"""Native validation with explicit compile configurations and retained commands."""

from __future__ import annotations

import re
from pathlib import Path

from crosstl.translator.codegen.hlsl_specialization_macros import (
    HLSL_SPECIALIZATION_MACRO_RE,
)
from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_MACROS,
)

from .interface import hlsl_reflected_strides
from .native import METAL_LANGUAGE_VERSIONS
from .process import CommandResult, require_artifact, run
from .toolchains import tool

TARGETS = {"metal": ".metal", "hlsl": ".hlsl", "glsl": ".comp", "spirv": ".spvasm"}
BACKENDS = {"metal": "metal", "hlsl": "directx", "glsl": "opengl", "spirv": "vulkan"}
SOURCE_BACKENDS = {"glsl": "opengl", "hlsl": "directx"}
BINARIES = {"metal": ".air", "hlsl": ".dxil", "glsl": ".spv", "spirv": ".spv"}
HLSL_PROFILE = "cs_6_6"
# The later compute profiles a kernel compiles under, oldest first, when DXC rejects it at an older one only because it
# calls an operation of a later shader model: SampleCmpLevel and GatherRaw need Shader Model 6.7, SampleCmpGrad,
# SampleCmpBias and Barrier 6.8. The Direct3D 12 device of the Windows CI creates pipelines at each of them
# (test_native_runtimes.test_d3d12_device_runs_the_harness_shader_model) but not at cs_6_9.
HLSL_LATER_PROFILES = ("cs_6_7", "cs_6_8")
# DXC's diagnostics for an operation that a later shader model introduced: Sema's availability check names the
# model, and the validator names the opcode it does not allow under the profile.
_LATER_SHADER_MODEL = re.compile(
    r"requires shader model \d+\.\d+ or greater|"
    r"Opcode \w+ not valid in shader model \w+_\d+_\d+"
)
CONFIGS = ("strict", "relaxed")
ENVIRONMENTS = ("vulkan1.1", "vulkan1.2", "vulkan1.3")
# HLSL 2021 native 16-bit scalar, vector and matrix types exist only under `dxc -enable-16bit-types` (Shader Model 6.2).
NATIVE_16_BIT = re.compile(r"\b(?:float16_t|int16_t|uint16_t)(?:[1-4](?:x[1-4])?)?\b")


# A thread-group size the source's host chooses reads three macros that the host defines when it compiles the shader
# (crosstl/translator/codegen/hlsl_host_workgroup_size.py); the check gives them the size of one invocation, as the
# GLSL check does (corpus/checks.py).
HOST_WORKGROUP_SIZE_DEFAULT = "1"
# A specialization constant without a default reads its value from a macro the host defines
# (crosstl/translator/codegen/hlsl_specialization_macros.py) and casts it to the constant's scalar type. The check
# defines each macro as 0, the zero of that type, which is the value the GLSL check encodes (corpus/checks.py) and the
# default a SPIR-V module is validated under.
HLSL_SPECIALIZATION_DEFAULT = "0"


def hlsl_arguments(path):
    """The DXC arguments an HLSL artifact needs beyond its profile and entry point."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return []
    arguments = ["-enable-16bit-types"] if NATIVE_16_BIT.search(text) else []
    if any(f"defined({macro})" in text for macro in HOST_WORKGROUP_SIZE_MACROS):
        for macro in HOST_WORKGROUP_SIZE_MACROS:
            arguments += ["-D", f"{macro}={HOST_WORKGROUP_SIZE_DEFAULT}"]
    specialization_macros = sorted(
        {match["macro"] for match in HLSL_SPECIALIZATION_MACRO_RE.finditer(text)}
    )
    for macro in specialization_macros:
        arguments += ["-D", f"{macro}={HLSL_SPECIALIZATION_DEFAULT}"]
    return arguments


def _missing(name):
    found = tool(name)
    return (
        None
        if found.available
        else CommandResult((name,), "unavailable", None, found.reason)
    )


def dxil_is_signed(path):
    """DXC fills the container digest only after the validator accepted the module."""
    header = Path(path).read_bytes()[:20]
    return header[:4] == b"DXBC" and any(header[4:20])


def later_profiles(profile):
    """``profile`` and the later profiles of :data:`HLSL_LATER_PROFILES` a kernel may fall back to, oldest first."""
    match = re.fullmatch(r"cs_(\d+)_(\d+)", profile)
    if match is None:
        return (profile,)
    model = (int(match.group(1)), int(match.group(2)))
    later = tuple(
        candidate
        for candidate in HLSL_LATER_PROFILES
        if tuple(int(part) for part in candidate.split("_")[1:]) > model
    )
    return (profile, *later)


def _compile_hlsl(dxc, path, output, entry, config, profile):
    """Compile under the oldest of :func:`later_profiles` that DXC accepts the kernel under.

    A kernel moves to a later profile only when DXC names a later shader model in its rejection; any other rejection,
    and a rejection under every profile, reports the last profile's diagnostics.
    """
    for candidate in later_profiles(profile):
        command = [*dxc, path, "-T", candidate, "-E", entry, "-HV", "2021"]
        command += ["-Fo", output, *hlsl_arguments(path)]
        if config == "strict":
            command.append("-Gis")
        result = require_artifact(run(command), output)
        if result.status != "rejected" or not _LATER_SHADER_MODEL.search(
            result.log or ""
        ):
            break
    return result


def _hlsl(path, output, entry, config, profile):
    missing = _missing("dxc")
    if missing:
        return [missing], {}
    dxc = tool("dxc").command
    result = _compile_hlsl(dxc, path, output, entry, config, profile)
    if result.status != "accepted":
        return [result], {}
    if not dxil_is_signed(output):
        return [
            CommandResult(
                result.command,
                "rejected",
                result.returncode,
                result.log
                + "\nDXC wrote an unsigned container: DXIL validation did not run.",
            )
        ], {}
    dump = run([*dxc, "-dumpbin", output])
    if dump.status != "accepted":
        return [
            result,
            CommandResult(dump.command, "harness-error", dump.returncode, dump.log),
        ], {}
    return [result, dump], {"strides": hlsl_reflected_strides(dump.log)}


def _metal(path, output, config):
    """Compile under the oldest MSL version that accepts the artifact and name it for the driver compile.

    A rejection under every version reports the newest compiler's diagnostics.
    """
    for major, minor in METAL_LANGUAGE_VERSIONS:
        command = [*tool("metal").command, f"-std=metal{major}.{minor}"]
        command += ["-c", path, "-o", output]
        command += ["-fno-fast-math"] if config == "strict" else ["-ffast-math"]
        result = require_artifact(run(command), output)
        if result.status != "rejected":
            break
    reflection = {"metal_language": [major, minor]}
    return [result], reflection if result.status == "accepted" else {}


def validate(
    path,
    target,
    output,
    entry,
    *,
    config="strict",
    environment="vulkan1.1",
    profile=HLSL_PROFILE,
):
    """Run the target's native checker; returns the command results and any reflection data."""
    path, output = Path(path), Path(output)
    if config not in CONFIGS or environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown configuration {config!r} or environment {environment!r}"
        )
    if target == "metal":
        missing = _missing("metal")
        if missing:
            return [missing], {}
        return _metal(path, output, config)
    if target == "glsl":
        missing = _missing("glslangValidator")
        if missing:
            return [missing], {}
        # Checks the GLSL rules for OpenGL; the driver compile happens in the OpenGL runtime.
        return [run([*tool("glslangValidator").command, "-S", "comp", path])], {}
    if target == "hlsl":
        return _hlsl(path, output, entry, config, profile)
    missing = _missing("spirv-as") or _missing("spirv-val")
    if missing:
        return [missing], {}
    results = [
        require_artifact(
            run(
                [
                    *tool("spirv-as").command,
                    "--target-env",
                    environment,
                    path,
                    "-o",
                    output,
                ]
            ),
            output,
        )
    ]
    if results[0].status == "accepted":
        results.append(
            run(
                [
                    *tool("spirv-val").command,
                    "--target-env",
                    environment,
                    *spirv_layout_flags(path),
                    output,
                ]
            )
        )
    return results, {}


def spirv_layout_flags(path):
    """``spirv-val`` flags for the block layouts a module declares it relies on.

    A module whose blocks need the scalar block layout says so with ``OpSourceExtension
    "GL_EXT_scalar_block_layout"``, as glslang does; the host then enables ``scalarBlockLayout``. Any other module
    is checked against Vulkan's default layout rules.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if 'OpSourceExtension "GL_EXT_scalar_block_layout"' in text:
        return ["--scalar-block-layout"]
    return []
