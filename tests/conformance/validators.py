"""Native validation with explicit compile configurations and retained commands."""

from __future__ import annotations

import re
from pathlib import Path

from .interface import hlsl_reflected_strides
from .process import CommandResult, require_artifact, run
from .toolchains import tool

TARGETS = {"metal": ".metal", "hlsl": ".hlsl", "glsl": ".comp", "spirv": ".spvasm"}
BACKENDS = {"metal": "metal", "hlsl": "directx", "glsl": "opengl", "spirv": "vulkan"}
SOURCE_BACKENDS = {"glsl": "opengl", "hlsl": "directx"}
BINARIES = {"metal": ".air", "hlsl": ".dxil", "glsl": ".spv", "spirv": ".spv"}
HLSL_PROFILE = "cs_6_6"
CONFIGS = ("strict", "relaxed")
ENVIRONMENTS = ("vulkan1.1", "vulkan1.2", "vulkan1.3")
# HLSL 2021 native 16-bit scalar, vector and matrix types exist only under `dxc -enable-16bit-types` (Shader Model 6.2).
NATIVE_16_BIT = re.compile(r"\b(?:float16_t|int16_t|uint16_t)(?:[1-4](?:x[1-4])?)?\b")


def hlsl_arguments(path):
    """The DXC arguments an HLSL artifact needs beyond its profile and entry point."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return []
    return ["-enable-16bit-types"] if NATIVE_16_BIT.search(text) else []


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


def _hlsl(path, output, entry, config, profile):
    missing = _missing("dxc")
    if missing:
        return [missing], {}
    dxc = tool("dxc").command
    command = [*dxc, path, "-T", profile, "-E", entry, "-HV", "2021", "-Fo", output]
    command += hlsl_arguments(path)
    if config == "strict":
        command.append("-Gis")
    result = require_artifact(run(command), output)
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
        command = [*tool("metal").command, "-c", path, "-o", output]
        command += ["-fno-fast-math"] if config == "strict" else ["-ffast-math"]
        return [require_artifact(run(command), output)], {}
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
