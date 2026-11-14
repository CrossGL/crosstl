"""How each language's reference compiler lowers a probe to SPIR-V, which :mod:`.semantic`, :mod:`.decorations`,
:mod:`.execution` and :mod:`.spirv_input` compare and run, and the parts of that SPIR-V a host must not see: the
capabilities rust-gpu declares and the counter buffers DXC declares that the module never uses.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from . import rustgpu, validators
from .validators import (
    GLSL_STAGES,
    HLSL_PROFILES,
    SLANG_STAGES,
    SLANG_TARGET,
    SPIRV_1_4_ENVIRONMENT,
    SPIRV_1_4_STAGES,
    SPIRV_ENVIRONMENT,
    SUFFIX,
    TIMEOUT,
)

# How each language's reference compiler lowers a probe to SPIR-V; ``{source}``, ``{entry}``, ``{stage}``, the HLSL
# ``{profile}`` and the Vulkan ``{environment}`` are filled in.
SPIRV_COMPILERS = {
    "glsl": (
        "glslangValidator",
        ("-V", "--target-env", "{environment}", "{source}", "-o", "probe.spv"),
    ),
    "hlsl": (
        "dxc",
        (
            "-spirv",
            "-T",
            "{profile}",
            "-E",
            "{entry}",
            "-HV",
            "2021",
            "-enable-16bit-types",
            "-O0",
            "-fspv-target-env={environment}",
            "{source}",
            "-Fo",
            "probe.spv",
        ),
    ),
    "slang": (
        "slangc",
        (
            "{source}",
            "-target",
            SLANG_TARGET,
            "-entry",
            "{entry}",
            "-stage",
            "{stage}",
            "-O0",
            "-o",
            "probe.spv",
        ),
    ),
    "wgsl": ("naga", ("{source}", "probe.spv")),
    # rust-gpu names the module after the crate, which is named after the file.
    "rust": ("rust-gpu", ("{source}", "--out-dir", ".")),
}
# slangc removes unused locals even at -O0; with debug information it keeps them, so the baseline probe keeps the
# argument reads the call's probe has. Debug instructions never join an id. A SPIR-V input probe is lowered without
# it, as a project ships its SPIR-V.
DEBUG_INFO = {"slang": ("-g2",)}
# The flags that make a compiler keep the resources a shader declares but never reads, so its SPIR-V keeps the
# bindings the source gives them. glslang and naga keep them without one.
KEEP_BINDINGS = {"hlsl": ("-fspv-preserve-bindings",), "slang": ("-preserve-params",)}
# The flags that let glslang lower OpenGL GLSL, whose stage inputs and outputs may have no location, for Vulkan.
AUTO_LOCATIONS = {"glsl": ("--auto-map-locations",)}
# GLSL for Vulkan has no gl_VertexID or gl_InstanceID: it names the vertex and instance index gl_VertexIndex and
# gl_InstanceIndex (GL_KHR_vulkan_glsl), so OpenGL GLSL is lowered with those names. gl_InstanceIndex also counts the
# draw's first instance, which a lowered probe, never drawn, does not observe.
VULKAN_NAMES = {"gl_VertexID": "gl_VertexIndex", "gl_InstanceID": "gl_InstanceIndex"}
_OPENGL_NAME = re.compile(rf"\b(?:{'|'.join(VULKAN_NAMES)})\b")


def vulkan_glsl(source: str) -> str:
    """OpenGL GLSL ``source`` with the vertex and instance index given their Vulkan names (:data:`VULKAN_NAMES`)."""
    return _OPENGL_NAME.sub(lambda match: VULKAN_NAMES[match[0]], source)


def can_lower(language: str) -> bool:
    compiler = SPIRV_COMPILERS.get(language)
    return bool(
        compiler and validators.command(compiler[0]) and validators.command("spirv-dis")
    )


def lower(
    language: str,
    source: str,
    entry: str = "main",
    stage: str = "compute",
    *,
    raw_ids: bool = True,
    keep_bindings: bool = False,
    auto_locations: bool = False,
    debug: bool = True,
) -> tuple[str, str]:
    """Compile a probe to SPIR-V and disassemble it: ``(assembly, "")`` on success, ``("", log)`` if not.

    ``raw_ids`` numbers every id instead of naming it after its debug name, so assemblies compare by instruction.
    ``keep_bindings`` keeps the resources the probe never reads (:data:`KEEP_BINDINGS`), ``auto_locations``
    numbers the stage inputs and outputs that have no location (:data:`AUTO_LOCATIONS`), and ``debug`` adds the
    compiler's debug information (:data:`DEBUG_INFO`). GLSL is lowered with the Vulkan names of its vertex and
    instance index (:func:`vulkan_glsl`). Call only when :func:`can_lower` is true.
    """
    tool, arguments = SPIRV_COMPILERS[language]
    name = "probe" + (
        f".{GLSL_STAGES[stage]}" if language == "glsl" else SUFFIX[language]
    )
    fields = {
        "source": name,
        "entry": entry,
        "stage": SLANG_STAGES.get(stage, stage),
        "profile": HLSL_PROFILES.get(stage, ""),
        "environment": (
            SPIRV_1_4_ENVIRONMENT if stage in SPIRV_1_4_STAGES else SPIRV_ENVIRONMENT
        ),
    }
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        Path(scratch, name).write_text(
            vulkan_glsl(source) if language == "glsl" else source
        )
        filled = [a.format(**fields) for a in arguments]
        if keep_bindings:
            filled += KEEP_BINDINGS.get(language, ())
        if auto_locations:
            filled += AUTO_LOCATIONS.get(language, ())
        if debug:
            filled += DEBUG_INFO.get(language, ())
        compiler = (
            rustgpu.command(rustgpu.target(stage))
            if tool == "rust-gpu"
            else validators.command(tool)
        )
        compiled = validators.run([*compiler, *filled], scratch)
        if not compiled.accepted:
            return "", compiled.log
        try:
            result = subprocess.run(
                [
                    *validators.command("spirv-dis"),
                    *(["--raw-id"] if raw_ids else []),
                    "probe.spv",
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            return "", f"spirv-dis timed out after {TIMEOUT} s"
        if result.returncode != 0:
            return "", result.stderr.strip()[-2000:]
        return result.stdout, ""


_CAPABILITY = re.compile(r"^[ \t]*OpCapability (\w+)[ \t]*\n", re.M)


def without_unused_capabilities(assembly: str) -> str:
    """``assembly`` without each capability that spirv-val accepts it without.

    rust-gpu compiles every probe with all of ``rustgpu.CAPABILITIES`` enabled and removes only the ones it knows how
    to prove unused. A Vulkan module may only declare capabilities the device enables (Vulkan 1.4,
    VUID-VkShaderModuleCreateInfo-pCode-08740), and lavapipe will not create a pipeline from one that declares more.
    spirv-val requires every capability an instruction, type, decoration or built-in of the module needs, so what it
    accepts is what the module uses.
    """
    target = validators.spirv_environment([assembly])
    assemble, validate = validators.command("spirv-as"), validators.command("spirv-val")
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:

        def valid(text: str) -> bool:
            Path(scratch, "probe.spvasm").write_text(text)
            steps = (
                [*assemble, "--target-env", target, "probe.spvasm", "-o", "probe.spv"],
                [*validate, "--target-env", target, "probe.spv"],
            )
            return all(validators.run(step, scratch).accepted for step in steps)

        # Removing a later line leaves the offsets of the earlier ones in place.
        for found in reversed(list(_CAPABILITY.finditer(assembly))):
            trimmed = assembly[: found.start()] + assembly[found.end() :]
            if found[1] != "Shader" and valid(trimmed):
                assembly = trimmed
    return assembly


# DXC declares a counter buffer, of this type, for every RWStructuredBuffer a function takes as a parameter, and
# -fspv-preserve-bindings keeps it, with a binding of its own, when nothing uses it.
_COUNTER = re.compile(r'^\s*OpName (%\S+) "type\.ACSBuffer\.counter"$', re.M)
# One instruction that defines an id: the id, the opcode and the operands.
DEFINITION = re.compile(r"^\s*(%\S+) = (Op\w+)((?: \S+)*)\s*$")
# Instructions that name or decorate an id without using it.
_ANNOTATIONS = frozenset({"OpName", "OpDecorate"})


def without_unused_counters(assembly: str) -> str:
    """``assembly`` without the counter buffers DXC declares that nothing uses.

    The source declares no such buffer and the shader never reads it, so it is no resource the host binds.
    """
    types = set(_COUNTER.findall(assembly))
    if not types:
        return assembly
    lines = assembly.splitlines(keepends=True)
    defined = [DEFINITION.match(line) for line in lines]
    pointers = {
        found[1]
        for found in defined
        if found and found[2] == "OpTypePointer" and found[3].split()[-1] in types
    }
    counters = {
        found[1]
        for found in defined
        if found and found[2] == "OpVariable" and found[3].split()[0] in pointers
    }
    used = set()
    for line, found in zip(lines, defined):
        words = line.split()
        if (found and found[1] in counters) or words[:1] == ["OpEntryPoint"]:
            continue
        used |= counters.intersection(words[2:] if _annotates(words) else words)
    unused = counters - used
    kept = []
    for line, found in zip(lines, defined):
        words = line.split()
        if (found and found[1] in unused) or (_annotates(words) and words[1] in unused):
            continue
        if words[:1] == ["OpEntryPoint"]:
            line = re.sub(
                r"[ \t]+(%\S+)", lambda m: "" if m[1] in unused else m[0], line
            )
        kept.append(line)
    return "".join(kept)


def _annotates(words: list[str]) -> bool:
    return bool(words) and words[0] in _ANNOTATIONS
