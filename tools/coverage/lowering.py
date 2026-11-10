"""How each language's reference compiler lowers a probe to SPIR-V, which :mod:`.semantic`, :mod:`.decorations`,
:mod:`.execution` and :mod:`.spirv_input` compare and run."""

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
            # slangc removes unused locals even at -O0; with debug information it keeps them, so the baseline
            # probe keeps the argument reads the call's probe has. Debug instructions never join an id.
            "-g2",
            "-o",
            "probe.spv",
        ),
    ),
    "wgsl": ("naga", ("{source}", "probe.spv")),
    # rust-gpu names the module after the crate, which is named after the file.
    "rust": ("rust-gpu", ("{source}", "--out-dir", ".")),
}
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
) -> tuple[str, str]:
    """Compile a probe to SPIR-V and disassemble it: ``(assembly, "")`` on success, ``("", log)`` if not.

    ``raw_ids`` numbers every id instead of naming it after its debug name, so assemblies compare by instruction.
    ``keep_bindings`` keeps the resources the probe never reads (:data:`KEEP_BINDINGS`), and ``auto_locations``
    numbers the stage inputs and outputs that have no location (:data:`AUTO_LOCATIONS`). GLSL is lowered with the
    Vulkan names of its vertex and instance index (:func:`vulkan_glsl`). Call only when :func:`can_lower` is true.
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
