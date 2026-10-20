"""Reference validators, one per target language.

A validator is looked up on ``PATH``; ``CROSSTL_COVERAGE_<TOOL>`` (for example ``CROSSTL_COVERAGE_DXC``) points at a
specific binary instead.
"""

from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

TIMEOUT = 60
# Terminal colour codes some validators (naga) write even when their output is not a terminal.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The tool each language is validated with, and the suffix its compiler expects.
TOOLS = {
    "cuda": "nvcc",
    "glsl": "glslangValidator",
    "hip": "hipcc",
    "hlsl": "dxc",
    "metal": "metal",
    "spirv": "spirv-val",
    "wgsl": "naga",
}
SUFFIX = {
    "cuda": ".cu",
    "glsl": ".comp",
    "hip": ".hip",
    "hlsl": ".hlsl",
    "metal": ".metal",
    "spirv": ".spvasm",
    "wgsl": ".wgsl",
}
GLSL_STAGES = {"compute": "comp", "fragment": "frag", "vertex": "vert"}
HLSL_PROFILES = {"compute": "cs_6_6", "fragment": "ps_6_6", "vertex": "vs_6_6"}
SPIRV_ENVIRONMENT = "vulkan1.1"
# The language version of the Metal feature list (features/metal.txt); probes compile against the same version.
METAL_STANDARD = "metal4.1"
# CUDA probes compile for compute capability 8.0, which has every atomic and half-precision operation the CUDA
# feature list names, and for C++17.
CUDA_ARCH = "sm_80"
# HIP probes compile for RDNA3 (gfx1100), which has the image instructions texture and surface access need; the CDNA
# targets have none. No GPU is needed: hipcc compiles for the named target.
HIP_ARCH = "gfx1100"


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    log: str = ""


def _executable(name: str) -> str | None:
    override = os.environ.get("CROSSTL_COVERAGE_" + re.sub(r"\W", "_", name).upper())
    return override or shutil.which(name)


@functools.lru_cache(maxsize=None)
def command(name: str) -> tuple[str, ...]:
    """The command that runs ``name``, or ``()`` when it is not installed."""
    if name == "metal":
        # `xcrun -f metal` succeeds even when the Metal toolchain component is not installed; --version does not.
        metal = ("xcrun", "-sdk", "macosx", "metal")
        if platform.system() != "Darwin" or not shutil.which("xcrun"):
            return ()
        try:
            probe = subprocess.run(
                [*metal, "--version"], capture_output=True, text=True, timeout=TIMEOUT
            )
        except (OSError, subprocess.TimeoutExpired):
            return ()
        return metal if probe.returncode == 0 else ()
    found = _executable(name)
    return (found,) if found else ()


@functools.lru_cache(maxsize=None)
def version(name: str) -> str:
    """The first line the tool prints for ``--version``, or an empty string when it is not installed."""
    base = command(name)
    if not base:
        return ""
    try:
        result = subprocess.run(
            [*base, "--version"], capture_output=True, text=True, timeout=TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    return lines[0] if lines else ""


def available(language: str) -> bool:
    if language == "spirv":
        return bool(command("spirv-as") and command("spirv-val"))
    return bool(command(TOOLS[language]))


def _run(arguments: list[str], cwd: str) -> Verdict:
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=TIMEOUT, cwd=cwd
        )
    except subprocess.TimeoutExpired:
        return Verdict(False, f"{arguments[0]} timed out after {TIMEOUT} s")
    log = _ANSI.sub("", result.stdout + result.stderr).replace(cwd + os.sep, "").strip()
    # Compilers report the first error first and end with notes and counts, so the head of the log is kept.
    return Verdict(result.returncode == 0, log[:2000])


# Entry-point names HLSL writers conventionally give each stage, tried when a shader has no ``main``.
HLSL_STAGE_ENTRIES = {"vertex": "VSMain", "fragment": "PSMain"}


def hlsl_entry(source: str, stage: str) -> str:
    """The entry point of an HLSL shader: for compute the function that follows ``[numthreads(...)]``, otherwise
    ``main``, or the stage's conventional name when there is no ``main``."""
    if stage == "compute":
        match = re.search(
            r"\[numthreads\([^)]*\)\]\s*(?:\[[^\]]*\]\s*)*void\s+(\w+)", source
        )
        if match:
            return match.group(1)
        return "main"
    conventional = HLSL_STAGE_ENTRIES.get(stage, "")
    if not re.search(r"\bmain\s*\(", source) and conventional:
        if re.search(rf"\b{conventional}\s*\(", source):
            return conventional
    return "main"


def validate(
    language: str, source: str, stage: str = "compute", entry: str = ""
) -> Verdict:
    """Check ``source`` with the language's reference validator. Call only when :func:`available` is true."""
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        name = "probe" + (
            f".{GLSL_STAGES[stage]}" if language == "glsl" else SUFFIX[language]
        )
        Path(scratch, name).write_text(source)
        if language == "glsl":
            return _run([*command("glslangValidator"), name], scratch)
        if language == "hlsl":
            entry = entry or hlsl_entry(source, stage)
            profile = HLSL_PROFILES[stage]
            return _run(
                [
                    *command("dxc"),
                    "-T",
                    profile,
                    "-E",
                    entry,
                    "-HV",
                    "2021",
                    "-enable-16bit-types",
                    name,
                    "-Fo",
                    "probe.bin",
                ],
                scratch,
            )
        if language == "metal":
            return _run(
                [
                    *command("metal"),
                    f"-std={METAL_STANDARD}",
                    "-c",
                    name,
                    "-o",
                    "probe.air",
                ],
                scratch,
            )
        if language == "wgsl":
            return _run([*command("naga"), name], scratch)
        if language == "cuda":
            return _run(
                [
                    *command("nvcc"),
                    f"-arch={CUDA_ARCH}",
                    "-std=c++17",
                    "-c",
                    name,
                    "-o",
                    "probe.o",
                ],
                scratch,
            )
        if language == "hip":
            return _run(
                [
                    *command("hipcc"),
                    f"--offload-arch={HIP_ARCH}",
                    "-std=c++17",
                    "-c",
                    name,
                    "-o",
                    "probe.o",
                ],
                scratch,
            )
        environment = ["--target-env", SPIRV_ENVIRONMENT]
        assembled = _run(
            [*command("spirv-as"), *environment, name, "-o", "probe.spv"], scratch
        )
        if not assembled.accepted:
            return assembled
        return _run([*command("spirv-val"), *environment, "probe.spv"], scratch)


# How each language's reference compiler lowers a compute probe to SPIR-V; ``{source}`` and ``{entry}`` are filled in.
SPIRV_COMPILERS = {
    "glsl": (
        "glslangValidator",
        ("-V", "--target-env", SPIRV_ENVIRONMENT, "{source}", "-o", "probe.spv"),
    ),
    "hlsl": (
        "dxc",
        (
            "-spirv",
            "-T",
            HLSL_PROFILES["compute"],
            "-E",
            "{entry}",
            "-HV",
            "2021",
            "-enable-16bit-types",
            "-O0",
            "-fspv-target-env=" + SPIRV_ENVIRONMENT,
            "{source}",
            "-Fo",
            "probe.spv",
        ),
    ),
    "wgsl": ("naga", ("{source}", "probe.spv")),
}


def can_lower(language: str) -> bool:
    compiler = SPIRV_COMPILERS.get(language)
    return bool(compiler and command(compiler[0]) and command("spirv-dis"))


def lower(language: str, source: str, entry: str = "main") -> tuple[str, str]:
    """Compile a compute probe to SPIR-V and disassemble it: ``(assembly, "")`` on success, ``("", log)`` if not.

    Call only when :func:`can_lower` is true.
    """
    tool, arguments = SPIRV_COMPILERS[language]
    name = "probe" + (".comp" if language == "glsl" else SUFFIX[language])
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        Path(scratch, name).write_text(source)
        filled = [a.format(source=name, entry=entry) for a in arguments]
        compiled = _run([*command(tool), *filled], scratch)
        if not compiled.accepted:
            return "", compiled.log
        try:
            result = subprocess.run(
                [*command("spirv-dis"), "--raw-id", "probe.spv"],
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
