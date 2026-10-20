"""Reference validators, one per target language.

A validator is looked up on ``PATH``; ``CROSSTL_COVERAGE_<TOOL>`` (for example ``CROSSTL_COVERAGE_DXC``) points at a
specific binary instead. rust-gpu is a rustc backend rather than a binary; :mod:`.rustgpu` says how to provide it.
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

from . import rustgpu
from .model import RAYTRACING_STAGES

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
    "mojo": "mojo",
    "rust": "rust-gpu",
    "slang": "slangc",
    "spirv": "spirv-val",
    "wgsl": "naga",
}
SUFFIX = {
    "cuda": ".cu",
    "glsl": ".comp",
    "hip": ".hip",
    "hlsl": ".hlsl",
    "metal": ".metal",
    "mojo": ".mojo",
    "rust": ".rs",
    "slang": ".slang",
    "spirv": ".spvasm",
    "wgsl": ".wgsl",
}
GLSL_STAGES = {
    "compute": "comp",
    "fragment": "frag",
    "vertex": "vert",
    "geometry": "geom",
    "tessellation-control": "tesc",
    "tessellation-evaluation": "tese",
    "task": "task",
    "mesh": "mesh",
    # glslang's stage suffixes for GL_EXT_ray_tracing shaders.
    "ray-generation": "rgen",
    "intersection": "rint",
    "any-hit": "rahit",
    "closest-hit": "rchit",
    "miss": "rmiss",
    "callable": "rcall",
}
# The newest shader model the pinned DXC accepts. Feature lists leave out intrinsics that need a newer one.
HLSL_SHADER_MODEL = (6, 9)
# Ray-tracing shaders compile as a library, whose ``[shader(...)]`` attributes name the entry points.
HLSL_PROFILES = {
    stage: f"{prefix}_{HLSL_SHADER_MODEL[0]}_{HLSL_SHADER_MODEL[1]}"
    for stage, prefix in {
        "compute": "cs",
        "fragment": "ps",
        "vertex": "vs",
        "geometry": "gs",
        "tessellation-control": "hs",
        "tessellation-evaluation": "ds",
        "task": "as",
        "mesh": "ms",
        **dict.fromkeys(RAYTRACING_STAGES, "lib"),
    }.items()
}
# slangc's names for the stages it does not spell as the report does.
SLANG_STAGES = {
    **RAYTRACING_STAGES,
    "tessellation-control": "hull",
    "tessellation-evaluation": "domain",
    "task": "amplification",
}
SPIRV_ENVIRONMENT = "vulkan1.1"
# Ray tracing and mesh shading need SPIR-V 1.4, which Vulkan 1.1 allows only as an extension; DXC lowers them for
# Vulkan 1.2.
SPIRV_1_4_ENVIRONMENT = "vulkan1.2"
SPIRV_1_4_STAGES = frozenset({*RAYTRACING_STAGES, "task", "mesh"})
# The language version of the Metal feature list (features/metal.txt); probes compile against the same version.
METAL_STANDARD = "metal4.1"
# CUDA probes compile for compute capability 8.0, which has every atomic and half-precision operation the CUDA
# feature list names, and for C++17.
CUDA_ARCH = "sm_80"
# Slang probes compile to SPIR-V, the target whose validator (SPIRV-Tools, built into slangc) checks the result.
SLANG_TARGET = "spirv"
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
    if name == "rust-gpu":
        return rustgpu.command()
    found = _executable(name)
    if (
        found
        and name == "mojo"
        and not _MOJO_RELEASE.match(_first_line([found, "--version"]))
    ):
        # Calendar-versioned Mojo (24.x, 25.x) predates the language the probes are written in; treat it as absent
        # so a stale install reads as untested rather than wrong.
        return ()
    return (found,) if found else ()


# The Mojo line the probes target: the compute job pins mojo-compiler 1.1.0.
_MOJO_RELEASE = re.compile(r"mojo\s+1\.", re.IGNORECASE)


def _first_line(arguments: list[str]) -> str:
    """The first non-empty line ``arguments`` prints, or an empty string when it cannot run."""
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    return lines[0] if lines else ""


@functools.lru_cache(maxsize=None)
def version(name: str) -> str:
    """The first line the tool prints for ``--version``, or an empty string when it is not installed."""
    if name == "rust-gpu":
        return rustgpu.version()
    base = command(name)
    if not base:
        return ""
    # slangc takes single-dash options only.
    flag = "-version" if name == "slangc" else "--version"
    return _first_line([*base, flag])


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
    log = _ANSI.sub("", result.stdout + result.stderr)
    # Some compilers (mojo) print the scratch directory's real path, which on macOS is under /private; the longer
    # spelling goes first, since it contains the other.
    for directory in sorted({os.path.realpath(cwd), cwd}, key=len, reverse=True):
        log = log.replace(directory + os.sep, "")
    log = log.strip()
    if result.returncode != 0 and not log:
        log = f"error: {Path(arguments[0]).name} exited with status {result.returncode} and no message"
    # Compilers report the first error first and end with notes and counts, so the head of the log is kept.
    return Verdict(result.returncode == 0, log[:2000])


# Entry-point names HLSL writers conventionally give each stage, tried when a shader has no ``main``.
HLSL_STAGE_ENTRIES = {
    "vertex": "VSMain",
    "fragment": "PSMain",
    "geometry": "GSMain",
    "tessellation-control": "HSMain",
    "tessellation-evaluation": "DSMain",
    "task": "ASMain",
    "mesh": "MSMain",
}


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


# The AIR metadata that marks a Metal entry point of each stage; a post-tessellation vertex function also describes
# its patch.
METAL_ENTRIES = {
    "compute": (r"^!air\.kernel = ",),
    "vertex": (r"^!air\.vertex = ",),
    "fragment": (r"^!air\.fragment = ",),
    "tessellation-evaluation": (r"^!air\.vertex = ", r'!"air\.patch"'),
    "task": (r"^!air\.object = ",),
    "mesh": (r"^!air\.mesh = ",),
}


def _metal(name: str, stage: str, scratch: str) -> Verdict:
    """Compile a Metal probe to textual AIR and require an entry point of ``stage``: the compiler accepts a library
    of plain functions, which no pipeline can run."""
    compiled = _run(
        [
            *command("metal"),
            f"-std={METAL_STANDARD}",
            "-S",
            "-emit-llvm",
            name,
            "-o",
            "probe.ll",
        ],
        scratch,
    )
    if not compiled.accepted:
        return compiled
    air = Path(scratch, "probe.ll").read_text()
    for pattern in METAL_ENTRIES.get(stage, ()):
        if not re.search(pattern, air, re.MULTILINE):
            return Verdict(False, f"error: the library has no {stage} entry point")
    return compiled


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
            # GL_EXT_ray_tracing and GL_EXT_mesh_shader exist only for SPIR-V 1.4 and later, so those shaders are
            # checked for Vulkan.
            environment = (
                ["--target-env", SPIRV_1_4_ENVIRONMENT]
                if stage in SPIRV_1_4_STAGES
                else []
            )
            return _run([*command("glslangValidator"), *environment, name], scratch)
        if language == "hlsl":
            profile = HLSL_PROFILES[stage]
            entry_point = (
                []
                if profile.startswith("lib_")
                else ["-E", entry or hlsl_entry(source, stage)]
            )
            return _run(
                [
                    *command("dxc"),
                    "-T",
                    profile,
                    *entry_point,
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
            return _metal(name, stage, scratch)
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
        if language == "mojo":
            # The Mojo compiler type-checks every function it parses, called or not; emitting LLVM IR needs no main.
            return _run(
                [
                    *command("mojo"),
                    "build",
                    "--emit",
                    "llvm",
                    "--disable-warnings",
                    name,
                    "-o",
                    "probe.ll",
                ],
                scratch,
            )
        if language == "rust":
            return _run([*command("rust-gpu"), name, "--out-dir", "."], scratch)
        if language == "slang":
            # A ``[shader(...)]`` attribute names the entry point and its stage; without one, slangc is told both.
            entry_point = (
                []
                if "[shader(" in source
                else ["-entry", entry or "main", "-stage", stage]
            )
            return _run(
                [
                    *command("slangc"),
                    name,
                    *entry_point,
                    "-target",
                    SLANG_TARGET,
                    "-o",
                    "probe.spv",
                ],
                scratch,
            )
        environment = [
            "--target-env",
            (SPIRV_1_4_ENVIRONMENT if stage in SPIRV_1_4_STAGES else SPIRV_ENVIRONMENT),
        ]
        assembled = _run(
            [*command("spirv-as"), *environment, name, "-o", "probe.spv"], scratch
        )
        if not assembled.accepted:
            return assembled
        return _run([*command("spirv-val"), *environment, "probe.spv"], scratch)


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


def can_lower(language: str) -> bool:
    compiler = SPIRV_COMPILERS.get(language)
    return bool(compiler and command(compiler[0]) and command("spirv-dis"))


def lower(
    language: str, source: str, entry: str = "main", stage: str = "compute"
) -> tuple[str, str]:
    """Compile a probe to SPIR-V and disassemble it: ``(assembly, "")`` on success, ``("", log)`` if not.

    Call only when :func:`can_lower` is true.
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
        Path(scratch, name).write_text(source)
        filled = [a.format(**fields) for a in arguments]
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
