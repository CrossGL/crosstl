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
    "opencl": "clang",
    "rust": "rust-gpu",
    "slang": "slangc",
    "spirv": "spirv-val",
    "webgl": "glslangValidator",
    "wgsl": "naga",
}
SUFFIX = {
    "cuda": ".cu",
    "glsl": ".comp",
    "hip": ".hip",
    "hlsl": ".hlsl",
    "metal": ".metal",
    "mojo": ".mojo",
    "opencl": ".cl",
    "rust": ".rs",
    "slang": ".slang",
    "spirv": ".spvasm",
    "webgl": ".frag",
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
# Ray-tracing and work-graph node shaders compile as a library, whose ``[shader(...)]`` attributes name the entry
# points.
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
        "node": "lib",
    }.items()
}
# The ``[shader(...)]`` kind of each stage a library holds.
HLSL_LIBRARY_KINDS = {**RAYTRACING_STAGES, "node": "node"}
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
# OpenCL C 3.0 for SPIR: Clang's SPIR targets support every optional feature and Khronos extension, so a probe needs
# no -cl-ext flags. Only half needs the probe to enable cl_khr_fp16 with a pragma.
OPENCL_STANDARD = "CL3.0"
OPENCL_TARGET = "spir64"
# Slang probes compile to SPIR-V, the target whose validator (SPIRV-Tools, built into slangc) checks the result.
SLANG_TARGET = "spirv"
# slangc runs that validator only when this variable is set; without it, slangc writes invalid SPIR-V and exits 0.
SLANG_VALIDATION = {"SLANG_RUN_SPIRV_VALIDATION": "1"}
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


def missing(language: str, target: str = "") -> str:
    """The first tool that checking ``language`` probes for ``target`` needs and this machine lacks, or ``""``."""
    needed = {
        "spirv": ("spirv-as", "spirv-val"),
        "slang": ("slangc", "dxc") if target == "hlsl" else ("slangc",),
        # An imported OpenCL C probe is checked as the HLSL it is emitted to (runner.ROUND_TRIPS).
        "opencl": ("clang", "dxc"),
    }.get(language, (TOOLS[language],))
    return next((tool for tool in needed if not command(tool)), "")


def available(language: str, target: str = "") -> bool:
    return not missing(language, target)


def _run(
    arguments: list[str],
    cwd: str,
    environment: dict[str, str] | None = None,
    stdout: bool = True,
) -> Verdict:
    """Run a validator in ``cwd``. ``environment`` adds variables to the inherited environment; without ``stdout``,
    the log leaves out standard output, where slangc lists the SPIR-V it failed to validate ahead of the error.
    """
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=cwd,
            env={**os.environ, **environment} if environment else None,
        )
    except subprocess.TimeoutExpired:
        return Verdict(False, f"{arguments[0]} timed out after {TIMEOUT} s")
    log = _ANSI.sub("", (result.stdout if stdout else "") + result.stderr)
    # Some compilers (mojo) print the scratch directory's real path, which on macOS is under /private; the longer
    # spelling goes first, since it contains the other.
    for directory in sorted({os.path.realpath(cwd), cwd}, key=len, reverse=True):
        log = log.replace(directory + os.sep, "")
    log = log.strip()
    if result.returncode != 0 and not log:
        log = f"error: {Path(arguments[0]).name} exited with status {result.returncode} and no message"
    # Compilers report the first error first and end with notes and counts, so the head of the log is kept.
    return Verdict(result.returncode == 0, log[:2000])


def _suffix(language: str, stage: str) -> str:
    """The file suffix of a probe, from which glslang tells its stage. A WebGL kernel probe is a fragment shader."""
    if language == "glsl":
        return f".{GLSL_STAGES[stage]}"
    if language == "webgl":
        return ".vert" if stage == "vertex" else ".frag"
    return SUFFIX[language]


# glslang checks a shader against the version its ``#version`` line names, so a WebGL shader must name GLSL ES 3.00.
_ESSL_VERSION = re.compile(r"\A(?:\s|//[^\n]*\n)*#\s*version\s+300\s+es\b")


def _essl(name: str, source: str, scratch: str) -> Verdict:
    """Check a WebGL shader with glslang as GLSL ES 3.00, the shading language of WebGL 2."""
    if not _ESSL_VERSION.match(source):
        return Verdict(False, "error: the shader does not begin with #version 300 es")
    return _run([*command("glslangValidator"), name], scratch)


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
    "intersection": (r"^!air\.intersection = ",),
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


# What Slang's library emits for an operation that its HLSL target cannot express, as for ``debugBreak``.
_UNSUPPORTED_IN_HLSL = re.compile(r"/\*[^*]*not currently supported for HLSL\s*\*/")


def _dxc(name: str, source: str, stage: str, entry: str, scratch: str) -> Verdict:
    """Compile an HLSL probe. A library must also declare an entry point of ``stage``: DXC accepts a library of plain
    functions, which no pipeline can run."""
    profile = HLSL_PROFILES[stage]
    library = profile.startswith("lib_")
    entry_point = [] if library else ["-E", entry or hlsl_entry(source, stage)]
    kind = HLSL_LIBRARY_KINDS.get(stage, "")
    attribute = rf'\[\s*shader\s*\(\s*"{kind}"\s*\)\s*\]'
    if library and not re.search(attribute, source, re.IGNORECASE):
        return Verdict(False, f"error: the library has no {stage} entry point")
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


def validate(
    language: str,
    source: str,
    stage: str = "compute",
    entry: str = "",
    target: str = "",
) -> Verdict:
    """Check ``source`` with the language's reference validator, compiling for ``target`` when the language has
    more than one (Slang). Call only when :func:`available` is true."""
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        name = "probe" + _suffix(language, stage)
        Path(scratch, name).write_text(source)
        if language == "webgl":
            return _essl(name, source, scratch)
        if language == "glsl":
            # GL_EXT_ray_tracing and GL_EXT_mesh_shader exist only for SPIR-V 1.4 and later, so those shaders are
            # checked for Vulkan, as are probes whose feature only GLSL for Vulkan has.
            if stage in SPIRV_1_4_STAGES:
                environment = ["--target-env", SPIRV_1_4_ENVIRONMENT]
            elif target == "vulkan":
                environment = ["--target-env", SPIRV_ENVIRONMENT]
            else:
                environment = []
            return _run([*command("glslangValidator"), *environment, name], scratch)
        if language == "hlsl":
            return _dxc(name, source, stage, entry, scratch)
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
        if language == "opencl":
            return _run(
                [
                    *command("clang"),
                    "-x",
                    "cl",
                    f"-cl-std={OPENCL_STANDARD}",
                    "-target",
                    OPENCL_TARGET,
                    "-Xclang",
                    "-finclude-default-header",
                    "-fsyntax-only",
                    name,
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
            compiler = rustgpu.command(rustgpu.target(stage))
            return _run([*compiler, name, "--out-dir", "."], scratch)
        if language == "slang":
            # A ``[shader(...)]`` attribute names the entry point and its stage; without one, slangc is told both.
            # For a source target the output must follow the entry point it holds.
            entry_point = [
                *(
                    ["-entry", entry or "main"]
                    if target or "[shader(" not in source
                    else []
                ),
                *(
                    []
                    if "[shader(" in source
                    else ["-stage", SLANG_STAGES.get(stage, stage)]
                ),
            ]
            output = "probe.hlsl" if target == "hlsl" else "probe.spv"
            compiled = _run(
                [
                    *command("slangc"),
                    name,
                    *entry_point,
                    "-target",
                    target or SLANG_TARGET,
                    "-o",
                    output,
                ],
                scratch,
                SLANG_VALIDATION,
                stdout=False,
            )
            if target != "hlsl" or not compiled.accepted:
                return compiled
            # slangc writes HLSL without compiling it, so DXC checks the result as it checks an HLSL probe. Where
            # HLSL lacks an operation, Slang's library emits a comment in place of the call.
            generated = Path(scratch, output).read_text()
            dropped = _UNSUPPORTED_IN_HLSL.search(generated)
            if dropped:
                return Verdict(False, f"error: slangc emits {dropped.group(0)}")
            return _dxc(output, generated, stage, "", scratch)
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


# slangc names only the kind of an error in its header (``error[E30015]: undefined identifier``) and puts the
# particulars, such as the undefined name, in the label under the quoted source line. Its first error is the header
# joined with that label, so two probes that fail on different names have different causes.
LABELLED = frozenset({"slang"})
_HEADER = re.compile(r"error\[(?P<code>E\d+)\]: (?P<message>.+)")
_LABEL = re.compile(r"\s*\d*\s*\|\s*\^+\s+(?P<label>\S.*)")


def first_error(log: str, language: str = "") -> str:
    """The first line of a tool's log that reports an error, or the log's start when none says so. For a
    ``language`` in :data:`LABELLED`, the error's header is joined with the label under its source line.
    """
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if "error" not in line.lower():
            continue
        header = _HEADER.fullmatch(line.strip()) if language in LABELLED else None
        return _labelled(header, lines[index + 1 :]) if header else line.strip()
    return log[:300].strip()


def _labelled(header: re.Match[str], following: list[str]) -> str:
    """The error ``header`` with the first label of its own block. A label that only repeats the header adds
    nothing, and one that starts with the header's words replaces them."""
    for line in following:
        if _HEADER.fullmatch(line.strip()):
            break
        label = _LABEL.fullmatch(line)
        if label:
            text, message = label["label"].strip().rstrip("."), header["message"]
            if text.lower() == message.lower():
                break
            if not text.lower().startswith(message.lower()):
                text = f"{message}: {text}"
            return f"error[{header['code']}]: {text}"
    return header.group(0)


def lower(
    language: str,
    source: str,
    entry: str = "main",
    stage: str = "compute",
    *,
    raw_ids: bool = True,
) -> tuple[str, str]:
    """Compile a probe to SPIR-V and disassemble it: ``(assembly, "")`` on success, ``("", log)`` if not.

    ``raw_ids`` numbers every id instead of naming it after its debug name, so assemblies compare by instruction.
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
        compiler = (
            rustgpu.command(rustgpu.target(stage))
            if tool == "rust-gpu"
            else command(tool)
        )
        compiled = _run([*compiler, *filled], scratch)
        if not compiled.accepted:
            return "", compiled.log
        try:
            result = subprocess.run(
                [
                    *command("spirv-dis"),
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
