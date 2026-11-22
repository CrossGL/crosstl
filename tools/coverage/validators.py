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
    "gles": "glslangValidator",
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
    "gles": ".comp",
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
# The newest shader model the pinned DXC accepts. Feature lists leave out intrinsics that need a newer one, except
# those the Shader Model 6.10 preview compiler checks.
HLSL_SHADER_MODEL = (6, 9)
# HLSL probes of features that need shader model 6.10 are checked for this target, by DXC's Shader Model 6.10 preview
# (v1.10.2605.37), which CI installs next to the pinned release and :func:`command` finds as ``dxc-preview``.
HLSL_PREVIEW = "sm6.10"
HLSL_PREVIEW_SHADER_MODEL = (6, 10)
# HLSL probes of features DXC compiles only to SPIR-V, such as printf, are checked for this target.
HLSL_SPIRV = "spirv"
# Ray-tracing and work-graph node shaders compile as a library, whose ``[shader(...)]`` attributes name the entry
# points.
HLSL_PREFIXES = {
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
}
HLSL_PROFILES = {
    stage: f"{prefix}_{HLSL_SHADER_MODEL[0]}_{HLSL_SHADER_MODEL[1]}"
    for stage, prefix in HLSL_PREFIXES.items()
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
# The SPIR-V version spirv-dis prints in an assembly's header, as CrossTL's generator does, and the highest minor
# version each Vulkan environment accepts (Vulkan 1.1 takes SPIR-V 1.3, Vulkan 1.2 takes 1.5, Vulkan 1.3 takes 1.6).
_SPIRV_VERSION = re.compile(r"^; Version: 1\.(\d+)$", re.M)
_ENVIRONMENTS = ((3, SPIRV_ENVIRONMENT), (5, SPIRV_1_4_ENVIRONMENT), (6, "vulkan1.3"))
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
# The targets other than SPIR-V a Slang probe compiles for, each with the compiler that checks what slangc writes.
# slangc writes HLSL and CUDA source without compiling it, so DXC and nvcc check that source as they check the probes
# of those languages. The CUDA source includes the prelude header that ships with slangc.
SLANG_SOURCE_TARGETS = {"hlsl": "dxc", "cuda": "nvcc"}
# HIP probes compile for RDNA3 (gfx1100), which has the image instructions texture and surface access need; the CDNA
# targets have none. No GPU is needed: hipcc compiles for the named target.
HIP_ARCH = "gfx1100"


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    log: str = ""
    # The tool ran out of time, which says more about the machine's load than about the source.
    timed_out: bool = False


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


# The Mojo line the probes target: the compute job pins mojo-compiler 1.2.0.
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


def toolchain(language: str, target: str = "") -> tuple[str, ...]:
    """The tools that checking ``language`` probes for ``target`` runs."""
    if language == "slang" and target in SLANG_SOURCE_TARGETS:
        return ("slangc", SLANG_SOURCE_TARGETS[target])
    if language == "hlsl" and target == HLSL_PREVIEW:
        return ("dxc-preview",)
    return {
        "spirv": ("spirv-as", "spirv-val"),
        # An imported OpenCL C probe is checked as the HLSL it is emitted to (runner.ROUND_TRIPS).
        "opencl": ("clang", "dxc"),
    }.get(language, (TOOLS[language],))


def missing(language: str, target: str = "") -> str:
    """The first tool that checking ``language`` probes for ``target`` needs and this machine lacks, or ``""``."""
    return next((tool for tool in toolchain(language, target) if not command(tool)), "")


def available(language: str, target: str = "") -> bool:
    return not missing(language, target)


def run(
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
        return Verdict(
            False, f"{arguments[0]} timed out after {TIMEOUT} s", timed_out=True
        )
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


def suffix(language: str, stage: str) -> str:
    """The file suffix of a probe, from which glslang tells its stage. A WebGL kernel probe is a fragment shader."""
    if language in ("glsl", "gles"):
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
    return run([*command("glslangValidator"), name], scratch)


# An OpenGL ES 3.2 shader names one of the GLSL ES versions OpenGL ES 3.x runs.
_GLES_VERSION = re.compile(r"\A(?:\s|//[^\n]*\n)*#\s*version\s+3[012]0\s+es\b")


def _gles(name: str, source: str, stage: str, target: str, scratch: str) -> Verdict:
    """Check a shader with glslang as GLSL ES 3.00, 3.10 or 3.20, as its ``#version`` line names."""
    if not _GLES_VERSION.match(source):
        return Verdict(
            False, "error: the shader does not begin with #version 300, 310 or 320 es"
        )
    return _glsl(name, source, stage, target, scratch)


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


def _air(name: str, scratch: str) -> Verdict:
    """Compile the Metal source ``name`` in ``scratch`` to textual AIR in ``probe.ll``."""
    return run(
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


def air(source: str) -> str:
    """Metal ``source`` as textual AIR, or an empty string when the Metal compiler is not installed or rejects it."""
    if not command("metal"):
        return ""
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        Path(scratch, "probe.metal").write_text(source)
        compiled = _air("probe.metal", scratch)
        return Path(scratch, "probe.ll").read_text() if compiled.accepted else ""


def _metal(name: str, stage: str, scratch: str) -> Verdict:
    """Compile a Metal probe to textual AIR and require an entry point of ``stage``: the compiler accepts a library
    of plain functions, which no pipeline can run."""
    compiled = _air(name, scratch)
    if not compiled.accepted:
        return compiled
    air = Path(scratch, "probe.ll").read_text()
    for pattern in METAL_ENTRIES.get(stage, ()):
        if not re.search(pattern, air, re.MULTILINE):
            return Verdict(False, f"error: the library has no {stage} entry point")
    return compiled


# What Slang's library emits for an operation that its HLSL target cannot express, as for ``debugBreak``. The feature
# list probes such a function for CUDA instead (features.slang.lacking); this catches a comment it does not find.
_UNSUPPORTED_IN_HLSL = re.compile(r"/\*[^*]*not currently supported for HLSL\s*\*/")


def hlsl_profile(stage: str, target: str = "") -> str:
    """The DXC profile an HLSL probe of ``stage`` compiles with for ``target``."""
    model = HLSL_PREVIEW_SHADER_MODEL if target == HLSL_PREVIEW else HLSL_SHADER_MODEL
    return f"{HLSL_PREFIXES[stage]}_{model[0]}_{model[1]}"


# A kernel whose source leaves its workgroup size to the host is written so that a compiler of GLSL, HLSL or Slang
# source reads the size from three macros the host defines, and stops at an #error without them
# (crosstl/translator/codegen/host_workgroup_size_ids.py). Such a translation is compiled as its host compiles it,
# with a size of one in each dimension, as the conformance harness does (tests/conformance/validators.py).
HOST_WORKGROUP_SIZE = "1"


_GLSL_VERSION_LINE = re.compile(r"^[ \t]*#[ \t]*version\b[^\n]*\n", re.M)


def _host_workgroup_macros(source: str) -> tuple[str, ...]:
    from crosstl.translator.codegen.host_workgroup_size_ids import (
        HOST_WORKGROUP_SIZE_MACROS,
    )

    if any(f"defined({macro})" in source for macro in HOST_WORKGROUP_SIZE_MACROS):
        return HOST_WORKGROUP_SIZE_MACROS
    return ()


def host_workgroup_defines(source: str) -> list[str]:
    """The ``-D`` arguments that size a translation whose host chooses its workgroup size, or none for any other."""
    return [
        argument
        for macro in _host_workgroup_macros(source)
        for argument in ("-D", f"{macro}={HOST_WORKGROUP_SIZE}")
    ]


def host_sized_glsl(source: str) -> str:
    """``source`` as an OpenGL host compiles it: with the size macros defined after ``#version``, and a ``#line``
    that keeps the log's line numbers those of ``source``. glslang takes ``-D`` only when it links.
    """
    macros = _host_workgroup_macros(source)
    version = _GLSL_VERSION_LINE.search(source)
    if not macros or version is None:
        return source
    following = source.count("\n", 0, version.end()) + 1
    defines = "".join(f"#define {macro} {HOST_WORKGROUP_SIZE}\n" for macro in macros)
    end = version.end()
    return f"{source[:end]}{defines}#line {following}\n{source[end:]}"


def _dxc(
    name: str, source: str, stage: str, entry: str, scratch: str, target: str = ""
) -> Verdict:
    """Compile an HLSL probe for ``target``: DXIL by default, DXIL at shader model 6.10 with the preview compiler for
    :data:`HLSL_PREVIEW`, and Vulkan SPIR-V, which DXC validates as it writes it, for :data:`HLSL_SPIRV`. A library
    must also declare an entry point of ``stage``: DXC accepts a library of plain functions, which no pipeline can
    run."""
    profile = hlsl_profile(stage, target)
    library = profile.startswith("lib_")
    entry_point = [] if library else ["-E", entry or hlsl_entry(source, stage)]
    kind = HLSL_LIBRARY_KINDS.get(stage, "")
    attribute = rf'\[\s*shader\s*\(\s*"{kind}"\s*\)\s*\]'
    if library and not re.search(attribute, source, re.IGNORECASE):
        return Verdict(False, f"error: the library has no {stage} entry point")
    spirv = ["-spirv", f"-fspv-target-env={SPIRV_ENVIRONMENT}"]
    return run(
        [
            *command(toolchain("hlsl", target)[0]),
            *(spirv if target == HLSL_SPIRV else []),
            "-T",
            profile,
            *entry_point,
            "-HV",
            "2021",
            "-enable-16bit-types",
            *host_workgroup_defines(source),
            name,
            "-Fo",
            "probe.bin",
        ],
        scratch,
    )


# The arguments of the validators (:data:`TOOLS`) that only compile the probe; ``{source}`` is its file name.
COMPILE_ARGUMENTS = {
    "wgsl": ("{source}",),
    "cuda": (f"-arch={CUDA_ARCH}", "-std=c++17", "-c", "{source}", "-o", "probe.o"),
    "hip": (
        f"--offload-arch={HIP_ARCH}",
        "-std=c++17",
        "-c",
        "{source}",
        "-o",
        "probe.o",
    ),
    "opencl": (
        "-x",
        "cl",
        f"-cl-std={OPENCL_STANDARD}",
        "-target",
        OPENCL_TARGET,
        "-Xclang",
        "-finclude-default-header",
        "-fsyntax-only",
        "{source}",
    ),
    # The Mojo compiler type-checks every function it parses, called or not; emitting LLVM IR needs no main.
    "mojo": (
        "build",
        "--emit",
        "llvm",
        "--disable-warnings",
        "{source}",
        "-o",
        "probe.ll",
    ),
}


def _glsl(name: str, source: str, stage: str, target: str, scratch: str) -> Verdict:
    # GL_EXT_ray_tracing and GL_EXT_mesh_shader exist only for SPIR-V 1.4 and later, so those shaders are checked
    # for Vulkan, as are probes whose feature only GLSL for Vulkan has.
    if stage in SPIRV_1_4_STAGES:
        environment = ["--target-env", SPIRV_1_4_ENVIRONMENT]
    elif target == "vulkan":
        environment = ["--target-env", SPIRV_ENVIRONMENT]
    else:
        environment = []
    sized = host_sized_glsl(source)
    if sized != source:
        Path(scratch, name).write_text(sized)
    return run([*command("glslangValidator"), *environment, name], scratch)


def _slang(
    name: str, source: str, stage: str, entry: str, target: str, scratch: str
) -> Verdict:
    # A ``[shader(...)]`` attribute names the entry point and its stage; without one, slangc is told both. For a
    # source target the output must follow the entry point it holds.
    attributed = "[shader(" in source
    entry_point = [
        *(["-entry", entry or "main"] if target or not attributed else []),
        *([] if attributed else ["-stage", SLANG_STAGES.get(stage, stage)]),
    ]
    output = f"probe{SUFFIX[target]}" if target in SLANG_SOURCE_TARGETS else "probe.spv"
    compiled = run(
        [
            *command("slangc"),
            name,
            *entry_point,
            *host_workgroup_defines(source),
            "-target",
            target or SLANG_TARGET,
            "-o",
            output,
        ],
        scratch,
        SLANG_VALIDATION,
        stdout=False,
    )
    if target not in SLANG_SOURCE_TARGETS or not compiled.accepted:
        return compiled
    if target == "cuda":
        arguments = (a.format(source=output) for a in COMPILE_ARGUMENTS["cuda"])
        return run([*command("nvcc"), *arguments], scratch)
    # Where HLSL lacks an operation, Slang's library emits a comment in place of the call.
    generated = Path(scratch, output).read_text()
    dropped = _UNSUPPORTED_IN_HLSL.search(generated)
    if dropped:
        return Verdict(False, f"error: slangc emits {dropped.group(0)}")
    return _dxc(output, generated, stage, "", scratch)


def spirv_environment(assemblies: list[str]) -> str:
    """The oldest Vulkan environment that accepts every one of ``assemblies``, by the SPIR-V version each declares."""
    minor = max(
        (
            int(found.group(1))
            for found in map(_SPIRV_VERSION.search, assemblies)
            if found
        ),
        default=0,
    )
    return next(
        (name for highest, name in _ENVIRONMENTS if minor <= highest),
        _ENVIRONMENTS[-1][1],
    )


def _spirv(name: str, source: str, stage: str, scratch: str) -> Verdict:
    # A module is checked for the Vulkan version its SPIR-V version needs, as slangc's SPIR-V 1.5 is, and the stages
    # that need SPIR-V 1.4 for Vulkan 1.2 at least.
    found = spirv_environment([source])
    if stage in SPIRV_1_4_STAGES and found == SPIRV_ENVIRONMENT:
        found = SPIRV_1_4_ENVIRONMENT
    environment = ["--target-env", found]
    assembled = run(
        [*command("spirv-as"), *environment, name, "-o", "probe.spv"], scratch
    )
    if not assembled.accepted:
        return assembled
    return run([*command("spirv-val"), *environment, "probe.spv"], scratch)


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
        name = "probe" + suffix(language, stage)
        Path(scratch, name).write_text(source)
        if language in COMPILE_ARGUMENTS:
            arguments = (a.format(source=name) for a in COMPILE_ARGUMENTS[language])
            return run([*command(TOOLS[language]), *arguments], scratch)
        if language == "webgl":
            return _essl(name, source, scratch)
        if language == "gles":
            return _gles(name, source, stage, target, scratch)
        if language == "glsl":
            return _glsl(name, source, stage, target, scratch)
        if language == "hlsl":
            return _dxc(name, source, stage, entry, scratch, target)
        if language == "metal":
            return _metal(name, stage, scratch)
        if language == "rust":
            compiler = rustgpu.command(rustgpu.target(stage))
            return run([*compiler, name, "--out-dir", "."], scratch)
        if language == "slang":
            return _slang(name, source, stage, entry, target, scratch)
        return _spirv(name, source, stage, scratch)


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
