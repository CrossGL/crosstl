"""Native checks of translated corpus artifacts, per target and stage.

Each target's own checker decides validity: ``xcrun metal`` for MSL, DXC with validation (signed DXIL) for HLSL,
glslang for desktop GLSL and GLSL ES, and spirv-as plus spirv-val for SPIR-V. Translated programs keep several
stages in one artifact, so every entry point found in the artifact is checked and all of them must pass.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from ..process import run
from ..toolchains import tool
from ..validators import dxil_is_signed
from .manifest import GLSLANG_STAGE, RAY_TRACING

TIMEOUT = 120
COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
DXC_PREFIX = {
    "compute": "cs",
    "vertex": "vs",
    "fragment": "ps",
    "geometry": "gs",
    "tess_control": "hs",
    "tess_eval": "ds",
    "mesh": "ms",
    "task": "as",
}
HLSL_SHADER_ATTRIBUTE = {
    "vertex": "vertex",
    "pixel": "fragment",
    "fragment": "fragment",
    "compute": "compute",
    "geometry": "geometry",
    "hull": "tess_control",
    "domain": "tess_eval",
    "mesh": "mesh",
    "amplification": "task",
}
HLSL_CONVENTIONAL = {
    "VSMain": "vertex",
    "PSMain": "fragment",
    "CSMain": "compute",
    "GSMain": "geometry",
    "HSMain": "tess_control",
    "DSMain": "tess_eval",
    "MSMain": "mesh",
    "ASMain": "task",
}
GLSL_GUARDS = {
    "GL_VERTEX_SHADER": "vertex",
    "GL_FRAGMENT_SHADER": "fragment",
    "GL_COMPUTE_SHADER": "compute",
    "GL_GEOMETRY_SHADER": "geometry",
    "GL_TESS_CONTROL_SHADER": "tess_control",
    "GL_TESS_EVALUATION_SHADER": "tess_eval",
}
ENTRY_PATTERNS = {
    "metal": re.compile(
        r"\b(?:kernel|vertex|fragment)\s+(?:(?:const|static|inline)\s+)*[\w:]+(?:\s*<[^;{}]*?>)?[\s*&]+\w+\s*\("
        r"|\[\[\s*(?:kernel|vertex|fragment|mesh|object|visible)\s*(?:\]\]|,)|\[\[host_name\("
    ),
    "glsl": re.compile(r"\bvoid\s+main\s*\("),
    "essl": re.compile(r"\bvoid\s+main\s*\("),
    "hlsl": re.compile(
        r"\[\s*numthreads\s*\(|\[\s*shader\s*\(|\b(?:main|[VPGHDCMA]SMain)\s*\("
    ),
    "spirv": re.compile(r"^\s*OpEntryPoint\b", re.M),
}
SPIRV_ENVIRONMENT = {"1.4": "vulkan1.2", "1.5": "vulkan1.2", "1.6": "vulkan1.3"}


def count_entries(text, target):
    """Entry points an artifact declares; an artifact without one is not a usable translation."""
    if target != "spirv":
        text = COMMENTS.sub(" ", text)
    return len(ENTRY_PATTERNS[target].findall(text))


def _verdict(results, **extra):
    statuses = [result.status for result in results]
    for status in ("unavailable", "harness-error", "timeout", "rejected"):
        if status in statuses:
            verdict = status
            break
    else:
        verdict = "accepted"
    failing = next((r for r in results if r.status != "accepted"), None)
    return {
        "verdict": verdict,
        "commands": [" ".join(result.command) for result in results],
        "log": (failing.log if failing else "")[-1500:],
        **extra,
    }


def _missing(*names):
    for name in names:
        if not tool(name).available:
            return {"verdict": "unavailable", "commands": [], "log": tool(name).reason}
    return None


def check_metal(path, scratch):
    return _missing("metal") or _verdict(
        [
            run(
                [*tool("metal").command, "-c", path, "-o", Path(scratch) / "unit.air"],
                timeout=TIMEOUT,
            )
        ]
    )


def hlsl_entries(text, stages):
    """(name, stage) of each entry point: attributes, conventional names, else ``main`` for a one-stage unit."""
    text = COMMENTS.sub(" ", text)
    function = r"(?:\[[^\]]*\]\s*)*[\w<>,\s:]+?\s+(\w+)\s*\("
    entries = {}
    for match in re.finditer(r"\[\s*numthreads\s*\([^)]*\)\s*\]\s*" + function, text):
        entries[match.group(1)] = "compute"
    for match in re.finditer(
        r"\[\s*shader\s*\(\s*\"(\w+)\"\s*\)\s*\]\s*" + function, text
    ):
        entries[match.group(2)] = HLSL_SHADER_ATTRIBUTE.get(
            match.group(1).lower(), match.group(1).lower()
        )
    for name, stage in HLSL_CONVENTIONAL.items():
        if name not in entries and re.search(rf"\b{name}\s*\(", text):
            entries[name] = stage
    if not entries and len(stages) == 1 and re.search(r"\bmain\s*\(", text):
        entries["main"] = stages[0]
    return sorted(entries.items())


def check_hlsl(path, scratch, stages):
    missing = _missing("dxc")
    if missing:
        return missing
    text = Path(path).read_text(errors="replace")
    extra = (
        ["-enable-16bit-types"]
        if re.search(r"\b(?:u?int16_t|float16_t)\b", text)
        else []
    )
    entries = hlsl_entries(text, stages)
    jobs = [
        (
            (name, f"{DXC_PREFIX[stage]}_6_6", ["-E", name])
            if stage in DXC_PREFIX
            else (name, "lib_6_6", ["-default-linkage", "external"])
        )
        for name, stage in entries
    ] or [("library", "lib_6_6", ["-default-linkage", "external"])]
    results = []
    for index, (name, profile, arguments) in enumerate(jobs):
        output = Path(scratch) / f"entry{index}.dxil"
        result = run(
            [
                *tool("dxc").command,
                path,
                "-T",
                profile,
                *arguments,
                "-HV",
                "2021",
                *extra,
                "-Fo",
                output,
            ],
            timeout=TIMEOUT,
        )
        if result.status == "accepted" and not (
            output.is_file() and dxil_is_signed(output)
        ):
            result = type(result)(
                result.command,
                "rejected",
                result.returncode,
                result.log + "\nunsigned DXIL",
            )
        results.append(result)
        if result.status != "accepted":
            break
    return _verdict(results, entries=[name for name, _, _ in jobs])


def glsl_environment(stages):
    """glslang options for the API a GLSL artifact's stages exist in.

    Ray tracing stages exist only in Vulkan: GL_EXT_ray_tracing is defined for SPIR-V generation, and glslang
    rejects the extension under OpenGL rules. They are checked with Vulkan 1.2 rules, the first Vulkan whose SPIR-V
    (1.5) includes SPV_KHR_ray_tracing's 1.4 requirement. Every other stage is checked with the OpenGL or OpenGL ES
    rules the artifact's ``#version`` selects.
    """
    if stages and set(stages) <= RAY_TRACING:
        return ["--target-env", "vulkan1.2"]
    return []


def _glsl_command(path, stage, scratch):
    environment = glsl_environment([stage])
    # Vulkan rules make glslang write SPIR-V, by default to ``<stage>.spv`` in the working directory.
    output = ["-o", str(Path(scratch) / f"{stage}.spv")] if environment else []
    return [
        *tool("glslangValidator").command,
        *environment,
        *output,
        "-S",
        GLSLANG_STAGE[stage],
        path,
    ]


def check_glsl(path, stages):
    """glslang with the rules of the API the stages exist in (:func:`glsl_environment`)."""
    missing = _missing("glslangValidator")
    if missing:
        return missing
    text = Path(path).read_text(errors="replace")
    guarded = [
        stage
        for guard, stage in GLSL_GUARDS.items()
        if re.search(rf"#\s*if(?:def)?\s+(?:defined\s*\(\s*)?{guard}\b", text)
    ]
    selected = guarded or list(stages)
    results = []
    with tempfile.TemporaryDirectory(prefix="crosstl-glslang-") as scratch:
        for stage in selected:
            result = run(_glsl_command(path, stage, scratch), timeout=TIMEOUT)
            results.append(result)
            if result.status != "accepted":
                break
    return _verdict(results, entries=selected)


def spirv_environment(text):
    match = re.search(r"^; Version: (\d+\.\d+)", text, re.M)
    return SPIRV_ENVIRONMENT.get(match.group(1), "vulkan1.1") if match else "vulkan1.1"


def check_spirv(path, scratch):
    missing = _missing("spirv-as", "spirv-val")
    if missing:
        return missing
    environment = spirv_environment(Path(path).read_text(errors="replace"))
    binary = Path(scratch) / "unit.spv"
    results = [
        run(
            [
                *tool("spirv-as").command,
                "--target-env",
                environment,
                path,
                "-o",
                binary,
            ],
            timeout=TIMEOUT,
        )
    ]
    if results[0].status == "accepted":
        results.append(
            run(
                [*tool("spirv-val").command, "--target-env", environment, binary],
                timeout=TIMEOUT,
            )
        )
    return _verdict(results, environment=environment)


# GLSL ES requires #version before anything else, comments included (GLSL ES 3.00 section 3.4).
ESSL_VERSION = re.compile(r"\A#[ \t]*version[ \t]+300[ \t]+es\b")


def check_essl(path, stages):
    """WebGL 2 compiles GLSL ES 3.00 only; glslang would judge a desktop ``#version`` by desktop rules instead."""
    if not ESSL_VERSION.match(Path(path).read_text(errors="replace")):
        return {
            "verdict": "rejected",
            "commands": [],
            "log": "the WebGL 2 artifact does not begin with #version 300 es",
        }
    return check_glsl(path, stages)


def check(path, target, stages):
    """Validate one artifact; returns verdict (accepted, rejected, timeout, unavailable, harness-error) and log."""
    with tempfile.TemporaryDirectory() as scratch:
        if target == "metal":
            return check_metal(path, scratch)
        if target == "hlsl":
            return check_hlsl(path, scratch, stages)
        if target == "essl":
            return check_essl(path, stages)
        if target == "glsl":
            return check_glsl(path, stages)
        if target == "spirv":
            return check_spirv(path, scratch)
    raise ValueError(f"Unknown corpus target {target!r}")
