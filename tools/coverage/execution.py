"""Run supported compute probes on a GPU runtime and compare what they compute with the native probe.

A translation the reference tool accepts can still compute something else. Where Core's conformance harness
(``tests/conformance``) has a runtime for a language, the native probe and CrossTL's translation run there on the
same inputs, and a translation that writes other results, writes outside its output buffer, or that the runtime
rejects is wrong.

HLSL, Slang and WGSL have no runtime on the coverage hosts, so both probes run as the SPIR-V that the language's
reference compiler (DXC, slangc or naga) writes for them, on the harness's Vulkan runtime. The two go through the
same compiler, so a difference between them comes from the translation. A probe the compiler cannot lower keeps its
validation result, and so does one whose SPIR-V declares a 16-bit float, because lavapipe skips rounding through it.
The harness assembles and validates that SPIR-V for the oldest Vulkan version whose SPIR-V version is at least the
one the compiler wrote: slangc writes SPIR-V 1.5, which Vulkan 1.1 does not accept.

The harness binds one input buffer at slot 0 and one output buffer at slot 1, so only compute probes that declare one
input buffer run; a probe whose call takes no argument declares one it never reads. The other probes keep their
validation result, and so does one that reads a clock or queries a ray query, because two runs of it need not agree.
Integer inputs are small and non-negative, so no probe shifts, extracts or inserts bits outside what every language
defines.
"""

from __future__ import annotations

import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import validators
from .model import Result

# The languages the harness runs, with the suffix it reads each one's sources from (tests/conformance/cases.py).
SUFFIXES = {"glsl": ".comp", "metal": ".metal", "spirv": ".spvasm"}
# The languages that run as the SPIR-V their reference compiler lowers them to.
VIA_SPIRV = frozenset({"hlsl", "slang", "wgsl"})
THREADS = 64
# Output words one thread may write: a 4x4 float matrix is the largest result a probe stores.
WORDS = 16
# Input elements one thread may read past its own: three 4x4 matrix arguments.
SPAN = 48
_INPUT = re.compile(r"\b(fin|iin|uin)\b")
# llvmpipe implements OpenGL 4.5, so GLSL written for 4.60 runs as GLSL 4.50. A native probe that needs 4.60 is then
# rejected by the driver and keeps its validation result, and so does a translation that declares 4.60, since it may
# use what 4.60 adds.
_GLSL_460 = re.compile(r"^#version 460\b.*$", re.M)
# lavapipe drops a conversion of a 32-bit float to 16 bits that is converted straight back, so a native probe that
# rounds through a 16-bit float writes its unrounded input there. Probes whose SPIR-V declares a 16-bit float keep
# their validation result on every Vulkan runtime, so a verdict does not depend on the runtime that ran it.
_FLOAT16 = re.compile(r"\bOpTypeFloat 16\b")
# The SPIR-V version spirv-dis prints in an assembly's header, and the highest minor version each Vulkan environment
# of the harness accepts (Vulkan 1.1 takes SPIR-V 1.3, Vulkan 1.2 takes 1.5, Vulkan 1.3 takes 1.6).
_SPIRV_VERSION = re.compile(r"^; Version: 1\.(\d+)$", re.M)
_ENVIRONMENTS = ((3, "vulkan1.1"), (5, "vulkan1.2"), (6, "vulkan1.3"))
# Two runs of a probe need not agree when it reads a clock, or queries a ray query: the harness binds no acceleration
# structure, so the query is never initialized and SPV_KHR_ray_query leaves what it returns undefined. Such probes
# keep their validation result.
_UNREPEATABLE = re.compile(r"\b(OpReadClockKHR|OpTypeRayQueryKHR)\b")
# Harness outcomes of a translation that ran, or failed to, where its native probe ran.
_BROKEN = {
    "memory-corruption": "writes outside its output buffer or into its input buffer",
    "driver-rejected": "is rejected by the runtime",
}
DIVERGED = "computes different results from the native probe"


def found(detail: str) -> bool:
    """Whether the diagnostic ``detail`` of a wrong result comes from running its probe."""
    first = detail.partition("\n")[0]
    return first.startswith("on ") and any(
        f"the translation {text}" in first for text in (DIVERGED, *_BROKEN.values())
    )


def executable(result: Result) -> bool:
    """Whether ``result`` is a supported compute probe whose native probe and translation declare one input buffer."""
    if result.outcome != "supported" or len(result.sources) != 2:
        return False
    buffers = [set(_INPUT.findall(source)) for source in result.sources]
    return len(buffers[0]) == 1 and buffers[0] == buffers[1]


def runnable(language: str, source: str) -> str:
    """``source`` as the runtime for ``language`` compiles it."""
    if language == "glsl":
        return _GLSL_460.sub("#version 450 core", source, count=1)
    return source


def runs(language: str) -> bool:
    """Whether the harness runs probes in ``language``, directly or as SPIR-V."""
    return language in SUFFIXES or language in VIA_SPIRV


def target(language: str) -> str:
    """The language the harness runs ``language``'s probes as."""
    return "spirv" if language in VIA_SPIRV else language


def as_spirv(language: str, source: str) -> str:
    """The SPIR-V assembly ``language``'s reference compiler writes for a compute ``source``, or an empty string when
    it rejects the source."""
    # naga takes no entry point; HLSL and Slang name theirs after [numthreads(...)].
    entry = "main" if language == "wgsl" else validators.hlsl_entry(source, "compute")
    assembly, _ = validators.lower(language, source, entry, keep_bindings=True)
    return assembly


def environment(assemblies: list[str]) -> str:
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


def inputs(buffer: str):
    """The input every probe reading ``buffer`` gets, the same on every run."""
    import numpy as np

    count = THREADS + SPAN
    generator = np.random.default_rng(0)
    if buffer == "fin":
        return generator.uniform(-4.0, 4.0, count).astype(np.float32)
    dtype = np.int32 if buffer == "iin" else np.uint32
    return generator.integers(0, 16, count).astype(dtype)


def difference(native, translated) -> str:
    """How the output words ``translated`` differ from ``native``, or an empty string when they match.

    Two words that are both NaN as 32-bit floats match, whatever their payload.
    """
    import numpy as np

    floats = (native.view(np.float32), translated.view(np.float32))
    differ = (native != translated) & ~(np.isnan(floats[0]) & np.isnan(floats[1]))
    if not differ.any():
        return ""
    word = int(np.argmax(differ))
    return (
        f"{DIVERGED}\n{int(differ.sum())} output words differ. The first, word {word}, is "
        f"{int(translated[word]):#010x}; the native probe writes {int(native[word]):#010x}."
    )


def _case(identity: str, path: Path, language: str, data, capture):
    import numpy as np

    from tests.conformance.cases import Case

    def check(_, output):
        capture[identity] = np.array(output)
        return True, {}

    return Case(
        id=identity,
        cls="coverage",
        title=identity,
        rule="coverage probe",
        inputs=data,
        out_dtype=np.uint32,
        threads=THREADS,
        group=1,
        out_count=THREADS * WORDS,
        check=check,
        src=path,
        lang=language,
    )


def _execute(language: str, chunk: list[tuple[int, Result]], selection, root: Path):
    """Run one chunk of results; return ``{index: (native record, translated record, outputs, lowered)}``.

    ``lowered`` says whether the runtime compiles the translation at a lower language version than it declares.
    """
    from tests.conformance import runner

    runs_as = target(language)
    suffix, cases, outputs, lowered = SUFFIXES[runs_as], [], {}, set()
    assemblies = []
    for index, result in chunk:
        [buffer] = set(_INPUT.findall(result.sources[0]))
        data = inputs(buffer)
        texts = [runnable(language, source) for source in result.sources]
        if language in VIA_SPIRV:
            texts = [as_spirv(language, text) for text in texts]
            if not all(texts):
                continue
        elif texts[1] != result.sources[1]:
            lowered.add(index)
        if runs_as == "spirv" and any(
            _FLOAT16.search(text) or _UNREPEATABLE.search(text) for text in texts
        ):
            continue
        if language in VIA_SPIRV:
            assemblies += texts
        for role, text in zip(("native", "translated"), texts):
            path = root / "sources" / f"{index:05d}-{role}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            cases.append(_case(f"{index:05d}-{role}", path, runs_as, data, outputs))
    if not cases:
        return {}
    # SPIR-V that a compiler wrote is assembled for the oldest environment that takes its version; a SPIR-V probe,
    # which declares none, for Vulkan 1.1 as the validators check it.
    records = runner.run_suite(
        cases,
        [runs_as],
        ["strict"],
        selection,
        root,
        reference=True,
        environment=environment(assemblies),
    )
    found = {}
    for record in records:
        index, role = record["case"].split("-")
        found.setdefault(int(index), {})[role] = record
    return {
        index: (roles.get("native"), roles.get("translated"), outputs, index in lowered)
        for index, roles in found.items()
    }


def _judge(
    result: Result, native, translated, outputs, lowered: bool, index: int
) -> Result:
    if not native or not translated or native["outcome"] != "preserved":
        return result
    runtime = native["runtime"]
    if lowered and translated["outcome"] == "driver-rejected":
        return result
    language = result.feature.language
    where = f"on {runtime}" + (
        f" as SPIR-V from {validators.TOOLS[language]}" if language in VIA_SPIRV else ""
    )
    if translated["outcome"] in _BROKEN:
        detail = f"{where}, the translation {_BROKEN[translated['outcome']]}"
        return replace(result, outcome="wrong", detail=detail, runtime=runtime)
    if translated["outcome"] != "preserved":
        return result
    found = difference(
        outputs[f"{index:05d}-native"], outputs[f"{index:05d}-translated"]
    )
    if found:
        first, _, rest = found.partition("\n")
        detail = f"{where}, the translation {first}\n{rest}"
        return replace(result, outcome="wrong", detail=detail, runtime=runtime)
    return replace(result, runtime=runtime)


def run(results: list[Result], runtimes: str = "available", workers: int = 4):
    """Execute every :func:`executable` result where a runtime is available; return the results and the runtimes.

    ``runtimes`` is the harness's selection: ``available``, or a comma-separated list of runtime ids.
    """
    from tests.conformance import runner

    results = list(results)
    by_language: dict[str, list[tuple[int, Result]]] = {}
    for index, result in enumerate(results):
        language = result.feature.language
        if runs(language) and executable(result):
            by_language.setdefault(language, []).append((index, result))
    used = set()
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-execution-") as scratch:
        for language, items in by_language.items():
            if language in VIA_SPIRV and not validators.can_lower(language):
                continue
            runs_as = target(language)
            selection = runner.select_runtimes(runtimes, [runs_as])
            available = [pair for pair in selection[runs_as] if pair[1].available]
            if not available:
                continue
            # One runtime per language keeps a result's verdict independent of how many devices the host has.
            selection = {runs_as: available[:1]}
            chunks = [items[n::workers] for n in range(workers) if items[n::workers]]
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                done = pool.map(
                    lambda pair: _execute(
                        language,
                        pair[1],
                        selection,
                        Path(scratch, language, str(pair[0])),
                    ),
                    enumerate(chunks),
                )
                for found in done:
                    for index, records in found.items():
                        results[index] = _judge(results[index], *records, index)
            used.add(available[0][0].id)
    return results, sorted(used)


def idle(results: list[Result]) -> list[str]:
    """The languages with :func:`executable` results of which none ran.

    A runtime that rejects every native probe of a language, as one does for SPIR-V newer than it is told to accept,
    runs none of them, just as a missing runtime does.
    """
    ran = {r.feature.language for r in results if r.runtime}
    return sorted({r.feature.language for r in results if executable(r)} - ran)
