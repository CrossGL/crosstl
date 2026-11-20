"""Run supported compute probes on a GPU runtime and compare what they compute with the native probe.

A translation the reference tool accepts can still compute something else. Where Core's conformance harness
(``tests/conformance``) has a runtime for a language, the native probe and CrossTL's translation run there on the
same inputs, and a translation that writes other results, writes outside its output buffer, binds other resources
than the native probe, or that the runtime rejects is wrong. The diagnostic of a rejected translation quotes the first
lines of the runtime's message. The harness binds both probes the way it finds the native probe's buffers bound, so a
translation that turns those buffers into something else, such as pointers in a uniform block, cannot run in the native
probe's place.

HLSL, Slang, WGSL and Rust have no runtime on the coverage hosts, so both probes run as the SPIR-V that the
language's reference compiler (DXC, slangc, naga or rust-gpu) writes for them, on the harness's Vulkan runtime. The
two go through the same compiler, so a difference between them comes from the translation. A probe the compiler cannot
lower keeps its validation result, and so does one whose SPIR-V declares a 16-bit float, because lavapipe skips
rounding through it. rust-gpu leaves out a storage buffer the probe never reads, so a Rust probe of a call that takes
no argument binds only its output and keeps its validation result too.
DXC declares a counter buffer for each RWStructuredBuffer a function takes as a parameter; the source declares no such
buffer, so one that nothing uses is left out of the SPIR-V instead of asking the host for a binding.
The harness assembles and validates that SPIR-V for the oldest Vulkan version whose SPIR-V version is at least the
one the compiler wrote: slangc writes SPIR-V 1.5, which Vulkan 1.1 does not accept.

GLSL runs on llvmpipe's OpenGL, which lacks extensions such as the explicit arithmetic types. When it rejects a native
probe, or a translation that it compiles at a lower version than the translation declares, both probes run again as
the SPIR-V glslang writes for them, on the Vulkan runtime, and that run decides the result.

Each probe runs :data:`THREADS` invocations, each in a workgroup of its own, except a subgroup operation's, which
runs them as one workgroup (:mod:`.lanes`).
The harness binds input buffers in declaration order at slots 0 through n-1, followed by the output at slot n.
A SPIR-V module's names are debug information without semantic impact, which a producer may drop (SPIR-V 1.6,
section 3.42.2, ``OpName``), and CrossTL's generator renames a buffer block's member. So the inputs of a SPIR-V
probe are its storage buffers in binding order, each known by the scalar type of its runtime array, and the last
binding is the output. A probe whose call takes no argument declares one input it never reads. When the native probe
also declares one uniform or constant buffer, the harness fills it with :data:`CONSTANTS` for both probes, so the
translation must declare that block too. Probes with different ordered inputs keep their validation result, and so does
one that reads a clock or queries a ray query, because two runs of it need not agree.
Integer inputs are small and non-negative, so no probe shifts, extracts or inserts bits outside what every language
defines.
"""

from __future__ import annotations

import re
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import lanes, lowering, validators
from .model import Result

# The languages the harness runs, with the suffix it reads each one's sources from (tests/conformance/cases.py).
SUFFIXES = {"glsl": ".comp", "metal": ".metal", "spirv": ".spvasm"}
# The languages that run as the SPIR-V their reference compiler lowers them to.
VIA_SPIRV = frozenset({"hlsl", "rust", "slang", "wgsl"})
THREADS = 64
# Output words one thread may write: a 4x4 float matrix is the largest result a probe stores.
WORDS = 16
# Input elements one thread may read past its own: three 4x4 matrix arguments.
SPAN = 48
_INPUT = re.compile(r"\b(fin|iin|uin)\b")
_SPIRV_ENTRY = re.compile(r"^\s*OpEntryPoint\b", re.M)
_BINDING = re.compile(r"^\s*OpDecorate (%\S+) Binding (\d+)\s*$", re.M)
# The input each scalar type of a SPIR-V storage buffer gets.
_SCALAR_INPUTS = {
    "OpTypeFloat 32": "fin",
    "OpTypeInt 32 1": "iin",
    "OpTypeInt 32 0": "uin",
}
# The bytes of the constants block a probe declares: sixteen 32-bit floats, four 16-byte std140 rows. A probe reads the
# first row, the first word of each row, or all four rows as a 4x4 matrix. The words differ so that a translation
# reading another offset computes something else.
CONSTANTS = struct.pack(
    "<16f",
    *(2.5, -1.75, 3.25, 0.5),
    *(1.5, -0.25, 4.0, 0.75),
    *(-2.0, 1.25, 0.125, -3.5),
    *(2.75, 0.375, -1.0, 5.0),
)
# llvmpipe implements OpenGL 4.5, so GLSL written for 4.60 runs as GLSL 4.50. A native probe that needs 4.60 is then
# rejected by the driver and keeps its validation result, and so does a translation that declares 4.60, since it may
# use what 4.60 adds.
_GLSL_460 = re.compile(r"^#version 460\b.*$", re.M)
# GLSL 4.60 took its group vote functions from ARB_shader_group_vote, which llvmpipe implements. At 4.50 they need
# that extension, which names them with an ARB suffix.
_GROUP_VOTE = re.compile(
    r"\b(anyInvocation|allInvocations|allInvocationsEqual)(?=\s*\()"
)
# lavapipe drops a conversion of a 32-bit float to 16 bits that is converted straight back, so a native probe that
# rounds through a 16-bit float writes its unrounded input there. Probes whose SPIR-V declares a 16-bit float keep
# their validation result on every Vulkan runtime, so a verdict does not depend on the runtime that ran it.
_FLOAT16 = re.compile(r"\bOpTypeFloat 16\b")
# Two runs of a probe need not agree when it reads a clock, or queries a ray query: the harness binds no acceleration
# structure, so the query is never initialized and SPV_KHR_ray_query leaves what it returns undefined. Such probes
# keep their validation result.
_UNREPEATABLE = re.compile(r"\b(OpReadClockKHR|OpTypeRayQueryKHR)\b")
DIVERGED = "computes different results from the native probe"
REBOUND = "binds other resources than the native probe"
REJECTED = "is rejected by the runtime"
# Harness outcomes of a translation that ran, or failed to, where its native probe ran.
_BROKEN = {
    "memory-corruption": "writes outside its output buffer or into its input buffer",
    "driver-rejected": REJECTED,
}
# How much of the runtime's message for a rejected translation its diagnostic quotes, after the first line.
_QUOTED_LINES = 6
_QUOTED_CHARS = 600
# Harness outcomes of a probe the runtime would not compile.
_UNCOMPILED = frozenset({"invalid", "driver-rejected"})


def found(detail: str) -> bool:
    """Whether the diagnostic ``detail`` of a wrong result comes from running its probe."""
    first = detail.partition("\n")[0]
    return first.startswith("on ") and any(
        f"the translation {text}" in first
        for text in (DIVERGED, REBOUND, *_BROKEN.values())
    )


def _quoted(record: dict) -> str:
    """The first lines of what the runtime said when it rejected the artifact of the harness ``record``."""
    message = (record.get("execution") or {}).get("message") or ""
    lines = [line.rstrip() for line in message.splitlines() if line.strip()]
    text = "\n".join(lines[:_QUOTED_LINES])
    if len(text) > _QUOTED_CHARS:
        text = text[: _QUOTED_CHARS - 1].rstrip() + "…"
    return text


def input_buffers(source: str) -> tuple[str, ...]:
    """Probe input names in declaration order, before their uses in the entry point, or in binding order for SPIR-V."""
    if _SPIRV_ENTRY.search(source):
        return _spirv_inputs(source)
    return tuple(dict.fromkeys(_INPUT.findall(source)))


def _spirv_inputs(assembly: str) -> tuple[str, ...]:
    """The inputs of a SPIR-V module: the variables it binds to a block of one runtime array, in binding order and
    without the last one, the output. Each is named for the array's scalar type as :func:`inputs` fills it, or for
    the instruction that declares any other element type."""
    defined = {}
    for line in assembly.splitlines():
        found = lowering.DEFINITION.match(line)
        if found:
            defined[found[1]] = (found[2], found[3].split())

    def operand(ident: str, opcode: str, position: int) -> str:
        op, words = defined.get(ident, ("", []))
        return words[position] if op == opcode and words else ""

    buffers = []
    for variable, slot in _BINDING.findall(assembly):
        block = operand(operand(variable, "OpVariable", 0), "OpTypePointer", -1)
        op, members = defined.get(block, ("", []))
        array = members[0] if op == "OpTypeStruct" and len(members) == 1 else ""
        scalar = operand(array, "OpTypeRuntimeArray", 0)
        if scalar in defined:
            kind = " ".join((defined[scalar][0], *defined[scalar][1]))
            buffers.append((int(slot), _SCALAR_INPUTS.get(kind, kind)))
    return tuple(name for _, name in sorted(buffers)[:-1])


def executable(result: Result) -> bool:
    """Whether ``result`` is a supported compute probe whose native probe and translation declare the same ordered
    inputs, each one :func:`inputs` fills."""
    if result.outcome != "supported" or len(result.sources) != 2:
        return False
    buffers = [input_buffers(source) for source in result.sources]
    known = set(_SCALAR_INPUTS.values())
    return (
        bool(buffers[0]) and buffers[0] == buffers[1] and known.issuperset(buffers[0])
    )


def runnable(language: str, source: str) -> str:
    """``source`` as the runtime for ``language`` compiles it."""
    if language != "glsl" or not _GLSL_460.search(source):
        return source
    version = "#version 450 core"
    if _GROUP_VOTE.search(source):
        version += "\n#extension GL_ARB_shader_group_vote : require"
        source = _GROUP_VOTE.sub(r"\1ARB", source)
    return _GLSL_460.sub(version, source, count=1)


def runs(language: str) -> bool:
    """Whether the harness runs probes in ``language``, directly or as SPIR-V."""
    return language in SUFFIXES or language in VIA_SPIRV


def as_spirv(language: str, source: str) -> str:
    """The SPIR-V assembly ``language``'s reference compiler writes for a compute ``source``, or an empty string when
    it rejects the source."""
    # GLSL's entry point is main, and naga and rust-gpu take none; HLSL and Slang name theirs after [numthreads(...)].
    entry = (
        "main"
        if language in ("glsl", "rust", "wgsl")
        else validators.hlsl_entry(source, "compute")
    )
    assembly, _ = lowering.lower(language, source, entry, keep_bindings=True)
    if language == "hlsl":
        return lowering.without_unused_counters(assembly)
    if language == "rust" and assembly:
        return lowering.without_unused_capabilities(assembly)
    return assembly


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


def _case(
    identity: str, path: Path, language: str, data, constants, capture, group: int
):
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
        group=group,
        out_count=THREADS * WORDS,
        check=check,
        constants=constants,
        src=path,
        lang=language,
    )


def declares_constants(runs_as: str, path: Path, *, input_count: int = 1) -> bool:
    """Whether the probe at ``path`` declares the one constants block the harness can bind, besides its buffers."""
    from tests.conformance.interface import InterfaceChanged, interface

    try:
        interface(path, runs_as, 1, True, input_count=input_count)
    except InterfaceChanged:
        return False
    return True


def rebinding(
    runs_as: str,
    native: Path,
    translated: Path,
    constants: bool,
    *,
    input_count: int = 1,
) -> str:
    """Why the harness cannot bind the ``translated`` probe the way it binds the ``native`` one, or an empty string
    when it can, or when it cannot bind the native probe either. ``constants`` says whether both get a constants
    block. ``input_count`` is the number of storage inputs the host supplies to both probes.
    """
    from tests.conformance.interface import InterfaceChanged, interface

    objections = []
    for path in (native, translated):
        try:
            interface(path, runs_as, 1, constants, input_count=input_count)
            objections.append("")
        except InterfaceChanged as error:
            objections.append(str(error))
    return "" if objections[0] else objections[1]


def _execute(
    language: str,
    chunk: list[tuple[int, Result]],
    selection,
    root: Path,
    via_spirv: bool,
):
    """Run one chunk of results, as SPIR-V when ``via_spirv``; return ``{index: (native record, translated record,
    outputs, lowered, rebound, via_spirv)}``.

    ``lowered`` says whether the runtime compiles the translation at a lower language version than it declares, and
    ``rebound`` is :func:`rebinding`'s objection to the translation's resources.
    """
    from tests.conformance import runner

    runs_as = "spirv" if via_spirv else language
    suffix, cases, outputs, lowered = SUFFIXES[runs_as], [], {}, set()
    rebound = {}
    assemblies = []
    for index, result in chunk:
        data = tuple(inputs(buffer) for buffer in input_buffers(result.sources[0]))
        if via_spirv:
            texts = [as_spirv(language, text) for text in result.sources]
            if not all(texts):
                continue
        else:
            texts = [runnable(language, source) for source in result.sources]
            if texts[1] != result.sources[1]:
                lowered.add(index)
        if runs_as == "spirv" and any(
            _FLOAT16.search(text) or _UNREPEATABLE.search(text) for text in texts
        ):
            continue
        if via_spirv:
            assemblies += texts
        paths = []
        for role, text in zip(("native", "translated"), texts):
            path = root / "sources" / f"{index:05d}-{role}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            paths.append(path)
        block = declares_constants(runs_as, paths[0], input_count=len(data))
        constants = CONSTANTS if block else None
        # A subgroup operation's probes run as one workgroup (:mod:`.lanes`), every other probe one per invocation.
        group = THREADS if lanes.cooperative(result.feature) else 1
        for role, path in zip(("native", "translated"), paths):
            cases.append(
                _case(
                    f"{index:05d}-{role}",
                    path,
                    runs_as,
                    data,
                    constants,
                    outputs,
                    group,
                )
            )
        rebound[index] = rebinding(runs_as, *paths, block, input_count=len(data))
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
        environment=validators.spirv_environment(assemblies),
    )
    found = {}
    for record in records:
        index, role = record["case"].split("-")
        found.setdefault(int(index), {})[role] = record
    return {
        index: (
            roles.get("native"),
            roles.get("translated"),
            outputs,
            index in lowered,
            rebound.get(index, ""),
            via_spirv,
        )
        for index, roles in found.items()
    }


def _judge(
    result: Result,
    native,
    translated,
    outputs,
    lowered: bool,
    rebound: str,
    via_spirv: bool,
    index: int,
) -> Result:
    if not native or not translated or native["outcome"] != "preserved":
        return result
    runtime = native["runtime"]
    if lowered and translated["outcome"] == "driver-rejected":
        return result
    language = result.feature.language
    where = f"on {runtime}" + (
        f" as SPIR-V from {validators.TOOLS[language]}" if via_spirv else ""
    )
    if rebound:
        detail = f"{where}, the translation {REBOUND}\n{rebound}"
        return replace(result, outcome="wrong", detail=detail, runtime=runtime)
    if translated["outcome"] in _BROKEN:
        detail = f"{where}, the translation {_BROKEN[translated['outcome']]}"
        message = _quoted(translated)
        if message:
            detail += "\n" + message
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


def _unjudged(native, translated, outputs, lowered: bool, *_) -> bool:
    """Whether a run on llvmpipe's OpenGL left no verdict because the runtime would not compile the native probe, or a
    translation it compiled at a lower version than the translation declares."""
    if not native or not translated:
        return False
    return native["outcome"] in _UNCOMPILED or (
        lowered and translated["outcome"] == "driver-rejected"
    )


def _pass(
    language: str,
    items: list[tuple[int, Result]],
    runtimes: str,
    root: Path,
    workers: int,
    via_spirv: bool,
):
    """Run ``items`` on the first available runtime; return ``({index: records}, runtime id)``, with an empty id when
    there is nothing to run or no runtime to run it on."""
    from tests.conformance import runner

    if not items or (via_spirv and not lowering.can_lower(language)):
        return {}, ""
    runs_as = "spirv" if via_spirv else language
    selection = runner.select_runtimes(runtimes, [runs_as])
    available = [pair for pair in selection[runs_as] if pair[1].available]
    if not available:
        return {}, ""
    # One runtime per language keeps a result's verdict independent of how many devices the host has.
    selection = {runs_as: available[:1]}
    chunks = [items[n::workers] for n in range(workers) if items[n::workers]]
    found = {}
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        done = pool.map(
            lambda pair: _execute(
                language, pair[1], selection, root / str(pair[0]), via_spirv
            ),
            enumerate(chunks),
        )
        for records in done:
            found.update(records)
    return found, available[0][0].id


def run(results: list[Result], runtimes: str = "available", workers: int = 4):
    """Execute every :func:`executable` result where a runtime is available; return the results and the runtimes.

    ``runtimes`` is the harness's selection: ``available``, or a comma-separated list of runtime ids.
    """
    results = list(results)
    by_language: dict[str, list[tuple[int, Result]]] = {}
    for index, result in enumerate(results):
        language = result.feature.language
        if runs(language) and executable(result):
            by_language.setdefault(language, []).append((index, result))
    used = set()
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-execution-") as scratch:
        for language, items in by_language.items():
            via_spirv = language in VIA_SPIRV
            found, runtime = _pass(
                language, items, runtimes, Path(scratch, language), workers, via_spirv
            )
            used.add(runtime)
            if language == "glsl":
                again = [
                    (i, r) for i, r in items if i in found and _unjudged(*found[i])
                ]
                rerun, runtime = _pass(
                    language,
                    again,
                    runtimes,
                    Path(scratch, "glsl-spirv"),
                    workers,
                    True,
                )
                found.update(rerun)
                if rerun:
                    used.add(runtime)
            for index, records in found.items():
                results[index] = _judge(results[index], *records, index)
    return results, sorted(used - {""})


def idle(results: list[Result]) -> list[str]:
    """The languages with :func:`executable` results of which none ran.

    A runtime that rejects every native probe of a language, as one does for SPIR-V newer than it is told to accept,
    runs none of them, just as a missing runtime does.
    """
    ran = {r.feature.language for r in results if r.runtime}
    return sorted({r.feature.language for r in results if executable(r)} - ran)
