"""Run supported compute probes on a GPU runtime and compare what they compute with the native probe.

A translation the reference tool accepts can still compute something else. Where Core's conformance harness
(``tests/conformance``) has a runtime for a language, the native probe and CrossTL's translation run there on the
same inputs, and a translation that writes other results, writes outside its output buffer, binds other resources
than the native probe, or that the runtime rejects is wrong. The diagnostic of a rejected translation quotes the first
lines of the runtime's message. The harness binds both probes the way it finds the native probe's buffers bound, so a
translation that turns those buffers into something else, such as pointers in a uniform block, cannot run in the native
probe's place.

Slang, WGSL and Rust have no runtime on the coverage hosts, so both probes run as the SPIR-V that the language's
reference compiler (slangc, naga or rust-gpu) writes for them, on the harness's Vulkan runtime. So do HLSL probes on a
host without the harness's Direct3D 12 runtime, as the SPIR-V DXC writes. The two go through the same compiler, so a
difference between them comes from the translation. A probe the compiler cannot
lower keeps its validation result, and so does one whose SPIR-V declares a 16-bit float, because lavapipe skips
rounding through it. rust-gpu leaves out a storage buffer the probe never reads, so a Rust probe of a call that takes
no argument binds only its output and keeps its validation result too.
DXC declares a counter buffer for each RWStructuredBuffer a function takes as a parameter; the source declares no such
buffer, so one that nothing uses is left out of the SPIR-V instead of asking the host for a binding.
The harness assembles and validates that SPIR-V for the oldest Vulkan version whose SPIR-V version is at least the
one the compiler wrote: slangc writes SPIR-V 1.5, which Vulkan 1.1 does not accept.

Where the selected runtimes include the harness's Direct3D 12 WARP, HLSL probes run there as HLSL instead, with the
entry point :func:`validators.hlsl_entry` names, compiled by DXC for the harness's shader model
(``tests.conformance.validators.HLSL_PROFILE``). A probe that needs a later shader model fails to compile there and keeps
its validation result, and so does one that queries a ray query. The Windows coverage job runs them so, and
``tools.coverage merge`` adds those verdicts to the rows the Linux job measured (:mod:`.report`).

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

The two probes must write the same words, with two exceptions, both where the native probe's compiler or runtime
computes another value than the function the languages define. CrossGL's ``fmod`` is the exact remainder of C and
MSL, which the HLSL documentation also gives (``x = i * y + f``). SPIR-V's ``OpFRem`` need not compute it: the Vulkan
specification's precision table gives ``OpFRem`` the error inherited from ``x - y * trunc(x / y)``, and lavapipe
computes that expression. DXC lowers HLSL's ``fmod`` to DXIL as ``frac(abs(x / y)) * y``, the fraction taking the
sign of ``x / y``, which inherits the division's error too. So where the native probe of a CrossGL ``fmod`` runs as SPIR-V that uses
``OpFRem``, or as HLSL that calls ``fmod``, a translated word matches when it is the native probe's word or the exact
remainder of the probe's arguments (:func:`remainders`). CrossGL's ``frexp`` returns the mantissa of C's ``frexp``,
which keeps the sign of ``x``, as GLSL, MSL and the HLSL documentation (``x = mantissa * 2^exp``) do. DXC lowers HLSL's
``frexp`` to DXIL as ``(bits & 0x007fffff) | 0x3f000000``, which drops the sign (its SPIR-V keeps it). So where the
native probe runs as HLSL that calls ``frexp``, a translated word also matches when it is C's mantissa of the probe's
argument (:func:`mantissas`). Any other word still differs.

A cell of the translation matrix (:mod:`.pairs`) translates a native probe in one language into another. The two run
on the same runtime only as SPIR-V (:func:`runs_pair`): each through its own language's reference compiler, as DXC
for HLSL and glslang for GLSL, on the Vulkan runtime. A cell with Metal, CUDA or another language at either end keeps
its validation result, because the native probe and the translation would run on different runtimes.
"""

from __future__ import annotations

import re
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import lanes, lowering, validators
from .model import Feature, Result, Type

# The languages the harness runs, with the suffix it reads each one's sources from (tests/conformance/cases.py).
SUFFIXES = {"glsl": ".comp", "hlsl": ".hlsl", "metal": ".metal", "spirv": ".spvasm"}
# The languages that run as the SPIR-V their reference compiler lowers them to, HLSL only where the selected runtimes
# have none for HLSL itself.
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
# The same in HLSL source, which reads no clock DXIL can express.
_UNREPEATABLE_HLSL = re.compile(r"\bRayQuery\s*<")
# The SPIR-V remainder whose precision Vulkan inherits from x - y * trunc(x / y).
_INEXACT_REMAINDER = re.compile(r"\bOpFRem\b")
# The HLSL calls DXC lowers to DXIL as another value than the function's definition: an inexact remainder, and a
# mantissa without the argument's sign.
_DXIL_REMAINDER = re.compile(r"\bfmod\s*\(")
_DXIL_MANTISSA = re.compile(r"\bfrexp\s*\(")
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


def _said(record: dict) -> str:
    """What the harness ``record`` says went wrong: the runtime's message, the harness's own message for a
    ``harness-error``, or the log of the first validation step that did not accept the artifact.
    """
    message = (record.get("execution") or {}).get("message") or record.get("message")
    if message:
        return str(message)
    failed = next(
        (
            step
            for step in record.get("validation") or ()
            if isinstance(step, dict) and step.get("status") != "accepted"
        ),
        None,
    )
    return str(failed.get("log") or "") if failed else ""


def _quoted(record: dict) -> str:
    """The first lines of what the runtime said when it rejected the artifact of the harness ``record``."""
    message = _said(record)
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


def languages(result: Result) -> tuple[str, str]:
    """The languages of ``result``'s native probe and of its translation: the feature's own for a row's result, and
    the cell's source and target for a translation matrix result (:mod:`.pairs`)."""
    return result.feature.language, result.into or result.feature.language


def runs_pair(source: str, target: str) -> bool:
    """Whether the harness can run a native probe in ``source`` and its translation into ``target`` on one runtime.

    Both run on the Vulkan runtime as SPIR-V: a SPIR-V source as it is, and any other as its language's reference
    compiler lowers it. Two different runtimes, such as Metal's and Vulkan's, need not round a function alike, so a
    translation with either end in a language that does not lower to SPIR-V keeps its validation result.
    """
    spirv = VIA_SPIRV | {"glsl", "spirv"}
    return source != target and source in spirv and target in spirv


def as_spirv(language: str, source: str) -> str:
    """The SPIR-V assembly ``language``'s reference compiler writes for a compute ``source``, or an empty string when
    it rejects the source. A SPIR-V ``source`` is its own assembly."""
    if language == "spirv":
        return source
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


def remainders(feature: Feature):
    """The output words a probe of ``feature`` writes when it computes the exact remainder of its arguments, or
    ``None`` when ``feature`` is not CrossGL's ``fmod`` of two float scalars or vectors.

    The probe reads its arguments in order from consecutive elements of the float input, so a ``floatN`` call takes
    ``x`` from ``fin[i]`` to ``fin[i + N - 1]`` and ``y`` from the next N elements, and stores the ``floatN`` result
    in the N words from ``N * i`` (:mod:`.probes`).
    """
    import numpy as np

    signature = feature.signature
    if feature.crossgl != "fmod" or signature is None:
        return None
    kinds = [param.type for param in signature.params]
    directions = {param.direction for param in signature.params}
    result = signature.result
    if (
        len(kinds) != 2
        or directions != {"in"}
        or any(kind != result for kind in kinds)
        or not _elementwise_float(result)
    ):
        return None
    x, y = _arguments(result, 2)
    return np.fmod(x, y).astype(np.float32).reshape(-1).view(np.uint32)


def mantissas(feature: Feature):
    """The output words a probe of ``feature`` writes when it returns C's ``frexp`` mantissa of its argument, which
    keeps the argument's sign, or ``None`` when ``feature`` is not CrossGL's ``frexp`` of a float scalar or vector.

    The probe reads the argument as :func:`remainders` describes, passes a local for the exponent, and stores the
    mantissa.
    """
    import numpy as np

    signature = feature.signature
    if feature.crossgl != "frexp" or signature is None:
        return None
    params = signature.params
    result = signature.result
    if (
        len(params) != 2
        or [param.direction for param in params] != ["in", "out"]
        or params[0].type != result
        or not _elementwise_float(result)
    ):
        return None
    [x] = _arguments(result, 1)
    return np.frexp(x)[0].astype(np.float32).reshape(-1).view(np.uint32)


def _elementwise_float(kind: Type) -> bool:
    return (
        kind.scalar == "float"
        and not kind.opaque
        and not kind.count
        and len(kind.shape) <= 1
    )


def _arguments(kind: Type, count: int):
    """The ``count`` arguments of the type ``kind`` that each invocation of a probe reads, one row per invocation."""
    import numpy as np

    width = kind.shape[0] if kind.shape else 1
    data = inputs("fin")
    elements = np.arange(THREADS)[:, None] + np.arange(width)[None, :]
    return [data[elements + n * width] for n in range(count)]


def exact_words(feature: Feature, runs_as: str, native: str):
    """The words a translation of ``feature``'s native probe ``native`` may also write where the native probe's
    compiler computes another value than the function's definition, or ``None`` where it must write the native
    probe's words. ``runs_as`` is the language the runtime runs: ``spirv`` or ``hlsl`` (DXIL).
    """
    if runs_as == "spirv" and _INEXACT_REMAINDER.search(native):
        return remainders(feature)
    if runs_as == "hlsl" and _DXIL_REMAINDER.search(native):
        return remainders(feature)
    if runs_as == "hlsl" and _DXIL_MANTISSA.search(native):
        return mantissas(feature)
    return None


def difference(native, translated, exact=None) -> str:
    """How the output words ``translated`` differ from ``native``, or an empty string when they match.

    Two words that are both NaN as 32-bit floats match, whatever their payload. ``exact``, when given, holds the words
    the exact result writes at the start of the output (:func:`exact_words`); a translated word there also matches
    when it is the exact one.
    """
    import numpy as np

    floats = (native.view(np.float32), translated.view(np.float32))
    differ = (native != translated) & ~(np.isnan(floats[0]) & np.isnan(floats[1]))
    if exact is not None:
        count = len(exact)
        written = translated[:count]
        nan = np.isnan(written.view(np.float32)) & np.isnan(exact.view(np.float32))
        differ[:count] &= (written != exact) & ~nan
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
    outputs, lowered, rebound, via_spirv, exact)}`` and ``{index: reason}`` for the results it does not run.

    ``lowered`` says whether the runtime compiles the translation at a lower language version than it declares,
    ``rebound`` is :func:`rebinding`'s objection to the translation's resources, and ``exact`` holds the words of an
    exact result where the native probe's compiler computes another (:func:`exact_words`), or ``None``.
    """
    from tests.conformance import runner

    runs_as = "spirv" if via_spirv else language
    suffix, cases, outputs, lowered = SUFFIXES[runs_as], [], {}, set()
    rebound, exact, skipped = {}, {}, {}
    assemblies = []
    for index, result in chunk:
        data = tuple(inputs(buffer) for buffer in input_buffers(result.sources[0]))
        if via_spirv:
            texts = [
                as_spirv(spoken, text)
                for spoken, text in zip(languages(result), result.sources)
            ]
            if not all(texts):
                failed = list(
                    dict.fromkeys(
                        validators.TOOLS[spoken]
                        for spoken, text in zip(languages(result), texts)
                        if not text
                    )
                )
                verb = "writes" if len(failed) == 1 else "write"
                skipped[index] = f"{' and '.join(failed)} {verb} no SPIR-V for it"
                continue
        else:
            texts = [runnable(language, source) for source in result.sources]
            if texts[1] != result.sources[1]:
                lowered.add(index)
        if runs_as == "spirv" and any(_FLOAT16.search(text) for text in texts):
            skipped[index] = "its SPIR-V declares a 16-bit float, which lavapipe does not round through"
            continue
        if runs_as == "spirv" and any(_UNREPEATABLE.search(text) for text in texts):
            skipped[index] = (
                "it reads a clock or a ray query, which no second run repeats"
            )
            continue
        if runs_as == "hlsl" and any(_UNREPEATABLE_HLSL.search(t) for t in texts):
            skipped[index] = "it reads a ray query, which no second run repeats"
            continue
        if via_spirv:
            assemblies += texts
        exact[index] = exact_words(result.feature, runs_as, texts[0])
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
        return {}, skipped
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
            exact.get(index),
        )
        for index, roles in found.items()
    }, skipped | {
        index: "the harness recorded no run of it"
        for index, _ in chunk
        if index not in found and index not in skipped
    }


def _judge(
    result: Result,
    native,
    translated,
    outputs,
    lowered: bool,
    rebound: str,
    via_spirv: bool,
    exact,
    index: int,
) -> Result:
    """``result`` with the verdict of its run, or with the reason the run gave none in ``unrun``."""
    if not native or not translated:
        role = "its translation" if native else "its native probe"
        return replace(result, unrun=f"the harness recorded no run of {role}")
    if native["outcome"] != "preserved":
        return replace(result, unrun=_unran("its native probe", native))
    runtime = native["runtime"]
    if lowered and translated["outcome"] == "driver-rejected":
        return replace(
            result,
            unrun="the runtime compiles the translation only at a lower version than it declares, and rejects it",
        )
    compilers = " and ".join(
        dict.fromkeys(
            validators.TOOLS[spoken]
            for spoken in languages(result)
            if spoken != "spirv"
        )
    )
    where = f"on {runtime}" + (
        f" as SPIR-V from {compilers}" if via_spirv and compilers else ""
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
        return replace(result, unrun=_unran("the translation", translated))
    found = difference(
        outputs[f"{index:05d}-native"], outputs[f"{index:05d}-translated"], exact
    )
    if found:
        first, _, rest = found.partition("\n")
        detail = f"{where}, the translation {first}\n{rest}"
        return replace(result, outcome="wrong", detail=detail, runtime=runtime)
    return replace(result, runtime=runtime)


def _unran(role: str, record: dict) -> str:
    """Why the harness ``record`` of ``role`` gives no verdict, with what the runtime said."""
    message = _quoted(record)
    return f"{role} did not run ({record['outcome']})" + (
        f"\n{message}" if message else ""
    )


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
    """Run ``items`` on the first available runtime; return ``({index: records}, runtime id, {index: reason})``, with
    an empty id when there is nothing to run or no runtime to run it on, and the reasons for the items it did not
    run."""
    from tests.conformance import runner

    lowered = {spoken for _, result in items for spoken in languages(result)}
    if not items or (
        via_spirv
        and not all(lowering.can_lower(spoken) for spoken in lowered - {"spirv"})
    ):
        return {}, "", {}
    runs_as = "spirv" if via_spirv else language
    selection = runner.select_runtimes(runtimes, [runs_as])
    available = [pair for pair in selection[runs_as] if pair[1].available]
    if not available:
        return {}, "", {}
    # One runtime per language keeps a result's verdict independent of how many devices the host has.
    selection = {runs_as: available[:1]}
    chunks = [items[n::workers] for n in range(workers) if items[n::workers]]
    found, skipped = {}, {}
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        done = pool.map(
            lambda pair: _execute(
                language, pair[1], selection, root / str(pair[0]), via_spirv
            ),
            enumerate(chunks),
        )
        for records, reasons in done:
            found.update(records)
            skipped.update(reasons)
    return found, available[0][0].id, skipped


def run(results: list[Result], runtimes: str = "available", workers: int = 4):
    """Execute every :func:`executable` result where a runtime is available; return the results and the runtimes.

    ``runtimes`` is the harness's selection: ``available``, or a comma-separated list of runtime ids. A translation
    matrix result runs with its native probe as SPIR-V when :func:`runs_pair` allows it. A result the selected runtime
    did not run, or ran without a verdict, says why in ``unrun``.
    """
    results = list(results)
    by_language: dict[tuple[str, str], list[tuple[int, Result]]] = {}
    for index, result in enumerate(results):
        native, translated = languages(result)
        if native == translated:
            if not runs(native):
                continue
        elif not runs_pair(native, translated):
            continue
        if executable(result):
            by_language.setdefault((native, translated), []).append((index, result))
    used = set()
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-execution-") as scratch:
        for (language, into), items in by_language.items():
            found, runtime, skipped = {}, "", {}
            folder = language if into == language else f"{language}-to-{into}"
            # A row's language the harness runs as itself runs so where a selected runtime can, and otherwise as
            # SPIR-V; a translation matrix cell runs only as SPIR-V.
            if into == language and language in SUFFIXES:
                found, runtime, skipped = _pass(
                    language, items, runtimes, Path(scratch, folder), workers, False
                )
            if not runtime and (language in VIA_SPIRV or into != language):
                found, runtime, skipped = _pass(
                    language,
                    items,
                    runtimes,
                    Path(scratch, f"{folder}-spirv"),
                    workers,
                    True,
                )
            used.add(runtime)
            where = dict.fromkeys([*found, *skipped], runtime)
            if (language, into) == ("glsl", "glsl"):
                again = [
                    (i, r) for i, r in items if i in found and _unjudged(*found[i])
                ]
                rerun, runtime, unran = _pass(
                    language,
                    again,
                    runtimes,
                    Path(scratch, "glsl-spirv"),
                    workers,
                    True,
                )
                found.update(rerun)
                skipped.update(unran)
                where.update(dict.fromkeys([*rerun, *unran], runtime))
                if rerun:
                    used.add(runtime)
            for index in where:
                judged = results[index]
                if index in found:
                    judged = _judge(judged, *found[index], index)
                # A run that gave no verdict ends with a reason; a later pass's reason for not running it wins.
                reason = "" if judged.runtime else skipped.get(index, judged.unrun)
                if reason:
                    judged = replace(judged, unrun=f"on {where[index]}, {reason}")
                results[index] = judged
    return results, sorted(used - {""})


def _label(result: Result) -> str:
    native, translated = languages(result)
    return native if native == translated else f"{native} to {translated}"


def idle(results: list[Result]) -> list[str]:
    """The languages, or translation matrix cells as ``"<source> to <target>"``, with :func:`executable` results of
    which none ran.

    A runtime that rejects every native probe of a language, as one does for SPIR-V newer than it is told to accept,
    runs none of them, just as a missing runtime does.
    """
    ran = {_label(r) for r in results if r.runtime}
    return sorted({_label(r) for r in results if executable(r)} - ran)
