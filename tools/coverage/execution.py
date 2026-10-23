"""Run supported compute probes on a GPU runtime and compare what they compute with the native probe.

A translation the reference tool accepts can still compute something else. Where Core's conformance harness
(``tests/conformance``) has a runtime for a language, the native probe and CrossTL's translation run there on the
same inputs, and a translation that writes other results, writes outside its output buffer, or that the runtime
rejects is wrong.

The harness binds one input buffer at slot 0 and one output buffer at slot 1, so only compute probes that read one
input buffer run. The other probes keep their validation result. Integer inputs are small and non-negative, so no
probe shifts, extracts or inserts bits outside what every language defines.
"""

from __future__ import annotations

import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .model import Result

# The languages the harness runs, with the suffix it reads each one's sources from (tests/conformance/cases.py).
SUFFIXES = {"metal": ".metal", "spirv": ".spvasm"}
THREADS = 64
# Output words one thread may write: a 4x4 float matrix is the largest result a probe stores.
WORDS = 16
# Input elements one thread may read past its own: three 4x4 matrix arguments.
SPAN = 48
_INPUT = re.compile(r"\b(fin|iin|uin)\b")
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
    """Whether ``result`` is a supported compute probe whose native probe and translation read one input buffer."""
    if result.outcome != "supported" or len(result.sources) != 2:
        return False
    buffers = [set(_INPUT.findall(source)) for source in result.sources]
    return len(buffers[0]) == 1 and buffers[0] == buffers[1]


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
    """Run one chunk of results; return ``{index: (native record, translated record, outputs)}``."""
    from tests.conformance import runner

    suffix, cases, outputs = SUFFIXES[language], [], {}
    for index, result in chunk:
        [buffer] = set(_INPUT.findall(result.sources[0]))
        data = inputs(buffer)
        for role, source in zip(("native", "translated"), result.sources):
            path = root / "sources" / f"{index:05d}-{role}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
            cases.append(_case(f"{index:05d}-{role}", path, language, data, outputs))
    records = runner.run_suite(
        cases, [language], ["strict"], selection, root, reference=True
    )
    found = {}
    for record in records:
        index, role = record["case"].split("-")
        found.setdefault(int(index), {})[role] = record
    return {
        index: (roles.get("native"), roles.get("translated"), outputs)
        for index, roles in found.items()
    }


def _judge(result: Result, native, translated, outputs, index: int) -> Result:
    if not native or not translated or native["outcome"] != "preserved":
        return result
    runtime = native["runtime"]
    if translated["outcome"] in _BROKEN:
        detail = f"on {runtime}, the translation {_BROKEN[translated['outcome']]}"
        return replace(result, outcome="wrong", detail=detail, runtime=runtime)
    if translated["outcome"] != "preserved":
        return result
    found = difference(
        outputs[f"{index:05d}-native"], outputs[f"{index:05d}-translated"]
    )
    if found:
        first, _, rest = found.partition("\n")
        detail = f"on {runtime}, the translation {first}\n{rest}"
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
        if language in SUFFIXES and executable(result):
            by_language.setdefault(language, []).append((index, result))
    used = set()
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-execution-") as scratch:
        for language, items in by_language.items():
            selection = runner.select_runtimes(runtimes, [language])
            available = [pair for pair in selection[language] if pair[1].available]
            if not available:
                continue
            # One runtime per language keeps a result's verdict independent of how many devices the host has.
            selection = {language: available[:1]}
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
                    for index, (native, translated, outputs) in found.items():
                        results[index] = _judge(
                            results[index], native, translated, outputs, index
                        )
            used.add(available[0][0].id)
    return results, sorted(used)
