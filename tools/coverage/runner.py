"""Run probes and classify each one as supported, refused, wrong or untested."""

from __future__ import annotations

import hashlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import execution, probes, spirv_input, validators
from .model import ROUND_TRIPS, Feature, Result
from .semantic import lower

OUTPUT_BACKENDS = {
    "cuda": "cuda",
    "glsl": "opengl",
    "hip": "hip",
    "hlsl": "directx",
    "metal": "metal",
    "mojo": "mojo",
    "rust": "rust",
    "slang": "slang",
    "spirv": "vulkan",
    "webgl": "webgl",
    "wgsl": "wgsl",
}
# The CrossGL dialect (a key of :data:`.probes.DIALECTS`) each language's output probes are written in, when it is not
# plain CrossGL: GLSL ES 3.00 runs no compute kernels.
CROSSGL_DIALECTS = {"webgl": "crossgl-es"}
INPUT_BACKENDS = {
    "cuda": "cuda",
    "glsl": "opengl",
    "hip": "hip",
    "hlsl": "directx",
    "metal": "metal",
    "mojo": "mojo",
    "opencl": "opencl",
    "rust": "rust",
    "slang": "slang",
    "spirv": "vulkan",
}


class Refused(Exception):
    pass


class Broken(Exception):
    pass


def _first_line(error: BaseException, scratch: str) -> str:
    text = str(error)
    if scratch:
        text = text.replace(scratch + os.sep, "")
    text = text.strip().splitlines()
    return f"{type(error).__name__}: {text[0][:300] if text else ''}".rstrip(": ")


def translate(source: str, suffix: str, backend: str, source_backend: str) -> str:
    """Translate ``source`` with CrossTL, raising :class:`Refused` or :class:`Broken` on failure.

    A failure is a refusal only when it carries a stable diagnostic code, the rule the conformance harness uses
    (``tests/conformance/worker.py``). Any other exception is a translator bug.
    """
    from crosstl import translate as crosstl_translate

    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-") as scratch:
        path = Path(scratch) / ("probe" + suffix)
        path.write_text(source)
        try:
            return crosstl_translate(
                str(path),
                backend=backend,
                source_backend=source_backend,
                format_output=False,
            )
        except Exception as error:  # noqa: BLE001 - every failure is classified
            code = getattr(error, "project_diagnostic_code", None)
            if isinstance(code, str) and code:
                raise Refused(f"{code}: {_first_line(error, scratch)}") from error
            raise Broken(_first_line(error, scratch)) from error


# A feature's output and input probes both validate its native probe, and a round trip often reproduces a source
# already checked, so a run validates each distinct source once. The directory lives for one run and is shared by its
# worker processes; a verdict is never reused across runs, where the tool might differ.
_verdicts: str | None = None


def _remember(directory: str | None) -> None:
    global _verdicts
    _verdicts = directory


def _verdict(
    language: str, source: str, stage: str, entry: str, target: str
) -> validators.Verdict:
    if not _verdicts:
        return validators.validate(language, source, stage, entry, target)
    digest = hashlib.sha256(
        "\0".join((language, target, stage, entry, source)).encode()
    ).hexdigest()
    path = Path(_verdicts, digest)
    try:
        accepted, _, log = path.read_text().partition("\n")
        return validators.Verdict(accepted == "1", log)
    except FileNotFoundError:
        pass
    verdict = validators.validate(language, source, stage, entry, target)
    # Written aside and renamed, so a worker validating the same source at the same time never reads half a file.
    partial = path.with_suffix(f".{os.getpid()}")
    partial.write_text(("1" if verdict.accepted else "0") + "\n" + verdict.log)
    partial.replace(path)
    return verdict


def _validated(
    language: str, source: str, stage: str, entry: str = "", target: str = ""
) -> validators.Verdict:
    verdict = _verdict(language, source, stage, entry, target)
    if verdict.accepted:
        return verdict
    return validators.Verdict(False, validators.first_error(verdict.log, language))


def parse_crossgl(source: str) -> None:
    """Parse a CrossGL probe, raising :class:`Refused` or :class:`Broken` like :func:`translate`."""
    from crosstl.translator.source_registry import (
        SOURCE_REGISTRY,
        register_default_sources,
    )

    register_default_sources()
    try:
        SOURCE_REGISTRY.get("cgl").parse(source)
    except Exception as error:  # noqa: BLE001 - every failure is classified
        code = getattr(error, "project_diagnostic_code", None)
        detail = f"CrossGL does not parse the probe: {_first_line(error, '')}"
        if isinstance(code, str) and code:
            raise Refused(f"{code}: {detail}") from error
        raise Broken(detail) from error


def _native_rejection(feature: Feature, language: str) -> str:
    """Why the target rejects the feature's own probe, or an empty string when it accepts it or cannot tell."""
    if language not in probes.DIALECTS or not validators.available(
        language, feature.target
    ):
        return ""
    try:
        native = probes.build(feature, language)
    except probes.ProbeError:
        return ""
    verdict = _validated(
        language, native.source, feature.stage, native.entry, feature.target
    )
    return "" if verdict.accepted else verdict.log


def _runs(feature: Feature, language: str) -> bool:
    """Whether :mod:`.execution` can run ``feature``'s probes in ``language``."""
    return feature.stage == "compute" and language in execution.SUFFIXES


def run_output(feature: Feature, language: str) -> Result:
    semantic, _ = lower(feature)

    def result(outcome: str, detail: str = "") -> Result:
        tool = validators.TOOLS[language] if outcome in ("supported", "wrong") else ""
        return Result(feature, "output", outcome, detail, tool, semantic)

    try:
        probe = probes.build(feature, CROSSGL_DIALECTS.get(language, "crossgl"))
    except probes.NoSpelling as error:
        return result("refused", str(error))
    except probes.ProbeError as error:
        return result("untested", str(error))
    rejected = _native_rejection(feature, language)
    if rejected:
        return result("untested", f"the probe itself is rejected: {rejected}")
    try:
        parse_crossgl(probe.source)
        translated = translate(
            probe.source, probe.suffix, OUTPUT_BACKENDS[language], "cgl"
        )
    except Refused as error:
        return result("refused", str(error))
    except Broken as error:
        return result("wrong", str(error))
    if not validators.available(language, feature.target):
        tool = (
            validators.missing(language, feature.target) or validators.TOOLS[language]
        )
        return result("untested", f"{tool} is not installed")
    verdict = _validated(language, translated, feature.stage, "", feature.target)
    if not verdict.accepted:
        return result("wrong", verdict.log)
    # The CrossGL probe reads its inputs like the native probe unless it calls a counterpart overload.
    if _runs(feature, language) and not feature.counterpart:
        try:
            native = (
                spirv_input.build(feature)
                if language == "spirv"
                else probes.build(feature, language)
            ).source
        except probes.ProbeError:
            return result("supported")
        return replace(result("supported"), sources=(native, translated))
    return result("supported")


def run_input(feature: Feature, language: str) -> Result:
    def result(outcome: str, detail: str = "") -> Result:
        tool = validators.TOOLS[language] if outcome in ("supported", "wrong") else ""
        return Result(feature, "input", outcome, detail, tool)

    try:
        if language == "spirv":
            probe = spirv_input.build(feature)
        else:
            probe = probes.build(feature, language)
    except probes.ProbeError as error:
        return result("untested", str(error))
    if not validators.available(language, feature.target):
        tool = (
            validators.missing(language, feature.target) or validators.TOOLS[language]
        )
        return result("untested", f"{tool} is not installed")
    native = _validated(
        language, probe.source, feature.stage, probe.entry, feature.target
    )
    if not native.accepted:
        return result("untested", f"the probe itself is rejected: {native.log}")
    backend = INPUT_BACKENDS[language]
    emitted = ROUND_TRIPS.get(language, language)
    try:
        # A language CrossTL cannot generate goes through CrossTL's own pipeline to the emitted language, which
        # imports the probe to CrossGL and, for OpenCL C, lowers its kernel parameters for the target in one call.
        if emitted == language:
            crossgl = translate(probe.source, probe.suffix, "cgl", backend)
            translated = translate(crossgl, ".cgl", OUTPUT_BACKENDS[language], "cgl")
        else:
            translated = translate(
                probe.source, probe.suffix, OUTPUT_BACKENDS[emitted], backend
            )
    except Refused as error:
        return result("refused", str(error))
    except Broken as error:
        return result("wrong", str(error))
    verdict = _validated(emitted, translated, feature.stage, "", feature.target)
    if not verdict.accepted:
        return result("wrong", f"after the round trip: {verdict.log}")
    lost = spirv_input.dropped(feature, translated) if language == "spirv" else ""
    if lost:
        return result("wrong", lost)
    if emitted == language and _runs(feature, language):
        return replace(result("supported"), sources=(probe.source, translated))
    return result("supported")


def _job(job: tuple[Feature, str, str]) -> Result:
    feature, direction, language = job
    return (
        run_output(feature, language)
        if direction == "output"
        else run_input(feature, language)
    )


def run(
    jobs: list[tuple[Feature, str, str]], workers: int | None = None
) -> list[Result]:
    """Run every ``(feature, direction, language)`` job; results come back in job order."""
    workers = workers or min(8, os.cpu_count() or 1)
    with tempfile.TemporaryDirectory(prefix="crosstl-coverage-verdicts-") as verdicts:
        if workers == 1 or len(jobs) < 2:
            _remember(verdicts)
            try:
                return [_job(job) for job in jobs]
            finally:
                _remember(None)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_remember, initargs=(verdicts,)
        ) as pool:
            return list(pool.map(_job, jobs, chunksize=4))
