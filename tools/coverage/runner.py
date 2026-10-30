"""Run probes and classify each one as supported, refused, wrong or untested."""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import decorations, execution, probes, rustgpu, spirv_input, validators
from .features.spirv import is_core
from .model import ROUND_TRIPS, Feature, Result, kept_folder
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
# The target a language's probes fall back to when the reference compiler rejects a native probe for its usual target.
# slangc writes invalid SPIR-V for some operations its HLSL target compiles, such as an OpAtomicCompareExchange on
# floats for InterlockedCompareExchangeFloatBitwise, so those are probed for HLSL instead of being left untested.
FALLBACK_TARGETS = {"slang": "hlsl"}


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
# already checked, so a run validates each distinct source once. The verdicts live in a directory shared by the run's
# worker processes: a temporary one, or the one ``CROSSTL_COVERAGE_VERDICTS`` names, which CI carries from one run to
# the next and which holds only the verdicts the latest run used. A verdict is filed under the versions of the tools
# that gave it and the code that ran them, so a kept verdict is reused only for the same check.
VERDICTS = "CROSSTL_COVERAGE_VERDICTS"
_verdicts: str | None = None
# Where ``run --keep`` writes each probe and CrossTL's translations of it; set in each worker process like
# ``_verdicts``.
_kept: str | None = None


def _remember(verdicts: str | None, kept: str | None = None) -> None:
    global _verdicts, _kept
    _verdicts, _kept = verdicts, kept


def _keep(
    feature: Feature, direction: str, language: str, name: str, text: str
) -> None:
    if _kept:
        folder = Path(_kept, kept_folder(language, direction, feature.name))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text)


@functools.lru_cache(maxsize=None)
def _checker(language: str, target: str) -> str:
    """The versions of the tools that check ``language`` for ``target``, and a digest of the code that runs them."""
    versions = [
        validators.version(tool) for tool in validators.toolchain(language, target)
    ]
    code = [Path(module.__file__).read_bytes() for module in (validators, rustgpu)]
    return "\n".join(versions + [hashlib.sha256(b"\0".join(code)).hexdigest()])


def _verdict(
    language: str, source: str, stage: str, entry: str, target: str
) -> validators.Verdict:
    if not _verdicts:
        return validators.validate(language, source, stage, entry, target)
    digest = hashlib.sha256(
        "\0".join(
            (language, target, stage, entry, _checker(language, target), source)
        ).encode()
    ).hexdigest()
    path = Path(_verdicts, digest)
    try:
        accepted, _, log = path.read_text().partition("\n")
    except FileNotFoundError:
        pass
    else:
        # A verdict this run used survives _forget_unused.
        os.utime(path)
        return validators.Verdict(accepted == "1", log)
    verdict = validators.validate(language, source, stage, entry, target)
    if verdict.timed_out:
        return verdict
    # Written aside and renamed, so a worker validating the same source at the same time never reads half a file.
    partial = path.with_suffix(f".{os.getpid()}")
    partial.write_text(("1" if verdict.accepted else "0") + "\n" + verdict.log)
    partial.replace(path)
    return verdict


def _forget_unused(verdicts: Path, started: float) -> None:
    """Delete the kept verdicts this run did not use, so the directory holds only verdicts for current sources."""
    for path in verdicts.iterdir():
        # One second of slack covers file systems that store whole-second times.
        if path.stat().st_mtime < started - 1:
            path.unlink()


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


def _native_verdict(
    feature: Feature, language: str, source: str, entry: str
) -> tuple[Feature, validators.Verdict]:
    """The reference compiler's verdict on ``feature``'s native probe ``source``, and the feature as it is probed.

    A probe the compiler rejects for the feature's usual target is retried for the language's
    :data:`FALLBACK_TARGETS` entry; when that target accepts it, the feature is probed for that target from then on.
    Otherwise the verdict for the usual target stands.
    """
    verdict = _validated(language, source, feature.stage, entry, feature.target)
    fallback = FALLBACK_TARGETS.get(language, "")
    if (
        verdict.accepted
        or feature.target
        or not fallback
        or not validators.available(language, fallback)
    ):
        return feature, verdict
    retried = _validated(language, source, feature.stage, entry, fallback)
    if retried.accepted:
        return replace(feature, target=fallback), retried
    return feature, verdict


def _native_rejection(feature: Feature, language: str) -> tuple[Feature, str]:
    """The feature as it is probed, and why the target rejects its own probe: an empty string when the target accepts
    the probe or the check cannot run."""
    if language not in probes.DIALECTS or not validators.available(
        language, feature.target
    ):
        return feature, ""
    try:
        native = probes.build(feature, language)
    except probes.ProbeError:
        return feature, ""
    _keep(feature, "output", language, "native" + native.suffix, native.source)
    feature, verdict = _native_verdict(feature, language, native.source, native.entry)
    return feature, "" if verdict.accepted else verdict.log


def _runs(feature: Feature, language: str) -> bool:
    """Whether :mod:`.execution` can run ``feature``'s probes in ``language``."""
    return feature.stage == "compute" and execution.runs(language)


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
    _keep(feature, "output", language, "probe" + probe.suffix, probe.source)
    probed, rejected = _native_rejection(feature, language)
    if rejected:
        return result("untested", f"the probe itself is rejected: {rejected}")
    # The semantic id stays the one :func:`.semantic.lower` found for the usual target.
    feature = probed
    try:
        parse_crossgl(probe.source)
        translated = translate(
            probe.source, probe.suffix, OUTPUT_BACKENDS[language], "cgl"
        )
    except Refused as error:
        return result("refused", str(error))
    except Broken as error:
        return result("wrong", str(error))
    _keep(
        feature,
        "output",
        language,
        "translated" + validators.suffix(language, feature.stage),
        translated,
    )
    if not validators.available(language, feature.target):
        tool = (
            validators.missing(language, feature.target) or validators.TOOLS[language]
        )
        return result("untested", f"{tool} is not installed")
    verdict = _validated(language, translated, feature.stage, "", feature.target)
    if not verdict.accepted:
        return result("wrong", verdict.log)
    lost = decorations.lost(feature, language, translated)
    if lost:
        return result("wrong", lost)
    ignored = (
        spirv_input.ignores_nan(feature, translated) if language == "spirv" else ""
    )
    if ignored:
        return result("wrong", ignored)
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


def _encodes_otherwise(feature: Feature) -> bool:
    """Whether CrossTL's SPIR-V generator writes a core instruction's operation without the instruction, as it
    writes ``isnan`` as a test of the exponent bits: its translation of the feature's own CrossGL probe lacks the
    instruction too. A round trip that drops it is then judged by its validation and execution, as the output row is.
    """
    if not is_core(feature):
        return False
    try:
        probe = probes.build(feature, "crossgl")
        direct = translate(probe.source, probe.suffix, OUTPUT_BACKENDS["spirv"], "cgl")
    except (probes.ProbeError, Refused, Broken):
        return False
    return bool(spirv_input.dropped(feature, direct))


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
    _keep(feature, "input", language, "probe" + probe.suffix, probe.source)
    if not validators.available(language, feature.target):
        tool = (
            validators.missing(language, feature.target) or validators.TOOLS[language]
        )
        return result("untested", f"{tool} is not installed")
    feature, native = _native_verdict(feature, language, probe.source, probe.entry)
    if not native.accepted:
        return result("untested", f"the probe itself is rejected: {native.log}")
    backend = INPUT_BACKENDS[language]
    emitted = ROUND_TRIPS.get(language, language)
    try:
        # A language CrossTL cannot generate goes through CrossTL's own pipeline to the emitted language, which
        # imports the probe to CrossGL and, for OpenCL C, lowers its kernel parameters for the target in one call.
        if emitted == language:
            crossgl = translate(probe.source, probe.suffix, "cgl", backend)
            _keep(feature, "input", language, "crossgl.cgl", crossgl)
            translated = translate(crossgl, ".cgl", OUTPUT_BACKENDS[language], "cgl")
        else:
            translated = translate(
                probe.source, probe.suffix, OUTPUT_BACKENDS[emitted], backend
            )
    except Refused as error:
        return result("refused", str(error))
    except Broken as error:
        return result("wrong", str(error))
    _keep(
        feature,
        "input",
        language,
        "translated" + validators.suffix(emitted, feature.stage),
        translated,
    )
    verdict = _validated(emitted, translated, feature.stage, "", feature.target)
    if not verdict.accepted:
        return result("wrong", f"after the round trip: {verdict.log}")
    lost = spirv_input.dropped(feature, translated) if language == "spirv" else ""
    if lost and not _encodes_otherwise(feature):
        return result("wrong", lost)
    lost = decorations.lost(feature, emitted, translated)
    if lost:
        return result("wrong", f"after the round trip: {lost}")
    if emitted == language and _runs(feature, language):
        return replace(result("supported"), sources=(probe.source, translated))
    return result("supported")


def _job(job: tuple[Feature, str, str]) -> Result:
    feature, direction, language = job
    if _kept:
        # Files an earlier run kept for this probe would sit beside this run's as if they belonged to it.
        shutil.rmtree(
            Path(_kept, kept_folder(language, direction, feature.name)),
            ignore_errors=True,
        )
    return (
        run_output(feature, language)
        if direction == "output"
        else run_input(feature, language)
    )


def run(
    jobs: list[tuple[Feature, str, str]],
    workers: int | None = None,
    keep: Path | None = None,
) -> list[Result]:
    """Run every ``(feature, direction, language)`` job; results come back in job order.

    With ``keep``, each job writes its probe and CrossTL's translations of it to the job's :func:`.model.kept_folder`
    under ``keep``.
    """
    workers = workers or min(8, os.cpu_count() or 1)
    kept = str(keep) if keep else None
    lasting = os.environ.get(VERDICTS)
    if not lasting:
        with tempfile.TemporaryDirectory(
            prefix="crosstl-coverage-verdicts-"
        ) as verdicts:
            return _run_jobs(jobs, workers, verdicts, kept)
    Path(lasting).mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = _run_jobs(jobs, workers, lasting, kept)
    _forget_unused(Path(lasting), started)
    return results


def _run_jobs(
    jobs: list[tuple[Feature, str, str]], workers: int, verdicts: str, kept: str | None
) -> list[Result]:
    if workers == 1 or len(jobs) < 2:
        _remember(verdicts, kept)
        try:
            return [_job(job) for job in jobs]
        finally:
            _remember(None)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_remember, initargs=(verdicts, kept)
    ) as pool:
        return list(pool.map(_job, jobs, chunksize=4))
