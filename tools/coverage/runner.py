"""Run probes and classify each one as supported, refused, wrong or untested."""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import probes, validators
from .model import Feature, Result
from .semantic import lower

OUTPUT_BACKENDS = {
    "cuda": "cuda",
    "glsl": "opengl",
    "hip": "hip",
    "hlsl": "directx",
    "metal": "metal",
    "spirv": "vulkan",
    "wgsl": "wgsl",
}
INPUT_BACKENDS = {
    "cuda": "cuda",
    "glsl": "opengl",
    "hip": "hip",
    "hlsl": "directx",
    "metal": "metal",
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


def _validated(
    language: str, source: str, stage: str, entry: str = ""
) -> validators.Verdict:
    verdict = validators.validate(language, source, stage, entry)
    if verdict.accepted:
        return verdict
    errors = [line for line in verdict.log.splitlines() if "error" in line.lower()]
    return validators.Verdict(
        False, (errors[0] if errors else verdict.log[:300]).strip()
    )


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
    if language not in probes.DIALECTS or not validators.available(language):
        return ""
    try:
        native = probes.build(feature, language)
    except probes.ProbeError:
        return ""
    verdict = _validated(language, native.source, feature.stage, native.entry)
    return "" if verdict.accepted else verdict.log


def run_output(feature: Feature, language: str) -> Result:
    semantic, _ = lower(feature)

    def result(outcome: str, detail: str = "") -> Result:
        tool = validators.TOOLS[language] if outcome in ("supported", "wrong") else ""
        return Result(feature, "output", outcome, detail, tool, semantic)

    try:
        probe = probes.build(feature, "crossgl")
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
    if not validators.available(language):
        return result("untested", f"{validators.TOOLS[language]} is not installed")
    verdict = _validated(language, translated, feature.stage)
    return result("supported") if verdict.accepted else result("wrong", verdict.log)


def run_input(feature: Feature, language: str) -> Result:
    def result(outcome: str, detail: str = "") -> Result:
        tool = validators.TOOLS[language] if outcome in ("supported", "wrong") else ""
        return Result(feature, "input", outcome, detail, tool)

    try:
        probe = probes.build(feature, language)
    except probes.ProbeError as error:
        return result("untested", str(error))
    if not validators.available(language):
        return result("untested", f"{validators.TOOLS[language]} is not installed")
    native = _validated(language, probe.source, feature.stage, probe.entry)
    if not native.accepted:
        return result("untested", f"the probe itself is rejected: {native.log}")
    backend = INPUT_BACKENDS[language]
    try:
        crossgl = translate(probe.source, probe.suffix, "cgl", backend)
        translated = translate(crossgl, ".cgl", OUTPUT_BACKENDS[language], "cgl")
    except Refused as error:
        return result("refused", str(error))
    except Broken as error:
        return result("wrong", str(error))
    verdict = _validated(language, translated, feature.stage)
    return (
        result("supported")
        if verdict.accepted
        else result("wrong", f"after the round trip: {verdict.log}")
    )


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
    if workers == 1 or len(jobs) < 2:
        return [_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_job, jobs, chunksize=4))
