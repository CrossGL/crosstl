"""Generated-artifact pins for the MLX corpus tests.

A pin records the exact bytes one translator state produced for an MLX kernel when
the artifact was reviewed. Other work keeps changing the translator on ``main``, so a
merge-queue group or a nightly run builds on a translator the pins were not reviewed
against. Those runs set ``CROSSTL_MLX_PIN_MODE=report``: a changed pin is reported as
drift instead of failing, while translation, diagnostics, native compilation and
numerical parity stay enforced. Every other run, including pull requests and
dispatched runs, enforces the pins.

Drift is reported through :class:`PinDriftWarning` and, when
``CROSSTL_MLX_PIN_DRIFT_DIR`` names a directory, as one JSON line per drifted pin in
``drift-<pid>.jsonl`` there, which is enough to refresh the pins.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PIN_MODE_ENV = "CROSSTL_MLX_PIN_MODE"
PIN_DRIFT_DIR_ENV = "CROSSTL_MLX_PIN_DRIFT_DIR"
STRICT_PIN_MODE = "strict"
REPORT_PIN_MODE = "report"
PIN_MODES = (STRICT_PIN_MODE, REPORT_PIN_MODE)
DRIFT_SUMMARY_LIMIT = 20


class PinDriftWarning(UserWarning):
    """A generated artifact no longer matches the bytes it was pinned to."""


def pin_mode() -> str:
    """Return the configured pin mode; an unknown mode is a configuration error."""

    mode = os.environ.get(PIN_MODE_ENV, "").strip() or STRICT_PIN_MODE
    if mode not in PIN_MODES:
        raise ValueError(
            f"{PIN_MODE_ENV} must be one of {', '.join(PIN_MODES)}, not {mode!r}"
        )
    return mode


def _record_drift(record: Mapping[str, Any]) -> None:
    directory = os.environ.get(PIN_DRIFT_DIR_ENV, "").strip()
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with (path / f"drift-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)


def check_pin(subject: str, field: str, actual: Any, expected: Any) -> bool:
    """Compare one pinned value; return whether it still matches.

    Strict mode raises ``AssertionError`` on a mismatch. Report mode records the
    drift and returns ``False`` so the caller continues with its other checks.
    """

    if actual == expected:
        return True
    message = (
        f"{subject}: pinned {field} changed from {expected!r} to {actual!r}; "
        "review the new artifact and update the pin"
    )
    if pin_mode() == STRICT_PIN_MODE:
        raise AssertionError(message)
    _record_drift(
        {
            "subject": subject,
            "field": field,
            "expected": expected,
            "actual": actual,
            "test": os.environ.get("PYTEST_CURRENT_TEST", ""),
        }
    )
    warnings.warn(PinDriftWarning(message), stacklevel=3)
    return False


def check_generated_artifact(
    artifact: Mapping[str, Any],
    *,
    sha256: str,
    size_bytes: int | None = None,
) -> bool:
    """Check a portability-report artifact against its pinned bytes.

    The artifact must always carry a SHA-256 ``generatedHash`` and an integer
    ``generatedSizeBytes``; only the pinned values themselves may drift.
    """

    subject = str(artifact.get("path") or artifact.get("source") or "artifact")
    generated_hash = artifact.get("generatedHash")
    assert isinstance(generated_hash, Mapping), f"{subject}: generatedHash is missing"
    assert set(generated_hash) == {"algorithm", "value"}, generated_hash
    assert generated_hash["algorithm"] == "sha256", generated_hash
    value = generated_hash["value"]
    assert isinstance(value, str) and len(value) == 64, generated_hash
    matches = check_pin(subject, "generatedHash", value, sha256)
    if size_bytes is not None:
        actual_size = artifact.get("generatedSizeBytes")
        assert isinstance(actual_size, int) and actual_size > 0, actual_size
        matches = (
            check_pin(subject, "generatedSizeBytes", actual_size, size_bytes)
            and matches
        )
    return matches


def drift_records(directory: Path) -> list[dict]:
    """Return every drift record under ``directory`` once, in a stable order."""

    records = {}
    for path in sorted(directory.glob("drift-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[json.dumps(record, sort_keys=True)] = record
    return sorted(
        records.values(),
        key=lambda record: tuple(
            str(record.get(name, ""))
            for name in ("subject", "field", "test", "expected", "actual")
        ),
    )


def drift_step_summary(records: list[dict]) -> str:
    """Render drift records as a short Markdown section for a CI step summary."""

    lines = [
        f"### MLX generated-artifact pin drift: {len(records)} pins",
        "",
        "Report mode recorded these pins instead of failing; refresh them from the "
        "`MLX-PIN-DRIFT` lines in the job log.",
        "",
    ]
    for record in records[:DRIFT_SUMMARY_LIMIT]:
        lines.append(
            f"- `{str(record.get('test', '')).split(' ')[0]}` {record['field']}: "
            f"`{record['expected']}` -> `{record['actual']}`"
        )
    if len(records) > DRIFT_SUMMARY_LIMIT:
        lines.append(f"- ... and {len(records) - DRIFT_SUMMARY_LIMIT} more")
    return "\n".join(lines) + "\n"
