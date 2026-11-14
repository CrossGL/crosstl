"""Session-wide reporting of generated-artifact pin drift for the MLX tests.

In report mode (see :mod:`demos.integrations.mlx.tests.pins`) a drifted pin does
not fail its test. This plugin gives the session one drift directory that every
pytest-xdist worker inherits, then lists each drifted pin once at the end of the
run: in the terminal summary as ``MLX-PIN-DRIFT`` JSON lines, which are enough to
refresh the pins from a CI log, and as a count in the GitHub step summary.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from demos.integrations.mlx.tests import pins

_OWNED_DRIFT_DIR = "_crosstl_mlx_owned_pin_drift_dir"


def _is_worker(config) -> bool:
    return hasattr(config, "workerinput")


def pytest_configure(config):
    if _is_worker(config) or pins.pin_mode() != pins.REPORT_PIN_MODE:
        return
    if os.environ.get(pins.PIN_DRIFT_DIR_ENV, "").strip():
        return
    # Workers start after configuration, so they inherit this directory.
    directory = tempfile.mkdtemp(prefix="crosstl-mlx-pin-drift-")
    os.environ[pins.PIN_DRIFT_DIR_ENV] = directory
    setattr(config, _OWNED_DRIFT_DIR, directory)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _is_worker(config):
        return
    directory = os.environ.get(pins.PIN_DRIFT_DIR_ENV, "").strip()
    if not directory or not Path(directory).is_dir():
        return
    records = pins.drift_records(Path(directory))
    if not records:
        return
    terminalreporter.section("MLX generated-artifact pin drift")
    terminalreporter.write_line(
        f"{len(records)} pinned values drifted; translation, validation and "
        "numerical checks still ran."
    )
    for record in records:
        terminalreporter.write_line(
            "MLX-PIN-DRIFT " + json.dumps(record, sort_keys=True)
        )
    summary = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(pins.drift_step_summary(records))


def pytest_unconfigure(config):
    directory = getattr(config, _OWNED_DRIFT_DIR, None)
    if directory is None:
        return
    shutil.rmtree(directory, ignore_errors=True)
    if os.environ.get(pins.PIN_DRIFT_DIR_ENV) == directory:
        del os.environ[pins.PIN_DRIFT_DIR_ENV]
