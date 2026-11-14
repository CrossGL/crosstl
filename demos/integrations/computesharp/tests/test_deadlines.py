"""Check the deadlines of the translated-run harness.

A hung dispatch must fail the run instead of holding the job until its
timeout: each ``dotnet test`` run has a deadline after which its whole process
group is stopped, and MSTest stops the test host when one case exceeds the
per-case deadline.
"""

import os
import sys
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

RUN_TESTS = load_module("computesharp_run_tests_deadlines", ROOT / "run_tests.py")

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the translated runs use POSIX process groups"
)


def test_commands_within_the_deadline_report_their_status(tmp_path):
    log = tmp_path / "logs" / "ok.log"
    status = RUN_TESTS.run_logged(
        [sys.executable, "-c", "print('done'); raise SystemExit(3)"], log, deadline=30
    )

    assert status == 3
    text = log.read_text()
    assert "done" in text
    assert "[exit 3 after " in text


def test_a_command_past_its_deadline_is_stopped_with_its_children(tmp_path):
    marker = tmp_path / "child-still-running"
    child = (
        "import pathlib, time; time.sleep(3); "
        f"pathlib.Path({str(marker)!r}).write_text('late')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    )
    log = tmp_path / "hung.log"
    started = time.monotonic()

    with pytest.raises(RUN_TESTS.DeadlineExceeded, match="deadline of 1s"):
        RUN_TESTS.run_logged([sys.executable, "-c", parent], log, deadline=1)

    assert time.monotonic() - started < 30
    assert "[deadline of 1s exceeded; the command was stopped]" in log.read_text()
    time.sleep(4)
    assert not marker.exists()


def test_test_runs_stop_hung_cases_and_have_a_run_deadline(tmp_path, monkeypatch):
    calls = []

    def fake_run_logged(command, log, *, deadline=None, **kwargs):
        calls.append((command, deadline))
        results = Path(command[command.index("--results-directory") + 1])
        (results / "results.trx").write_text("<TestRun/>")
        return 0

    monkeypatch.setattr(RUN_TESTS, "run_logged", fake_run_logged)
    common = dict(
        adapted=tmp_path / "adapted",
        artifacts=tmp_path / "artifacts",
        target="metal",
        native_library=tmp_path / "libcsbackend_metal.dylib",
        excluded=["ComputeSharp.Tests.InteropTests.Method"],
        evidence=None,
        fault=None,
        extra_filter=None,
        log=tmp_path / "test-run.log",
    )
    RUN_TESTS.run_tests(results_dir=tmp_path / "trx", **common)
    RUN_TESTS.run_tests(
        results_dir=tmp_path / "control",
        deadline=RUN_TESTS.CONTROL_RUN_DEADLINE_SECONDS,
        **common,
    )

    (full, full_deadline), (control, control_deadline) = calls
    hang = full.index("--blame-hang-timeout")
    assert full[hang + 1] == RUN_TESTS.TEST_CASE_DEADLINE == "5m"
    assert full[full.index("--blame-hang-dump-type") + 1] == "none"
    assert full_deadline == RUN_TESTS.RUN_DEADLINE_SECONDS
    assert control_deadline == RUN_TESTS.CONTROL_RUN_DEADLINE_SECONDS
    assert "--blame-hang-timeout" in control
    assert RUN_TESTS.DEADLINES == {
        "testCase": "5m",
        "runSeconds": 1500,
        "controlRunSeconds": 240,
    }
