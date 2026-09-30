import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_bounded_command.py"


def _run(*args, timeout=10):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def test_bounded_command_streams_child_output_and_reports_success():
    result = _run(
        "--timeout-seconds",
        "5",
        "--label",
        "live-success",
        "--",
        sys.executable,
        "-c",
        (
            "import sys; "
            "print('child-stdout', flush=True); "
            "print('child-stderr', file=sys.stderr, flush=True)"
        ),
    )

    assert result.returncode == 0
    assert "[bounded-command] START label='live-success' timeout=5s" in result.stdout
    assert "child-stdout" in result.stdout
    assert "child-stderr" in result.stderr
    assert "status=passed exit=0" in result.stdout


def test_bounded_command_propagates_child_failure():
    result = _run(
        "--timeout-seconds",
        "5",
        "--label",
        "expected-failure",
        "--",
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    )

    assert result.returncode == 7
    assert "status=failed exit=7" in result.stdout
    assert "TIMEOUT" not in result.stderr


def test_bounded_command_times_out_with_case_identity():
    started_at = time.monotonic()
    result = _run(
        "--timeout-seconds",
        "0.2",
        "--label",
        "hung-runtime-entry",
        "--",
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    )

    assert result.returncode == 124
    assert time.monotonic() - started_at < 5
    assert "TIMEOUT label='hung-runtime-entry'" in result.stderr
    assert "limit=0.2s" in result.stderr
    assert "status=timeout exit=124" in result.stdout


def test_bounded_command_timeout_kills_descendants(tmp_path):
    marker = tmp_path / "descendant-survived"
    descendant = (
        "import pathlib,time; "
        "time.sleep(1); "
        f"pathlib.Path({str(marker)!r}).write_text('survived', encoding='utf-8')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        "time.sleep(30)"
    )

    result = _run(
        "--timeout-seconds",
        "0.2",
        "--label",
        "process-tree",
        "--",
        sys.executable,
        "-c",
        parent,
    )
    time.sleep(1.2)

    assert result.returncode == 124
    assert not marker.exists()
