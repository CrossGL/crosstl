"""Metal downloads must be executable before native CI tests start."""

import subprocess

import pytest

from tools import ensure_metal_toolchain as setup


def _commands(monkeypatch, statuses, *, download_status=0, probe_seconds=0):
    calls = []
    sleeps = []
    clock = [0.0]
    statuses = iter(statuses)

    def run(command, *, check, text, capture_output, timeout):
        assert check is False and text is True
        calls.append((command, timeout))
        if command[0] == "xcodebuild":
            assert command == ["xcodebuild", "-downloadComponent", "MetalToolchain"]
            assert not capture_output
            return subprocess.CompletedProcess(command, download_status)
        assert command == ["xcrun", "--sdk", "macosx", "metal", "--version"]
        assert capture_output and 0 < timeout <= setup.PROBE_TIMEOUT_SECONDS
        clock[0] += min(probe_seconds, timeout)
        status, message = next(statuses)
        return subprocess.CompletedProcess(
            command, status, "Metal compiler\n" if not status else "", message
        )

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(setup.time, "sleep", sleep)
    return calls, sleeps, clock


MISSING = (17, "error: cannot execute 'metal' due to missing Metal Toolchain\n")
READY = (0, "")


@pytest.mark.parametrize("pending_probes", (0, 1, 4, 9))
def test_download_waits_for_executable_without_repeated_downloads(
    monkeypatch, capsys, pending_probes
):
    calls, sleeps, _ = _commands(
        monkeypatch, [MISSING] * (pending_probes + 1) + [READY]
    )
    assert setup.main([]) == 0
    assert sum(command[0] == "xcodebuild" for command, _ in calls) == 1
    assert len(calls) == pending_probes + 3
    assert sleeps == [setup.RETRY_DELAY_SECONDS] * pending_probes
    output = capsys.readouterr()
    assert "Metal compiler" in output.out
    assert "still unavailable" not in output.err


def test_active_compiler_does_not_download_or_wait(monkeypatch):
    calls, sleeps, _ = _commands(monkeypatch, [READY])
    assert setup.main([]) == 0
    assert len(calls) == 1 and not sleeps


@pytest.mark.parametrize("after_download", (False, True))
def test_other_compiler_failures_are_not_retried(monkeypatch, after_download):
    failure = (23, "unsupported platform or compiler configuration\n")
    calls, sleeps, _ = _commands(
        monkeypatch, [MISSING, failure] if after_download else [failure]
    )
    assert setup.main([]) == 23
    assert len(calls) == (3 if after_download else 1)
    assert not sleeps


def test_failed_download_keeps_original_exit_status(monkeypatch):
    calls, sleeps, _ = _commands(monkeypatch, [MISSING], download_status=19)
    assert setup.main([]) == 19
    assert len(calls) == 2 and not sleeps
    assert calls[-1][1] == setup.DOWNLOAD_TIMEOUT_SECONDS


def test_unavailable_compiler_exhausts_bounded_attempts(monkeypatch, capsys):
    calls, sleeps, _ = _commands(
        monkeypatch, [MISSING] * (setup.ACTIVATION_ATTEMPTS + 1)
    )
    assert setup.main([]) == 17
    assert len(calls) == setup.ACTIVATION_ATTEMPTS + 2
    assert sleeps == [setup.RETRY_DELAY_SECONDS] * (setup.ACTIVATION_ATTEMPTS - 1)
    assert "still unavailable" in capsys.readouterr().err


def test_activation_deadline_includes_probe_execution(monkeypatch):
    calls, sleeps, clock = _commands(
        monkeypatch, [MISSING] * 20, probe_seconds=setup.PROBE_TIMEOUT_SECONDS
    )
    assert setup.main([]) == 17
    assert len(calls) < setup.ACTIVATION_ATTEMPTS + 2
    assert clock[0] == setup.PROBE_TIMEOUT_SECONDS + setup.ACTIVATION_TIMEOUT_SECONDS
    assert all(0 < delay <= setup.RETRY_DELAY_SECONDS for delay in sleeps)


@pytest.mark.parametrize("stage", ("probe", "download"))
@pytest.mark.parametrize("failure,expected", (("timeout", 124), ("missing", 127)))
def test_unusable_commands_fail_without_retry(monkeypatch, stage, failure, expected):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if stage == "download" and command[0] == "xcrun":
            return subprocess.CompletedProcess(command, MISSING[0], "", MISSING[1])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main([]) == expected
    assert len(calls) == (2 if stage == "download" else 1)


@pytest.mark.parametrize("arguments,exit_code", ((["--help"], 0), (["--unknown"], 2)))
def test_cli_argument_handling_does_not_probe_compilers(
    monkeypatch, arguments, exit_code
):
    def unexpected(*args, **kwargs):
        pytest.fail("CLI argument handling must not invoke a compiler or download")

    monkeypatch.setattr(setup, "_probe", unexpected)
    with pytest.raises(SystemExit) as error:
        setup.main(arguments)
    assert error.value.code == exit_code
