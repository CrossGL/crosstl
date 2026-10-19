"""Optional toolchain probes do not replace required native validation."""

import subprocess
from pathlib import Path

import pytest

import conftest as toolchains


@pytest.fixture
def metal_probe(monkeypatch):
    probe = toolchains._xcrun_resolves_missing_metal_toolchain
    probe.cache_clear()
    monkeypatch.setattr(toolchains.shutil, "which", lambda name: f"/tools/{name}")
    yield probe
    probe.cache_clear()


@pytest.mark.parametrize("stage", ("lookup", "compile"))
def test_timed_out_metal_probe_keeps_toolchain_discovery_unchanged(
    monkeypatch, metal_probe, stage
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {
            "capture_output": True,
            "check": False,
            "text": True,
            "timeout": 30,
        }
        if stage == "lookup" or len(calls) == 2:
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, "/tools/metal", "")

    monkeypatch.setattr(toolchains.subprocess, "run", run)
    assert metal_probe() is False
    assert metal_probe() is False
    toolchains.hide_xcrun_when_metal_toolchain_is_incomplete.__wrapped__(monkeypatch)
    assert toolchains.shutil.which("xcrun") == "/tools/xcrun"
    assert toolchains.shutil.which("clang") == "/tools/clang"
    assert len(calls) == (1 if stage == "lookup" else 2)
    if stage == "compile":
        source_path = Path(calls[1][calls[1].index("-c") + 1])
        assert not source_path.parent.exists()


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_confirmed_missing_metal_component_is_still_hidden(
    monkeypatch, metal_probe, stream
):
    message = (
        "missing Metal Toolchain; run xcodebuild -downloadComponent MetalToolchain"
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, "/tools/metal", "")
        return subprocess.CompletedProcess(
            command,
            1,
            message if stream == "stdout" else "",
            message if stream == "stderr" else "",
        )

    monkeypatch.setattr(toolchains.subprocess, "run", run)
    assert metal_probe() is True
    toolchains.hide_xcrun_when_metal_toolchain_is_incomplete.__wrapped__(monkeypatch)
    assert toolchains.shutil.which("xcrun") is None
    assert toolchains.shutil.which("clang") == "/tools/clang"
    assert len(calls) == 2


@pytest.mark.parametrize("lookup_status,compile_status", ((1, 0), (0, 0), (0, 1)))
def test_other_probe_results_do_not_hide_the_toolchain(
    monkeypatch, metal_probe, lookup_status, compile_status
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        status = lookup_status if len(calls) == 1 else compile_status
        return subprocess.CompletedProcess(command, status, "", "unrelated diagnostic")

    monkeypatch.setattr(toolchains.subprocess, "run", run)
    assert metal_probe() is False
    toolchains.hide_xcrun_when_metal_toolchain_is_incomplete.__wrapped__(monkeypatch)
    assert toolchains.shutil.which("xcrun") == "/tools/xcrun"
    assert len(calls) == (1 if lookup_status else 2)


def test_absent_xcrun_does_not_start_a_probe(monkeypatch, metal_probe):
    monkeypatch.setattr(toolchains.shutil, "which", lambda name: None)

    def run(*args, **kwargs):
        pytest.fail("A missing executable must not be invoked")

    monkeypatch.setattr(toolchains.subprocess, "run", run)
    assert metal_probe() is False
