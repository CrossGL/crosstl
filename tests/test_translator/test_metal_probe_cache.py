"""Metal device probes: persistent runtimes keep the probe that found a device."""

import json
import subprocess

import pytest

from crosstl.project import metal_runtime as module


class Probes:
    """A stand-in for the helper's ``--probe`` command that counts its runs."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.commands = []

    def __call__(self, command, *, input_text=None, timeout_seconds=120):
        _ = input_text, timeout_seconds
        self.commands.append(list(command))
        available = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        payload = {"available": available, "device": "Test GPU" if available else None}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def _runtime(monkeypatch, probes, *, injected=False, workers=("/helper/a",)):
    if not injected:
        monkeypatch.setattr(module, "run_metal_command", probes)
    runtime = module.MetalComputeRuntime(
        platform_name="darwin", command_runner=probes if injected else None
    )
    paths = list(workers)
    monkeypatch.setattr(
        runtime, "_worker", lambda: paths.pop(0) if len(paths) > 1 else paths[0]
    )
    return runtime


def test_a_persistent_runtime_probes_its_helper_once(monkeypatch):
    probes = Probes(True)
    runtime = _runtime(monkeypatch, probes)
    assert runtime._persistent
    first = runtime.is_available(None, None)
    second = runtime.is_available(None, None)
    assert first.available and second.available
    assert first.details == second.details
    assert first.details["device"] == "Test GPU"
    assert probes.commands == [["/helper/a", "--probe"]]
    # Each answer owns its details.
    assert first.details is not second.details


def test_a_probe_without_a_device_is_repeated(monkeypatch):
    probes = Probes(False, True)
    runtime = _runtime(monkeypatch, probes)
    missing = runtime.is_available(None, None)
    assert not missing.available
    assert missing.details["reasonKind"] == "device-unavailable"
    assert runtime.is_available(None, None).available
    assert runtime.is_available(None, None).available
    assert len(probes.commands) == 2


def test_a_new_helper_is_probed_again(monkeypatch):
    probes = Probes(True)
    runtime = _runtime(monkeypatch, probes, workers=("/helper/a", "/helper/b"))
    runtime.is_available(None, None)
    runtime.is_available(None, None)
    runtime.is_available(None, None)
    assert probes.commands == [["/helper/a", "--probe"], ["/helper/b", "--probe"]]


def test_closing_forgets_the_probe(monkeypatch):
    probes = Probes(True)
    runtime = _runtime(monkeypatch, probes)
    runtime.is_available(None, None)
    runtime.close()
    runtime.is_available(None, None)
    assert len(probes.commands) == 2


def test_an_injected_runner_sees_every_probe(monkeypatch):
    probes = Probes(True)
    runtime = _runtime(monkeypatch, probes, injected=True)
    assert not runtime._persistent
    runtime.is_available(None, None)
    runtime.is_available(None, None)
    assert len(probes.commands) == 2


@pytest.mark.parametrize("stdout", ["[]", "not json"])
def test_an_invalid_probe_reply_is_not_kept(monkeypatch, stdout):
    calls = []

    def runner(command, *, input_text=None, timeout_seconds=120):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout, "")

    runtime = _runtime(monkeypatch, runner)
    for _ in range(2):
        assert not runtime.is_available(None, None).available
    assert len(calls) == 2
