"""Metal helper reuse never caches shader execution or injected test runners."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project import metal_runtime as module
from crosstl.project.runtime_verification import RuntimeAdapterSetupError


@pytest.fixture
def isolated_cache():
    module._clear_worker_cache()
    yield
    module._clear_worker_cache()


@pytest.fixture
def toolchain(tmp_path, monkeypatch, isolated_cache):
    source = tmp_path / "metal_runtime_worker.swift"
    source.write_text("import Metal\n", encoding="utf-8")
    compiler = tmp_path / "swiftc"
    compiler.write_bytes(b"compiler")
    sdk = tmp_path / "MacOSX.sdk"
    sdk.mkdir()
    settings = sdk / "SDKSettings.json"
    settings.write_text('{"Version": "1"}', encoding="utf-8")
    xcrun = tmp_path / "xcrun"
    xcrun.write_bytes(b"resolver")
    state = SimpleNamespace(
        source=source,
        compiler=compiler,
        sdk=sdk,
        settings=settings,
        xcrun=xcrun,
        version="Swift version 1\nTarget: arm64-apple-macosx\n",
        calls=[],
        builds=[],
        failure=None,
        entered=None,
        release=None,
    )

    def run(command, *, input_text=None, timeout_seconds=120):
        state.calls.append((list(command), input_text, timeout_seconds))
        if "--find" in command:
            output = str(state.compiler)
        elif "--show-sdk-path" in command:
            output = str(state.sdk)
        elif "--version" in command:
            output = state.version
        elif "-o" in command:
            path = Path(command[command.index("-o") + 1])
            state.builds.append(path)
            if state.entered is not None:
                state.entered.set()
                assert state.release.wait(5)
            path.write_bytes(b"verified helper")
            path.chmod(0o700)
            if isinstance(state.failure, BaseException):
                raise state.failure
            if state.failure == "missing":
                path.unlink()
            elif state.failure == "empty":
                path.write_bytes(b"")
            elif state.failure == "exit":
                return subprocess.CompletedProcess(command, 1, "", "compile failure")
            output = ""
        else:
            output = '{"available": true, "device": "test device"}'
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module, "_WORKER_SOURCE", source)
    monkeypatch.setattr(module, "run_metal_command", run)
    monkeypatch.setattr(module.shutil, "which", lambda name: str(state.xcrun))
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")
    state.run = run
    return state


def _runtime(**options):
    return module.MetalComputeRuntime(platform_name="darwin", **options)


def test_default_instances_share_helper_but_probe_separately(toolchain):
    first, second = _runtime(), _runtime(timeout_seconds=37)
    assert first.is_available(None, None).available
    path = first._worker_path
    first.close()
    assert path.is_file()
    assert second.is_available(None, None).available
    assert second._worker_path == path
    second.close()
    assert first.is_available(None, None).available
    assert first._worker_path == path
    assert len(toolchain.builds) == 1
    probes = [call for call in toolchain.calls if "--probe" in call[0]]
    assert len(probes) == 3
    assert [call[2] for call in probes] == [120, 37, 120]
    module._clear_worker_cache()
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "change",
    ("source", "compiler", "version", "sdk", "architecture", "flags", "environment"),
)
def test_cache_identity_includes_build_inputs(toolchain, monkeypatch, change):
    runtime = _runtime()
    first = runtime._worker()
    if change == "source":
        toolchain.source.write_text("import Metal\n// changed\n", encoding="utf-8")
    elif change == "compiler":
        toolchain.compiler.write_bytes(b"replacement compiler")
    elif change == "version":
        toolchain.version = "Swift version 2"
    elif change == "sdk":
        toolchain.settings.write_text('{"Version": "2"}', encoding="utf-8")
    elif change == "architecture":
        monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    elif change == "flags":
        monkeypatch.setattr(
            module, "_WORKER_OPTIONS", ("-Onone", "-warnings-as-errors")
        )
    else:
        monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "15.0")
    assert runtime._worker() != first
    assert len(toolchain.builds) == 2


def test_unrelated_environment_changes_do_not_rebuild(toolchain, monkeypatch):
    first = _runtime()._worker()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "another test")
    assert _runtime()._worker() == first
    assert len(toolchain.builds) == 1


@pytest.mark.parametrize("change", ("missing", "corrupt"))
def test_missing_or_corrupt_executable_is_rebuilt(toolchain, change):
    runtime = _runtime()
    first = runtime._worker()
    if change == "missing":
        first.unlink()
    else:
        first.write_bytes(b"corrupted helper")
    replacement = runtime._worker()
    assert replacement != first
    assert replacement.read_bytes() == b"verified helper"
    assert not first.parent.exists()
    assert len(toolchain.builds) == 2


@pytest.mark.parametrize(
    "failure", ("missing", "empty", "exit", "timeout", "interrupt")
)
def test_failed_builds_are_cleaned_and_retryable(toolchain, failure):
    error = RuntimeAdapterSetupError
    if failure == "timeout":
        toolchain.failure = RuntimeAdapterSetupError("deadline")
    elif failure == "interrupt":
        toolchain.failure, error = KeyboardInterrupt(), KeyboardInterrupt
    else:
        toolchain.failure = failure
    with pytest.raises(error):
        _runtime()._worker()
    assert not module._WORKERS
    assert not toolchain.builds[0].parent.exists()
    toolchain.failure = None
    assert _runtime()._worker().is_file()
    assert len(toolchain.builds) == 2


def test_concurrent_instances_publish_one_complete_build(toolchain):
    toolchain.entered, toolchain.release = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_runtime()._worker) for _ in range(4)]
        assert toolchain.entered.wait(5)
        assert not module._WORKERS
        toolchain.release.set()
        paths = [future.result(timeout=5) for future in futures]
    assert len(set(paths)) == 1
    assert len(toolchain.builds) == 1


def test_waiting_for_another_build_has_a_deadline(toolchain):
    with module._BUILD_LOCK:
        with pytest.raises(RuntimeAdapterSetupError) as caught:
            _runtime(timeout_seconds=0.01)._worker()
    assert caught.value.details["reasonKind"] == "worker-build-timeout"
    assert not toolchain.builds
    assert _runtime()._worker().is_file()


@pytest.mark.parametrize("injection", ("runner", "resolver", "both"))
def test_injected_runners_and_resolvers_keep_instance_owned_builds(
    toolchain, injection
):
    shared = _runtime()._worker()
    options = {}
    if injection in {"runner", "both"}:
        options["command_runner"] = toolchain.run
    if injection in {"resolver", "both"}:
        options["tool_resolver"] = lambda name: str(toolchain.xcrun)
    first, second = _runtime(**options), _runtime(**options)
    paths = [first._worker(), second._worker()]
    assert len({shared, *paths}) == 3
    assert first._worker() == paths[0]
    first.close()
    assert not paths[0].exists()
    assert paths[1].is_file() and shared.is_file()
    second.close()
    assert not paths[1].exists()


def test_compile_uses_keyed_source_snapshot_and_explicit_sdk(toolchain):
    path = _runtime()._worker()
    command = next(call[0] for call in toolchain.calls if "-o" in call[0])
    assert command[0] == str(toolchain.compiler)
    assert command[command.index("-sdk") + 1] == str(toolchain.sdk)
    assert "-O" in command and "-warnings-as-errors" in command
    snapshot = path.parent / "metal_runtime_worker.swift"
    assert str(snapshot) in command
    assert snapshot.read_bytes() == toolchain.source.read_bytes()
    assert command[command.index("-module-cache-path") + 1] == str(
        path.parent / "modules"
    )


def test_unavailable_toolchain_never_reuses_a_stale_helper(toolchain, monkeypatch):
    _runtime()._worker()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    result = _runtime().is_available(None, None)
    assert not result.available
    assert result.details["reasonKind"] == "tool-unavailable"


def test_native_helper_reuse_is_required_in_existing_package_job():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate native Metal package execution"
    )
    assert 'CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME: "1"' in step
    assert "tests/test_translator/test_metal_worker_cache.py" in step
    assert "-n auto" in step and "continue-on-error" not in step


def test_native_instances_reuse_helper_and_execute_independent_requests(
    tmp_path, monkeypatch, isolated_cache
):
    if os.environ.get("CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME") != "1":
        pytest.skip("requires the macOS native package runtime gate")
    assert sys.platform == "darwin"
    from tests.test_translator.test_metal_native_runtime import (
        _constant_request,
        _request,
    )
    from tests.test_translator.test_native_loader_dispatch_integration import _executor

    calls, records = [], []
    run = module.run_metal_command

    def observe(command, **kwargs):
        calls.append((list(command), kwargs.get("input_text")))
        return run(command, **kwargs)

    monkeypatch.setattr(module, "run_metal_command", observe)
    for index in range(3):
        root = tmp_path / str(index)
        root.mkdir()
        if index == 2:
            request = _constant_request(root, enabled=False)
            expected = {"result": {"values": [-1] * 4}}
        else:
            request, _, _, _, expected = _request(root, constant_pointer=index == 1)
        executor = _executor("metal")
        try:
            availability = executor.is_available(request)
            assert availability.available, availability
            result = executor.run(request)
            assert result.status == "ok", result
            assert result.outputs["result"]["values"] == expected["result"]["values"]
            path = executor.runtime_adapter.runtime._worker_path
            records.append(
                {
                    "worker": str(path),
                    "workerSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "outputs": result.outputs,
                    "details": result.details,
                }
            )
        finally:
            executor.runtime_adapter.runtime.close()
    (tmp_path / "helper-reuse.json").write_text(
        json.dumps({"instances": records, "commands": calls}, indent=2),
        encoding="utf-8",
    )
    assert len({record["worker"] for record in records}) == 1
    assert (
        len(
            [
                call
                for call in calls
                if Path(call[0][0]).name == "swiftc" and "-o" in call[0]
            ]
        )
        == 1
    )
    assert len([call for call in calls if "metal" in call[0]]) == 3
    assert len([call for call in calls if "metallib" in call[0]]) == 3
    assert len([call for call in calls if "--probe" in call[0]]) >= 3
    assert len([call for call in calls if call[1] is not None]) == 3


@pytest.mark.parametrize("failure", ("empty", "nonzero"))
def test_toolchain_lookup_failure_does_not_publish_a_helper(
    toolchain, monkeypatch, failure
):
    monkeypatch.setattr(
        module,
        "run_metal_command",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1 if failure == "nonzero" else 0,
            "failed" if failure == "nonzero" else "",
            "diagnostic",
        ),
    )
    result = _runtime().is_available(None, None)
    assert not result.available
    assert result.details["reasonKind"] == "toolchain-unavailable"
    assert result.details["stderr"] == "diagnostic"
    assert not module._WORKERS and not toolchain.builds


def test_forked_cache_does_not_own_parent_helpers(toolchain):
    path = _runtime()._worker()
    entries = dict(module._WORKERS)
    lock = module._BUILD_LOCK
    try:
        with lock:
            module._reset_worker_cache_after_fork()
            assert module._BUILD_LOCK is not lock
            module._clear_worker_cache()
        assert path.is_file()
        child = _runtime()._worker()
        assert child != path
        module._clear_worker_cache()
        assert not child.exists() and path.is_file()
    finally:
        module._WORKERS.update(entries)
