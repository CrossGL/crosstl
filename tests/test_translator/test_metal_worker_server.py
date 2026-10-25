"""Serve-mode Metal workers: the line protocol, pooling and native reuse."""

import json
import os
import sys
import threading

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project import metal_runtime as module
from crosstl.project.metal_worker_server import MetalWorkerServer
from crosstl.project.runtime_verification import (
    RuntimeAdapterSetupError,
    RuntimeExecutionError,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="serve-mode workers use POSIX process groups"
)

# A stand-in worker that speaks the serve protocol; each request names its
# action. Replies carry the worker's process id so reuse is observable.
FAKE_WORKER = """\
import json
import os
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    action = request["action"]
    if action == "echo":
        reply = {"pid": os.getpid(), "size": len(request.get("data", ""))}
        sys.stdout.write("+" + json.dumps(reply) + "\\n")
    elif action == "fail":
        failure = {"failure": "Invalid group count.", "error": {"field": "x"}}
        sys.stdout.write("-" + json.dumps(failure) + "\\n")
    elif action == "message":
        sys.stdout.write("-" + json.dumps({"failure": "Missing entry."}) + "\\n")
    elif action == "exit":
        sys.stderr.write("worker stopped\\n")
        sys.stderr.flush()
        sys.exit(3)
    elif action == "slow":
        time.sleep(0.5)
        sys.stdout.write("+" + json.dumps({"pid": os.getpid()}) + "\\n")
    elif action == "hang":
        time.sleep(60)
    else:
        sys.stdout.write("?" + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture
def worker(tmp_path):
    path = tmp_path / "worker"
    path.write_text(f"#!{sys.executable}\n{FAKE_WORKER}", encoding="utf-8")
    path.chmod(0o700)
    return path


def _request(action, **fields):
    return json.dumps({"action": action, **fields})


def _pid(result):
    assert result.returncode == 0, result
    return json.loads(result.stdout)["pid"]


def test_server_runs_requests_in_one_process(worker):
    server = MetalWorkerServer(worker)
    try:
        # Several megabytes cross the pipe in large writes, not 512-byte chunks.
        data = "x" * (6 << 20)
        first = server.exchange(_request("echo", data=data), 30)
        second = server.exchange(_request("echo"), 30)
        assert json.loads(first.stdout)["size"] == len(data)
        assert _pid(first) == _pid(second) == server.process.pid
        assert first.args == [str(worker), "--serve"]
        assert server.usable
    finally:
        server.terminate()
    assert not server.usable
    assert server.process.returncode is not None


def test_failure_replies_match_one_shot_results(worker):
    server = MetalWorkerServer(worker)
    try:
        detailed = server.exchange(_request("fail"), 30)
        assert detailed.returncode == 1
        assert json.loads(detailed.stdout) == {"error": {"field": "x"}}
        assert detailed.stderr == "Invalid group count.\n"
        plain = server.exchange(_request("message"), 30)
        assert (plain.returncode, plain.stdout, plain.stderr) == (
            1,
            "",
            "Missing entry.\n",
        )
        # A failed request leaves the worker serving.
        assert _pid(server.exchange(_request("echo"), 30)) == server.process.pid
    finally:
        server.terminate()


def test_worker_exit_reports_status_and_stderr(worker):
    server = MetalWorkerServer(worker)
    result = server.exchange(_request("exit"), 30)
    assert result.returncode == 3
    assert result.stderr == "worker stopped\n"
    assert not server.usable


def test_deadline_kills_the_worker(worker):
    server = MetalWorkerServer(worker)
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        server.exchange(_request("hang"), 0.5)
    details = caught.value.details
    assert details["reasonKind"] == "operation-timeout"
    assert details["timeoutSeconds"] == 0.5
    assert details["command"] == [str(worker), "--serve"]
    assert server.process.returncode is not None
    assert not server.usable


def test_malformed_reply_is_rejected(worker):
    server = MetalWorkerServer(worker)
    with pytest.raises(RuntimeAdapterSetupError) as caught:
        server.exchange(_request("garbage"), 30)
    assert caught.value.details["reasonKind"] == "worker-protocol-invalid"
    assert not server.usable


def test_inherited_server_is_not_used_or_stopped(worker, monkeypatch):
    server = MetalWorkerServer(worker)
    try:
        monkeypatch.setattr(server, "owner", server.owner + 1)
        assert not server.usable
        server.terminate()
        # The owning process still has a running worker.
        assert server.process.poll() is None
    finally:
        monkeypatch.undo()
        server._closed = False
        server.terminate()


def test_runtime_reuses_idle_workers_and_replaces_stopped_ones(worker):
    runtime = module.MetalComputeRuntime(platform_name="darwin", timeout_seconds=30)
    try:
        first = _pid(runtime._serve(worker, _request("echo")))
        assert _pid(runtime._serve(worker, _request("echo"))) == first
        assert runtime._serve(worker, _request("exit")).returncode == 3
        assert not runtime._servers
        assert _pid(runtime._serve(worker, _request("echo"))) != first
    finally:
        runtime.close()
    assert not runtime._servers


def test_concurrent_dispatches_use_separate_workers(worker):
    runtime = module.MetalComputeRuntime(platform_name="darwin", timeout_seconds=30)
    barrier = threading.Barrier(2)
    pids = []

    def dispatch():
        barrier.wait()
        pids.append(_pid(runtime._serve(worker, _request("slow"))))

    try:
        threads = [threading.Thread(target=dispatch) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # Neither dispatch waited for the other's worker, and both workers
        # return to the pool for later dispatches.
        assert len(set(pids)) == 2
        assert len(runtime._servers) == 2
        assert _pid(runtime._serve(worker, _request("echo"))) in pids
    finally:
        runtime.close()


def test_runtime_discards_a_worker_past_its_deadline(worker):
    runtime = module.MetalComputeRuntime(platform_name="darwin", timeout_seconds=0.5)
    try:
        with pytest.raises(RuntimeAdapterSetupError):
            runtime._serve(worker, _request("hang"))
        assert not runtime._servers
    finally:
        runtime.close()


def test_injected_runner_keeps_one_command_per_dispatch():
    runtime = module.MetalComputeRuntime(
        platform_name="darwin", command_runner=lambda *args, **kwargs: None
    )
    assert not runtime._persistent
    assert module.MetalComputeRuntime(platform_name="darwin")._persistent


def test_native_worker_serves_independent_requests(tmp_path, monkeypatch):
    if os.environ.get("CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME") != "1":
        pytest.skip("requires the macOS native package runtime gate")
    assert sys.platform == "darwin"
    from tests.test_translator.test_metal_native_runtime import _request
    from tests.test_translator.test_native_loader_dispatch_integration import (
        _executor,
    )

    for name in ("valid", "other"):
        (tmp_path / name).mkdir()
    request, descriptor, package, inputs, outputs = _request(tmp_path / "valid")
    oversized = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [2, 2, 1], "workgroupSize": [65536, 1, 1]},
        expected_target="metal",
    )
    # A second library under another path keeps its own cached pipeline.
    other, *_ = _request(tmp_path / "other")
    executor = _executor("metal")
    runtime = executor.runtime_adapter.runtime
    pids = []
    original = MetalWorkerServer.exchange

    def observe(server, request_text, timeout_seconds):
        pids.append(server.process.pid)
        return original(server, request_text, timeout_seconds)

    monkeypatch.setattr(MetalWorkerServer, "exchange", observe)
    try:
        for item in (request, oversized, other, request):
            if item is oversized:
                with pytest.raises(RuntimeExecutionError) as caught:
                    executor.run(item)
                details = caught.value.details
                assert details["dispatchValidation"]["field"] == "workgroupSize"
                assert "group dimension" in details["stderr"]
                continue
            result = executor.run(item)
            assert result.status == "ok", result
            assert result.outputs["result"]["values"] == (outputs["result"]["values"])
    finally:
        runtime.close()
    assert len(pids) == 4 and len(set(pids)) == 1
    (tmp_path / "serve.json").write_text(
        json.dumps({"pids": pids}, indent=2), encoding="utf-8"
    )
