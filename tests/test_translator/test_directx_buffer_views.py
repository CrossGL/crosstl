"""Native descriptors retain ranges of shared Direct3D allocations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project import directx_runtime
from crosstl.project.native_runtime_drivers import DirectXComputeRuntime
from crosstl.project.runtime_verification import (
    RuntimeAdapterSetupError,
    RuntimeAllocationView,
    RuntimeExecutorUnavailable,
)
from tests.test_translator.test_native_runtime_drivers import (
    _directx_dispatch_request,
    _FakeCompushady,
)


def _request(tmp_path, constant_offset=256):
    request = _directx_dispatch_request(tmp_path)
    buffers = {
        name: replace(
            binding,
            allocation=RuntimeAllocationView(
                allocation_id=name,
                byte_offset=offset,
                byte_length=size,
                allocation_byte_length=allocation_size,
            ),
        )
        for name, binding, offset, size, allocation_size in (
            (
                "params",
                request.buffers["params"],
                constant_offset,
                4,
                constant_offset + 256,
            ),
            ("lhs", request.buffers["lhs"], 4, 8, 32),
            ("out", request.buffers["out"], 12, 8, 32),
        )
    }
    return replace(
        request,
        buffers=buffers,
        dispatch=replace(request.dispatch, workgroup_count=(2, 1, 1)),
    )


def _plan(requests):
    captured = []
    module = _FakeCompushady()
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
        buffer_view_executor=lambda *args: captured.append(args),
    )
    runtime.dispatch_sequence(None, None, requests)
    assert not module.buffers and not module.computes
    return captured[0][:3]


@pytest.fixture(scope="module")
def protocol_worker(tmp_path_factory):
    if sys.platform == "win32":
        try:
            return directx_runtime._worker()
        except RuntimeAdapterSetupError as exc:
            if (
                exc.details.get("reasonKind") == "compiler-unavailable"
                and os.environ.get("CROSTL_REQUIRE_DIRECTX_BUFFER_VIEWS") != "1"
            ):
                pytest.skip("MSVC and the Windows SDK are unavailable")
            raise
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    root = tmp_path_factory.mktemp("directx-protocol")
    worker = root / "worker"
    source = Path(directx_runtime.__file__).with_name("directx_runtime_worker.cpp")
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(worker),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return worker


@pytest.mark.parametrize("constant_offset", [0, 256, 512])
def test_buffer_view_packet_preserves_offsets_and_uploads(
    tmp_path, protocol_worker, constant_offset
):
    nodes, allocations, keys = _plan((_request(tmp_path, constant_offset),))
    payload, descriptions = directx_runtime.encode_dispatches(nodes, allocations, keys)
    packet = tmp_path / "request.bin"
    packet.write_bytes(payload)
    result = subprocess.run(
        [str(protocol_worker), str(packet), "--validate"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "3 allocations, 1 dispatches\n"
    by_name = {item.allocation_id: item for item in allocations}
    assert by_name["params"].upload_payload[
        constant_offset : constant_offset + 4
    ] == struct.pack("<I", 3)
    assert by_name["lhs"].upload_payload[4:12] == struct.pack("<2f", 1, 2)
    assert by_name["lhs"].upload_payload[:4] == b"\0" * 4
    assert by_name["lhs"].upload_payload[12:] == b"\0" * 20
    assert {item["allocationId"] for item in descriptions} == {"params", "lhs", "out"}


@pytest.mark.parametrize(
    "corruption", ["version", "truncated", "trailing", "count", "shader"]
)
def test_native_view_protocol_rejects_malformed_packets(
    tmp_path, protocol_worker, corruption
):
    payload, _ = directx_runtime.encode_dispatches(*_plan((_request(tmp_path),)))
    if corruption == "version":
        payload = b"BAD!" + payload[4:]
    elif corruption == "truncated":
        payload = payload[:-1]
    elif corruption == "trailing":
        payload += b"!"
    elif corruption == "count":
        payload = payload[:4] + b"\xff" * 4 + payload[8:]
    else:
        payload = payload.replace(b"DXBC", b"BAD!")
    packet = tmp_path / "invalid.bin"
    packet.write_bytes(payload)
    result = subprocess.run(
        [str(protocol_worker), str(packet), "--validate"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr.strip()


def _response(allocations):
    return struct.pack("<II", 0x31525844, len(allocations)) + b"".join(
        struct.pack("<QQ", (index + 1) * 65536, allocation.views[0].allocation_size)
        + bytes(allocation.views[0].allocation_size)
        for index, allocation in enumerate(allocations)
    )


@pytest.mark.parametrize(
    "corruption",
    ["alignment", "extent", "stride", "length", "register", "mixed-access", "overlap"],
)
def test_native_view_protocol_rejects_invalid_descriptors(
    tmp_path, protocol_worker, corruption
):
    nodes, allocations, keys = _plan((_request(tmp_path),))
    payload, _ = directx_runtime.encode_dispatches(nodes, allocations, keys)
    payload = bytearray(payload)
    start = len(payload) - 32 * len(nodes[0].buffers)
    positions = {
        view.name: start + 32 * index for index, view in enumerate(nodes[0].buffers)
    }
    params, lhs, out = (positions[name] for name in ("params", "lhs", "out"))
    if corruption == "alignment":
        struct.pack_into("<Q", payload, params + 16, 4)
    elif corruption == "extent":
        struct.pack_into("<Q", payload, out + 16, 31)
    elif corruption == "stride":
        struct.pack_into("<I", payload, out + 12, 0)
    elif corruption == "length":
        struct.pack_into("<Q", payload, out + 24, 7)
    elif corruption == "register":
        struct.pack_into("<I", payload, out, 1)
    else:
        payload[out + 8 : out + 12] = payload[lhs + 8 : lhs + 12]
        if corruption == "overlap":
            struct.pack_into("<I", payload, lhs, 2)
            struct.pack_into("<I", payload, out + 4, 1)
            struct.pack_into("<Q", payload, out + 16, 4)
    packet = tmp_path / "descriptor.bin"
    packet.write_bytes(payload)
    result = subprocess.run(
        [str(protocol_worker), str(packet), "--validate"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stdout
    assert result.stderr.strip()


@pytest.mark.parametrize(
    "corruption", ["version", "count", "size", "address", "truncated", "trailing"]
)
def test_view_readback_rejects_invalid_allocations(tmp_path, corruption):
    _, allocations, _ = _plan((_request(tmp_path),))
    response = bytearray(_response(allocations))
    if corruption == "version":
        response[:4] = b"BAD!"
    elif corruption == "count":
        response[4:8] = struct.pack("<I", 0)
    elif corruption == "size":
        response[16:24] = struct.pack("<Q", 1)
    elif corruption == "address":
        response[8:16] = bytes(8)
    elif corruption == "truncated":
        response.pop()
    else:
        response.append(0)
    with pytest.raises(RuntimeAdapterSetupError) as error:
        directx_runtime.decode_readbacks(response, allocations)
    assert error.value.details["reasonKind"] == "worker-output-invalid"


def test_view_executor_reads_the_requested_output_slice(tmp_path, monkeypatch):
    nodes, allocations, keys = _plan((_request(tmp_path),))
    monkeypatch.setattr(directx_runtime, "_worker", lambda: Path("worker.exe"))

    def execute(command, **kwargs):
        assert command[-1] == "selected adapter"
        payload = bytearray(_response(allocations))
        offset = 8
        for allocation in allocations:
            size = allocation.views[0].allocation_size
            if allocation.allocation_id == "out":
                payload[offset + 16 + 12 : offset + 16 + 20] = struct.pack("<2f", 3, 6)
            offset += 16 + size
        Path(command[2]).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(directx_runtime, "_run", execute)
    state = SimpleNamespace(details={})
    result = directx_runtime.execute_buffer_views(
        nodes, allocations, keys, state, device=SimpleNamespace(name="selected adapter")
    )
    assert result["result"]["values"] == [3.0, 6.0]
    assert len(state.details["directxRuntime"]["allocations"]) == 3


SOURCE = """cbuffer Params : register(b0) { uint multiplier; };
StructuredBuffer<float> lhs : register(t0);
RWStructuredBuffer<float> out_values : register(u0);
[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
    out_values[id.x] = lhs[id.x] * multiplier;
}
"""


@pytest.mark.parametrize("constant_offset", [256, 512])
def test_directx_buffer_views_execute(tmp_path, constant_offset, retain_native_packets):
    module = _compile_native(tmp_path, SOURCE)
    request = replace(_request(tmp_path, constant_offset), loaded_artifact=module)
    state = SimpleNamespace(details={})
    outputs = DirectXComputeRuntime().dispatch(None, state, request)
    assert outputs["result"]["values"] == [3.0, 6.0]
    expected = bytearray(32)
    expected[12:20] = struct.pack("<2f", 3, 6)
    _assert_allocations(tmp_path, state, {"out": expected})
    by_id = {
        item["allocationId"]: item
        for item in state.details["directxRuntime"]["allocations"]
    }
    assert by_id["params"]["views"][0]["byteOffset"] == constant_offset


@pytest.fixture
def retain_native_packets(tmp_path, monkeypatch):
    execute = directx_runtime._run

    def record(command, **kwargs):
        result = execute(command, **kwargs)
        if (
            not isinstance(command, str)
            and Path(command[0]).name == "directx-runtime.exe"
        ):
            shutil.copy2(command[1], tmp_path / "native-request.bin")
            if Path(command[2]).is_file():
                shutil.copy2(command[2], tmp_path / "native-readbacks.bin")
            (tmp_path / "worker.json").write_text(
                json.dumps(
                    {
                        "workerSHA256": (
                            hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()
                        ),
                        "adapter": command[-1],
                        "returnCode": result.returncode,
                        "stderr": result.stderr.decode(errors="replace"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(directx_runtime, "_run", record)


def _compile_native(tmp_path, source_text, stem="views"):
    if os.environ.get("CROSTL_REQUIRE_DIRECTX_BUFFER_VIEWS") != "1":
        pytest.skip("set CROSTL_REQUIRE_DIRECTX_BUFFER_VIEWS=1 for native buffer views")
    assert sys.platform == "win32", "native buffer view proof requires Windows"
    dxc = shutil.which("dxc")
    assert dxc, "DXC is required"
    source, module = tmp_path / f"{stem}.hlsl", tmp_path / f"{stem}.dxil"
    source.write_text(source_text, encoding="utf-8")
    compiled = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "main", "-WX", str(source), "-Fo", str(module)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    (tmp_path / f"{stem}-identity.json").write_text(
        json.dumps(
            {
                "sourceSHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "moduleSHA256": hashlib.sha256(module.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return module.read_bytes()


def _assert_allocations(tmp_path, state, expected):
    native = state.details["directxRuntime"]
    assert native["runtime"] == "native-buffer-views"
    by_id = {item["allocationId"]: item for item in native["allocations"]}
    assert len(by_id) == len(native["allocations"])
    for name, payload in expected.items():
        assert by_id[name]["readbackSHA256"] == hashlib.sha256(payload).hexdigest()
        (tmp_path / f"{name}-expected.bin").write_bytes(payload)
    (tmp_path / "evidence.json").write_text(
        json.dumps(native, indent=2), encoding="utf-8"
    )


def _shared_requests(tmp_path, mode):
    request = _request(tmp_path)
    buffers = dict(request.buffers)
    if mode == "read-only":
        lhs = buffers["lhs"]
        buffers["rhs"] = replace(
            lhs,
            name="rhs",
            binding=replace(lhs.binding, name="rhs", binding=1),
            value=[5.0, 7.0],
            allocation=replace(lhs.allocation, byte_offset=12),
        )
        return (replace(request, buffers=buffers),)
    if mode == "disjoint-writes":
        out = buffers["out"]
        buffers["other"] = replace(
            out,
            name="other",
            binding=replace(out.binding, name="other", binding=1),
            allocation=replace(out.allocation, byte_offset=24),
            metadata={"runtimeValueName": "other"},
        )
        return (replace(request, buffers=buffers),)
    second = dict(buffers)
    second["lhs"] = replace(
        buffers["lhs"],
        value=None,
        source=None,
        allocation=buffers["out"].allocation,
    )
    second["out"] = replace(
        buffers["out"],
        allocation=replace(
            buffers["out"].allocation, allocation_id="final", byte_offset=4
        ),
        metadata={"runtimeValueName": "final"},
    )
    return (request, replace(request, buffers=second))


@pytest.mark.parametrize("mode", ["read-only", "disjoint-writes", "sequence"])
def test_shared_allocation_packet(tmp_path, protocol_worker, mode):
    nodes, allocations, keys = _plan(_shared_requests(tmp_path, mode))
    payload, descriptions = directx_runtime.encode_dispatches(nodes, allocations, keys)
    packet = tmp_path / "shared.bin"
    packet.write_bytes(payload)
    result = subprocess.run(
        [str(protocol_worker), str(packet), "--validate"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert len(descriptions) == (4 if mode == "sequence" else 3)
    repeated = {
        item["allocationId"]: item["views"]
        for item in descriptions
        if len(item["views"]) > 1
    }
    assert ("lhs" if mode == "read-only" else "out") in repeated


@pytest.mark.parametrize("mode", ["read-only", "disjoint-writes", "sequence"])
def test_shared_allocations_execute(tmp_path, mode, retain_native_packets):
    source = SOURCE
    if mode == "read-only":
        source = source.replace(
            "RWStructuredBuffer",
            "StructuredBuffer<float> rhs : register(t1);\nRWStructuredBuffer",
        )
        source = source.replace(
            "lhs[id.x] * multiplier", "(lhs[id.x] + rhs[id.x]) * multiplier"
        )
    elif mode == "disjoint-writes":
        source = source.replace(
            "[numthreads",
            "RWStructuredBuffer<float> other : register(u1);\n[numthreads",
        )
        source = source.replace("}\n", "    other[id.x] = lhs[id.x] + multiplier;\n}\n")
    module = _compile_native(tmp_path, source)
    requests = tuple(
        replace(item, loaded_artifact=module)
        for item in _shared_requests(tmp_path, mode)
    )
    state = SimpleNamespace(details={})
    outputs = DirectXComputeRuntime().dispatch_sequence(None, state, requests)
    values = [18.0, 27.0] if mode == "read-only" else [3.0, 6.0]
    assert outputs["result"]["values"] == values
    output = bytearray(32)
    output[12:20] = struct.pack("<2f", *values)
    expected = {"out": output}
    if mode == "disjoint-writes":
        assert outputs["other"]["values"] == [4.0, 5.0]
        output[24:32] = struct.pack("<2f", 4, 5)
    elif mode == "sequence":
        assert outputs["final"]["values"] == [9.0, 18.0]
        final = bytearray(32)
        final[4:12] = struct.pack("<2f", 9, 18)
        expected["final"] = final
    _assert_allocations(tmp_path, state, expected)
    assert len(state.details["directxRuntime"]["allocations"]) == (
        4 if mode == "sequence" else 3
    )


def test_ranged_execution_rejects_wrong_backend(tmp_path):
    module = _FakeCompushady(backend="vulkan")
    runtime = DirectXComputeRuntime(
        module_loader=lambda name: module,
        platform_name="win32",
        buffer_view_executor=lambda *args: pytest.fail(
            "wrong backend reached executor"
        ),
    )
    with pytest.raises(RuntimeExecutorUnavailable, match="Direct3D 12 backend"):
        runtime.dispatch(None, None, _request(tmp_path))


@pytest.mark.parametrize("limit", ["allocation", "total"])
def test_ranged_execution_checks_limits_before_merging_uploads(
    tmp_path, monkeypatch, limit
):
    from crosstl.project import native_runtime_drivers

    request = _request(tmp_path)
    buffers = dict(request.buffers)
    for name in ("out",) if limit == "allocation" else buffers:
        binding = buffers[name]
        buffers[name] = replace(
            binding,
            allocation=replace(
                binding.allocation,
                allocation_byte_length=(257 if limit == "allocation" else 200)
                * 1024
                * 1024,
            ),
        )
    monkeypatch.setattr(
        native_runtime_drivers,
        "_prepared_allocation_payload",
        lambda *args, **kwargs: pytest.fail(
            "oversized request reached host allocation"
        ),
    )
    with pytest.raises(RuntimeAdapterSetupError) as error:
        _plan((replace(request, buffers=buffers),))
    assert error.value.details["reasonKind"] == (
        "allocation-size-limit" if limit == "allocation" else "request-size-limit"
    )


def test_ranged_execution_rejects_mixed_access_states(tmp_path):
    request = _request(tmp_path)
    buffers = dict(request.buffers)
    buffers["out"] = replace(
        buffers["out"],
        allocation=replace(buffers["out"].allocation, allocation_id="lhs"),
    )
    with pytest.raises(RuntimeAdapterSetupError) as error:
        directx_runtime.encode_dispatches(*_plan((replace(request, buffers=buffers),)))
    assert error.value.details["reasonKind"] == "allocation-state-incompatible"
    assert error.value.details["allocationId"] == "lhs"


def test_native_process_timeout_reaps_child():
    with pytest.raises(RuntimeAdapterSetupError) as error:
        directx_runtime._run(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1
        )
    assert error.value.details["reasonKind"] == "operation-timeout"


@pytest.mark.parametrize(
    "tree_kill_error",
    [OSError("missing taskkill"), subprocess.TimeoutExpired("taskkill", 15)],
)
def test_windows_timeout_reaps_worker_when_tree_kill_fails(
    monkeypatch, tree_kill_error
):
    calls = []

    class Process:
        pid = 123
        returncode = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert self.returncode == -1

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            calls.append("reaped")
            return b"", b""

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -1
            calls.append("killed")

    def tree_kill(*args, **kwargs):
        raise tree_kill_error

    monkeypatch.setattr(directx_runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        directx_runtime.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    monkeypatch.setattr(directx_runtime.subprocess, "run", tree_kill)
    with pytest.raises(RuntimeAdapterSetupError) as error:
        directx_runtime._run(["worker"], timeout=1)
    assert error.value.details["reasonKind"] == "operation-timeout"
    assert calls == ["killed", "reaped"]


def test_worker_cache_retries_failed_compilation(tmp_path, monkeypatch):
    compiler = tmp_path / "cl.exe"
    compiler.touch()
    monkeypatch.setattr(
        directx_runtime,
        "sys",
        SimpleNamespace(platform="win32", executable=str(tmp_path / "python.exe")),
    )
    monkeypatch.setattr(
        directx_runtime, "_compiler_environment", lambda: (str(compiler), {})
    )
    monkeypatch.setattr(directx_runtime, "_WORKERS", {})
    builds = []

    def compile_worker(command, **kwargs):
        output = Path(
            next(argument[3:] for argument in command if argument.startswith("/Fe"))
        )
        builds.append(output)
        if len(builds) == 1:
            return subprocess.CompletedProcess(command, 1, b"failed", b"")
        output.write_bytes(b"test worker")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(directx_runtime, "_run", compile_worker)
    with pytest.raises(RuntimeAdapterSetupError, match="compilation failed"):
        directx_runtime._worker()
    assert not builds[0].parent.exists()
    assert not directx_runtime._WORKERS
    worker = directx_runtime._worker()
    assert directx_runtime._worker() == worker
    assert len(builds) == 2
    for directory, _ in directx_runtime._WORKERS.values():
        directory.cleanup()


def test_required_native_view_gate_keeps_existing_windows_runner():
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    steps = workflow["jobs"]["mlx-metal-porting"]["steps"]
    step = next(
        item for item in steps if item["name"] == "Validate DirectX allocation views"
    )
    assert step["if"] == "runner.os == 'Windows'"
    assert step["env"]["CROSTL_REQUIRE_DIRECTX_BUFFER_VIEWS"] == "1"
    assert "--timeout-seconds 180" in step["run"]
    assert "-n auto" in step["run"]
    assert "tests/test_translator/test_directx_buffer_views.py" in step["run"]
    upload = next(
        item
        for item in steps
        if item["name"] == "Upload DirectX allocation view evidence"
    )
    assert upload["if"] == "always() && runner.os == 'Windows'"
    assert upload["with"]["if-no-files-found"] == "error"
