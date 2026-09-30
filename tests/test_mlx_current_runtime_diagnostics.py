import json
from types import SimpleNamespace

import pytest

from tests.test_translator import test_mlx_current_arg_reduce as proof
from tests.test_translator.test_directx_reduction_primitives import _buffers
from tests.test_translator.test_native_runtime_drivers import _directx_dispatch_request


def test_runtime_stage_records_failure_before_propagating(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="dispatch failed"):
        with proof._runtime_stage(
            tmp_path,
            "execute",
            entry="argmin_float32",
            target="directx",
            case="strided",
        ):
            raise RuntimeError("dispatch failed")
    records = [
        json.loads(line)
        for line in (tmp_path / "runtime-progress.jsonl").read_text().splitlines()
    ]
    assert [record["status"] for record in records] == ["started", "failed"]
    assert all(record["case"] == "strided" for record in records)
    assert '"status": "failed"' in capsys.readouterr().out


def test_runtime_parity_retains_inputs_and_abi_on_dispatch_failure(
    tmp_path, monkeypatch
):
    descriptor = {"target": "directx", "bindings": []}
    monkeypatch.setattr(
        proof, "_runtime_descriptor", lambda *args: (descriptor, tmp_path)
    )
    monkeypatch.setattr(
        proof, "_runtime_dispatch_request", lambda *args: (object(), "out_")
    )
    request = {
        "rows": 1,
        "axisSize": 2,
        "axisStride": 1,
        "rowStride": 2,
        "inputBits": [0, 1065353216],
    }
    monkeypatch.setattr(
        proof, "_cases", lambda: iter([("sample", request, [[0.0, 1.0]], [0.0, 1.0])])
    )

    def execute(_request):
        assert json.loads((tmp_path / "sample-input.json").read_text()) == {
            **request,
            "expected": [0],
        }
        assert (
            json.loads((tmp_path / "native-loader-abi.json").read_text()) == descriptor
        )
        raise RuntimeError("native dispatch failed")

    monkeypatch.setattr(
        proof,
        "_runtime_executor",
        lambda _target, _work: SimpleNamespace(
            is_available=lambda _request: SimpleNamespace(available=True),
            run=execute,
        ),
    )
    with pytest.raises(RuntimeError, match="native dispatch failed"):
        proof._run_runtime_parity(None, tmp_path, "directx", "argmin_float32")
    records = [
        json.loads(line)
        for line in (tmp_path / "runtime-progress.jsonl").read_text().splitlines()
    ]
    assert [(record["stage"], record["status"]) for record in records] == [
        ("package", "started"),
        ("package", "completed"),
        ("availability", "started"),
        ("availability", "completed"),
        ("execute", "started"),
        ("execute", "failed"),
    ]
    assert not (tmp_path / "runtime-cases.json").exists()


def test_directx_dispatch_retains_actual_module_and_packed_resources(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    request = replace(
        _directx_dispatch_request(tmp_path),
        buffers=_buffers("combined"),
        loaded_artifact=b"actual runtime module",
    )
    runtime = proof._RecordingDirectXRuntime(tmp_path)

    def dispatch(self, adapter, state, requests):
        assert self is runtime
        assert requests == (request,)
        directory = tmp_path / "native-dispatches" / "0"
        assert (directory / "kernel.dxil").read_bytes() == request.loaded_artifact
        recorded = json.loads((directory / "dispatch.json").read_text())
        assert (
            recorded["shaderSha256"]
            == proof.hashlib.sha256(request.loaded_artifact).hexdigest()
        )
        assert recorded["dispatch"] == request.dispatch.to_json()
        resources = {item["name"]: item for item in recorded["resources"]}
        assert resources["AxisSize"]["payloadHex"].startswith("2000000000000000")
        assert resources["AxisSize"]["binding"] == 7
        assert resources["AxisSize"]["namespace"] == "cbv"
        assert resources["AxisSize"]["allocationSize"] == 256
        assert resources["outputValues"]["upload"] is False
        assert resources["outputValues"]["payloadHex"] == ""
        assert resources["__crosstl_descriptor_gap_srv1"]["stride"] == 4
        raise RuntimeError("dispatch stalled")

    monkeypatch.setattr(proof.DirectXComputeRuntime, "dispatch_sequence", dispatch)
    with pytest.raises(RuntimeError, match="dispatch stalled"):
        runtime.dispatch(None, None, request)
