import json
import math
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.tests.kernels import test_current_arg_reduce as proof
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

    closed = []
    monkeypatch.setattr(
        proof,
        "_runtime_executor",
        lambda _target, _work: SimpleNamespace(
            is_available=lambda _request: SimpleNamespace(available=True),
            run=execute,
            runtime_adapter=SimpleNamespace(
                runtime=SimpleNamespace(close=lambda: closed.append(True))
            ),
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
    assert closed == [True]


@pytest.mark.parametrize("entry", proof.RUNTIME_ENTRIES + proof.INTEGER_RUNTIME_ENTRIES)
@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
def test_checked_dispatch_preserves_rows_guards_and_input_observations(
    tmp_path, monkeypatch, target, entry
):
    parameters = (
        "in",
        "out",
        "shape",
        "in_strides",
        "out_strides",
        "ndim",
        "axis_stride",
        "axis_size",
    )
    descriptor = {
        "bindings": [
            {
                "name": f"binding_{index}",
                "provenance": {"sourceResource": {"parameter": parameter}},
            }
            for index, parameter in enumerate(parameters)
        ]
    }
    if target == "directx":
        descriptor["bindings"].append(
            {
                "name": "launch_geometry",
                "provenance": {"kind": "generated-execution-input"},
            }
        )
    captured = []

    def build(abi, package, inputs, outputs, dispatch, **options):
        assert abi is descriptor and package == tmp_path
        captured.append((inputs, outputs, dispatch, options))
        return SimpleNamespace(
            fixture=SimpleNamespace(
                inputs=[
                    SimpleNamespace(
                        name="launch_geometry",
                        dtype="uint32",
                        shape=(3,),
                        values=dispatch["workgroupCount"],
                        metadata={"source": "dispatch.workgroupCount"},
                    )
                ]
            ),
            execution_plan=SimpleNamespace(
                diagnostics=(),
                dispatch=SimpleNamespace(
                    workgroup_size=tuple(dispatch["workgroupSize"]),
                    workgroup_count=tuple(dispatch["workgroupCount"]),
                ),
            ),
        )

    monkeypatch.setattr(proof, "build_native_loader_dispatch_request", build)
    cases = (
        proof._integer_cases(entry)
        if entry in proof.INTEGER_RUNTIME_ENTRIES
        else proof._cases()
    )
    for _name, request, rows, storage in cases:
        original = storage.copy()
        expected = proof._expected_indices(entry, rows)
        _, output = proof._runtime_dispatch_request(
            descriptor, tmp_path, target, entry, request, storage, expected
        )
        inputs, outputs, dispatch, options = captured[-1]
        assert output == "binding_1"
        assert storage == original
        assert dispatch == {"workgroupCount": [1, 2, 1], "workgroupSize": [32, 1, 1]}
        assert options == {
            "expected_target": target,
            "observe_inputs": [f"binding_{index}" for index in (0, 2, 3, 4, 5, 6, 7)],
        }
        assert inputs[output]["values"] == [proof.OUTPUT_GUARD] * 34
        assert outputs[output] == {
            "dtype": "uint32",
            "shape": [34],
            "values": expected + [proof.OUTPUT_GUARD] * 32,
        }
        values = inputs["binding_0"]["values"]
        assert inputs["binding_0"]["dtype"] == proof._runtime_input_type(target, entry)
        assert inputs["binding_0"]["shape"] == [len(values)]
        assert len(values) > request["rowStride"] + (
            (request["axisSize"] + 127) // 128 * 128 * request["axisStride"]
        )
        for observed, value in zip(values, original):
            if isinstance(value, float) and not math.isfinite(value):
                assert observed == (
                    "nan"
                    if math.isnan(value)
                    else "+infinity" if value > 0 else "-infinity"
                )
            else:
                assert observed == value
        dtype = entry.split("_", 1)[1]
        if dtype.endswith("16"):
            assert len(values) % 2 == 0
        if dtype == "float32":
            sentinel = "+infinity" if entry.startswith("argmax") else "-infinity"
        else:
            signed = dtype.startswith("int")
            width = int(dtype[3:] if signed else dtype[4:])
            low = -(1 << (width - 1)) if signed else 0
            high = (1 << (width - int(signed))) - 1
            sentinel = high if entry.startswith("argmax") else low
        assert values[len(original) :] == [sentinel] * (len(values) - len(original))
        json.dumps(inputs, allow_nan=False)
    assert len(captured) == 8


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
