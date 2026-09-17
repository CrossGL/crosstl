import json
from types import SimpleNamespace

import pytest

from tests.test_translator import test_mlx_current_arg_reduce as proof


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
        lambda _target: SimpleNamespace(
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
