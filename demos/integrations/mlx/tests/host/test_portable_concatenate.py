"""Concatenation payload, copy-sequence, upstream and CI evidence contracts."""

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import runtime
from demos.integrations.mlx.portable_host import verify_concatenate as proof
from tests.ci_helpers import assert_paths_covered


def evidence(target, native=True):
    records, trace = [], []
    for case in proof.cases():
        arrays = proof.inputs(np, case)
        expected = np.concatenate(arrays, axis=case["axis"])
        nonempty = [a for a in arrays if a.size]
        records.append(
            {
                **case,
                "inputPayloads": [proof.payload(np, a) for a in arrays],
                "actualPayload": proof.payload(np, expected),
                "resultShape": list(expected.shape),
                "resultDtype": "mlx.core." + case["dtype"].removesuffix("_"),
                "inputUnchanged": True,
                "dispatchCount": len(nonempty) if native else 0,
            }
        )
        staged = np.zeros_like(expected)
        axis = 0 if case["axis"] is None else case["axis"]
        offset = 0
        for position, source in enumerate(nonempty if native else []):
            if case["axis"] is None:
                source = source.reshape(-1)
            shape = list(source.shape)
            strides = [
                stride // source.itemsize if size != 1 else 0
                for size, stride in zip(shape, source.strides)
            ]
            dst_strides = [stride // expected.itemsize for stride in expected.strides]
            while len(shape) < 2:
                shape.insert(0, 1)
                strides.insert(0, 0)
                dst_strides.insert(0, 0)
            extents = [(size - 1) * stride for size, stride in zip(shape, strides)]
            low, high = sum(min(v, 0) for v in extents), sum(max(v, 0) for v in extents)
            selection = [slice(None)] * expected.ndim
            selection[axis] = slice(offset, offset + source.shape[axis])
            staged[tuple(selection)] = source
            grid = [(shape[-1] + 1) // 2, shape[-2], int(np.prod(shape[:-2]))]
            boolean = case["dtype"] == "bool_"
            guard = runtime.BOOLEAN_GUARD if boolean else runtime.COPY_GUARD
            if boolean and target != "metal":
                guard = [int(v) for v in guard]
            trace.append(
                {
                    "entry": proof.BOOLEAN_COPY_ENTRY if boolean else proof.COPY_ENTRY,
                    "target": target,
                    "dispatchVersion": runtime.DISPATCH_VERSION,
                    "threads": source.size,
                    "workgroupSize": [1, 1, 1],
                    "workgroupCount": grid,
                    "copyGuardWords": guard,
                    "copyValues": proof.storage(np, staged),
                    "copyMetadata": {
                        "shape": shape,
                        "sourceStrides": strides,
                        "destinationStrides": dst_strides,
                        "sourceOffset": -low,
                        "destinationOffset": (
                            offset * (expected.strides[axis] // expected.itemsize)
                        ),
                        "sourceCount": high - low + 1,
                        "destinationCount": expected.size,
                        "preserveDestination": bool(position),
                        "workgroupCount": grid,
                    },
                }
            )
            offset += source.shape[axis]
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "case",
        "input",
        "payload",
        "dtype",
        "shape",
        "preserved",
        "count",
        "missing",
        "extra",
        "copy",
        "guard",
        "geometry",
        "offset",
        "stride",
        "mode",
    ],
)
def test_concatenate_requires_exact_evidence(target, fault):
    records, trace = evidence(target)
    if fault == "case":
        records.pop()
    elif fault == "input":
        records[0]["inputPayloads"][0] = ""
    elif fault == "payload":
        records[0]["actualPayload"] = ""
    elif fault == "dtype":
        records[0]["resultDtype"] = "mlx.core.float16"
    elif fault == "shape":
        records[0]["resultShape"] = [24]
    elif fault == "preserved":
        records[0]["inputUnchanged"] = False
    elif fault == "count":
        records[0]["dispatchCount"] = 1
    elif fault == "missing":
        trace.pop()
    elif fault == "extra":
        trace.append(trace[-1])
    elif fault == "copy":
        trace[1]["copyValues"][0] ^= 1
    elif fault == "guard":
        trace[0]["copyGuardWords"] = []
    elif fault == "geometry":
        trace[0]["workgroupCount"] = [1, 1, 1]
    elif fault == "offset":
        trace[1]["copyMetadata"]["destinationOffset"] = 0
    elif fault == "stride":
        trace[0]["copyMetadata"]["destinationStrides"][0] += 1
    elif fault == "mode":
        trace[1]["copyMetadata"]["preserveDestination"] = False
    if fault:
        with pytest.raises(ValueError, match="Concatenation"):
            proof.validate(records, trace, native=True)
    else:
        proof.validate(records, trace, native=True)
        records, trace = evidence(target, native=False)
        proof.validate(records, trace, native=False)
        assert len(records) == 72


def test_concatenate_retains_input_output_abi_layout():
    assert runtime.Buffer.output.offset == 32
    assert ctypes.sizeof(runtime.Buffer) == 40
    assert runtime.copy_layout.INOUT == 2
    header = Path("demos/integrations/mlx/portable_host/dispatch.h").read_text()
    assert "CROSTL_MLX_BUFFER_INOUT = 2" in header


def test_concatenate_ci_requires_all_native_targets():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["small-row-reductions"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Execute concatenation host operations"
    )
    assert "portable_host.verify_concatenate" in step["run"]
    assert "--timeout-seconds 1500" in step["run"]
    assert "continue-on-error" not in step and "if" not in step
    translation = next(
        step
        for step in job["steps"]
        if step.get("name") == "Translate concatenation test reduction"
    )
    assert "--entry all_reduce_andbool_ --width 32" in translation["run"]
    triggers = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "push"):
        assert_paths_covered(
            triggers[event]["paths"],
            "demos/integrations/mlx/tests/host/test_portable_concatenate.py",
        )
    assert any(
        "test_portable_concatenate.py" in step.get("run", "")
        for step in workflow["jobs"]["portable-host"]["steps"]
    )


def test_concatenate_failed_worker_retains_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from demos.integrations.mlx.portable_host import verify_concatenate

    base, reductions = tmp_path / "base", tmp_path / "reduce"
    base.mkdir()
    reductions.mkdir()
    (base / "index.json").write_text(json.dumps({"target": "metal"}))
    monkeypatch.setattr(proof, "verify_prepared", lambda root: {})
    monkeypatch.setattr(
        proof,
        "load_index",
        lambda *args: {"descriptors": {"w32/all_reduce_andbool_": {}}},
    )
    monkeypatch.setattr(
        proof.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=124)
    )
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="cpu worker failed"):
        verify_concatenate.verify(
            SimpleNamespace(
                mlx_root=tmp_path,
                packages=base,
                reductions=reductions,
                output_dir=output,
            )
        )
    assert json.loads((output / "cpu.command.json").read_text())["returncode"] == 124
    assert not (output / "evidence.json").exists()
