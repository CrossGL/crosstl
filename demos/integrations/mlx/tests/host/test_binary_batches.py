"""Binary batch coverage, byte offsets and retained native execution contracts."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import verify_binary_batches as verifier


def test_inventory_covers_every_base_operation_and_large_layouts():
    cases = verifier.cases()
    assert len(cases) == len({case["id"] for case in cases}) == 48
    assert sum(len(verifier.batch_sizes(case)) for case in cases) == 98
    assert sum(verifier.copy_count(case) for case in cases) == 4
    assert {case["entry"] for case in cases} == (
        set(verifier.BINARY_ENTRIES) | set(verifier.COMPARISON_ENTRIES)
    ) - {"vv_NaNEqualfloat32"}
    assert {case["count"] for case in cases} >= {65535, 65536, 131075}
    for case in cases:
        for side in range(2):
            storage, view, _, _, offset, low, _ = verifier.operand(np, case, side)
            assert view.size == case["count"] and offset + low > 0
            assert np.shares_memory(storage, view)
            chunks = [
                view.reshape(-1)[i : i + 65535] for i in range(0, view.size, 65535)
            ]
            assert len(chunks) == len(
                {hashlib.sha256(v.tobytes()).hexdigest() for v in chunks}
            )
        expected = verifier.reference(np, case)
        assert expected.dtype in (
            np.dtype("float32"),
            np.dtype("uint32"),
            np.dtype("int32"),
            np.dtype("bool"),
        )
        assert np.isfinite(expected).all()


def event_for(target, entry, count, grid, values, types, **extra):
    inputs = {
        name: {"dtype": types[name], "shape": [len(value)], "values": value}
        for name, value in values.items()
    }
    bindings = {
        name: {
            "dtype": value["dtype"],
            "shape": value["shape"],
            "binding": {
                "metadata": {
                    "scalarLayout": {
                        "memberName": (
                            name
                            if target != "directx"
                            else entry.rstrip("_") + "_" + name
                        ),
                        "elementType": value["dtype"],
                        "elementStrideBytes": np.dtype(value["dtype"]).itemsize,
                    }
                }
            },
        }
        for name, value in inputs.items()
    }
    native = {"opengl": "main", "directx": "CSMain"}.get(target, entry)
    return {
        "entry": entry,
        "target": target,
        "threads": count,
        "dispatchVersion": 3,
        "workgroupCount": grid,
        "workgroupSize": [1, 1, 1],
        "inputs": inputs,
        "details": {
            "request": {
                "buffers": bindings,
                "target": target,
                "entryPoint": native,
                "dispatch": {
                    "entryPoint": native,
                    "workgroupCount": grid,
                    "workgroupSize": [1, 1, 1],
                },
            }
        },
        **extra,
    }


def synthetic_case(directory, target, case):
    saved, sources, events = {}, [], []
    for side in range(2):
        storage, view, shape, strides, offset, low, high = verifier.operand(
            np, case, side
        )
        saved[f"before{side}"] = storage.copy()
        saved[f"after{side}"] = storage.copy()
        sources.append(view.reshape(-1))
        if view.flags.c_contiguous:
            continue
        padding = max(0, 2 - len(shape))
        destination = [0] * padding + [
            int(np.prod(shape[i + 1 :])) for i in range(len(shape))
        ]
        shape, strides = [1] * padding + shape, [0] * padding + strides
        grid = [(shape[-1] + 1) // 2, shape[-2], int(np.prod(shape[:-2]))]
        metadata = {
            "shape": shape,
            "sourceStrides": strides,
            "destinationStrides": destination,
            "sourceOffset": -low,
            "destinationOffset": 0,
            "sourceCount": high - low + 1,
            "destinationCount": case["count"],
            "preserveDestination": False,
            "workgroupCount": grid,
        }
        logical = "bool_" if case["dtype"] == "bool_" else "uint32"
        physical = "bool" if logical == "bool_" and target == "metal" else "uint32"
        source, result = storage[offset + low : offset + high + 1], view.copy().reshape(
            -1
        )
        if logical == "bool_":
            source, result = source.astype(physical), result.astype(physical)
            guard = np.asarray(verifier.BOOLEAN_GUARD, dtype=physical).tolist()
        else:
            source, result, guard = (
                source.view("uint32"),
                result.view("uint32"),
                verifier.COPY_GUARD,
            )
        values = {
            "src": source.tolist(),
            "dst": [False if physical == "bool" else 0] * case["count"] + guard,
            "src_shape": shape,
            "src_strides": strides,
            "dst_strides": destination,
            "ndim": [len(shape)],
            "src_offset": [-low],
            "dst_offset": [0],
        }
        types = {
            name: physical if name in {"src", "dst"} else "int32" for name in values
        }
        events.append(
            event_for(
                target,
                "ggn2_dynamic_copy" + logical + logical,
                case["count"],
                grid,
                values,
                types,
                copyMetadata=metadata,
                copyValues=result.tolist(),
                copyGuardWords=guard,
            )
        )
    expected = verifier.reference(np, case)
    saved["actual"] = expected.copy()
    np.savez(directory / (case["id"] + ".npz"), **saved)
    first = 0
    for size in verifier.batch_sizes(case):
        boolean = "bool" if target == "metal" else "uint32"
        source_type = boolean if case["dtype"] == "bool_" else case["dtype"]
        output_type = boolean if expected.dtype == np.bool_ else str(expected.dtype)
        guard = np.asarray(
            (
                verifier.BOOLEAN_GUARD
                if expected.dtype == np.bool_
                else verifier.COPY_GUARD
            ),
            dtype=boolean if expected.dtype == np.bool_ else "uint32",
        )
        guard = (
            guard.view("float32")
            if output_type == "float32"
            else guard.astype(output_type)
        ).tolist()
        values = {
            name: source[first : first + size].astype(source_type).tolist()
            for name, source in zip(("a", "b"), sources)
        }
        values.update(
            c=[False if output_type == "bool" else 0] * size + guard, size=[size]
        )
        events.append(
            event_for(
                target,
                case["entry"],
                size,
                [size, 1, 1],
                values,
                {
                    "a": source_type,
                    "b": source_type,
                    "c": output_type,
                    "size": "uint32",
                },
                binaryValues=expected.reshape(-1)[first : first + size]
                .astype(output_type)
                .tolist(),
                binaryGuardValues=guard,
            )
        )
        first += size
    item = verifier.record(np, case, saved)
    item.update(dispatchStart=0, dispatchCount=len(events))
    return item, events


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("index", [0, 3, 17, 19, 39, 42, 43, 44, 45, 46, 47])
def test_binary_batches_require_native_readbacks_and_receipts(
    tmp_path, monkeypatch, target, index
):
    case = verifier.cases()[index]
    item, events = synthetic_case(tmp_path, target, case)
    audited = []
    monkeypatch.setattr(verifier, "audit_native_execution", audited.append)
    verifier.audit_case(np, case, item, events, tmp_path)
    assert audited == events


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "first-input",
        "second-input",
        "output",
        "guard",
        "initialization",
        "dtype",
        "coercion",
        "request",
        "receipt",
        "native-audit",
    ],
)
def test_binary_audit_rejects_batch_corruption(tmp_path, monkeypatch, fault):
    case = verifier.cases()[0]
    item, events = synthetic_case(tmp_path, "opengl", case)

    def audit(event):
        if fault == "native-audit":
            raise ValueError("Native receipt missing")

    monkeypatch.setattr(verifier, "audit_native_execution", audit)
    if fault == "missing":
        events.pop()
    elif fault == "duplicate":
        events[-1] = copy.deepcopy(events[0])
    elif fault in {"first-input", "second-input"}:
        name = "a" if fault == "first-input" else "b"
        events[-1]["inputs"][name]["values"][0] += 1
    elif fault == "output":
        events[-1]["binaryValues"][0] += 1
    elif fault == "guard":
        events[-1]["binaryGuardValues"][0] = 0
    elif fault == "initialization":
        events[-1]["inputs"]["c"]["values"][0] = 1
    elif fault == "dtype":
        events[-1]["inputs"]["c"]["dtype"] = "float64"
    elif fault == "coercion":
        events[-1]["binaryValues"][0] = str(events[-1]["binaryValues"][0])
    elif fault == "request":
        events[-1]["details"]["request"]["dispatch"]["workgroupSize"] = [2, 1, 1]
    elif fault == "receipt":
        item["hashes"]["actual"] = "0" * 64
    with pytest.raises(ValueError):
        verifier.audit_case(np, case, item, events, tmp_path)


@pytest.mark.parametrize("fault", ["source", "result", "offset", "guard", "request"])
def test_binary_materialization_evidence_is_required(tmp_path, monkeypatch, fault):
    case = verifier.cases()[-1]
    item, events = synthetic_case(tmp_path, "directx", case)
    monkeypatch.setattr(verifier, "audit_native_execution", lambda event: None)
    if fault == "source":
        events[1]["inputs"]["src"]["values"][0] ^= 1
    elif fault == "result":
        events[1]["copyValues"][-1] ^= 1
    elif fault == "offset":
        events[1]["copyMetadata"]["sourceOffset"] += 1
    elif fault == "guard":
        events[1]["copyGuardWords"][-1] ^= 1
    else:
        events[1]["details"]["request"]["target"] = "metal"
    with pytest.raises(ValueError):
        verifier.audit_case(np, case, item, events, tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "timeout",
        "missing",
        "passed",
        "commit",
        "target",
        "casesPerPath",
        "dispatchCount",
        "fullUpstreamSuite",
    ],
)
def test_bounded_binary_proof_requires_complete_evidence(tmp_path, monkeypatch, fault):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    output = tmp_path / "proof"
    summary = {
        "passed": True,
        "commit": verifier.COMMIT,
        "target": "opengl",
        "casesPerPath": 48,
        "dispatchCount": 102,
        "fullUpstreamSuite": False,
    }
    if fault in summary:
        summary[fault] = None

    def run(command, **kwargs):
        assert command[command.index("--timeout-seconds") + 1] == "600"
        assert command[command.index("-m") + 1] == verifier.__name__
        output.mkdir()
        if fault != "missing":
            (output / "evidence.json").write_text(json.dumps(summary))
        return SimpleNamespace(returncode=124 if fault == "timeout" else 0)

    monkeypatch.setattr(verifier.subprocess, "run", run)
    if fault:
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            verifier.run_bounded(tmp_path / "mlx", packages, output)
    else:
        assert verifier.run_bounded(tmp_path / "mlx", packages, output) == summary
    assert (
        output.with_suffix(".stdout").exists()
        and output.with_suffix(".stderr").exists()
    )
    assert output.with_suffix(".command.json").exists()


def test_binary_batches_are_required_in_existing_three_platform_job():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["portable-host"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "opengl",
        "directx",
    }
    step = next(
        s
        for s in job["steps"]
        if s.get("name") == "Execute upstream tests on the translated backend"
    )
    assert not step.get("if") and not step.get("continue-on-error")
    assert "--cast-batches --large-copies --binary-batches" in step["run"]
    assert "--timeout-seconds 4000" in step["run"]
    contracts = next(
        s for s in job["steps"] if s.get("name") == "Validate portable host contracts"
    )
    assert contracts["if"] == "runner.os == 'Linux'"
    for filename in (
        "test_cast_batches.py",
        "test_large_copies.py",
        "test_binary_batches.py",
    ):
        assert "demos/integrations/mlx/tests/host/" + filename in contracts["run"]
