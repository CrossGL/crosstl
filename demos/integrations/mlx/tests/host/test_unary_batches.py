"""Stored-layout unary batches and fail-closed native evidence."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import verify_unary_batches as verifier


def test_inventory_distinguishes_logical_and_stored_counts():
    cases = verifier.cases()
    assert len(cases) == len({case["id"] for case in cases}) == 18
    assert sum(len(verifier.batch_sizes(case)) for case in cases) == 36
    assert {case["operation"] for case in cases} == set(verifier.OPERATIONS)
    for case in cases:
        storage, view, _, _, offset, count = verifier.operand(np, case)
        assert view.size == case["count"]
        assert offset == 7 and np.shares_memory(storage, view)
        assert sum(verifier.batch_sizes(case)) == count
        chunks = [
            storage[offset + first : offset + min(first + 65535, count)]
            for first in range(0, count, 65535)
        ]
        assert len(chunks) == len(
            {hashlib.sha256(chunk.tobytes()).hexdigest() for chunk in chunks}
        )
        assert np.isfinite(verifier.apply(np, case, view)).all()
        if "broadcast" in case["layout"]:
            assert count < view.size


def test_round_reference_preserves_ties_to_even_and_negative_zero():
    actual = verifier.apply(
        np,
        {"operation": "Round"},
        np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype="float32"),
    )
    expected = np.array([-2, -2, -0.0, 0.0, 2, 2], dtype="float32")
    assert verifier.equal_storage(actual, expected)


def synthetic_case(directory, target, case):
    storage, view, _, strides, offset, _ = verifier.operand(np, case)
    saved = {
        "before": storage.copy(),
        "after": storage.copy(),
        "actual": verifier.apply(np, case, view),
    }
    np.savez(directory / (case["id"] + ".npz"), **saved)
    dtype = verifier.physical_type(case, target)
    entry = case["entry"]
    native = {"opengl": "main", "directx": "CSMain"}.get(target, entry)
    guard = verifier.guard_values(np, case, target)
    events, first = [], 0
    for size in verifier.batch_sizes(case):
        source = storage[offset + first : offset + first + size]
        values = {
            "in": source.astype(dtype).tolist(),
            "out": np.concatenate((np.zeros(size, dtype=dtype), guard)).tolist(),
            "size": [size],
        }
        inputs = {
            name: {
                "dtype": "uint32" if name == "size" else dtype,
                "shape": [len(value)],
                "values": value,
            }
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
                                entry.rstrip("_") + "_" + name
                                if target == "directx"
                                else name + "_" if name in {"in", "out"} else name
                            ),
                            "elementType": value["dtype"],
                            "elementStrideBytes": np.dtype(value["dtype"]).itemsize,
                        }
                    }
                },
            }
            for name, value in inputs.items()
        }
        events.append(
            {
                "entry": entry,
                "target": target,
                "threads": size,
                "dispatchVersion": 3,
                "workgroupCount": [size, 1, 1],
                "workgroupSize": [1, 1, 1],
                "inputs": inputs,
                "unaryValues": verifier.apply(np, case, source).astype(dtype).tolist(),
                "unaryGuardValues": guard.tolist(),
                "details": {
                    "request": {
                        "target": target,
                        "entryPoint": native,
                        "buffers": bindings,
                        "dispatch": {
                            "entryPoint": native,
                            "workgroupCount": [size, 1, 1],
                            "workgroupSize": [1, 1, 1],
                        },
                    }
                },
            }
        )
        first += size
    item = verifier.record(np, case, saved)
    item.update(dispatchStart=0, dispatchCount=len(events), strides=strides)
    return item, events


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "layout", ["dense", "transpose", "broadcast", "scalar-broadcast"]
)
def test_audit_requires_stored_order_and_native_receipts(
    tmp_path, monkeypatch, target, layout
):
    case = next(case for case in verifier.cases() if case["layout"] == layout)
    item, events = synthetic_case(tmp_path, target, case)
    receipts = []
    monkeypatch.setattr(verifier, "audit_native_execution", receipts.append)
    verifier.audit_case(np, case, item, events, tmp_path)
    assert receipts == events


@pytest.mark.parametrize(
    "fault",
    [
        "missing-batch",
        "repeated-batch",
        "input-offset",
        "output",
        "guard",
        "initialization",
        "strides",
        "size",
        "launch",
        "request",
        "type",
        "coercion",
        "receipt",
        "hash",
        "source",
    ],
)
def test_audit_rejects_incomplete_or_corrupt_batches(tmp_path, monkeypatch, fault):
    case = next(
        case
        for case in verifier.cases()
        if case["operation"] == "Negative" and case["count"] == 131075
    )
    item, events = synthetic_case(tmp_path, "directx", case)
    monkeypatch.setattr(verifier, "audit_native_execution", lambda event: None)
    if fault == "missing-batch":
        events.pop()
    elif fault == "repeated-batch":
        events[1] = copy.deepcopy(events[0])
    elif fault == "input-offset":
        events[1]["inputs"]["in"]["values"][0] += 1
    elif fault == "output":
        events[1]["unaryValues"][0] += 1
    elif fault == "guard":
        events[-1]["unaryGuardValues"][0] = 0
    elif fault == "initialization":
        events[0]["inputs"]["out"]["values"][0] = 1
    elif fault == "strides":
        item["strides"] = [0]
    elif fault == "size":
        events[1]["inputs"]["size"]["values"] = [1]
    elif fault == "launch":
        events[1]["workgroupCount"] = [1, 1, 1]
    elif fault == "request":
        events[0]["details"]["request"]["dispatch"]["entryPoint"] = "wrong"
    elif fault == "type":
        events[1]["details"]["request"]["buffers"]["in"]["binding"]["metadata"][
            "scalarLayout"
        ]["elementStrideBytes"] = 8
    elif fault == "coercion":
        events[1]["unaryValues"][0] = str(events[1]["unaryValues"][0])
    elif fault == "receipt":

        def reject(event):
            raise ValueError("Native receipt differs")

        monkeypatch.setattr(verifier, "audit_native_execution", reject)
    elif fault == "hash":
        item["hashes"]["actual"] = "wrong"
    else:
        path = tmp_path / (case["id"] + ".npz")
        with np.load(path) as saved:
            values = dict(saved)
        values["after"][0] += 1
        np.savez(path, **values)
    with pytest.raises(ValueError):
        verifier.audit_case(np, case, item, events, tmp_path)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
def test_boolean_batches_preserve_physical_width(tmp_path, monkeypatch, target):
    case = next(case for case in verifier.cases() if case["operation"] == "LogicalNot")
    item, events = synthetic_case(tmp_path, target, case)
    monkeypatch.setattr(verifier, "audit_native_execution", lambda event: None)
    verifier.audit_case(np, case, item, events, tmp_path)
    events[-1]["unaryValues"][0] = "true"
    with pytest.raises(ValueError):
        verifier.audit_case(np, case, item, events, tmp_path)


@pytest.mark.parametrize(
    "fault", [None, "command", "missing", "commit", "target", "coverage", "scope"]
)
def test_bounded_verification_rejects_missing_or_incomplete_evidence(
    tmp_path, monkeypatch, fault
):
    packages, output = tmp_path / "packages", tmp_path / "proof"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "metal"}))

    def run(command, **kwargs):
        assert command[2:4] == ["--timeout-seconds", "600"]
        assert command[-2:] == ["--output-dir", str(output)]
        output.mkdir()
        evidence = {
            "passed": True,
            "commit": verifier.COMMIT,
            "target": "metal",
            "casesPerPath": 18,
            "dispatchCount": 36,
            "fullUpstreamSuite": False,
        }
        if fault == "commit":
            evidence["commit"] = "wrong"
        elif fault == "target":
            evidence["target"] = "opengl"
        elif fault == "coverage":
            evidence["dispatchCount"] -= 1
        elif fault == "scope":
            evidence["fullUpstreamSuite"] = True
        if fault != "missing":
            (output / "evidence.json").write_text(json.dumps(evidence))
        return SimpleNamespace(returncode=1 if fault == "command" else 0)

    monkeypatch.setattr(verifier.subprocess, "run", run)
    if fault:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            verifier.run_bounded(tmp_path, packages, output)
    else:
        assert verifier.run_bounded(tmp_path, packages, output)["passed"] is True


def test_unary_batches_run_in_the_existing_three_platform_proof():
    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["portable-host"]
    command = next(
        step["run"]
        for step in job["steps"]
        if "--binary-batches" in step.get("run", "")
    )
    assert "--unary-batches" in command and "--timeout-seconds 4000" in command
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "opengl",
        "directx",
    }
    contract = next(
        step for step in job["steps"] if "test_portable_host.py" in step.get("run", "")
    )
    assert "test_unary_batches.py" in contract["run"]
    assert contract["if"] == "runner.os == 'Linux'"
