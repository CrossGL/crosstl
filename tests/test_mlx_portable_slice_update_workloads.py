"""Slice-update parity evidence and unchanged upstream test contracts."""

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import slice_update_workloads as workloads
from demos.integrations.mlx.portable_host import verify_slice_updates as proof


def evidence(target="metal", *, native=True):
    records, trace = [], []
    for case in workloads.cases():
        base, value, _ = workloads.inputs(np, case)
        expected = workloads.reference(np, case)
        events = workloads.expected_trace(np, case, target) if native else []
        records.append(
            {
                **case,
                "inputPayloads": [
                    workloads.copies.payload(np, item) for item in (base, value)
                ],
                "inputUnchanged": True,
                "resultPayload": workloads.copies.payload(np, expected),
                "resultShape": list(expected.shape),
                "resultDtype": expected.dtype.name,
                "dispatchCount": len(events),
            }
        )
        trace.extend(events)
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("native", [True, False])
def test_complete_slice_update_evidence(target, native):
    records, trace = evidence(target, native=native)
    assert len(records) == 304
    assert {case["layout"] for case in records} == set(workloads.LAYOUTS)
    assert {case["dtype"] for case in records} == set(workloads.copies.DTYPES)
    workloads.validate(records, trace, native=native)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-case",
        "extra-case",
        "case-id",
        "input",
        "result",
        "input-mutated",
        "result-shape",
        "result-type",
        "count",
        "count-bool",
        "missing-event",
        "extra-event",
        "entry",
        "target",
        "launch",
        "metadata",
        "result-values",
        "storage-word",
        "storage-type",
        "nan-payload",
        "zero-sign",
        "guard",
        "copy-values",
        "copy-guard",
        "copy-stride",
        "wide-float",
        "bool-int",
    ],
)
def test_slice_updates_reject_corrupt_evidence(fault):
    records, trace = evidence()
    event = next(item for item in trace if item["entry"] == "slice_update_sumfloat32")
    if fault == "missing-case":
        records.pop()
    elif fault == "extra-case":
        records.append(records[-1])
    elif fault == "case-id":
        records[0]["operation"] = "max"
    elif fault == "input":
        records[0]["inputPayloads"][0] = ""
    elif fault == "result":
        records[0]["resultPayload"] = ""
    elif fault == "input-mutated":
        records[0]["inputUnchanged"] = False
    elif fault == "result-shape":
        records[0]["resultShape"] = [24]
    elif fault == "result-type":
        records[0]["resultDtype"] = "float16"
    elif fault in {"count", "count-bool"}:
        records[0]["dispatchCount"] = 0 if fault == "count" else True
    elif fault == "missing-event":
        trace.pop()
    elif fault == "extra-event":
        trace.append(trace[-1])
    elif fault == "entry":
        event["entry"] = "slice_update_maxfloat32"
    elif fault == "target":
        trace[0]["target"] = "cpu"
    elif fault == "launch":
        event["workgroupCount"] = [1, 1, 1]
    elif fault == "metadata":
        event["sliceUpdateMetadata"]["destinationOffset"] += 1
    elif fault == "result-values":
        event["sliceUpdateValues"][0] = 4.0
    elif fault == "storage-word":
        event["sliceUpdateStorageWords"][0] ^= 1
    elif fault == "storage-type":
        event["sliceUpdateStorageWords"][0] = float(event["sliceUpdateStorageWords"][0])
    elif fault in {"nan-payload", "zero-sign"}:
        event = next(
            item
            for item in trace
            if 0x7FC12345 in item.get("sliceUpdateStorageWords", [])
        )
        words = event["sliceUpdateStorageWords"]
        words[words.index(0x7FC12345 if fault == "nan-payload" else 0x80000000)] = (
            0x7FC00000 if fault == "nan-payload" else 0
        )
    elif fault == "guard":
        event["sliceUpdateGuardValues"] = []
    elif fault == "copy-values":
        trace[0]["copyValues"][0] ^= 1
    elif fault == "copy-guard":
        trace[0]["copyGuardWords"] = []
    elif fault == "copy-stride":
        trace[0]["copyMetadata"]["sourceStrides"][0] = -6
    elif fault == "wide-float":
        event = next(
            item for item in trace if item["entry"] == "slice_update_sumuint64"
        )
        event["sliceUpdateValues"][0] = float(event["sliceUpdateValues"][0])
    else:
        event = next(item for item in trace if item["entry"] == "slice_update_sumbool_")
        event["sliceUpdateValues"][0] = int(event["sliceUpdateValues"][0])
    with pytest.raises(ValueError, match="Slice update"):
        workloads.validate(records, trace, native=True)


@pytest.mark.parametrize("operation", [*workloads.METHODS, "replace"])
@pytest.mark.parametrize("layout", ["negative-bounds", "clipped-stop"])
def test_slice_bounds_keep_expected_elements_and_destination(operation, layout):
    case = {"dtype": "int64", "operation": operation, "layout": layout}
    base, update, selection = workloads.inputs(np, case)
    reference = base.copy()
    selected = reference[selection]
    if operation == "replace":
        selected[:] = update
    else:
        for row in range(selected.shape[0]):
            for column in range(selected.shape[1]):
                left, right = int(selected[row, column]), int(update[row, column])
                value = {
                    "sum": lambda: left + right,
                    "prod": lambda: left * right,
                    "min": lambda: min(left, right),
                    "max": lambda: max(left, right),
                }[operation]()
                selected[row, column] = value
    assert reference.tobytes() == workloads.reference(np, case).tobytes()
    trace = workloads.expected_trace(np, case, "metal")
    metadata = trace[-1][
        "copyMetadata" if operation == "replace" else "sliceUpdateMetadata"
    ]
    assert metadata["destinationOffset"] == (7 if layout == "negative-bounds" else 6)
    assert metadata["destinationStrides"] == (
        [6, 2] if layout == "negative-bounds" else [12, 2]
    )


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "test",
        "testsRun",
        "failures",
        "errors",
        "skips",
        "dispatchCount",
        "missing",
    ],
)
def test_unchanged_slice_update_test_is_required(native, fault):
    record = {
        "tests": [
            {
                "test": proof.UPSTREAM_TESTS[0],
                "testsRun": 1,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                "dispatchCount": 1 if native else 0,
            }
        ],
        "workloadDispatchCount": 2 if native else 0,
    }
    if fault == "missing":
        record["tests"] = []
    elif fault == "test":
        record["tests"][0]["test"] = "replacement"
    elif fault:
        record["tests"][0][fault] = (
            0 if fault == "testsRun" or (fault == "dispatchCount" and native) else 1
        )
    if fault:
        with pytest.raises(ValueError, match="Slice update"):
            proof.validate_upstream(record, native=native)
    else:
        proof.validate_upstream(record, native=native)


def test_slice_update_ci_requires_native_execution_and_evidence():
    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["slice-update-host"]
    assert job["needs"] == "integer64-host"
    assert "continue-on-error" not in job and "if" not in job
    assert {
        (item["target"], item["os"]) for item in job["strategy"]["matrix"]["include"]
    } == {
        ("metal", "macos-26"),
        ("opengl", "ubuntu-24.04"),
        ("directx", "windows-2025"),
    }
    for step in job["steps"]:
        if step.get("name") in {
            "Translate slice update kernels",
            "Execute slice update reductions",
        }:
            assert "if" not in step and "continue-on-error" not in step
    translation = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Translate slice update kernels"
    )
    execution = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Execute slice update reductions"
    )
    assert "--family slice-update" in translation
    assert "portable_host.verify_slice_updates" in execution
    assert (
        "--slice-updates .mlx-portable-slice-update/slice-update-packages" in execution
    )
    assert "--packages .mlx-portable-slice-update/base/base/packages" in execution
    assert "--integer64 .mlx-portable-slice-update/base/integer64-packages" in execution
    assert "--output-dir .mlx-portable-slice-update/slice-update-evidence" in execution
    deadlines = [
        int(value)
        for step in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert sorted(deadlines) == [900, 1800, 3000, 3600, 3600]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60 <= 360 * 60
    selector = next(
        step
        for step in job["steps"]
        if step.get("name") == "Select same-run integer64 packages"
    )
    assert "run_id: context.runId" in selector["with"]["script"]
    assert "mlx-portable-integer64-" in selector["with"]["script"]
    upload = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()" and upload["with"]["include-hidden-files"] is True
    assert upload["with"]["path"] == ".mlx-portable-slice-update"
