"""Row reduction metadata retains upstream geometry and checked source spans."""

import copy
import ctypes
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    reduction_packages,
    row_reduction_layout,
    row_workloads,
    runtime,
)
from tests.ci_helpers import assert_paths_covered


def _buffers(simple=False, dimension=1):
    if simple:
        data = {
            "in": ("float32", [1.0] * (33 * 65)),
            "out": ("float32", [0.0] * 33),
            "reduction_size": ("uint64", [65]),
            "out_size": ("int64", [33]),
        }
        entry = "row_reduce_simple_sumfloat32"
        logical_size, groups = 33 * 65, 9
    else:
        shape = [3] * dimension
        reductions = [2] * dimension
        stride = 65
        strides, reduce_strides = [], []
        for _ in range(dimension):
            strides.insert(0, stride)
            reduce_strides.insert(0, stride * 3)
            stride *= 6
        data = {
            "in": ("float32", [1.0] * stride),
            "out": ("float32", [0.0] * 3**dimension),
            "row_size": ("int64", [65]),
            "non_row_reductions": ("int64", [2**dimension]),
            "shape": ("int32", shape),
            "strides": ("int64", strides),
            "ndim": ("int32", [dimension]),
            "reduce_shape": ("int32", reductions),
            "reduce_strides": ("int64", reduce_strides),
            "reduce_ndim": ("int32", [dimension]),
        }
        entry = (
            f"row_reduce_looped_{dimension if dimension <= 2 else 5}_reduce_sumfloat32"
        )
        logical_size, groups = stride, 3**dimension
    memory = {
        name: (runtime.TYPES[dtype] * len(values))(*values)
        for name, (dtype, values) in data.items()
    }
    buffers = {
        name: runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            int(name == "out"),
        )
        for name, (dtype, values) in data.items()
    }
    return (
        entry,
        buffers,
        logical_size,
        {"workgroupCount": [1, groups, 1], "workgroupSize": [32, 1, 1]},
        memory,
    )


@pytest.mark.parametrize(
    "size,expected",
    [
        (65, 32),
        (512, 32),
        (513, 128),
        (1024, 128),
        (1025, 288),
        (4096, 1024),
        (65535, 1024),
    ],
)
def test_row_width_matches_upstream_thresholds(size, expected):
    assert row_reduction_layout.width(size) == expected


@pytest.mark.parametrize("size", [0, 64, 65536, True, "65", 65.0])
def test_invalid_row_width_is_rejected(size):
    with pytest.raises(ValueError):
        row_reduction_layout.width(size)


def test_row_packages_cover_every_upstream_width():
    assert set(reduction_packages.ROW_WIDTHS) == {
        row_reduction_layout.width(size) for size in range(65, 65536)
    }
    assert len(reduction_packages.ROW_ENTRIES) == 4 * len(reduction_packages.ENTRIES)


@pytest.mark.parametrize(
    "simple,dimension", [(True, 1), (False, 1), (False, 2), (False, 3)]
)
def test_valid_row_metadata(simple, dimension):
    entry, buffers, count, execution, memory = _buffers(simple, dimension)
    assert row_reduction_layout.validate(entry, buffers, count, execution)


@pytest.mark.parametrize(
    "fault",
    [
        "span",
        "dtype",
        "rank",
        "array-size",
        "logical-size",
        "rows",
        "stride",
        "reduction-count",
        "template",
        "width",
        "grid",
        "null",
        "direction-name",
    ],
)
def test_invalid_looped_row_metadata_is_rejected_before_dispatch(fault):
    entry, buffers, count, execution, memory = _buffers()
    if fault == "span":
        buffers["in"].count -= 1
    elif fault == "dtype":
        buffers["row_size"].dtype = b"uint32"
    elif fault == "rank":
        memory["ndim"][0] = 65
    elif fault == "array-size":
        buffers["strides"].count = 2
    elif fault == "logical-size":
        count -= 1
    elif fault == "rows":
        buffers["out"].count += 1
    elif fault == "stride":
        memory["strides"][0] = -65
    elif fault == "reduction-count":
        memory["non_row_reductions"][0] += 1
    elif fault == "template":
        entry = entry.replace("looped_1", "looped_2")
    elif fault == "width":
        execution["workgroupSize"][0] = 128
    elif fault == "grid":
        execution["workgroupCount"][0] = 2
    elif fault == "null":
        buffers["shape"].data = None
    elif fault == "direction-name":
        buffers["missing"] = buffers.pop("in")
    with pytest.raises(ValueError):
        row_reduction_layout.validate(entry, buffers, count, execution)


@pytest.mark.parametrize("fault", ["size", "count", "span", "width", "grid"])
def test_simple_row_preconditions_are_required(fault):
    entry, buffers, count, execution, memory = _buffers(True)
    if fault == "size":
        memory["out_size"][0] -= 1
    elif fault == "count":
        count -= 1
    elif fault == "span":
        buffers["in"].count -= 1
    elif fault == "width":
        execution["workgroupSize"][0] = 128
    elif fault == "grid":
        execution["workgroupCount"][1] = 8
    with pytest.raises(ValueError):
        row_reduction_layout.validate(entry, buffers, count, execution)


def test_rank_mismatch_is_rejected_before_metadata_array_read(monkeypatch):
    entry, buffers, count, execution, memory = _buffers()
    buffers["shape"].count = 2
    original = ctypes.cast

    def checked(address, pointer):
        assert address != buffers["shape"].data
        return original(address, pointer)

    monkeypatch.setattr(row_reduction_layout.ctypes, "cast", checked)
    with pytest.raises(ValueError, match="rank does not match"):
        row_reduction_layout.validate(entry, buffers, count, execution)


@pytest.fixture(scope="module")
def row_evidence():
    import numpy as np

    records, trace = [], []
    for ordinal, case in enumerate(row_workloads.cases((32, 128))):
        _, logical, expected = row_workloads.reference(np, case)
        record = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in case.items()
        }
        record.update(
            actual=row_workloads.wire_values(expected.tolist()),
            expected=row_workloads.wire_values(expected.tolist()),
            resultShape=list(expected.shape),
            logicalShape=list(logical.shape),
            resultDtype="mlx.core." + case["dtype"].removesuffix("_"),
            dispatchStart=ordinal,
            dispatchEnd=ordinal + 1,
        )
        guards = (
            [index % 2 == 0 for index in range(32)]
            if case["dtype"] == "bool_"
            else [
                (
                    struct.unpack("=f", struct.pack("=I", 0x6A15BEEF))[0]
                    if case["dtype"] == "float32"
                    else 0x6A15BEEF
                )
            ]
            * 32
        )
        trace.append(
            dict(
                entry=case["entry"],
                threads=logical.size,
                dispatchVersion=3,
                workgroupSize=[case["width"], 1, 1],
                workgroupCount=[
                    1,
                    (
                        (expected.size + 3) // 4
                        if "simple" in case["entry"]
                        else expected.size
                    ),
                    1,
                ],
                reductionValues=row_workloads.wire_values(
                    expected.reshape(-1).tolist()
                ),
                reductionGuardValues=guards,
            )
        )
        records.append(record)
    return records, trace


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing-case",
        "extra-case",
        "identity",
        "expected",
        "actual",
        "shape",
        "dtype",
        "missing-dispatch",
        "extra-dispatch",
        "duplicate-interval",
        "float-interval",
        "entry",
        "width",
        "float-width",
        "grid",
        "threads",
        "float-threads",
        "version",
        "float-version",
        "readback",
        "guard",
        "guard-count",
        "nan",
        "negative-zero",
    ],
)
def test_row_evidence_requires_complete_native_execution(row_evidence, fault):
    records, trace = copy.deepcopy(row_evidence)
    if fault == "missing-case":
        records.pop()
    elif fault == "extra-case":
        records.append(records[0])
    elif fault == "identity":
        records[0]["id"] = "other"
    elif fault in {"actual", "expected"}:
        records[0][fault][0] += 1
    elif fault == "shape":
        records[0]["resultShape"] = []
    elif fault == "dtype":
        records[0]["resultDtype"] = "mlx.core.int32"
    elif fault == "missing-dispatch":
        trace.pop()
    elif fault == "extra-dispatch":
        trace.append(trace[0])
    elif fault == "duplicate-interval":
        records[1].update(dispatchStart=0, dispatchEnd=1)
    elif fault == "float-interval":
        records[0]["dispatchStart"] = 0.0
    elif fault == "entry":
        trace[0]["entry"] = "other"
    elif fault == "width":
        trace[0]["workgroupSize"][0] = 128
    elif fault == "float-width":
        trace[0]["workgroupSize"][0] = 32.0
    elif fault == "grid":
        trace[0]["workgroupCount"][1] += 1
    elif fault == "threads":
        trace[0]["threads"] += 1
    elif fault == "float-threads":
        trace[0]["threads"] = float(trace[0]["threads"])
    elif fault == "version":
        trace[0]["dispatchVersion"] = 1
    elif fault == "float-version":
        trace[0]["dispatchVersion"] = 2.0
    elif fault == "readback":
        trace[0]["reductionValues"][0] += 1
    elif fault == "guard":
        trace[0]["reductionGuardValues"][0] = 0
    elif fault == "guard-count":
        trace[0]["reductionGuardValues"].pop()
    elif fault in {"nan", "negative-zero"}:
        record = next(
            record
            for record in records
            if record.get("profile")
            == ("early-nan" if fault == "nan" else "negative-zero")
            and record["entry"] == "row_reduce_simple_prodfloat32"
        )
        record["actual"][0] = 0.0
    if fault:
        with pytest.raises(ValueError):
            row_workloads.validate(records, (32, 128), trace=trace)
    else:
        row_workloads.validate(records, (32, 128), trace=trace)


def test_row_special_profiles_cover_every_float_entry():
    cases = list(row_workloads.cases((32, 128)))
    special = [case for case in cases if "profile" in case]
    assert len(special) == 64
    assert {(case["entry"], case["profile"]) for case in special} == {
        (entry, profile)
        for entry, dtype in reduction_packages.ROW_ENTRIES.items()
        if dtype == "float32"
        for profile in ("early-nan", "late-nan", "positive-infinity", "negative-zero")
    }


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("fault", [False, True])
def test_row_trace_checks_physical_boolean_storage(row_evidence, target, fault):
    records, trace = copy.deepcopy(row_evidence)
    for record, dispatch in zip(records, trace):
        dispatch["target"] = target
        if record["dtype"] == "bool_":
            if target != "metal":
                dispatch["reductionValues"] = [
                    int(value) for value in dispatch["reductionValues"]
                ]
            if fault:
                dispatch["reductionValues"][0] = 2
    if fault:
        with pytest.raises(ValueError):
            row_workloads.validate(records, (32, 128), trace=trace)
    else:
        row_workloads.validate(records, (32, 128), trace=trace)


def test_row_ci_preserves_all_three_targets_and_full_width_sets():
    import yaml

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    dependencies = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Install CrossTL and MLX build dependencies"
    )
    assert '"PyYAML>=6,<7"' in dependencies
    requirements = (
        Path(__file__).resolve().parents[5] / "requirements.txt"
    ).read_text()
    assert "PyYAML>=6,<7" in requirements.splitlines()
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "demos/integrations/mlx/tests/host/test_portable_rows.py",
        )
    job = yaml.safe_load(workflow)["jobs"]["reductions"]
    assert {
        (item["target"], item["os"], item["family"], item["verifier"])
        for item in job["strategy"]["matrix"]["include"]
    } == {
        (target, os, family, verifier)
        for target, os in (
            ("metal", "macos-26"),
            ("opengl", "ubuntu-24.04"),
            ("directx", "windows-2025"),
        )
        for family, verifier in (
            ("all", "verify"),
            ("row", "verify_rows"),
            ("column", "verify_columns"),
        )
    }
    steps = {step.get("name"): step for step in job["steps"]}
    translation = steps["Translate reduction launch variants"]
    execution = steps["Execute MLX reductions"]
    assert "--family ${{ matrix.family }}" in translation["run"]
    assert "--width" not in translation["run"] and "--entry" not in translation["run"]
    assert "portable_host.${{ matrix.verifier }}" in execution["run"]
    assert translation["if"] == "matrix.family != 'row'"
    assert "if" not in execution
    for step in (translation, execution):
        assert not step.get("continue-on-error")


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize(
    "fault",
    [None, "count", "direction", "missing-variant", "null", "rank", "span", "guard"],
)
def test_row_host_dispatch_checks_metadata_before_native_execution(
    tmp_path, monkeypatch, target, simple, fault
):
    entry, supplied, logical, execution, memory = _buffers(simple)
    buffers = (runtime.Buffer * len(supplied))(*supplied.values())
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.target, host.descriptors = target, {}
    host.reduction_directories = {f"w32/{entry}": tmp_path}
    host.trace = tmp_path / "trace.jsonl"
    bindings = []
    for name, buffer in supplied.items():
        member = {"in": "in_", "out": "out_"}.get(name, name)
        if target == "directx":
            member = entry + "_" + member
        dtype = buffer.dtype.decode()
        bindings.append(
            {
                "name": name,
                "scalarLayout": {
                    "memberName": member,
                    "elementType": dtype,
                    "elementStrideBytes": ctypes.sizeof(runtime.TYPES[dtype]),
                },
            }
        )
    host.reduction_descriptors = {
        f"w32/{entry}": {
            "bindings": bindings,
            "artifact": {"hash": {"algorithm": "sha256", "value": "0" * 64}},
        }
    }
    calls = []

    def request(descriptor, directory, inputs, outputs, launch, **kwargs):
        assert directory == tmp_path / "package"
        assert launch == execution
        assert inputs["in"]["shape"] == [logical]
        return outputs

    def execute(outputs):
        calls.append(outputs)
        outputs["out"]["values"][: supplied["out"].count] = [65.0] * supplied[
            "out"
        ].count
        if fault == "guard":
            outputs["out"]["values"][-1] = 0
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", request)
    host.executor = SimpleNamespace(run=execute)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    if fault == "direction":
        buffers[0].output = 1
    elif fault == "missing-variant":
        host.reduction_descriptors = {}
    elif fault == "null":
        buffers[0].data = None
    elif fault == "span":
        buffers[0].count -= 1
    elif fault == "rank":
        if simple:
            memory["out_size"][0] -= 1
        else:
            memory["ndim"][0] = 65
    if fault:
        with pytest.raises(RuntimeError if fault == "guard" else ValueError):
            host.dispatch(
                entry,
                buffers,
                len(buffers) - int(fault == "count"),
                logical,
                launch=launch,
            )
        assert len(calls) == int(fault == "guard")
        assert list(memory["out"]) == [0.0] * supplied["out"].count
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), logical, launch=launch)
        assert list(memory["out"]) == [65.0] * supplied["out"].count
        assert json.loads(host.trace.read_text())[
            "reductionMetadata"
        ] == row_reduction_layout.validate(entry, supplied, logical, execution)


@pytest.mark.parametrize(
    "fault", [None, "target", "hash", "size", "content", "trace", "escape"]
)
def test_row_artifact_evidence_retains_package_identity(tmp_path, fault):
    import hashlib

    from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

    package = tmp_path / "package"
    package.mkdir()
    (package / "kernel.metal").write_bytes(b"kernel")
    artifact = {
        "packagePath": "kernel.metal",
        "hash": {"algorithm": "sha256", "value": hashlib.sha256(b"kernel").hexdigest()},
        "sizeBytes": 6,
    }
    index = {"target": "metal", "descriptors": {"w32/entry": {"artifact": artifact}}}
    trace = [
        {
            "entry": "entry",
            "target": "metal",
            "workgroupSize": [32, 1, 1],
            "artifact": copy.deepcopy(artifact),
        }
    ]
    if fault == "target":
        trace[0]["target"] = "opengl"
    elif fault == "hash":
        artifact["hash"]["value"] = "0" * 64
    elif fault == "size":
        artifact["sizeBytes"] = 5
    elif fault == "content":
        (package / "kernel.metal").write_bytes(b"changed")
    elif fault == "trace":
        trace[0].pop("artifact")
    elif fault == "escape":
        artifact["packagePath"] = "../kernel.metal"
    if fault in {"hash", "size", "escape"}:
        trace[0]["artifact"] = copy.deepcopy(artifact)
    if fault:
        with pytest.raises(ValueError):
            verify_artifacts(trace, tmp_path, index)
    else:
        verify_artifacts(trace, tmp_path, index)
