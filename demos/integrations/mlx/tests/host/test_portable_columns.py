"""Column metadata, native pass routing, and retained numerical evidence."""

import copy
import ctypes
import json
import struct
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    column_reduction_layout,
    column_workloads,
    reduction_packages,
    runtime,
)
from tests.ci_helpers import assert_paths_covered


def _buffers(two_pass=False, dimension=1, *, rows=None, columns=33):
    rows = (257 if two_pass else 33) if rows is None else rows
    reductions = [2] * (dimension - 1) + [rows]
    steps, extent = [], columns
    for size in reversed(reductions):
        steps.insert(0, extent)
        extent *= size
    data = {
        "in": ("float32", [1.0] * extent),
        "out": ("float32", [0.0] * (columns * (32 if two_pass else 1))),
        "reduction_size": ("uint64", [rows]),
        "reduction_stride": ("int64", [columns]),
        "shape": ("int32", [0]),
        "strides": ("int64", [0]),
        "ndim": ("int32", [0]),
        "reduce_shape": ("int32", reductions),
        "reduce_strides": ("int64", steps),
        "reduce_ndim": ("int32", [dimension]),
        "non_col_reductions": ("uint64", [2 ** (dimension - 1)]),
    }
    if two_pass:
        data["out_size"] = ("uint64", [1])
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
    entry = f"col_reduce_{'2pass' if two_pass else 'looped'}_{dimension if dimension <= 2 else 5}_32_32_reduce_sumfloat32"
    return (
        entry,
        buffers,
        extent,
        {
            "workgroupCount": [(columns + 31) // 32, 32 if two_pass else 1, 1],
            "workgroupSize": [256, 1, 1],
        },
        memory,
    )


def test_column_package_family_covers_both_modes_and_every_template():
    assert len(reduction_packages.COLUMN_ENTRIES) == 84
    assert reduction_packages.FAMILY_WIDTHS["column"] == (256,)
    assert {case["entry"] for case in column_workloads.cases()} == set(
        reduction_packages.COLUMN_ENTRIES
    )


@pytest.mark.parametrize(
    "two_pass,rows,columns",
    [(False, 31, 33), (True, 1024, 16), (False, 257, 33), (True, 256, 33)],
)
def test_column_plan_selection_is_not_replaced_by_a_supported_kernel(
    two_pass, rows, columns
):
    entry, buffers, count, execution, memory = _buffers(
        two_pass, rows=rows, columns=columns
    )
    with pytest.raises(ValueError, match="upstream plan"):
        column_reduction_layout.validate(entry, buffers, count, execution)


def test_two_pass_outer_count_is_not_the_full_output_size():
    entry, buffers, count, execution, memory = _buffers(True)
    memory["out_size"][0] = 33
    with pytest.raises(ValueError, match="upstream plan"):
        column_reduction_layout.validate(entry, buffers, count, execution)


@pytest.mark.parametrize("width", [32, 128, 256.0, True, 512])
def test_column_packages_reject_non_source_widths(tmp_path, width):
    with pytest.raises(ValueError, match="widths"):
        reduction_packages.build_packages(
            tmp_path, tmp_path / "out", "metal", family="column", widths=[width]
        )


def test_column_cases_cover_upstream_thresholds_and_repeated_tiles():
    workload = list(column_workloads.cases())
    for entry in reduction_packages.COLUMN_ENTRIES:
        if "_1_32_32_" not in entry:
            continue
        rows = {case["shape"][0] for case in workload if case["entry"] == entry}
        assert rows.issuperset({257, 1025, 1985} if "2pass" in entry else {32, 33, 256})


def test_column_ci_includes_metadata_tests_on_changes():
    from pathlib import Path

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "demos/integrations/mlx/tests/host/test_portable_columns.py",
        )


@pytest.mark.parametrize("mode", ["allocation", "small", "long", "limit"])
@pytest.mark.parametrize("fault", [None, "message", "dispatch", "accepted"])
def test_column_rejection_requires_the_expected_error_before_dispatch(mode, fault):
    import numpy as np

    from demos.integrations.mlx.portable_host.verify_columns import (
        NEGATIVE_CHECKS,
        reject_invalid_plan,
    )

    def evaluate(value):
        if fault != "accepted":
            raise ValueError(
                "unexpected" if fault == "message" else NEGATIVE_CHECKS[mode]
            )

    mx = SimpleNamespace(
        array=lambda values: values,
        as_strided=lambda values, *args: values,
        sum=lambda values, **kwargs: values,
        eval=evaluate,
    )
    host = SimpleNamespace(dispatch_count=int(fault == "dispatch"))
    if fault:
        with pytest.raises(RuntimeError):
            reject_invalid_plan(mx, np, mode, host)
    else:
        assert reject_invalid_plan(mx, np, mode, host) == {
            "rejected": True,
            "message": NEGATIVE_CHECKS[mode],
            "dispatchCount": 0,
        }


@pytest.mark.parametrize("two_pass", [False, True])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_valid_column_metadata(two_pass, dimension):
    entry, buffers, count, execution, memory = _buffers(two_pass, dimension)
    assert column_reduction_layout.validate(entry, buffers, count, execution)


@pytest.mark.parametrize("two_pass", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "null-dtype",
        "span",
        "rank",
        "array-size",
        "null",
        "direction",
        "size",
        "stride",
        "empty-shape",
        "non-columns",
        "last-stride",
        "logical-size",
        "width",
        "grid",
        "template",
        "out-count",
    ],
)
def test_invalid_column_metadata_is_rejected(two_pass, fault):
    entry, buffers, count, execution, memory = _buffers(two_pass)
    if fault == "dtype":
        buffers["reduction_size"].dtype = b"uint32"
    elif fault == "null-dtype":
        buffers["in"].dtype = None
    elif fault == "span":
        buffers["in"].count -= 1
    elif fault == "rank":
        memory["reduce_ndim"][0] = 65
    elif fault == "array-size":
        buffers["reduce_shape"].count = 2
    elif fault == "null":
        buffers["shape"].data = None
    elif fault == "direction":
        buffers["out"].output = 0
    elif fault == "size":
        memory["reduction_size"][0] -= 1
    elif fault == "stride":
        memory["reduction_stride"][0] = -1
    elif fault == "empty-shape":
        memory["shape"][0] = 1
    elif fault == "non-columns":
        memory["non_col_reductions"][0] = 2
    elif fault == "last-stride":
        memory["reduce_strides"][0] -= 1
    elif fault == "logical-size":
        count -= 1
    elif fault == "width":
        execution["workgroupSize"][0] = 128
    elif fault == "grid":
        execution["workgroupCount"][1] += 1
    elif fault == "template":
        entry = entry.replace("_1_32_", "_2_32_")
    elif fault == "out-count":
        buffers["out"].count -= 1
    with pytest.raises(ValueError):
        column_reduction_layout.validate(entry, buffers, count, execution)


def test_column_rank_checked_before_array_read(monkeypatch):
    entry, buffers, count, execution, memory = _buffers()
    buffers["reduce_shape"].count = 2
    original = ctypes.cast

    def checked(address, pointer):
        assert address != buffers["reduce_shape"].data
        return original(address, pointer)

    monkeypatch.setattr(column_reduction_layout.ctypes, "cast", checked)
    with pytest.raises(ValueError, match="rank does not match"):
        column_reduction_layout.validate(entry, buffers, count, execution)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("two_pass", [False, True])
@pytest.mark.parametrize("fault", [None, "span", "missing-variant", "guard"])
def test_column_runtime_checks_layout_and_guards(
    tmp_path, monkeypatch, target, two_pass, fault
):
    entry, supplied, count, execution, memory = _buffers(two_pass)
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.retain_native_modules = False
    host.target, host.descriptors = target, {}
    host.reduction_directories = {f"w256/{entry}": tmp_path}
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
    if target == "directx":
        bindings.append(
            {
                "name": "CrossGLDispatchInfo",
                "scalarLayout": {"memberName": "crossglNumWorkGroups"},
                "provenance": {
                    "kind": "generated-execution-input",
                    "executionInput": {
                        "kind": "dispatch-workgroup-count",
                        "coordinateSpace": "physical",
                        "dimensions": 3,
                        "memberName": "crossglNumWorkGroups",
                        "valueSource": "dispatch.workgroupCount",
                    },
                },
            }
        )
    host.reduction_descriptors = {
        f"w256/{entry}": {
            "bindings": bindings,
            "artifact": {"hash": {"algorithm": "sha256", "value": "0" * 64}},
        }
    }
    calls = []

    def request(descriptor, directory, inputs, outputs, launch, **kwargs):
        assert launch == execution and directory == tmp_path / "package"
        assert inputs["in"]["shape"] == [count]
        assert "CrossGLDispatchInfo" not in inputs
        assert "CrossGLDispatchInfo" not in outputs
        return outputs

    def execute(outputs):
        calls.append(outputs)
        outputs["out"]["values"][: supplied["out"].count] = [3.0] * supplied[
            "out"
        ].count
        if fault == "guard":
            outputs["out"]["values"][-1] = 0
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(runtime, "build_native_loader_dispatch_request", request)
    host.executor = SimpleNamespace(run=execute)
    if fault == "span":
        supplied["in"].count -= 1
    elif fault == "missing-variant":
        host.reduction_descriptors = {}
    buffers = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    if fault:
        with pytest.raises(RuntimeError if fault == "guard" else ValueError):
            host.dispatch(entry, buffers, len(buffers), count, launch=launch)
        assert len(calls) == int(fault == "guard")
        assert list(memory["out"]) == [0.0] * supplied["out"].count
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), count, launch=launch)
        assert list(memory["out"]) == [3.0] * supplied["out"].count
        assert json.loads(host.trace.read_text())[
            "reductionMetadata"
        ] == column_reduction_layout.validate(entry, supplied, count, execution)


@pytest.fixture(scope="module")
def column_evidence():
    import numpy as np

    records, trace = [], []
    for case in column_workloads.cases():
        _, logical, expected, partials = column_workloads.reference(np, case)
        start = len(trace)
        dtype = case["dtype"]
        passes = [expected] if partials is None else [partials, expected]
        for stage, values in enumerate(passes):
            stride = logical.shape[-1] if stage == 0 else expected.size
            guards = (
                [index % 2 == 0 for index in range(32)]
                if dtype == "bool_"
                else [
                    (
                        struct.unpack("=f", struct.pack("=I", 0x6A15BEEF))[0]
                        if dtype == "float32"
                        else 0x6A15BEEF
                    )
                ]
                * 32
            )
            trace.append(
                dict(
                    entry=(
                        case["entry"]
                        if stage == 0
                        else "col_reduce_looped_1_32_32_reduce_"
                        + case["operation"]
                        + dtype
                    ),
                    dispatchVersion=3,
                    threads=logical.size if stage == 0 else partials.size,
                    workgroupSize=[256, 1, 1],
                    workgroupCount=[
                        (stride + 31) // 32,
                        (expected.size // stride)
                        * (32 if stage == 0 and partials is not None else 1),
                        1,
                    ],
                    reductionValues=column_workloads.wire_values(
                        values.reshape(-1).tolist()
                    ),
                    reductionGuardValues=guards,
                )
            )
        records.append(
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in case.items()
            }
            | dict(
                actual=column_workloads.wire_values(expected.tolist()),
                expected=column_workloads.wire_values(expected.tolist()),
                resultShape=list(expected.shape),
                logicalShape=list(logical.shape),
                resultDtype="mlx.core." + dtype.removesuffix("_"),
                dispatchStart=start,
                dispatchEnd=len(trace),
            )
        )
    return records, trace


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing-case",
        "actual",
        "reference",
        "interval",
        "entry",
        "width",
        "float-width",
        "partial",
        "second-pass",
        "guard",
        "extra-pass",
    ],
)
def test_column_evidence_checks_every_native_pass(column_evidence, fault):
    records, trace = copy.deepcopy(column_evidence)
    two_pass = next(record for record in records if "2pass" in record["entry"])
    first = trace[two_pass["dispatchStart"]]
    if fault == "missing-case":
        records.pop()
    elif fault == "actual":
        records[0]["actual"][0] += 1
    elif fault == "reference":
        records[0]["expected"][0] += 1
    elif fault == "interval":
        two_pass["dispatchEnd"] -= 1
    elif fault == "entry":
        first["entry"] = "wrong"
    elif fault == "width":
        first["workgroupSize"][0] = 128
    elif fault == "float-width":
        first["workgroupSize"][0] = 256.0
    elif fault == "partial":
        first["reductionValues"][0] += 1
    elif fault == "second-pass":
        trace[two_pass["dispatchStart"] + 1]["reductionValues"][0] += 1
    elif fault == "guard":
        first["reductionGuardValues"][-1] = 0
    elif fault == "extra-pass":
        trace.append(trace[-1])
    if fault:
        with pytest.raises(ValueError):
            column_workloads.validate(records, trace=trace)
    else:
        column_workloads.validate(records, trace=trace)
