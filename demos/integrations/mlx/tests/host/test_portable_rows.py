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


def _wide_metadata(simple, rows, row_size, *, stride=None, reductions=2):
    """Detached row metadata over ``rows`` rows of ``row_size`` elements."""
    if simple:
        metadata = {"reduction_size": [row_size], "out_size": [rows]}
        span, logical = rows * row_size, rows * row_size
        entry, grid = "row_reduce_simple_sumfloat32", [1, (rows + 3) // 4, 1]
    else:
        stride = row_size if stride is None else stride
        reduce_stride = rows * stride
        # Each shape axis stays below 65,536, so more rows take two axes.
        shape, strides = (
            ([rows], [stride])
            if rows <= 65535
            else ([2, rows // 2], [rows // 2 * stride, stride])
        )
        metadata = {
            "row_size": [row_size],
            "non_row_reductions": [reductions],
            "shape": shape,
            "strides": strides,
            "ndim": [len(shape)],
            "reduce_shape": [reductions],
            "reduce_strides": [reduce_stride],
            "reduce_ndim": [1],
        }
        span = row_size + (rows - 1) * stride + (reductions - 1) * reduce_stride
        logical = rows * row_size * reductions
        entry, grid = "row_reduce_looped_1_reduce_sumfloat32", [1, rows, 1]
    execution = {
        "workgroupCount": grid,
        "workgroupSize": [row_reduction_layout.width(row_size), 1, 1],
    }
    return entry, metadata, logical, span, rows, execution


@pytest.mark.parametrize("simple", [True, False])
def test_row_plans_bind_more_than_65535_elements(simple):
    # 2,048-element rows over 81,920 and 163,840 elements, with a non-row
    # reduction stride of 81,920 for the looped plan.
    entry, metadata, logical, span, rows, execution = _wide_metadata(simple, 40, 2048)
    assert logical > 65535 and span > 65535
    assert row_reduction_layout.validate_metadata(
        entry, metadata, logical, span, rows, execution
    )
    assert row_reduction_layout.MAX_ELEMENTS == 2**28 - 1
    assert reduction_packages.ROW_MAX_ELEMENTS == row_reduction_layout.MAX_ELEMENTS


def test_sliced_row_views_bind_spans_beyond_their_logical_size():
    # Every other 2,048-element row: a 161,792-element span for 81,920 values.
    entry, metadata, logical, span, rows, execution = _wide_metadata(
        False, 40, 2048, stride=4096, reductions=1
    )
    assert span == 2048 + 39 * 4096 and logical == 40 * 2048
    assert row_reduction_layout.validate_metadata(
        entry, metadata, logical, span, rows, execution
    )


@pytest.mark.parametrize("simple", [True, False])
def test_row_launch_axes_are_bounded_by_65535_workgroups(simple):
    # 65,535 workgroups fit one launch axis; one more row is refused.
    rows = 4 * 65535 if simple else 65535
    entry, metadata, logical, span, rows, execution = _wide_metadata(
        simple, rows, 65, reductions=1
    )
    assert row_reduction_layout.validate_metadata(
        entry, metadata, logical, span, rows, execution
    )
    entry, metadata, logical, span, rows, execution = _wide_metadata(
        simple, rows + (4 if simple else 1), 65, reductions=1
    )
    assert execution["workgroupCount"][1] == 65536
    with pytest.raises(ValueError, match="launch exceeds 65535 workgroups"):
        row_reduction_layout.validate_metadata(
            entry, metadata, logical, span, rows, execution
        )


@pytest.mark.parametrize("fault", ["span", "logical-size", "simple-lanes", "stride"])
def test_row_spans_are_bounded_by_the_row_element_limit(fault):
    maximum = row_reduction_layout.MAX_ELEMENTS
    simple = fault == "simple-lanes"
    entry, metadata, logical, span, rows, execution = _wide_metadata(simple, 40, 2048)
    if fault == "span":
        span = maximum + 1
    elif fault == "logical-size":
        logical = maximum + 1
    elif fault == "simple-lanes":
        # The final workgroup's inactive lanes form indexes four values per
        # lane past the span, which static GLSL packages assert below MAX.
        entry, metadata, logical, span, rows, execution = _wide_metadata(
            True, 4096, 65535
        )
        assert span <= maximum < span + 4 * 1024
    else:
        metadata["strides"] = [maximum + 1]
    with pytest.raises(ValueError):
        row_reduction_layout.validate_metadata(
            entry, metadata, logical, span, rows, execution
        )


def test_static_glsl_row_packages_assert_the_row_element_limit():
    source = Path(reduction_packages.__file__).read_text()
    start = source.index('if family == "row" and target == "opengl":')
    body = source[start : source.index("config.write_text(", start)]
    assert "maximum = {ROW_MAX_ELEMENTS}" in body
    assert "131071" not in body


def test_backend_bounds_row_passes_by_elements_and_workgroups():
    source = Path(row_reduction_layout.__file__).with_name("backend.cpp").read_text()
    assert "constexpr uint64_t max_row_elements = max_column_elements;" in source
    start = source.index("void dispatch_row_reduce(")
    body = source[start : source.index("\n}\n", start)]
    for text in (
        "uint64_t(steps[axis]) > max_row_elements",
        "uint64_t(span) > max_row_elements",
        "(simple && uint64_t(span) + uint64_t(width) * 4 > max_row_elements)",
        "if (row_groups > 65535) {",
    ):
        assert text in body, text
    assert "span > 65535" not in body
    # The launch bound is checked before the output allocation and dispatch.
    assert body.index("if (row_groups > 65535) {") < body.index("out.set_data(")


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
    import tomllib

    import yaml

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    dependencies = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Install CrossTL and MLX build dependencies"
    )
    assert '"PyYAML>=6,<7"' in dependencies
    project = tomllib.loads(
        (Path(__file__).resolve().parents[5] / "pyproject.toml").read_text()
    )
    assert "PyYAML>=6,<7" in project["project"]["optional-dependencies"]["test"]
    for event in ("pull_request", "push"):
        assert_paths_covered(
            workflow_text.workflow_event_path_filters(workflow, event),
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
    [
        None,
        "count",
        "direction",
        "missing-variant",
        "null",
        "rank",
        "span",
        "element-size",
        "guard",
    ],
)
def test_row_host_dispatch_checks_metadata_before_native_execution(
    tmp_path, monkeypatch, target, simple, fault
):
    entry, supplied, logical, execution, memory = _buffers(simple)
    buffers = (runtime.Buffer * len(supplied))(*supplied.values())
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.retain_native_modules = False
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
                    "elementSizeBytes": ctypes.sizeof(runtime.TYPES[dtype]),
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
    elif fault == "element-size":
        bindings[0]["scalarLayout"]["elementSizeBytes"] += 1
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


def test_row_source_spans_accept_compacted_uploads_of_the_same_length():
    from demos.integrations.mlx.portable_host.reduction_cache import (
        check_source_span,
    )
    from demos.integrations.mlx.portable_host.trace_records import (
        compact_trace_values,
    )

    values = [float(i) for i in range(420)]
    full = {"dtype": "float32", "shape": [420], "values": values}
    compact = compact_trace_values(full)
    assert "compactedValues" in compact["values"]
    for packet in (full, compact):
        check_source_span(packet, "float32", 420)
    short = compact_trace_values({**full, "values": values[:-1]})
    digest = {**compact["values"]["compactedValues"], "sha256": "0"}
    for packet, dtype, size in (
        (full, "float32", 419),
        (full, "int32", 420),
        ({**full, "shape": [419]}, "float32", 420),
        (short, "float32", 420),
        ({**compact, "values": {"compactedValues": digest}}, "float32", 420),
        ({**compact, "values": {"length": 420}}, "float32", 420),
        ({**full, "values": None}, "float32", 420),
        (None, "float32", 420),
    ):
        with pytest.raises(ValueError, match="source span"):
            check_source_span(packet, dtype, size)
