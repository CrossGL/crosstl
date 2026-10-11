"""Whole-array MLX reductions preserve explicit source plans and package widths."""

import ctypes
import json
import math
import re
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    prepare,
    reduction_layout,
    reduction_packages,
    reduction_workloads,
    runtime,
)
from tests.ci_helpers import assert_paths_covered


@pytest.mark.parametrize(
    "count", [1, 7, 127, 128, 129, 257, 512, 513, 4095, 4096, 4097, 8191, 65535]
)
def test_all_reduce_stage_preserves_upstream_geometry(count):
    rows = 1 if count <= 4096 else 128
    row_size = math.ceil(count / rows)
    width = min(1024, math.ceil(math.ceil(row_size / 4) / 32) * 32)
    assert reduction_layout.stage(count) == {
        "rowSize": row_size,
        "workgroupCount": [1, rows, 1],
        "workgroupSize": [width, 1, 1],
    }
    if rows > 1:
        assert reduction_layout.stage(rows)["workgroupSize"] == [32, 1, 1]


@pytest.mark.parametrize("count", [0, -1, 2**31, True, 32.0, "32"])
def test_invalid_all_reduce_stage_is_rejected(count):
    with pytest.raises(ValueError, match="stored elements"):
        reduction_layout.stage(count)


@pytest.mark.parametrize(
    "dtype,itemsize",
    [("float32", 4), ("int32", 4), ("uint32", 4), ("bool_", 1), ("bfloat16", 2)],
)
@pytest.mark.parametrize(
    "boundary", [65536, 131075, "before", "at", "after", 2**31 - 1]
)
def test_large_reduction_stage_uses_logical_bytes(dtype, itemsize, boundary):
    count = {
        "before": 2**26 // itemsize - 1,
        "at": 2**26 // itemsize,
        "after": 2**26 // itemsize + 1,
    }.get(boundary, boundary)
    rows = 128 if count * itemsize <= 2**26 else 4096
    row_size = (count + rows - 1) // rows
    width = ((min((row_size + 3) // 4, 1024) + 31) // 32) * 32
    assert reduction_layout.stage(count, dtype) == {
        "rowSize": row_size,
        "workgroupCount": [1, rows, 1],
        "workgroupSize": [width, 1, 1],
    }
    final = reduction_layout.stage(rows, dtype)
    assert final["workgroupCount"] == [1, 1, 1]
    assert final["workgroupSize"] == [32 if rows == 128 else 1024, 1, 1]


@pytest.mark.parametrize("dtype", [None, [], "bool", "float16", "uint64", 4])
def test_reduction_plan_rejects_unsupported_logical_dtype(dtype):
    with pytest.raises(ValueError, match="dtype"):
        reduction_layout.stage(65536, dtype)


@pytest.mark.parametrize("dtype", ["float32", "bool_"])
@pytest.mark.parametrize("fault", [None, "rows", "width", "row-size"])
def test_large_reduction_metadata_is_validated_without_reading_input(dtype, fault):
    count = 2**26 // reduction_layout.ITEM_SIZES[dtype] + 1
    plan = reduction_layout.stage(count, dtype)
    size, row = ctypes.c_uint64(count), ctypes.c_uint64(plan["rowSize"])
    buffers = {
        "in_size": runtime.Buffer(b"in_size", b"uint64", ctypes.addressof(size), 1, 0),
        "row_size": runtime.Buffer(b"row_size", b"uint64", ctypes.addressof(row), 1, 0),
    }
    execution = {key: plan[key][:] for key in ("workgroupCount", "workgroupSize")}
    if fault == "rows":
        execution["workgroupCount"][1] = 128
    elif fault == "width":
        execution["workgroupSize"][0] = 32
    elif fault == "row-size":
        row.value += 1
    if fault:
        with pytest.raises(ValueError, match="upstream plan"):
            reduction_layout.validate(buffers, count, execution, dtype)
    else:
        assert reduction_layout.validate(buffers, count, execution, dtype) == plan


@pytest.mark.parametrize("fault", [None, "row", "size", "width", "rows", "axis"])
def test_reduction_launch_and_size_metadata_are_checked(fault):
    count = 4097
    size, row = ctypes.c_uint64(count), ctypes.c_uint64(33)
    buffers = {
        "in_size": runtime.Buffer(b"in_size", b"uint64", ctypes.addressof(size), 1, 0),
        "row_size": runtime.Buffer(b"row_size", b"uint64", ctypes.addressof(row), 1, 0),
    }
    execution = {"workgroupCount": [1, 128, 1], "workgroupSize": [32, 1, 1]}
    if fault == "row":
        row.value += 1
    elif fault == "size":
        size.value += 1
    elif fault == "width":
        execution["workgroupSize"][0] = 128
    elif fault == "rows":
        execution["workgroupCount"][1] = 127
    elif fault == "axis":
        execution["workgroupCount"] = [128, 1, 1]
    if fault:
        with pytest.raises(ValueError, match="upstream plan"):
            reduction_layout.validate(buffers, count, execution)
    else:
        assert reduction_layout.validate(buffers, count, execution)["rowSize"] == 33


@pytest.mark.parametrize(
    "widths", [[], [1], [31], [33], [1056], [32, 32], [32.0], [True]]
)
def test_package_builder_rejects_invalid_widths_before_checkout(tmp_path, widths):
    with pytest.raises(ValueError, match="widths"):
        reduction_packages.build_packages(
            tmp_path, tmp_path / "out", "opengl", widths=widths
        )


@pytest.mark.parametrize("entries", [[], ["missing"], ["all_reduce_sumfloat32"] * 2])
def test_package_builder_rejects_ambiguous_entries(tmp_path, entries):
    with pytest.raises(ValueError, match="entries"):
        reduction_packages.build_packages(
            tmp_path, tmp_path / "out", "opengl", entries=entries
        )


def test_package_widths_cover_every_small_array_plan():
    required = {
        reduction_layout.stage(count)["workgroupSize"][0] for count in range(1, 4097)
    }
    assert required == set(reduction_packages.WIDTHS)


@pytest.fixture(scope="module", params=["metal", "opengl", "directx"])
def packages(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"reductions-{request.param}")
    source = root / reduction_packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("""template<typename T> kernel void reduce_values(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant ulong& in_size [[buffer(2)]], constant ulong& row_size [[buffer(3)]],
uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  T value = in[tid];
  T result = simd_sum(value);
  if (tid == 0) out[group.y] = result + T(in_size - row_size);
}
template [[host_name("all_reduce_sumfloat32")]] [[kernel]]
decltype(reduce_values<float>) reduce_values<float>;
""")
    output = root / "packages"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            reduction_packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = reduction_packages.build_packages(
            root,
            output,
            request.param,
            widths=(32, 64, 128),
            entries=("all_reduce_sumfloat32",),
        )
    assert reduction_packages.load_index(output, request.param) == index
    assert (
        "subgroup_width_rules" not in (output / "translation/crosstl.toml").read_text()
    )
    return output, index


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "launch",
        "variant",
        "count",
        "input-count",
        "output-count",
        "dtype",
        "metadata-type",
        "metadata-count",
        "size",
        "row",
        "rows",
        "direction",
        "duplicate",
        "null",
        "guard",
        "readback-count",
    ],
)
@pytest.mark.parametrize("count", [129, 65536])
def test_reduction_dispatch_contract(packages, tmp_path, monkeypatch, fault, count):
    directory, index = packages
    entry = "all_reduce_sumfloat32"
    host = runtime.HostRuntime.__new__(runtime.HostRuntime)
    host.retain_native_modules = False
    host.target = index["target"]
    host.descriptors = {}
    host.reduction_directories = {key: directory for key in index["descriptors"]}
    host.reduction_descriptors = index["descriptors"].copy()
    host.trace = tmp_path / "trace.jsonl"
    plan = reduction_layout.stage(count)
    rows = plan["workgroupCount"][1]
    width = plan["workgroupSize"][0]
    values = [8256.0] * rows
    memory = [
        (ctypes.c_float * count)(*range(count)),
        (ctypes.c_float * rows)(),
        ctypes.c_uint64(count),
        ctypes.c_uint64(plan["rowSize"]),
    ]
    buffers = (runtime.Buffer * 4)(
        *[
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(data),
                count,
                int(name == "out"),
            )
            for name, dtype, count, data in zip(
                ("in", "out", "in_size", "row_size"),
                ("float32", "float32", "uint64", "uint64"),
                (count, rows, 1, 1),
                memory,
            )
        ]
    )
    launch = runtime.Launch((1, rows, 1), (width, 1, 1))
    if fault == "variant":
        host.reduction_descriptors = {}
    elif fault == "input-count":
        buffers[0].count -= 1
    elif fault == "output-count":
        buffers[1].count = 2
    elif fault in {"dtype", "metadata-type"}:
        buffers[0 if fault == "dtype" else 2].dtype = b"uint32"
    elif fault == "metadata-count":
        buffers[2].count = 2
    elif fault in {"size", "row"}:
        memory[2 if fault == "size" else 3].value -= 1
    elif fault == "rows":
        launch.workgroup_count[0] = 2
    elif fault == "direction":
        buffers[0].output = 1
    elif fault == "duplicate":
        buffers[1].name = b"in"
    elif fault == "null":
        buffers[0].data = None
    guard = [
        ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
        for word in runtime.COPY_GUARD
    ]
    descriptor = index["descriptors"][f"w{width}/{entry}"]
    name = next(
        binding["name"]
        for binding in descriptor["bindings"]
        if binding["access"] == "read_write"
    )
    calls = []

    def execute(request):
        calls.append(request)
        return SimpleNamespace(
            status="ok",
            outputs={
                name: {
                    "dtype": "float32",
                    "shape": [rows + 32],
                    "values": (
                        values
                        + (
                            []
                            if fault == "readback-count"
                            else [0.0] * 32 if fault == "guard" else guard
                        )
                    ),
                }
            },
            details={},
        )

    host.executor = SimpleNamespace(run=execute)
    if fault in {"guard", "readback-count"}:
        with pytest.raises(RuntimeError, match="guard|readback size"):
            host.dispatch(entry, buffers, 4, count, launch=launch)
        assert list(memory[1]) == [0.0] * rows and not host.trace.exists()
    elif fault:
        with pytest.raises(ValueError):
            host.dispatch(
                entry,
                buffers,
                3 if fault == "count" else 4,
                count,
                launch=None if fault == "launch" else launch,
            )
        assert not calls and list(memory[1]) == [0.0] * rows
    else:
        host.dispatch(entry, buffers, 4, count, launch=launch)
        assert list(memory[1]) == values
        assert calls[0].execution_plan.dispatch.workgroup_size == (width, 1, 1)
        assert calls[0].execution_plan.dispatch.workgroup_count == (1, rows, 1)
        assert not calls[0].execution_plan.diagnostics
        assert json.loads(host.trace.read_text())["reductionGuardValues"] == guard


@pytest.mark.parametrize(
    "fault",
    [
        "target",
        "float-width",
        "duplicate-width",
        "empty-widths",
        "entry",
        "duplicate-entry",
        "missing",
        "extra",
        "descriptor-target",
    ],
)
def test_reduction_index_rejects_ambiguous_variants(packages, tmp_path, fault):
    _, index = packages
    index = json.loads(json.dumps(index))
    if fault == "target":
        index["target"] = "other"
    elif fault == "float-width":
        index["widths"][0] = 32.0
    elif fault == "duplicate-width":
        index["widths"] = [32, 32]
    elif fault == "empty-widths":
        index["widths"] = []
    elif fault == "entry":
        index["entries"] = ["unknown"]
    elif fault == "duplicate-entry":
        index["entries"] *= 2
    elif fault == "missing":
        index["descriptors"].pop(next(iter(index["descriptors"])))
    elif fault == "extra":
        index["descriptors"]["w96/all_reduce_sumfloat32"] = {}
    else:
        next(iter(index["descriptors"].values()))["target"] = "other"
    (tmp_path / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        reduction_packages.load_index(tmp_path, packages[1]["target"])


def reduction_evidence():
    records, trace = [], []
    for case in reduction_workloads.cases((32, 64, 128)):
        start = len(trace)
        if case.get("layout") in {"reverse", "broadcast"}:
            trace.append(
                {
                    "entry": (
                        "ggn2_dynamic_copybool_bool_"
                        if case["dtype"] == "bool_"
                        else "ggn2_dynamic_copyuint32uint32"
                    )
                }
            )
        rows = reduction_layout.stage(case["count"], case["dtype"])["workgroupCount"][1]
        partials = case["inputs"]
        for count in [case["count"]] + ([rows] if rows > 1 else []):
            plan = reduction_layout.stage(count, case["dtype"])
            trace.append(
                {
                    "entry": case["entry"],
                    "threads": count,
                    "dispatchVersion": 3,
                    "workgroupCount": plan["workgroupCount"],
                    "workgroupSize": plan["workgroupSize"],
                    "reductionGuardValues": [],
                }
            )
            if case.get("offset"):
                partials = reduction_workloads.partial_reference(
                    partials, case["operation"], case["dtype"], plan
                )
                guard = (
                    list(runtime.BOOLEAN_GUARD)
                    if case["dtype"] == "bool_"
                    else (
                        [
                            ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
                            for word in runtime.COPY_GUARD
                        ]
                        if case["dtype"] == "float32"
                        else runtime.COPY_GUARD
                    )
                )
                trace[-1].update(
                    target="metal",
                    reductionValues=partials,
                    reductionGuardValues=guard,
                    reductionMetadata={"in_size": count, "row_size": plan["rowSize"]},
                )
        value = (
            float(case["expected"]) if case["dtype"] == "float32" else case["expected"]
        )
        dtype = "bool" if case["dtype"] == "bool_" else case["dtype"]
        records.append(
            {
                key: reduction_workloads.wire(value)
                for key, value in case.items()
                if key != "inputs"
            }
            | {
                "actual": reduction_workloads.wire(value),
                "resultShape": [],
                "resultDtype": f"mlx.core.{dtype}",
                "dispatchStart": start,
                "dispatchEnd": len(trace),
            }
        )
    return records, trace


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "extra",
        "identity",
        "expected",
        "actual",
        "shape",
        "dtype",
        "negative-zero",
        "nan",
        "missing-pass",
        "missing-copy",
        "width",
        "float-width",
        "metadata",
        "float-version",
        "guard",
        "interval",
        "trailing-dispatch",
        "large-partial",
        "large-guard",
        "large-metadata",
        "large-result-count",
    ],
)
def test_reduction_evidence_requires_complete_numerical_execution(fault):
    records, trace = reduction_evidence()
    if fault == "missing":
        records.pop()
    elif fault == "extra":
        records.append(records[0])
    elif fault == "identity":
        records[0]["id"] = "different"
    elif fault in {"expected", "actual"}:
        records[0][fault] += 1
    elif fault == "shape":
        records[0]["resultShape"] = [1]
    elif fault == "dtype":
        records[0]["resultDtype"] = "mlx.core.int32"
    elif fault == "negative-zero":
        next(
            record
            for record in records
            if record["id"] == "all_reduce_minfloat32-negative-zero"
        )["actual"] = 0.0
    elif fault == "nan":
        next(
            record for record in records if record["id"] == "all_reduce_minfloat32-nan"
        )["actual"] = 0.0
    elif fault == "missing-pass":
        trace.pop(0)
    elif fault == "missing-copy":
        next(record for record in trace if record["entry"].startswith("ggn2"))[
            "entry"
        ] = "all_reduce_sumfloat32"
    elif fault == "width":
        trace[0]["workgroupSize"][0] = 64
    elif fault == "float-width":
        trace[0]["workgroupSize"][0] = 32.0
    elif fault == "metadata":
        trace[0]["threads"] += 1
    elif fault == "float-version":
        trace[0]["dispatchVersion"] = 2.0
    elif fault == "guard":
        trace[0].pop("reductionGuardValues")
    elif fault == "interval":
        records[0]["dispatchStart"] = True
    elif fault == "trailing-dispatch":
        trace.append(trace[-1])
    elif fault and fault.startswith("large-"):
        dispatch = next(record for record in trace if record.get("threads", 0) > 65535)
        if fault == "large-partial":
            dispatch["reductionValues"][0] = 7.0
        elif fault == "large-guard":
            dispatch["reductionGuardValues"] = []
        elif fault == "large-metadata":
            dispatch["reductionMetadata"]["row_size"] += 1
        elif fault == "large-result-count":
            dispatch["reductionValues"].pop()
    if fault:
        with pytest.raises(RuntimeError):
            reduction_workloads.validate(records, (32, 64, 128), trace=trace)
    else:
        reduction_workloads.validate(records, (32, 64, 128), trace=trace)


def test_reduction_workloads_cover_every_declared_width_and_operation():
    cases = list(reduction_workloads.cases())
    assert len({case["id"] for case in cases}) == len(cases)
    assert {
        (case["entry"], reduction_layout.stage(case["count"])["workgroupSize"][0])
        for case in cases
    } == {
        (entry, width)
        for entry in reduction_packages.ENTRIES
        for width in reduction_packages.WIDTHS
    }


def test_reduction_host_ci_requires_every_variant_and_retains_failures():
    from pathlib import Path

    from tools import ci_coverage

    root = Path(__file__).resolve().parents[5]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "demos/integrations/mlx/tests/host/test_portable_reductions.py",
        )
    translation = ci_coverage.workflow_job_step_section(
        workflow, "reductions", "Translate reduction launch variants"
    )
    execution = ci_coverage.workflow_job_step_section(
        workflow, "reductions", "Execute MLX reductions"
    )
    for step in (translation, execution):
        assert "continue-on-error" not in step
        assert "set -euo pipefail" in step and "run_bounded_command.py" in step
        assert "tee .mlx-portable-reductions/" in step
    assert "--width" not in translation and "--entry" not in translation
    assert "--jobs 2" in translation and "--target ${{ matrix.target }}" in translation
    assert "--reductions .mlx-portable-reductions/packages" in execution
    assert ci_coverage.workflow_job_step_after(
        workflow,
        "reductions",
        "Execute MLX reductions",
        "Translate reduction launch variants",
    )
    job = ci_coverage.workflow_job_text(workflow, "reductions")
    assert "needs: [portable-host, row-packages]" in job
    assert "artifact-ids: ${{ steps.host-artifact.outputs.id }}" in job
    assert "run_id: context.runId" in job and "merge-multiple: true" in job
    assert "if: always()" in job and "include-hidden-files: true" in job
    import yaml

    config = yaml.safe_load(workflow)["jobs"]["reductions"]
    assert "if: matrix.family != 'row'" in translation and "if:" not in execution
    for case in config["strategy"]["matrix"]["include"]:
        steps = [
            step
            for step in config["steps"]
            if step.get("if")
            not in (
                (
                    "matrix.family != 'row'"
                    if case["family"] == "row"
                    else "matrix.family == 'row'"
                ),
            )
        ]
        deadline = sum(
            int(value)
            for step in steps
            for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
        )
        assert (
            deadline + case.get("verification_timeout", 5400) + 1800
            < config["timeout-minutes"] * 60
        )
