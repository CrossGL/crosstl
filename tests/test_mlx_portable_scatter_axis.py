"""Axis-scatter allocation validation, native writeback and evidence contracts."""

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    gather_packages,
    runtime,
    scatter_axis_evidence,
)
from demos.integrations.mlx.portable_host import scatter_axis_layout as layout
from demos.integrations.mlx.portable_host import scatter_axis_workloads as workloads
from demos.integrations.mlx.portable_host import verify_scatter_axis
from demos.integrations.mlx.portable_host.gather_workloads import words


def buffers(case):
    source, indices, updates, expected = workloads.reference(np, case)
    if any(step < 0 for step in indices.strides):
        indices = np.ascontiguousarray(indices)
    if any(step < 0 for step in updates.strides):
        updates = np.ascontiguousarray(updates)
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    update_steps = [step // updates.itemsize for step in updates.strides]
    index_steps = [step // indices.itemsize for step in indices.strides]
    metadata = {
        "shape": [size for i, size in enumerate(indices.shape) if i != axis] or [1],
        "upd_strides": (
            [step for i, step in enumerate(update_steps) if i != axis] or [0]
        ),
        "idx_strides": [step for i, step in enumerate(index_steps) if i != axis] or [0],
        "ndim": [source.ndim - 1],
        "axis": [axis],
        "out_axis_size": [source.shape[axis]],
        "upd_ax_stride": [update_steps[axis]],
        "idx_ax_stride": [index_steps[axis]],
    }
    arrays = {
        name: np.array(value, dtype=layout.METADATA[name])
        for name, value in metadata.items()
    }
    arrays.update(
        upd=updates,
        indices=indices,
        out=(
            source.copy(order="C")
            if case["operation"] == "none"
            else np.zeros_like(source, order="C")
        ),
    )
    supplied = {}
    for name, array in arrays.items():
        span = 1 + sum(
            (size - 1) * (step // array.itemsize)
            for size, step in zip(array.shape, array.strides)
        )
        supplied[name] = runtime.Buffer(
            name.encode(),
            str(array.dtype).encode(),
            array.ctypes.data,
            span,
            int(name == "out"),
        )
    entry = (
        f"scatter_axis{case['dtype']}{case['index_dtype']}_{case['operation']}_int"
        + ("c" if updates.flags.c_contiguous else "nc")
        + ("c" if indices.flags.c_contiguous else "nc")
    )
    execution = {
        "workgroupCount": [
            int(np.prod(indices.shape[axis + 1 :])),
            indices.shape[axis],
            int(np.prod(indices.shape[:axis])),
        ],
        "workgroupSize": [1, 1, 1],
    }
    return entry, supplied, arrays, expected, execution


@pytest.mark.parametrize(
    "case",
    [case for case in workloads.cases() if case["layout"] != "empty"],
    ids=lambda case: case["id"],
)
def test_axis_scatter_validates_update_and_output_geometry(case):
    entry, supplied, arrays, expected, execution = buffers(case)
    result = layout.validate(entry, supplied, expected.size, execution)
    assert result["outputShape"] == list(expected.shape)
    assert result["updateShape"] == list(arrays["upd"].shape)
    assert result["indexStrides"] == [
        step // arrays["indices"].itemsize for step in arrays["indices"].strides
    ]
    assert result["updateStrides"] == [
        step // arrays["upd"].itemsize for step in arrays["upd"].strides
    ]
    assert result["maximumIndex"] == expected.size - 1


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "dtype",
        "null",
        "direction",
        "scalar-length",
        "rank",
        "axis",
        "axis-size",
        "axis-step",
        "shape",
        "stride",
        "metadata-length",
        "update-span",
        "index-span",
        "index",
        "negative-index",
        "output-size",
        "overlap",
        "geometry",
        "group-size",
        "update-contiguous",
        "index-contiguous",
    ),
)
def test_axis_scatter_rejects_invalid_metadata_before_dispatch(fault):
    case = dict(next(workloads.cases()), layout="both-strided", axis=1)
    entry, supplied, arrays, expected, execution = buffers(case)
    if fault == "missing":
        supplied.pop("axis")
    elif fault == "dtype":
        supplied["indices"].dtype = b"uint32"
    elif fault == "null":
        supplied["upd"].data = None
    elif fault == "direction":
        supplied["upd"].output = 1
    elif fault == "scalar-length":
        supplied["axis"].count = 2
    elif fault == "rank":
        arrays["ndim"][0] = 64
    elif fault == "axis":
        arrays["axis"][0] = 3
    elif fault == "axis-size":
        arrays["out_axis_size"][0] = 0
    elif fault == "axis-step":
        arrays["upd_ax_stride"][0] = 65536
    elif fault == "shape":
        arrays["shape"][0] = 0
    elif fault == "stride":
        arrays["upd_strides"][0] = -1
    elif fault == "metadata-length":
        supplied["shape"].count += 1
    elif fault == "update-span":
        supplied["upd"].count -= 1
    elif fault == "index-span":
        supplied["indices"].count -= 1
    elif fault in {"index", "negative-index"}:
        arrays["indices"].flat[0] = 3 if fault == "index" else -4
    elif fault == "output-size":
        supplied["out"].count -= 1
    elif fault == "overlap":
        supplied["out"].data = supplied["upd"].data
    elif fault == "geometry":
        execution["workgroupCount"][0] += 1
    elif fault == "group-size":
        execution["workgroupSize"][0] = 2
    elif fault == "update-contiguous":
        entry = entry.removesuffix("ncnc") + "cnc"
    elif fault == "index-contiguous":
        entry = entry.removesuffix("ncnc") + "ncc"
    with pytest.raises(ValueError):
        layout.validate(entry, supplied, expected.size, execution)


@pytest.mark.parametrize(
    "entry",
    (
        "scatter_axisfloat32int32_none_intcc",
        "scatter_axisint64int32_none_intcc",
        "scatter_axisint32float32_none_intcc",
        "scatter_axisint32int32_prod_intcc",
        "scatter_axisint32int32_none_int64_tcc",
        "scatter_axisint32int32_none_intcc_extra",
    ),
)
def test_axis_scatter_rejects_unsupported_specializations(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


@pytest.mark.parametrize("operation", ("none", "sum"))
def test_axis_scatter_source_is_only_pinned_includes_and_instantiation(
    tmp_path, operation
):
    entry = f"scatter_axisuint32int64_{operation}_intncc"
    text = gather_packages.source(tmp_path, entry)
    assert '#include "mlx/backend/metal/kernels/indexing/scatter_axis.h"' in text
    op = "None" if operation == "none" else "Sum<uint>"
    assert f"decltype(scatter_axis<uint, int64_t, int, {op}, false, true>)" in text
    assert "{" not in text
    host = SimpleNamespace(descriptors={}, gathers=object())
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1
    host.gathers = None
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 0


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize(
    "fault",
    (
        None,
        "missing",
        "duplicate",
        "actual",
        "shape",
        "dtype",
        "source",
        "count",
        "start",
    ),
)
def test_axis_scatter_requires_complete_matching_workloads(native, fault):
    records = []
    for i, case in enumerate(workloads.cases()):
        *_, expected = workloads.reference(np, case)
        records.append(
            {
                **case,
                "actual": words(np, expected),
                "shape": list(expected.shape),
                "resultDtype": "mlx.core." + case["dtype"],
                "inputUnchanged": True,
                "dispatchStart": i if native else 0,
                "dispatchCount": int(native),
            }
        )
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records[-1] = records[0]
    elif fault == "actual":
        records[0]["actual"][0] ^= 1
    elif fault == "shape":
        records[0]["shape"] = [48]
    elif fault == "dtype":
        records[0]["resultDtype"] = "mlx.core.float32"
    elif fault == "source":
        records[0]["inputUnchanged"] = False
    elif fault == "count":
        records[0]["dispatchCount"] = not native
    elif fault == "start":
        records[1]["dispatchStart"] += 1
    if fault:
        with pytest.raises(ValueError):
            verify_scatter_axis.validate_records(np, records, native=native)
    else:
        verify_scatter_axis.validate_records(np, records, native=native)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault", (None, "guard", "shape", "dtype", "missing", "storage", "index")
)
def test_axis_scatter_uploads_initialized_output_and_guards_writeback(
    tmp_path, monkeypatch, target, fault
):
    entry, supplied, arrays, expected, execution = buffers(next(workloads.cases()))
    initial = arrays["out"].copy()
    descriptor = {"artifact": {}, "bindings": []}
    for name, buffer in supplied.items():
        scalar = {
            "memberName": name,
            "elementType": buffer.dtype.decode(),
            "elementStrideBytes": ctypes.sizeof(runtime.TYPES[buffer.dtype.decode()]),
        }
        if name == "out":
            scalar.update(
                componentCount=1,
                structMembers=[
                    {"name": "val", "offsetBytes": 0, "physicalType": "int"}
                ],
            )
        descriptor["bindings"].append({"name": name, "scalarLayout": scalar})
    if fault == "storage":
        descriptor["bindings"][-1]["scalarLayout"]["componentCount"] = 2
    if fault == "index":
        arrays["indices"].flat[0] = 4
    calls = []

    def package(name, maximum):
        calls.append((name, maximum))
        assert name == entry and maximum == expected.size - 1
        return descriptor, tmp_path

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        assert (
            inputs["out"]["values"] == initial.flatten().tolist() + runtime.COPY_GUARD
        )
        assert inputs["out"]["shape"] == [expected.size + 32, 1]
        return SimpleNamespace(inputs=inputs, outputs=outputs)

    def execute(_host, request):
        result = {
            **request.outputs["out"],
            "values": expected.flatten().tolist() + runtime.COPY_GUARD,
        }
        if fault == "guard":
            result["values"][-1] = 0
        elif fault == "shape":
            result["shape"] = [expected.size + 32]
        elif fault == "dtype":
            result["dtype"] = "uint32"
        return SimpleNamespace(
            status="ok",
            outputs={} if fault == "missing" else {"out": result},
            details={},
        )

    monkeypatch.setattr(
        gather_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(gather_dispatch, "execute", execute)
    host = SimpleNamespace(
        target=target,
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
        gathers=SimpleNamespace(get=package),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch(tuple(execution["workgroupCount"]), (1, 1, 1))
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, entry, table, len(table), expected.size, launch
            )
        assert np.array_equal(arrays["out"], initial)
        assert host.dispatch_count == 0 and not host.trace.exists()
        if fault == "index":
            assert not calls
    else:
        gather_dispatch.dispatch(host, entry, table, len(table), expected.size, launch)
        assert np.array_equal(arrays["out"], expected)
        assert host.dispatch_count == 1
        assert (
            json.loads(host.trace.read_text())["scatterValues"]
            == expected.flatten().tolist()
        )


def scatter_event(tmp_path, case, target="metal"):
    from tests.test_mlx_portable_gather import evidence_event

    event = evidence_event(tmp_path)
    for key in list(event):
        if key.startswith("gather"):
            del event[key]
    entry, supplied, arrays, expected, execution = buffers(case)
    event.update(entry=entry, target=target, threads=expected.size, **execution)
    event["scatterMetadata"] = layout.validate(
        entry, supplied, expected.size, execution
    )
    event["scatterValues"] = words(np, expected)
    event["scatterGuardValues"] = runtime.COPY_GUARD.copy()
    event["scatterStorageType"] = case["dtype"]
    event["outputHash"] = hashlib.sha256(expected.tobytes()).hexdigest()
    event["inputs"] = {}
    request = event["details"]["request"]
    request.update(
        entryPoint={"metal": entry, "directx": "CSMain", "opengl": "main"}[target],
        target=target,
        dispatch=execution,
        buffers={},
    )
    request["dispatch"] = {
        **execution,
        "entryPoint": request["entryPoint"],
        "globalSize": execution["workgroupCount"].copy(),
        "gridSize": execution["workgroupCount"].copy(),
        "metadata": {
            "source": "native-loader-dispatch",
            "workgroupSizeSource": "dispatch" if target == "metal" else "reflection",
        },
    }
    for name, buffer in supplied.items():
        dtype = buffer.dtype.decode()
        values = list(
            ctypes.cast(
                buffer.data, ctypes.POINTER(runtime.TYPES[dtype] * buffer.count)
            ).contents
        )
        if name == "out":
            values += runtime.COPY_GUARD
        value = {
            "dtype": dtype,
            "shape": [len(values), 1] if name == "out" else [len(values)],
            "values": values,
        }
        event["inputs"][name] = value
        scalar = {"elementType": dtype, "elementStrideBytes": np.dtype(dtype).itemsize}
        if name == "out":
            scalar.update(
                componentCount=1,
                structMembers=[
                    {
                        "name": "val",
                        "offsetBytes": 0,
                        "physicalType": "int" if dtype == "int32" else "uint",
                    }
                ],
            )
        request["buffers"][name] = {
            key: item for key, item in value.items() if key != "values"
        }
        request["buffers"][name]["binding"] = {"metadata": {"scalarLayout": scalar}}
    if target != "metal":
        module = tmp_path / ("test.dxil" if target == "directx" else "test.glsl")
        module.write_bytes(b"retained-test-module")
        event["details"]["module"] = {
            "file": str(module),
            "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        }
        event["details"]["validationModules"] = []
        event["details"]["adapterSteps"] = [
            {
                "action": (
                    "compile-hlsl-for-directx-runtime"
                    if target == "directx"
                    else "validate-glsl-for-opengl-runtime"
                ),
                "status": "passed",
            }
        ]
        event["details"]["artifactIdentityVerification"]["target"] = target
    return event


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "case",
    [case for case in workloads.cases() if case["layout"] != "empty"],
    ids=lambda case: case["id"],
)
def test_axis_scatter_audit_reconstructs_every_workload(tmp_path, target, case):
    event = scatter_event(tmp_path, case, target)
    initial, indices, updates, actual = scatter_axis_evidence.audit_event(np, event)
    source, expected_indices, expected_updates, expected = workloads.reference(np, case)
    assert np.array_equal(
        initial, source if case["operation"] == "none" else np.zeros_like(source)
    )
    assert np.array_equal(indices, expected_indices)
    assert np.array_equal(updates, expected_updates)
    assert actual == words(np, expected)


@pytest.mark.parametrize(
    "fault",
    (
        "result",
        "guard",
        "hash",
        "index",
        "update",
        "initial",
        "axis",
        "stride",
        "metadata",
        "geometry",
        "binding",
        "extra-binding",
        "missing-binding",
        "atomic-member",
        "atomic-stride",
        "encoding",
        "entry",
        "target",
        "source",
        "module",
        "compiler",
        "validation",
        "identity",
        "output-count",
        "global-grid",
        "grid-size",
        "dispatch-entry",
        "thread-grid",
    ),
)
def test_axis_scatter_audit_rejects_corrupt_native_evidence(tmp_path, fault):
    case = dict(next(workloads.cases()), operation="sum", layout="both-strided", axis=1)
    event = scatter_event(tmp_path, case)
    inputs, details = event["inputs"], event["details"]
    request = details["request"]
    if fault == "result":
        event["scatterValues"][0] ^= 1
    elif fault == "guard":
        event["scatterGuardValues"][-1] = 0
    elif fault == "hash":
        event["outputHash"] = "0" * 64
    elif fault == "index":
        inputs["indices"]["values"][0] = 3
    elif fault == "update":
        inputs["upd"]["values"][0] += 1
    elif fault == "initial":
        inputs["out"]["values"][0] += 1
    elif fault == "axis":
        inputs["axis"]["values"][0] = 2
    elif fault == "stride":
        inputs["upd_strides"]["values"][0] += 1
    elif fault == "metadata":
        event["scatterMetadata"]["axis"] = 2
    elif fault == "geometry":
        event["workgroupCount"][0] += 1
    elif fault == "binding":
        request["buffers"]["upd"]["dtype"] = "uint32"
    elif fault == "extra-binding":
        request["buffers"]["unexpected"] = request["buffers"]["upd"]
    elif fault == "missing-binding":
        del request["buffers"]["upd"]
    elif fault == "atomic-member":
        request["buffers"]["out"]["binding"]["metadata"]["scalarLayout"][
            "structMembers"
        ][0]["offsetBytes"] = 4
    elif fault == "atomic-stride":
        request["buffers"]["out"]["binding"]["metadata"]["scalarLayout"][
            "elementStrideBytes"
        ] = 8
    elif fault == "encoding":
        inputs["out"]["encoding"] = "ieee754-binary32"
    elif fault == "entry":
        request["entryPoint"] = "other"
    elif fault == "target":
        event["target"] = "unknown"
    elif fault == "source":
        (tmp_path / "source.metal").write_bytes(b"changed")
    elif fault == "module":
        (tmp_path / "test.metallib").write_bytes(b"changed")
    elif fault == "compiler":
        details["validationModules"] = []
    elif fault == "validation":
        details["adapterSteps"][0]["status"] = "skipped"
    elif fault == "identity":
        details["artifactIdentityVerification"]["verificationStatus"] = "unverified"
    elif fault == "output-count":
        event["threads"] = event["scatterMetadata"]["updateCount"]
    elif fault == "global-grid":
        request["dispatch"]["globalSize"][0] += 1
    elif fault == "grid-size":
        request["dispatch"]["gridSize"][0] += 1
    elif fault == "dispatch-entry":
        request["dispatch"]["entryPoint"] = "other"
    elif fault == "thread-grid":
        request["dispatch"]["threadGridSize"] = event["workgroupCount"]
    with pytest.raises(ValueError):
        scatter_axis_evidence.audit_event(np, event)


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "case",
        "initial",
        "extra",
        "missing",
        "boundary",
        "count",
        "empty-result",
        "empty-dispatch",
        "mixed-target",
    ),
)
def test_axis_scatter_requires_complete_workload_traces(tmp_path, monkeypatch, fault):
    from demos.integrations.mlx.portable_host import verify_bitwise

    cases = [
        case
        for case in workloads.cases()
        if case["dtype"] == "int32"
        and case["operation"] == "none"
        and case["layout"] in {"dense", "empty"}
    ]
    monkeypatch.setattr(workloads, "cases", lambda: iter(cases))
    event = scatter_event(tmp_path, cases[0])
    trace = [event, {"entry": "copy-empty-scatter-source", "target": "metal"}]
    audited = []
    monkeypatch.setattr(
        verify_bitwise,
        "verify_native_identity",
        lambda events, target: audited.append((events, target)),
    )
    records = []
    for i, case in enumerate(cases):
        *_, expected = workloads.reference(np, case)
        records.append(
            {
                **case,
                "actual": words(np, expected),
                "shape": list(expected.shape),
                "resultDtype": "mlx.core.int32",
                "inputUnchanged": True,
                "dispatchStart": i,
                "dispatchCount": 1,
            }
        )
    if fault == "case":
        records[0]["id"] = "other"
    elif fault == "initial":
        # Dense replacement overwrites every cell, so its final result cannot
        # expose a wrong initial upload. Workload reconciliation must catch it.
        event["inputs"]["out"]["values"][0] += 1
    elif fault == "extra":
        trace.append(event)
    elif fault == "missing":
        trace.pop()
    elif fault == "boundary":
        records[1]["dispatchStart"] = 0
    elif fault == "count":
        records[0]["dispatchCount"] = True
    elif fault == "empty-result":
        records[1]["actual"][0] ^= 1
    elif fault == "empty-dispatch":
        trace[1] = event
    elif fault == "mixed-target":
        trace[1]["target"] = "opengl"
    if fault:
        with pytest.raises(ValueError):
            scatter_axis_evidence.validate(np, records, trace)
    else:
        assert scatter_axis_evidence.validate(np, records, trace) == {
            "target": "metal",
            "scatterDispatchCount": 1,
        }
        assert audited == [([trace[1]], "metal")]


def test_axis_scatter_ci_requires_native_execution_on_all_targets():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["slice-update-host"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    assert "if" not in job and "continue-on-error" not in job
    steps = {step.get("name"): step for step in job["steps"]}
    step = steps["Execute axis scatter host operations"]
    assert "if" not in step and "continue-on-error" not in step
    for text in (
        "set -euo pipefail",
        "--timeout-seconds 3600",
        "portable_host.verify_scatter_axis",
        "--integer64",
        "scatter-axis-evidence",
        "tee",
    ):
        assert text in step["run"]
    assert steps["Retain slice update execution evidence"]["if"] == "always()"
    for trigger in ("pull_request", "push"):
        assert (
            "tests/test_mlx_portable_scatter_axis.py"
            in workflow.get("on", workflow.get(True))[trigger]["paths"]
        )
