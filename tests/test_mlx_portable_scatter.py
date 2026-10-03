"""General-scatter host layout, specialization and writeback contracts."""

import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    gather_packages,
    runtime,
    scatter_evidence,
)
from demos.integrations.mlx.portable_host import scatter_layout as layout
from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host import verify_scatter
from demos.integrations.mlx.portable_host.gather_workloads import words


def test_scatter_retains_existing_workloads_and_adds_product_layouts():
    cases = list(workloads.cases())
    legacy = cases[:112]
    digest = hashlib.sha256()
    for case in legacy:
        digest.update(case["id"].encode())
        source, indices, updates, expected = workloads.reference(np, case)
        for array in (source, *indices, updates, expected):
            digest.update(str((str(array.dtype), array.shape, array.strides)).encode())
            digest.update(array.tobytes())
    assert digest.hexdigest() == (
        "0866069c11124bfa9d5295f6e67beda7ebb8adabada57aa27021950801836021"
    )
    products = cases[112:]
    assert len(cases) == len({case["id"] for case in cases}) == 144
    assert all(case["operation"] == "prod" for case in products)
    assert {(case["dtype"], case["layout"]) for case in products} == {
        (dtype, layout)
        for dtype in ("int32", "uint32")
        for layout in (*workloads.LAYOUTS, "work4", "work8", "work16", "work32")
    }
    assert {case["index_dtype"] for case in products} == {
        "int32",
        "uint32",
        "int64",
        "uint64",
    }


@pytest.mark.parametrize(
    "case",
    [case for case in workloads.cases() if case["operation"] == "prod"],
    ids=lambda case: case["id"],
)
def test_scatter_product_reference_and_all_order_intermediate_bounds(case):
    source, indices, updates, expected = workloads.reference(np, case)
    independent = source.copy()
    np.multiply.at(independent, tuple(indices), updates)
    assert np.array_equal(expected, independent)
    # Python integers bound all partial products, even if zero arrives last.
    bounds = np.abs(source.astype(object))
    for coordinate in np.ndindex(indices[0].shape):
        destination = tuple(int(index[coordinate]) for index in indices)
        bounds[destination] *= np.maximum(1, np.abs(updates[coordinate].astype(object)))
    assert np.all(bounds <= np.iinfo(case["dtype"]).max)
    if case["layout"] not in {"empty", "alias", "update-broadcast"}:
        assert 0 in expected
        assert np.any(expected != 0)
        assert not np.array_equal(source, expected)
        if case["dtype"] == "int32":
            assert np.any(updates < 0)
    if case["layout"].startswith("work"):
        assert indices[0].size % int(case["layout"][4:]) != 0
        assert len(set(zip(*(index.flat for index in indices)))) < indices[0].size


def buffers(case, *, normalize_scalar=False):
    source, indices, updates, expected = workloads.reference(np, case)
    if normalize_scalar and indices[0].ndim == 0:
        indices = [index.reshape(1) for index in indices]
        updates = np.expand_dims(updates, 0)
    indices = [
        (
            np.ascontiguousarray(index)
            if any(step < 0 for step in index.strides)
            else index
        )
        for index in indices
    ]
    if any(step < 0 for step in updates.strides):
        updates = np.ascontiguousarray(updates)
    rank = indices[0].ndim
    for i in range(len(indices)):
        updates = np.expand_dims(updates, rank + i)
    output = source.copy(order="C")
    metadata = {
        "upd_shape": list(updates.shape),
        "upd_strides": [step // updates.itemsize for step in updates.strides],
        "upd_ndim": [updates.ndim],
        "upd_size": [int(np.prod(updates.shape[rank:]))],
        "out_shape": list(output.shape),
        "out_strides": [step // output.itemsize for step in output.strides],
        "out_ndim": [output.ndim],
        "axes": list(range(len(indices))),
        "idx_shapes": [size for index in indices for size in index.shape] or [0],
        "idx_strides": (
            [step // index.itemsize for index in indices for step in index.strides]
            or [0]
        ),
        "idx_contigs": (
            [int(index.flags.c_contiguous) for index in indices]
            + ([0] if rank == 0 else [])
        ),
        "idx_ndim": [rank],
        "idx_size": [indices[0].size],
    }
    arrays = {
        name: np.array(
            value, dtype="uint8" if name == "idx_contigs" else layout.METADATA[name]
        )
        for name, value in metadata.items()
    }
    arrays.update(
        updates=updates,
        out=output,
        **{f"idx{i}": index for i, index in enumerate(indices)},
    )
    supplied = {}
    for name, array in arrays.items():
        span = 1 + sum(
            (size - 1) * (step // array.itemsize)
            for size, step in zip(array.shape, array.strides)
        )
        supplied[name] = runtime.Buffer(
            name.encode(),
            b"bool_" if name == "idx_contigs" else str(array.dtype).encode(),
            array.ctypes.data,
            span,
            int(name == "out"),
        )
    work = layout.work_per_thread(rank, indices[0].size, output.size)
    entry = (
        f"scatter{case['dtype']}{case['index_dtype']}_{case['operation']}_{len(indices)}"
        f"_updc_{str(updates.flags.c_contiguous).lower()}_nwork{work}_int"
    )
    execution = {
        "workgroupCount": [
            metadata["upd_size"][0],
            (indices[0].size + work - 1) // work,
            1,
        ],
        "workgroupSize": [1, 1, 1],
    }
    return entry, supplied, arrays, expected, execution


@pytest.mark.parametrize(
    "case",
    [case for case in workloads.cases() if case["layout"] != "empty"],
    ids=lambda case: case["id"],
)
def test_scatter_validates_workload_allocations_and_launch(case):
    entry, supplied, arrays, expected, execution = buffers(case)
    result = layout.validate(entry, supplied, expected.size, execution)
    assert result["outputShape"] == list(expected.shape)
    assert result["updateShape"] == list(arrays["updates"].shape)
    assert result["updateCount"] == arrays["updates"].size
    assert (
        result["maximumIndex"] == max(buffer.count for buffer in supplied.values()) - 1
    )
    if case["layout"].startswith("work"):
        assert result["workPerThread"] == int(case["layout"][4:])


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "dtype",
        "null",
        "direction",
        "scalar-length",
        "rank",
        "index-rank",
        "axis",
        "shape",
        "slice",
        "stride",
        "metadata-length",
        "index-shape",
        "index-stride",
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
        "work",
        "size",
    ),
)
def test_scatter_rejects_invalid_layout_before_translation(fault):
    case = dict(next(workloads.cases()), layout="update-strided")
    entry, supplied, arrays, expected, execution = buffers(case)
    if fault == "missing":
        supplied.pop("axes")
    elif fault == "dtype":
        supplied["idx0"].dtype = b"uint32"
    elif fault == "null":
        supplied["updates"].data = None
    elif fault == "direction":
        supplied["updates"].output = 1
    elif fault == "scalar-length":
        supplied["idx_size"].count = 2
    elif fault == "rank":
        arrays["upd_ndim"][0] = 65
    elif fault == "index-rank":
        arrays["idx_ndim"][0] = -1
    elif fault == "axis":
        arrays["axes"][0] = 2
    elif fault == "shape":
        arrays["out_shape"][0] = 0
    elif fault == "slice":
        arrays["upd_shape"][-1] = 5
    elif fault == "stride":
        arrays["out_strides"][0] = 0
    elif fault == "metadata-length":
        supplied["upd_shape"].count += 1
    elif fault == "index-shape":
        arrays["idx_shapes"][0] = 4
    elif fault == "index-stride":
        arrays["idx_strides"][0] = -1
    elif fault == "update-span":
        supplied["updates"].count -= 1
    elif fault == "index-span":
        supplied["idx0"].count -= 1
    elif fault in {"index", "negative-index"}:
        arrays["idx0"].flat[0] = 3 if fault == "index" else -4
    elif fault == "output-size":
        supplied["out"].count -= 1
    elif fault == "overlap":
        supplied["out"].data = supplied["updates"].data
    elif fault == "geometry":
        execution["workgroupCount"][0] += 1
    elif fault == "group-size":
        execution["workgroupSize"][0] = 2
    elif fault == "update-contiguous":
        entry = entry.replace("updc_false", "updc_true")
    elif fault == "index-contiguous":
        arrays["idx_contigs"][0] = 2
    elif fault == "work":
        entry = entry.replace("nwork1", "nwork4")
    elif fault == "size":
        arrays["idx_size"][0] += 1
    with pytest.raises(ValueError):
        layout.validate(entry, supplied, expected.size, execution)


@pytest.mark.parametrize(
    "entry",
    (
        "scatterfloat32int32_sum_1_updc_true_nwork1_int",
        "scatterint32int32_median_1_updc_true_nwork1_int",
        "scatterint32int32_sum_0_updc_true_nwork1_int",
        "scatterint32int32_sum_01_updc_true_nwork1_int",
        "scatterint32int32_sum_11_updc_true_nwork1_int",
        "scatterint32int32_sum_1_updc_true_nwork2_int",
        "scatterint32int32_sum_1_updc_true_nwork1_int64_t",
    ),
)
def test_scatter_rejects_unsupported_specializations(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)
    host = SimpleNamespace(descriptors={}, gathers=object())
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 0


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault", (None, "guard", "shape", "dtype", "missing", "storage", "index")
)
def test_scatter_preserves_initialized_output_and_guards_writeback(
    tmp_path, monkeypatch, target, fault
):
    entry, supplied, arrays, expected, execution = buffers(next(workloads.cases()))
    descriptor = {"artifact": {}, "bindings": []}
    for name, buffer in supplied.items():
        dtype = buffer.dtype.decode()
        storage = runtime.physical_dtype(dtype, target)
        scalar = {
            "memberName": name,
            "elementType": storage,
            "elementStrideBytes": (
                1 if storage == "bool" else ctypes.sizeof(runtime.TYPES[storage])
            ),
        }
        if name == "out":
            scalar.update(
                componentCount=1,
                structMembers=[
                    {"name": "val", "offsetBytes": 0, "physicalType": "int"}
                ],
            )
            if fault == "storage":
                scalar["structMembers"][0]["offsetBytes"] = 4
        descriptor["bindings"].append({"name": name, "scalarLayout": scalar})
    if fault == "index":
        arrays["idx0"].flat[0] = 3

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        assert (
            inputs["out"]["values"]
            == arrays["out"].flatten().tolist() + runtime.COPY_GUARD
        )
        assert inputs["out"]["shape"] == [expected.size + 32, 1]
        return SimpleNamespace(outputs=outputs)

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
        gathers=SimpleNamespace(get=lambda *_: (descriptor, tmp_path)),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch(tuple(execution["workgroupCount"]), (1, 1, 1))
    initial = arrays["out"].copy()
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, entry, table, len(table), expected.size, launch
            )
        assert np.array_equal(arrays["out"], initial)
        assert host.dispatch_count == 0
    else:
        gather_dispatch.dispatch(host, entry, table, len(table), expected.size, launch)
        assert np.array_equal(arrays["out"], expected)
        assert host.dispatch_count == 1


@pytest.mark.parametrize("operation", ("sum", "prod", "min", "max"))
def test_scatter_source_requires_original_jit_definition(tmp_path, operation):
    entry = f"scatterint32int64_{operation}_2_updc_false_nwork4_int"
    header = tmp_path / gather_packages.JIT_HEADER
    header.parent.mkdir(parents=True)
    header.write_text(
        'constexpr auto scatter_kernels = R"({0} {1} {2} {3} {4} {5} {6} {7} {8} {9})";'
    )
    text = gather_packages.source(tmp_path, entry)
    assert '#include "mlx/backend/metal/kernels/indexing/scatter.h"' in text
    assert f"int32int64_{operation} int int64_t {operation.title()}<int> 2" in text
    assert "idx0 [[buffer(20)]]" in text and "idx1 [[buffer(21)]]" in text
    assert "idx0, idx1 false 4 int" in text
    header.write_text("missing")
    with pytest.raises(ValueError, match="JIT definition"):
        gather_packages.source(tmp_path, entry)


def test_scatter_audit_accounts_for_generated_dispatch_input(tmp_path):
    from tests.test_mlx_portable_scatter_axis import generated_dispatch_input

    event = scatter_event(tmp_path, next(workloads.cases()), "directx")
    event["details"]["request"]["buffers"][
        "CrossGLDispatchInfo"
    ] = generated_dispatch_input()
    scatter_evidence.audit_event(np, event)


def scatter_event(tmp_path, case, target="metal", *, normalize_scalar=False):
    from tests.test_mlx_portable_gather import evidence_event

    event = evidence_event(tmp_path)
    for key in list(event):
        if key.startswith("gather"):
            del event[key]
    entry, supplied, arrays, expected, execution = buffers(
        case, normalize_scalar=normalize_scalar
    )
    event.update(entry=entry, target=target, threads=expected.size, **execution)
    event.update(
        scatterMetadata=layout.validate(entry, supplied, expected.size, execution),
        scatterValues=expected.reshape(-1).tolist(),
        scatterGuardValues=runtime.COPY_GUARD.copy(),
        scatterStorageType=case["dtype"],
        outputHash=hashlib.sha256(expected.tobytes()).hexdigest(),
        inputs={},
    )
    request = event["details"]["request"]
    native_entry = {"metal": entry, "directx": "CSMain", "opengl": "main"}[target]
    request.update(
        entryPoint=native_entry,
        target=target,
        buffers={},
        dispatch={
            **execution,
            "entryPoint": native_entry,
            "globalSize": execution["workgroupCount"].copy(),
            "gridSize": execution["workgroupCount"].copy(),
        },
    )
    for name, buffer in supplied.items():
        kind = buffer.dtype.decode()
        storage = runtime.physical_dtype(kind, target)
        values = layout.values(buffer)
        if name == "idx_contigs":
            values = [
                bool(value) if target == "metal" else int(value) for value in values
            ]
        if name == "out":
            values += runtime.COPY_GUARD
        shape = [len(values), 1] if name == "out" else [len(values)]
        event["inputs"][name] = {"dtype": storage, "shape": shape, "values": values}
        scalar = {
            "elementType": storage,
            "elementStrideBytes": (
                1 if storage == "bool" else np.dtype(storage).itemsize
            ),
        }
        if name == "out":
            scalar.update(
                componentCount=1,
                structMembers=[
                    {
                        "name": "val",
                        "offsetBytes": 0,
                        "physicalType": "int" if case["dtype"] == "int32" else "uint",
                    }
                ],
            )
        request["buffers"][name] = {
            "dtype": storage,
            "shape": shape,
            "binding": {"metadata": {"scalarLayout": scalar}},
        }
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
@pytest.mark.parametrize("operation", ("none", "sum", "prod", "min", "max"))
@pytest.mark.parametrize("fault", (None, "rank", "dtype", "value"))
def test_scatter_workload_audit_requires_upstream_scalar_index_shape(
    tmp_path, monkeypatch, target, operation, fault
):
    case = next(
        case
        for case in workloads.cases()
        if case["layout"] == "scalar" and case["operation"] == operation
    )
    monkeypatch.setattr(workloads, "cases", lambda: iter([case]))
    event_case = dict(case, index_dtype="int32") if fault == "dtype" else case
    event = scatter_event(
        tmp_path, event_case, target, normalize_scalar=fault != "rank"
    )
    if fault == "value":
        event["inputs"]["idx0"]["values"][0] += 1
    *_, expected = workloads.reference(np, case)
    records = [
        {
            **case,
            "actual": words(np, expected),
            "shape": list(expected.shape),
            "resultDtype": "mlx.core." + case["dtype"],
            "inputUnchanged": True,
            "dispatchCount": 1,
            "dispatchStart": 0,
        }
    ]
    upstream = {
        "testsRun": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "dispatchStart": 1,
        "dispatchCount": 1,
    }
    if fault:
        with pytest.raises(ValueError):
            scatter_evidence.validate(np, records, [event, event], upstream)
    else:
        assert scatter_evidence.validate(np, records, [event, event], upstream) == {
            "target": target,
            "scatterDispatchCount": 1,
            "upstreamScatterDispatchCount": 1,
        }


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "case",
    [case for case in workloads.cases() if case["layout"] != "empty"],
    ids=lambda case: case["id"],
)
def test_scatter_audit_reconstructs_workload_from_uploads(tmp_path, target, case):
    event = scatter_event(tmp_path, case, target)
    initial, indices, updates, actual = scatter_evidence.audit_event(np, event)
    source, expected_indices, expected_updates, expected = workloads.reference(np, case)
    assert np.array_equal(initial, source)
    assert all(
        np.array_equal(index, expected_index)
        for index, expected_index in zip(indices, expected_indices)
    )
    assert words(np, updates) == words(np, expected_updates)
    assert actual == words(np, expected)


@pytest.mark.parametrize("fault", ("unsigned-word", "float", "bool", "missing"))
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_scatter_audit_requires_signed_native_readback(tmp_path, target, fault):
    case = next(case for case in workloads.cases() if case["id"] == "int32-prod-dense")
    event = scatter_event(tmp_path, case, target)
    assert event["scatterValues"][1] == -22
    if fault == "unsigned-word":
        event["scatterValues"][1] &= 0xFFFFFFFF
    elif fault == "float":
        event["scatterValues"][1] = -22.0
    elif fault == "bool":
        event["scatterValues"][0] = False
    else:
        event["scatterValues"].pop()
    with pytest.raises(ValueError, match="outside its storage type"):
        scatter_evidence.audit_event(np, event)


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
        "missing-binding",
        "atomic-member",
        "encoding",
        "entry",
        "source",
        "module",
        "compiler",
        "validation",
        "identity",
        "global-grid",
        "grid-size",
        "dispatch-entry",
        "thread-grid",
        "contiguity",
        "index-shape",
    ),
)
@pytest.mark.parametrize("operation", ("sum", "prod"))
def test_scatter_audit_rejects_corrupt_execution_evidence(tmp_path, fault, operation):
    event = scatter_event(
        tmp_path,
        dict(next(workloads.cases()), operation=operation, layout="update-strided"),
    )
    inputs, details = event["inputs"], event["details"]
    request = details["request"]
    if fault == "result":
        event["scatterValues"][0] ^= 1
    elif fault == "guard":
        event["scatterGuardValues"][-1] = 0
    elif fault == "hash":
        event["outputHash"] = "0" * 64
    elif fault == "index":
        inputs["idx0"]["values"][0] = 3
    elif fault == "update":
        inputs["updates"]["values"][0] += 1
    elif fault == "initial":
        inputs["out"]["values"][1 if operation == "prod" else 0] += 1
    elif fault == "axis":
        inputs["axes"]["values"][0] = 2
    elif fault == "stride":
        inputs["upd_strides"]["values"][0] += 1
    elif fault == "metadata":
        event["scatterMetadata"]["workPerThread"] = 4
    elif fault == "geometry":
        event["workgroupCount"][0] += 1
    elif fault == "binding":
        request["buffers"]["updates"]["dtype"] = "uint32"
    elif fault == "missing-binding":
        del request["buffers"]["updates"]
    elif fault == "atomic-member":
        request["buffers"]["out"]["binding"]["metadata"]["scalarLayout"][
            "structMembers"
        ][0]["offsetBytes"] = 4
    elif fault == "encoding":
        inputs["out"]["encoding"] = "ieee754-binary32"
    elif fault == "entry":
        request["entryPoint"] = "other"
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
    elif fault == "global-grid":
        request["dispatch"]["globalSize"][0] += 1
    elif fault == "grid-size":
        request["dispatch"]["gridSize"][0] += 1
    elif fault == "dispatch-entry":
        request["dispatch"]["entryPoint"] = "other"
    elif fault == "thread-grid":
        request["dispatch"]["threadGridSize"] = event["workgroupCount"]
    elif fault == "contiguity":
        inputs["idx_contigs"]["values"][0] = 2
    elif fault == "index-shape":
        inputs["idx_shapes"]["values"][0] -= 1
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


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
def test_scatter_requires_complete_matching_workloads(native, fault):
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
        records[0]["shape"] = [12]
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
            verify_scatter.validate_records(np, records, native=native)
    else:
        verify_scatter.validate_records(np, records, native=native)


def test_scatter_ci_requires_separate_three_os_host_execution():
    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["scatter-host"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    assert "if" not in job and "continue-on-error" not in job
    assert job["needs"] == "integer64-host"
    assert 90 <= job["timeout-minutes"] <= 360
    steps = {step.get("name"): step for step in job["steps"]}
    step = steps["Execute general scatter host operations"]
    assert "if" not in step and "continue-on-error" not in step
    for text in (
        "set -euo pipefail",
        "--timeout-seconds 3600",
        "portable_host.verify_scatter",
        "--integer64",
        "scatter-evidence",
        "tee",
    ):
        assert text in step["run"]
    assert steps["Retain general scatter execution evidence"]["if"] == "always()"
    for trigger in ("pull_request", "push"):
        assert (
            "tests/test_mlx_portable_scatter.py"
            in workflow.get("on", workflow.get(True))[trigger]["paths"]
        )


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "case",
        "initial",
        "extra",
        "missing",
        "boundary",
        "empty-dispatch",
        "upstream-skipped",
        "upstream-boundary",
        "upstream-missing",
        "mixed-target",
    ),
)
def test_scatter_requires_complete_workload_and_upstream_traces(
    tmp_path, monkeypatch, fault
):
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
    trace = [event, {"entry": "copy-empty-scatter-source", "target": "metal"}, event]
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
    upstream = {
        "testsRun": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "dispatchStart": 2,
        "dispatchCount": 1,
    }
    if fault == "case":
        records[0]["id"] = "other"
    elif fault == "initial":
        event["inputs"]["out"]["values"][0] += 1
    elif fault == "extra":
        trace.append(event)
    elif fault == "missing":
        trace.pop()
    elif fault == "boundary":
        records[1]["dispatchStart"] = 0
    elif fault == "empty-dispatch":
        trace[1] = event
    elif fault == "upstream-skipped":
        upstream["skipped"] = 1
    elif fault == "upstream-boundary":
        upstream["dispatchStart"] = 1
    elif fault == "upstream-missing":
        trace[-1] = {"entry": "copy-only", "target": "metal"}
    elif fault == "mixed-target":
        trace[1]["target"] = "opengl"
    if fault:
        with pytest.raises(ValueError):
            scatter_evidence.validate(np, records, trace, upstream)
    else:
        assert scatter_evidence.validate(np, records, trace, upstream) == {
            "target": "metal",
            "scatterDispatchCount": 1,
            "upstreamScatterDispatchCount": 1,
        }
        assert audited == [([trace[1]], "metal")]
