"""Axis-scatter allocation validation, native writeback and evidence contracts."""

import ctypes
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    gather_layout,
    gather_packages,
    runtime,
    scatter_axis_evidence,
)
from demos.integrations.mlx.portable_host import scatter_axis_layout as layout
from demos.integrations.mlx.portable_host import scatter_axis_workloads as workloads
from demos.integrations.mlx.portable_host import scatter_layout, verify_scatter_axis
from demos.integrations.mlx.portable_host.gather_workloads import words
from tests.ci_helpers import assert_paths_covered

# Workloads that fit one launch; the wide workloads split into several.
SINGLE = [case for case in workloads.cases() if case["layout"] not in workloads.WIDE]
WIDE = list(workloads.wide_cases())


def test_axis_scatter_retains_existing_workloads_and_adds_signed_layouts():
    cases = list(workloads.cases())
    digest = hashlib.sha256()
    for case in cases[:56]:
        digest.update(json.dumps(case, sort_keys=True).encode())
        for array in workloads.reference(np, case):
            digest.update(
                json.dumps([array.dtype.str, array.shape, array.strides]).encode()
            )
            digest.update(array.tobytes())
    assert digest.hexdigest() == (
        "b45dd7465484e9604282442e9ce6ef87902be69d3a2a541c51d34c7a4020a64d"
    )
    signed, floating, wide = cases[56:84], cases[84:112], cases[112:]
    assert len(cases) == len({case["id"] for case in cases}) == 132
    assert floating == list(workloads.float_cases())
    assert wide == WIDE and SINGLE == cases[:112]
    assert {(case["operation"], case["layout"]) for case in floating} == {
        (operation, layout)
        for operation in ("none", "sum")
        for layout in workloads.LAYOUTS
    }
    assert all(case["dtype"] == "float32" for case in floating)
    for case in floating:
        source, _, updates, expected = workloads.reference(np, case)
        assert source.dtype == expected.dtype == np.float32
        if case["layout"] != "empty":
            # Negative and fractional values in every non-empty workload.
            assert np.any(expected < 0) and np.any(expected % 1 != 0)
    assert all(
        case["dtype"] == "int32" and case["values"] == "signed" for case in signed
    )
    assert {(case["operation"], case["layout"]) for case in signed} == {
        (operation, layout)
        for operation in ("none", "sum")
        for layout in workloads.LAYOUTS
    }
    assert {case["index_dtype"] for case in signed} == {
        "int32",
        "uint32",
        "int64",
        "uint64",
    }


@pytest.mark.parametrize("case", list(workloads.cases()), ids=lambda case: case["id"])
def test_axis_scatter_reference_matches_independent_indexed_updates(case):
    source, indices, updates, expected = workloads.reference(np, case)
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    independent = (
        source.copy() if case["operation"] == "none" else np.zeros_like(source)
    )
    if case["operation"] == "none":
        np.put_along_axis(independent, indices, updates, axis=axis)
    else:
        coordinates = list(np.indices(indices.shape))
        coordinates[axis] = indices.astype(np.int64) % source.shape[axis]
        np.add.at(independent, tuple(coordinates), updates)
        bounds = np.zeros_like(source, dtype=object)
        np.add.at(bounds, tuple(coordinates), np.abs(updates.astype(object)))
        if case["dtype"] == "float32":
            # Quarter steps far below 2^24 sum exactly in any order.
            assert np.all(bounds * 4 < 2**24)
            assert np.all(np.asarray(updates) * 4 == np.round(np.asarray(updates) * 4))
            assert scatter_axis_evidence.exact_float_sums(
                np,
                np.zeros_like(source),
                indices,
                updates,
                axis,
                source.shape[axis],
            )
        else:
            assert np.all(bounds <= np.iinfo(case["dtype"]).max)
    assert np.array_equal(expected, independent)
    if case.get("values") == "signed":
        assert np.any(source < 0)
        if case["layout"] != "empty":
            assert np.any(updates < 0) and np.any(expected < 0)


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
    [case for case in SINGLE if case["layout"] != "empty"],
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
        "scatter_axisfloat16int32_none_intcc",
        "scatter_axisbfloat16int32_sum_intcc",
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


def scatter_event(tmp_path, case, target="metal", launch=None):
    from demos.integrations.mlx.tests.host.test_portable_gather import evidence_event

    event = evidence_event(tmp_path)
    for key in list(event):
        if key.startswith("gather"):
            del event[key]
    if launch is None:
        entry, supplied, arrays, expected, execution = buffers(case)
    else:
        # One launch of a split workload reads back its own output slab.
        entry, supplied, arrays, execution, _ = launch
        trial = dict(arrays, out=arrays["out"].copy())
        run_kernel(entry, trial, execution)
        expected = trial["out"]
    event.update(entry=entry, target=target, threads=expected.size, **execution)
    event["scatterMetadata"] = layout.validate(
        entry, supplied, expected.size, execution
    )
    event["scatterValues"] = (
        words(np, expected)
        if case["dtype"] == "float32"
        else expected.reshape(-1).tolist()
    )
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
        kind = buffer.dtype.decode()
        dtype = (
            scatter_layout.atomic_storage_dtype(kind, target) if name == "out" else kind
        )
        values = list(
            ctypes.cast(
                buffer.data,
                ctypes.POINTER(
                    (ctypes.c_uint32 if kind == "float32" else runtime.TYPES[kind])
                    * buffer.count
                ),
            ).contents
        )
        if name == "out":
            values += runtime.COPY_GUARD
        value = {
            "dtype": dtype,
            "shape": [len(values), 1] if name == "out" else [len(values)],
            "values": values,
        }
        if dtype == "float32":
            value["encoding"] = runtime.FLOAT32_BITS
        event["inputs"][name] = value
        scalar = {"elementType": dtype, "elementStrideBytes": np.dtype(dtype).itemsize}
        if name == "out":
            scalar.update(
                componentCount=1,
                structMembers=[
                    {
                        "name": "val",
                        "offsetBytes": 0,
                        "physicalType": scatter_layout.ATOMIC_TYPES[dtype],
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


def generated_dispatch_input():
    execution = {
        "coordinateSpace": "physical",
        "dimensions": 3,
        "kind": "dispatch-workgroup-count",
        "memberName": "crossglNumWorkGroups",
        "valueSource": "dispatch.workgroupCount",
    }
    return {
        "name": "CrossGLDispatchInfo",
        "source": "input",
        "dtype": "uint32",
        "shape": [3],
        "metadata": {
            "source": "dispatch.workgroupCount",
            "executionInput": execution,
            "runtimeValueName": "CrossGLDispatchInfo",
        },
        "binding": {
            "name": "CrossGLDispatchInfo",
            "kind": "constant-buffer",
            "access": "read",
            "metadata": {
                "provenance": {
                    "kind": "generated-execution-input",
                    "executionInput": execution,
                },
                "scalarLayout": {
                    "elementType": "uint32",
                    "vectorWidth": 3,
                    "memberName": "crossglNumWorkGroups",
                    "memberOffsetBytes": 0,
                    "elementStrideBytes": 12,
                    "blockSizeBytes": 16,
                    "storageLayout": "hlsl-constant-buffer",
                },
            },
        },
    }


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "dtype",
        "shape",
        "provenance",
        "source",
        "layout",
        "count",
        "target",
        "duplicate",
    ),
)
def test_axis_scatter_audit_accounts_for_generated_dispatch_input(tmp_path, fault):
    event = scatter_event(tmp_path, next(workloads.cases()), "directx")
    request = event["details"]["request"]
    buffer = generated_dispatch_input()
    request["buffers"]["CrossGLDispatchInfo"] = buffer
    if fault == "dtype":
        buffer["dtype"] = "int32"
    elif fault == "shape":
        buffer["shape"] = [4]
    elif fault == "provenance":
        del buffer["binding"]["metadata"]["provenance"]
    elif fault == "source":
        buffer["metadata"]["source"] = "dispatch.workgroupSize"
    elif fault == "layout":
        buffer["binding"]["metadata"]["scalarLayout"]["memberOffsetBytes"] = 4
    elif fault == "count":
        request["dispatch"]["workgroupCount"] = [2, 3, 4]
    elif fault == "target":
        event["target"] = "metal"
    elif fault == "duplicate":
        request["buffers"]["extra"] = generated_dispatch_input()
    if fault:
        with pytest.raises(ValueError):
            scatter_axis_evidence.audit_event(np, event)
    else:
        scatter_axis_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "case",
    [case for case in SINGLE if case["layout"] != "empty"],
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


def test_axis_scatter_float_sums_are_exact_only_without_rounding():
    zeros = np.zeros((2,), dtype=np.float32)
    indices = np.array([0, 0, 0, 1], dtype=np.int32)
    exact = np.array([0.25, -1.5, 3.0, -0.0], dtype=np.float32)
    assert scatter_axis_evidence.exact_float_sums(np, zeros, indices, exact, 0, 2)
    # 2^24 + 1 - 1 rounds in one order and not in the other.
    rounding = np.array([2.0**24, 1.0, -1.0, 0.0], dtype=np.float32)
    assert not scatter_axis_evidence.exact_float_sums(
        np, zeros, indices, rounding, 0, 2
    )
    for value in (np.inf, np.nan):
        special = np.array([value, 0.0, 0.0, 0.0], dtype=np.float32)
        assert not scatter_axis_evidence.exact_float_sums(
            np, zeros, indices, special, 0, 2
        )
    # Tiny steps are exact while their magnitudes fit in 24 bits of the step.
    tiny = np.array([2.0**-140, 2.0**-149, -(2.0**-149), 0.0], dtype=np.float32)
    assert scatter_axis_evidence.exact_float_sums(np, zeros, indices, tiny, 0, 2)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("fault", ("order", "float-word", "encoding", "storage"))
def test_axis_scatter_audit_requires_exact_raw_float32_evidence(
    tmp_path, target, fault
):
    case = next(
        case
        for case in workloads.float_cases()
        if case["operation"] == "sum" and case["layout"] == "dense"
    )
    event = scatter_event(tmp_path, case, target)
    scatter_axis_evidence.audit_event(np, event)
    upload = next(
        value for key, value in event["inputs"].items() if key.endswith("upd")
    )
    output = next(
        value for key, value in event["inputs"].items() if key.endswith("out")
    )
    if fault == "order":
        upload["values"][0] = words(np, np.array([2.0**30], dtype=np.float32))[0]
        message = "update order"
    elif fault == "float-word":
        upload["values"][0] = float(upload["values"][0])
        message = "raw binary32"
    elif fault == "encoding":
        del upload["encoding"]
        message = "binding layout"
    else:
        output["dtype"] = "float32" if target == "opengl" else "uint32"
        message = "binding layout|storage type"
    with pytest.raises(ValueError, match=message):
        scatter_axis_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("operation", ("none", "sum"))
@pytest.mark.parametrize(
    "fault", ("unsigned-word", "float", "bool", "missing", "underflow")
)
def test_axis_scatter_audit_requires_typed_signed_native_readback(
    tmp_path, target, operation, fault
):
    case = next(
        case
        for case in workloads.cases()
        if case["id"] == f"int32-{operation}-dense-signed"
    )
    event = scatter_event(tmp_path, case, target)
    assert event["scatterValues"][0] < 0
    if fault == "unsigned-word":
        event["scatterValues"][0] &= 0xFFFFFFFF
    elif fault == "float":
        event["scatterValues"][0] = float(event["scatterValues"][0])
    elif fault == "bool":
        event["scatterValues"][0] = False
    elif fault == "underflow":
        event["scatterValues"][0] = -(2**31) - 1
    else:
        event["scatterValues"].pop()
    with pytest.raises(ValueError, match="outside its storage type"):
        scatter_axis_evidence.audit_event(np, event)


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
        and "values" not in case
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
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["slice-update-host"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    assert (
        job["if"] == "needs.ci-plan.outputs.heavy == 'true'"
        and "continue-on-error" not in job
    )
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
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[trigger].get("paths", ["**"]),
            "demos/integrations/mlx/tests/host/test_portable_scatter_axis.py",
        )


def launches(case):
    """Per-launch buffers as backend.cpp's ScatterAxis::eval_gpu supplies them."""
    source, indices, updates, expected = workloads.reference(np, case)
    axis = case["axis"] % source.ndim
    out = (
        source.copy(order="C")
        if case["operation"] == "none"
        else np.zeros_like(source, order="C")
    )
    physical = [
        np.ascontiguousarray(array) if any(s < 0 for s in array.strides) else array
        for array in (updates, indices)
    ]
    steps = [[s // array.itemsize for s in array.strides] for array in physical]
    bases = [
        np.lib.stride_tricks.as_strided(
            array, (1 + sum((n - 1) * s for n, s in zip(array.shape, step)),),
            (array.itemsize,),
        )
        for array, step in zip(physical, steps)
    ]
    entry = (
        f"scatter_axis{case['dtype']}{case['index_dtype']}_{case['operation']}_int"
        + "".join("c" if array.flags.c_contiguous else "nc" for array in physical)
    )
    post = math.prod(indices.shape[axis + 1 :])
    keep = [i for i in range(source.ndim) if i != axis]
    for slab, rows in layout.launch_plan(list(indices.shape), axis):
        start = [part.start or 0 for part in slab]
        lead = [len(range(*part.indices(indices.shape[i]))) for i, part in enumerate(slab)]
        tile = lead + [rows.stop - rows.start] + list(indices.shape[axis + 1 :])
        first = int(np.ravel_multi_index(start, indices.shape[:axis])) if slab else 0
        outputs = math.prod(lead) * source.shape[axis] * post
        arrays = {
            "shape": [tile[i] for i in keep] or [1],
            "upd_strides": [steps[0][i] for i in keep] or [0],
            "idx_strides": [steps[1][i] for i in keep] or [0],
            "ndim": [source.ndim - 1],
            "axis": [axis],
            "out_axis_size": [source.shape[axis]],
            "upd_ax_stride": [steps[0][axis]],
            "idx_ax_stride": [steps[1][axis]],
        }
        arrays = {
            name: np.array(value, dtype=layout.METADATA[name])
            for name, value in arrays.items()
        }
        for name, base, step in zip(("upd", "indices"), bases, steps):
            offset = sum(a * s for a, s in zip(start, step)) + rows.start * step[axis]
            extent = 1 + sum((n - 1) * s for n, s in zip(tile, step))
            arrays[name] = base[offset : offset + extent]
        arrays["out"] = out.reshape(-1)[
            first * source.shape[axis] * post : first * source.shape[axis] * post
            + outputs
        ]
        supplied = {
            name: runtime.Buffer(
                name.encode(),
                str(array.dtype).encode(),
                array.ctypes.data,
                array.size,
                int(name == "out"),
            )
            for name, array in arrays.items()
        }
        execution = {
            "workgroupCount": [post, rows.stop - rows.start, math.prod(lead)],
            "workgroupSize": [1, 1, 1],
        }
        yield entry, supplied, arrays, execution, outputs
    yield out, expected


def run_kernel(entry, arrays, execution):
    """Evaluate the pinned scatter_axis kernel over one launch grid."""
    _, _, operation, update_contiguous, index_contiguous = layout.signature(entry)
    gx, gy, gz = execution["workgroupCount"]
    x, y, z = (
        part.reshape(-1)
        for part in np.meshgrid(
            np.arange(gx), np.arange(gy), np.arange(gz), indexing="ij"
        )
    )
    element = z * gx
    ndim = int(arrays["ndim"][0])

    def location(step, strides, contiguous):
        if contiguous:
            return y * step + element * gy + x
        flat, result = element + x, y * step
        for size, stride in zip(
            reversed(arrays["shape"][:ndim]), reversed(strides[:ndim])
        ):
            result = result + flat % int(size) * int(stride)
            flat = flat // int(size)
        return result

    index = arrays["indices"][
        location(
            int(arrays["idx_ax_stride"][0]), arrays["idx_strides"], index_contiguous
        )
    ].astype(np.int64)
    axis_size = int(arrays["out_axis_size"][0])
    index = np.where(index < 0, index + axis_size, index)
    update = arrays["upd"][
        location(
            int(arrays["upd_ax_stride"][0]), arrays["upd_strides"], update_contiguous
        )
    ]
    destination = element * axis_size + index * gx + x
    if operation == "none":
        arrays["out"][destination] = update
    else:
        np.add.at(arrays["out"], destination, update)


def test_axis_scatter_launch_plan_splits_each_wide_axis():
    plan = layout.launch_plan
    assert plan([4, 7, 1], 2) == [((slice(0, 4), slice(None)), slice(0, 1))]
    assert plan([70001, 3], 0) == [((), slice(0, 65535)), ((), slice(65535, 70001))]
    assert [rows for _, rows in plan([131073, 1], 1)] == [slice(0, 1)] * 3
    assert [slab for slab, _ in plan([131073, 1], 1)] == [
        (slice(0, 65535),),
        (slice(65535, 131070),),
        (slice(131070, 131073),),
    ]
    assert [slab for slab, _ in plan([300, 300, 1], 2)] == [
        (slice(0, 218), slice(None)),
        (slice(218, 300), slice(None)),
    ]
    # Splitting the index axis keeps one leading position per launch.
    assert plan([3, 70000], 1) == [
        ((slice(start, start + 1),), rows)
        for start in range(3)
        for rows in (slice(0, 65535), slice(65535, 70000))
    ]
    assert plan([2, 2, 65536], 2) == [
        ((slice(a, a + 1), slice(b, b + 1)), rows)
        for a in range(2)
        for b in range(2)
        for rows in (slice(0, 65535), slice(65535, 65536))
    ]
    assert plan([65535, 2], 0) == [((), slice(0, 65535))]
    with pytest.raises(ValueError, match="per-axis workgroup"):
        plan([2, 65536], 0)


@pytest.mark.parametrize("case", WIDE, ids=lambda case: case["id"])
def test_axis_scatter_wide_launches_validate_and_compose_the_result(case):
    *tiles, (out, expected) = launches(case)
    shape, _, rows, strided = workloads.WIDE[case["layout"]]
    assert expected.size > 65535 or rows > 65535 or len(tiles) > 1
    assert ("nc" in tiles[0][0]) == strided
    for entry, supplied, arrays, execution, outputs in tiles:
        assert all(1 <= size <= 65535 for size in execution["workgroupCount"])
        result = layout.validate(entry, supplied, outputs, execution)
        assert result["outputCount"] == outputs
        run_kernel(entry, arrays, execution)
    assert words(np, out) == words(np, expected)


def test_axis_scatter_wide_validation_bounds_storage_and_launch_axes():
    case = next(case for case in WIDE if case["layout"] == "wide-output")
    (entry, supplied, arrays, execution, outputs), _ = launches(case)
    assert outputs == 4 * 7 * 4096 > layout.MAX_ELEMENTS
    layout.validate(entry, supplied, outputs, execution)
    for fault in ("grid", "storage", "alignment"):
        trial = {name: runtime.Buffer(*buffer) for name, buffer in (
            (name, (b.name, b.dtype, b.data, b.count, b.output))
            for name, b in supplied.items()
        )}
        launch = json.loads(json.dumps(execution))
        if fault == "grid":
            launch["workgroupCount"][2] = 65536
        elif fault == "storage":
            trial["out"].count = gather_layout.MAX_STORAGE_ELEMENTS + 1
        else:
            trial["upd"].data += 1
        with pytest.raises(ValueError):
            layout.validate(entry, trial, outputs, launch)


def test_axis_scatter_audit_composes_split_launches(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import verify_bitwise

    case = next(
        case
        for case in WIDE
        if case["layout"] == "wide-split" and case["operation"] == "sum"
    )
    monkeypatch.setattr(workloads, "cases", lambda: iter([case]))
    monkeypatch.setattr(verify_bitwise, "verify_native_identity", lambda *_: None)
    *tiles, (out, expected) = launches(case)
    trace = []
    for entry, supplied, arrays, execution, outputs in tiles:
        launch = (entry, supplied, arrays, execution, outputs)
        trace.append(scatter_event(tmp_path, case, launch=launch))
        run_kernel(entry, arrays, execution)
    record = {
        **case,
        "actual": words(np, expected),
        "shape": list(expected.shape),
        "resultDtype": "mlx.core." + case["dtype"],
        "inputUnchanged": True,
        "dispatchStart": 0,
        "dispatchCount": len(trace),
    }
    assert len(trace) == 6
    assert scatter_axis_evidence.validate(np, [record], trace) == {
        "target": "metal",
        "scatterDispatchCount": 6,
    }
    for fault in ("order", "missing", "initial"):
        events = json.loads(json.dumps(trace))
        if fault == "order":
            events[0], events[1] = events[1], events[0]
        elif fault == "missing":
            events.pop()
        else:
            events[1]["inputs"]["out"]["values"][0] += 1
        with pytest.raises(ValueError):
            scatter_axis_evidence.validate(
                np, [dict(record, dispatchCount=len(events))], events
            )
