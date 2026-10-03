"""Axis-gather allocation, specialization and evidence contracts."""

import ctypes
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import gather_axis_layout as layout
from demos.integrations.mlx.portable_host import gather_axis_workloads as workloads
from demos.integrations.mlx.portable_host import (
    gather_evidence,
    gather_packages,
    runtime,
    verify_gather,
)
from demos.integrations.mlx.portable_host.gather_workloads import words


def buffers(case):
    source, (indices,), expected = workloads.reference(np, case)
    if any(step < 0 for step in source.strides):
        source = np.ascontiguousarray(source)
    if any(step < 0 for step in indices.strides):
        indices = np.ascontiguousarray(indices)
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    source_steps = [step // source.itemsize for step in source.strides]
    index_steps = [step // indices.itemsize for step in indices.strides]
    metadata = {
        "shape": [size for i, size in enumerate(indices.shape) if i != axis] or [1],
        "src_strides": (
            [step for i, step in enumerate(source_steps) if i != axis] or [0]
        ),
        "idx_strides": [step for i, step in enumerate(index_steps) if i != axis] or [0],
        "ndim": [source.ndim - 1],
        "axis": [axis],
        "axis_size": [source.shape[axis]],
        "src_ax_stride": [source_steps[axis]],
        "idx_ax_stride": [index_steps[axis]],
    }
    arrays = {
        name: np.array(value, dtype=layout.METADATA[name])
        for name, value in metadata.items()
    }
    arrays.update(src=source, indices=indices, out=np.zeros_like(expected, order="C"))
    supplied = {}
    for name, array in arrays.items():
        span = 1 + sum(
            (size - 1) * (step // array.itemsize)
            for size, step in zip(array.shape, array.strides)
        )
        kind = "bool_" if array.dtype == np.bool_ else str(array.dtype)
        supplied[name] = runtime.Buffer(
            name.encode(), kind.encode(), array.ctypes.data, span, int(name == "out")
        )
    entry = (
        f"gather_axis{case['dtype']}{case['index_dtype']}_int"
        + ("c" if source.flags.c_contiguous else "nc")
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


@pytest.mark.parametrize("case", list(workloads.cases()), ids=lambda case: case["id"])
def test_axis_gather_preserves_all_workload_layouts(case):
    entry, supplied, arrays, expected, execution = buffers(case)
    metadata = layout.validate(entry, supplied, expected.size, execution)
    assert metadata["sourceShape"] == list(arrays["src"].shape)
    assert metadata["indexShape"] == list(arrays["indices"].shape)
    assert metadata["sourceStrides"] == [
        step // arrays["src"].itemsize for step in arrays["src"].strides
    ]
    assert metadata["indexStrides"] == [
        step // arrays["indices"].itemsize for step in arrays["indices"].strides
    ]


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "null",
        "dtype",
        "direction",
        "scalar-length",
        "rank",
        "axis",
        "axis-size",
        "axis-step",
        "shape",
        "stride",
        "metadata-length",
        "source-span",
        "index-span",
        "index",
        "negative-index",
        "output-size",
        "output-overlap",
        "geometry",
        "group-size",
        "source-contiguous",
        "index-contiguous",
    ),
)
def test_axis_gather_rejects_invalid_layout_before_dispatch(fault):
    case = dict(next(workloads.cases()), layout="both-strided", axis=1)
    entry, supplied, arrays, expected, execution = buffers(case)
    if fault == "missing":
        supplied.pop("axis")
    elif fault == "null":
        supplied["src"].data = None
    elif fault == "dtype":
        supplied["indices"].dtype = b"uint32"
    elif fault == "direction":
        supplied["indices"].output = 1
    elif fault == "scalar-length":
        supplied["axis"].count = 2
    elif fault == "rank":
        arrays["ndim"][0] = 64
    elif fault == "axis":
        arrays["axis"][0] = 3
    elif fault == "axis-size":
        arrays["axis_size"][0] = 0
    elif fault == "axis-step":
        arrays["src_ax_stride"][0] = 65536
    elif fault == "shape":
        arrays["shape"][0] = 0
    elif fault == "stride":
        arrays["src_strides"][0] = -1
    elif fault == "metadata-length":
        supplied["shape"].count += 1
    elif fault == "source-span":
        supplied["src"].count -= 1
    elif fault == "index-span":
        supplied["indices"].count -= 1
    elif fault in {"index", "negative-index"}:
        arrays["indices"].flat[0] = 3 if fault == "index" else -4
    elif fault == "output-size":
        supplied["out"].count -= 1
    elif fault == "output-overlap":
        supplied["out"].data = supplied["src"].data
    elif fault == "geometry":
        execution["workgroupCount"][0] += 1
    elif fault == "group-size":
        execution["workgroupSize"][0] = 2
    elif fault == "source-contiguous":
        entry = entry.removesuffix("ncnc") + "cnc"
    elif fault == "index-contiguous":
        entry = entry.removesuffix("ncnc") + "ncc"
    with pytest.raises(ValueError):
        layout.validate(entry, supplied, expected.size, execution)


@pytest.mark.parametrize(
    "entry",
    (
        "gather_axisfloat32float32_intcc",
        "gather_axisfloat32int32_int64_tcc",
        "gather_axisint16int32_intcc",
        "gather_axisint64int64_int",
        "gather_axisint64int64_intcc_extra",
    ),
)
def test_axis_gather_rejects_unknown_specialization(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_axis_gather_source_keeps_exact_upstream_template_instantiation(tmp_path):
    entry = "gather_axisint64uint64_intncc"
    text = gather_packages.source(tmp_path, entry)
    assert '#include "mlx/backend/metal/kernels/indexing/gather_axis.h"' in text
    assert f'[[host_name("{entry}")]]' in text
    assert "decltype(gather_axis<int64_t, uint64_t, int, false, true>)" in text
    host = SimpleNamespace(descriptors={}, gathers=object())
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1
    host.gathers = None
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 0


def test_axis_gather_ci_requires_native_unchanged_upstream_test():
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
    step = steps["Execute axis gather host operations"]
    assert "if" not in step and "continue-on-error" not in step
    for text in (
        "set -euo pipefail",
        "--timeout-seconds 3600",
        "portable_host.verify_gather --axis",
        "--integer64",
        "--reductions",
        "gather-axis-evidence",
        "tee",
    ):
        assert text in step["run"]
    assert steps["Retain slice update execution evidence"]["if"] == "always()"
    assert verify_gather.AXIS_UPSTREAM_TESTS == (
        "test_ops.TestOps.test_take_along_axis",
    )
    for trigger in ("pull_request", "push"):
        assert (
            "tests/test_mlx_portable_gather_axis.py"
            in workflow.get("on", workflow.get(True))[trigger]["paths"]
        )


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize(
    "fault", (None, "case", "index-type", "axis", "value", "missing", "boundary")
)
def test_axis_gather_verifier_rejects_incomplete_records(native, fault):
    records = []
    for i, case in enumerate(workloads.cases()):
        _, _, expected = workloads.reference(np, case)
        records.append(
            {
                **case,
                "dtype": "mlx.core." + case["dtype"].removesuffix("_"),
                "actual": words(np, expected),
                "expected": words(np, expected),
                "shape": list(expected.shape),
                "inputUnchanged": True,
                "dispatchStart": i if native else 0,
                "dispatchCount": int(native),
            }
        )
    if fault == "case":
        records[-1] = records[0]
    elif fault == "index-type":
        records[0]["index_dtype"] = "uint32"
    elif fault == "axis":
        records[0]["axis"] = 2
    elif fault == "value":
        records[0]["actual"][0] ^= 1
    elif fault == "missing":
        records.pop()
    elif fault == "boundary":
        records[1]["dispatchStart"] += 1
    if fault:
        with pytest.raises(ValueError):
            verify_gather.validate_records(np, records, native=native, axis=True)
    else:
        verify_gather.validate_records(np, records, native=native, axis=True)


def axis_event(tmp_path):
    from tests.test_mlx_portable_gather import evidence_event

    event = evidence_event(tmp_path)
    case = dict(next(workloads.cases()), layout="both-strided", axis=1)
    entry, supplied, arrays, expected, execution = buffers(case)
    event.update(entry=entry, threads=expected.size, **execution)
    event["gatherMetadata"] = layout.validate(entry, supplied, expected.size, execution)
    event["gatherValues"] = words(np, expected)
    event["outputHash"] = hashlib.sha256(expected.tobytes()).hexdigest()
    event["inputs"] = {}
    request = event["details"]["request"]
    request.update(entryPoint=entry, dispatch=execution, buffers={})
    for name, buffer in supplied.items():
        dtype = buffer.dtype.decode()
        ctype = ctypes.c_uint32 if dtype == "float32" else layout.TYPES[dtype]
        values = list(
            ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents
        )
        if name == "out":
            values = [runtime.COPY_GUARD[0]] * expected.size + runtime.COPY_GUARD
        value = {"dtype": dtype, "shape": [len(values)], "values": values}
        if dtype == "float32":
            value["encoding"] = "ieee754-binary32"
        event["inputs"][name] = value
        request["buffers"][name] = {
            key: item for key, item in value.items() if key != "values"
        }
        request["buffers"][name]["binding"] = {
            "metadata": {"scalarLayout": {"elementType": dtype}}
        }
    return event


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "value",
        "index",
        "source",
        "axis",
        "metadata",
        "geometry",
        "compiler",
        "binding",
        "guard",
    ),
)
def test_axis_gather_audit_reconstructs_outputs_and_rejects_corruption(tmp_path, fault):
    event = axis_event(tmp_path)
    if fault == "value":
        event["gatherValues"][0] ^= 1
    elif fault == "index":
        event["inputs"]["indices"]["values"][0] = 3
    elif fault == "source":
        event["inputs"]["src"]["values"][0] ^= 1
    elif fault == "axis":
        event["inputs"]["axis"]["values"][0] = 2
    elif fault == "metadata":
        event["gatherMetadata"]["axis"] = 2
    elif fault == "geometry":
        event["workgroupCount"][0] += 1
    elif fault == "compiler":
        event["details"]["adapterSteps"].pop()
    elif fault == "binding":
        event["details"]["request"]["buffers"]["src"]["dtype"] = "uint32"
    elif fault == "guard":
        event["gatherGuardValues"][-1] ^= 1
    if fault:
        with pytest.raises(ValueError):
            gather_evidence.audit_event(np, event)
    else:
        _, _, actual = gather_evidence.audit_event(np, event)
        assert actual == event["gatherValues"]
