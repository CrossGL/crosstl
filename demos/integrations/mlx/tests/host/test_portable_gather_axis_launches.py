"""Axis-gather launches that split MLX's grid past one launch axis."""

import ctypes
import hashlib
import json
import math

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    gather_axis_evidence,
)
from demos.integrations.mlx.portable_host import gather_axis_layout as layout
from demos.integrations.mlx.portable_host import gather_axis_workloads as workloads
from demos.integrations.mlx.portable_host import (
    gather_evidence,
    gather_layout,
    runtime,
    scatter_axis_layout,
)
from demos.integrations.mlx.portable_host.gather_workloads import words

WIDE = [case for case in workloads.cases() if case["layout"] in workloads.WIDE]


def physical(array):
    """The view the host binds: negative strides are densified first."""
    if any(step < 0 for step in array.strides):
        return np.ascontiguousarray(array)
    return array


def launches(case):
    """Mirror GatherAxis::eval_gpu: rebased views and output slices per launch."""
    source, (indices,), expected = workloads.reference(np, case)
    source, indices = physical(source), physical(indices)
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    steps = [
        [step // array.itemsize for step in array.strides]
        for array in (source, indices)
    ]
    bases = [
        np.lib.stride_tricks.as_strided(
            array,
            (1 + sum((n - 1) * s for n, s in zip(array.shape, step)),),
            (array.itemsize,),
        )
        for array, step in zip((source, indices), steps)
    ]
    out = np.zeros(expected.shape, expected.dtype)
    wide = layout.wide(list(indices.shape), axis)
    entry = f"gather_axis{case['dtype']}{case['index_dtype']}_int" + (
        "ncnc"
        if wide
        else "".join(
            "c" if array.flags.c_contiguous else "nc" for array in (source, indices)
        )
    )
    keep = [i for i in range(source.ndim) if i != axis]
    for region in layout.launch_plan(list(indices.shape), axis):
        start = [part.indices(n)[0] for part, n in zip(region, indices.shape)]
        tile = [len(range(*part.indices(n))) for part, n in zip(region, indices.shape)]
        first = int(np.ravel_multi_index(start, indices.shape))
        outputs = math.prod(tile)
        arrays = {
            "shape": [tile[i] for i in keep] or [1],
            "src_strides": [steps[0][i] for i in keep] or [0],
            "idx_strides": [steps[1][i] for i in keep] or [0],
            "ndim": [source.ndim - 1],
            "axis": [axis],
            "axis_size": [source.shape[axis]],
            "src_ax_stride": [steps[0][axis]],
            "idx_ax_stride": [steps[1][axis]],
        }
        arrays = {
            name: np.array(value, dtype=layout.METADATA[name])
            for name, value in arrays.items()
        }
        source_offset = sum(start[i] * steps[0][i] for i in keep)
        source_extent = (
            1
            + (source.shape[axis] - 1) * steps[0][axis]
            + sum((tile[i] - 1) * steps[0][i] for i in keep)
        )
        index_offset = sum(a * s for a, s in zip(start, steps[1]))
        index_extent = 1 + sum((n - 1) * s for n, s in zip(tile, steps[1]))
        arrays["src"] = bases[0][source_offset : source_offset + source_extent]
        arrays["indices"] = bases[1][index_offset : index_offset + index_extent]
        arrays["out"] = out.reshape(-1)[first : first + outputs]
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
            "workgroupCount": [
                math.prod(tile[axis + 1 :]),
                tile[axis],
                math.prod(tile[:axis]),
            ],
            "workgroupSize": [1, 1, 1],
        }
        yield entry, supplied, arrays, execution, outputs, region
    yield out, expected


def run_kernel(entry, arrays, execution):
    """Evaluate the pinned gather_axis kernel over one launch grid."""
    _, index_dtype, source_contiguous, index_contiguous = layout.signature(entry)
    gx, gy, gz = execution["workgroupCount"]
    x, y, z = (
        part.reshape(-1)
        for part in np.meshgrid(
            np.arange(gx), np.arange(gy), np.arange(gz), indexing="ij"
        )
    )
    element = z * gx
    ndim = int(arrays["ndim"][0])

    def elem_to_loc(flat, strides):
        result = np.zeros_like(flat)
        for size, stride in zip(
            reversed(arrays["shape"][:ndim]), reversed(strides[:ndim])
        ):
            result = result + flat % int(size) * int(stride)
            flat = flat // int(size)
        return result

    output = element * gy + x
    index = y * int(arrays["idx_ax_stride"][0]) + (
        output if index_contiguous else elem_to_loc(element + x, arrays["idx_strides"])
    )
    axis_size = int(arrays["axis_size"][0])
    value = arrays["indices"][index].astype(np.int64)
    if index_dtype.startswith("int"):
        value = np.where(value < 0, value + axis_size, value)
    source = value * int(arrays["src_ax_stride"][0]) + (
        element * axis_size + x
        if source_contiguous
        else elem_to_loc(element + x, arrays["src_strides"])
    )
    arrays["out"][output + y * gx] = arrays["src"][source]


def test_axis_gather_launch_plan_splits_each_wide_axis():
    plan = layout.launch_plan
    full = slice(None)
    assert plan([4, 7, 1], 2) == [(slice(0, 4), full, slice(0, 1))]
    assert plan([3], 0) == [(slice(0, 3),)]
    assert plan([70001], 0) == [(slice(0, 65535),), (slice(65535, 70001),)]
    assert plan([131073, 1], 1) == [
        (slice(0, 65535), slice(0, 1)),
        (slice(65535, 131070), slice(0, 1)),
        (slice(131070, 131073), slice(0, 1)),
    ]
    assert plan([300, 300, 3], 2) == [
        (slice(0, 218), full, slice(0, 3)),
        (slice(218, 300), full, slice(0, 3)),
    ]
    # Splitting the index axis keeps one leading position per launch.
    assert plan([3, 70000], 1) == [
        (slice(start, start + 1), rows)
        for start in range(3)
        for rows in (slice(0, 65535), slice(65535, 70000))
    ]
    # A trailing extent past one launch axis runs per leading position and
    # index row, over rectangular slabs of the trailing axes.
    assert plan([2, 70001], 0) == [
        (slice(row, row + 1), part)
        for row in range(2)
        for part in (slice(0, 65535), slice(65535, 70001))
    ]
    assert plan([2, 3, 257, 257], 1) == [
        (slice(a, a + 1), slice(b, b + 1), rows, full)
        for a in range(2)
        for b in range(3)
        for rows in (slice(0, 255), slice(255, 257))
    ]
    assert plan([65535, 2], 0) == [(slice(0, 65535), full)]


@pytest.mark.parametrize(
    "shape, axis",
    (
        ([4, 7, 1], 2),
        ([70001, 3], 0),
        ([131073, 1], 1),
        ([300, 300, 1], 2),
        ([3, 70000], 1),
        ([2, 2, 65536], 2),
        ([65535, 2], 0),
    ),
)
def test_axis_gather_narrow_plan_matches_axis_scatter(shape, axis):
    """Both axis kernels share the host's leading-slab and row planner."""
    trailing = (slice(None),) * (len(shape) - axis - 1)
    assert layout.launch_plan(shape, axis) == [
        slab + (rows,) + trailing
        for slab, rows in scatter_axis_layout.launch_plan(shape, axis)
    ]


def test_axis_gather_wide_workloads_split_every_launch_axis():
    assert len(WIDE) == 2 * len(workloads.WIDE)
    assert {case["dtype"] for case in WIDE} == {"float32", "int64"}
    for name, (shape, axis, rows, _) in workloads.WIDE.items():
        index_shape = list(shape)
        index_shape[axis] = rows
        assert (
            math.prod(index_shape[axis + 1 :]) > 65535
            or rows > 65535
            or math.prod(index_shape[:axis]) > 65535
        ), name
        assert len(layout.launch_plan(index_shape, axis)) > 1, name


@pytest.mark.parametrize("case", WIDE, ids=lambda case: case["id"])
def test_axis_gather_wide_launches_validate_and_compose_the_result(case):
    *tiles, (out, expected) = launches(case)
    strided = workloads.WIDE[case["layout"]][3]
    entry = tiles[0][0]
    assert ("nc" in entry) == (
        strided or layout.wide(list(expected.shape), tiles[0][2]["axis"][0])
    )
    covered = np.zeros(expected.shape, dtype=np.int64)
    for entry, supplied, arrays, execution, outputs, region in tiles:
        assert all(1 <= size <= 65535 for size in execution["workgroupCount"])
        metadata = layout.validate(entry, supplied, outputs, execution)
        assert metadata["outputCount"] == outputs
        run_kernel(entry, arrays, execution)
        covered[region] += 1
    assert np.all(covered == 1)
    assert words(np, out) == words(np, expected)


def test_axis_gather_wide_validation_bounds_storage_and_launch_axes():
    case = next(case for case in WIDE if case["layout"] == "wide-trailing")
    entry, supplied, _, execution, outputs, _ = next(launches(case))
    assert outputs == 65535 and entry.endswith("ncnc")
    layout.validate(entry, supplied, outputs, execution)
    for fault in ("grid", "storage", "alignment", "span"):
        trial = {
            name: runtime.Buffer(b.name, b.dtype, b.data, b.count, b.output)
            for name, b in supplied.items()
        }
        launch = json.loads(json.dumps(execution))
        if fault == "grid":
            launch["workgroupCount"][0] = 65536
        elif fault == "storage":
            trial["src"].count = gather_layout.MAX_STORAGE_ELEMENTS + 1
        elif fault == "alignment":
            trial["src"].data += 1
        else:
            trial["src"].count -= 1
        with pytest.raises(ValueError):
            layout.validate(entry, trial, outputs, launch)


def launch_event(tmp_path, entry, supplied, execution, outputs, values):
    from demos.integrations.mlx.tests.host.test_portable_gather import evidence_event

    event = evidence_event(tmp_path)
    values = np.ascontiguousarray(values)
    event.update(entry=entry, threads=outputs, **execution)
    event["gatherMetadata"] = layout.validate(entry, supplied, outputs, execution)
    event["gatherValues"] = words(np, values)
    event["outputHash"] = hashlib.sha256(values.tobytes()).hexdigest()
    event["inputs"] = {}
    request = event["details"]["request"]
    request.update(entryPoint=entry, dispatch=execution, buffers={})
    for name, buffer in supplied.items():
        dtype = buffer.dtype.decode()
        ctype = ctypes.c_uint32 if dtype == "float32" else layout.TYPES[dtype]
        items = list(
            ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents
        )
        if name == "out":
            items = [runtime.COPY_GUARD[0]] * outputs + runtime.COPY_GUARD
        value = {"dtype": dtype, "shape": [len(items)], "values": items}
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


@pytest.mark.parametrize("layout_name", ("wide-split", "wide-trailing"))
def test_axis_gather_audit_composes_split_launches(tmp_path, layout_name):
    case = next(
        case
        for case in WIDE
        if case["layout"] == layout_name and case["dtype"] == "float32"
    )
    *tiles, (out, expected) = launches(case)
    source, (indices,), _ = workloads.reference(np, case)
    axis = case["axis"] % source.ndim
    trace = []
    for entry, supplied, arrays, execution, outputs, region in tiles:
        trace.append(
            launch_event(
                tmp_path, entry, supplied, execution, outputs, expected[region]
            )
        )
    assert len(trace) == 6
    actual = gather_axis_evidence.audit_batches(
        np, trace, source, indices, expected, axis
    )
    assert actual == words(np, expected)
    for fault in ("order", "missing", "value", "source"):
        events = json.loads(json.dumps(trace))
        if fault == "order":
            events[0], events[1] = events[1], events[0]
        elif fault == "missing":
            events.pop()
        elif fault == "value":
            events[-1]["gatherValues"][0] ^= 1
        else:
            events[1]["inputs"]["src"]["values"][0] ^= 1
        with pytest.raises(ValueError):
            gather_axis_evidence.audit_batches(
                np, events, source, indices, expected, axis
            )


def test_axis_gather_validate_requires_the_planned_launches(tmp_path, monkeypatch):
    case = next(
        case
        for case in WIDE
        if case["layout"] == "wide-split" and case["dtype"] == "float32"
    )
    monkeypatch.setattr(workloads, "cases", lambda: iter([case]))
    *tiles, (_, expected) = launches(case)
    trace = [
        launch_event(tmp_path, entry, supplied, execution, outputs, expected[region])
        for entry, supplied, _, execution, outputs, region in tiles
    ]
    record = {
        "dispatchStart": 0,
        "dispatchCount": len(trace),
        "actual": words(np, expected),
    }
    with pytest.raises(ValueError, match="Upstream gather trace is incomplete"):
        gather_evidence.validate(
            np,
            [record],
            trace,
            {"dispatchStart": 6, "dispatchCount": 0},
            workloads=workloads,
        )
    with pytest.raises(ValueError, match="launch count"):
        gather_evidence.validate(
            np,
            [dict(record, dispatchCount=5)],
            trace[:5],
            {"dispatchStart": 5, "dispatchCount": 0},
            workloads=workloads,
        )
