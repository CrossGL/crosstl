"""Validate axis-scatter allocations, indices and launch geometry."""

import ctypes
import itertools
import math
import re

from demos.integrations.mlx.portable_host.gather_layout import (
    MAX_ELEMENTS,
    TYPES,
    values,
)

METADATA = {
    "shape": "int32",
    "upd_strides": "int64",
    "idx_strides": "int64",
    "ndim": "uint64",
    "axis": "int32",
    "out_axis_size": "int32",
    "upd_ax_stride": "uint64",
    "idx_ax_stride": "uint64",
}


def signature(entry):
    match = re.fullmatch(
        r"scatter_axis(int32|uint32)(int32|uint32|int64|uint64)_(none|sum)_int(c|nc)(c|nc)",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native axis-scatter entry")
    dtype, index_dtype, operation, update, index = match.groups()
    return dtype, index_dtype, operation, update == "c", index == "c"


def validate(entry, buffers, logical_size, execution):
    dtype, index_dtype, operation, update_contiguous, index_contiguous = signature(
        entry
    )
    dtypes = {**METADATA, "upd": dtype, "indices": index_dtype, "out": dtype}
    if set(buffers) != set(dtypes) or not 0 < logical_size <= MAX_ELEMENTS:
        raise ValueError("Native axis-scatter buffers or output size do not match")
    for name, buffer in buffers.items():
        if (
            not buffer.data
            or buffer.dtype != dtypes[name].encode("ascii")
            or not 0 < buffer.count <= MAX_ELEMENTS
            or buffer.output != int(name == "out")
        ):
            raise ValueError(
                "Native axis-scatter buffer layout or direction is invalid"
            )
    scalars = ("ndim", "axis", "out_axis_size", "upd_ax_stride", "idx_ax_stride")
    if any(buffers[name].count != 1 for name in scalars):
        raise ValueError("Native axis-scatter scalar metadata length is invalid")
    ndim, axis, axis_size, update_step, index_step = (
        values(buffers[name])[0] for name in scalars
    )
    if (
        not 0 <= ndim < 64
        or not 0 <= axis <= ndim
        or not 1 <= axis_size <= MAX_ELEMENTS
        or not 0 <= update_step <= MAX_ELEMENTS
        or not 0 <= index_step <= MAX_ELEMENTS
        or buffers["out"].count != logical_size
        or any(
            buffers[name].count != max(1, ndim)
            for name in ("shape", "upd_strides", "idx_strides")
        )
    ):
        raise ValueError("Native axis-scatter metadata bounds or lengths are invalid")
    shape, update_strides, index_strides = (
        values(buffers[name])[:ndim] for name in ("shape", "upd_strides", "idx_strides")
    )
    if any(not 1 <= size <= MAX_ELEMENTS for size in shape) or any(
        not 0 <= step <= MAX_ELEMENTS for step in update_strides + index_strides
    ):
        raise ValueError("Native axis-scatter shape or strides are invalid")
    grid = execution.get("workgroupCount")
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or any(type(size) is not int or not 1 <= size <= MAX_ELEMENTS for size in grid)
        or execution != {"workgroupCount": grid, "workgroupSize": [1, 1, 1]}
        or grid[0] != math.prod(shape[axis:])
        or grid[2] != math.prod(shape[:axis])
        or math.prod(grid) > MAX_ELEMENTS
        or logical_size != math.prod(shape) * axis_size
    ):
        raise ValueError("Native axis-scatter launch does not match its shapes")
    index_shape, output_shape = shape.copy(), shape.copy()
    index_shape.insert(axis, grid[1])
    output_shape.insert(axis, axis_size)
    update_strides.insert(axis, update_step)
    index_strides.insert(axis, index_step)
    for name, strides, contiguous in (
        ("upd", update_strides, update_contiguous),
        ("indices", index_strides, index_contiguous),
    ):
        span = 1 + sum((size - 1) * step for size, step in zip(index_shape, strides))
        if span != buffers[name].count:
            raise ValueError(
                "Native axis-scatter storage span does not match its allocation"
            )
        if contiguous and any(
            size > 1 and step != math.prod(index_shape[i + 1 :])
            for i, (size, step) in enumerate(zip(index_shape, strides))
        ):
            raise ValueError("Native axis-scatter contiguity specialization is invalid")
    indices = values(buffers["indices"])
    for coordinates in itertools.product(*(range(size) for size in index_shape)):
        location = sum(index * step for index, step in zip(coordinates, index_strides))
        value = indices[location]
        if value < 0:
            value += axis_size
        if not 0 <= value < axis_size:
            raise ValueError("Native axis-scatter index would exceed the output shape")
    output = buffers["out"]
    output_end = output.data + output.count * ctypes.sizeof(TYPES[dtype])
    for name, buffer in buffers.items():
        end = buffer.data + buffer.count * ctypes.sizeof(TYPES[dtypes[name]])
        if name != "out" and max(buffer.data, output.data) < min(end, output_end):
            raise ValueError("Native axis-scatter output overlaps an input allocation")
    return {
        "operation": operation,
        "outputShape": output_shape,
        "updateShape": index_shape,
        "updateStrides": update_strides,
        "indexShape": index_shape,
        "indexStrides": index_strides,
        "updateContiguous": update_contiguous,
        "indexContiguous": index_contiguous,
        "axis": axis,
        "updateCount": math.prod(grid),
        "outputCount": logical_size,
        "maximumIndex": logical_size - 1,
    }
