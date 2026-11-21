"""Validate axis-scatter allocations, indices and launch geometry.

The pinned ``indexing.cpp`` instantiates ``scatter_axis`` for every MLX dtype
with replacement and additive updates. The host runs the 32-bit types whose
atomics every target provides: int32, uint32 and float32. Float32 outputs
are raw binary32 words; OpenGL keeps them in uint32 atomics
(``scatter_layout.atomic_storage_dtype``).

Storage follows the general gather's bound. Each launch keeps the portable
per-axis workgroup limit: MLX's grid is [post, index axis, pre], and the host
splits the index axis and the leading extents into launches whose views start
at the launch's first element (``launch_plan``).
"""

import ctypes
import itertools
import math
import re

from demos.integrations.mlx.portable_host.gather_axis_layout import slabs
from demos.integrations.mlx.portable_host.gather_layout import (
    MAX_ELEMENTS,
    MAX_STORAGE_ELEMENTS,
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
        r"scatter_axis(int32|uint32|float32)(int32|uint32|int64|uint64)_(none|sum)_int"
        r"(c|nc)(c|nc)",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native axis-scatter entry")
    dtype, index_dtype, operation, update, index = match.groups()
    return dtype, index_dtype, operation, update == "c", index == "c"


def launch_plan(index_shape, axis):
    """Return the host's launches as (leading slices, index-row slice) pairs.

    The kernel strides its output by the launch's x extent, so the trailing
    extent must fit one launch axis. Leading extents split into rectangular
    slabs of at most MAX_ELEMENTS positions; a launch that splits the index
    axis covers one leading position, because the contiguous specializations
    stride leading positions by the launch's y extent. This mirrors
    ``axis_launch_tiles`` in backend.cpp.
    """
    leading, rows = list(index_shape[:axis]), index_shape[axis]
    if math.prod(index_shape[axis + 1 :]) > MAX_ELEMENTS:
        raise ValueError("Native axis-scatter launch exceeds per-axis workgroup limits")
    limit = 1 if rows > MAX_ELEMENTS else MAX_ELEMENTS
    return [
        (slab, slice(row, min(rows, row + MAX_ELEMENTS)))
        for slab in slabs(leading, limit)
        for row in range(0, rows, MAX_ELEMENTS)
    ]


def validate(entry, buffers, logical_size, execution):
    dtype, index_dtype, operation, update_contiguous, index_contiguous = signature(
        entry
    )
    dtypes = {**METADATA, "upd": dtype, "indices": index_dtype, "out": dtype}
    if set(buffers) != set(dtypes) or not 0 < logical_size <= MAX_STORAGE_ELEMENTS:
        raise ValueError("Native axis-scatter buffers or output size do not match")
    for name, buffer in buffers.items():
        if (
            not buffer.data
            or buffer.dtype != dtypes[name].encode("ascii")
            or not 0 < buffer.count <= MAX_STORAGE_ELEMENTS
            or buffer.output != int(name == "out")
            or buffer.data % ctypes.alignment(TYPES[dtypes[name]])
            or buffer.data + buffer.count * ctypes.sizeof(TYPES[dtypes[name]])
            > 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
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
        or not 1 <= axis_size <= MAX_STORAGE_ELEMENTS
        or not 0 <= update_step <= MAX_STORAGE_ELEMENTS
        or not 0 <= index_step <= MAX_STORAGE_ELEMENTS
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
    if any(not 1 <= size <= MAX_STORAGE_ELEMENTS for size in shape) or any(
        not 0 <= step <= MAX_STORAGE_ELEMENTS for step in update_strides + index_strides
    ):
        raise ValueError("Native axis-scatter shape or strides are invalid")
    grid = execution.get("workgroupCount")
    # Storage is bounded as a whole; each launch axis keeps the portable limit.
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or any(type(size) is not int or not 1 <= size <= MAX_ELEMENTS for size in grid)
        or execution != {"workgroupCount": grid, "workgroupSize": [1, 1, 1]}
        or grid[0] != math.prod(shape[axis:])
        or grid[2] != math.prod(shape[:axis])
        or math.prod(grid) > MAX_STORAGE_ELEMENTS
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
