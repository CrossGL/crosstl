"""Validate the pinned axis-gather metadata before native execution.

Storage follows the general gather's bound. Each launch keeps the portable
per-axis workgroup limit: MLX's grid is [post, index axis, pre], and the host
splits it into launches whose views start at the launch's first element
(``launch_plan``).
"""

import ctypes
import itertools
import math
import re

from demos.integrations.mlx.portable_host.gather_layout import (
    MAX_ELEMENTS,
    MAX_STORAGE_ELEMENTS,
    TYPES,
    VALUE_PATTERN,
    values,
)

METADATA = {
    "shape": "int32",
    "src_strides": "int64",
    "idx_strides": "int64",
    "ndim": "uint64",
    "axis": "int32",
    "axis_size": "int32",
    "src_ax_stride": "uint64",
    "idx_ax_stride": "uint64",
}


def signature(entry):
    match = re.fullmatch(
        rf"gather_axis({VALUE_PATTERN})(int32|uint32|int64|uint64)_int(c|nc)(c|nc)",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native axis-gather entry")
    dtype, index_dtype, source, index = match.groups()
    return dtype, index_dtype, source == "c", index == "c"


def slabs(extents, limit):
    """Split row-major positions over ``extents`` into rectangular slabs.

    Each slab holds at most ``limit`` positions and is contiguous in
    row-major order: it fixes the axes before its split axis, takes a range
    of the split axis, and spans every later axis. This mirrors
    ``axis_launch_tiles`` in backend.cpp.
    """
    if not extents:
        return [()]
    split, suffix = 0, math.prod(extents[1:])
    while suffix > limit:
        split += 1
        suffix //= extents[split]
    extent, result, first = limit // suffix, [], 0
    while first < math.prod(extents):
        start, remaining = [0] * len(extents), first
        for i in reversed(range(len(extents))):
            start[i] = remaining % extents[i]
            remaining //= extents[i]
        count = min(extent, extents[split] - start[split])
        result.append(
            tuple(slice(start[i], start[i] + 1) for i in range(split))
            + (slice(start[split], start[split] + count),)
            + (slice(None),) * (len(extents) - split - 1)
        )
        first += count * suffix
    return result


def launch_plan(index_shape, axis):
    """Return the host's launches as slices of the index (and output) array.

    The kernel strides its output by the launch's x extent. When the trailing
    extent fits one launch axis, leading extents split into slabs and the
    index axis into row ranges; a launch that splits the index axis covers
    one leading position, because the contiguous specializations stride
    leading positions by the launch's y extent. A wider trailing extent runs
    per leading position and index row, as slabs of the trailing axes
    through the strided specializations. Launches follow the output's
    row-major order, as ``GatherAxis::eval_gpu`` in backend.cpp issues them.
    """
    leading, rows = list(index_shape[:axis]), index_shape[axis]
    trailing = list(index_shape[axis + 1 :])
    if math.prod(trailing) <= MAX_ELEMENTS:
        limit = 1 if rows > MAX_ELEMENTS else MAX_ELEMENTS
        return [
            slab
            + (slice(row, min(rows, row + MAX_ELEMENTS)),)
            + (slice(None),) * len(trailing)
            for slab in slabs(leading, limit)
            for row in range(0, rows, MAX_ELEMENTS)
        ]
    return [
        tuple(slice(i, i + 1) for i in position) + (slice(row, row + 1),) + slab
        for position in itertools.product(*(range(size) for size in leading))
        for row in range(rows)
        for slab in slabs(trailing, MAX_ELEMENTS)
    ]


def wide(index_shape, axis):
    """Whether the host runs the strided specializations for this launch."""
    return math.prod(index_shape[axis + 1 :]) > MAX_ELEMENTS


def validate(entry, buffers, logical_size, execution):
    dtype, index_dtype, source_contiguous, index_contiguous = signature(entry)
    dtypes = {**METADATA, "src": dtype, "indices": index_dtype, "out": dtype}
    if set(buffers) != set(dtypes) or not 0 < logical_size <= MAX_STORAGE_ELEMENTS:
        raise ValueError("Native axis-gather buffers or logical size do not match")
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
            raise ValueError("Native axis-gather buffer layout or direction is invalid")
    for name in ("ndim", "axis", "axis_size", "src_ax_stride", "idx_ax_stride"):
        if buffers[name].count != 1:
            raise ValueError("Native axis-gather scalar metadata length is invalid")
    ndim, axis, axis_size, source_step, index_step = (
        values(buffers[name])[0]
        for name in ("ndim", "axis", "axis_size", "src_ax_stride", "idx_ax_stride")
    )
    if (
        not 0 <= ndim < 64
        or not 0 <= axis <= ndim
        or not 1 <= axis_size <= MAX_STORAGE_ELEMENTS
        or not 0 <= source_step <= MAX_STORAGE_ELEMENTS
        or not 0 <= index_step <= MAX_STORAGE_ELEMENTS
        or buffers["out"].count != logical_size
        or any(
            buffers[name].count != max(1, ndim)
            for name in ("shape", "src_strides", "idx_strides")
        )
    ):
        raise ValueError("Native axis-gather metadata bounds or lengths are invalid")
    shape, source_strides, index_strides = (
        values(buffers[name])[:ndim] for name in ("shape", "src_strides", "idx_strides")
    )
    if any(not 1 <= size <= MAX_STORAGE_ELEMENTS for size in shape) or any(
        not 0 <= step <= MAX_STORAGE_ELEMENTS for step in source_strides + index_strides
    ):
        raise ValueError("Native axis-gather shape or strides are invalid")
    outer = math.prod(shape)
    if logical_size % outer:
        raise ValueError("Native axis-gather output does not match its shape")
    index_size = logical_size // outer
    grid = [math.prod(shape[axis:]), index_size, math.prod(shape[:axis])]
    # Storage is bounded as a whole; each launch axis keeps the portable limit.
    if any(not 1 <= dimension <= MAX_ELEMENTS for dimension in grid):
        raise ValueError("Native axis-gather launch exceeds per-axis workgroup limits")
    if execution != {
        "workgroupCount": grid,
        "workgroupSize": [1, 1, 1],
    }:
        raise ValueError("Native axis-gather launch does not match its output shape")
    source_shape, index_shape = shape.copy(), shape.copy()
    source_shape.insert(axis, axis_size)
    index_shape.insert(axis, index_size)
    source_strides.insert(axis, source_step)
    index_strides.insert(axis, index_step)
    for name, extents, strides, contiguous in (
        ("src", source_shape, source_strides, source_contiguous),
        ("indices", index_shape, index_strides, index_contiguous),
    ):
        span = 1 + sum((size - 1) * step for size, step in zip(extents, strides))
        if span != buffers[name].count:
            raise ValueError(
                "Native axis-gather storage span does not match its allocation"
            )
        if contiguous and any(
            size > 1 and step != math.prod(extents[i + 1 :])
            for i, (size, step) in enumerate(zip(extents, strides))
        ):
            raise ValueError("Native axis-gather contiguity specialization is invalid")
    indices = values(buffers["indices"])
    for coordinates in itertools.product(*(range(size) for size in index_shape)):
        location = sum(i * step for i, step in zip(coordinates, index_strides))
        value = indices[location]
        if value < 0:
            value += axis_size
        if not 0 <= value < axis_size:
            raise ValueError("Native axis-gather index would exceed the source shape")
    output = buffers["out"]
    output_end = output.data + output.count * ctypes.sizeof(TYPES[dtype])
    for name, buffer in buffers.items():
        end = buffer.data + buffer.count * ctypes.sizeof(TYPES[dtypes[name]])
        if name != "out" and max(buffer.data, output.data) < min(end, output_end):
            raise ValueError("Native axis-gather output overlaps an input allocation")
    return {
        "sourceShape": source_shape,
        "sourceStrides": source_strides,
        "indexShape": index_shape,
        "indexStrides": index_strides,
        "sourceContiguous": source_contiguous,
        "indexContiguous": index_contiguous,
        "axis": axis,
        "sourceCount": buffers["src"].count,
        "outputCount": logical_size,
        "maximumIndex": max(buffer.count for buffer in buffers.values()) - 1,
    }
