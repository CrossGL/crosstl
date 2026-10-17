"""Validate the pinned axis-gather metadata before native execution."""

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
        r"gather_axis(float32|int32|uint32|int64|uint64|bool_)"
        r"(int32|uint32|int64|uint64)_int(c|nc)(c|nc)",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native axis-gather entry")
    dtype, index_dtype, source, index = match.groups()
    return dtype, index_dtype, source == "c", index == "c"


def validate(entry, buffers, logical_size, execution):
    dtype, index_dtype, source_contiguous, index_contiguous = signature(entry)
    dtypes = {**METADATA, "src": dtype, "indices": index_dtype, "out": dtype}
    if set(buffers) != set(dtypes) or not 0 < logical_size <= MAX_ELEMENTS:
        raise ValueError("Native axis-gather buffers or logical size do not match")
    for name, buffer in buffers.items():
        if (
            not buffer.data
            or buffer.dtype != dtypes[name].encode("ascii")
            or not 0 < buffer.count <= MAX_ELEMENTS
            or buffer.output != int(name == "out")
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
        or not 1 <= axis_size <= MAX_ELEMENTS
        or not 0 <= source_step <= MAX_ELEMENTS
        or not 0 <= index_step <= MAX_ELEMENTS
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
    if any(not 1 <= size <= MAX_ELEMENTS for size in shape) or any(
        not 0 <= step <= MAX_ELEMENTS for step in source_strides + index_strides
    ):
        raise ValueError("Native axis-gather shape or strides are invalid")
    outer = math.prod(shape)
    if logical_size % outer:
        raise ValueError("Native axis-gather output does not match its shape")
    index_size = logical_size // outer
    grid = [math.prod(shape[axis:]), index_size, math.prod(shape[:axis])]
    if index_size < 1 or execution != {
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
