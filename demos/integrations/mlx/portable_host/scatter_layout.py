"""Validate general-scatter storage, slice bounds and upstream launch planning."""

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
    "upd_shape": "int32",
    "upd_strides": "int64",
    "upd_ndim": "uint64",
    "upd_size": "uint64",
    "out_shape": "int32",
    "out_strides": "int64",
    "out_ndim": "uint64",
    "axes": "int32",
    "idx_shapes": "int32",
    "idx_strides": "int64",
    "idx_contigs": "bool_",
    "idx_ndim": "int32",
    "idx_size": "uint64",
}
ATOMIC_TYPES = {"int32": "int", "uint32": "uint", "float32": "float"}


def signature(entry):
    match = re.fullmatch(
        r"scatter(int32|uint32|float32)(int32|uint32|int64|uint64)_(none|sum|prod|min|max)_"
        r"([1-9]|10)_updc_(true|false)_nwork(1|4|8|16|32)_int",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native general-scatter entry")
    dtype, index_dtype, operation, count, contiguous, work = match.groups()
    return dtype, index_dtype, int(count), operation, contiguous == "true", int(work)


def work_per_thread(index_rank, index_size, output_size):
    ratio = index_size // output_size
    if index_rank <= 1 or ratio < 1:
        return 1
    return 4 if ratio <= 4 else 8 if ratio < 16 else 16 if ratio < 32 else 32


def validate(entry, buffers, logical_size, execution):
    dtype, index_dtype, count, operation, contiguous, work = signature(entry)
    dtypes = {
        **METADATA,
        "updates": dtype,
        "out": dtype,
        **{f"idx{i}": index_dtype for i in range(count)},
    }
    if set(buffers) != set(dtypes) or not 0 < logical_size <= MAX_ELEMENTS:
        raise ValueError("Native scatter buffers or output size do not match")
    for name, buffer in buffers.items():
        if (
            not buffer.data
            or buffer.dtype != dtypes[name].encode("ascii")
            or not 0 < buffer.count <= MAX_ELEMENTS
            or buffer.output != int(name == "out")
        ):
            raise ValueError("Native scatter buffer layout or direction is invalid")
    scalars = ("upd_ndim", "upd_size", "out_ndim", "idx_ndim", "idx_size")
    if any(buffers[name].count != 1 for name in scalars):
        raise ValueError("Native scatter scalar metadata length is invalid")
    update_rank, slice_size, rank, index_rank, index_size = (
        values(buffers[name])[0] for name in scalars
    )
    if (
        not 1 <= rank <= 64
        or not 0 <= index_rank <= 64
        or update_rank != rank + index_rank
        or update_rank > 64
        or not 1 <= index_size <= MAX_ELEMENTS
        or not 1 <= slice_size <= MAX_ELEMENTS
    ):
        raise ValueError("Native scatter ranks or extents are invalid")
    lengths = {
        "out": logical_size,
        "upd_shape": update_rank,
        "upd_strides": update_rank,
        "out_shape": rank,
        "out_strides": rank,
        "axes": count,
        "idx_shapes": max(1, count * index_rank),
        "idx_strides": max(1, count * index_rank),
        "idx_contigs": count + int(index_rank == 0),
    }
    if any(buffers[name].count != size for name, size in lengths.items()):
        raise ValueError("Native scatter metadata lengths do not match")
    shape, strides, update_shape, update_strides, axes, shapes, steps, contigs = (
        values(buffers[name])
        for name in (
            "out_shape",
            "out_strides",
            "upd_shape",
            "upd_strides",
            "axes",
            "idx_shapes",
            "idx_strides",
            "idx_contigs",
        )
    )
    slices = update_shape[index_rank:]
    index_shape = update_shape[:index_rank]
    if (
        any(not 1 <= size <= MAX_ELEMENTS for size in shape + update_shape)
        or any(not 0 <= step <= MAX_ELEMENTS for step in strides + update_strides)
        or strides != [math.prod(shape[i + 1 :]) for i in range(rank)]
        or math.prod(shape) != logical_size
        or math.prod(index_shape) != index_size
        or math.prod(slices) != slice_size
        or index_size * slice_size > MAX_ELEMENTS
        or any(size > extent for size, extent in zip(slices, shape))
        or len(set(axes)) != count
        or any(not 0 <= axis < rank for axis in axes)
    ):
        raise ValueError("Native scatter shapes, strides or axes are invalid")
    span = 1 + sum(
        (size - 1) * step for size, step in zip(update_shape, update_strides)
    )
    if span != buffers["updates"].count or (
        contiguous
        and any(
            size > 1 and step != math.prod(update_shape[i + 1 :])
            for i, (size, step) in enumerate(zip(update_shape, update_strides))
        )
    ):
        raise ValueError("Native scatter update span or contiguity is invalid")
    if work != work_per_thread(index_rank, index_size, logical_size) or execution != {
        "workgroupCount": [slice_size, (index_size + work - 1) // work, 1],
        "workgroupSize": [1, 1, 1],
    }:
        raise ValueError("Native scatter launch does not match its specialization")
    if any(flag not in (0, 1) for flag in contigs):
        raise ValueError("Native scatter index contiguity flag is invalid")
    for i, axis in enumerate(axes):
        current_shape = shapes[i * index_rank : (i + 1) * index_rank]
        current_steps = steps[i * index_rank : (i + 1) * index_rank]
        if current_shape != index_shape or any(
            not 0 <= step <= MAX_ELEMENTS for step in current_steps
        ):
            raise ValueError("Native scatter index layouts must have matching shapes")
        span = 1 + sum(
            (size - 1) * step for size, step in zip(index_shape, current_steps)
        )
        if span != buffers[f"idx{i}"].count or (
            contigs[i]
            and any(
                size > 1 and step != math.prod(index_shape[j + 1 :])
                for j, (size, step) in enumerate(zip(index_shape, current_steps))
            )
        ):
            raise ValueError("Native scatter index span or contiguity is invalid")
        indices = values(buffers[f"idx{i}"])
        for coordinates in itertools.product(*(range(size) for size in index_shape)):
            location = sum(
                index * step for index, step in zip(coordinates, current_steps)
            )
            value = indices[location]
            if value < 0:
                value += shape[axis]
            if not 0 <= value <= shape[axis] - slices[axis]:
                raise ValueError("Native scatter index would exceed its output slice")
    output = buffers["out"]
    output_end = output.data + output.count * ctypes.sizeof(TYPES[dtype])
    for name, buffer in buffers.items():
        end = buffer.data + buffer.count * ctypes.sizeof(TYPES[dtypes[name]])
        if name != "out" and max(buffer.data, output.data) < min(end, output_end):
            raise ValueError("Native scatter output overlaps an input allocation")
    return {
        "operation": operation,
        "outputShape": shape,
        "outputStrides": strides,
        "updateShape": update_shape,
        "updateStrides": update_strides,
        "updateContiguous": contiguous,
        "sliceSizes": slices,
        "axes": axes,
        "indexShape": index_shape,
        "indexStrides": steps,
        "indexContiguous": contigs,
        "workPerThread": work,
        "updateCount": index_size * slice_size,
        "outputCount": logical_size,
        "maximumIndex": max(buffer.count for buffer in buffers.values()) - 1,
    }
