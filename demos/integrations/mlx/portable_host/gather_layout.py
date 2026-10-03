"""Validate general-gather storage and index bounds before native submission."""

import ctypes
import itertools
import math
import re

MAX_ELEMENTS = 65535
TYPES = {
    "float32": ctypes.c_float,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    "bool_": ctypes.c_uint8,
}
METADATA = {
    "src_shape": "int32",
    "src_strides": "int64",
    "src_ndim": "uint64",
    "slice_sizes": "int32",
    "axes": "int32",
    "idx_shapes": "int32",
    "idx_strides": "int64",
    "idx_contigs": "bool_",
    "idx_ndim": "int32",
}


def signature(entry):
    match = re.fullmatch(
        r"gather(float32|int32|uint32|int64|uint64|bool_)"
        r"(int32|uint32|int64|uint64)_([0-9]+)_([0-9]+)_int",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native gather entry")
    dtype, index_dtype, count, ndim = match.groups()
    count, ndim = int(count), int(ndim)
    if not 1 <= count <= 10 or not 0 <= ndim <= 64:
        raise ValueError("Native gather specialization exceeds its bounds")
    if entry != f"gather{dtype}{index_dtype}_{count}_{ndim}_int":
        raise ValueError("Native gather entry must use canonical dimensions")
    return dtype, index_dtype, count, ndim


def values(buffer):
    dtype = buffer.dtype.decode("ascii")
    return list(
        ctypes.cast(buffer.data, ctypes.POINTER(TYPES[dtype] * buffer.count)).contents
    )


def validate(entry, buffers, logical_size, execution):
    dtype, index_dtype, count, ndim = signature(entry)
    dtypes = {
        **METADATA,
        "src": dtype,
        "out": dtype,
        **{f"idx{i}": index_dtype for i in range(count)},
    }
    if set(buffers) != set(dtypes) or not 0 < logical_size <= MAX_ELEMENTS:
        raise ValueError("Native gather buffers or logical size do not match")
    for name, buffer in buffers.items():
        if (
            not buffer.data
            or buffer.dtype != dtypes[name].encode("ascii")
            or not 0 < buffer.count <= MAX_ELEMENTS
            or buffer.output != int(name == "out")
        ):
            raise ValueError("Native gather buffer layout or direction is invalid")
    rank = buffers["src_shape"].count
    lengths = {
        "out": logical_size,
        "src_shape": rank,
        "src_strides": rank,
        "src_ndim": 1,
        "slice_sizes": rank,
        "axes": count,
        "idx_shapes": max(1, count * ndim),
        "idx_strides": max(1, count * ndim),
        "idx_contigs": count,
        "idx_ndim": 1,
    }
    if not 1 <= rank <= 64 or any(
        buffers[name].count != size for name, size in lengths.items()
    ):
        raise ValueError("Native gather metadata lengths do not match")
    if values(buffers["src_ndim"]) != [rank] or values(buffers["idx_ndim"]) != [ndim]:
        raise ValueError("Native gather ranks do not match their specialization")
    shape, strides, slices, axes = (
        values(buffers[name])
        for name in ("src_shape", "src_strides", "slice_sizes", "axes")
    )
    if (
        any(not 1 <= extent <= MAX_ELEMENTS for extent in shape)
        or any(not 0 <= stride <= MAX_ELEMENTS for stride in strides)
        or any(not 1 <= size <= extent for size, extent in zip(slices, shape))
        or len(set(axes)) != count
        or any(not 0 <= axis < rank for axis in axes)
    ):
        raise ValueError(
            "Native gather source shape, strides, slices or axes are invalid"
        )
    if (
        1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
        != buffers["src"].count
    ):
        raise ValueError("Native gather source span does not match its allocation")
    shapes, steps, contigs = (
        values(buffers[name]) for name in ("idx_shapes", "idx_strides", "idx_contigs")
    )
    index_shape = shapes[:ndim]
    if any(not 1 <= size <= MAX_ELEMENTS for size in index_shape):
        raise ValueError("Native gather index shape is invalid")
    grid = [
        index_shape[0] if ndim else 1,
        math.prod(index_shape[1:]),
        math.prod(slices),
    ]
    if math.prod(grid) != logical_size or execution != {
        "workgroupCount": grid,
        "workgroupSize": [1, 1, 1],
    }:
        raise ValueError("Native gather launch does not match its output shape")
    for i, axis in enumerate(axes):
        current_shape = shapes[i * ndim : (i + 1) * ndim]
        current_steps = steps[i * ndim : (i + 1) * ndim]
        if current_shape != index_shape or any(
            not 0 <= stride <= MAX_ELEMENTS for stride in current_steps
        ):
            raise ValueError("Native gather index layouts must have matching shapes")
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(index_shape, current_steps)
        )
        if span != buffers[f"idx{i}"].count:
            raise ValueError("Native gather index span does not match its allocation")
        dense = all(
            size == 1 or stride == math.prod(index_shape[j + 1 :])
            for j, (size, stride) in enumerate(zip(index_shape, current_steps))
        )
        if contigs[i] not in (0, 1) or (contigs[i] and not dense):
            raise ValueError("Native gather index contiguity flag is invalid")
        indices = values(buffers[f"idx{i}"])
        for coordinates in itertools.product(*(range(size) for size in index_shape)):
            location = sum(
                index * stride for index, stride in zip(coordinates, current_steps)
            )
            value = indices[location]
            if value < 0:
                value += shape[axis]
            if not 0 <= value <= shape[axis] - slices[axis]:
                raise ValueError("Native gather index would exceed the source shape")
    output = buffers["out"]
    output_end = output.data + output.count * ctypes.sizeof(TYPES[dtype])
    for name, buffer in buffers.items():
        end = buffer.data + buffer.count * ctypes.sizeof(TYPES[dtypes[name]])
        if name != "out" and max(buffer.data, output.data) < min(end, output_end):
            raise ValueError("Native gather output overlaps an input allocation")
    return {
        "sourceShape": shape,
        "sourceStrides": strides,
        "sliceSizes": slices,
        "axes": axes,
        "indexShape": index_shape,
        "indexStrides": steps,
        "indexContiguous": contigs,
        "sourceCount": buffers["src"].count,
        "outputCount": logical_size,
        "maximumIndex": max(buffer.count for buffer in buffers.values()) - 1,
    }
