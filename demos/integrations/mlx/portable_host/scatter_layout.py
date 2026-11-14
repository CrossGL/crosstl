"""Validate general-scatter storage, slice bounds and upstream launch planning.

Storage follows the general gather's bound. Each launch keeps the per-axis
workgroup limit: the update slice runs along x and the host splits index rows
into launches of at most 65535 rows, each with MLX's work-per-thread choice
for its own rows.

As in MLX's ``Scatter::eval_gpu``, complex64 is accepted only for sums.
``Sum::atomic_update`` adds each element's real and imaginary float32 lanes
through a float view of the output, so the output binds as two 32-bit atomic
words per element and its word count must stay within the storage bound.
"""

import ctypes
import itertools
import math
import re

from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from demos.integrations.mlx.portable_host import bfloat_storage, half_storage
from demos.integrations.mlx.portable_host.gather_layout import (
    MAX_ELEMENTS,
    MAX_STORAGE_ELEMENTS,
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
# 32-bit atomic words per output element.
ATOMIC_LANES = {"int32": 1, "uint32": 1, "float32": 1, "complex64": 2}
# The pinned mlx_atomic<half> and mlx_atomic<bfloat16_t> are one atomic uint
# that packs two elements, lowest lane first (kernels/atomic.h uint_or_packed).
PACKED_TYPES = ("float16", "bfloat16")
PACKING = 2
# Host element types; binary16 and bfloat16 elements are raw 16-bit words.
ELEMENT_TYPES = {**TYPES, "float16": ctypes.c_uint16, "bfloat16": ctypes.c_uint16}
# An odd output's last word keeps this value in its unused high lane.
PADDING = {"float16": half_storage.GUARD[0], "bfloat16": bfloat_storage.GUARD[0]}
# The wire encodings of float carriers; integer storage has none.
ENCODINGS = {
    "float32": FLOAT32_BITS,
    "float16": FLOAT16_BITS,
    "bfloat16": BFLOAT16_BITS,
}


def atomic_storage_dtype(dtype, target):
    """Return the reflected physical type of the output atomic allocation.

    ``mlx_atomic<complex64_t>`` is MLX's generic ``atomic<uint>`` wrapper on
    every target, and the packed 16-bit atomics are one ``atomic<uint>`` word
    per two elements.
    """
    if (
        dtype in PACKED_TYPES
        or dtype == "complex64"
        or dtype == "float32"
        and target == "opengl"
    ):
        return "uint32"
    return dtype


def storage_count(dtype, count):
    """Return the number of atomic words that hold ``count`` output elements."""
    if dtype in PACKED_TYPES:
        return -(-count // PACKING)
    return count * ATOMIC_LANES[dtype]


def pack_lanes(lanes, padding):
    """Pack 16-bit elements into atomic words; an odd tail pads its high lane."""
    lanes = list(lanes)
    if any(type(lane) is not int or not 0 <= lane <= 0xFFFF for lane in lanes) or (
        type(padding) is not int or not 0 <= padding <= 0xFFFF
    ):
        raise ValueError("Packed scatter lanes must be unsigned 16-bit words")
    if len(lanes) % PACKING:
        lanes.append(padding)
    return [low | (high << 16) for low, high in zip(lanes[::2], lanes[1::2])]


def carriers(dtype, lanes, target):
    """Return the native carriers of 16-bit update words, without rounding."""
    storage = half_storage if dtype == "float16" else bfloat_storage
    return storage.pack(lanes, target)


def unpack_lanes(words, count):
    """Return the ``count`` packed elements and the padding lanes after them."""
    words = list(words)
    if any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in words):
        raise ValueError("Packed scatter words must be unsigned 32-bit values")
    if len(words) != storage_count("float16", count):
        raise ValueError("Packed scatter word count does not match its elements")
    lanes = [lane for word in words for lane in (word & 0xFFFF, word >> 16)]
    return lanes[:count], lanes[count:]


def signature(entry):
    match = re.fullmatch(
        r"scatter(?:(int32|uint32|float32|float16|bfloat16)(int32|uint32|int64|uint64)_"
        r"(none|sum|prod|min|max)|(complex64)(int32|uint32|int64|uint64)_(sum))_"
        r"([1-9]|10)_updc_(true|false)_nwork(1|4|8|16|32)_int",
        entry,
    )
    if match is None:
        raise ValueError("Unsupported native general-scatter entry")
    groups = match.groups()
    dtype, index_dtype, operation = groups[0:3] if groups[0] else groups[3:6]
    count, contiguous, work = groups[6:]
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
    if (
        set(buffers) != set(dtypes)
        or not 0 < storage_count(dtype, logical_size) <= MAX_STORAGE_ELEMENTS
    ):
        raise ValueError("Native scatter buffers or output size do not match")
    for name, buffer in buffers.items():
        if (
            not buffer.data
            or buffer.dtype != dtypes[name].encode("ascii")
            or not 0 < buffer.count <= MAX_STORAGE_ELEMENTS
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
        or not 1 <= index_size <= MAX_STORAGE_ELEMENTS
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
        any(not 1 <= size <= MAX_STORAGE_ELEMENTS for size in shape + update_shape)
        or any(
            not 0 <= step <= MAX_STORAGE_ELEMENTS for step in strides + update_strides
        )
        or strides != [math.prod(shape[i + 1 :]) for i in range(rank)]
        or math.prod(shape) != logical_size
        or math.prod(index_shape) != index_size
        or math.prod(slices) != slice_size
        or index_size * slice_size > MAX_STORAGE_ELEMENTS
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
    if (
        work != work_per_thread(index_rank, index_size, logical_size)
        or (index_size + work - 1) // work > MAX_ELEMENTS
    ):
        raise ValueError("Native scatter launch exceeds per-axis workgroup limits")
    if execution != {
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
            not 0 <= step <= MAX_STORAGE_ELEMENTS for step in current_steps
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
    output_end = output.data + output.count * ctypes.sizeof(ELEMENT_TYPES[dtype])
    for name, buffer in buffers.items():
        end = buffer.data + buffer.count * ctypes.sizeof(ELEMENT_TYPES[dtypes[name]])
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
        # The atomic words of a complex64 output are its largest index space;
        # a packed 16-bit output indexes no further than its elements.
        "maximumIndex": (
            max(
                buffer.count * (ATOMIC_LANES.get(dtype, 1) if name == "out" else 1)
                for name, buffer in buffers.items()
            )
            - 1
        ),
    }
