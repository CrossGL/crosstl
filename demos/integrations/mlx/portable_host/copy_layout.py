"""Validate bounded copy metadata before reading or submitting source storage."""

import ctypes
import itertools
import math

DTYPES = {
    "src": "uint32",
    "dst": "uint32",
    "src_shape": "int32",
    "src_strides": "int64",
    "dst_strides": "int64",
    "ndim": "int32",
    "src_offset": "int64",
    "dst_offset": "int64",
}


MAX_DESTINATION_ELEMENTS = 2**31 - 1
MAX_GROUPS = 65535
INOUT = 2


def destination_indices(metadata):
    """Yield each destination element index in row-major coordinate order.

    The index of coordinates ``c`` is ``destinationOffset + sum(c[i] * s[i])``
    over the strides the shape has (a missing stride counts as zero). Rows of
    the last axis are produced as arithmetic ranges, so a copy of millions of
    elements does not sum strides per element.
    """
    shape = list(metadata["shape"])
    if not shape:
        yield metadata["destinationOffset"]
        return
    if any(size <= 0 for size in shape):
        return
    strides = list(metadata["destinationStrides"])[: len(shape)]
    strides += [0] * (len(shape) - len(strides))
    bases = [metadata["destinationOffset"]]
    for size, stride in zip(shape[:-1], strides[:-1]):
        bases = [base + index * stride for base in bases for index in range(size)]
    length, stride = shape[-1], strides[-1]
    for base in bases:
        if stride:
            yield from range(base, base + length * stride, stride)
        else:
            yield from itertools.repeat(base, length)


def launch_regions(shape, source_strides, destination_strides, destination_offset):
    """Mirror the host's split of one copy into portable exact launches.

    The copy kernel guards only its X axis, so the host runs a copy whose
    launch would exceed 65535 groups on an axis as sub-copies along the array
    axes, in ascending order: the last axis in chunks of 65535 element pairs,
    then rows, then the leading axes that fold into Z. Each region is
    ``(shape, source_origin, destination_offset)``, where ``source_origin`` is
    the element offset of the region origin from the source view.
    """
    regions = []

    def append(region_shape, origin, offset):
        rank = len(region_shape)
        axis, chunk = rank - 1, 2 * MAX_GROUPS
        if (region_shape[-1] + 1) // 2 <= MAX_GROUPS:
            axis, chunk = rank - 2, MAX_GROUPS
            if region_shape[-2] <= MAX_GROUPS:
                if math.prod(region_shape[:-2]) <= MAX_GROUPS:
                    regions.append((list(region_shape), origin, offset))
                    return
                axis = next(i for i, size in enumerate(region_shape) if size > 1)
                inner = math.prod(region_shape[axis + 1 : -2])
                chunk = MAX_GROUPS // inner if inner <= MAX_GROUPS else 1
        for first in range(0, region_shape[axis], chunk):
            part = list(region_shape)
            part[axis] = min(chunk, region_shape[axis] - first)
            append(
                part,
                origin + first * source_strides[axis],
                offset + first * destination_strides[axis],
            )

    append(list(shape), 0, destination_offset)
    return regions


def validate(buffers, logical_size, *, dtype="uint32"):
    if (
        type(logical_size) is not int
        or not 0 < logical_size <= MAX_DESTINATION_ELEMENTS
    ):
        raise ValueError("Native copy logical size exceeds signed index bounds")
    if dtype not in {"uint32", "bool_", "int64", "uint64", "float16", "bfloat16"}:
        raise ValueError("Unsupported native copy storage dtype")
    rank = buffers["src_shape"].count
    if not 2 <= rank <= 64:
        raise ValueError("Native copy rank must be between 2 and 64")
    for name, buffer in buffers.items():
        expected = (
            logical_size
            if name == "dst"
            else rank if name in {"src_shape", "src_strides", "dst_strides"} else 1
        )
        if name == "dst":
            if not logical_size <= buffer.count <= MAX_DESTINATION_ELEMENTS:
                raise ValueError("Native copy destination span exceeds its bounds")
        elif name == "src":
            if not 0 < buffer.count <= MAX_DESTINATION_ELEMENTS:
                raise ValueError("Native copy source span exceeds signed index bounds")
        elif buffer.count != expected:
            raise ValueError("Native copy metadata shape does not match")
        if buffer.dtype.decode("ascii") != (
            dtype if name in {"src", "dst"} else DTYPES[name]
        ):
            raise ValueError("Native copy metadata dtype does not match")

    def values(name, ctype):
        buffer = buffers[name]
        return list(
            ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents
        )

    if values("ndim", ctypes.c_int32) != [rank]:
        raise ValueError("Native copy rank does not match its metadata")
    shape = values("src_shape", ctypes.c_int32)
    strides = values("src_strides", ctypes.c_int64)
    if any(size <= 0 for size in shape) or math.prod(shape) != logical_size:
        raise ValueError("Native copy shape does not match the logical size")
    if any(abs(stride) > MAX_DESTINATION_ELEMENTS for stride in strides):
        raise ValueError("Native copy source stride exceeds signed index bounds")
    offset = values("src_offset", ctypes.c_int64)[0]
    extents = [(size - 1) * stride for size, stride in zip(shape, strides)]
    low = offset + sum(min(extent, 0) for extent in extents)
    high = offset + sum(max(extent, 0) for extent in extents)
    if (
        low != 0
        or high != buffers["src"].count - 1
        or sum(abs(extent) for extent in extents) >= MAX_DESTINATION_ELEMENTS
    ):
        raise ValueError("Native copy addresses do not match the uploaded source span")
    destination_strides = values("dst_strides", ctypes.c_int64)
    destination_offset = values("dst_offset", ctypes.c_int64)[0]
    if any(abs(stride) > MAX_DESTINATION_ELEMENTS for stride in destination_strides):
        raise ValueError("Native copy destination stride exceeds signed index bounds")
    extents = [(size - 1) * stride for size, stride in zip(shape, destination_strides)]
    if (
        destination_offset < 0
        or destination_offset + sum(min(extent, 0) for extent in extents) < 0
        or destination_offset + sum(max(extent, 0) for extent in extents)
        >= buffers["dst"].count
        or sum(abs(extent) for extent in extents) >= MAX_DESTINATION_ELEMENTS
    ):
        raise ValueError("Native copy destination addresses exceed its allocation")
    itemsize = {"bool_": 1, "float16": 2, "bfloat16": 2, "int64": 8, "uint64": 8}.get(
        dtype, 4
    )
    source, destination = buffers["src"], buffers["dst"]
    if max(source.data, destination.data) < min(
        source.data + source.count * itemsize,
        destination.data + destination.count * itemsize,
    ):
        raise ValueError("Native copy source and destination allocations overlap")
    metadata = {
        "shape": shape,
        "sourceStrides": strides,
        "destinationStrides": destination_strides,
        "sourceOffset": offset,
        "destinationOffset": destination_offset,
        "sourceCount": source.count,
        "destinationCount": destination.count,
        "preserveDestination": destination.output == INOUT,
        "workgroupCount": [(shape[-1] + 1) // 2, shape[-2], math.prod(shape[:-2])],
    }
    if any(not 1 <= count <= 65535 for count in metadata["workgroupCount"]):
        raise ValueError("Native copy launch exceeds 65535 groups per axis")
    if len(set(destination_indices(metadata))) != logical_size:
        raise ValueError("Native copy destination elements overlap")
    return metadata
