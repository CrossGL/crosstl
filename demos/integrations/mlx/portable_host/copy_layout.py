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
INOUT = 2


def destination_indices(metadata):
    for coordinates in itertools.product(*(range(size) for size in metadata["shape"])):
        yield metadata["destinationOffset"] + sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, metadata["destinationStrides"])
        )


def validate(buffers, logical_size, *, dtype="uint32"):
    if dtype not in {"uint32", "bool_", "int64", "uint64"}:
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
            if not 0 < buffer.count <= 65535:
                raise ValueError("Native copy source span exceeds 65535")
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
    if any(abs(stride) > 65535 for stride in strides):
        raise ValueError("Native copy source stride exceeds 65535")
    offset = values("src_offset", ctypes.c_int64)[0]
    extents = [(size - 1) * stride for size, stride in zip(shape, strides)]
    low = offset + sum(min(extent, 0) for extent in extents)
    high = offset + sum(max(extent, 0) for extent in extents)
    if low != 0 or high != buffers["src"].count - 1:
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
        or sum(abs(extent) for extent in extents) > MAX_DESTINATION_ELEMENTS
    ):
        raise ValueError("Native copy destination addresses exceed its allocation")
    itemsize = 1 if dtype == "bool_" else 8 if dtype in {"int64", "uint64"} else 4
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
    if len(set(destination_indices(metadata))) != logical_size:
        raise ValueError("Native copy destination elements overlap")
    return metadata
