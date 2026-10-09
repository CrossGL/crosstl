"""Check dense updates and unique strided destinations before native dispatch."""

import ctypes
import math

from demos.integrations.mlx.portable_host.copy_layout import INOUT, destination_indices
from demos.integrations.mlx.portable_host.packages import SLICE_UPDATE_TYPES

MAX_STORAGE_ELEMENTS = 2**28 - 33
MAX_INDEX = 2**31 - 1
TYPES = {
    "bool_": ctypes.c_uint8,
    "float32": ctypes.c_float,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}

DTYPES = {
    "updates": None,
    "out": None,
    "update_shape": "int32",
    "update_strides": "int64",
    "update_ndim": "int32",
    "update_size": "int64",
    "output_strides": "int64",
    "output_offset": "int64",
}


def validate(buffers, logical_size, *, dtype):
    if dtype not in SLICE_UPDATE_TYPES or set(buffers) != set(DTYPES):
        raise ValueError("Native slice update signature does not match")
    if type(logical_size) is not int or not 0 < logical_size <= 65535:
        raise ValueError("Native slice update size exceeds 65535")
    rank = buffers["update_shape"].count
    if not 1 <= rank <= 64:
        raise ValueError("Native slice update rank must be between 1 and 64")
    for name, buffer in buffers.items():
        expected = (
            logical_size
            if name == "updates"
            else (
                rank
                if name in {"update_shape", "update_strides", "output_strides"}
                else 1
            )
        )
        if name == "out":
            if not logical_size <= buffer.count <= MAX_STORAGE_ELEMENTS:
                raise ValueError("Native slice update output span exceeds its bounds")
        elif buffer.count != expected:
            raise ValueError("Native slice update metadata shape does not match")
        if (
            not buffer.data
            or buffer.dtype.decode("ascii") != (DTYPES[name] or dtype)
            or buffer.output != (INOUT if name == "out" else 0)
        ):
            raise ValueError(
                "Native slice update buffer type or direction does not match"
            )
        scalar = TYPES[DTYPES[name] or dtype]
        address_limit = 1 << (ctypes.sizeof(ctypes.c_void_p) * 8)
        if (
            buffer.data % ctypes.alignment(scalar)
            or buffer.data + buffer.count * ctypes.sizeof(scalar) > address_limit
        ):
            raise ValueError("Native slice update buffer address is invalid")

    def values(name, ctype):
        buffer = buffers[name]
        return list(
            ctypes.cast(buffer.data, ctypes.POINTER(ctype * buffer.count)).contents
        )

    if values("update_ndim", ctypes.c_int32) != [rank]:
        raise ValueError("Native slice update rank does not match its metadata")
    if values("update_size", ctypes.c_int64) != [logical_size]:
        raise ValueError("Native slice update size does not match its metadata")
    shape = values("update_shape", ctypes.c_int32)
    if any(size <= 0 for size in shape) or math.prod(shape) != logical_size:
        raise ValueError("Native slice update shape does not match its size")
    stride, dense = 1, []
    for size in reversed(shape):
        dense.insert(0, stride)
        stride *= size
    if values("update_strides", ctypes.c_int64) != dense:
        raise ValueError("Native slice update inputs must be dense row-major arrays")
    strides = values("output_strides", ctypes.c_int64)
    offset = values("output_offset", ctypes.c_int64)[0]
    extents = [(size - 1) * stride for size, stride in zip(shape, strides)]
    if (
        any(abs(stride) > MAX_INDEX for stride in strides)
        or sum(abs(extent) for extent in extents) >= MAX_INDEX
        or offset < 0
        or offset + sum(min(extent, 0) for extent in extents) < 0
        or offset + sum(max(extent, 0) for extent in extents) >= buffers["out"].count
    ):
        raise ValueError("Native slice update destination exceeds its allocation")
    metadata = {
        "shape": shape,
        "destinationStrides": strides,
        "destinationOffset": offset,
        "destinationCount": buffers["out"].count,
    }
    if len(set(destination_indices(metadata))) != logical_size:
        raise ValueError("Native slice update destinations overlap")
    size = ctypes.sizeof(TYPES[dtype])
    source, destination = buffers["updates"], buffers["out"]
    if max(source.data, destination.data) < min(
        source.data + source.count * size,
        destination.data + destination.count * size,
    ):
        raise ValueError("Native slice update source and destination overlap")
    return metadata
