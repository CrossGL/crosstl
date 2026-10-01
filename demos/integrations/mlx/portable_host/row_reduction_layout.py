"""Check MLX row metadata and source address spans before native dispatch."""

import ctypes
import math

from demos.integrations.mlx.portable_host.reduction_packages import ROW_ENTRIES

SIMPLE = {"in": None, "out": None, "reduction_size": "uint64", "out_size": "int64"}
LOOPED = {
    "in": None,
    "out": None,
    "row_size": "int64",
    "non_row_reductions": "int64",
    "shape": "int32",
    "strides": "int64",
    "ndim": "int32",
    "reduce_shape": "int32",
    "reduce_strides": "int64",
    "reduce_ndim": "int32",
}
METADATA_TYPES = {
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}


def signature(entry):
    return SIMPLE if entry.startswith("row_reduce_simple_") else LOOPED


def width(row_size):
    if type(row_size) is not int or not 65 <= row_size <= 65535:
        raise ValueError("Row kernels require 65 to 65535 elements per row")
    if row_size <= 512:
        return 32
    if row_size <= 1024:
        return 128
    return min(1024, ((row_size + 127) // 128) * 32)


def validate(entry, buffers, logical_size, execution):
    spec = signature(entry)
    if set(buffers) != set(spec) or entry not in ROW_ENTRIES:
        raise ValueError("Native row reduction metadata does not match")
    for name, buffer in buffers.items():
        expected_type = spec[name] or ROW_ENTRIES[entry]
        if buffer.dtype.decode("ascii") != expected_type:
            raise ValueError("Native row reduction dtype does not match")
        maximum = (
            65535
            if name in {"in", "out"}
            else (
                64
                if name in {"shape", "strides", "reduce_shape", "reduce_strides"}
                else 1
            )
        )
        if not 1 <= buffer.count <= maximum or not buffer.data:
            raise ValueError("Native row reduction buffer shape does not match")

    def read(name):
        return list(
            ctypes.cast(
                buffers[name].data,
                ctypes.POINTER(METADATA_TYPES[spec[name]] * buffers[name].count),
            ).contents
        )

    metadata = {
        name: read(name)
        for name, dtype in spec.items()
        if dtype is not None
        and name not in {"shape", "strides", "reduce_shape", "reduce_strides"}
    }
    rows = buffers["out"].count
    row_size = metadata["reduction_size" if spec is SIMPLE else "row_size"][0]
    group_width = width(row_size)
    if spec is SIMPLE:
        if (
            rows < 32
            or metadata["out_size"] != [rows]
            or logical_size != rows * row_size
            or buffers["in"].count != logical_size
        ):
            raise ValueError("Simple row reduction does not match the upstream plan")
        grid = [1, (rows + 3) // 4, 1]
        # Source pointer construction may include inactive lanes beyond the last row.
        if buffers["in"].count + group_width * 4 > 131071:
            raise ValueError("Simple row reduction exceeds its index contract")
    else:
        ndim, reduce_ndim = metadata["ndim"][0], metadata["reduce_ndim"][0]
        if not 0 <= ndim <= 64 or not 0 <= reduce_ndim <= 64:
            raise ValueError("Native row reduction rank exceeds its bounds")
        for prefix, rank in (("", ndim), ("reduce_", reduce_ndim)):
            if buffers[prefix + "shape"].count != max(1, rank) or buffers[
                prefix + "strides"
            ].count != max(1, rank):
                raise ValueError("Native row reduction rank does not match its arrays")
            shape, strides = read(prefix + "shape"), read(prefix + "strides")
            metadata.update({prefix + "shape": shape, prefix + "strides": strides})
            if rank == 0:
                if shape != [0] or strides != [0]:
                    raise ValueError("Native row reduction empty metadata must be zero")
            elif any(not 1 <= size <= 65535 for size in shape) or any(
                not 0 <= stride <= 65535 for stride in strides
            ):
                raise ValueError(
                    "Native row reduction shape or stride exceeds its bounds"
                )
        shape, strides = metadata["shape"][:ndim], metadata["strides"][:ndim]
        reduce_shape, reduce_strides = (
            metadata["reduce_shape"][:reduce_ndim],
            metadata["reduce_strides"][:reduce_ndim],
        )
        non_rows = math.prod(reduce_shape)
        if (
            math.prod(shape) != rows
            or metadata["non_row_reductions"] != [non_rows]
            or rows * non_rows * row_size != logical_size
        ):
            raise ValueError(
                "Native row reduction shape does not match its logical size"
            )
        extent = row_size + sum(
            (size - 1) * stride
            for size, stride in zip(shape + reduce_shape, strides + reduce_strides)
        )
        if extent != buffers["in"].count:
            raise ValueError("Native row reduction view does not match its source span")
        dimension = 1 if reduce_ndim <= 1 else 2 if reduce_ndim == 2 else 5
        if not entry.startswith(f"row_reduce_looped_{dimension}_reduce_"):
            raise ValueError("Native row reduction template rank does not match")
        grid = [1, rows, 1]
    if execution != {"workgroupCount": grid, "workgroupSize": [group_width, 1, 1]}:
        raise ValueError("Native row reduction launch does not match the upstream plan")
    return metadata
