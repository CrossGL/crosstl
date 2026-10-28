"""Validate MLX column plans and their input spans before native dispatch."""

import ctypes
import math

from demos.integrations.mlx.portable_host.reduction_packages import (
    COLUMN_ENTRIES,
    PARTITION_SUM_ENTRIES,
)

LOOPED = {
    "in": None,
    "out": None,
    "reduction_size": "uint64",
    "reduction_stride": "int64",
    "shape": "int32",
    "strides": "int64",
    "ndim": "int32",
    "reduce_shape": "int32",
    "reduce_strides": "int64",
    "reduce_ndim": "int32",
    "non_col_reductions": "uint64",
}
TWO_PASS = {**LOOPED, "out_size": "uint64"}
METADATA_TYPES = {
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}
ARRAYS = {"shape", "strides", "reduce_shape", "reduce_strides"}
# The largest input, output or logical size a column launch binds: the bound
# the host's matrix products use, far below the INT32_MAX at which reduce.cpp
# switches to 64-bit indexing. Each shape axis stays below 65536, each stride
# inside that bound, and each launch axis at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535


def signature(entry):
    return TWO_PASS if entry.startswith("col_reduce_2pass_") else LOOPED


def read_metadata(entry, entries, spec, buffers, logical_size):
    """Check one column launch's buffers and read its shape metadata.

    ``entries`` maps the family's entries to their element types and ``spec``
    the entry's buffers to their metadata types. Returns the metadata and the
    plan quantities ``(size, stride, total, outer, reduce_ndim)``.
    """
    if (
        entry not in entries
        or set(buffers) != set(spec)
        or type(logical_size) is not int
        or not 1 <= logical_size <= MAX_ELEMENTS
    ):
        raise ValueError("Native column reduction metadata does not match")
    for name, buffer in buffers.items():
        maximum = MAX_ELEMENTS if name in {"in", "out"} else 64 if name in ARRAYS else 1
        if (
            not buffer.dtype
            or buffer.dtype.decode("ascii") != (spec[name] or entries[entry])
            or not 1 <= buffer.count <= maximum
            or not buffer.data
            or buffer.output != int(name == "out")
        ):
            raise ValueError("Native column reduction buffer layout does not match")

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
        if dtype is not None and name not in ARRAYS
    }
    ndim, reduce_ndim = metadata["ndim"][0], metadata["reduce_ndim"][0]
    if not 0 <= ndim <= 64 or not 1 <= reduce_ndim <= 64:
        raise ValueError("Native column reduction rank exceeds its bounds")
    for prefix, rank in (("", ndim), ("reduce_", reduce_ndim)):
        if any(
            buffers[prefix + name].count != max(1, rank)
            for name in ("shape", "strides")
        ):
            raise ValueError("Native column reduction rank does not match its arrays")
        shape, strides = read(prefix + "shape"), read(prefix + "strides")
        metadata.update({prefix + "shape": shape, prefix + "strides": strides})
        if rank == 0:
            if shape != [0] or strides != [0]:
                raise ValueError("Native column reduction empty metadata must be zero")
        elif any(not 1 <= size <= 65535 for size in shape) or any(
            not 0 <= stride <= MAX_ELEMENTS for stride in strides
        ):
            raise ValueError(
                "Native column reduction shape or stride exceeds its bounds"
            )
    shape, strides = metadata["shape"][:ndim], metadata["strides"][:ndim]
    reductions, steps = metadata["reduce_shape"], metadata["reduce_strides"]
    size, stride = metadata["reduction_size"][0], metadata["reduction_stride"][0]
    non_columns = math.prod(reductions[:-1])
    outer = math.prod(shape)
    total = size * non_columns
    output_size = outer * stride
    if (
        not 1 <= size <= 65535
        or not 1 <= stride <= MAX_ELEMENTS
        or reductions[-1] != size
        or steps[-1] != stride
        or metadata["non_col_reductions"] != [non_columns]
        or total * output_size != logical_size
    ):
        raise ValueError(
            "Native column reduction shape does not match its logical size"
        )
    if outer > MAX_GROUPS:
        raise ValueError("Native column reduction launch exceeds 65535 workgroups")
    extent = stride + sum(
        (size - 1) * step for size, step in zip(shape + reductions, strides + steps)
    )
    if extent != buffers["in"].count:
        raise ValueError("Native column reduction view does not match its source span")
    return metadata, (size, stride, total, outer, reduce_ndim)


def template_dimension(reduce_ndim):
    """reduce.cpp get_kernel_reduce_ndim."""
    return 1 if reduce_ndim <= 1 else 2 if reduce_ndim == 2 else 5


def validate(entry, buffers, logical_size, execution):
    spec = signature(entry)
    metadata, (size, stride, total, outer, reduce_ndim) = read_metadata(
        entry, {**COLUMN_ENTRIES, **PARTITION_SUM_ENTRIES}, spec, buffers, logical_size
    )
    output_size = outer * stride
    two_pass = spec is TWO_PASS
    if buffers["out"].count != output_size * (32 if two_pass else 1):
        raise ValueError(
            "Native column reduction shape does not match its logical size"
        )
    if (
        total < 32
        or (stride < 32 and total >= 1024)
        or two_pass != (total > 256 and output_size // 32 < 1024)
        or (two_pass and metadata["out_size"] != [outer])
    ):
        raise ValueError("Native column reduction does not match the upstream plan")
    dimension = template_dimension(reduce_ndim)
    mode = "2pass" if two_pass else "looped"
    if not entry.startswith(f"col_reduce_{mode}_{dimension}_32_32_reduce_"):
        raise ValueError("Native column reduction template rank does not match")
    if outer * (32 if two_pass else 1) > MAX_GROUPS or -(-stride // 32) > MAX_GROUPS:
        raise ValueError("Native column reduction launch exceeds 65535 workgroups")
    if execution != {
        "workgroupCount": [(stride + 31) // 32, outer * (32 if two_pass else 1), 1],
        "workgroupSize": [256, 1, 1],
    }:
        raise ValueError(
            "Native column reduction launch does not match the upstream plan"
        )
    return metadata
