"""Check MLX row metadata and source address spans before native dispatch."""

import ctypes
import math

from demos.integrations.mlx.portable_host.reduction_packages import (
    ALL_ROW_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    ROW_MAX_ELEMENTS as MAX_ELEMENTS,
)
from demos.integrations.mlx.portable_host.small_row_packages import (
    ALL_SMALL_ROW_ENTRIES,
    SMALL_ROW_OUTPUTS,
)

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
ARRAY_METADATA = frozenset({"shape", "strides", "reduce_shape", "reduce_strides"})
# Inputs, outputs and logical sizes hold at most MAX_ELEMENTS elements; every
# shape axis stays below 65,536, every stride inside MAX_ELEMENTS and every
# launch axis at most MAX_GROUPS workgroups.
MAX_GROUPS = 65535


def signature(entry):
    return SIMPLE if entry.startswith("row_reduce_simple_") else LOOPED


def width(row_size):
    """The threadgroup width reduce.cpp picks for a row of ``row_size``.

    Looped and simple kernels stride through any row, so a row may span every
    element a source view holds.
    """
    if type(row_size) is not int or not 65 <= row_size <= MAX_ELEMENTS:
        raise ValueError("Row kernels require 65 to 2**28 - 1 elements per row")
    if row_size <= 512:
        return 32
    if row_size <= 1024:
        return 128
    return min(1024, ((row_size + 127) // 128) * 32)


def validate(entry, buffers, logical_size, execution):
    spec = signature(entry)
    entries = {**ALL_ROW_ENTRIES, **ALL_SMALL_ROW_ENTRIES}
    if set(buffers) != set(spec) or entry not in entries:
        raise ValueError("Native row reduction metadata does not match")
    for name, buffer in buffers.items():
        expected_type = spec[name] or (
            SMALL_ROW_OUTPUTS.get(entry, entries[entry])
            if name == "out"
            else entries[entry]
        )
        if buffer.dtype.decode("ascii") != expected_type:
            raise ValueError("Native row reduction dtype does not match")
        maximum = (
            MAX_ELEMENTS
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
        if dtype is not None and name not in ARRAY_METADATA
    }
    if spec is LOOPED:
        for prefix in ("", "reduce_"):
            rank = metadata[prefix + "ndim"][0]
            if not 0 <= rank <= 64:
                raise ValueError("Native row reduction rank exceeds its bounds")
            for suffix in ("shape", "strides"):
                if buffers[prefix + suffix].count != max(1, rank):
                    raise ValueError(
                        "Native row reduction rank does not match its arrays"
                    )
        metadata.update({name: read(name) for name in ARRAY_METADATA})
    return validate_metadata(
        entry,
        metadata,
        logical_size,
        buffers["in"].count,
        buffers["out"].count,
        execution,
    )


def validate_metadata(
    entry, metadata, logical_size, input_count, output_count, execution
):
    """Validate a detached row layout before cache lookup or source translation."""
    entries = {**ALL_ROW_ENTRIES, **ALL_SMALL_ROW_ENTRIES}
    if not isinstance(entry, str) or entry not in entries:
        raise ValueError("Unknown row reduction entry")
    spec = signature(entry)
    small = entry in ALL_SMALL_ROW_ENTRIES
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {name for name, dtype in spec.items() if dtype is not None}
        or any(
            type(value) is not int or not 1 <= value <= MAX_ELEMENTS
            for value in (logical_size, input_count, output_count)
        )
        or not isinstance(execution, dict)
        or any(
            not isinstance(values, list)
            or len(values) != 3
            or any(type(value) is not int or value <= 0 for value in values)
            for values in execution.values()
        )
    ):
        raise ValueError("Invalid row reduction layout")
    for name, values in metadata.items():
        dtype = spec[name]
        bits = 32 if dtype == "int32" else 64
        minimum = 0 if dtype == "uint64" else -(1 << (bits - 1))
        maximum = (1 << (bits if dtype == "uint64" else bits - 1)) - 1
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= (64 if name in ARRAY_METADATA else 1)
            or any(
                type(value) is not int or not minimum <= value <= maximum
                for value in values
            )
        ):
            raise ValueError("Invalid row reduction metadata values")
    metadata = {name: list(values) for name, values in metadata.items()}
    rows = output_count
    row_size = metadata["reduction_size" if spec is SIMPLE else "row_size"][0]
    if small and not 1 <= row_size <= 64:
        raise ValueError("Small-row kernels require 1 to 64 elements per row")
    group_width = 32 if small else width(row_size)
    exact_grid = None
    if spec is SIMPLE:
        if (
            rows < 32
            or metadata["out_size"] != [rows]
            or logical_size != rows * row_size
            or input_count != logical_size
        ):
            raise ValueError("Simple row reduction does not match the upstream plan")
        grid = [1, (rows + 3) // 4, 1]
        # Source pointer construction may include inactive lanes beyond the
        # last row; static GLSL packages assert these indexes inside the bound.
        if input_count + group_width * 4 > MAX_ELEMENTS:
            raise ValueError("Simple row reduction exceeds its index contract")
    else:
        ndim, reduce_ndim = metadata["ndim"][0], metadata["reduce_ndim"][0]
        if not 0 <= ndim <= 64 or not 0 <= reduce_ndim <= 64:
            raise ValueError("Native row reduction rank exceeds its bounds")
        for prefix, rank in (("", ndim), ("reduce_", reduce_ndim)):
            if len(metadata[prefix + "shape"]) != max(1, rank) or len(
                metadata[prefix + "strides"]
            ) != max(1, rank):
                raise ValueError("Native row reduction rank does not match its arrays")
            shape, strides = metadata[prefix + "shape"], metadata[prefix + "strides"]
            if rank == 0:
                if shape != [0] or strides != [0]:
                    raise ValueError("Native row reduction empty metadata must be zero")
            elif any(not 1 <= size <= 65535 for size in shape) or any(
                not 0 <= stride <= MAX_ELEMENTS for stride in strides
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
        if extent != input_count:
            raise ValueError("Native row reduction view does not match its source span")
        dimension = 1 if reduce_ndim <= 1 else 2 if reduce_ndim == 2 else 5
        family = "small" if small else "looped"
        if not entry.startswith(f"row_reduce_{family}_{dimension}_reduce_"):
            raise ValueError("Native row reduction template rank does not match")
        grid = [1, rows, 1]
        if small:
            scalar = (non_rows < 32 and row_size <= 8) or non_rows <= 8
            if scalar:
                group_width = min(rows, 1024)
                grid = [(rows + group_width - 1) // group_width, 1, 1]
                exact_grid = [rows, 1, 1]
            else:
                exact_grid = [32, rows, 1]
    if any(groups > MAX_GROUPS for groups in grid):
        raise ValueError("Native row reduction launch exceeds 65535 workgroups")
    expected = {"workgroupCount": grid, "workgroupSize": [group_width, 1, 1]}
    if exact_grid is not None:
        expected["threadGridSize"] = exact_grid
    if execution != expected:
        raise ValueError("Native row reduction launch does not match the upstream plan")
    return metadata
