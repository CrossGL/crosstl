"""Validate MLX ``col_reduce_small`` launches before native dispatch.

reduce.cpp's ``strided_reduce_general_dispatch`` sends column reductions with
fewer than 32 reduced elements per output to ``strided_reduce_small``. It
appends the column axis to the reduction loop, so ``reduce_shape`` ends with
``reduction_size``, and sizes the threadgroup to the columns: up to 32 lanes
of four columns each along ``x`` and up to 8 rows along ``y``. The kernel reads
the threadgroup size at run time, so every geometry is its own translation.
The host's buffers hold at most ``column_reduction_layout.MAX_ELEMENTS``
elements, far below ``INT32_MAX``, so it never selects the ``_large`` (64-bit
index) variants.

``PARTITION_SUM_ENTRIES`` are the 16-bit float sums over one axis that
quantized.cpp's split-K products run over their partial products. Their
accumulator is the element type, as in reduce.metal: each addition rounds.

``WIDE_SMALL_COLUMN_ENTRIES`` are the remaining instantiations whose storage
is 8 or 16 bits wide: int64, uint64 and complex64 numeric reductions, 16-bit
float sums and products of any rank, and 16-bit float extrema. They keep the
element type as their accumulator, as ``remap_reduce_types`` does, and the
runtime dispatches them through ``wide_column_dispatch`` with exact storage
words. The launch checks below are the same for both sets.
"""

import ctypes
from types import SimpleNamespace

from demos.integrations.mlx.portable_host.column_reduction_layout import (
    ARRAYS,
    LOOPED,
    MAX_GROUPS,
    METADATA_TYPES,
    read_metadata,
    template_dimension,
)
from demos.integrations.mlx.portable_host.reduction_packages import ENTRIES
from demos.integrations.mlx.portable_host.wide_reduction_packages import (
    OPERATIONS as WIDE_OPERATIONS,
)

ENTRY_PREFIX = "col_reduce_small_"
SMALL_COLUMN_ENTRIES = {
    f"{ENTRY_PREFIX}{dimension}_reduce_" + entry.removeprefix("all_reduce_"): dtype
    for dimension in (1, 2, 5)
    for entry, dtype in ENTRIES.items()
}
PARTITION_SUM_ENTRIES = {
    f"{ENTRY_PREFIX}1_reduce_sum{dtype}": dtype for dtype in ("float16", "bfloat16")
}
ALL_SMALL_COLUMN_ENTRIES = {**SMALL_COLUMN_ENTRIES, **PARTITION_SUM_ENTRIES}
WIDE_OPERATIONS_AND_EXTREMA = {
    **WIDE_OPERATIONS,
    **{
        operation: (*WIDE_OPERATIONS[operation], "float16", "bfloat16")
        for operation in ("min", "max")
    },
}
WIDE_SMALL_COLUMN_ENTRIES = {
    entry: dtype
    for entry, dtype in (
        (f"{ENTRY_PREFIX}{dimension}_reduce_{operation}{dtype}", dtype)
        for dimension in (1, 2, 5)
        for operation, dtypes in WIDE_OPERATIONS_AND_EXTREMA.items()
        for dtype in dtypes
    )
    if entry not in PARTITION_SUM_ENTRIES
}
# Every entry whose launch this module checks.
LAUNCH_ENTRIES = {**ALL_SMALL_COLUMN_ENTRIES, **WIDE_SMALL_COLUMN_ENTRIES}
N_READS = 4
MAX_LANES = 32
MAX_ROWS = 8
# Apple GPUs report 1024 threads per threadgroup for the reduction kernels,
# so this bound never constrains MAX_ROWS.
MAX_THREADS = 1024
SEPARATOR = "@"


def signature(entry):
    if entry not in LAUNCH_ENTRIES:
        raise ValueError("Unsupported small column reduction entry")
    return LOOPED


def geometry(stride, total):
    """strided_reduce_small's threadgroup size for one plan."""
    blocks = -(-stride // N_READS)
    lanes = min(blocks, MAX_LANES)
    return lanes, min(MAX_ROWS, min(MAX_THREADS // lanes, total))


def variant(entry, workgroup):
    """The cache key of one entry at one threadgroup size."""
    signature(entry)
    lanes, rows, depth = workgroup
    if (
        any(type(value) is not int for value in workgroup)
        or not 1 <= lanes <= MAX_LANES
        or not 1 <= rows <= MAX_ROWS
        or depth != 1
    ):
        raise ValueError("Small column reduction threadgroup exceeds its bounds")
    return f"{entry}{SEPARATOR}{lanes}x{rows}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid small column reduction variant")
    entry, separator, size = key.partition(SEPARATOR)
    lanes, _, rows = size.partition("x")
    if not separator or not lanes.isdigit() or not rows.isdigit():
        raise ValueError("Invalid small column reduction variant")
    workgroup = (int(lanes), int(rows), 1)
    if variant(entry, workgroup) != key:
        raise ValueError("Invalid small column reduction variant")
    return entry, workgroup


def validate(entry, buffers, logical_size, execution):
    """Check one launch against strided_reduce_small and return its metadata."""
    spec = signature(entry)
    metadata, (size, stride, total, outer, reduce_ndim) = read_metadata(
        entry, LAUNCH_ENTRIES, spec, buffers, logical_size
    )
    if buffers["out"].count != outer * stride:
        raise ValueError(
            "Native column reduction shape does not match its logical size"
        )
    if total >= 32:
        raise ValueError("Native column reduction does not match the upstream plan")
    if not entry.startswith(f"{ENTRY_PREFIX}{template_dimension(reduce_ndim)}_reduce_"):
        raise ValueError("Native column reduction template rank does not match")
    lanes, rows = geometry(stride, total)
    blocks = -(-stride // N_READS)
    if -(-blocks // lanes) > MAX_GROUPS:
        raise ValueError("Native column reduction launch exceeds 65535 workgroups")
    if execution != {
        "workgroupCount": [-(-blocks // lanes), outer, 1],
        "workgroupSize": [lanes, rows, 1],
    }:
        raise ValueError(
            "Native column reduction launch does not match the upstream plan"
        )
    return metadata


def validate_metadata(
    entry, metadata, logical_size, input_count, output_count, execution
):
    """Validate a detached column layout, as ``validate`` checks a launch.

    The metadata values are range-checked, bound again as the buffers the
    native host passes and run through the same plan checks.
    """
    spec = signature(entry)
    names = {name for name, kind in spec.items() if kind is not None}
    if (
        not isinstance(metadata, dict)
        or set(metadata) != names
        or any(type(value) is not int for value in (input_count, output_count))
    ):
        raise ValueError("Invalid small column reduction layout")
    storage, buffers = {}, {}
    for name, kind in spec.items():
        if kind is None:
            # Element storage is not read here; one byte stands in for it.
            words, count = (ctypes.c_uint8 * 1)(), (
                input_count if name == "in" else output_count
            )
        else:
            values = metadata[name]
            bits = 32 if kind == "int32" else 64
            low = 0 if kind == "uint64" else -(1 << (bits - 1))
            high = (1 << (bits if kind == "uint64" else bits - 1)) - 1
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= (64 if name in ARRAYS else 1)
                or any(
                    type(value) is not int or not low <= value <= high
                    for value in values
                )
            ):
                raise ValueError("Invalid small column reduction metadata values")
            words, count = (METADATA_TYPES[kind] * len(values))(*values), len(values)
        storage[name] = words
        buffers[name] = SimpleNamespace(
            dtype=(kind or LAUNCH_ENTRIES[entry]).encode("ascii"),
            count=count,
            data=ctypes.addressof(words),
            output=int(name == "out"),
        )
    return validate(entry, buffers, logical_size, execution)
