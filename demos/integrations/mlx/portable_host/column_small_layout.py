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
"""

from demos.integrations.mlx.portable_host.column_reduction_layout import (
    LOOPED,
    MAX_GROUPS,
    read_metadata,
    template_dimension,
)
from demos.integrations.mlx.portable_host.reduction_packages import ENTRIES

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
N_READS = 4
MAX_LANES = 32
MAX_ROWS = 8
# Apple GPUs report 1024 threads per threadgroup for the reduction kernels,
# so this bound never constrains MAX_ROWS.
MAX_THREADS = 1024
SEPARATOR = "@"


def signature(entry):
    if entry not in ALL_SMALL_COLUMN_ENTRIES:
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
        entry, ALL_SMALL_COLUMN_ENTRIES, spec, buffers, logical_size
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
