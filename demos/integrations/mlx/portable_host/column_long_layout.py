"""Validate MLX ``col_reduce_longcolumn`` launches before native dispatch.

reduce.cpp's ``strided_reduce_general_dispatch`` sends a column plan whose
reduction stride is below 32 and that reduces 1,024 or more elements per
output to ``strided_reduce_longcolumn``. That pass appends the column axis to
the reduction loop, so ``reduce_shape`` ends with ``reduction_size`` as for
``col_reduce_small``, and splits the reduced rows over ``outer_blocks``
threadgroups along ``z``: 32 blocks, or 128 from 32,768 reduced elements. Each
threadgroup holds one thread per column along ``x`` and up to 32 rows along
``y``, and block ``z`` writes its partial results ``out_size`` elements into
an intermediate of ``outer_blocks`` copies of the output. A looped column pass
over that intermediate's first axis then produces the result.

The kernel reads its threadgroup size at run time, so every column count is
its own translation. The host's buffers hold at most
``column_reduction_layout.MAX_ELEMENTS`` elements, far below ``INT32_MAX``, so
it never selects the ``_large`` (64-bit index) variants.
"""

from demos.integrations.mlx.portable_host.column_reduction_layout import (
    MAX_GROUPS,
    TWO_PASS,
    read_metadata,
    template_dimension,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    ENTRIES,
    HALF_FLOAT_ENTRIES,
)

ENTRY_PREFIX = "col_reduce_longcolumn_"
# reduce.metal instantiates the long-column kernel beside col_reduce_small for
# every type; these are the float32, int32, uint32, Boolean and 16-bit float
# ones the looped and two-pass column passes also run. 16-bit floats
# accumulate in the element type, as instantiate_sum_prod and
# instantiate_min_max declare them.
LONG_COLUMN_ENTRIES = {
    f"{ENTRY_PREFIX}{dimension}_reduce_" + entry.removeprefix("all_reduce_"): dtype
    for dimension in (1, 2, 5)
    for entry, dtype in {**ENTRIES, **HALF_FLOAT_ENTRIES}.items()
}
# strided_reduce_general_dispatch's thresholds.
MAX_STRIDE = 31
MIN_TOTAL = 1024
WIDE_TOTAL = 32768
MAX_ROWS = 32
SEPARATOR = "@"


def signature(entry):
    if entry not in LONG_COLUMN_ENTRIES:
        raise ValueError("Unsupported long column reduction entry")
    return TWO_PASS


def outer_blocks(total):
    """strided_reduce_longcolumn's number of partial results per output."""
    return 128 if total >= WIDE_TOTAL else 32


def geometry(stride, total):
    """strided_reduce_longcolumn's threadgroup size for one plan."""
    return stride, min(MAX_ROWS, -(-total // outer_blocks(total)))


def variant(entry, workgroup):
    """The cache key of one entry at one threadgroup size."""
    signature(entry)
    columns, rows, depth = workgroup
    if (
        any(type(value) is not int for value in workgroup)
        or not 1 <= columns <= MAX_STRIDE
        or not 1 <= rows <= MAX_ROWS
        or depth != 1
    ):
        raise ValueError("Long column reduction threadgroup exceeds its bounds")
    return f"{entry}{SEPARATOR}{columns}x{rows}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid long column reduction variant")
    entry, separator, size = key.partition(SEPARATOR)
    columns, _, rows = size.partition("x")
    if not separator or not columns.isdigit() or not rows.isdigit():
        raise ValueError("Invalid long column reduction variant")
    workgroup = (int(columns), int(rows), 1)
    if variant(entry, workgroup) != key:
        raise ValueError("Invalid long column reduction variant")
    return entry, workgroup


def validate(entry, buffers, logical_size, execution):
    """Check one launch against strided_reduce_longcolumn; return its metadata."""
    spec = signature(entry)
    metadata, (_, stride, total, outer, reduce_ndim) = read_metadata(
        entry, LONG_COLUMN_ENTRIES, spec, buffers, logical_size
    )
    if stride > MAX_STRIDE or total < MIN_TOTAL:
        raise ValueError("Native column reduction does not match the upstream plan")
    blocks = outer_blocks(total)
    if (
        buffers["out"].count != outer * stride * blocks
        or metadata["out_size"] != [outer * stride]
    ):
        raise ValueError(
            "Native column reduction shape does not match its logical size"
        )
    if not entry.startswith(f"{ENTRY_PREFIX}{template_dimension(reduce_ndim)}_reduce_"):
        raise ValueError("Native column reduction template rank does not match")
    columns, rows = geometry(stride, total)
    if outer > MAX_GROUPS:
        raise ValueError("Native column reduction launch exceeds 65535 workgroups")
    if execution != {
        "workgroupCount": [outer, 1, blocks],
        "workgroupSize": [columns, rows, 1],
    }:
        raise ValueError(
            "Native column reduction launch does not match the upstream plan"
        )
    return metadata
