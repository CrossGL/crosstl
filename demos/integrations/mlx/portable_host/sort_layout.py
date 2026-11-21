"""Validate MLX merge-sort launches before native dispatch.

The pinned sort.metal instantiates, for every sortable dtype, contiguous
(``c``) and strided (``nc``) single-block sorts and argsorts at 32 to 512
threads (256 for the 64-bit types), and the multi-block ``sort_mbsort``,
``partition_mbsort`` and ``merge_mbsort`` stages at 512 threads (256).
``Sort``, ``ArgSort``, ``Partition`` and ``ArgPartition::eval_gpu`` all call
``gpu_merge_sort``: it picks ``bn`` from the sorted axis's length at four
values per thread, sorts each row in one threadgroup when the row fits in
``bn * 4`` values, and otherwise sorts ``ceil(size / (bn * 4))`` tiles per row
and merges tile pairs, partitioning each pass first, until one run remains.
Every check below is that plan's; a launch outside it is refused before any
package is translated.

``partition_mbsort`` strides over its threadgroup size, which
``multi_block_sort`` sets to ``min(n_blocks + 1, 1024)``, so each width is
its own package; every other entry runs at its ``bn``. Keys are
``<entry>@<width>``. The host binds views with non-negative strides (it sorts
a negative-stride view from a dense copy), every bound extent stays far below
the signed 32-bit offsets OpenGL indexes with, and ``index_assertions`` names
the 64-bit offsets the host bounds for that target.
"""

import ctypes
from types import SimpleNamespace

SOURCE = "mlx/backend/metal/kernels/sort.metal"
# instantiate_block_sort_bn and instantiate_multi_block_sort_base.
BASE_DTYPES = (
    "uint8",
    "uint16",
    "uint32",
    "int8",
    "int16",
    "int32",
    "float16",
    "float32",
    "bfloat16",
)
# instantiate_block_sort_long and instantiate_multi_block_sort_long.
LONG_DTYPES = ("uint64", "int64", "complex64")
DTYPES = BASE_DTYPES + LONG_DTYPES
INDEX_TYPE = "uint32"
SCALAR_TYPE = "int32"
N_PER_THREAD = 4
MAX_PARTITION_WIDTH = 1024
STAGES = ("sort", "partition", "merge")


def block_widths(dtype):
    """The ``bn`` values sort.metal instantiates single-block kernels at."""
    return (32, 64, 128, 256, 512) if dtype in BASE_DTYPES else (32, 64, 128, 256)


def multi_block_width(dtype):
    """The ``bn`` sort.metal instantiates the multi-block stages at."""
    return 512 if dtype in BASE_DTYPES else 256


def _entries():
    entries = {}
    for dtype in DTYPES:
        for bn in block_widths(dtype):
            for layout in ("c", "nc"):
                name = f"{layout}_block_sort_{dtype}_{dtype}_bn{bn}_tn{N_PER_THREAD}"
                entries[name] = ("block", layout, False, dtype, bn)
                name = (
                    f"{layout}arg_block_sort_{dtype}_{INDEX_TYPE}_bn{bn}"
                    f"_tn{N_PER_THREAD}"
                )
                entries[name] = ("block", layout, True, dtype, bn)
        bn = multi_block_width(dtype)
        for stage in STAGES:
            name = f"{stage}_mbsort_{dtype}_{INDEX_TYPE}_bn{bn}_tn{N_PER_THREAD}"
            entries[name] = (stage, None, True, dtype, bn)
    return entries


# entry -> (stage, layout, argsort, value type, bn); "block" stages carry the
# "c" or "nc" layout.
ENTRIES = _entries()
# Each stage's kernel parameters in buffer order, and the ones it writes.
OPERANDS = {
    ("block", "c"): (
        "inp",
        "out",
        "size_sorted_axis",
        "in_stride_sorted_axis",
        "out_stride_sorted_axis",
        "in_stride_segment_axis",
        "out_stride_segment_axis",
    ),
    ("block", "nc"): (
        "inp",
        "out",
        "size_sorted_axis",
        "in_stride_sorted_axis",
        "out_stride_sorted_axis",
        "nc_dim",
        "nc_shape",
        "in_nc_strides",
        "out_nc_strides",
    ),
    ("sort", None): (
        "inp",
        "out_vals",
        "out_idxs",
        "size_sorted_axis",
        "stride_sorted_axis",
        "nc_dim",
        "nc_shape",
        "nc_strides",
    ),
    ("partition", None): (
        "block_partitions",
        "dev_vals",
        "dev_idxs",
        "size_sorted_axis",
        "merge_tiles",
        "n_blocks",
    ),
    ("merge", None): (
        "block_partitions",
        "dev_vals_in",
        "dev_idxs_in",
        "dev_vals_out",
        "dev_idxs_out",
        "size_sorted_axis",
        "merge_tiles",
        "num_tiles",
    ),
}
OUTPUTS = {
    "block": ("out",),
    "sort": ("out_vals", "out_idxs"),
    "partition": ("block_partitions",),
    "merge": ("dev_vals_out", "dev_idxs_out"),
}
ARRAYS = {
    "nc_shape": "int32",
    "in_nc_strides": "int64",
    "out_nc_strides": "int64",
    "nc_strides": "int64",
}
VALUES = frozenset(("inp", "dev_vals", "dev_vals_in", "dev_vals_out", "out_vals"))
INDICES = frozenset(
    ("block_partitions", "dev_idxs", "dev_idxs_in", "dev_idxs_out", "out_idxs")
)
# Element storage; every other parameter is ``int`` metadata or a rank array.
ELEMENTS = VALUES | INDICES | {"out"}
METADATA_TYPES = {"int32": ctypes.c_int32, "int64": ctypes.c_int64}
MAX_RANK = 64
# The bounds the host's other row families use: every bound buffer and stride
# stays far below the INT32_MAX OpenGL's signed offsets allow, and one launch
# axis carries at most 65535 workgroups.
MAX_ELEMENTS = 2**28 - 1
MAX_GROUPS = 65535
SIGNED_MAX = 2**31 - 1
SEPARATOR = "@"
# single_block_sort's segment stride when every unsorted axis has extent one.
UNUSED_SEGMENT_STRIDE = SIGNED_MAX


def signature(entry):
    """``(stage, layout, argsort, value type, bn)`` of a sort entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unsupported sort entry") from error


def operand_types(entry):
    """Every kernel parameter's element type, in buffer order."""
    stage, layout, argsort, dtype, _ = signature(entry)
    types = {}
    for name in OPERANDS[stage, layout]:
        if name in VALUES:
            types[name] = dtype
        elif name in INDICES:
            types[name] = INDEX_TYPE
        elif name == "out":
            types[name] = INDEX_TYPE if argsort else dtype
        else:
            types[name] = ARRAYS.get(name, SCALAR_TYPE)
    return types


def outputs(entry):
    """The parameters ``entry`` writes."""
    return OUTPUTS[signature(entry)[0]]


def launch(size, dtype):
    """``gpu_merge_sort``'s ``(bn, n_blocks)`` for a sorted axis of ``size``."""
    if type(size) is not int or not 1 <= size <= MAX_ELEMENTS or dtype not in DTYPES:
        raise ValueError("Sorted axis must be non-empty and bounded")
    potential = (size + N_PER_THREAD - 1) // N_PER_THREAD
    bn = 32
    for threshold, threads in ((256, 512), (128, 256), (64, 128), (32, 64)):
        if potential > threshold:
            bn = threads
            break
    if bn == 512 and dtype in LONG_DTYPES:
        bn = 256
    per_block = bn * N_PER_THREAD
    return bn, (size + per_block - 1) // per_block


def entry_name(stage, dtype, size, argsort=False, contiguous=True):
    """The entry ``gpu_merge_sort`` selects for one stage of one axis length."""
    bn, n_blocks = launch(size, dtype)
    if (stage == "block") != (n_blocks == 1):
        raise ValueError("Sort stage does not match the sorted axis length")
    if stage == "block":
        layout = "c" if contiguous else "nc"
        if argsort:
            entry = f"{layout}arg_block_sort_{dtype}_{INDEX_TYPE}"
        else:
            entry = f"{layout}_block_sort_{dtype}_{dtype}"
    else:
        entry = f"{stage}_mbsort_{dtype}_{INDEX_TYPE}"
    entry += f"_bn{bn}_tn{N_PER_THREAD}"
    signature(entry)
    return entry


def width(entry, n_blocks):
    """The threadgroup width ``entry`` runs at for ``n_blocks`` tiles per row."""
    stage, _, _, _, bn = signature(entry)
    if stage == "partition":
        return min(n_blocks + 1, MAX_PARTITION_WIDTH)
    return bn


def variant(entry, size):
    """The cache key of one entry at one threadgroup width."""
    stage, _, _, _, bn = signature(entry)
    valid = 3 <= size <= MAX_PARTITION_WIDTH if stage == "partition" else size == bn
    if type(size) is not int or not valid:
        raise ValueError("Sort threadgroup width does not match its kernel")
    return f"{entry}{SEPARATOR}{size}"


def split_variant(key):
    """``(entry, workgroup)`` of a cache key ``variant`` built."""
    if not isinstance(key, str):
        raise ValueError("Invalid sort variant")
    entry, separator, size = key.partition(SEPARATOR)
    if not separator or not size.isdigit() or variant(entry, int(size)) != key:
        raise ValueError("Invalid sort variant")
    return entry, (int(size), 1, 1)


def merge_levels(n_blocks):
    """``merge_tiles`` of every pass ``multi_block_sort`` runs, in order."""
    levels, tiles = [], 2
    while tiles // 2 < n_blocks:
        levels.append(tiles)
        tiles *= 2
    return levels


def index_assertions(entry):
    """The OpenGL 64-bit offsets the host bounds to ``[0, INT32_MAX]``.

    The strided kernels advance their pointers by ``elem_to_loc`` row offsets
    and the sorted axis's stride; ``plan`` bounds every such offset by the
    bound span.
    """
    stage, layout, _, _, _ = signature(entry)
    if stage == "block" and layout == "nc":
        return (
            "inp + in_block_idx",
            "out_ + out_block_idx",
            "inp + i * in_stride_sorted_axis",
            "out_ + i * out_stride_sorted_axis",
        )
    if stage == "sort":
        return ("inp + block_idx", "inp + idx * stride_sorted_axis")
    return ()


# OpenGL cannot prove the absolute tile index of a threadgroup view passed to
# a helper (merge partitions are data-dependent binary searches). MLX's merge
# sort keeps every access inside its ``bn * 4`` tile; these host
# preconditions state that range for each helper view.
_MERGE_VIEWS = (
    ("BlockMergeSort_*__merge_partition", ("As", "Bs")),
    ("BlockMergeSort_*__merge_step", ("As", "Bs")),
)
_ARG_MERGE_VIEWS = (
    ("BlockMergeSort_*__merge_partition", ("As", "Bs")),
    ("BlockMergeSort_*__merge_step", ("As", "Bs", "As_idx", "Bs_idx")),
)


def workgroup_views(entry):
    """``(function, parameters)`` of every threadgroup view OpenGL bounds."""
    stage, _, argsort, _, _ = signature(entry)
    if stage == "block":
        if argsort:
            return (
                ("KernelMergeSort_*__block_sort", ("tgp_vals", "tgp_idxs")),
            ) + _ARG_MERGE_VIEWS
        return (("KernelMergeSort_*__block_sort", ("tgp_vals",)),) + _MERGE_VIEWS
    if stage == "sort":
        return (
            ("KernelMultiBlockMergeSort_*__block_sort", ("tgp_vals", "tgp_idxs")),
        ) + _ARG_MERGE_VIEWS
    if stage == "merge":
        return _ARG_MERGE_VIEWS
    return ()


def _rows(size, logical_size):
    if (
        type(logical_size) is not int
        or not 1 <= logical_size <= MAX_ELEMENTS
        or logical_size % size
    ):
        raise ValueError("Sort launch does not cover whole rows")
    rows = logical_size // size
    if rows > MAX_GROUPS:
        raise ValueError("Sort launch exceeds its group limit")
    return rows


def _strides(values):
    if any(not 0 <= value <= MAX_ELEMENTS for value in values):
        raise ValueError("Sort strides must be non-negative and bounded")


def _row_layout(metadata, rows, strides, placeholder):
    """Check an ``nc`` row shape; returns the furthest row offset per stride set."""
    rank = metadata["nc_dim"]
    if not 0 <= rank <= MAX_RANK:
        raise ValueError("Sort rank exceeds its bounds")
    names = ("nc_shape", *strides)
    if any(len(metadata[name]) != max(rank, 1) for name in names):
        raise ValueError("Sort metadata ranks do not match")
    if rank == 0:
        if metadata["nc_shape"] != [0] or any(
            metadata[name] != [placeholder] for name in strides
        ):
            raise ValueError("Sort rank-zero placeholders differ from upstream")
        if rows != 1:
            raise ValueError("Sort rows do not match their shape")
        return [0] * len(strides)
    shape = metadata["nc_shape"]
    count = 1
    for extent in shape:
        if not 1 <= extent <= MAX_GROUPS:
            raise ValueError("Sort row shape exceeds its bounds")
        count *= extent
    if count != rows:
        raise ValueError("Sort rows do not match their shape")
    furthest = []
    for name in strides:
        _strides(metadata[name])
        furthest.append(
            sum((extent - 1) * stride for extent, stride in zip(shape, metadata[name]))
        )
    return furthest


def _require_counts(counts, expected):
    if any(counts[name] != count for name, count in expected.items()):
        raise ValueError("Sort buffer extents do not match the upstream plan")


def _plan_block(entry, metadata, counts, logical_size):
    _, layout, _, dtype, bn = signature(entry)
    size = metadata["size_sorted_axis"]
    if launch(size, dtype) != (bn, 1):
        raise ValueError("Sort kernel does not match its sorted axis length")
    rows = _rows(size, logical_size)
    in_sorted = metadata["in_stride_sorted_axis"]
    out_sorted = metadata["out_stride_sorted_axis"]
    _strides((in_sorted, out_sorted))
    if layout == "c":
        segments = []
        for name in ("in_stride_segment_axis", "out_stride_segment_axis"):
            stride = metadata[name]
            if not (rows == 1 and stride == UNUSED_SEGMENT_STRIDE):
                _strides((stride,))
            segments.append((rows - 1) * stride)
        in_row, out_row = segments
        span = in_row + (size - 1) * in_sorted + 1
        valid = span <= counts["inp"]
    else:
        in_row, out_row = _row_layout(
            metadata, rows, ("in_nc_strides", "out_nc_strides"), 0
        )
        span = in_row + (size - 1) * in_sorted + 1
        valid = span == counts["inp"]
    if not valid or counts["inp"] > MAX_ELEMENTS:
        raise ValueError("Sort source span does not match its view")
    if counts["out"] != logical_size or out_row + (size - 1) * out_sorted >= (
        logical_size
    ):
        raise ValueError("Sort output does not match its rows")
    return {"workgroupCount": [1, rows, 1], "workgroupSize": [bn, 1, 1]}


def _plan_sort(entry, metadata, counts, logical_size):
    _, _, _, dtype, bn = signature(entry)
    size = metadata["size_sorted_axis"]
    expected, n_blocks = launch(size, dtype)
    if expected != bn or n_blocks < 2:
        raise ValueError("Sort kernel does not match its sorted axis length")
    rows = _rows(size, logical_size)
    if rows > 1 and n_blocks + 1 > MAX_PARTITION_WIDTH:
        raise ValueError("Multi-block sorts of several rows exceed 1023 tiles")
    stride = metadata["stride_sorted_axis"]
    _strides((stride,))
    (row,) = _row_layout(metadata, rows, ("nc_strides",), 1)
    if counts["inp"] != row + (size - 1) * stride + 1:
        raise ValueError("Sort source span does not match its view")
    _require_counts(counts, {"out_vals": logical_size, "out_idxs": logical_size})
    return {"workgroupCount": [n_blocks, rows, 1], "workgroupSize": [bn, 1, 1]}


def _plan_pass(entry, metadata, counts, logical_size):
    stage, _, _, dtype, bn = signature(entry)
    size = metadata["size_sorted_axis"]
    expected, n_blocks = launch(size, dtype)
    tiles = metadata["n_blocks" if stage == "partition" else "num_tiles"]
    if expected != bn or n_blocks < 2 or tiles != n_blocks:
        raise ValueError("Sort kernel does not match its sorted axis length")
    if metadata["merge_tiles"] not in merge_levels(n_blocks):
        raise ValueError("Sort merge pass is not one upstream runs")
    rows = _rows(size, logical_size)
    if rows > 1 and n_blocks + 1 > MAX_PARTITION_WIDTH:
        # partition_mbsort strides rows by its threadgroup width while
        # merge_mbsort reads n_blocks + 1 partitions per row.
        raise ValueError("Multi-block sorts of several rows exceed 1023 tiles")
    names = (
        ("dev_vals", "dev_idxs")
        if stage == "partition"
        else ("dev_vals_in", "dev_idxs_in", "dev_vals_out", "dev_idxs_out")
    )
    _require_counts(
        counts,
        {
            "block_partitions": rows * (n_blocks + 1),
            **{name: logical_size for name in names},
        },
    )
    if stage == "partition":
        return {
            "workgroupCount": [1, rows, 1],
            "workgroupSize": [width(entry, n_blocks), 1, 1],
        }
    return {"workgroupCount": [n_blocks, rows, 1], "workgroupSize": [bn, 1, 1]}


def plan(entry, metadata, counts, logical_size):
    """Check one launch's metadata and extents against ``gpu_merge_sort``.

    ``metadata`` holds each scalar parameter's value and each array's values,
    ``counts`` every buffer's bound extent, and ``logical_size`` the number of
    values the launch sorts. Returns the launch the host must use.
    """
    stage = signature(entry)[0]
    if stage == "block":
        return _plan_block(entry, metadata, counts, logical_size)
    if stage == "sort":
        return _plan_sort(entry, metadata, counts, logical_size)
    return _plan_pass(entry, metadata, counts, logical_size)


def _read(buffer, kind):
    return list(
        ctypes.cast(
            buffer.data, ctypes.POINTER(METADATA_TYPES[kind] * buffer.count)
        ).contents
    )


def _check_buffers(entry, buffers):
    types = operand_types(entry)
    written = outputs(entry)
    if set(buffers) != set(types):
        raise ValueError("Sort buffers do not match their kernel")
    for name, kind in types.items():
        buffer = buffers[name]
        maximum = (
            MAX_ELEMENTS if name in ELEMENTS else MAX_RANK if name in ARRAYS else 1
        )
        if (
            buffer.dtype != kind.encode("ascii")
            or not buffer.data
            or not 1 <= buffer.count <= maximum
            or buffer.output != int(name in written)
        ):
            raise ValueError("Sort buffer type, extent or direction differs")


def metadata_of(entry, buffers):
    """Every scalar's value and every array's values a launch binds."""
    types = operand_types(entry)
    metadata = {}
    for name, kind in types.items():
        if name in ARRAYS:
            metadata[name] = _read(buffers[name], kind)
        elif name not in ELEMENTS:
            metadata[name] = _read(buffers[name], kind)[0]
    return metadata


def validate(entry, buffers, logical_size, execution):
    """Check one launch against ``gpu_merge_sort`` and return its metadata."""
    signature(entry)
    _check_buffers(entry, buffers)
    metadata = metadata_of(entry, buffers)
    counts = {name: buffer.count for name, buffer in buffers.items()}
    if plan(entry, metadata, counts, logical_size) != execution:
        raise ValueError("Native sort launch does not match the upstream plan")
    return metadata


def validate_metadata(entry, metadata, counts, logical_size, execution):
    """Validate a detached launch, binding its values as the native host does.

    ``counts`` names the extent of every element buffer; the values are
    range-checked, bound again as the buffers the native host passes and run
    through the same plan.
    """
    types = operand_types(entry)
    if not isinstance(metadata, dict) or not isinstance(counts, dict):
        raise ValueError("Invalid sort layout")
    storage, buffers = {}, {}
    written = outputs(entry)
    for name, kind in types.items():
        if name not in ELEMENTS:
            values = metadata.get(name)
            values = values if name in ARRAYS else [values]
            bits = 32 if kind == "int32" else 64
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= (MAX_RANK if name in ARRAYS else 1)
                or any(
                    type(value) is not int
                    or not -(1 << (bits - 1)) <= value < 1 << (bits - 1)
                    for value in values
                )
            ):
                raise ValueError("Invalid sort metadata values")
            words = (METADATA_TYPES[kind] * len(values))(*values)
            count = len(values)
        else:
            count = counts.get(name)
            if type(count) is not int:
                raise ValueError("Invalid sort buffer extent")
            # Element storage is not read here; one byte stands in for it.
            words = (ctypes.c_uint8 * 1)()
        storage[name] = words
        buffers[name] = SimpleNamespace(
            dtype=kind.encode("ascii"),
            count=count,
            data=ctypes.addressof(words),
            output=int(name in written),
        )
    if set(metadata) != set(metadata_of(entry, buffers)):
        raise ValueError("Invalid sort layout")
    return validate(entry, buffers, logical_size, execution)
