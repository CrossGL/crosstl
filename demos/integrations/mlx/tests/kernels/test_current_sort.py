"""Native parity for every pinned MLX ``sort.metal`` kernel on all three backends.

MLX sorts and argsorts with a stable merge sort under ``LessThan`` (NaN after
every ordered value, complex values compared lexicographically). Single-block
kernels sort one row per threadgroup; rows longer than one block run the
multi-block path: ``sort_mbsort`` sorts tiles, then ``partition_mbsort`` and
``merge_mbsort`` merge tile pairs until one run remains (``sort.cpp``). Every
result is exact, so each output word is compared exactly.
"""

import ast
import functools
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.runtime_value_encoding import FLOAT16_BITS
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _element_type,
    _is_nan,
    _less_than,
    _levels,
    _metal_bytes,
    _runtime_value,
)
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/sort.metal"
SOURCE_SHA256 = "be97ea6106fb2e0c0c26f006037408e246fafa012fd2577bfc20cf356475ee61"
HEADER = "mlx/backend/metal/kernels/sort.h"
HEADER_SHA256 = "3171f47dabf2cf501dffc2e388e3d66446d6f71712e8d070028258a084b63574"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SORT"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/sort.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# instantiate_block_sort_bn / instantiate_multi_block_sort_base in sort.metal.
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
# instantiate_block_sort_long / instantiate_multi_block_sort_long.
LONG_DTYPES = ("uint64", "int64", "complex64")
DTYPES = BASE_DTYPES + LONG_DTYPES
FLOAT_DTYPES = ("float16", "float32", "bfloat16")
ITEM_SIZES = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "uint64": 8,
    "int64": 8,
    "complex64": 8,
}
N_PER_THREAD = 4
MB_STAGES = ("sort", "partition", "merge")


def _block_threads(dtype):
    return (32, 64, 128, 256, 512) if dtype in BASE_DTYPES else (32, 64, 128, 256)


def _multi_block_threads(dtype):
    return 512 if dtype in BASE_DTYPES else 256


BLOCK_ENTRIES = tuple(
    f"{layout}{name}_{dtype}_{output}_bn{bn}_tn{N_PER_THREAD}"
    for dtype in DTYPES
    for bn in _block_threads(dtype)
    for name, output in (("_block_sort", dtype), ("arg_block_sort", "uint32"))
    for layout in ("c", "nc")
)
MB_ENTRIES = tuple(
    f"{stage}_mbsort_{dtype}_uint32_bn{_multi_block_threads(dtype)}_tn{N_PER_THREAD}"
    for dtype in DTYPES
    for stage in MB_STAGES
)
ENTRIES = BLOCK_ENTRIES + MB_ENTRIES
BLOCK_PATTERN = re.compile(
    r"(?P<layout>c|nc)(?P<arg>arg)?_block_sort_(?P<dtype>[a-z0-9]+)_"
    r"(?P<output>[a-z0-9]+)_bn(?P<bn>\d+)_tn4"
)
MB_PATTERN = re.compile(
    r"(?P<stage>sort|partition|merge)_mbsort_(?P<dtype>[a-z0-9]+)_uint32_"
    r"bn(?P<bn>\d+)_tn4"
)
INT32_MAX = 2**31 - 1
GUARD = 0xABCDEF01
GUARD_COUNT = 16
# Output guards are values no case contains, so a stray or missing store shows.
VALUE_GUARDS = {
    "uint8": 0xA5,
    "uint16": 0xA5A5,
    "uint32": 0xABCDEF01,
    "uint64": 0xABCDEF0123456789,
    "int8": -91,
    "int16": -23131,
    "int32": -1412567295,
    "int64": -6066930334832433271,
    "float16": 3.75,
    "float32": 3.75,
    "bfloat16": 3.75,
    "complex64": (3.75, -3.75),
}
# multi_block_sort() dispatches partition_mbsort with min(n_blocks + 1, 1024)
# threads; the pipeline cases use three tiles, so the artifact is built for 4.
MB_TILE_COUNT = 3
PARTITION_WORKGROUP = MB_TILE_COUNT + 1
SIGNED_MAX = 2**31 - 1
# OpenGL cannot prove the absolute tile index of a threadgroup view passed to
# a helper (merge partitions are data-dependent binary searches). MLX's merge
# sort keeps every access inside its N_PER_BLOCK tile; these host
# preconditions state that range for each helper view.
_MERGE_VIEWS = (
    ("BlockMergeSort_*__merge_partition", ("As", "Bs")),
    ("BlockMergeSort_*__merge_step", ("As", "Bs")),
)
_ARG_MERGE_VIEWS = (
    ("BlockMergeSort_*__merge_partition", ("As", "Bs")),
    ("BlockMergeSort_*__merge_step", ("As", "Bs", "As_idx", "Bs_idx")),
)
OPENGL_WORKGROUP_VIEWS = {
    "block": (("KernelMergeSort_*__block_sort", ("tgp_vals",)),) + _MERGE_VIEWS,
    "arg_block": (
        (("KernelMergeSort_*__block_sort", ("tgp_vals", "tgp_idxs")),)
        + _ARG_MERGE_VIEWS
    ),
    "sort": (
        (("KernelMultiBlockMergeSort_*__block_sort", ("tgp_vals", "tgp_idxs")),)
        + _ARG_MERGE_VIEWS
    ),
    "partition": (),
    "merge": _ARG_MERGE_VIEWS,
}
# OpenGL indexes 32-bit storage arrays: signed 64-bit element offsets of the
# non-contiguous kernels are bounded to the int range by the host.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "nc": (
        "inp + in_block_idx",
        "out_ + out_block_idx",
        "inp + i * in_stride_sorted_axis",
        "out_ + i * out_stride_sorted_axis",
    ),
    "sort": ("inp + block_idx", "inp + idx * stride_sorted_axis"),
}


def _parse(entry):
    match = BLOCK_PATTERN.fullmatch(entry)
    if match is not None:
        return {
            "kind": "arg_block" if match["arg"] else "block",
            "layout": match["layout"],
            "arg": bool(match["arg"]),
            "dtype": match["dtype"],
            "bn": int(match["bn"]),
        }
    match = MB_PATTERN.fullmatch(entry)
    assert match is not None, entry
    return {
        "kind": match["stage"],
        "layout": None,
        "arg": True,
        "dtype": match["dtype"],
        "bn": int(match["bn"]),
    }


def _n_per_block(bn):
    return bn * N_PER_THREAD


def _workgroup(entry):
    info = _parse(entry)
    return [PARTITION_WORKGROUP if info["kind"] == "partition" else info["bn"], 1, 1]


def _mlx_launch(size, dtype):
    """Mirror ``gpu_merge_sort``: block threads and tile count for ``size``."""
    potential = (size + N_PER_THREAD - 1) // N_PER_THREAD
    bn = 32
    for threshold, threads in ((256, 512), (128, 256), (64, 128), (32, 64)):
        if potential > threshold:
            bn = threads
            break
    if bn == 512 and ITEM_SIZES[dtype] > 4:
        bn = 256
    n_per_block = _n_per_block(bn)
    return bn, (size + n_per_block - 1) // n_per_block


# ---------------------------------------------------------------------------
# Ordering model
# ---------------------------------------------------------------------------


def _pool(dtype):
    """Distinct logical values plus values MLX ``LessThan`` treats as equal."""
    values = list(_levels(dtype))
    if dtype in FLOAT_DTYPES:
        # Equal to +0.0: a stable sort keeps both zero signs in input order.
        values.append(-0.0)
    elif dtype == "complex64":
        values += [(-0.0, 0.0), (0.0, float("nan")), (1.5, -0.0)]
    elif dtype in {"uint64", "int64"}:
        # Equal low words with different high words catch 32-bit comparison.
        values += [2**32, 2**32 + 1]
    return values


def _lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        yield state >> 33


def _values(dtype, count, salt):
    """Deterministic values with every pool member and many ties."""
    pool = _pool(dtype)
    seed = int.from_bytes(
        hashlib.sha256(f"{dtype}:{salt}".encode()).digest()[:8], "little"
    )
    draws = _lcg(seed)
    values = [pool[index % len(pool)] for index in range(min(count, len(pool)))]
    values += [pool[next(draws) % len(pool)] for _ in range(count - len(values))]
    for index in range(len(values) - 1, 0, -1):
        other = next(draws) % (index + 1)
        values[index], values[other] = values[other], values[index]
    return values


def _compare(dtype, left, right):
    if _less_than(dtype, left, right):
        return -1
    if _less_than(dtype, right, left):
        return 1
    return 0


def _stable_order(dtype, values):
    """Indices of the stable sort of ``values`` under MLX ``LessThan``."""
    return sorted(
        range(len(values)),
        key=functools.cmp_to_key(
            lambda left, right: _compare(dtype, values[left], values[right])
        ),
    )


def _merge_partition(dtype, a_values, b_values, a_size, b_size, sort_md):
    """``merge_partition`` from sort.h: A elements in the first ``sort_md``."""
    a_start = max(0, sort_md - b_size)
    a_end = min(sort_md, a_size)
    while a_start < a_end:
        middle = a_start + (a_end - a_start) // 2
        if _less_than(dtype, b_values[sort_md - 1 - middle], a_values[middle]):
            a_end = middle
        else:
            a_start = middle + 1
    return a_end


def _stable_merge(dtype, first, second):
    """Merge ``(value, index)`` runs; ``merge_step`` takes B only when B < A."""
    merged, a, b = [], 0, 0
    while a < len(first) or b < len(second):
        take_b = b < len(second) and (
            a >= len(first) or _less_than(dtype, second[b][0], first[a][0])
        )
        if take_b:
            merged.append(second[b])
            b += 1
        else:
            merged.append(first[a])
            a += 1
    return merged


def _mb_sort_stage(dtype, rows, n_per_block):
    """``mb_block_sort``: stable-sort each tile, recording row positions."""
    values, indices = [], []
    for row in rows:
        for start in range(0, len(row), n_per_block):
            tile = list(range(start, min(start + n_per_block, len(row))))
            order = [
                tile[index] for index in _stable_order(dtype, [row[i] for i in tile])
            ]
            values += [row[index] for index in order]
            indices += order
    return values, indices


def _mb_partition_stage(dtype, values, size, merge_tiles, n_blocks, n_per_block):
    """``mb_block_partition`` for every row (``n_blocks + 1`` partitions each)."""
    partitions = []
    for start in range(0, len(values), size):
        row = values[start : start + size]
        for tile in range(n_blocks + 1):
            merge_group, merge_lane = divmod(tile, merge_tiles)
            sort_sz = n_per_block * merge_tiles
            sort_st = n_per_block * merge_tiles * merge_group
            a_st = min(size, sort_st)
            a_ed = min(size, sort_st + sort_sz // 2)
            b_st = a_ed
            b_ed = min(size, b_st + sort_sz // 2)
            partition_at = min(b_ed - a_st, n_per_block * merge_lane)
            partition = _merge_partition(
                dtype, row[a_st:], row[b_st:], a_ed - a_st, b_ed - b_st, partition_at
            )
            partitions.append(a_st + partition)
    return partitions


def _mb_merge_stage(
    dtype, values, indices, partitions, size, merge_tiles, n_blocks, n_per_block
):
    """``mb_block_merge``: each tile merges its A and B slices stably."""
    out_values, out_indices = [], []
    for row_index, start in enumerate(range(0, len(values), size)):
        row = list(zip(values[start : start + size], indices[start : start + size]))
        bounds = partitions[
            row_index * (n_blocks + 1) : (row_index + 1) * (n_blocks + 1)
        ]
        merged_row = [None] * size
        for block in range(n_blocks):
            merge_group = block // merge_tiles
            sort_st = n_per_block * merge_tiles * merge_group
            sort_sz = n_per_block * merge_tiles
            sort_md = n_per_block * block - sort_st
            a_st, a_ed = bounds[block], bounds[block + 1]
            b_st = min(size, 2 * sort_st + sort_sz // 2 + sort_md - a_st)
            b_ed = min(size, 2 * sort_st + sort_sz // 2 + sort_md + n_per_block - a_ed)
            if block % merge_tiles == merge_tiles - 1:
                a_ed = min(size, sort_st + sort_sz // 2)
                b_ed = min(size, sort_st + sort_sz)
            merged = _stable_merge(dtype, row[a_st:a_ed], row[b_st:b_ed])
            for offset, item in enumerate(merged[:n_per_block]):
                position = n_per_block * block + offset
                if position < size:
                    merged_row[position] = item
        assert None not in merged_row
        out_values += [item[0] for item in merged_row]
        out_indices += [item[1] for item in merged_row]
    return out_values, out_indices


def _merge_levels(n_blocks):
    """``merge_tiles`` values of the ``multi_block_sort`` merge loop."""
    levels, merge_tiles = [], 2
    while merge_tiles // 2 < n_blocks:
        levels.append(merge_tiles)
        merge_tiles *= 2
    return levels


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _elem_to_loc(element, shape, strides):
    """``elem_to_loc<int64_t>`` from utils.h."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if element <= 0:
            break
        location += (element % extent) * stride
        element //= extent
    return location


def _block_cases(layout, bn):
    """Shapes MLX routes to ``bn``-thread single-block kernels (``sort.cpp``)."""
    n_per_block = _n_per_block(bn)
    if layout == "c":
        return {
            # A 1-D row: no other axis, so both segment strides are INT32_MAX.
            "padded": {
                "size": 2 * bn + 3,
                "rows": 1,
                "inSorted": 1,
                "outSorted": 1,
                "inSegment": INT32_MAX,
                "outSegment": INT32_MAX,
            },
            # (2, N_PER_BLOCK) sorted along the contiguous last axis.
            "rows": {
                "size": n_per_block,
                "rows": 2,
                "inSorted": 1,
                "outSorted": 1,
                "inSegment": n_per_block,
                "outSegment": n_per_block,
            },
            # (size, 3) sorted along axis 0: the sorted stride is the largest.
            "columns": {
                "size": n_per_block - 1,
                "rows": 3,
                "inSorted": 3,
                "outSorted": 3,
                "inSegment": 1,
                "outSegment": 1,
            },
        }
    size = 3 * bn + 1
    return {
        # x[::2]: a non-contiguous 1-D view, so nc_dim is 0 (shape 0, stride 0).
        "strided": {
            "size": 2 * bn + 3,
            "rows": 1,
            "inSorted": 2,
            "outSorted": 1,
            "shape": [0],
            "inStrides": [0],
            "outStrides": [0],
            "ncDim": 0,
        },
        # transpose(0, 2, 1) of a (2, 3, size) array sorted along axis 1; the
        # output is a fresh row-contiguous (2, size, 3) array.
        "planes": {
            "size": size,
            "rows": 6,
            "inSorted": 1,
            "outSorted": 3,
            "shape": [2, 3],
            "inStrides": [3 * size, size],
            "outStrides": [3 * size, 1],
            "ncDim": 2,
        },
    }


def _mb_cases(dtype):
    """Rows MLX sorts with the multi-block path: three tiles, two merge levels."""
    n_per_block = _n_per_block(_multi_block_threads(dtype))
    size = (MB_TILE_COUNT - 1) * n_per_block + 37
    return {
        # (2, size) sorted along the contiguous last axis.
        "rows": {"size": size, "rows": 2, "sorted": 1, "shape": [2], "strides": [size]},
        # (size, 2) sorted along axis 0.
        "columns": {"size": size, "rows": 2, "sorted": 2, "shape": [2], "strides": [1]},
    }


# The cases above in terms of the entry's block threads (``bn``), its tile
# (``N_PER_BLOCK``) and the sorted length (``size``).
BLOCK_CASE_CONTRACT = {
    "c": {
        "padded": {
            "size": "2 * bn + 3",
            "rows": 1,
            "inSorted": 1,
            "outSorted": 1,
            "inSegment": INT32_MAX,
            "outSegment": INT32_MAX,
        },
        "rows": {
            "size": "4 * bn",
            "rows": 2,
            "inSorted": 1,
            "outSorted": 1,
            "inSegment": "4 * bn",
            "outSegment": "4 * bn",
        },
        "columns": {
            "size": "4 * bn - 1",
            "rows": 3,
            "inSorted": 3,
            "outSorted": 3,
            "inSegment": 1,
            "outSegment": 1,
        },
    },
    "nc": {
        "strided": {
            "size": "2 * bn + 3",
            "rows": 1,
            "inSorted": 2,
            "outSorted": 1,
            "shape": [0],
            "inStrides": [0],
            "outStrides": [0],
            "ncDim": 0,
        },
        "planes": {
            "size": "3 * bn + 1",
            "rows": 6,
            "inSorted": 1,
            "outSorted": 3,
            "shape": [2, 3],
            "inStrides": ["3 * size", "size"],
            "outStrides": ["3 * size", 1],
            "ncDim": 2,
        },
    },
}
MB_CASE_CONTRACT = {
    "rows": {
        "size": "2 * N_PER_BLOCK + 37",
        "rows": 2,
        "sorted": 1,
        "shape": [2],
        "strides": ["size"],
    },
    "columns": {
        "size": "2 * N_PER_BLOCK + 37",
        "rows": 2,
        "sorted": 2,
        "shape": [2],
        "strides": [1],
    },
}


def _evaluate(value, names):
    """Evaluate a contract value built from integers, names, ``+ - *``."""
    if isinstance(value, list):
        return [_evaluate(item, names) for item in value]
    if not isinstance(value, str):
        return value

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp):
            operation = {
                ast.Add: int.__add__,
                ast.Sub: int.__sub__,
                ast.Mult: int.__mul__,
            }
            return operation[type(node.op)](walk(node.left), walk(node.right))
        raise AssertionError(ast.dump(node))

    return walk(ast.parse(value, mode="eval"))


def _resolve_case(contract, names):
    names = dict(names)
    names["size"] = _evaluate(contract["size"], names)
    return {key: _evaluate(value, names) for key, value in contract.items()}


def _row_bases(case, prefix):
    if "shape" in case:
        strides = case[f"{prefix}Strides"]
        return [
            _elem_to_loc(row, case["shape"], strides) for row in range(case["rows"])
        ]
    segment = case[f"{prefix}Segment"]
    return [row * segment for row in range(case["rows"])]


def _extent(bases, sorted_stride, size):
    return max(bases) + (size - 1) * sorted_stride + 1


def _block_data(entry, name):
    info = _parse(entry)
    dtype = info["dtype"]
    case = _block_cases(info["layout"], info["bn"])[name]
    size, rows = case["size"], case["rows"]
    kind = "argsort" if info["arg"] else "sort"
    logical = [
        _values(dtype, size, f"{kind}:{info['bn']}:{name}:{row}") for row in range(rows)
    ]
    in_bases = _row_bases(case, "in")
    out_bases = _row_bases(case, "out")
    # Unread storage holds the smallest value: reading it moves it to the front.
    storage = [_levels(dtype)[0]] * _extent(in_bases, case["inSorted"], size)
    written = [False] * len(storage)
    for base, row in zip(in_bases, logical):
        for index, value in enumerate(row):
            position = base + index * case["inSorted"]
            assert not written[position]
            written[position] = True
            storage[position] = value
    guard = GUARD if info["arg"] else VALUE_GUARDS[dtype]
    output_count = _extent(out_bases, case["outSorted"], size)
    expected = [guard] * (output_count + GUARD_COUNT)
    for base, row in zip(out_bases, logical):
        order = _stable_order(dtype, row)
        for index, source in enumerate(order):
            expected[base + index * case["outSorted"]] = (
                source if info["arg"] else row[source]
            )
    # Every output element is written, and no sorted value equals the guard.
    assert guard not in expected[:output_count]
    return {
        "case": case,
        "rows": logical,
        "storage": storage,
        "initial": [guard] * (output_count + GUARD_COUNT),
        "expected": expected,
    }


def _mb_data(dtype, name):
    case = _mb_cases(dtype)[name]
    size, rows = case["size"], case["rows"]
    logical = [_values(dtype, size, f"mbsort:{name}:{row}") for row in range(rows)]
    bases = [_elem_to_loc(row, case["shape"], case["strides"]) for row in range(rows)]
    storage = [None] * _extent(bases, case["sorted"], size)
    for base, row in zip(bases, logical):
        for index, value in enumerate(row):
            storage[base + index * case["sorted"]] = value
    assert None not in storage
    return {"case": case, "rows": logical, "storage": storage}


def _mb_pipeline(dtype, name):
    """Every stage's exact outputs for one multi-block case."""
    data = _mb_data(dtype, name)
    case = data["case"]
    size = case["size"]
    bn = _multi_block_threads(dtype)
    n_per_block = _n_per_block(bn)
    launch = _mlx_launch(size, dtype)
    assert launch == (bn, MB_TILE_COUNT)
    values, indices = _mb_sort_stage(dtype, data["rows"], n_per_block)
    stages = [{"stage": "sort", "values": values, "indices": indices}]
    for merge_tiles in _merge_levels(MB_TILE_COUNT):
        partitions = _mb_partition_stage(
            dtype, values, size, merge_tiles, MB_TILE_COUNT, n_per_block
        )
        values, indices = _mb_merge_stage(
            dtype,
            values,
            indices,
            partitions,
            size,
            merge_tiles,
            MB_TILE_COUNT,
            n_per_block,
        )
        stages.append(
            {
                "stage": "merge",
                "mergeTiles": merge_tiles,
                "partitions": partitions,
                "values": values,
                "indices": indices,
            }
        )
    return data, stages


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _config(output, target, entry):
    info = _parse(entry)
    assertions = ""
    if target == "opengl":
        maximum = _n_per_block(info["bn"]) - 1
        for function, parameters in OPENGL_WORKGROUP_VIEWS[info["kind"]]:
            for parameter in parameters:
                assertions += f"""
[[project.workgroup_access_assertions]]
source = "{SOURCE}"
entry_point = "{entry}"
function = "{function}"
parameter = "{parameter}"
minimum = 0
maximum = {maximum}
"""
        group = "nc" if info["layout"] == "nc" else info["kind"]
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS.get(group, ()):
            assertions += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    size = ", ".join(str(item) for item in _workgroup(entry))
    return assertions + f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [{size}]
[project.source_options.metal]
max_template_specializations = 256
max_template_materialization_work = 16384
"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _dispatch_counts():
    block = sum(
        len(_block_cases(_parse(entry)["layout"], _parse(entry)["bn"]))
        for entry in BLOCK_ENTRIES
    )
    stages = 1 + 2 * len(_merge_levels(MB_TILE_COUNT))
    multi = sum(len(_mb_cases(dtype)) * stages for dtype in DTYPES)
    return block, multi


def _expected_validation_contract():
    block_dispatches, multi_dispatches = _dispatch_counts()
    return {
        "ordering": (
            "stable sort under MLX LessThan: NaN after every ordered value, "
            "complex values lexicographic by (real, imag)"
        ),
        "comparison": "exact words",
        "nPerThread": N_PER_THREAD,
        "workgroupSize": {
            "blockSort": "[bn, 1, 1]",
            "sortMbsort": "[bn, 1, 1]",
            "partitionMbsort": [PARTITION_WORKGROUP, 1, 1],
            "mergeMbsort": "[bn, 1, 1]",
        },
        "grid": {
            "blockSort": "[1, rows, 1]",
            "sortMbsort": "[n_blocks, rows, 1]",
            "partitionMbsort": "[1, rows, 1]",
            "mergeMbsort": "[n_blocks, rows, 1]",
        },
        "blockCases": BLOCK_CASE_CONTRACT,
        "multiBlockCases": MB_CASE_CONTRACT,
        "multiBlockTileCount": MB_TILE_COUNT,
        "multiBlockMergeTiles": _merge_levels(MB_TILE_COUNT),
        "multiBlockChaining": (
            "each stage consumes the previous stage's verified native outputs"
        ),
        "openglHostPreconditions": {
            "workgroupAccessAssertions": {
                kind: [
                    {
                        "function": function,
                        "parameters": list(parameters),
                        "minimum": 0,
                        "maximum": "N_PER_BLOCK - 1",
                    }
                    for function, parameters in views
                ]
                for kind, views in OPENGL_WORKGROUP_VIEWS.items()
            },
            "indexRangeAssertions": {
                group: [
                    {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                    for expression in expressions
                ]
                for group, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
            },
        },
        "valueGuards": {
            dtype: list(value) if isinstance(value, tuple) else value
            for dtype, value in VALUE_GUARDS.items()
        },
        "indexGuard": GUARD,
        "outputGuardCount": GUARD_COUNT,
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
            for target in TARGETS
        },
        "dispatchCountByTarget": {
            "blockSort": block_dispatches,
            "multiBlock": multi_dispatches,
        },
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    block_dispatches, multi_dispatches = _dispatch_counts()
    return {
        "instantiatedEntryCount": len(ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": block_dispatches + multi_dispatches,
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_sort_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-sort-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
        assert len({record["sha256"] for record in artifacts.values()}) == len(ENTRIES)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_sort_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "sort.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 264
    assert len(BLOCK_ENTRIES) == 228 and len(MB_ENTRIES) == 36
    assert ENTRIES[:4] == (
        "c_block_sort_uint8_uint8_bn32_tn4",
        "nc_block_sort_uint8_uint8_bn32_tn4",
        "carg_block_sort_uint8_uint32_bn32_tn4",
        "ncarg_block_sort_uint8_uint32_bn32_tn4",
    )
    assert ENTRIES[227] == "ncarg_block_sort_complex64_uint32_bn256_tn4"
    assert ENTRIES[228:231] == (
        "sort_mbsort_uint8_uint32_bn512_tn4",
        "partition_mbsort_uint8_uint32_bn512_tn4",
        "merge_mbsort_uint8_uint32_bn512_tn4",
    )
    assert ENTRIES[-1] == "merge_mbsort_complex64_uint32_bn256_tn4"
    for entry in ENTRIES:
        info = _parse(entry)
        assert info["dtype"] in DTYPES
        if info["layout"] is None:
            assert info["bn"] == _multi_block_threads(info["dtype"])
        else:
            assert info["bn"] in _block_threads(info["dtype"])


@pytest.mark.parametrize("entry", BLOCK_ENTRIES)
def test_block_sort_cases_follow_mlx_dispatch(entry):
    info = _parse(entry)
    cases = _block_cases(info["layout"], info["bn"])
    assert set(cases) == set(BLOCK_CASE_CONTRACT[info["layout"]])
    for name, case in cases.items():
        assert case == _resolve_case(
            BLOCK_CASE_CONTRACT[info["layout"]][name], {"bn": info["bn"]}
        )
        # gpu_merge_sort() routes each case to this entry's block size.
        assert _mlx_launch(case["size"], info["dtype"]) == (info["bn"], 1), name
        data = _block_data(entry, name)
        assert len(data["expected"]) == len(data["initial"])
        assert data["expected"][-GUARD_COUNT:] == data["initial"][-GUARD_COUNT:]
        if info["layout"] == "c":
            # single_block_sort() takes the contiguous kernel only when the
            # sorted stride is the smallest or the largest of the array.
            strides = {case["inSorted"], case["inSegment"]}
            assert case["rows"] == 1 or case["inSorted"] in {
                min(strides),
                max(strides),
            }
        else:
            assert len(case["shape"]) == len(case["inStrides"])
            assert case["ncDim"] == (0 if case["shape"] == [0] else len(case["shape"]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_multi_block_cases_follow_mlx_dispatch(dtype):
    bn = _multi_block_threads(dtype)
    cases = _mb_cases(dtype)
    assert set(cases) == set(MB_CASE_CONTRACT)
    for name, case in cases.items():
        assert case == _resolve_case(
            MB_CASE_CONTRACT[name], {"N_PER_BLOCK": _n_per_block(bn)}
        )
        # gpu_merge_sort() sends the case to multi_block_sort() with this bn.
        assert _mlx_launch(case["size"], dtype) == (bn, MB_TILE_COUNT)
    assert _merge_levels(MB_TILE_COUNT) == [2, 4]
    # multi_block_sort() launches partitions with min(n_blocks + 1, 1024).
    assert PARTITION_WORKGROUP == min(MB_TILE_COUNT + 1, 1024)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_cases_cover_ties_nan_and_signed_zero(dtype):
    values = _values(dtype, 4 * 512, "coverage")
    pool = _pool(dtype)
    assert all(value in values or _is_nan(dtype, value) for value in pool)
    assert len(values) > 2 * len(pool)
    order = _stable_order(dtype, values)
    ordered = [values[index] for index in order]
    assert not any(
        _less_than(dtype, right, left) for left, right in zip(ordered, ordered[1:])
    )
    # Equal values keep their input order, so ties are observable.
    for left, right in zip(order, order[1:]):
        if _compare(dtype, values[left], values[right]) == 0:
            assert left < right
    if dtype in FLOAT_DTYPES:
        signs = [str(value)[0] == "-" for value in ordered if value == 0.0]
        assert True in signs and False in signs
        assert _is_nan(dtype, ordered[-1])
    if dtype == "complex64":
        assert _is_nan(dtype, ordered[-1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ("rows", "columns"))
def test_multi_block_model_reaches_the_stable_sort(dtype, name):
    data, stages = _mb_pipeline(dtype, name)
    size = data["case"]["size"]
    final = stages[-1]
    assert [stage.get("mergeTiles") for stage in stages] == [None, 2, 4]
    for row_index, row in enumerate(data["rows"]):
        order = _stable_order(dtype, row)
        start = row_index * size
        assert final["indices"][start : start + size] == order
        # The model moves the same value objects, so NaNs compare by identity.
        assert final["values"][start : start + size] == [row[i] for i in order]
    for stage in stages[1:]:
        # Partitions are monotone tile boundaries within each row.
        width = MB_TILE_COUNT + 1
        for row_index in range(data["case"]["rows"]):
            bounds = stage["partitions"][row_index * width : (row_index + 1) * width]
            assert bounds == sorted(bounds) and bounds[-1] <= size
    # The first merge level leaves the last tile unmerged; the second merges it.
    assert stages[1]["indices"] != stages[2]["indices"]
    assert stages[0]["indices"] != stages[1]["indices"]


def _is_negative_zero(value):
    return value == 0.0 and str(value).startswith("-")


SORT_DEFECTS = {
    # Equal keys in reverse input order: an unstable merge.
    "unstable": lambda dtype, row, a, b: _compare(dtype, row[a], row[b]) or (b - a),
    # NaN before every ordered value.
    "nan-first": lambda dtype, row, a, b: (
        (_is_nan(dtype, row[b]) - _is_nan(dtype, row[a]))
        or _compare(dtype, row[a], row[b])
        or (a - b)
    ),
    # -0.0 before +0.0, as a sort of the IEEE bit patterns would order them.
    "zero-sign": lambda dtype, row, a, b: (
        _compare(dtype, row[a], row[b])
        or (_is_negative_zero(row[b]) - _is_negative_zero(row[a]))
        or (a - b)
    ),
}


@pytest.mark.parametrize("defect", sorted(SORT_DEFECTS))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sort_expectations_reject_defects(defect, dtype):
    """Every case set separates the exact stable order from common defects."""
    compare = SORT_DEFECTS[defect]

    def words(rows, orders):
        return [
            _runtime_value("metal", dtype, [row[i] for i in order])["values"]
            for row, order in zip(rows, orders)
        ]

    for entry in (
        f"c_block_sort_{dtype}_{dtype}_bn32_tn4",
        f"ncarg_block_sort_{dtype}_uint32_bn512_tn4",
        f"nc_block_sort_{dtype}_{dtype}_bn512_tn4",
    ):
        info = _parse(entry)
        exposed = []
        for name in _block_cases(info["layout"], info["bn"]):
            rows = _block_data(entry, name)["rows"]
            exact = [_stable_order(dtype, row) for row in rows]
            mutated = [
                sorted(
                    range(len(row)),
                    key=functools.cmp_to_key(
                        lambda a, b, row=row: compare(dtype, row, a, b)
                    ),
                )
                for row in rows
            ]
            # Argsort indices expose the defect in every case.
            assert mutated != exact, (entry, name)
            exposed.append(words(rows, mutated) != words(rows, exact))
        # Value sorts expose it through the stored words in some case.
        assert any(exposed), entry
    # Swapping rows (a wrong segment offset) changes every multi-row case.
    data = _block_data(f"nc_block_sort_{dtype}_{dtype}_bn32_tn4", "planes")
    assert data["rows"] != list(reversed(data["rows"]))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry",
    (
        "c_block_sort_uint8_uint8_bn32_tn4",
        "ncarg_block_sort_float32_uint32_bn512_tn4",
        "sort_mbsort_int64_uint32_bn256_tn4",
        "partition_mbsort_float16_uint32_bn512_tn4",
        "merge_mbsort_complex64_uint32_bn256_tn4",
    ),
)
def test_sort_configs_scope_opengl_host_preconditions(tmp_path, target, entry):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    info = _parse(entry)
    assert list(config.targets) == [target]
    workgroup = [item.to_json() for item in config.workgroup_access_assertions]
    indices = [item.to_json() for item in config.index_range_assertions]
    if target != "opengl":
        assert workgroup == [] and indices == []
        return
    maximum = _n_per_block(info["bn"]) - 1
    assert workgroup == [
        {
            "source": SOURCE,
            "entryPoint": entry,
            "function": function,
            "parameter": parameter,
            "minimum": 0,
            "maximum": maximum,
        }
        for function, parameters in OPENGL_WORKGROUP_VIEWS[info["kind"]]
        for parameter in parameters
    ]
    group = "nc" if info["layout"] == "nc" else info["kind"]
    assert indices == [
        {
            "source": SOURCE,
            "expression": expression,
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS.get(group, ())
    ]


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native sort gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256((root / HEADER).read_bytes()).hexdigest() == HEADER_SHA256
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("sort-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _scalar(dtype, value):
    return {"dtype": dtype, "shape": [1], "values": [value]}


def _array(dtype, values):
    return {"dtype": dtype, "shape": [len(values)], "values": list(values)}


def _translate(root, work, target, entry):
    """Translate one entry and check it against its pinned artifact."""
    record = _contract()["artifacts"][target][entry]
    stage = work / entry
    stage.mkdir()
    config = stage / "crosstl.toml"
    config.write_text(
        _config(f"{work.name}/{entry}/out", target, entry), encoding="utf-8"
    )
    translation = translate_project(
        load_project_config(root, config),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(stage / "report.json")
    validation = validate_project_report(stage / "report.json")
    assert validation["success"] is True, validation
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    assert report["summary"]["failedCount"] == 0
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": record["sha256"],
    }
    assert artifact["generatedSizeBytes"] == record["sizeBytes"]
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == _workgroup(entry)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, stage / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, stage, target)
    descriptor, package = _prepare_native_package(translation, stage)
    return descriptor, package, record


def _request(descriptor, package, target, entry, data, outputs, grid):
    """Bind ``data`` by source parameter name and build one dispatch."""
    inputs, observed, output_names = {}, [], {}
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry + "_")
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"]).removeprefix(prefix)
        value = data[name]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name in outputs:
            output_names[name] = binding["name"]
        else:
            observed.append(binding["name"])
    assert set(output_names) == set(outputs) and len(inputs) == len(data)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_names[name]: dict(data[name]) for name in outputs},
        {"workgroupCount": grid, "workgroupSize": _workgroup(entry)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_names, len(observed)


def _execute(executor, work, label, request, output_names, observed):
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    (work / f"{label}-result.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "outputs": result.outputs,
                "details": result.details,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    assert result.status == "ok"
    observations = result.details["inputObservations"]
    assert len(observations) == observed
    assert all(item["unchanged"] for item in observations.values())
    return {name: result.outputs[binding] for name, binding in output_names.items()}


def _upstream(work, reference, entry, label, payloads, grid, outputs):
    """Run the upstream metallib on raw buffers in ``[[buffer(i)]]`` order."""
    runner, library = reference
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": grid,
                "workgroupSize": _workgroup(entry),
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    raw = {}
    for index, payload in enumerate(payloads):
        data = (readback / f"buffer-{index}.bin").read_bytes()
        if index in outputs:
            raw[index] = data
        else:
            assert data == payload, (label, index)
    return device, raw


def _block_inputs(target, entry, data):
    info = _parse(entry)
    dtype, case = info["dtype"], data["case"]
    output_dtype = "uint32" if info["arg"] else dtype
    values = {
        "inp": _runtime_value(target, dtype, data["storage"]),
        "out_": _runtime_value(target, output_dtype, data["initial"]),
        "size_sorted_axis": _scalar("int32", case["size"]),
        "in_stride_sorted_axis": _scalar("int32", case["inSorted"]),
        "out_stride_sorted_axis": _scalar("int32", case["outSorted"]),
    }
    if info["layout"] == "c":
        values["in_stride_segment_axis"] = _scalar("int32", case["inSegment"])
        values["out_stride_segment_axis"] = _scalar("int32", case["outSegment"])
    else:
        values["nc_dim"] = _scalar("int32", case["ncDim"])
        values["nc_shape"] = _array("int32", case["shape"])
        values["in_nc_strides"] = _array("int64", case["inStrides"])
        values["out_nc_strides"] = _array("int64", case["outStrides"])
    return values


def _block_payloads(entry, data):
    info = _parse(entry)
    dtype, case = info["dtype"], data["case"]
    output_dtype = "uint32" if info["arg"] else dtype
    payloads = [
        _metal_bytes(dtype, data["storage"]),
        _metal_bytes(output_dtype, data["initial"]),
        struct.pack("<i", case["size"]),
        struct.pack("<i", case["inSorted"]),
        struct.pack("<i", case["outSorted"]),
    ]
    if info["layout"] == "c":
        payloads += [
            struct.pack("<i", case["inSegment"]),
            struct.pack("<i", case["outSegment"]),
        ]
    else:
        payloads += [
            struct.pack("<i", case["ncDim"]),
            struct.pack(f"<{len(case['shape'])}i", *case["shape"]),
            struct.pack(f"<{len(case['inStrides'])}q", *case["inStrides"]),
            struct.pack(f"<{len(case['outStrides'])}q", *case["outStrides"]),
        ]
    return payloads


def _assert_words(label, actual, expected):
    assert len(actual) == len(expected), label
    mismatches = [
        {"index": index, "expected": want, "actual": got}
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


@pytest.mark.parametrize("entry", BLOCK_ENTRIES)
def test_current_mlx_block_sort_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    info = _parse(entry)
    dtype = info["dtype"]
    output_dtype = "uint32" if info["arg"] else dtype
    with tempfile.TemporaryDirectory(prefix=".current-sort-", dir=root) as directory:
        work = Path(directory)
        try:
            descriptor, package, record = _translate(root, work, target, entry)
            executor = _executor(target)
            evidence = []
            try:
                for name in _block_cases(info["layout"], info["bn"]):
                    data = _block_data(entry, name)
                    request, output_names, observed = _request(
                        descriptor,
                        package,
                        target,
                        entry,
                        _block_inputs(target, entry, data),
                        ("out_",),
                        [1, data["case"]["rows"], 1],
                    )
                    outputs = _execute(
                        executor, work, name, request, output_names, observed
                    )
                    output = outputs["out_"]
                    expected = _runtime_value(target, output_dtype, data["expected"])
                    assert output["dtype"] == expected["dtype"]
                    _assert_words(name, output["values"], expected["values"])
                    upstream = None
                    if target == "metal":
                        device, raw = _upstream(
                            work,
                            metal_reference,
                            entry,
                            name,
                            _block_payloads(entry, data),
                            [1, data["case"]["rows"], 1],
                            {1},
                        )
                        assert raw[1] == _metal_bytes(output_dtype, data["expected"])
                        upstream = {"device": device, "matched": True}
                    evidence.append(
                        {
                            "case": name,
                            **{
                                key: value
                                for key, value in data["case"].items()
                                if key != "ncDim"
                            },
                            "outputGuardCount": GUARD_COUNT,
                            "observedInputCount": observed,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "comparison": "exact words",
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def _stage_inputs(target, dtype, stage, case, previous, level, guards):
    """Runtime values and upstream raw buffers for one multi-block dispatch."""
    size = case["size"]
    values_guard, index_guard = guards
    if stage == "sort":
        storage = previous
        return {
            "inp": _runtime_value(target, dtype, storage),
            "out_vals": _runtime_value(target, dtype, values_guard),
            "out_idxs": _array("uint32", index_guard),
            "size_sorted_axis": _scalar("int32", size),
            "stride_sorted_axis": _scalar("int32", case["sorted"]),
            "nc_dim": _scalar("int32", len(case["shape"])),
            "nc_shape": _array("int32", case["shape"]),
            "nc_strides": _array("int64", case["strides"]),
        }, [
            _metal_bytes(dtype, storage),
            _metal_bytes(dtype, values_guard),
            _metal_bytes("uint32", index_guard),
            struct.pack("<i", size),
            struct.pack("<i", case["sorted"]),
            struct.pack("<i", len(case["shape"])),
            struct.pack(f"<{len(case['shape'])}i", *case["shape"]),
            struct.pack(f"<{len(case['strides'])}q", *case["strides"]),
        ]
    merge_tiles = level["mergeTiles"]
    if stage == "partition":
        partition_guard = [GUARD] * (len(level["partitions"]) + GUARD_COUNT)
        return {
            "block_partitions": _array("uint32", partition_guard),
            "dev_vals": _runtime_value(target, dtype, previous["values"]),
            "dev_idxs": _array("uint32", previous["indices"]),
            "size_sorted_axis": _scalar("int32", size),
            "merge_tiles": _scalar("int32", merge_tiles),
            "n_blocks": _scalar("int32", MB_TILE_COUNT),
        }, [
            _metal_bytes("uint32", partition_guard),
            _metal_bytes(dtype, previous["values"]),
            _metal_bytes("uint32", previous["indices"]),
            struct.pack("<i", size),
            struct.pack("<i", merge_tiles),
            struct.pack("<i", MB_TILE_COUNT),
        ]
    assert stage == "merge"
    return {
        "block_partitions": _array("uint32", level["partitions"]),
        "dev_vals_in": _runtime_value(target, dtype, previous["values"]),
        "dev_idxs_in": _array("uint32", previous["indices"]),
        "dev_vals_out": _runtime_value(target, dtype, values_guard),
        "dev_idxs_out": _array("uint32", index_guard),
        "size_sorted_axis": _scalar("int32", size),
        "merge_tiles": _scalar("int32", merge_tiles),
        "num_tiles": _scalar("int32", MB_TILE_COUNT),
    }, [
        _metal_bytes("uint32", level["partitions"]),
        _metal_bytes(dtype, previous["values"]),
        _metal_bytes("uint32", previous["indices"]),
        _metal_bytes(dtype, values_guard),
        _metal_bytes("uint32", index_guard),
        struct.pack("<i", size),
        struct.pack("<i", merge_tiles),
        struct.pack("<i", MB_TILE_COUNT),
    ]


# Output parameters of each multi-block kernel and their [[buffer(i)]] slots.
MB_OUTPUTS = {
    "sort": {"out_vals": 1, "out_idxs": 2},
    "partition": {"block_partitions": 0},
    "merge": {"dev_vals_out": 3, "dev_idxs_out": 4},
}


def _stage_expected(dtype, stage, level):
    """Exact output words of one stage, guards included."""
    values = level["values"] + [VALUE_GUARDS[dtype]] * GUARD_COUNT
    indices = level["indices"] + [GUARD] * GUARD_COUNT
    if stage == "partition":
        return {
            "block_partitions": ("uint32", level["partitions"] + [GUARD] * GUARD_COUNT)
        }
    names = list(MB_OUTPUTS[stage])
    return {names[0]: (dtype, values), names[1]: ("uint32", indices)}


def _run_stage(context, stage, label, case, previous, level):
    """Dispatch one multi-block stage natively (and upstream on Metal)."""
    target, dtype = context["target"], context["dtype"]
    entry = context["entries"][stage]
    descriptor, package, _ = context["translated"][stage]
    count = case["size"] * case["rows"]
    guards = (
        [VALUE_GUARDS[dtype]] * (count + GUARD_COUNT),
        [GUARD] * (count + GUARD_COUNT),
    )
    values, payloads = _stage_inputs(
        target, dtype, stage, case, previous, level, guards
    )
    grid = [1 if stage == "partition" else MB_TILE_COUNT, case["rows"], 1]
    request, names, observed = _request(
        descriptor, package, target, entry, values, tuple(MB_OUTPUTS[stage]), grid
    )
    outputs = _execute(
        context["executor"], context["work"], label, request, names, observed
    )
    for name, (output_dtype, words) in _stage_expected(dtype, stage, level).items():
        expected = _runtime_value(target, output_dtype, words)["values"]
        _assert_words(f"{label}:{name}", outputs[name]["values"], expected)
    record = {"stage": stage, "label": label, "grid": grid, "upstream": None}
    if target == "metal":
        slots = MB_OUTPUTS[stage]
        device, raw = _upstream(
            context["work"],
            context["reference"],
            entry,
            label,
            payloads,
            grid,
            set(slots.values()),
        )
        for name, (output_dtype, words) in _stage_expected(dtype, stage, level).items():
            assert raw[slots[name]] == _metal_bytes(output_dtype, words), (label, name)
        record["upstream"] = {"device": device, "matched": True}
    return record


@pytest.mark.parametrize("dtype", DTYPES)
def test_current_mlx_multi_block_sort_native_parity(
    current_mlx, metal_reference, tmp_path, dtype
):
    """Chain sort -> (partition -> merge) per level exactly as ``sort.cpp``."""
    root, target = current_mlx
    bn = _multi_block_threads(dtype)
    entries = {
        stage: f"{stage}_mbsort_{dtype}_uint32_bn{bn}_tn{N_PER_THREAD}"
        for stage in MB_STAGES
    }
    with tempfile.TemporaryDirectory(prefix=".current-mbsort-", dir=root) as directory:
        work = Path(directory)
        try:
            translated = {
                stage: _translate(root, work, target, entry)
                for stage, entry in entries.items()
            }
            context = {
                "target": target,
                "dtype": dtype,
                "entries": entries,
                "translated": translated,
                "executor": _executor(target),
                "work": work,
                "reference": metal_reference,
            }
            evidence = []
            try:
                for name in _mb_cases(dtype):
                    data, stages = _mb_pipeline(dtype, name)
                    case = data["case"]
                    records = [
                        _run_stage(
                            context,
                            "sort",
                            f"{name}-sort",
                            case,
                            data["storage"],
                            stages[0],
                        )
                    ]
                    previous = stages[0]
                    for level in stages[1:]:
                        # Each stage reads the previous stage's outputs, which
                        # were just verified word for word.
                        for stage in ("partition", "merge"):
                            records.append(
                                _run_stage(
                                    context,
                                    stage,
                                    f"{name}-{stage}-{level['mergeTiles']}",
                                    case,
                                    previous,
                                    level,
                                )
                            )
                        previous = level
                    size = case["size"]
                    for row_index, row in enumerate(data["rows"]):
                        start = row_index * size
                        assert previous["indices"][start : start + size] == (
                            _stable_order(dtype, row)
                        )
                    evidence.append(
                        {
                            "case": name,
                            "size": size,
                            "rows": case["rows"],
                            "sortedStride": case["sorted"],
                            "dispatches": records,
                        }
                    )
            finally:
                runtime = context["executor"].runtime_adapter.runtime
                close = getattr(runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entries": entries,
                        "target": target,
                        "artifactSha256": {
                            stage: translated[stage][2]["sha256"] for stage in entries
                        },
                        "cases": evidence,
                        "comparison": "exact words",
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_sort_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX sort native parity", group="ordering"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/sort",
        "--junitxml=mlx-current-results/sort.xml",
        "demos/integrations/mlx/tests/kernels/test_current_sort.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_sort.py"
    )
