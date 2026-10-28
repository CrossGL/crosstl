"""Prove every current MLX reduce kernel natively on Metal, OpenGL and DirectX.

``reduce.metal`` instantiates 2,396 kernels: ``init_reduce`` for 40 operator
and type pairs, and for 62 operator, input and output type triples
``all_reduce``, ``row_reduce_small``, ``row_reduce_simple``,
``row_reduce_looped``, ``col_reduce_small``, ``col_reduce_longcolumn``,
``col_reduce_looped`` and ``col_reduce_2pass`` with 32- and 64-bit indices
over one, two and five reduction dimensions. Every entry translates with a
threadgroup size MLX's host chooses for it and executes several geometries
on Metal, Mesa EGL and Direct3D 12 WARP.

A model of each kernel follows its thread loops, SIMD-group reductions,
threadgroup combines and guarded loads and stores step by step and gives the
expected words of every case. Reductions of types narrower than eight bytes
use Apple's SIMD-group intrinsics, whose association order is unspecified, so
their inputs make every partial result exact and every fold order give the
same words; eight-byte types reduce with the kernel's explicit shuffle loop,
which the model follows in order. The model is checked against an
independent fold of each output's elements.
"""

import functools
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels import (
    test_reduce_complete_opengl as complete_opengl,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    FLOAT_DTYPES,
    INF,
    INTEGER_DTYPES,
    NAN,
    _below,
    _bits,
    _complex_mul,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _draws as _scan_draws,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _float_add,
    _float_mul,
    _is_nan,
    _limits,
    _real,
    _signed,
    _wrap,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    METAL_FORMATS,
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/reduce.metal"
SOURCE_SHA256 = "f1e410ab635eaa940ec195461069cbb013ab88ab0d1f8d5dd8790d30b32c454a"
HEADERS = {
    f"mlx/backend/metal/kernels/{name}": digest
    for name, digest in (
        (
            "reduce.h",
            "2b4ca93fd1f43ee91129440359b18c0fa49bc70098bd4ca3033c6550b2c07d94",
        ),
        (
            "reduction/ops.h",
            "78d06730fc9564a73944e7f1fe3897d25c8789b28a939bf418e1968db311da41",
        ),
        (
            "reduction/reduce_all.h",
            "44ea27207c4d976524960ae615de4889c8e5998bf5e88b88850398cc293b1abb",
        ),
        (
            "reduction/reduce_col.h",
            "9bb7b8b15612230c5f91cea3220048d09219809dbbc1adab4b2dabfd44315810",
        ),
        (
            "reduction/reduce_init.h",
            "c5ef9c65886f9aba64a5df03607cf175950fd99237ca068bd6fcaec72d63d519",
        ),
        (
            "reduction/reduce_row.h",
            "8045c77cb0dde4f97a77ddcdcfad825ccf855928d31369a1dc1832629a63da50",
        ),
        (
            "defines.h",
            "a6c2c824e855078cd092fd9592457616f995755f0e865475c3f006a3f5d6bf1d",
        ),
        ("utils.h", "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"),
        (
            "atomic.h",
            "4c35ea2798a2335502865247aee878149fc9ada0d7e84c05d771baef0c7fcc60",
        ),
        (
            "complex.h",
            "16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1",
        ),
        ("bf16.h", "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb"),
        (
            "bf16_math.h",
            "1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496",
        ),
    )
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_REDUCE_FAMILY"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_REDUCE_FAMILY_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_REDUCE_FAMILY_SHARD_COUNT"
CI_SHARD_COUNT = 12
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/reduce.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# ---------------------------------------------------------------------------
# reduce.metal's instantiation table
# ---------------------------------------------------------------------------

# instantiate_init_reduce rows: (op, dtype).
INIT_PAIRS = (
    ("and", "bool_"),
    ("or", "bool_"),
    *(
        (op, dtype)
        for op in ("sum", "prod")
        for dtype in ("int32", "int64", "float16", "bfloat16", "float32", "complex64")
    ),
    *(
        (op, dtype)
        for op in ("min", "max")
        for dtype in (
            "bool_",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "bfloat16",
            "float32",
            "complex64",
        )
    ),
)
# instantiate_reduce_functions rows: (op, input dtype, output dtype).
REDUCTIONS = (
    *(
        (op, itype, "bool_")
        for op in ("and", "or")
        for itype in (
            "bool_",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
            "float32",
        )
    ),
    *(
        (op, itype, otype)
        for op in ("sum", "prod")
        for itype, otype in (
            ("uint8", "int32"),
            ("uint16", "uint32"),
            ("uint32", "uint32"),
            ("uint64", "uint64"),
            ("int8", "int32"),
            ("int16", "int32"),
            ("int32", "int32"),
            ("int64", "int64"),
            ("float16", "float16"),
            ("bfloat16", "bfloat16"),
            ("float32", "float32"),
            ("complex64", "complex64"),
        )
    ),
    *(
        (op, dtype, dtype)
        for op in ("min", "max")
        for dtype in (
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "bfloat16",
            "float32",
            "complex64",
        )
    ),
)
DIMS = (1, 2, 5)
# Kernel kind -> (name prefixes by 32/64-bit index, dimensional, tiled).
KINDS = {
    "row_small": ("row_reduce_small_", "row_reduce_small_large_"),
    "row_looped": ("row_reduce_looped_", "row_reduce_looped_large_"),
    "col_small": ("col_reduce_small_", "col_reduce_small_large_"),
    "col_longcolumn": ("col_reduce_longcolumn_", "col_reduce_longcolumn_large_"),
    "col_looped": ("col_reduce_looped_", "col_reduce_looped_large_"),
    "col_2pass": ("col_reduce_2pass_", "col_reduce_2pass_large_"),
}
# The threadgroup MLX's host (backend/metal/reduce.cpp) launches each kernel
# with for the geometries this gate executes. ``all_reduce`` takes 64 threads
# for 129..256 elements per threadgroup row, the looped and simple row
# kernels 128 threads for rows of 513..1024 elements, ``col_reduce_small``
# 32 x 8 for strides of at least 125 columns and 8..31 reduced rows,
# ``col_reduce_longcolumn`` its stride (here 7) x 32 and the tiled column
# kernels 8 SIMD-groups.
WORKGROUPS = {
    "init": (32, 1, 1),
    "all": (64, 1, 1),
    "row_small": (32, 1, 1),
    "row_simple": (128, 1, 1),
    "row_looped": (128, 1, 1),
    "col_small": (32, 8, 1),
    "col_longcolumn": (7, 32, 1),
    "col_looped": (256, 1, 1),
    "col_2pass": (256, 1, 1),
}
SIMD_SIZE = 32
# Kernels without SIMD-group operations translate without software subgroups.
SIMD_FREE_KINDS = ("init", "col_small", "col_longcolumn")


def _reduction_entries(op, itype, otype):
    """The 38 entries of one ``instantiate_reduce_functions`` row."""
    name = f"{op}{itype}"
    rows = [(f"all_reduce_{name}", "all", False, None)]
    for kind in ("row_small", "row_looped"):
        small, large = KINDS[kind]
        for dim in DIMS:
            rows.append((f"{small}{dim}_reduce_{name}", kind, False, dim))
            rows.append((f"{large}{dim}_reduce_{name}", kind, True, dim))
    rows.append((f"row_reduce_simple_{name}", "row_simple", False, None))
    for kind in ("col_small", "col_longcolumn", "col_looped", "col_2pass"):
        small, large = KINDS[kind]
        tile = "_32_32" if kind in ("col_looped", "col_2pass") else ""
        for dim in DIMS:
            rows.append((f"{small}{dim}{tile}_reduce_{name}", kind, False, dim))
            rows.append((f"{large}{dim}{tile}_reduce_{name}", kind, True, dim))
    return [
        (
            entry,
            {
                "kind": kind,
                "op": op,
                "itype": itype,
                "otype": otype,
                "large": large,
                "dims": dims,
            },
        )
        for entry, kind, large, dims in rows
    ]


INFO = {
    f"init_reduce_{op}{dtype}": {
        "kind": "init",
        "op": op,
        "itype": dtype,
        "otype": dtype,
        "large": False,
        "dims": None,
    }
    for op, dtype in INIT_PAIRS
}
for _row in REDUCTIONS:
    INFO.update(_reduction_entries(*_row))
ENTRIES = tuple(sorted(INFO))


def _workgroup(entry):
    return list(WORKGROUPS[INFO[entry]["kind"]])


def _options(target, entry):
    if target == "metal" or INFO[entry]["kind"] in SIMD_FREE_KINDS:
        return {}
    options = {"software_subgroup_width": SIMD_SIZE}
    if target == "directx":
        # Lanes past the SIMD-group keep their own value, as Metal's
        # simd_shuffle_down does; float32 additions keep IEEE semantics.
        options["relative_wave_shuffle_out_of_range"] = "self"
        options["binary32_additive_profile"] = "native-precise"
    return options


def _index_assertions(target):
    """OpenGL's 32-bit indexing preconditions for the 64-bit index kernels.

    These are the complete-family assertions, except that the corpus commit
    widens ``col_reduce_longcolumn``'s lane offset: ``in += in_idx +
    IdxT(lid.x)``.
    """
    if target != "opengl":
        return []
    expressions = [
        (
            (
                "in_ + in_idx + int64(lid.x)"
                if expression == "in_ + in_idx + lid.x"
                else expression
            ),
            low,
            high,
        )
        for expression, low, high in complete_opengl.INDEX_RANGE_ASSERTIONS
    ]
    assert len({expression for expression, *_ in expressions}) == len(expressions)
    return [
        dict(source=SOURCE, expression=expression, minimum=low, maximum=high)
        for expression, low, high in expressions
    ]


def _config(directory, target, entry):
    """The translation of one entry, as the gate and its contract run it."""
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: tuple(_workgroup(entry))}},
        index_range_assertions=_index_assertions(target),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 1024,
                "max_template_materialization_work": 1048576,
                "target_options": {target: _options(target, entry)},
            }
        },
    )


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"reduce family shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


# ---------------------------------------------------------------------------
# Element arithmetic with Metal semantics
# ---------------------------------------------------------------------------


def _quiet_nan(dtype):
    return (
        (np.float32(NAN), np.float32(NAN))
        if dtype == "complex64"
        else _real(dtype, NAN)
    )


def _init(info):
    """``Op::init`` of the entry's output type."""
    op, dtype = info["op"], info["otype"]
    if op in ("and", "or"):
        return op == "and"
    if op in ("sum", "prod"):
        one = 1 if op == "prod" else 0
        if dtype == "complex64":
            return (np.float32(one), np.float32(0.0))
        return _real(dtype, one) if dtype in FLOAT_DTYPES else one
    # Min starts at Limits<U>::max, Max at Limits<U>::min.
    return _limits(dtype, op == "min")


def _convert(info, value):
    """``cast_to<U>(T)``: integral promotion, or a bool from any nonzero bit."""
    if info["otype"] != "bool_":
        return value
    if info["itype"] in FLOAT_DTYPES:
        return bool(_is_nan(value) or value != 0)
    return bool(value)


def _complex_less(a, b):
    """complex.h ``a < b``: ``b > a`` lexicographically."""
    return b[0] > a[0] or (b[0] == a[0] and b[1] > a[1])


def _complex_extremum(minimum):
    def combine(a, b):
        real_nan = _is_nan(a[0]) or _is_nan(b[0])
        imag_nan = _is_nan(a[1]) or _is_nan(b[1])
        if not real_nan and not imag_nan:
            keep = _complex_less(a, b) if minimum else _complex_less(b, a)
            return a if keep else b
        nan = np.float32(NAN)
        if real_nan and imag_nan:
            return (nan, nan)
        if real_nan:
            first = a[1] < b[1] if minimum else a[1] > b[1]
            return (nan, a[1] if first else b[1])
        first = a[0] < b[0] if minimum else a[0] > b[0]
        return (a[0] if first else b[0], nan)

    return combine


def _operator(info):
    """``Op::operator()(a, b)`` of reduction/ops.h for the output type."""
    op, dtype = info["op"], info["otype"]
    if op == "and":
        return lambda a, b: bool(a) and bool(b)
    if op == "or":
        return lambda a, b: bool(a) or bool(b)
    if op == "sum":
        if dtype in INTEGER_DTYPES:
            return lambda a, b: _wrap(dtype, a + b)
        if dtype == "complex64":
            return lambda a, b: (
                _float_add("float32", a[0], b[0]),
                _float_add("float32", a[1], b[1]),
            )
        return lambda a, b: _float_add(dtype, a, b)
    if op == "prod":
        if dtype in INTEGER_DTYPES:
            return lambda a, b: _wrap(dtype, a * b)
        if dtype == "complex64":
            return _complex_mul
        return lambda a, b: _float_mul(dtype, a, b)
    minimum = op == "min"
    if dtype == "complex64":
        return _complex_extremum(minimum)
    if dtype in INTEGER_DTYPES:
        return lambda a, b: a if (a < b if minimum else a > b) else b

    def extremum(a, b):
        if _is_nan(a) or _is_nan(b):
            return _quiet_nan(dtype)
        return a if (a < b if minimum else a > b) else b

    return extremum


def _simd_total(op, lanes):
    """Lane 0 of ``simd_reduce``: ``val = op(val, simd_shuffle_down(val, d))``.

    Eight-byte types run exactly this loop. Narrower types call Apple's
    SIMD-group intrinsics, whose order is unspecified; their inputs make every
    order give the same words, so the same tree models them.
    """
    values = list(lanes)
    assert len(values) == SIMD_SIZE
    for delta in (16, 8, 4, 2, 1):
        values = [op(values[lane], values[lane + delta]) for lane in range(delta)]
    return values[0]


# ---------------------------------------------------------------------------
# Case geometries
# ---------------------------------------------------------------------------

# Gaps between rows and blocks hold decoys that no reduction may read.
ROW_GAP, BLOCK_GAP = 2, 3
GUARD_COUNT = 16
# init_reduce writes three threadgroups of outputs.
INIT_GROUPS = 3
# all_reduce: one threadgroup over 203 elements (MLX's single pass, whose last
# active thread reads three), and the first pass of MLX's two-pass reduction
# of 16,589 elements: 128 rows of 130, the last one cut to 79 by in_size.
ALL_CASES = {
    "single": {"in_size": 203, "row_size": 203, "grid": [1, 1, 1]},
    "rows": {"in_size": 16589, "row_size": 130, "grid": [1, 128, 1]},
}
# Row kernels: output dims, reduced non-row dims per NDIMS and the row.
# row_reduce_small reduces each row in one thread when there are at most 8
# non-row reductions and in one SIMD-group otherwise.
ROW_CASES = {
    "row_small": {
        "thread": {
            "row": 7,
            "out": [2, 32],
            "reduce": {1: [3], 2: [2, 2], 5: [2, 2, 2]},
            "grid": [1, 2, 1],
        },
        "simd": {
            "row": 5,
            "out": [3],
            "reduce": {1: [40], 2: [5, 8], 5: [2, 4, 5]},
            "grid": [1, 3, 1],
        },
    },
    "row_looped": {
        "rows": {
            "row": 522,
            "out": [3],
            "reduce": {1: [2], 2: [2, 2], 5: [1, 2, 2]},
            "grid": [1, 3, 1],
        },
    },
}
# row_reduce_simple: 33 contiguous rows of 522 elements; the last threadgroup
# moves back to rows 29..32 and rewrites three outputs.
ROW_SIMPLE = {"row": 522, "rows": 33, "grid": [1, 9, 1]}
# Column kernels: outer outputs, reduced dims (the reduced axis last) per
# NDIMS and the contiguous columns.
COL_CASES = {
    # 32 x 8 threads over 130 columns: the second x block keeps one partial
    # four-column thread and every other thread returns early.
    "col_small": {
        "stride": 130,
        "out": [2],
        "reduce": {1: [12], 2: [3, 5], 5: [2, 2, 5]},
        "grid": [2, 2, 1],
    },
    # 7 x 32 threads, 32 outer blocks over 1,100 rows; writes the
    # [32, out_size] intermediate.
    "col_longcolumn": {
        "stride": 7,
        "out": [2],
        "reduce": {1: [1100], 2: [4, 275], 5: [2, 2, 275]},
        "grid": [2, 1, 32],
    },
    # 32 x 32 tiles over 70 columns: the third tile has six.
    "col_looped": {
        "stride": 70,
        "out": [2],
        "reduce": {1: [40], 2: [2, 20], 5: [2, 2, 10]},
        "grid": [3, 2, 1],
    },
    # 32 outer blocks of 32 x 32 tiles over 260 rows and 33 columns; writes
    # the [32, out.size()] intermediate.
    "col_2pass": {
        "stride": 33,
        "out": [2],
        "reduce": {1: [260], 2: [2, 130], 5: [2, 2, 65]},
        "grid": [2, 64, 1],
    },
}
CASES = {
    "init": ("grid",),
    "all": tuple(ALL_CASES),
    "row_small": tuple(ROW_CASES["row_small"]),
    "row_simple": ("rows",),
    "row_looped": tuple(ROW_CASES["row_looped"]),
    "col_small": ("columns",),
    "col_longcolumn": ("columns",),
    "col_looped": ("columns",),
    "col_2pass": ("columns",),
}


def _cases(entry):
    return CASES[INFO[entry]["kind"]]


def _elem_to_loc(elem, shape, strides):
    """utils.h ``elem_to_loc``."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if elem <= 0:
            break
        location += (elem % extent) * stride
        elem //= extent
    return location


def _blocked_strides(shape, inner, gap):
    """Row-major strides over ``shape`` of blocks of ``inner`` elements."""
    strides, step = [], inner
    for extent in reversed(shape):
        strides.append(step)
        step = step * extent + gap
    return list(reversed(strides)), step


def _vector(dtype, values):
    """A metadata vector; MLX pushes one zero for an empty one."""
    return (dtype, list(values) or [0])


def _layout(entry, case):
    """Return the dispatch, metadata, input extent and logical outputs.

    ``groups`` lists, for every logical output of the reduction the kernel
    takes part in, the input indices it reduces; ``parts`` maps every output
    element the kernel writes to the input indices it must combine.
    """
    info = INFO[entry]
    kind = info["kind"]
    if kind == "init":
        count = INIT_GROUPS * WORKGROUPS["init"][0]
        return {
            "grid": [INIT_GROUPS, 1, 1],
            "metadata": {},
            "size": 0,
            "groups": [],
            "parts": {index: [] for index in range(count)},
        }
    if kind == "all":
        spec = ALL_CASES[case]
        size, row = spec["in_size"], spec["row_size"]
        rows = [
            list(range(start, min(start + row, size))) for start in range(0, size, row)
        ]
        assert len(rows) == spec["grid"][1]
        return {
            "grid": spec["grid"],
            "metadata": {"in_size": ("uint64", [size]), "row_size": ("uint64", [row])},
            "size": size,
            "groups": rows,
            "parts": dict(enumerate(rows)),
        }
    if kind == "row_simple":
        row, count = ROW_SIMPLE["row"], ROW_SIMPLE["rows"]
        rows = [list(range(index * row, (index + 1) * row)) for index in range(count)]
        return {
            "grid": ROW_SIMPLE["grid"],
            "metadata": {
                "reduction_size": ("uint64", [row]),
                "out_size": ("int64", [count]),
            },
            "size": row * count,
            "groups": rows,
            "parts": dict(enumerate(rows)),
        }
    if kind in ("row_small", "row_looped"):
        spec = ROW_CASES[kind][case]
        row, reduce = spec["row"], spec["reduce"][info["dims"]]
        reduce_strides, block = _blocked_strides(reduce, row + ROW_GAP, BLOCK_GAP)
        out_strides, size = _blocked_strides(spec["out"], block, BLOCK_GAP)
        count = math.prod(reduce)
        groups = []
        for out in range(math.prod(spec["out"])):
            base = _elem_to_loc(out, spec["out"], out_strides)
            groups.append(
                [
                    base + _elem_to_loc(r, reduce, reduce_strides) + column
                    for r in range(count)
                    for column in range(row)
                ]
            )
        return {
            "grid": spec["grid"],
            "metadata": {
                "row_size": ("int64", [row]),
                "non_row_reductions": ("int64", [count]),
                "shape": _vector("int32", spec["out"]),
                "strides": _vector("int64", out_strides),
                "ndim": ("int32", [len(spec["out"])]),
                "reduce_shape": _vector("int32", reduce),
                "reduce_strides": _vector("int64", reduce_strides),
                "reduce_ndim": ("int32", [len(reduce)]),
            },
            "size": size,
            "groups": groups,
            "parts": dict(enumerate(groups)),
        }
    spec = COL_CASES[kind]
    stride, reduce = spec["stride"], spec["reduce"][info["dims"]]
    # The reduced axis steps by whole rows of ``stride`` columns; the other
    # reduced dims and the outer outputs leave gaps.
    inner_strides, block = _blocked_strides(reduce[:-1], reduce[-1] * stride, BLOCK_GAP)
    reduce_strides = inner_strides + [stride]
    out_strides, size = _blocked_strides(spec["out"], block, BLOCK_GAP)
    total = math.prod(reduce)
    outer = math.prod(spec["out"])
    groups = {}
    for out in range(outer):
        base = _elem_to_loc(out, spec["out"], out_strides)
        for column in range(stride):
            groups[out, column] = [
                base + _elem_to_loc(r, reduce, reduce_strides) + column
                for r in range(total)
            ]
    metadata = {
        "reduction_size": ("uint64", [reduce[-1]]),
        "reduction_stride": (
            "uint64" if kind == "col_longcolumn" else "int64",
            [stride],
        ),
        "shape": _vector("int32", spec["out"]),
        "strides": _vector("int64", out_strides),
        "ndim": ("int32", [len(spec["out"])]),
        "reduce_shape": _vector("int32", reduce),
        "reduce_strides": _vector("int64", reduce_strides),
        "reduce_ndim": ("int32", [len(reduce)]),
        "non_col_reductions": ("uint64", [math.prod(reduce[:-1])]),
    }
    parts = {}
    if kind in ("col_small", "col_looped"):
        for (out, column), indices in groups.items():
            parts[out * stride + column] = indices
    else:
        # Each outer block takes the rows whose 32-row tile, counted modulo
        # 1,024 rows, is its index.
        blocks = 32
        width = outer * stride
        metadata["out_size"] = (
            "uint64",
            [width if kind == "col_longcolumn" else outer],
        )
        for block_index in range(blocks):
            for (out, column), indices in groups.items():
                rows = [
                    index
                    for r, index in enumerate(indices)
                    if (r % (blocks * 32)) // 32 == block_index
                ]
                parts[block_index * width + out * stride + column] = rows
    return {
        "grid": spec["grid"],
        "metadata": metadata,
        "size": size,
        "groups": list(groups.values()),
        "parts": parts,
    }


# ---------------------------------------------------------------------------
# Kernel models
# ---------------------------------------------------------------------------


class _Run:
    """One dispatch of the model: typed loads, recorded reads and writes."""

    def __init__(self, entry, case, values):
        self.info = INFO[entry]
        self.layout = _layout(entry, case)
        self.meta = {name: data for name, (_, data) in self.layout["metadata"].items()}
        self.values = values
        self.op = _operator(self.info)
        self.init = _init(self.info)
        self.reads = set()
        self.writes = {}

    def scalar(self, name):
        return self.meta[name][0]

    def load(self, index):
        assert 0 <= index < len(self.values), index
        self.reads.add(index)
        return _convert(self.info, self.values[index])

    def store(self, index, value):
        if index in self.writes:
            # row_reduce_simple's last threadgroup rewrites equal outputs.
            assert _words(self.info["otype"], self.writes[index]) == _words(
                self.info["otype"], value
            ), index
        self.writes[index] = value

    def location(self, r):
        shape = self.meta["reduce_shape"][: self.scalar("reduce_ndim")]
        strides = self.meta["reduce_strides"][: self.scalar("reduce_ndim")]
        if self.info["dims"] == 1:
            # LoopedElemToLoc<1, IdxT, false> steps by reduce_strides[0] only.
            return r * self.meta["reduce_strides"][0]
        return _elem_to_loc(r, shape, strides)

    def out_location(self, out):
        ndim = self.scalar("ndim")
        return _elem_to_loc(out, self.meta["shape"][:ndim], self.meta["strides"][:ndim])

    def groups(self):
        x, y, z = self.layout["grid"]
        return [(gx, gy, gz) for gz in range(z) for gy in range(y) for gx in range(x)]

    def threadgroup_total(self, totals):
        """reduce_row.h ``threadgroup_reduce`` of one register per thread."""
        sums = [
            _simd_total(self.op, totals[start : start + SIMD_SIZE])
            for start in range(0, len(totals), SIMD_SIZE)
        ]
        if len(sums) == 1:
            return sums[0]
        return _simd_total(self.op, sums + [self.init] * (SIMD_SIZE - len(sums)))


def _model_init(run):
    for index in range(INIT_GROUPS * WORKGROUPS["init"][0]):
        run.store(index, run.init)


def _model_all(run):
    width = WORKGROUPS["all"][0]
    in_size, row_size = run.scalar("in_size"), run.scalar("row_size")
    for _, gy, _ in run.groups():
        totals = []
        for lid in range(width):
            total = run.init
            start = gy * row_size
            actual = row_size if start + row_size <= in_size else in_size - start
            assert actual > 0
            blocks = actual // (width * 4)
            extra = actual - blocks * width * 4 - lid * 4
            pointer = start + lid * 4
            if extra >= 4:
                blocks, extra = blocks + 1, 0
            for _block in range(blocks):
                for i in range(4):
                    total = run.op(run.load(pointer + i), total)
                pointer += width * 4
            for i in range(max(extra, 0)):
                total = run.op(run.load(pointer + i), total)
            totals.append(total)
        run.store(gy, run.threadgroup_total(totals))


def _thread_reduce(run, total, pointer, blocks, extra):
    for _block in range(blocks):
        values = [run.load(pointer + j) for j in range(4)]
        for value in values:
            total = run.op(value, total)
        pointer += 4
    for _ in range(extra):
        total = run.op(run.load(pointer), total)
        pointer += 1
    return total


def _model_row_small(run):
    width = WORKGROUPS["row_small"][0]
    row, count = run.scalar("row_size"), run.scalar("non_row_reductions")
    blocks, extra = row // 4, row % 4
    grid = run.layout["grid"]
    if (count < 32 and row <= 8) or count <= 8:
        # One thread per output; tsize.x threads per grid row.
        for gx, gy, gz in run.groups():
            assert gz == 0
            for lid in range(width):
                out = gx * width + lid + grid[0] * width * gy
                base = run.out_location(out)
                total = run.init
                for r in range(count):
                    total = _thread_reduce(
                        run, total, base + run.location(r), blocks, extra
                    )
                run.store(out, total)
        return
    for _, gy, gz in run.groups():
        out = gy + grid[1] * gz
        base = run.out_location(out)
        lanes = []
        for lane in range(SIMD_SIZE):
            total = run.init
            for r in range(lane, count, SIMD_SIZE):
                total = _thread_reduce(
                    run, total, base + run.location(r), blocks, extra
                )
            lanes.append(total)
        run.store(out, _simd_total(run.op, lanes))


def _per_thread_row_reduce(run, pointers, blocks, extra, width, lid):
    """reduce_row.h ``per_thread_row_reduce`` over explicit row pointers."""
    totals = [run.init] * len(pointers)
    pointers = list(pointers)
    for _block in range(blocks):
        for j in range(len(pointers)):
            for i in range(4):
                totals[j] = run.op(run.load(pointers[j] + i), totals[j])
            pointers[j] += width * 4
    index = lid * 4
    for j in range(len(pointers)):
        for i in range(4 if index + 4 <= extra else max(extra - index, 0)):
            totals[j] = run.op(run.load(pointers[j] + i), totals[j])
    return totals


def _model_row_simple(run):
    width, writes = WORKGROUPS["row_simple"][0], 4
    row, out_size = run.scalar("reduction_size"), run.scalar("out_size")
    grid = run.layout["grid"]
    blocks = row // (width * 4)
    extra = row - blocks * width * 4
    for _, gy, gz in run.groups():
        out = writes * (gy + grid[1] * gz)
        if out + writes > out_size:
            out = out_size - writes
        registers = []
        for lid in range(width):
            pointers = [out * row + lid * 4 + j * row for j in range(writes)]
            registers.append(
                _per_thread_row_reduce(run, pointers, blocks, extra, width, lid)
            )
        for j in range(writes):
            run.store(
                out + j, run.threadgroup_total([values[j] for values in registers])
            )


def _model_row_looped(run):
    width = WORKGROUPS["row_looped"][0]
    row, count = run.scalar("row_size"), run.scalar("non_row_reductions")
    grid = run.layout["grid"]
    blocks = row // (width * 4)
    extra = row - blocks * width * 4
    for _, gy, gz in run.groups():
        out = gy + grid[1] * gz
        base = run.out_location(out)
        totals = []
        for lid in range(width):
            total = run.init
            for r in range(count):
                pointer = base + lid * 4 + run.location(r)
                (row_total,) = _per_thread_row_reduce(
                    run, [pointer], blocks, extra, width, lid
                )
                # Rows aggregate as op(total, row_total).
                total = run.op(total, row_total)
            totals.append(total)
        run.store(out, run.threadgroup_total(totals))


def _column_registers(run, base, column, rows, count=4):
    """Accumulate ``count`` columns over ``rows`` with the guarded loads."""
    stride = run.scalar("reduction_stride")
    totals = [run.init] * count
    safe = column + count <= stride
    for r in rows:
        pointer = base + column + run.location(r)
        if safe:
            values = [run.load(pointer + i) for i in range(count)]
        else:
            values = [
                run.load(pointer + i) if column + i < stride else run.init
                for i in range(count)
            ]
        for i in range(count):
            totals[i] = run.op(values[i], totals[i])
    return totals


def _write_columns(run, out_base, column, totals):
    stride = run.scalar("reduction_stride")
    for i, value in enumerate(totals):
        if column + i < stride:
            run.store(out_base + column + i, value)


def _model_col_small(run):
    lanes_x, rows_y, _ = WORKGROUPS["col_small"]
    grid = run.layout["grid"]
    stride = run.scalar("reduction_stride")
    total_rows = run.scalar("non_col_reductions") * run.scalar("reduction_size")
    for gx, gy, gz in run.groups():
        out = gy + grid[1] * gz
        base = run.out_location(out)
        for lx in range(lanes_x):
            column = gx * lanes_x * 4 + lx * 4
            if column >= stride:
                continue
            shared = [
                _column_registers(run, base, column, range(ly, total_rows, rows_y))
                for ly in range(rows_y)
            ]
            totals = shared[0]
            for j in range(1, rows_y):
                totals = [run.op(shared[j][i], totals[i]) for i in range(4)]
            _write_columns(run, out * stride, column, totals)


def _model_col_longcolumn(run):
    lanes_x, rows_y, _ = WORKGROUPS["col_longcolumn"]
    grid = run.layout["grid"]
    stride, out_size = run.scalar("reduction_stride"), run.scalar("out_size")
    assert lanes_x == stride
    total_rows = run.scalar("non_col_reductions") * run.scalar("reduction_size")
    step = rows_y * grid[2]
    for gx, gy, gz in run.groups():
        out = gx + grid[0] * gy
        base = run.out_location(out)
        for lx in range(lanes_x):
            shared = []
            for ly in range(rows_y):
                total = run.init
                for r in range(gz * rows_y + ly, total_rows, step):
                    total = run.op(run.load(base + lx + run.location(r)), total)
                shared.append(total)
            total = shared[0]
            for value in shared[1:]:
                total = run.op(total, value)
            run.store(gz * out_size + out * stride + lx, total)


def _model_col_tiles(run, two_pass):
    """col_reduce_looped and col_reduce_2pass with BM = BN = 32."""
    grid = run.layout["grid"]
    stride = run.scalar("reduction_stride")
    total_rows = run.scalar("non_col_reductions") * run.scalar("reduction_size")
    for gx, gy, gz in run.groups():
        full = gy + grid[1] * gz
        if two_pass:
            outer = run.scalar("out_size")
            block, out = full // outer, full % outer
            rows = lambda y: range(y + block * 32, total_rows, 32 * 32)  # noqa: E731
        else:
            out = full
            rows = lambda y: range(y, total_rows, 32)  # noqa: E731
        base = run.out_location(out)
        # shared[offset.y * 32 + offset.x + i] for 8 x 4 columns by 32 rows.
        shared = {}
        for lid in range(256):
            offset_x, offset_y = (lid % 8) * 4, lid // 8
            totals = _column_registers(run, base, 32 * gx + offset_x, rows(offset_y))
            for i, value in enumerate(totals):
                shared[offset_y * 32 + offset_x + i] = value
        for group in range(8):
            out_column = 32 * gx + group * 4
            totals = [
                _simd_total(
                    run.op, [shared[lane * 32 + group * 4 + i] for lane in range(32)]
                )
                for i in range(4)
            ]
            _write_columns(
                run, (full if two_pass else out) * stride, out_column, totals
            )


MODELS = {
    "init": _model_init,
    "all": _model_all,
    "row_small": _model_row_small,
    "row_simple": _model_row_simple,
    "row_looped": _model_row_looped,
    "col_small": _model_col_small,
    "col_longcolumn": _model_col_longcolumn,
    "col_looped": lambda run: _model_col_tiles(run, two_pass=False),
    "col_2pass": lambda run: _model_col_tiles(run, two_pass=True),
}


def _simulate(entry, case, values):
    run = _Run(entry, case, values)
    MODELS[run.info["kind"]](run)
    return run


# ---------------------------------------------------------------------------
# Input domains
# ---------------------------------------------------------------------------

PATTERNS = 6
DECOY_COUNT = 4
# Floating sums: multiples of 1/4 whose absolute values add up to at most
# this many quarters, so every partial sum of every order is exact.
SUM_QUARTERS = {"bfloat16": 32, "float16": 256, "float32": 2**18, "complex64": 2**18}
# Floating products: powers of two whose exponents add up to at most this
# much in each direction, so every partial product stays normal and exact.
PRODUCT_EXPONENTS = {"float16": 14, "bfloat16": 100, "float32": 100, "complex64": 60}
# Distinct finite extrema operands; patterns add infinities and NaN, and each
# group picks one zero sign.
EXTREMA_POOLS = {
    "float32": (
        -3.0e38,
        -1.5e10,
        -2.5,
        -1.0,
        -0.5,
        1.0e-30,
        0.5,
        1.0,
        1.5,
        2.5,
        3.0e38,
    ),
    "float16": (
        -65504.0,
        -1000.0,
        -2.5,
        -1.0,
        -0.5,
        2.0**-14,
        0.5,
        1.0,
        1.5,
        2.5,
        65504.0,
    ),
    "bfloat16": (
        -(2.0**100),
        -1024.0,
        -2.5,
        -1.0,
        -0.5,
        2.0**-100,
        0.5,
        1.0,
        1.5,
        2.5,
        2.0**100,
    ),
}
COMPLEX_PARTS = (-2.5, -1.0, 0.5, 1.0, 2.5, 1.0e30)


def _draws(*key):
    return _scan_draws("reduce", *key)


def _signed_limits(dtype):
    return _limits(dtype, False), _limits(dtype, True)


def _truth(itype, draws, value):
    """A true (nonzero) or false (zero) input of ``itype``."""
    if itype == "bool_":
        return value
    if itype in INTEGER_DTYPES:
        if not value:
            return 0
        low, high = _signed_limits(itype)
        choice = (low, high, 1, -1, _wrap(itype, _bits(draws, 62) | 1))[
            _below(draws, 5)
        ]
        return choice or 1
    if not value:
        return _real(itype, -0.0 if next(draws) & 1 else 0.0)
    pool = (NAN, INF, -INF, 1.0, -1.0, 2.0**-14, -2.5)
    return _real(itype, pool[_below(draws, len(pool))])


def _logical_values(op, itype, draws, pattern, length):
    """And keeps true, Or false (``neutral``) in two of every six outputs."""
    neutral = op == "and"
    values = [neutral] * length
    if pattern == 1:
        values[_below(draws, length)] = not neutral
    elif pattern == 2:
        values = [bool(next(draws) & 1) for _ in range(length)]
    elif pattern == 4:
        values[0] = not neutral
    elif pattern == 5:
        values[-1] = not neutral
    return [_truth(itype, draws, value) for value in values]


def _sparse(draws, length, count):
    positions = list(range(length))
    for index in range(len(positions) - 1, 0, -1):
        other = _below(draws, index + 1)
        positions[index], positions[other] = positions[other], positions[index]
    return positions[: min(count, length)]


def _signed_zero(draws):
    return -0.0 if next(draws) & 1 else 0.0


def _float_sum_values(dtype, draws, pattern, length):
    """Quarters with a bounded absolute sum, signed zeros and specials."""
    values = [_signed_zero(draws) for _ in range(length)]
    budget = SUM_QUARTERS[dtype]
    positions = _sparse(draws, length, 24)
    if pattern == 1:
        return values
    if pattern == 2:
        # Exact cancellation back to zero.
        for first, second in zip(positions[::2], positions[1::2]):
            move = _signed(draws, max(1, budget // 32)) or 1
            values[first], values[second] = move / 4, -move / 4
        return values
    for position in positions:
        move = _signed(draws, max(1, budget // (2 * len(positions))))
        values[position] = move / 4 if move else _signed_zero(draws)
    assert sum(abs(value) for value in values) * 4 <= budget
    special = positions[0]
    if pattern == 3:
        values[special] = INF if next(draws) & 1 else -INF
    elif pattern == 4 and length > 1:
        values[special], values[positions[1]] = INF, -INF
    elif pattern == 5:
        values[special] = NAN
    return values


def _integer_sum_values(itype, otype, draws, pattern, length):
    low, high = _signed_limits(otype)
    bits, _ = INTEGER_DTYPES[itype]
    if not INTEGER_DTYPES[otype][1] or bits < INTEGER_DTYPES[otype][0]:
        # Unsigned sums wrap; narrow sources cannot overflow int32 here.
        assert length * 2**bits < 2**31 or not INTEGER_DTYPES[otype][1]
        pool = _signed_limits(itype)
        return [
            (
                pool[_below(draws, 2)]
                if _below(draws, 8) == 0
                else _wrap(itype, _bits(draws, bits))
            )
            for _ in range(length)
        ]
    # Signed sums: positives add to at most high, negatives to at least low,
    # so every partial sum of every order is defined.
    limit = high // length
    values = [_signed(draws, limit) for _ in range(length)]
    if pattern == 1:
        values = [abs(value) // 2 for value in values]
        values[_below(draws, length)] = high - sum(values)
    elif pattern == 2:
        values = [-(abs(value) // 2) for value in values]
        values[_below(draws, length)] = low - sum(values)
    positive = sum(value for value in values if value > 0)
    negative = sum(value for value in values if value < 0)
    assert positive <= high and negative >= low
    return values


def _float_product_values(dtype, draws, pattern, length):
    """Signed powers of two with bounded exponent sums, zeros and specials."""
    values = [-1.0 if next(draws) & 1 else 1.0 for _ in range(length)]
    budget = PRODUCT_EXPONENTS[dtype]
    for position in _sparse(draws, length, budget // 2):
        values[position] *= 2.0 if next(draws) & 1 else 0.5
    special = _below(draws, length)
    if pattern == 1:
        values[special] = _signed_zero(draws)
    elif pattern == 2:
        values[special] = INF if next(draws) & 1 else -INF
    elif pattern == 3 and length > 1:
        values[special] = INF
        values[(special + 1) % length] = _signed_zero(draws)
    elif pattern == 4:
        values[special] = NAN
    return values


def _integer_product_values(itype, otype, draws, pattern, length):
    low, high = _signed_limits(otype)
    signed_out = INTEGER_DTYPES[otype][1]
    if not signed_out:
        # Unsigned products wrap; odd factors keep them from collapsing to 0.
        bits = INTEGER_DTYPES[itype][0]
        values = [_wrap(itype, _bits(draws, bits) | 1) for _ in range(length)]
        if pattern == 1:
            values[_below(draws, length)] = 0
        return values
    # Signed products stay inside the output type for every order: the
    # magnitudes multiply to at most high and MIN only meets factors of one.
    signed_in = INTEGER_DTYPES[itype][1]
    if pattern == 2:
        values = [1] * length
        values[_below(draws, length)] = _limits(itype, not signed_in)
        return values
    values = [(-1 if signed_in and next(draws) & 1 else 1) for _ in range(length)]
    magnitude = 1
    for position in _sparse(draws, length, 12):
        factor = (2, 3, 5, 7, 11, 13)[_below(draws, 6)]
        if magnitude * factor <= high:
            values[position] *= factor
            magnitude *= factor
    if pattern == 1:
        values[_below(draws, length)] = 0
    assert math.prod(abs(value) for value in values) <= high
    return values


def _complex_product_values(draws, pattern, length):
    """Units times powers of two and a few 1 +- i factors; every product exact."""
    units = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
    values = [units[_below(draws, 4)] for _ in range(length)]
    budget = PRODUCT_EXPONENTS["complex64"]
    for position in _sparse(draws, length, budget // 4):
        scale = 2.0 if next(draws) & 1 else 0.5
        values[position] = (values[position][0] * scale, values[position][1] * scale)
    for position in _sparse(draws, length, 6):
        values[position] = (
            1.0 if next(draws) & 1 else -1.0,
            1.0 if next(draws) & 1 else -1.0,
        )
    if pattern == 1:
        values[_below(draws, length)] = (_signed_zero(draws), _signed_zero(draws))
    elif pattern == 2:
        values[_below(draws, length)] = (NAN, 0.0)
    return values


def _extrema_values(op, dtype, draws, pattern, length):
    if dtype in INTEGER_DTYPES:
        bits = INTEGER_DTYPES[dtype][0]
        low, high = _signed_limits(dtype)
        # The type's extremes appear only in pattern 2, so a decoy extreme
        # changes every other output that read it.
        pool = sorted({low + 1, 0, 1, high - 1} | ({-1} if low else {2}))
        values = []
        while len(values) < length:
            value = (
                pool[_below(draws, len(pool))]
                if _below(draws, 8) == 0
                else _wrap(dtype, _bits(draws, bits))
            )
            if value not in (low, high):
                values.append(value)
        if pattern == 2:
            values[_below(draws, length)] = low if op == "min" else high
        elif pattern == 3:
            values = [values[0]] * length
        return values
    if dtype == "complex64":
        # One zero sign per component and group; complex.h compares
        # lexicographically and NaN pairs absorb every operand.
        reals = COMPLEX_PARTS + (_signed_zero(draws),)
        imags = COMPLEX_PARTS + (_signed_zero(draws),)
        values = [
            (reals[_below(draws, len(reals))], imags[_below(draws, len(imags))])
            for _ in range(length)
        ]
        if pattern == 1:
            values[_below(draws, length)] = (NAN, NAN)
        elif pattern == 2:
            infinite = INF if op == "max" else -INF
            values[_below(draws, length)] = (infinite, values[0][1])
        return values
    # One zero sign per group: Metal keeps the second operand of a tie.
    pool = EXTREMA_POOLS[dtype] + (_signed_zero(draws),)
    values = [pool[_below(draws, len(pool))] for _ in range(length)]
    if pattern == 1:
        values[_below(draws, length)] = NAN
    elif pattern == 2:
        values[_below(draws, length)] = INF if next(draws) & 1 else -INF
    elif pattern == 3:
        values = [values[0]] * length
    return values


def _group_values(info, draws, pattern, length):
    op, itype, otype = info["op"], info["itype"], info["otype"]
    if op in ("and", "or"):
        return _logical_values(op, itype, draws, pattern, length)
    if op == "sum":
        if itype in INTEGER_DTYPES:
            return _integer_sum_values(itype, otype, draws, pattern, length)
        if itype == "complex64":
            real = _float_sum_values("complex64", draws, pattern, length)
            imag = _float_sum_values(
                "complex64", draws, (pattern + 3) % PATTERNS, length
            )
            return list(zip(real, imag))
        return _float_sum_values(itype, draws, pattern, length)
    if op == "prod":
        if itype in INTEGER_DTYPES:
            return _integer_product_values(itype, otype, draws, pattern, length)
        if itype == "complex64":
            return _complex_product_values(draws, pattern, length)
        return _float_product_values(itype, draws, pattern, length)
    return _extrema_values(op, itype, draws, pattern, length)


def _logical(dtype, value):
    """A Python value rounded to ``dtype``."""
    if dtype == "complex64":
        return (np.float32(value[0]), np.float32(value[1]))
    if dtype in FLOAT_DTYPES:
        return _real(dtype, value)
    return value


def _decoy(info):
    """An unread input that would change every finite result if read."""
    op, itype = info["op"], info["itype"]
    if op in ("and", "or"):
        # A false input for And, a true one for Or.
        if itype == "bool_":
            return op == "or"
        if itype in INTEGER_DTYPES:
            return int(op == "or")
        return 1.0 if op == "or" else 0.0
    if op in ("sum", "prod"):
        if itype == "complex64":
            return (3.0, 3.0)
        return 3.0 if itype in FLOAT_DTYPES else 3
    if itype in INTEGER_DTYPES:
        return _limits(itype, op == "max")
    # Every finite extremum moves to an infinity that no pool contains.
    infinite = INF if op == "max" else -INF
    if itype == "complex64":
        return (infinite, infinite)
    return infinite


def _inputs(entry, case):
    """Logical input elements of one case, decoys in every unread position."""
    info = INFO[entry]
    if info["kind"] == "init":
        return []
    layout = _layout(entry, case)
    values = [None] * layout["size"]
    for index, indices in enumerate(layout["groups"]):
        draws = _draws(
            info["kind"], case, info["op"], info["itype"], info["otype"], index
        )
        for position, value in zip(
            indices, _group_values(info, draws, index % PATTERNS, len(indices))
        ):
            assert values[position] is None
            values[position] = value
    decoy = _decoy(info)
    values = [decoy if value is None else value for value in values]
    return [_logical(info["itype"], value) for value in values + [decoy] * DECOY_COUNT]


def _words(dtype, value):
    """Comparable bits of a logical value; every NaN compares equal."""
    if dtype == "complex64":
        return tuple(_words("float32", part) for part in value)
    if dtype == "bool_":
        return bool(value)
    if dtype in INTEGER_DTYPES:
        return _wrap(dtype, int(value))
    if _is_nan(value):
        return "nan"
    if dtype == "float16":
        return int(np.array(value, dtype=np.float16).view(np.uint16))
    return int(np.array(np.float32(value)).view(np.uint32))


def _different(dtype, value):
    """An initial output value whose bits differ from ``value``."""
    if dtype == "bool_":
        return not value
    if dtype in INTEGER_DTYPES:
        return _wrap(dtype, int(value) ^ 0x5A)
    if dtype == "complex64":
        return (_different("float32", value[0]), value[1])
    replacement = _real(
        dtype, 12.5 if _is_nan(value) or float(value) != 12.5 else -12.5
    )
    return replacement


def _guards(dtype):
    """Trailing output values that every case must leave unchanged."""
    if dtype == "bool_":
        return [bool(index % 2) for index in range(GUARD_COUNT)]
    if dtype in INTEGER_DTYPES:
        return [_wrap(dtype, 0x7B + 0x101 * index) for index in range(GUARD_COUNT)]
    if dtype == "complex64":
        return [
            (np.float32(1000.5 + index), np.float32(-index - 0.25))
            for index in range(GUARD_COUNT)
        ]
    return [_real(dtype, 100.0 + 8 * index) for index in range(GUARD_COUNT)]


def _unsigned_zeros(dtype, words):
    """Complex products keep zero signs only in the kernel's own order."""
    if dtype != "complex64":
        return words
    return tuple(0 if part in (0, 0x80000000) else part for part in words)


def _fold(info, values, indices):
    """An independent left fold of one output's elements."""
    op = _operator(info)
    return functools.reduce(
        lambda total, index: op(total, _convert(info, values[index])),
        indices,
        _init(info),
    )


@functools.lru_cache(maxsize=4096)
def _case_data(entry, case):
    """The dispatch, buffers and expected output of one case."""
    info = INFO[entry]
    layout = _layout(entry, case)
    values = _inputs(entry, case)
    run = _simulate(entry, case, values)
    count = len(layout["parts"])
    assert sorted(run.writes) == list(range(count)), entry
    expected = [run.writes[index] for index in range(count)]
    dtype = info["otype"]
    # The model reads exactly the reduced elements, never a gap or decoy,
    # and each output equals a left fold of the elements it covers.
    assert run.reads == {index for indices in layout["groups"] for index in indices}
    relaxed = info["op"] == "prod" and dtype == "complex64"
    for index, indices in layout["parts"].items():
        want = _words(dtype, _fold(info, values, indices))
        got = _words(dtype, expected[index])
        if relaxed:
            want, got = _unsigned_zeros(dtype, want), _unsigned_zeros(dtype, got)
        assert got == want, (entry, case, index, got, want)
    initial = [_different(dtype, value) for value in expected] + _guards(dtype)
    return {
        "grid": layout["grid"],
        "metadata": layout["metadata"],
        "storage": values,
        "initial": initial,
        "expected": expected + _guards(dtype),
        "size": len(values),
    }


def _source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_reduce_entries_match_the_complete_family_classification():
    from demos.integrations.mlx.tests.kernels import (
        test_reduce_complete_metal_roundtrip as complete_metal,
    )

    entries = complete_metal.REDUCE_METAL_CONTRACT["entries"]
    assert [item["entryPoint"] for item in entries] == list(ENTRIES)
    assert len(ENTRIES) == 2396
    names = complete_metal.REDUCE_METAL_CONTRACT["metalTypeNames"]
    dtypes = {
        "bool": "bool_",
        "int8_t": "int8",
        "int16_t": "int16",
        "int32_t": "int32",
        "int64_t": "int64",
        "uint8_t": "uint8",
        "uint16_t": "uint16",
        "uint32_t": "uint32",
        "uint64_t": "uint64",
        "float16_t": "float16",
        "bfloat16_t": "bfloat16",
        "float": "float32",
        "complex64_t": "complex64",
    }
    assert set(dtypes) == set(names)
    ops = {"And": "and", "Or": "or", "Sum": "sum", "Prod": "prod"}
    ops |= {"Min": "min", "Max": "max"}
    for item in entries:
        info = INFO[item["entryPoint"]]
        assert dtypes[item["inputType"]] == info["itype"], item
        assert dtypes[item["outputType"]] == info["otype"], item
        operator, _, otype = item["operator"].partition("<")
        assert ops[operator] == info["op"], item
        assert dtypes[otype.rstrip(">")] == info["otype"], item
        shape = item["shape"]
        if info["kind"] in KINDS:
            index = "i64" if info["large"] else "i32"
            prefix = KINDS[info["kind"]][0].removesuffix("_").replace("_", "-")
            assert shape == f"{prefix}-{index}-d{info['dims']}", item
        else:
            assert (
                shape == {
                    "init": "init",
                    "all": "all",
                    "row_simple": "row-simple",
                }[info["kind"]]
            )
    assert complete_metal.MLX_REDUCE_SHA256 == SOURCE_SHA256


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

DTYPES = (
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "bfloat16",
    "complex64",
)


def _dispatch_count():
    return sum(len(_cases(entry)) for entry in ENTRIES)


@functools.lru_cache(maxsize=None)
def _element_count():
    """Checked output elements per target, trailing guards included."""
    return sum(
        len(_layout(entry, case)["parts"]) + GUARD_COUNT
        for entry in ENTRIES
        for case in _cases(entry)
    )


def _expected_validation_contract():
    return {
        "model": (
            "init_reduce, all_reduce, row_reduce_small, row_reduce_simple, "
            "row_reduce_looped, col_reduce_small, col_reduce_longcolumn, "
            "col_reduce_looped and col_reduce_2pass step by step: thread loops, "
            "guarded loads, SIMD-group shuffle-down trees, threadgroup combines "
            "and guarded stores; checked against a left fold of each output"
        ),
        "comparison": {
            "default": "exact words; a NaN result accepts any NaN",
            "orderFreeReductions": (
                "types narrower than 8 bytes reduce with Apple's SIMD-group "
                "intrinsics; their inputs make every fold order give the same "
                "words"
            ),
            "kernelOrderReductions": (
                "8-byte types reduce with the explicit shuffle loop; complex "
                "products keep the zero signs of the kernel's own order"
            ),
        },
        "workgroupSize": {kind: list(size) for kind, size in WORKGROUPS.items()},
        "softwareSubgroupWidth": {
            "targets": ["opengl", "directx"],
            "width": SIMD_SIZE,
            "kinds": sorted(set(WORKGROUPS) - set(SIMD_FREE_KINDS)),
        },
        "directx": {
            "binary32AdditiveProfile": "native-precise",
            "relativeWaveShuffleOutOfRange": "self",
        },
        "openglIndexRangeAssertions": [
            {
                "expression": item["expression"],
                "minimum": item["minimum"],
                "maximum": item["maximum"],
            }
            for item in _index_assertions("opengl")
        ],
        "cases": {
            "init": {"grid": [INIT_GROUPS, 1, 1]},
            "all": ALL_CASES,
            "row_small": ROW_CASES["row_small"],
            "row_simple": {"rows": ROW_SIMPLE},
            "row_looped": ROW_CASES["row_looped"],
            **{kind: {"columns": spec} for kind, spec in COL_CASES.items()},
        },
        "layout": {
            "rowGap": ROW_GAP,
            "blockGap": BLOCK_GAP,
            "inputDecoys": DECOY_COUNT,
            "trailingOutputGuards": GUARD_COUNT,
            "output": "every element starts different from its expected result",
        },
        "inputs": {
            "patterns": PATTERNS,
            "floatSumQuarters": SUM_QUARTERS,
            "floatProductExponents": PRODUCT_EXPONENTS,
            "extremaPools": {
                dtype: [repr(value) for value in pool]
                for dtype, pool in EXTREMA_POOLS.items()
            },
            "complexExtremaParts": [repr(value) for value in COMPLEX_PARTS],
            "integers": (
                "unsigned sums and products wrap; signed sums and products keep "
                "every partial result of every order inside the output type"
            ),
            "logical": "true is any nonzero bit pattern, NaN included; -0.0 is false",
            "extrema": "one zero sign per output; NaN pairs for complex64",
        },
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype in DTYPES}
            for target in TARGETS
        },
        "dispatchCountByTarget": _dispatch_count(),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
        "ciShardCount": CI_SHARD_COUNT,
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": _dispatch_count(),
        "checkedElementCountByTarget": _element_count(),
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def expected_contract(artifacts=None):
    """Return the contract this module pins, for review and regeneration."""
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-reduce-family-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": HEADERS,
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        "artifacts": {
            target: {entry: artifacts[target][entry] for entry in ENTRIES}
            for target in TARGETS
        },
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(),
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_current_mlx_reduce_contract_is_exact():
    contract = _contract()
    assert contract == json.loads(json.dumps(expected_contract()))
    for target in TARGETS:
        records = contract["artifacts"][target]
        assert list(records) == list(ENTRIES)
        for record in records.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    assert contract["scope"]["dispatchCountByTarget"] == 2830


def _mlx_reduce_dims(reduce_ndim):
    """reduce.cpp ``get_kernel_reduce_ndim``."""
    return 1 if reduce_ndim <= 1 else 2 if reduce_ndim == 2 else 5


def _round32(value):
    return -(-value // 32) * 32


@pytest.mark.parametrize("kind", sorted(WORKGROUPS))
def test_reduce_cases_follow_mlx_dispatch(kind):
    """Each case is one MLX's host (backend/metal/reduce.cpp) would launch."""
    entries = [entry for entry in ENTRIES if INFO[entry]["kind"] == kind]
    for entry in entries:
        info = INFO[entry]
        for case in _cases(entry):
            layout = _layout(entry, case)
            meta = {name: values for name, (_, values) in layout["metadata"].items()}
            grid, size = layout["grid"], WORKGROUPS[kind]
            if kind in KINDS:
                reduce_ndim = meta["reduce_ndim"][0]
                assert _mlx_reduce_dims(reduce_ndim) == info["dims"], (entry, case)
            if kind == "all":
                in_size, row = meta["in_size"][0], meta["row_size"][0]
                if in_size <= 4 * 1024:
                    assert row == in_size and grid == [1, 1, 1]
                    assert size[0] == _round32(-(-in_size // 4))
                else:
                    assert size[0] == _round32(min(-(-row // 4), 1024))
                assert grid[1] == -(-in_size // row)
            elif kind == "row_small":
                row, count = meta["row_size"][0], meta["non_row_reductions"][0]
                assert row <= 64 and size == (32, 1, 1)
                outputs = math.prod(ROW_CASES[kind][case]["out"])
                if (count < 32 and row <= 8) or count <= 8:
                    # dispatch_threads(out grid, (min(width, 1024), 1, 1)).
                    assert ROW_CASES[kind][case]["out"][-1] == size[0]
                    assert grid[0] * size[0] * grid[1] == outputs
                else:
                    assert grid == [1, outputs, 1]
            elif kind == "row_simple":
                row, out = meta["reduction_size"][0], meta["out_size"][0]
                assert 512 < row <= 1024 and size == (128, 1, 1) and out >= 32
                assert grid == [1, -(-out // 4), 1]
            elif kind == "row_looped":
                row = meta["row_size"][0]
                assert 512 < row <= 1024 and size == (128, 1, 1)
                assert grid == [1, math.prod(ROW_CASES[kind][case]["out"]), 1]
            elif kind in COL_CASES:
                stride = meta["reduction_stride"][0]
                total = meta["non_col_reductions"][0] * meta["reduction_size"][0]
                outer = math.prod(COL_CASES[kind]["out"])
                if kind == "col_small":
                    blocks = -(-stride // 4)
                    assert total < 32
                    assert size == (min(blocks, 32), min(8, total), 1)
                    assert grid == [-(-blocks // size[0]), outer, 1]
                elif kind == "col_longcolumn":
                    assert stride < 32 and 1024 <= total < 32768
                    assert size == (stride, min(32, -(-total // 32)), 1)
                    assert grid == [outer, 1, 32]
                    assert meta["out_size"][0] == outer * stride
                else:
                    assert total >= 32 and not (stride < 32 and total >= 1024)
                    two_pass = total > 256 and outer * stride // 32 < 1024
                    assert two_pass == (kind == "col_2pass"), (entry, total)
                    assert size == (256, 1, 1)
                    columns = -(-stride // 32)
                    assert grid == [columns, outer * (32 if two_pass else 1), 1]
                    if two_pass:
                        assert meta["out_size"][0] == outer


@pytest.mark.parametrize("target", TARGETS)
def test_reduce_configs_scope_subgroups_and_index_preconditions(tmp_path, target):
    for entry in ENTRIES[::37]:
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        assert config.entry_workgroup_size_rules[SOURCE][entry] == tuple(
            str(value) for value in _workgroup(entry)
        )
        options = config.source_options["metal"]["target_options"][target]
        simd = INFO[entry]["kind"] not in SIMD_FREE_KINDS
        assert ("software_subgroup_width" in options) == (simd and target != "metal")
        assert config.source_options["metal"]["preserve_resource_origins"] is True
        assertions = [item.to_json() for item in config.index_range_assertions]
        assert assertions == _index_assertions(target)
    assert len(_index_assertions("opengl")) == 21
    assert {item["expression"] for item in _index_assertions("opengl")} >= {
        "in_ + in_idx + int64(lid.x)"
    }


@pytest.mark.parametrize("entry", ENTRIES[::7])
def test_reduce_model_matches_independent_fold(entry):
    # _case_data checks the model against a left fold of every output.
    for case in _cases(entry):
        data = _case_data(entry, case)
        assert len(data["expected"]) == len(data["initial"])
        guards = _guards(INFO[entry]["otype"])
        assert data["expected"][-GUARD_COUNT:] == guards
        for initial, expected in zip(data["initial"], data["expected"][:-GUARD_COUNT]):
            assert _words(INFO[entry]["otype"], initial) != _words(
                INFO[entry]["otype"], expected
            )


def _sum_entries(dtype):
    return [
        entry
        for entry in ENTRIES
        if INFO[entry]["op"] == "sum" and INFO[entry]["itype"] == dtype
    ]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_float_sum_domains_keep_every_partial_sum_exact(dtype):
    quarters = SUM_QUARTERS[dtype]
    seen = set()
    for entry in _sum_entries(dtype)[::5]:
        info = INFO[entry]
        for case in _cases(entry):
            values = _case_data(entry, case)["storage"]
            for indices in _layout(entry, case)["groups"]:
                finite = [float(values[i]) for i in indices if math.isfinite(values[i])]
                assert sum(abs(value) for value in finite) * 4 <= quarters
                assert all((value * 4).is_integer() for value in finite)
                specials = {
                    "nan" if _is_nan(values[i]) else repr(float(values[i]))
                    for i in indices
                    if not math.isfinite(values[i]) or values[i] == 0
                }
                seen |= specials
        assert info["otype"] == dtype
    assert {"nan", "inf", "-inf", "0.0", "-0.0"} <= seen


def test_reduce_inputs_cover_signed_zero_infinity_and_nan_results():
    results = set()
    for entry in ENTRIES[::11]:
        info = INFO[entry]
        if info["otype"] not in FLOAT_DTYPES:
            continue
        for case in _cases(entry):
            for value in _case_data(entry, case)["expected"][:-GUARD_COUNT]:
                if _is_nan(value):
                    results.add("nan")
                elif not math.isfinite(value):
                    results.add("inf")
                elif value == 0:
                    results.add("-0" if math.copysign(1.0, value) < 0 else "+0")
    assert results == {"nan", "inf", "-0", "+0"}


@pytest.mark.parametrize(
    "entry",
    [
        "all_reduce_sumfloat32",
        "row_reduce_small_2_reduce_maxint16",
        "row_reduce_simple_prodbfloat16",
        "row_reduce_looped_large_5_reduce_andfloat16",
        "col_reduce_small_5_reduce_sumcomplex64",
        "col_reduce_longcolumn_large_2_reduce_minfloat16",
        "col_reduce_looped_1_32_32_reduce_orint64",
        "col_reduce_2pass_large_5_32_32_reduce_produint16",
    ],
)
def test_reduce_decoys_change_reductions_that_read_them(entry):
    """A reduction that read one decoy would change most outputs."""
    info = INFO[entry]
    for case in _cases(entry):
        data = _case_data(entry, case)
        values = list(data["storage"])
        layout = _layout(entry, case)
        decoy = _logical(info["itype"], _decoy(info))
        changed = total = 0
        for indices in layout["parts"].values():
            if not indices:
                continue
            total += 1
            before = _words(info["otype"], _fold(info, values, indices))
            first = values[indices[0]]
            values[indices[0]] = decoy
            after = _words(info["otype"], _fold(info, values, indices))
            values[indices[0]] = first
            changed += before != after
        assert total and changed * 3 >= total, (case, changed, total)


def test_reduce_shards_partition_the_family(monkeypatch):
    import importlib

    module = importlib.import_module(__name__)
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(module._selected_entries())
    flat = [entry for shard in shards for entry in shard]
    assert sorted(flat) == sorted(ENTRIES) and len(flat) == len(set(flat))
    assert (
        max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1
    )
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        module._selected_entries()


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------

METADATA_FORMATS = {"int32": "<i", "int64": "<q", "uint64": "<Q"}


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native reduce gate")
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
    assert _source_digest(root / SOURCE) == SOURCE_SHA256
    for header, digest in HEADERS.items():
        assert _source_digest(root / header) == digest, header
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("reduce-metal-reference")
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


def _translate(root, work, target, entry):
    """Translate one entry and check it against its pinned artifact."""
    record = _contract()["artifacts"][target][entry]
    translation = translate_project(
        _config(work, target, entry),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    validation = validate_project_report(work / "report.json")
    assert validation["success"] is True, validation
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == _workgroup(entry)
    generated = root / artifact["path"]
    assert "unsupported" not in generated.read_text(encoding="utf-8")
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    descriptor, package = _prepare_native_package(translation, work)
    return descriptor, package, record


def _metadata_value(dtype, values):
    return {"dtype": dtype, "shape": [len(values)], "values": list(values)}


def _request(descriptor, package, target, entry, data):
    """Bind the case by source parameter and build one dispatch."""
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    info = INFO[entry]
    values = {"out": _runtime_value(target, info["otype"], data["initial"])}
    if info["kind"] != "init":
        values["in"] = _runtime_value(target, info["itype"], data["storage"])
    for name, (dtype, items) in data["metadata"].items():
        values[name] = _metadata_value(dtype, items)
    inputs, observed, output, generated = {}, [], None, 0
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            # HLSL has no threadgroups_per_grid builtin: the column kernels
            # that read it get a constant buffer the loader fills from the
            # dispatch.
            assert target == "directx", binding
            assert provenance["executionInput"]["valueSource"] == (
                "dispatch.workgroupCount"
            ), binding
            generated += 1
            continue
        source = provenance.get("sourceResource")
        assert source is not None, binding
        name = source["parameter"]
        value = values[name]
        layout = binding["scalarLayout"]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
        inputs[binding["name"]] = value
        if name == "out":
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert output is not None and len(inputs) == len(values), (sorted(values), inputs)
    expected = _runtime_value(target, info["otype"], data["expected"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: expected},
        {"workgroupCount": list(data["grid"]), "workgroupSize": _workgroup(entry)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    derived = [
        item.value.values
        for item in request.execution_plan.resource_bindings
        if item.value is not None and "executionInput" in item.value.metadata
    ]
    assert derived == [list(data["grid"])] * generated
    return request, {"out": output}, len(observed)


def _execute(executor, work, label, request, output_names, observed):
    """Run one dispatch; every observed input must come back unchanged."""
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
    # init_reduce binds only its output, so nothing is observed.
    observations = result.details.get("inputObservations", {})
    assert len(observations) == observed
    assert all(item["unchanged"] for item in observations.values())
    return {name: result.outputs[binding] for name, binding in output_names.items()}


def _nan_word(target, dtype, word):
    if dtype in ("float32", "complex64") or (
        target == "opengl" and dtype in FLOAT_DTYPES
    ):
        return word & 0x7FFFFFFF > 0x7F800000
    if dtype == "float16":
        return word & 0x7FFF > 0x7C00
    if dtype == "bfloat16":
        return word & 0x7FFF > 0x7F80
    return False


def _check(label, target, entry, actual, data):
    """Output words equal the model's; a NaN result accepts any NaN."""
    dtype = INFO[entry]["otype"]
    expected = _runtime_value(target, dtype, data["expected"])["values"]
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
        and not (_nan_word(target, dtype, want) and _nan_word(target, dtype, got))
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


def _upstream_words(dtype, raw):
    if dtype == "bool_":
        words = list(raw)
        assert set(words) <= {0, 1}
        return [bool(word) for word in words]
    if dtype in INTEGER_DTYPES:
        code = METAL_FORMATS[dtype]
        return list(struct.unpack(f"<{len(raw) // struct.calcsize(code)}{code}", raw))
    if dtype in ("float16", "bfloat16"):
        return list(struct.unpack(f"<{len(raw) // 2}H", raw))
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    info = INFO[entry]
    payloads = (
        [] if info["kind"] == "init" else [_metal_bytes(info["itype"], data["storage"])]
    )
    output = len(payloads)
    payloads.append(_metal_bytes(info["otype"], data["initial"]))
    for dtype, items in data["metadata"].values():
        payloads.append(
            struct.pack("<" + METADATA_FORMATS[dtype][1] * len(items), *items)
        )
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
                "workgroupCount": list(data["grid"]),
                "workgroupSize": _workgroup(entry),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    for index, payload in enumerate(payloads):
        if index != output:
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    raw = (readback / f"buffer-{output}.bin").read_bytes()
    return device, _upstream_words(info["otype"], raw)


def _assert_identical(label, actual, expected):
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_reduce_family_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    with tempfile.TemporaryDirectory(prefix=".current-reduce-", dir=root) as directory:
        work = Path(directory)
        try:
            descriptor, package, record = _translate(root, work, target, entry)
            executor = _executor(target)
            evidence = []
            try:
                for case in _cases(entry):
                    data = _case_data(entry, case)
                    request, output_names, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    outputs = _execute(
                        executor, work, case, request, output_names, observed
                    )
                    words = outputs["out"]["values"]
                    upstream = None
                    if target == "metal":
                        device, reference = _upstream(
                            work, metal_reference, entry, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        _assert_identical(case, words, reference)
                        upstream = {"device": device, "identicalToTranslation": True}
                    _check(case, target, entry, words, data)
                    evidence.append(
                        {
                            "case": case,
                            "grid": data["grid"],
                            "inputElementCount": data["size"],
                            "outputElementCount": len(data["expected"]),
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
                        "comparison": _expected_validation_contract()["comparison"],
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.air", "*.metallib", "out"),
            )


def test_ci_requires_native_reduce_family_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX reduce family native parity",
        family="reduce",
        count=CI_SHARD_COUNT,
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        f"{SHARD_INDEX_ENV}: ${{{{ matrix.shard }}}}",
        f'{SHARD_COUNT_ENV}: "{CI_SHARD_COUNT}"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/${{ matrix.group }}",
        "--junitxml=mlx-current-results/${{ matrix.group }}.xml",
        "demos/integrations/mlx/tests/kernels/test_current_reduce_family.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_reduce_family.py"
    )
