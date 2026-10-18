"""Native parity of the 224 pinned MLX GEMV kernels outside the wide family.

``gemv.metal`` instantiates ``gemv`` and ``gemv_t`` for float32, float16,
bfloat16 and complex64 over every tile shape ``matmul.cpp`` selects, each
with and without non-contiguous batches and ``axpby``, and ``gemv_gather``
and ``gemv_t_gather`` over their tile shapes. Its 32 ``gemv_wide`` and
``gemv_wide_gather`` entries have their own gate, ``test_current_gemv_wide.py``.

Every entry runs three cases on ``matmul.cpp``'s threadgroups of
``(32, bn, bm)`` threads and grids of ``(ceil(outputs / outputs per group), 1,
batches)``: a short one with fewer inputs than one block iteration, a ragged
one with partial row or column blocks and leftover inputs, and an aligned one.
A model of both kernels replays each thread's block position, iteration and
leftover loads, SIMD-group shuffle-down sums, threadgroup reduction and
guarded stores and must reach a direct dot product. Inputs are small integers:
float32 accumulates their products exactly in any order, and every result,
``alpha`` and ``beta * bias`` term is exact in the element type, so each
target must produce the model's words.
"""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
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
    _below,
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/gemv.metal"
SOURCE_SHA256 = "0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda"
HEADERS = {
    "mlx/backend/metal/kernels/gemv.h": (
        "296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
    "mlx/backend/metal/kernels/steel/utils.h": (
        "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a"
    ),
    "mlx/backend/metal/kernels/complex.h": (
        "16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1"
    ),
    "mlx/backend/metal/kernels/bf16.h": (
        "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb"
    ),
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMV"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/gemv.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# gemv.metal's element types and its tile shapes (bm, bn, sm, sn, tm, tn), in
# instantiation order.
DTYPES = (
    ("float32", "float"),
    ("float16", "half"),
    ("bfloat16", "bfloat16_t"),
    ("complex64", "complex64_t"),
)
GEMV_BLOCKS = (
    (1, 8, 1, 32, 4, 4),
    (1, 8, 1, 32, 1, 4),
    (1, 1, 8, 4, 4, 4),
    (1, 1, 8, 4, 1, 4),
    (4, 1, 1, 32, 1, 4),
    (4, 1, 1, 32, 4, 4),
    (8, 1, 1, 32, 4, 4),
)
GEMV_GATHER_BLOCKS = ((4, 1, 1, 32, 1, 4), (4, 1, 1, 32, 4, 4), (8, 1, 1, 32, 4, 4))
GEMV_T_BLOCKS = (
    (1, 2, 8, 4, 4, 1),
    (1, 2, 8, 4, 4, 4),
    (1, 4, 8, 4, 4, 4),
    (1, 16, 8, 4, 4, 4),
    (1, 16, 4, 8, 4, 4),
)
GEMV_T_GATHER_BLOCKS = GEMV_T_BLOCKS
# (kDoNCBatch, kDoAxpby) of instantiate_gemv and instantiate_gemv_t.
NC_AXPBY = ((0, 0), (0, 1), (1, 0), (1, 1))
WIDE_ENTRY_COUNT = 32

# [[buffer(i)]] parameters of the kernels, in index order.
GEMV_PARAMETERS = (
    "mat",
    "in_vec",
    "bias",
    "out_vec",
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "alpha",
    "beta",
    "batch_ndim",
    "batch_shape",
    "vector_batch_stride",
    "matrix_batch_stride",
    "bias_batch_stride",
    "bias_stride",
)
GATHER_PARAMETERS = (
    "mat",
    "in_vec",
    "bias",
    "out_vec",
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "alpha",
    "beta",
    "batch_ndim",
    "batch_shape",
    "index_batch_strides",
    "vector_batch_ndim",
    "vector_batch_shape",
    "vector_batch_stride",
    "matrix_batch_ndim",
    "matrix_batch_shape",
    "matrix_batch_stride",
    "vec_indices",
    "mat_indices",
)
SCALAR_TYPES = {
    "in_vec_size": "int32",
    "out_vec_size": "int32",
    "marix_ld": "int32",
    "alpha": "float32",
    "beta": "float32",
    "batch_ndim": "int32",
    "bias_stride": "int32",
    "vector_batch_ndim": "int32",
    "matrix_batch_ndim": "int32",
}
ARRAY_TYPES = {
    "batch_shape": "int32",
    "vector_batch_shape": "int32",
    "matrix_batch_shape": "int32",
    "vector_batch_stride": "int64",
    "matrix_batch_stride": "int64",
    "bias_batch_stride": "int64",
    "index_batch_strides": "int64",
    "vec_indices": "uint32",
    "mat_indices": "uint32",
}

SIMD_SIZE = 32
SIGNED_MAX = 2**31 - 1
CASES = ("short", "ragged", "aligned")
GUARD_COUNT = 16
ROW_PAD = 3
VECTOR_GAP = 5
MATRIX_GAP = 7
BIAS_GAP = 3
BIAS_STRIDES = {"short": 1, "ragged": 2, "aligned": 0}
ALPHA, BETA = 2.0, -0.5
# Bounds on |dot product|: 2 * bound + 7 stays exact in the element type.
RESULT_BOUNDS = {"float32": 2**20, "float16": 512, "bfloat16": 96}
PRECISION = {"float32": 24, "float16": 11, "bfloat16": 8}
# Exactly representable poison for every input element no case may read.
POISON = (640, -704, 768)
COMPLEX_POISON = ((640, -704), (-768, 640))
# Every output starts here; no result can equal it.
SENTINEL = -3.25
COMPLEX_SENTINEL = (-3.25, 0.75)

# Expressions OpenGL may evaluate in 32 bits: pointer offsets of the batch
# and gather prologues and of the row and column loads. The model shows each
# names an element inside its buffer.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "gemv": (
        "uint64(out_row) * matrix_ld",
        "mat + uint64(out_row) * matrix_ld",
        "src + src_offset + tn",
    ),
    "gemv_t": (
        "uint64(bm + tm) * marix_ld + out_col + tn",
        "mat + uint64(bm + tm) * marix_ld + out_col + tn",
    ),
    "batch": (
        "tid.z * buffer_load(vector_batch_stride, 0)",
        "in_vec + tid.z * buffer_load(vector_batch_stride, 0)",
        "tid.z * buffer_load(matrix_batch_stride, 0)",
        "mat + tid.z * buffer_load(matrix_batch_stride, 0)",
        "tid.z * buffer_load(bias_batch_stride, 0)",
        "bias + tid.z * buffer_load(bias_batch_stride, 0)",
    ),
    "gather": (
        "buffer_load(index_batch_strides, 0) * tid.z",
        "buffer_load(index_batch_strides, batch_ndim) * tid.z",
        "batch_offsets.x",
        "batch_offsets.y",
        "indx_vec * buffer_load(vector_batch_stride, 0)",
        "in_vec + indx_vec * buffer_load(vector_batch_stride, 0)",
        "indx_mat * buffer_load(matrix_batch_stride, 0)",
        "mat + indx_mat * buffer_load(matrix_batch_stride, 0)",
    ),
}


def _tile_name(block):
    bm, bn, sm, sn, tm, tn = block
    return f"bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}"


def _instantiations():
    """``(entry, info)`` in gemv.metal's instantiation order."""
    tables = (
        ("gemv", False, GEMV_BLOCKS, True),
        ("gemv", True, GEMV_GATHER_BLOCKS, False),
        ("gemv_t", False, GEMV_T_BLOCKS, True),
        ("gemv_t", True, GEMV_T_GATHER_BLOCKS, False),
    )
    for kind, gather, blocks, variants in tables:
        for dtype, _ in DTYPES:
            for block in blocks:
                bm, bn, sm, sn, tm, tn = block
                base = (
                    f"{kind}{'_gather' if gather else ''}_{dtype}_{_tile_name(block)}"
                )
                for nc, axpby in NC_AXPBY if variants else ((0, 0),):
                    name = f"{base}_nc{nc}_axpby{axpby}" if variants else base
                    yield name, {
                        "kind": kind,
                        "gather": gather,
                        "dtype": dtype,
                        "bm": bm,
                        "bn": bn,
                        "sm": sm,
                        "sn": sn,
                        "tm": tm,
                        "tn": tn,
                        "nc": bool(nc),
                        "axpby": bool(axpby),
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _workgroup(entry):
    info = INFO[entry]
    return [SIMD_SIZE, info["bn"], info["bm"]]


def _index_assertions(entry):
    info = INFO[entry]
    group = "gather" if info["gather"] else "batch"
    return OPENGL_INDEX_RANGE_ASSERTIONS[info["kind"]] + (
        OPENGL_INDEX_RANGE_ASSERTIONS[group]
    )


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
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (_index_assertions(entry) if target == "opengl" else ())
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
                "target_options": {
                    "directx": {
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {"software_subgroup_width": SIMD_SIZE},
                },
            }
        },
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _blocks(info):
    """``(outputs per threadgroup, inputs per iteration, TOut, TIn)``."""
    threads_m = info["bm"] * info["sm"]
    threads_n = info["bn"] * info["sn"]
    if info["kind"] == "gemv":
        return threads_m * info["tm"], threads_n * info["tn"], info["tm"], info["tn"]
    return threads_n * info["tn"], threads_m * info["tm"], info["tn"], info["tm"]


def _shape(entry, case):
    """``(out_vec_size, in_vec_size, marix_ld)``; ``gemv_t`` rows are inputs."""
    info = INFO[entry]
    per_group, per_iteration, t_out, t_in = _blocks(info)
    out_size = {
        "short": t_out + 2,
        "ragged": 2 * per_group + t_out + 1,
        "aligned": 2 * per_group,
    }[case]
    if info["kind"] == "gemv":
        in_size = {
            "short": per_iteration - t_in - 1,
            "ragged": 2 * per_iteration + t_in + 1,
            "aligned": per_iteration,
        }[case]
        row = in_size
    else:
        in_size = {
            "short": per_iteration - 1,
            "ragged": 2 * per_iteration + 3,
            "aligned": per_iteration,
        }[case]
        row = out_size
    return out_size, in_size, row + (0 if case == "aligned" else ROW_PAD)


def _rows(info, out_size, in_size):
    return out_size if info["kind"] == "gemv" else in_size


def _elem_to_loc(elem, shape, strides):
    """utils.h ``elem_to_loc`` and steel/utils.h ``elem_to_loc_broadcast``."""
    loc = [0] * len(strides)
    for i in range(len(shape) - 1, -1, -1):
        if elem <= 0:
            break
        position = elem % shape[i]
        elem //= shape[i]
        for k, stride in enumerate(strides):
            loc[k] += position * stride[i]
    return loc


# Gather layouts: batch shape and index strides, then the gathered vector and
# matrix batch layouts. Matrix indices are distinct over the grid's batches.
GATHER_LAYOUTS = {
    "short": {
        "batch_shape": [3],
        "index_batch_strides": [1, 2],
        "vec_indices": [2, 0, 2],
        "mat_indices": [3, 1, 1, 2, 0],
        "vector_batch_shape": [3],
        "matrix_batch_shape": [4],
    },
    "ragged": {
        "batch_shape": [3, 2],
        "index_batch_strides": [2, 1, 1, 3],
        "vec_indices": [1, 3, 0, 0, 2, 1],
        "mat_indices": [4, 1, 5, 0, 3, 2],
        "vector_batch_shape": [2, 2],
        "matrix_batch_shape": [3, 2],
    },
    "aligned": {
        "batch_shape": [2],
        "index_batch_strides": [0, 1],
        "vec_indices": [1],
        "mat_indices": [1, 0],
        "vector_batch_shape": [2],
        "matrix_batch_shape": [1, 2],
    },
}


def _batching(entry, case):
    """Batch metadata and per-batch ``(vector, matrix, bias)`` element offsets."""
    info = INFO[entry]
    out_size, in_size, ld = _shape(entry, case)
    vs = in_size + VECTOR_GAP
    ms = _rows(info, out_size, in_size) * ld + MATRIX_GAP
    bias_stride = BIAS_STRIDES[case]
    bs = out_size * max(bias_stride, 1) + BIAS_GAP
    if info["gather"]:
        layout = GATHER_LAYOUTS[case]
        shape = layout["batch_shape"]
        ndim = len(shape)
        strides = layout["index_batch_strides"]
        vector_shape = layout["vector_batch_shape"]
        matrix_shape = layout["matrix_batch_shape"]
        vector_strides = [vs] if len(vector_shape) == 1 else [vs, 2 * vs]
        matrix_strides = {1: [ms], 2: [ms, 3 * ms]}[len(matrix_shape)]
        if matrix_shape == [1, 2]:
            # A unit outer extent: elem_to_loc stops once the index is spent.
            matrix_strides = [5 * ms, ms]
        batches = []
        count = 1
        for extent in shape:
            count *= extent
        for z in range(count):
            if ndim > 1:
                loc_v, loc_m = _elem_to_loc(z, shape, (strides[:ndim], strides[ndim:]))
            else:
                loc_v, loc_m = strides[0] * z, strides[ndim] * z
            vector = layout["vec_indices"][loc_v]
            matrix = layout["mat_indices"][loc_m]
            if len(vector_shape) > 1:
                (v_offset,) = _elem_to_loc(vector, vector_shape, (vector_strides,))
            else:
                v_offset = vector * vector_strides[0]
            if len(matrix_shape) > 1:
                (m_offset,) = _elem_to_loc(matrix, matrix_shape, (matrix_strides,))
            else:
                m_offset = matrix * matrix_strides[0]
            batches.append(
                {
                    "vector": vector,
                    "matrix": matrix,
                    "in_vec": v_offset,
                    "mat": m_offset,
                }
            )
        metadata = {
            "batch_ndim": ndim,
            "batch_shape": shape,
            "index_batch_strides": strides,
            "vector_batch_ndim": len(vector_shape),
            "vector_batch_shape": vector_shape,
            "vector_batch_stride": vector_strides,
            "matrix_batch_ndim": len(matrix_shape),
            "matrix_batch_shape": matrix_shape,
            "matrix_batch_stride": matrix_strides,
            "vec_indices": layout["vec_indices"],
            "mat_indices": layout["mat_indices"],
            "bias_stride": ndim,  # gemv_gather passes batch_ndim; it is unused
        }
        vectors = 1
        for extent in vector_shape:
            vectors *= extent
        matrices = 1
        for extent in matrix_shape:
            matrices *= extent
        offsets = {
            "in_vec": [
                (
                    _elem_to_loc(v, vector_shape, (vector_strides,))[0]
                    if len(vector_shape) > 1
                    else v * vs
                )
                for v in range(vectors)
            ],
            "mat": [
                (
                    _elem_to_loc(m, matrix_shape, (matrix_strides,))[0]
                    if len(matrix_shape) > 1
                    else m * ms
                )
                for m in range(matrices)
            ],
        }
        return metadata, batches, offsets
    if info["nc"]:
        shape = [2, 2]
        vector_strides, matrix_strides, bias_strides = (
            [0, vs],
            [ms, 2 * ms],
            [2 * bs, bs],
        )
    else:
        shape = [2]
        vector_strides, matrix_strides, bias_strides = [vs], [ms], [bs]
    count = shape[0] * (shape[1] if len(shape) > 1 else 1)
    batches = []
    for z in range(count):
        if info["nc"]:
            v_offset, m_offset, b_offset = _elem_to_loc(
                z, shape, (vector_strides, matrix_strides, bias_strides)
            )
        else:
            v_offset, m_offset, b_offset = (
                z * vector_strides[0],
                z * matrix_strides[0],
                z * bias_strides[0],
            )
        batches.append(
            {
                "vector": v_offset // vs,
                "matrix": z,
                "in_vec": v_offset,
                "mat": m_offset,
                "bias": b_offset,
            }
        )
    metadata = {
        "batch_ndim": len(shape),
        "batch_shape": shape,
        "vector_batch_stride": vector_strides,
        "matrix_batch_stride": matrix_strides,
        "bias_batch_stride": bias_strides,
        "bias_stride": bias_stride,
    }
    offsets = {
        "in_vec": sorted({batch["in_vec"] for batch in batches}),
        "mat": [batch["mat"] for batch in batches],
    }
    return metadata, batches, offsets


def _grid(entry, case):
    per_group = _blocks(INFO[entry])[0]
    out_size = _shape(entry, case)[0]
    batches = len(_batching(entry, case)[1])
    return [-(-out_size // per_group), 1, batches]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _mul(a, b):
    if isinstance(a, tuple):
        return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
    return a * b


def _add(a, b):
    if isinstance(a, tuple):
        return (a[0] + b[0], a[1] + b[1])
    return a + b


def _zero(dtype):
    return (0, 0) if dtype == "complex64" else 0


def _exact(dtype, value):
    """Whether ``value`` (a number or a complex pair) is a value of ``dtype``."""
    parts = value if isinstance(value, tuple) else (value,)
    bits = PRECISION["float32" if dtype == "complex64" else dtype]
    for part in parts:
        number = Fraction(part)
        if number == 0:
            continue
        if number.denominator & (number.denominator - 1):
            return False
        significand = abs(number.numerator)
        while significand % 2 == 0:
            significand //= 2
        if significand >= 2**bits:
            return False
    return True


def _vector_values(dtype, draws, length):
    if dtype == "complex64":
        values = []
        for _ in range(length):
            value = (_signed(draws, 2), _signed(draws, 2))
            values.append(value if value != (0, 0) else (1, -1))
        return values
    return [
        (1 + _below(draws, 2)) * (1 if _below(draws, 2) else -1) for _ in range(length)
    ]


def _row_values(dtype, draws, vector):
    """A matrix row (or column) whose dot product with ``vector`` is nonzero
    and, for real types, at most ``RESULT_BOUNDS`` in magnitude: steps of a
    walk flip sign whenever the running sum would leave the bound."""
    if dtype == "complex64":
        values = [(_signed(draws, 2), _signed(draws, 2)) for _ in vector]
        total = (0, 0)
        for value, coefficient in zip(values, vector):
            total = _add(total, _mul(value, coefficient))
        if total == (0, 0):
            values[-1] = _add(values[-1], (1, 0))
        return values
    bound = RESULT_BOUNDS[dtype]
    values, total = [], 0
    for coefficient in vector:
        value = _signed(draws, 2)
        if abs(total + value * coefficient) > bound:
            value = -value
        values.append(value)
        total += value * coefficient
    if total == 0:
        last = max(k for k in range(len(values)) if values[k] or k == 0)
        if values[last] == 0:
            values[last] = 1
        else:
            values[last] = -values[last]
    return values


def _bias_values(dtype, draws, length):
    """Nonzero values ``2 mod 4``: ``beta * bias`` is odd, so no result is 0."""
    pick = (2, 6, 10, 14)

    def one():
        return pick[_below(draws, 4)] * (1 if _below(draws, 2) else -1)

    if dtype == "complex64":
        return [(one(), one()) for _ in range(length)]
    return [one() for _ in range(length)]


def _poison(dtype, index):
    if dtype == "complex64":
        return COMPLEX_POISON[index % len(COMPLEX_POISON)]
    return POISON[index % len(POISON)]


def _case_data(entry, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    info = INFO[entry]
    dtype = info["dtype"]
    out_size, in_size, ld = _shape(entry, case)
    rows = _rows(info, out_size, in_size)
    metadata, batches, offsets = _batching(entry, case)
    draws = _draws("gemv", entry, case)
    vector_length = max(offsets["in_vec"]) + in_size + VECTOR_GAP
    matrix_length = max(offsets["mat"]) + rows * ld + MATRIX_GAP
    in_vec = [_poison(dtype, k) for k in range(vector_length)]
    mat = [_poison(dtype, k) for k in range(matrix_length)]
    vectors = {}
    for offset in offsets["in_vec"]:
        values = _vector_values(dtype, draws, in_size)
        in_vec[offset : offset + in_size] = values
        vectors[offset] = values
    built = set()
    for batch in batches:
        # Each batch's matrix holds rows (gemv) or columns (gemv_t) whose
        # dot products with that batch's vector stay exact.
        assert batch["mat"] not in built, (entry, case)
        built.add(batch["mat"])
        vector = vectors[batch["in_vec"]]
        for out in range(out_size):
            line = _row_values(dtype, draws, vector)
            for k, value in enumerate(line):
                if info["kind"] == "gemv":
                    mat[batch["mat"] + out * ld + k] = value
                else:
                    mat[batch["mat"] + k * ld + out] = value
    for offset in offsets["mat"]:
        if offset not in built:
            for row in range(rows):
                for k in range(ld):
                    mat[offset + row * ld + k] = (
                        _signed(draws, 2)
                        if dtype != ("complex64")
                        else (_signed(draws, 2), _signed(draws, 2))
                    )
    bias_stride = metadata["bias_stride"]
    if info["axpby"]:
        bias_length = max(batch["bias"] for batch in batches) + (
            out_size * max(bias_stride, 1) + BIAS_GAP
        )
        bias = _bias_values(dtype, draws, bias_length)
    else:
        bias = [_poison(dtype, k) for k in range(GUARD_COUNT)]
    total = len(batches) * out_size + GUARD_COUNT
    sentinel = COMPLEX_SENTINEL if dtype == "complex64" else SENTINEL
    initial = [sentinel] * total
    expected = list(initial)
    for z, batch in enumerate(batches):
        for out in range(out_size):
            value = _zero(dtype)
            for k in range(in_size):
                if info["kind"] == "gemv":
                    element = mat[batch["mat"] + out * ld + k]
                else:
                    element = mat[batch["mat"] + k * ld + out]
                value = _add(value, _mul(element, in_vec[batch["in_vec"] + k]))
            if info["axpby"]:
                term = bias[batch["bias"] + out * bias_stride]
                if dtype == "complex64":
                    value = _add(_mul((ALPHA, 0), value), _mul((BETA, 0), term))
                else:
                    value = ALPHA * value + BETA * term
            assert _exact(dtype, value), (entry, case, z, out, value)
            expected[z * out_size + out] = value
    alpha, beta = (ALPHA, BETA) if info["axpby"] else (1.0, 0.0)
    parameters = GATHER_PARAMETERS if info["gather"] else GEMV_PARAMETERS
    scalars = {
        "in_vec_size": in_size,
        "out_vec_size": out_size,
        "marix_ld": ld,
        "alpha": alpha,
        "beta": beta,
        **{
            name: value
            for name, value in metadata.items()
            if name in SCALAR_TYPES and name in parameters
        },
    }
    arrays = {
        name: list(value)
        for name, value in metadata.items()
        if name in ARRAY_TYPES and name in parameters
    }
    if not info["gather"] and not info["axpby"]:
        # matmul.cpp binds no bias buffers without axpby; any value works.
        arrays["bias_batch_stride"] = [0] * len(metadata["bias_batch_stride"])
    values = {
        "mat": (dtype, mat),
        "in_vec": (dtype, in_vec),
        "bias": (dtype, bias),
        "out_vec": (dtype, initial),
        **{name: (SCALAR_TYPES[name], [value]) for name, value in scalars.items()},
        **{name: (ARRAY_TYPES[name], value) for name, value in arrays.items()},
    }
    assert set(values) == set(parameters), set(values) ^ set(parameters)
    return {
        "values": values,
        "initial": initial,
        "expected": expected,
        "grid": _grid(entry, case),
        "shape": {"outVecSize": out_size, "inVecSize": in_size, "matrixLd": ld},
        "batches": batches,
        "written": len(batches) * out_size,
        "size": total,
    }


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


class _Context:
    """One batch of one case: buffers, offsets and the model's records."""

    def __init__(self, entry, data, batch, mutation):
        info = INFO[entry]
        self.info = info
        self.dtype = info["dtype"]
        self.storage = {name: values for name, (_, values) in data["values"].items()}
        self.out_size = data["shape"]["outVecSize"]
        self.in_size = data["shape"]["inVecSize"]
        self.ld = data["shape"]["matrixLd"]
        self.batch = batch
        self.mutation = mutation
        self.bias_stride = (
            data["values"]["bias_stride"][1][0]
            if ("bias_stride" in data["values"])
            else None
        )
        self.accesses = {name: [] for name in ("mat", "in_vec", "bias")}
        self.paths = set()

    def load(self, name, index):
        self.accesses[name].append(index)
        values = self.storage[name]
        if self.mutation is not None and not 0 <= index < len(values):
            return _poison(self.dtype, 0)  # a defective kernel reads past its buffer
        return values[index]

    def load_safe(self, name, base, offset, size, count):
        """GEMVKernel::load_safe: whole when it fits, else element-guarded."""
        if offset + count <= size:
            return [self.load(name, base + offset + k) for k in range(count)]
        self.paths.add("partial-load")
        return [
            (
                self.load(name, base + offset + k)
                if offset + k < size
                else _zero(self.dtype)
            )
            for k in range(count)
        ]

    def finish(self, value, index):
        """The stored value: ``T(result)`` or ``alpha * T(result) + beta * bias``."""
        if not self.info["axpby"]:
            return value
        term = self.load("bias", self.batch["bias"] + index * self.bias_stride)
        if self.mutation == "bias":
            term = _zero(self.dtype)
        if self.dtype == "complex64":
            return _add(_mul((ALPHA, 0), value), _mul((BETA, 0), term))
        return ALPHA * value + BETA * term


def _sum(dtype, values):
    total = _zero(dtype)
    for value in values:
        total = _add(total, value)
    return total


def _gemv_group(ctx, gx):
    """GEMVKernel::run for threadgroup ``gx``: ``(row, value)`` stores."""
    info = ctx.info
    bm_, bn_, sm, sn, tm_, tn_ = (info[k] for k in ("bm", "bn", "sm", "sn", "tm", "tn"))
    block_m, block_n = bm_ * sm * tm_, bn_ * sn * tn_
    rows, size, ld = ctx.out_size, ctx.in_size, ctx.ld
    n_iter = size // block_n
    leftover = size - block_n * n_iter
    lanes = {}
    for simd_gid in range(bm_ * bn_):
        for lane in range(SIMD_SIZE):
            thr_m = lane // sn if sn != 32 else 0
            thr_n = lane % sn if sn != 32 else lane
            sg_n = simd_gid % bn_ if bn_ != 1 else 0
            simd_m = sm * (simd_gid // bn_) if bn_ != 1 else sm * simd_gid
            simd_n = sn * (simd_gid % bn_) if bn_ != 1 else 0
            bm = (simd_m + thr_m) * tm_
            bn = (simd_n + thr_n) * tn_
            out_row = gx * block_m + bm
            if out_row >= rows:
                ctx.paths.add("early-exit")
                continue
            if out_row + tm_ > rows and ctx.mutation != "shift":
                ctx.paths.add("tail-shift")
                out_row = rows - tm_
            row_base = ctx.batch["mat"] + out_row * ld
            result = [_zero(ctx.dtype)] * tm_
            for _ in range(n_iter):
                ctx.paths.add("iterations")
                coeff = [
                    ctx.load("in_vec", ctx.batch["in_vec"] + bn + k) for k in range(tn_)
                ]
                for t in range(tm_):
                    for k in range(tn_):
                        element = ctx.load("mat", row_base + t * ld + bn + k)
                        result[t] = _add(result[t], _mul(element, coeff[k]))
                bn += block_n
            if leftover > 0 and ctx.mutation != "leftover":
                ctx.paths.add("leftover")
                coeff = ctx.load_safe("in_vec", ctx.batch["in_vec"], bn, size, tn_)
                for t in range(tm_):
                    inter = ctx.load_safe("mat", row_base + t * ld, bn, size, tn_)
                    for k in range(tn_):
                        result[t] = _add(result[t], _mul(inter[k], coeff[k]))
            lanes[simd_gid, lane] = {
                "thr_n": thr_n,
                "sg_n": sg_n,
                "simd_n": simd_n,
                "bm": bm,
                "out_row": out_row,
                "result": result,
            }
    # simd_shuffle_down(sn = SN / 2 .. 1) leaves each row group's sum in its
    # thrN == 0 lane; the group's lanes share the row block and exit together.
    leaders = []
    for (simd_gid, lane), state in sorted(lanes.items()):
        if state["thr_n"] != 0:
            continue
        group = [lanes.get((simd_gid, lane + k)) for k in range(sn)]
        assert all(item and item["out_row"] == state["out_row"] for item in group)
        state["sum"] = [
            _sum(ctx.dtype, [item["result"][t] for item in group]) for t in range(tm_)
        ]
        leaders.append(state)
    if bn_ > 1:
        # Every thrN == 0 lane stores, a barrier, then sgN == 0 adds the rest.
        assert {state["bm"] for state in leaders} == {0}
        if ctx.mutation != "reduction":
            ctx.paths.add("threadgroup-reduction")
            memory = {}
            for state in leaders:
                for t in range(tm_):
                    memory[state["sg_n"] * (block_m + tm_) + state["bm"] + t] = state[
                        "sum"
                    ][t]
            for state in leaders:
                if state["sg_n"] == 0:
                    for sgn in range(1, bn_):
                        for t in range(tm_):
                            state["sum"][t] = _add(
                                state["sum"][t],
                                memory[sgn * (block_m + tm_) + state["bm"] + t],
                            )
    stores = []
    for state in leaders:
        if state["simd_n"] == 0:
            for t in range(tm_):
                row = state["out_row"] + t
                stores.append((row, ctx.finish(state["sum"][t], row)))
    return stores


def _gemv_t_group(ctx, gx):
    """GEMVTKernel::run for threadgroup ``gx``: ``(column, value)`` stores."""
    info = ctx.info
    bm_, bn_, sm, sn, tm_, tn_ = (info[k] for k in ("bm", "bn", "sm", "sn", "tm", "tn"))
    assert bm_ == 1  # gemv.metal never instantiates a threadgroup reduction
    block_m, block_n = bm_ * sm * tm_, bn_ * sn * tn_
    columns, size, ld = ctx.out_size, ctx.in_size, ctx.ld
    n_iter = size // block_m
    leftover = size - block_m * n_iter
    lanes = {}
    for simd_gid in range(bm_ * bn_):
        for lane in range(SIMD_SIZE):
            thr_m = lane // sn if sn != 32 else 0
            thr_n = lane % sn if sn != 32 else lane
            sg_m = simd_gid // bn_ if bn_ != 1 else simd_gid
            sg_n = simd_gid % bn_ if bn_ != 1 else 0
            cm = sm * sg_m + thr_m
            cn = sn * sg_n + thr_n
            bm, bn = cm * tm_, cn * tn_
            out_col = gx * block_n + bn
            result = [_zero(ctx.dtype)] * tn_
            if out_col < columns:
                if out_col + tn_ >= columns and ctx.mutation != "shift":
                    if out_col != columns - tn_:
                        ctx.paths.add("tail-shift")
                    out_col = columns - tn_
                for _ in range(n_iter):
                    # The row loop's threadgroup_barrier runs here.
                    ctx.paths.add("iterations")
                    coeff = [
                        ctx.load("in_vec", ctx.batch["in_vec"] + bm + t)
                        for t in range(tm_)
                    ]
                    for t in range(tm_):
                        for k in range(tn_):
                            element = ctx.load(
                                "mat", ctx.batch["mat"] + (bm + t) * ld + out_col + k
                            )
                            result[k] = _add(result[k], _mul(coeff[t], element))
                    bm += block_m
                if leftover > 0 and ctx.mutation != "leftover":
                    ctx.paths.add("leftover")
                    t = 0
                    while t < tm_ and bm + t < size:
                        coeff = ctx.load("in_vec", ctx.batch["in_vec"] + bm + t)
                        for k in range(tn_):
                            element = ctx.load(
                                "mat", ctx.batch["mat"] + (bm + t) * ld + out_col + k
                            )
                            result[k] = _add(result[k], _mul(coeff, element))
                        t += 1
                    if t < tm_:
                        ctx.paths.add("partial-load")
            else:
                ctx.paths.add("divergent-skip")
            lanes[simd_gid, lane] = {
                "thr_m": thr_m,
                "cm": cm,
                "out_col": out_col,
                "result": result,
            }
    stores = []
    for (simd_gid, lane), state in sorted(lanes.items()):
        if state["thr_m"] != 0:
            continue
        # simd_shuffle_down(SN * sm) sums the lanes of one column block.
        group = [lanes[simd_gid, lane + sn * m] for m in range(sm)]
        assert all(item["out_col"] == state["out_col"] for item in group)
        total = [
            _sum(ctx.dtype, [item["result"][k] for item in group]) for k in range(tn_)
        ]
        if state["cm"] == 0 and state["out_col"] < columns:
            for k in range(tn_):
                column = state["out_col"] + k
                stores.append((column, ctx.finish(total[k], column)))
    return stores


def _simulate(entry, case, data, mutation=None):
    """Replay every threadgroup: output words, input accesses and paths."""
    info = INFO[entry]
    output = list(data["initial"])
    stored = {}
    accesses = {name: [] for name in ("mat", "in_vec", "bias", "out_vec")}
    paths = set()
    group = _gemv_group if info["kind"] == "gemv" else _gemv_t_group
    batches = data["batches"]
    if mutation == "batch":
        batches = [batches[0]] * len(batches)
    for z, batch in enumerate(batches):
        ctx = _Context(entry, data, batch, mutation)
        for gx in range(data["grid"][0]):
            for index, value in group(ctx, gx):
                position = z * ctx.out_size + index
                accesses["out_vec"].append(position)
                if mutation is None and position in stored:
                    # Shifted tail blocks store the same words twice.
                    assert stored[position] == value, (entry, case, position)
                stored[position] = value
                if 0 <= position < len(output):
                    output[position] = value
        for name, items in ctx.accesses.items():
            accesses[name] += items
        paths |= ctx.paths
    return output, accesses, paths


def _expected_paths(entry):
    info = INFO[entry]
    paths = {"iterations", "leftover", "partial-load"}
    if _blocks(info)[2] > 1:
        # Blocks of one row or column never overhang the output.
        paths.add("tail-shift")
    if info["kind"] == "gemv":
        if info["bm"] * info["sm"] > 1:
            paths.add("early-exit")
        if info["bn"] > 1:
            paths.add("threadgroup-reduction")
    else:
        paths.add("divergent-skip")
    return paths


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _output_count(entry, case):
    return len(_batching(entry, case)[1]) * _shape(entry, case)[0] + GUARD_COUNT


def _element_count():
    return sum(_output_count(entry, case) for entry in ENTRIES for case in CASES)


def _expected_validation_contract():
    return {
        "model": (
            "GEMVKernel::run and GEMVTKernel::run thread by thread: block "
            "positions, iteration and leftover loads, SIMD-group shuffle-down "
            "sums, the threadgroup reduction and guarded stores"
        ),
        "comparison": "exact words",
        "workgroupSize": "[32, bn, bm]",
        "grid": "[ceil(out_vec_size / outputs per threadgroup), 1, batches]",
        "cases": {
            "gemv": {
                "outVecSize": {
                    "short": "TM + 2",
                    "ragged": "2 * BM * SM * TM + TM + 1",
                    "aligned": "2 * BM * SM * TM",
                },
                "inVecSize": {
                    "short": "BN * SN * TN - TN - 1",
                    "ragged": "2 * BN * SN * TN + TN + 1",
                    "aligned": "BN * SN * TN",
                },
            },
            "gemv_t": {
                "outVecSize": {
                    "short": "TN + 2",
                    "ragged": "2 * BN * SN * TN + TN + 1",
                    "aligned": "2 * BN * SN * TN",
                },
                "inVecSize": {
                    "short": "SM * TM - 1",
                    "ragged": "2 * SM * TM + 3",
                    "aligned": "SM * TM",
                },
            },
            "matrixLdPadding": {"short": ROW_PAD, "ragged": ROW_PAD, "aligned": 0},
        },
        "batching": {
            "contiguous": {"batchShape": [2], "strides": "one per batch"},
            "nonContiguous": {
                "batchShape": [2, 2],
                "vectorStrides": "[0, vector]",
                "matrixStrides": "[matrix, 2 * matrix]",
                "biasStrides": "[2 * bias, bias]",
            },
            "gather": GATHER_LAYOUTS,
        },
        "axpby": {"alpha": ALPHA, "beta": BETA, "biasStride": BIAS_STRIDES},
        "inputs": {
            "vectors": "nonzero integers in [-2, 2]",
            "matrices": (
                "integers in [-2, 2] whose dot product with the batch's vector "
                "is nonzero and at most the element type's bound"
            ),
            "resultBounds": RESULT_BOUNDS,
            "complex64": "integer parts in [-2, 2]",
            "bias": "nonzero integers 2 mod 4, so beta * bias is odd",
            "poison": {
                "real": list(POISON),
                "complex64": [list(item) for item in COMPLEX_POISON],
            },
        },
        "guards": {
            "output": "every element starts at the sentinel, which no result equals",
            "sentinel": {"real": SENTINEL, "complex64": list(COMPLEX_SENTINEL)},
            "trailingOutputGuards": GUARD_COUNT,
        },
        "openglIndexRangeAssertions": {
            group: [
                {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                for expression in expressions
            ]
            for group, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
        },
        "softwareSubgroupWidth": {"targets": ["opengl", "directx"], "width": SIMD_SIZE},
        "directx": {"relativeWaveShuffleOutOfRange": "self"},
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype, _ in DTYPES}
            for target in TARGETS
        },
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES) + WIDE_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        "excludedEntries": {
            "kernels": ["gemv_wide", "gemv_wide_gather"],
            "count": WIDE_ENTRY_COUNT,
            "reason": (
                "test_current_gemv_wide.py pins and executes their 80 "
                "function-constant specializations"
            ),
        },
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "checkedElementCountByTarget": _element_count(),
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_gemv_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-gemv-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "headers": HEADERS,
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


def test_gemv_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "gemv.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 224
    kinds = {}
    for info in INFO.values():
        key = info["kind"] + ("_gather" if info["gather"] else "")
        kinds[key] = kinds.get(key, 0) + 1
    assert kinds == {"gemv": 112, "gemv_gather": 12, "gemv_t": 80, "gemv_t_gather": 20}
    assert ENTRIES[0] == "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0"
    assert ENTRIES[-1] == "gemv_t_gather_complex64_bm1_bn16_sm4_sn8_tm4_tn4"
    for entry, info in INFO.items():
        # matmul.cpp: "gemv[_t]_{type}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc{}_axpby{}"
        # and "gemv[_t]_gather_{type}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}".
        name = (
            f"{info['kind']}{'_gather' if info['gather'] else ''}_{info['dtype']}"
            f"_bm{info['bm']}_bn{info['bn']}_sm{info['sm']}_sn{info['sn']}"
            f"_tm{info['tm']}_tn{info['tn']}"
        )
        if not info["gather"]:
            name += f"_nc{int(info['nc'])}_axpby{int(info['axpby'])}"
        assert entry == name
        # GEMVKernel / GEMVTKernel: SM * SN == 32 lanes per SIMD-group.
        assert info["sm"] * info["sn"] == SIMD_SIZE


@pytest.mark.parametrize("entry", ENTRIES[::5])
def test_gemv_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    per_group, per_iteration, t_out, t_in = _blocks(info)
    # matmul.cpp: n_out_per_tgp = bm * sm * tm (gemv) or bn * sn * tn
    # (gemv_t), group_dims (32, bn, bm), grid (n_tgp, 1, batch_size_out).
    if info["kind"] == "gemv":
        assert per_group == info["bm"] * info["sm"] * info["tm"]
    else:
        assert per_group == info["bn"] * info["sn"] * info["tn"]
    assert _workgroup(entry) == [32, info["bn"], info["bm"]]
    for case in CASES:
        out_size, in_size, ld = _shape(entry, case)
        grid = _grid(entry, case)
        assert grid[0] == (out_size + per_group - 1) // per_group and grid[1] == 1
        # The tail block shifts inwards by whole rows or columns.
        assert out_size >= t_out and in_size >= 1
        rows_long = in_size if info["kind"] == "gemv" else out_size
        assert ld >= rows_long
        metadata, batches, _ = _batching(entry, case)
        assert grid[2] == len(batches) == _product(metadata["batch_shape"])
        if not info["gather"]:
            assert metadata["batch_ndim"] == (2 if info["nc"] else 1)
    # The short case never completes a block iteration, the others do.
    assert _shape(entry, "short")[1] < per_iteration
    assert _shape(entry, "aligned")[1] % per_iteration == 0
    assert _shape(entry, "ragged")[1] % per_iteration != 0


def _product(values):
    total = 1
    for value in values:
        total *= value
    return total


MODEL_ENTRIES = tuple(
    entry for entry in ENTRIES if INFO[entry]["dtype"] in ("float32", "complex64")
)[::3]


@pytest.mark.parametrize("entry", MODEL_ENTRIES)
def test_gemv_model_matches_dot_products(entry):
    """The thread-level model stores each direct dot product, and only those."""
    paths = set()
    for case in CASES:
        data = _case_data(entry, case)
        output, accesses, seen = _simulate(entry, case, data)
        paths |= seen
        assert output == data["expected"], (entry, case)
        assert sorted(set(accesses["out_vec"])) == list(range(data["written"]))
    assert paths >= _expected_paths(entry), sorted(_expected_paths(entry) - paths)


@pytest.mark.parametrize("dtype", [dtype for dtype, _ in DTYPES])
def test_gemv_inputs_keep_every_result_exact(dtype):
    """Results, guards and poison are exact; no result is zero or the sentinel."""
    bits = PRECISION["float32" if dtype == "complex64" else dtype]
    if dtype != "complex64":
        assert 2 * RESULT_BOUNDS[dtype] + 7 < 2**bits
    sentinel = COMPLEX_SENTINEL if dtype == "complex64" else SENTINEL
    assert _exact(dtype, sentinel)
    for value in (*POISON, *COMPLEX_POISON):
        assert _exact(dtype, value)
    for entry in [entry for entry in ENTRIES if INFO[entry]["dtype"] == dtype][::4]:
        for case in CASES:
            data = _case_data(entry, case)
            written = data["expected"][: data["written"]]
            assert all(_exact(dtype, value) for value in written)
            assert all(value != _zero(dtype) and value != sentinel for value in written)
            assert data["expected"][data["written"] :] == [sentinel] * GUARD_COUNT
            # Products and their sums stay far inside float32's 24 bits.
            assert 8 * data["shape"]["inVecSize"] < 2**24


@pytest.mark.parametrize(
    "entry",
    [entry for entry in ENTRIES if INFO[entry]["dtype"] == "bfloat16"][::2],
)
def test_gemv_model_justifies_opengl_index_assertions(entry):
    """Every load and store the asserted offsets form lies inside its buffer."""
    for case in CASES:
        data = _case_data(entry, case)
        _, accesses, _ = _simulate(entry, case, data)
        for name, indices in accesses.items():
            if not indices:
                continue
            length = len(data["values"][name][1])
            assert 0 <= min(indices) and max(indices) < length <= SIGNED_MAX, (
                case,
                name,
            )
        metadata, batches, _ = _batching(entry, case)
        offsets = [value for batch in batches for value in batch.values()]
        assert min(offsets) >= 0
        for name in ("vector_batch_stride", "matrix_batch_stride", "bias_batch_stride"):
            assert min(metadata.get(name, [0])) >= 0


MUTATIONS = {
    "leftover": "drops the leftover inputs",
    "shift": "does not shift the tail block inwards",
    "batch": "addresses every batch as the first",
    "bias": "drops beta * bias",
    "reduction": "skips the threadgroup reduction",
}


@pytest.mark.parametrize(
    "entry,mutation",
    (
        ("gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc1_axpby1", "leftover"),
        ("gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc1_axpby1", "reduction"),
        ("gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc1_axpby1", "bias"),
        ("gemv_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0", "shift"),
        ("gemv_float16_bm1_bn1_sm8_sn4_tm4_tn4_nc1_axpby0", "batch"),
        ("gemv_gather_complex64_bm4_bn1_sm1_sn32_tm4_tn4", "batch"),
        ("gemv_t_float32_bm1_bn16_sm4_sn8_tm4_tn4_nc0_axpby1", "leftover"),
        ("gemv_t_bfloat16_bm1_bn2_sm8_sn4_tm4_tn4_nc1_axpby1", "shift"),
        ("gemv_t_float16_bm1_bn4_sm8_sn4_tm4_tn4_nc1_axpby1", "bias"),
        ("gemv_t_gather_float32_bm1_bn2_sm8_sn4_tm4_tn1", "batch"),
    ),
)
def test_gemv_expectations_reject_defects(entry, mutation):
    assert mutation in MUTATIONS
    exposed = False
    for case in CASES:
        data = _case_data(entry, case)
        output, _, _ = _simulate(entry, case, data, mutation)
        exposed |= output != data["expected"]
    assert exposed, (entry, mutation)


@pytest.mark.parametrize("target", TARGETS)
def test_gemv_configs_scope_opengl_index_assertions(tmp_path, target):
    for entry in ENTRIES[::9]:
        config = _config(tmp_path / "work", target, entry)
        assertions = [item.to_json() for item in config.index_range_assertions]
        if target != "opengl":
            assert assertions == []
            continue
        assert [item["expression"] for item in assertions] == list(
            _index_assertions(entry)
        )
        assert {(item["minimum"], item["maximum"]) for item in assertions} == {
            (0, SIGNED_MAX)
        }


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------


def _source_tables(text):
    """Tile shapes, element types and variants of gemv.metal's macros."""

    def blocks(helper):
        rows = re.findall(rf"\b{helper}\(name, itype,\s*([\d,\s]+?)\)", text)
        return tuple(tuple(int(part) for part in row.split(",")) for row in rows)

    def types(macro):
        return tuple(
            re.findall(rf"^{macro}\((\w+), (\w+)\);", text, flags=re.MULTILINE)
        )

    variants = tuple(
        (int(nc), int(axpby))
        for nc, axpby in re.findall(
            r"instantiate_gemv_helper\(name, itype, bm, bn, sm, sn, tm, tn, (\d), (\d)\)",
            text,
        )
    )
    return {
        "gemv": blocks("instantiate_gemv"),
        "gemv_gather": blocks("instantiate_gemv_bs_helper"),
        "gemv_t": blocks("instantiate_gemv_t"),
        "gemv_t_gather": blocks("instantiate_gemv_t_bs_helper"),
        "types": tuple(
            types(macro)
            for macro in (
                "instantiate_gemv_blocks",
                "instantiate_gemv_bs_blocks",
                "instantiate_gemv_t_blocks",
                "instantiate_gemv_t_bs_blocks",
            )
        ),
        "variants": variants,
    }


def _kernel_parameters(text, kernel):
    match = re.search(rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", text, re.DOTALL)
    assert match, kernel
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", match.group(1))
    assert [int(index) for _, index in found] == list(range(len(found)))
    return tuple(name for name, _ in found)


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native GEMV gate")
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
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in HEADERS.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # The tables above are gemv.metal's own instantiation lists, and the
    # parameter tuples gemv.h's [[buffer(i)]] signatures.
    tables = _source_tables(source.decode("utf-8"))
    assert tables["gemv"] == GEMV_BLOCKS
    assert tables["gemv_gather"] == GEMV_GATHER_BLOCKS
    assert tables["gemv_t"] == GEMV_T_BLOCKS
    assert tables["gemv_t_gather"] == GEMV_T_GATHER_BLOCKS
    assert tables["types"] == (DTYPES,) * 4
    assert tables["variants"] == NC_AXPBY
    header = (root / "mlx/backend/metal/kernels/gemv.h").read_text(encoding="utf-8")
    for kernel in ("gemv", "gemv_t"):
        assert _kernel_parameters(header, kernel) == GEMV_PARAMETERS
    for kernel in ("gemv_gather", "gemv_t_gather"):
        assert _kernel_parameters(header, kernel) == GATHER_PARAMETERS
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("gemv-metal-reference")
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
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    descriptor, package = _prepare_native_package(translation, work)
    return descriptor, package, record


def _request(descriptor, package, target, entry, data):
    """Bind every reflected binding by source parameter and build one dispatch."""
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    dtype = INFO[entry]["dtype"]
    values = {
        name: _runtime_value(target, kind, storage)
        for name, (kind, storage) in data["values"].items()
    }
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        source = binding.get("provenance", {}).get("sourceResource")
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
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "out_vec":
            assert binding["access"] == "read_write", binding
            output = binding["name"]
        else:
            observed.append(binding["name"])
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    assert output is not None and {"mat", "in_vec", "out_vec"} <= bound <= set(values)
    expected = _runtime_value(target, dtype, data["expected"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: expected},
        {"workgroupCount": data["grid"], "workgroupSize": _workgroup(entry)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, len(observed)


def _check(label, target, entry, actual, data):
    expected = _runtime_value(target, INFO[entry]["dtype"], data["expected"])["values"]
    _assert_identical(label, actual, expected)


def _assert_identical(label, actual, expected):
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    info = INFO[entry]
    parameters = GATHER_PARAMETERS if info["gather"] else GEMV_PARAMETERS
    payloads = [_metal_bytes(*data["values"][name]) for name in parameters]
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
                "workgroupCount": data["grid"],
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
    output = parameters.index("out_vec")
    for index, payload in enumerate(payloads):
        if index != output:
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    raw = (readback / f"buffer-{output}.bin").read_bytes()
    if info["dtype"] in {"float16", "bfloat16"}:
        words = list(struct.unpack(f"<{len(raw) // 2}H", raw))
    else:
        words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return device, words


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_gemv_native_parity(current_mlx, metal_reference, tmp_path, entry):
    root, target = current_mlx
    with tempfile.TemporaryDirectory(prefix=".current-gemv-", dir=root) as directory:
        work = Path(directory)
        try:
            descriptor, package, record = _translate(root, work, target, entry)
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    words = _execute(
                        executor, work, case, request, {"out": output}, observed
                    )["out"]["values"]
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work, metal_reference, entry, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        _assert_identical(case, words, upstream_words)
                        upstream = {"device": device, "identicalToTranslation": True}
                    _check(case, target, entry, words, data)
                    evidence.append(
                        {
                            "case": case,
                            **data["shape"],
                            "grid": data["grid"],
                            "batchCount": len(data["batches"]),
                            "writtenElementCount": data["written"],
                            "checkedElementCount": data["size"],
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


def test_ci_requires_native_gemv_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX GEMV native parity", group="gemv"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/gemv",
        "--junitxml=mlx-current-results/gemv.xml",
        "demos/integrations/mlx/tests/kernels/test_current_gemv.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_gemv.py"
    )
