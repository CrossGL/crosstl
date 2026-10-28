"""Natively prove quantized.metal's 162 affine qmv_fast matrix-vector kernels.

``quantized.metal`` instantiates ``affine_qmv_fast`` with and without batching
and ``affine_gather_qmv_fast`` for float, float16_t and bfloat16_t, group sizes
128, 64 and 32 and 2, 3, 4, 5, 6 and 8 bits: 162 of its 2,052 entries. Each is
translated and pinned per target, then runs ``quantized.cpp``'s ``qmv`` or
``gather_qmv`` dispatch: threadgroups of ``(32, 2, 1)`` threads, two
simdgroups of four output rows each, over a grid of ``(M, N / 8, B)``.

Unbatched entries run a single vector and a three-row matrix. Batched entries
run stacked batches, a broadcast two-dimensional batch with transposed
activations and broadcast weights, and strided activations sharing one weight
matrix, reaching both branches of ``adjust_matrix_offsets``. Gather entries
select activations and weights through one- and two-dimensional index
batches. Inputs are quarters in ``[-1, 1]``, power-of-two scales and biases
``-a * scale``: every product, partial sum, ``simd_sum`` and accumulator is a
multiple of ``2^-5`` below ``2^19``, so float32 holds each exactly in any
order and only the final ``static_cast<T>`` rounds, to nearest even. A model
replays each lane's loads, ``qdot`` unpacking and offsets and must reach a
direct dequantized dot product. Each target must produce the model's words
and leave every guard untouched; on Metal the translation must also match the
unchanged upstream kernel word for word.
"""

import functools
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
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
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    HEADERS,
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
    UNIT_ENTRY_COUNT,
    _macro,
    _pack_bitstream,
    _to_type,
    _words,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _below,
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QMV"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_qmv.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# adjust_matrix_offsets' general branches use steel/utils.h's
# elem_to_loc_broadcast.
QMV_HEADERS = {
    **HEADERS,
    "mlx/backend/metal/kernels/steel/utils.h": (
        "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a"
    ),
}

DTYPES = (("float", "float32"), ("float16_t", "float16"), ("bfloat16_t", "bfloat16"))
GROUP_SIZES = (128, 64, 32)
BITS = (2, 3, 4, 5, 6, 8)
# Per (type, group size, bits), instantiate_quantized_all_single instantiates
# affine_gather_qmv_fast before instantiate_quantized_all_batched's
# affine_qmv_fast with batched 1, then 0.
VARIANTS = ("gather", "batch_1", "batch_0")
KERNELS = {
    "gather": "affine_gather_qmv_fast",
    "batch_1": "affine_qmv_fast",
    "batch_0": "affine_qmv_fast",
}
# The 108 quantize and dequantize entries have their own gate,
# test_current_quantized.py, the 162 affine_qmv and affine_gather_qmv entries
# theirs, test_current_quantized_qmv_unaligned.py, the 432 affine_qmv_wide
# entries theirs, test_current_quantized_qmv_wide.py, the 162 affine_qvm
# and affine_gather_qvm entries theirs, test_current_quantized_qvm.py, the 108
# affine_qvm_split_k entries theirs, test_current_quantized_qvm_split_k.py,
# the 216 affine_qmm_t and 108 affine_qmm_n entries theirs,
# test_current_quantized_qmm_t.py and test_current_quantized_qmm_n.py, the 108
# affine_qmm_t_splitk entries theirs, test_current_quantized_qmm_t_splitk.py,
# the 162 affine_gather_qmm_t and affine_gather_qmm_n entries theirs,
# test_current_quantized_gather_qmm.py, and the 216 affine_qmv_quad entries
# theirs, test_current_quantized_qmv_quad.py.
QUANTIZE_ENTRY_COUNT = 108
QMV_UNALIGNED_ENTRY_COUNT = 162
QMV_WIDE_ENTRY_COUNT = 432
QVM_ENTRY_COUNT = 162
QVM_SPLIT_K_ENTRY_COUNT = 108
QMM_T_ENTRY_COUNT = 216
QMM_N_ENTRY_COUNT = 108
QMM_T_SPLITK_ENTRY_COUNT = 108
GATHER_QMM_ENTRY_COUNT = 162
QMV_QUAD_ENTRY_COUNT = 216
# Kernels of quantized.metal no gate covers yet.
EXCLUDED_KERNELS = (
    "affine_gather_qmm_rhs_nt",
    "affine_gather_qmm_rhs_nn",
)

SIMD_SIZE = 32
# qmv's and gather_qmv's group_dims (bk, 2, 1); each simdgroup produces
# results_per_simdgroup rows, so a threadgroup covers bn = 8 of them.
WORKGROUP = (32, 2, 1)
NUM_SIMDGROUPS = 2
RESULTS_PER_SIMDGROUP = 4
ROWS_PER_GROUP = NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP

# [[buffer(i)]] parameters of the kernels, as (name, index).
QMV_PARAMETERS = (
    ("w", 0),
    ("scales", 1),
    ("biases", 2),
    ("x", 3),
    ("y", 4),
    ("in_vec_size", 5),
    ("out_vec_size", 6),
    ("x_batch_ndims", 7),
    ("x_shape", 8),
    ("x_strides", 9),
    ("w_batch_ndims", 10),
    ("w_shape", 11),
    ("w_strides", 12),
    ("s_strides", 13),
    ("b_strides", 14),
)
GATHER_PARAMETERS = (
    ("w", 0),
    ("scales", 1),
    ("biases", 2),
    ("x", 3),
    ("lhs_indices", 4),
    ("rhs_indices", 5),
    ("y", 6),
    ("in_vec_size", 7),
    ("out_vec_size", 8),
    ("x_batch_ndims", 9),
    ("x_shape", 10),
    ("x_strides", 11),
    ("w_batch_ndims", 12),
    ("w_shape", 13),
    ("w_strides", 14),
    ("s_strides", 15),
    ("b_strides", 16),
    ("batch_ndims", 17),
    ("batch_shape", 18),
    ("lhs_strides", 19),
    ("rhs_strides", 20),
)
# qmv binds only w, scales, biases, x, y, K and N when B <= 1:
# add_strides_and_shapes skips the batch parameters.
UNBATCHED_BOUND = 7
SCALAR_TYPES = {
    "in_vec_size": "int32",
    "out_vec_size": "int32",
    "x_batch_ndims": "int32",
    "w_batch_ndims": "int32",
    "batch_ndims": "int32",
}
ARRAY_TYPES = {
    "x_shape": "int32",
    "w_shape": "int32",
    "batch_shape": "int32",
    "x_strides": "int64",
    "w_strides": "int64",
    "s_strides": "int64",
    "b_strides": "int64",
    "lhs_strides": "int64",
    "rhs_strides": "int64",
    "lhs_indices": "uint32",
    "rhs_indices": "uint32",
}

CASES = {
    "batch_0": ("vector", "matrix"),
    "batch_1": ("stacked", "broadcast", "shared"),
    "gather": ("experts", "broadcast"),
}
GUARD_COUNT = 8
SENTINEL = -3.25
# Exactly representable poison for every element no case may read.
POISON = 1.5
WORD_POISON = 0xA5A5A5A5
INDEX_POISON = 0x5A5A5A5A

# Expressions OpenGL may evaluate in 32 bits: the batch offsets of
# adjust_matrix_offsets' one-dimensional branches, the elem_to_loc_broadcast
# offsets of its general branches, and gather's index-buffer subscripts. The
# model shows each names an element inside its buffer, and every case's
# buffers hold far fewer than INDEX_BOUND elements.
INDEX_BOUND = 2**20 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "batch_0": (),
    "batch_1": (
        "x_idx * x_strides[0]",
        "w_idx * w_strides[0]",
        "w_idx * s_strides[0]",
        "w_idx * b_strides[0]",
        "idx.x",
        "idx.y",
        "idx.z",
    ),
    "gather": (
        "tid.z * lhs_strides[0]",
        "tid.z * rhs_strides[0]",
        "idx.x",
        "idx.y",
        "x_idx * x_strides[0]",
        "w_idx * w_strides[0]",
        "w_idx * s_strides[0]",
        "w_idx * b_strides[0]",
        "idx.z",
    ),
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for variant in VARIANTS:
                    stem = f"{KERNELS[variant]}_{itype}_gs_{group_size}_b_{bits}"
                    entry = stem if variant == "gather" else f"{stem}_{variant}"
                    yield entry, {
                        "kernel": KERNELS[variant],
                        "variant": variant,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry, variant=None, workgroup=WORKGROUP):
    """The translation of one entry, as the gate and contract run it.

    ``variant`` defaults to the entry's own; the qmv gate passes its
    entries' variants, and ``workgroup`` is the threadgroup size the host
    dispatches.
    """
    variant = variant or INFO[entry]["variant"]
    # simd_sum: software subgroups give DirectX and OpenGL MLX's 32 lanes.
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    assertions = OPENGL_INDEX_RANGE_ASSERTIONS[variant] if target == "opengl" else ()
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=INDEX_BOUND)
            for expression in assertions
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "target_options": options,
            }
        },
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits, group_size):
    """qmv_fast_impl's constants for one entry."""
    packs_per_thread = 1 if bits == 2 else 2
    # get_pack_factor<bits, 32>() and get_bytes_per_pack<bits, 32>().
    pack_factor = 8 if bits in (3, 5) else 4 if bits == 6 else 32 // bits
    power_of_2_bits = bits & (bits - 1) == 0
    bytes_per_pack = 4 if power_of_2_bits else 5 if bits == 5 else 3
    values_per_thread = pack_factor * packs_per_thread
    return {
        "packsPerThread": packs_per_thread,
        "packFactor": pack_factor,
        "bytesPerPack": bytes_per_pack,
        "valuesPerThread": values_per_thread,
        "blockSize": values_per_thread * SIMD_SIZE,
        "scaleStepPerThread": group_size // values_per_thread,
    }


# (M, N, K in blocks): N is a multiple of bn and K of qmv_fast_k_alignment,
# the conditions under which qmv and gather_qmv pick the fast kernels.
SHAPES = {
    "vector": (1, 8, 1),
    "matrix": (3, 24, 2),
    "stacked": (1, 16, 2),
    "broadcast": (2, 8, 1),
    "shared": (1, 8, 2),
    "experts": (1, 16, 2),
}
# Gather's index buffers: the experts case reads every other lhs index.
EXPERT_LHS = (1, 0, 0, 1)
EXPERT_RHS = (2, 0, 1, 2)
BROADCAST_LHS = (3, 0, 1, 2)
BROADCAST_RHS = (3, 0)


# The batch layout each case follows.
LAYOUTS = {
    "vector": "unbatched",
    "matrix": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
    "experts": "experts",
}


def _layout(entry, case):
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n, blocks = SHAPES[case]
    k = blocks * _geometry(info["bits"], info["groupSize"])["blockSize"]
    return layout_for(info, LAYOUTS[case], m, n, k)


def layout_for(info, kind, m, n, k, transposed=False):
    """quantized.cpp's shapes, strides and index buffers for one case.

    ``kind`` is one of LAYOUTS' values. ``batches`` gives, per ``tid.z``, the
    x element offset, the physical weight matrix and the output offset the
    arrays are built for; the kernel model derives the same from the bound
    parameters. qmv's weights are ``N`` rows of ``K`` codes; ``transposed``
    lays them out as qvm reads them, ``K`` rows of ``N`` codes, over qvm's
    ``(M, ceil(N / 64), B)`` grid.
    """
    variant, bits, group_size = info["variant"], info["bits"], info["groupSize"]
    rows, codes = (k, n) if transposed else (n, k)
    groups, words = codes // group_size, codes * bits // 32
    tile = 2 * SIMD_SIZE if transposed else ROWS_PER_GROUP
    matrix = m * k
    layout = {
        "M": m,
        "N": n,
        "K": k,
        "rows": rows,
        "groups": groups,
        "words": words,
        "transposed": transposed,
    }
    params = {}
    if kind == "unbatched":
        # add_strides_and_shapes would bind these unbatched shapes; with
        # B <= 1 qmv skips it and the kernel never reads them.
        params = {
            "x_batch_ndims": 0,
            "x_shape": [m, k],
            "x_strides": [k, 1],
            "w_batch_ndims": 0,
            "w_shape": [rows, words],
            "w_strides": [words, 1],
            "s_strides": [groups, 1],
            "b_strides": [groups, 1],
        }
        batches = [{"x": 0, "w": 0}]
        x_rows, matrices = m, 1
    elif kind == "stacked":
        params = {
            "x_batch_ndims": 1,
            "x_shape": [2, m, k],
            "x_strides": [matrix, k, 1],
            "w_batch_ndims": 1,
            "w_shape": [2, rows, words],
            "w_strides": [rows * words, words, 1],
            "s_strides": [rows * groups, groups, 1],
            "b_strides": [rows * groups, groups, 1],
        }
        batches = [{"x": z * matrix, "w": z} for z in range(2)]
        x_rows, matrices = 2 * m, 2
    elif kind == "shared":
        # Every other row of a strided activation view against one weight
        # matrix: w.ndim() == 2 gives w_batch_ndims 0.
        params = {
            "x_batch_ndims": 1,
            "x_shape": [3, m, k],
            "x_strides": [2 * matrix, k, 1],
            "w_batch_ndims": 0,
            "w_shape": [rows, words],
            "w_strides": [words, 1],
            "s_strides": [groups, 1],
            "b_strides": [groups, 1],
        }
        batches = [{"x": 2 * z * matrix, "w": 0} for z in range(3)]
        x_rows, matrices = 6 * m, 1
    else:
        # A (2, 2) batch: activations with transposed batch strides and
        # weights broadcast along the second batch dimension.
        x_params = {
            "x_batch_ndims": 2,
            "x_shape": [2, 2, m, k],
            "x_strides": [matrix, 2 * matrix, k, 1],
        }
        w_params = {
            "w_shape": [2, 2, rows, words],
            "w_strides": [rows * words, 0, words, 1],
            "s_strides": [rows * groups, 0, groups, 1],
            "b_strides": [rows * groups, 0, groups, 1],
        }
        x_rows, matrices = 4 * m, 2

        def x_offset(index):
            return (index % 2) * 2 * matrix + (index // 2) * matrix

        if kind == "experts":
            params = {
                "x_batch_ndims": 1,
                "x_shape": [2, m, k],
                "x_strides": [matrix, k, 1],
                "w_batch_ndims": 1,
                "w_shape": [3, rows, words],
                "w_strides": [rows * words, words, 1],
                "s_strides": [rows * groups, groups, 1],
                "b_strides": [rows * groups, groups, 1],
                "batch_ndims": 1,
                "batch_shape": [len(EXPERT_LHS)],
                "lhs_strides": [2],
                "rhs_strides": [1],
            }
            lhs = []
            for index in EXPERT_LHS:
                lhs += [index, INDEX_POISON]
            layout["lhs"], layout["rhs"] = lhs, list(EXPERT_RHS)
            batches = [
                {"x": x * matrix, "w": w} for x, w in zip(EXPERT_LHS, EXPERT_RHS)
            ]
            x_rows, matrices = 2 * m, 3
        elif variant == "gather":
            # Indices of shape (2, 2) against rhs indices broadcast from
            # (2, 1): collapse_contiguous_dims keeps both dimensions.
            params = {
                **x_params,
                "w_batch_ndims": 2,
                **w_params,
                "batch_ndims": 2,
                "batch_shape": [2, 2],
                "lhs_strides": [2, 1],
                "rhs_strides": [1, 0],
            }
            layout["lhs"], layout["rhs"] = list(BROADCAST_LHS), list(BROADCAST_RHS)
            batches = [
                {"x": x_offset(BROADCAST_LHS[z]), "w": BROADCAST_RHS[z // 2] // 2 % 2}
                for z in range(4)
            ]
        else:
            params = {**x_params, "w_batch_ndims": 2, **w_params}
            batches = [{"x": x_offset(z), "w": z // 2} for z in range(4)]
    for z, batch in enumerate(batches):
        batch["y"] = z * m * n
    layout.update(
        params=params,
        batches=batches,
        xRows=x_rows,
        matrices=matrices,
        grid=(m, -(-n // tile), len(batches)),
    )
    return layout


def _parameters(variant):
    return GATHER_PARAMETERS if variant == "gather" else QMV_PARAMETERS


def _exact_dyadic(value, scale=32):
    """Whether ``value`` is a multiple of ``1 / scale``."""
    return math.isfinite(value) and float(value * scale).is_integer()


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    return case_data_for(INFO[entry], entry, case, _layout(entry, case))


def case_data_for(info, entry, case, layout):
    """Logical storage of every parameter, the grid and the expected outputs."""
    dtype, bits, group_size = info["dtype"], info["bits"], info["groupSize"]
    n_bins = (1 << bits) - 1
    m, n, k = layout["M"], layout["N"], layout["K"]
    groups = layout["groups"]
    draws = _draws("quantized-qmv", entry, case)
    used_rows = {
        batch["x"] + row * k for batch in layout["batches"] for row in range(m)
    }
    x, inputs = [POISON] * (layout["xRows"] * k), []
    for start in sorted(used_rows):
        x[start : start + k] = [_signed(draws, 4) / 4 for _ in range(k)]
        inputs += x[start : start + k]
    codes, scales, biases = [], [], []
    for _ in range(layout["matrices"] * layout["rows"]):
        row = []
        for _ in range(groups):
            values = [_below(draws, n_bins + 1) for _ in range(group_size)]
            low = _below(draws, group_size)
            values[low] = 0
            values[(low + 1 + _below(draws, group_size - 1)) % group_size] = n_bins
            row += values
            scale = 2.0 ** -_below(draws, 4) * (1 if _below(draws, 2) else -1)
            scales.append(scale)
            biases.append(-_below(draws, n_bins + 1) * scale)
        codes.append(row)
    w = []
    for row in codes:
        payload = _pack_bitstream(row, bits)
        w += list(struct.unpack(f"<{len(payload) // 4}I", bytes(payload)))
    assert len(w) == layout["matrices"] * layout["rows"] * layout["words"]
    # Every element lies on the quarter, 2^-3 and 2^-5 grids the exactness
    # argument needs and is exact in the element type.
    assert all(_exact_dyadic(value, 4) and abs(value) <= 1 for value in inputs)
    assert all(abs(scale) in (1.0, 0.5, 0.25, 0.125) for scale in scales)
    assert all(_exact_dyadic(bias, 8) for bias in biases)
    for value in x + scales + biases:
        assert _to_type(dtype, value) == value, (entry, value)
    total = len(layout["batches"]) * m * n
    initial = [SENTINEL] * (total + GUARD_COUNT)
    params = layout["params"]
    values = {
        "w": ("uint32", w + [WORD_POISON] * GUARD_COUNT),
        "scales": (dtype, scales + [POISON] * GUARD_COUNT),
        "biases": (dtype, biases + [POISON] * GUARD_COUNT),
        "x": (dtype, x + [POISON] * GUARD_COUNT),
        "y": (dtype, initial),
        "in_vec_size": ("int32", [k]),
        "out_vec_size": ("int32", [n]),
        **{
            name: (SCALAR_TYPES[name], [value])
            for name, value in params.items()
            if name in SCALAR_TYPES
        },
        **{
            name: (ARRAY_TYPES[name], list(value))
            for name, value in params.items()
            if name in ARRAY_TYPES
        },
    }
    if info["variant"] == "gather":
        values["lhs_indices"] = ("uint32", list(layout["lhs"]))
        values["rhs_indices"] = ("uint32", list(layout["rhs"]))
    assert set(values) == {name for name, _ in _parameters(info["variant"])}, entry
    data = {
        "layout": layout,
        "codes": codes,
        "values": values,
        "initial": {"y": initial},
        "kinds": {"y": dtype},
        "grid": layout["grid"],
    }
    data["expected"], data["totals"], data["budget"] = _direct(info, data)
    return data


def _direct(info, data):
    """Each output as a dequantized dot product, its float32 total and budget.

    The budget bounds the magnitude of every partial sum of the kernel's
    float32 arithmetic, whatever its order: below 2^19, every multiple of
    2^-5 it can reach is exact in float32.
    """
    dtype, group_size = info["dtype"], info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    x = data["values"]["x"][1]
    scales, biases = data["values"]["scales"][1], data["values"]["biases"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    for batch in layout["batches"]:
        for row in range(m):
            vector = x[batch["x"] + row * k : batch["x"] + (row + 1) * k]
            for col in range(n):
                total, bound = 0.0, 0.0
                for index, value in enumerate(vector):
                    # Weight row and code of input ``index`` and output ``col``.
                    if layout["transposed"]:
                        line, position = batch["w"] * k + index, col
                    else:
                        line, position = batch["w"] * n + col, index
                    code = data["codes"][line][position]
                    group = line * groups + position // group_size
                    scale, bias = scales[group], biases[group]
                    total += (scale * code + bias) * value
                    bound += abs(value) * (abs(scale) * code + abs(bias))
                assert _exact_dyadic(total) and bound < 2**19, (info, total, bound)
                budget = max(budget, bound)
                index = batch["y"] + row * n + col
                totals[index] = total
                expected[index] = _to_type(dtype, total)
    return {"y": expected}, totals, budget


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _checked(value):
    """A float32 intermediate the exactness argument covers."""
    assert _f32(value) == value and _exact_dyadic(value), value
    return value


def _elem_to_loc(elem, shape, strides, ndim):
    """utils.h elem_to_loc and steel/utils.h elem_to_loc_broadcast."""
    locations = [0] * len(strides)
    for axis in range(ndim - 1, -1, -1):
        if elem <= 0:
            break
        position = elem % shape[axis]
        elem //= shape[axis]
        for index, values in enumerate(strides):
            locations[index] += position * values[axis]
    return locations


# load_vector's divisors: each input is pre-scaled by the bit position of its
# code within the packed byte or half word qdot masks it from.
X_DIVISORS = {
    2: (1, 4, 16, 64),
    3: (1, 8, 64, 2, 16, 128, 4, 32),
    4: (1, 16, 256, 4096),
    5: (1, 32, 4, 128, 16, 2, 64, 8),
    6: (1, 64, 16, 4),
    8: (1,),
}


def _load_vector(dtype, bits, values):
    """load_vector: the scaled inputs and their sum, runs added in ``T``."""
    if bits == 8:
        total = 0.0
        for value in values:
            total = _checked(total + value)
        return list(values), total
    run = 8 if bits in (3, 5) else 4
    total = 0.0
    for start in range(0, len(values), run):
        partial = values[start]
        for value in values[start + 1 : start + run]:
            exact = partial + value
            # x[i] + x[i + 1] + ... adds in the element type.
            partial = _to_type(dtype, exact)
            assert partial == exact, (dtype, exact)
        total = _checked(total + partial)
    divisors = X_DIVISORS[bits]
    return [
        value / divisors[index % len(divisors)] for index, value in enumerate(values)
    ], total


def _qdot(bits, byte, base, x_thread, scale, bias, total, mutation):
    """qdot over one lane's packed bytes, expression for expression."""
    count = len(x_thread)
    accum = 0.0

    def add(value):
        nonlocal accum
        accum = _checked(accum + value)

    x = x_thread
    if bits == 2:
        for i in range(count // 4):
            w = byte(base + i)
            add(
                x[4 * i] * (w & 0x03)
                + x[4 * i + 1] * (w & 0x0C)
                + x[4 * i + 2] * (w & 0x30)
                + x[4 * i + 3] * (w & 0xC0)
            )
    elif bits == 4:
        # qdot reads the bytes as little-endian uint16_t words.
        for i in range(count // 4):
            ws = byte(base + 2 * i) | byte(base + 2 * i + 1) << 8
            add(
                x[4 * i] * (ws & 0x000F)
                + x[4 * i + 1] * (ws & 0x00F0)
                + x[4 * i + 2] * (ws & 0x0F00)
                + x[4 * i + 3] * (ws & 0xF000)
            )
    elif bits == 8:
        for i in range(count):
            add(x[i] * byte(base + i))
    else:
        # 3, 5 and 6 bits advance x_thread and w by 8 * i (or 4 * i) and
        # width * i each iteration, cumulatively, as the kernel does.
        run, width = {3: (8, 3), 5: (8, 5), 6: (4, 3)}[bits]
        offset_x = offset_w = 0
        for i in range(count // run):
            offset_x += run * i
            offset_w += width * i
            w = [byte(base + offset_w + j) for j in range(width)]
            t = x[offset_x : offset_x + run]
            if bits == 3:
                terms = (
                    (w[0] & 0x07) * t[0],
                    (w[0] & 0x38) * t[1],
                    (w[0] & 0xC0) * t[2],
                    (w[1] & 0x01) * (t[2] * 256.0),
                    (w[1] & 0x0E) * t[3],
                    (w[1] & 0x70) * t[4],
                    (w[1] & 0x80) * t[5],
                    (w[2] & 0x03) * (t[5] * 256.0),
                    (w[2] & 0x1C) * t[6],
                    (w[2] & 0xE0) * t[7],
                )
            elif bits == 5:
                terms = (
                    (w[0] & 0x1F) * t[0],
                    (w[0] & 0xE0) * t[1],
                    (w[1] & 0x3) * (t[1] * 256.0),
                    (w[1] & 0x7C) * t[2],
                    (w[1] & 0x80) * t[3],
                    (w[2] & 0xF) * (t[3] * 256.0),
                    (w[2] & 0xF0) * t[4],
                    (w[3] & 0x1) * (t[4] * 256.0),
                    (w[3] & 0x3E) * t[5],
                    (w[3] & 0xC0) * t[6],
                    (w[4] & 0x7) * (t[6] * 256.0),
                    (w[4] & 0xF8) * t[7],
                )
            else:
                terms = (
                    (w[0] & 0x3F) * t[0],
                    (w[0] & 0xC0) * t[1],
                    (w[1] & 0x0F) * (t[1] * 256.0),
                    (w[1] & 0xF0) * t[2],
                    (w[2] & 0x03) * (t[2] * 256.0),
                    (w[2] & 0xFC) * t[3],
                )
            for term in terms:
                add(term)
    if mutation == "bias":
        return _checked(scale * accum)
    return _checked(_checked(scale * accum) + _checked(total * bias))


MUTATIONS = {
    "bias": "drops the sum * bias term",
    "scale-step": "gives every lane its block's first scale and bias",
    "order": "pairs each pack's inputs with its codes in reverse",
    "rows": "ignores the simdgroup's row offset",
    "batch": "ignores the weight batch offsets",
}


def _adjust(variant, storage, z, mutation, records):
    """adjust_matrix_offsets for ``tid.z``: x, w, scales, biases and y bases."""

    def scalar(name):
        return storage[name][0]

    if variant == "gather":
        if scalar("batch_ndims") == 1:
            lhs = z * storage["lhs_strides"][0]
            rhs = z * storage["rhs_strides"][0]
            records += [
                ("tid.z * lhs_strides[0]", "lhs_indices", lhs),
                ("tid.z * rhs_strides[0]", "rhs_indices", rhs),
            ]
        else:
            lhs, rhs = _elem_to_loc(
                z,
                storage["batch_shape"],
                (storage["lhs_strides"], storage["rhs_strides"]),
                scalar("batch_ndims"),
            )
            records += [("idx.x", "lhs_indices", lhs), ("idx.y", "rhs_indices", rhs)]
        x_idx, w_idx = storage["lhs_indices"][lhs], storage["rhs_indices"][rhs]
    else:
        x_idx = w_idx = z
    if scalar("x_batch_ndims") == 1:
        x_base = x_idx * storage["x_strides"][0]
        records.append(("x_idx * x_strides[0]", "x", x_base))
    else:
        (x_base,) = _elem_to_loc(
            x_idx, storage["x_shape"], (storage["x_strides"],), scalar("x_batch_ndims")
        )
    strides = (storage["w_strides"], storage["s_strides"], storage["b_strides"])
    if scalar("w_batch_ndims") == 1:
        bases = [w_idx * values[0] for values in strides]
        records += [
            (f"w_idx * {name}[0]", buffer, value)
            for name, buffer, value in zip(
                ("w_strides", "s_strides", "b_strides"),
                ("w", "scales", "biases"),
                bases,
            )
        ]
    else:
        bases = _elem_to_loc(
            w_idx, storage["w_shape"], strides, scalar("w_batch_ndims")
        )
        records += [
            (f"idx.{axis}", buffer, value)
            for axis, buffer, value in zip("xyz", ("w", "scales", "biases"), bases)
        ]
    if mutation == "batch":
        bases = [0, 0, 0]
    rows = storage["x_shape"][scalar("x_batch_ndims")]
    return (x_base, *bases, z * scalar("out_vec_size") * rows)


def _simulate(entry, data, mutation=None):
    """The kernel lane by lane; returns outputs, stores, reads and offsets."""
    info = INFO[entry]
    dtype, bits, group_size = info["dtype"], info["bits"], info["groupSize"]
    geometry = _geometry(bits, group_size)
    pack_factor, bytes_per_pack = geometry["packFactor"], geometry["bytesPerPack"]
    per_thread, block = geometry["valuesPerThread"], geometry["blockSize"]
    step = geometry["scaleStepPerThread"]
    storage = {name: values for name, (_, values) in data["values"].items()}
    in_vec_size, out_vec_size = storage["in_vec_size"][0], storage["out_vec_size"][0]
    in_vec_size_w = in_vec_size * bytes_per_pack // pack_factor
    in_vec_size_g = in_vec_size // group_size
    reads = {name: set() for name in storage}
    outputs = {"y": list(data["initial"]["y"])}
    stores, records = Counter(), []

    def load(name, index):
        assert 0 <= index < len(storage[name]) - GUARD_COUNT, (name, index)
        reads[name].add(index)
        return storage[name][index]

    def byte(index):
        word = load("w", index // 4)
        return (word >> (8 * (index % 4))) & 0xFF

    grid_x, grid_y, grid_z = data["grid"]
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if info["variant"] != "batch_0":
            bases = _adjust(info["variant"], storage, z, mutation, records)
        x_base, w_base, s_base, b_base, y_base = bases
        for tid_x in range(grid_x):
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    out_row = tid_y * ROWS_PER_GROUP
                    if mutation != "rows":
                        out_row += simd_gid * RESULTS_PER_SIMDGROUP
                    lanes = []
                    for lane in range(SIMD_SIZE):
                        ws = (
                            w_base * 4
                            + out_row * in_vec_size_w
                            + lane * geometry["packsPerThread"] * bytes_per_pack
                        )
                        group = 0 if mutation == "scale-step" else lane // step
                        scales = s_base + out_row * in_vec_size_g + group
                        biases = b_base + out_row * in_vec_size_g + group
                        x = x_base + tid_x * in_vec_size + lane * per_thread
                        result = [0.0] * RESULTS_PER_SIMDGROUP
                        for _ in range(0, in_vec_size, block):
                            values = [load("x", x + i) for i in range(per_thread)]
                            if mutation == "order":
                                values = [
                                    values[start + pack_factor - 1 - i]
                                    for start in range(0, per_thread, pack_factor)
                                    for i in range(pack_factor)
                                ]
                            x_thread, total = _load_vector(dtype, bits, values)
                            for row in range(RESULTS_PER_SIMDGROUP):
                                scale = load("scales", scales + row * in_vec_size_g)
                                bias = load("biases", biases + row * in_vec_size_g)
                                result[row] = _checked(
                                    result[row]
                                    + _qdot(
                                        bits,
                                        byte,
                                        ws + row * in_vec_size_w,
                                        x_thread,
                                        scale,
                                        bias,
                                        total,
                                        mutation,
                                    )
                                )
                            ws += block * bytes_per_pack // pack_factor
                            scales += block // group_size
                            biases += block // group_size
                            x += block
                        lanes.append(result)
                    for row in range(RESULTS_PER_SIMDGROUP):
                        # simd_sum: exact in any order; lane 0 stores T(sum).
                        total = 0.0
                        for result in lanes:
                            total = _checked(total + result[row])
                        index = y_base + tid_x * out_vec_size + out_row + row
                        assert 0 <= index < len(outputs["y"]) - GUARD_COUNT, index
                        stores["y", index] += 1
                        outputs["y"][index] = _to_type(dtype, total)
    return {"outputs": outputs, "stores": stores, "reads": reads, "records": records}


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_SHARD_COUNT"
CI_SHARD_COUNT = 2


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"quantized qmv shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _dispatch_count():
    return sum(len(CASES[INFO[entry]["variant"]]) for entry in ENTRIES)


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        len(_case_data(entry, case)["expected"]["y"])
        for entry in ENTRIES
        for case in CASES[INFO[entry]["variant"]]
    )


def _expected_validation_contract():
    return {
        "model": (
            "qmv_fast_impl lane by lane after adjust_matrix_offsets: "
            "load_vector's scaled inputs and element-type sums, qdot's masked "
            "unpacking, scale * accum + sum * bias, simd_sum and lane 0's "
            "static_cast stores, against direct dequantized dot products"
        ),
        "packing": "least-significant-bit-first code stream per weight row",
        "comparison": "exact words",
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, N / 8, B)",
        "cases": {
            variant: {
                case: {
                    "M": SHAPES[case][0],
                    "N": SHAPES[case][1],
                    "blocks": SHAPES[case][2],
                }
                for case in cases
            }
            for variant, cases in CASES.items()
        },
        "inputs": (
            "x in quarters of [-1, 1], codes with 0 and the top code in every "
            "group, scales +-2^e with e in [-3, 0] and biases -a * scale"
        ),
        "exactness": (
            "every float32 intermediate is a multiple of 2^-5 below 2^19 in "
            "magnitude; only static_cast<T> rounds, to nearest even"
        ),
        "guards": {"count": GUARD_COUNT, "sentinel": SENTINEL},
        "softwareSubgroups": {
            "width": SIMD_SIZE,
            "directxShuffleOutOfRange": "self",
        },
        "openglIndexRangeAssertions": {
            variant: {
                "expressions": list(expressions),
                "minimum": 0,
                "maximum": INDEX_BOUND,
            }
            for variant, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
        },
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": "unchanged quantized.metal, word for word",
    }


def _expected_scope():
    covered = (
        QUANTIZE_ENTRY_COUNT
        + len(ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + GATHER_QMM_ENTRY_COUNT
        + QMV_QUAD_ENTRY_COUNT
    )
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        "quantizeGateEntryCount": QUANTIZE_ENTRY_COUNT,
        "qmvUnalignedGateEntryCount": QMV_UNALIGNED_ENTRY_COUNT,
        "qmvWideGateEntryCount": QMV_WIDE_ENTRY_COUNT,
        "qvmGateEntryCount": QVM_ENTRY_COUNT,
        "qvmSplitKGateEntryCount": QVM_SPLIT_K_ENTRY_COUNT,
        "qmmTGateEntryCount": QMM_T_ENTRY_COUNT,
        "qmmNGateEntryCount": QMM_N_ENTRY_COUNT,
        "qmmTSplitKGateEntryCount": QMM_T_SPLITK_ENTRY_COUNT,
        "gatherQmmGateEntryCount": GATHER_QMM_ENTRY_COUNT,
        "qmvQuadGateEntryCount": QMV_QUAD_ENTRY_COUNT,
        "excludedEntries": {
            "kernels": list(EXCLUDED_KERNELS),
            "count": UNIT_ENTRY_COUNT - covered,
            "reason": "the gather_qmm_rhs matrix-matrix kernels are not gated yet",
        },
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": _dispatch_count(),
        "checkedElementCountByTarget": _checked_element_count(),
        "ciShardCount": CI_SHARD_COUNT,
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def expected_contract(artifacts=None):
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-quantized-qmv-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(QMV_HEADERS),
            "hostSource": {"path": HOST_SOURCE, "sha256": HOST_SHA256},
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        "artifacts": artifacts,
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(),
    }


def test_current_mlx_quantized_qmv_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    artifacts = contract["artifacts"]
    assert set(artifacts) == set(TARGETS)
    for target in TARGETS:
        assert list(artifacts[target]) == list(ENTRIES)
        for record in artifacts[target].values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            assert record["sizeBytes"] > 0


def test_quantized_qmv_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qmv.current-tree.translation.json" in readme
    assert len(ENTRIES) == 162 == len(set(ENTRIES))
    assert Counter(info["variant"] for info in INFO.values()) == {
        "gather": 54,
        "batch_1": 54,
        "batch_0": 54,
    }
    assert ENTRIES[:4] == (
        "affine_gather_qmv_fast_float_gs_128_b_2",
        "affine_qmv_fast_float_gs_128_b_2_batch_1",
        "affine_qmv_fast_float_gs_128_b_2_batch_0",
        "affine_gather_qmv_fast_float16_t_gs_128_b_2",
    )
    assert ENTRIES[-1] == "affine_qmv_fast_bfloat16_t_gs_32_b_8_batch_0"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + GATHER_QMM_ENTRY_COUNT
        + QMV_QUAD_ENTRY_COUNT
        + 108
        == UNIT_ENTRY_COUNT
    )


def test_quantized_qmv_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {81}
    for shard in shards:
        assert Counter(INFO[entry]["variant"] for entry in shard) == {
            "gather": 27,
            "batch_1": 27,
            "batch_0": 27,
        }
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES[::7])
def test_quantized_qmv_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"])
    # qmv_fast_k_alignment: pack_factor<bits, 32> * (bits == 2 ? 1 : 2) * 32.
    alignment = geometry["packFactor"] * (1 if info["bits"] == 2 else 2) * SIMD_SIZE
    assert geometry["blockSize"] == alignment
    assert geometry["packFactor"] * info["bits"] == 8 * geometry["bytesPerPack"]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        # fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0.
        assert n % ROWS_PER_GROUP == 0 and k % geometry["blockSize"] == 0
        assert data["grid"] == (m, n // ROWS_PER_GROUP, batches)
        if info["variant"] != "gather":
            # qmv names the kernel _batch_1 exactly when B > 1.
            assert (batches > 1) == (info["variant"] == "batch_1")
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        # add_strides_and_shapes: x.ndim() - 2 and w.ndim() - 2 batch dims.
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qmv_model_matches_dot_products(entry):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert len(run["stores"]) == written
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word, scale and bias of the used matrices is read, and
        # no guard or unused activation row is.
        layout = data["layout"]
        rows = {
            batch["x"] + row * layout["K"] + index
            for batch in layout["batches"]
            for row in range(layout["M"])
            for index in range(layout["K"])
        }
        assert run["reads"]["x"] == rows
        matrices = {batch["w"] for batch in layout["batches"]}
        n, words, groups = layout["N"], layout["words"], layout["groups"]
        assert run["reads"]["w"] == {
            (matrix * n + col) * words + word
            for matrix in matrices
            for col in range(n)
            for word in range(words)
        }
        for name in ("scales", "biases"):
            assert run["reads"][name] == {
                (matrix * n + col) * groups + group
                for matrix in matrices
                for col in range(n)
                for group in range(groups)
            }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qmv_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
    for entry in ENTRIES:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in CASES[INFO[entry]["variant"]]:
            data = _case_data(entry, case)
            assert data["budget"] < 2**19
            for name, (kind, values) in data["values"].items():
                if kind == dtype:
                    assert all(_to_type(dtype, value) == value for value in values)
            for total, value in zip(data["totals"], data["expected"]["y"]):
                assert _f32(total) == total and math.isfinite(value)
                if value != total:
                    rounded += 1
                    # Halfway cases that round-to-nearest-even decides.
                    neighbour = 2 * total - value
                    ties += _to_type(dtype, neighbour) == neighbour
    if dtype == "float32":
        assert rounded == 0
    else:
        # Only the final static_cast<T> rounds, including halfway cases.
        assert rounded > 0 and ties > 0, (rounded, ties)


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in ENTRIES
        for mutation in MUTATIONS
        if mutation != "batch" or INFO[entry]["variant"] != "batch_0"
    ],
)
def test_quantized_qmv_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qmv_opengl_assertions_name_buffer_elements(entry):
    variant = INFO[entry]["variant"]
    used = set()
    for case in CASES[variant]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _simulate(entry, data)["records"]:
            used.add(expression)
            assert (
                0
                <= value
                < lengths[buffer] - (0 if buffer.endswith("_indices") else GUARD_COUNT)
            ), (case, expression, buffer, value)
            assert value <= INDEX_BOUND
    # Every precondition is reached, and none is missing.
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[variant])


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qmv_configs_scope_target_options(tmp_path, target):
    for variant in VARIANTS:
        entry = next(e for e in ENTRIES if INFO[e]["variant"] == variant)
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        assert options["directx"]["software_subgroup_width"] == SIMD_SIZE
        assert options["directx"]["relative_wave_shuffle_out_of_range"] == "self"
        assert options["opengl"]["software_subgroup_width"] == SIMD_SIZE
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "opengl":
            assert expressions == list(OPENGL_INDEX_RANGE_ASSERTIONS[variant])
        else:
            assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native quantized qmv gate")
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
            "mlx/backend/metal",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # ENTRIES follow quantized.metal's own instantiation macros.
    text = source.decode("utf-8")
    single = re.findall(
        r"instantiate_quantized\((\w+),",
        _macro(text, "instantiate_quantized_all_single"),
    )
    assert single.index("affine_gather_qmv_fast") == 2
    batched = re.findall(
        r"instantiate_quantized_batched_wrap\((\w+),",
        _macro(text, "instantiate_quantized_all_batched"),
    )
    assert batched[0] == "affine_qmv_fast"
    wrap = _macro(text, "instantiate_quantized_batched_wrap")
    assert re.findall(r"bits, (\d)\)", wrap) == ["1", "0"]
    funcs = _macro(text, "instantiate_quantized_funcs")
    assert funcs.index("all_single") < funcs.index("all_batched")
    types = _macro(text, "instantiate_quantized_types")
    assert re.findall(r"instantiate_quantized_funcs\((\w+),", types) == [
        itype for itype, _ in DTYPES
    ]
    groups = _macro(text, "instantiate_quantized_groups")
    assert re.findall(r"instantiate_quantized_types\((\d+),", groups) == [
        str(size) for size in GROUP_SIZES
    ]
    widths = _macro(text, "instantiate_quantized_all")
    assert re.findall(r"instantiate_quantized_groups\((\d+)\)", widths) == [
        str(bits) for bits in BITS
    ]
    header = (root / "mlx/backend/metal/kernels/quantized.h").read_text(
        encoding="utf-8"
    )
    for kernel, parameters in (
        ("affine_qmv_fast", QMV_PARAMETERS),
        ("affine_gather_qmv_fast", GATHER_PARAMETERS),
    ):
        match = re.search(
            rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", header, re.DOTALL
        )
        assert match, kernel
        found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
        assert tuple((name, int(index)) for name, index in found) == parameters
    # The dispatch the cases follow.
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "int bk = 32;",
        "MTL::Size group_dims(bk, 2, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        "bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;",
        'B > 1 ? "_batch_1" : "_batch_0"',
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
        "add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);",
    ):
        assert required in host, required
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qmv-metal-reference")
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


def _translate(root, work, target, config, record, workgroup=WORKGROUP):
    """Translate one entry's ``config`` and check it against its pinned artifact."""
    translation = translate_project(
        config,
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
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == list(workgroup)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    return _prepare_native_package(translation, work)


def _request(
    descriptor, package, target, variant, data, parameters=None, workgroup=WORKGROUP
):
    """Bind every reflected binding by source parameter and build one dispatch.

    ``parameters`` overrides the variant's ``[[buffer(i)]]`` parameters and
    ``workgroup`` the threadgroup size.
    """
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    values = {
        name: _runtime_value(target, dtype, storage)
        for name, (dtype, storage) in data["values"].items()
    }
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        assert "executionInput" not in provenance, binding
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
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "y":
            assert binding["access"] == "read_write", binding
            output = binding["name"]
        else:
            observed.append(binding["name"])
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    # The translation reflects every kernel parameter, batched or not.
    parameters = parameters or _parameters(variant)
    assert output is not None and bound == {name for name, _ in parameters}
    expected = _runtime_value(target, data["kinds"]["y"], data["expected"]["y"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: expected},
        {"workgroupCount": list(data["grid"]), "workgroupSize": list(workgroup)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, len(observed)


def _upstream(
    work,
    reference,
    entry,
    variant,
    label,
    data,
    parameters=None,
    workgroup=WORKGROUP,
    bound=UNBATCHED_BOUND,
):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage.

    ``parameters`` overrides the variant's ``[[buffer(i)]]`` parameters,
    ``workgroup`` the threadgroup size and ``bound`` the buffers the host
    binds when ``B <= 1``.
    """
    runner, library = reference
    parameters = parameters or _parameters(variant)
    if variant == "batch_0":
        # The host skips add_strides_and_shapes when B <= 1: qmv binds only
        # buffers 0-6.
        parameters = parameters[:bound]
    files, payloads = [], {}
    for name, index in parameters:
        payload = _metal_bytes(*data["values"][name])
        payloads[name] = payload
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": list(data["grid"]),
                "workgroupSize": list(workgroup),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    words = None
    for name, index in parameters:
        raw = (readback / f"buffer-{index}.bin").read_bytes()
        if name == "y":
            words = _words(data["kinds"]["y"], raw)
        else:
            assert raw == payloads[name], name
    return device, words


def _assert_identical(label, actual, expected):
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_quantized_qmv_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-qmv-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = _translate(root, work, target, config, record)
            executor = _executor(target)
            evidence = []
            variant = INFO[entry]["variant"]
            try:
                for case in CASES[variant]:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, variant, data
                    )
                    words = _execute(
                        executor, work, case, request, {"y": output}, observed
                    )["y"]["values"]
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work, metal_reference, entry, variant, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        _assert_identical(case, words, upstream_words)
                        upstream = {"device": device, "identicalToTranslation": True}
                    expected = _runtime_value(
                        target, data["kinds"]["y"], data["expected"]["y"]
                    )["values"]
                    _assert_identical(case, words, expected)
                    evidence.append(
                        {
                            "case": case,
                            "grid": list(data["grid"]),
                            "M": data["layout"]["M"],
                            "N": data["layout"]["N"],
                            "K": data["layout"]["K"],
                            "checkedElementCount": len(data["expected"]["y"]),
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


def test_ci_requires_native_quantized_qmv_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized matrix-vector native parity",
        family="quantized-qmv",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv.py"
    )
