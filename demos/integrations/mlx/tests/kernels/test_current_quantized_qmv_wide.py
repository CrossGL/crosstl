"""Natively prove quantized.metal's 432 affine qmv_wide matrix-vector kernels.

``quantized.metal`` instantiates ``affine_qmv_wide`` with and without batching
for float, float16_t and bfloat16_t, group sizes 128, 64 and 32, 2, 3, 4, 5, 6
and 8 bits and tiles of 2, 3, 4 and 5 input vectors, each output row reduced
by eight lanes: 432 of its 2,052 entries. Each is translated and pinned per
target, then runs ``quantized.cpp``'s ``qmv_wide`` dispatch: threadgroups of
``(32, 2, 1)`` threads, two simdgroups of four output rows each, over a grid
of ``(ceil(M / vecs_per_tg), ceil(N / 8), B)``, where ``vecs_per_tg`` is the
tile ``qmv_wide`` picks for ``M``.

Unbatched entries run one tile against more groups per row than lanes per
row, and a tiled matrix whose last row group, and for tiles of 4 and 5 also
last vector tile, is partial. Batched entries run stacked batches, a
broadcast two-dimensional batch with transposed activations and broadcast
weights, and strided activations sharing one weight matrix, reaching both
branches of ``adjust_matrix_offsets``. Inputs are quarters in ``[-1, 1]``,
power-of-two scales and biases ``-a * scale``: every dequantized weight,
product, partial sum, shuffle and accumulator is a multiple of ``2^-5`` below
``2^19``, so float32 holds each exactly in any order and only the final
``static_cast<T>`` rounds, to nearest even. A model replays each lane's
loads, ``dequantize``'s eight-value sub-chunks, the shuffle-down ladder and
the partial-tile guards and must reach a direct dequantized dot product.
Each target must produce the model's words and leave every guard untouched;
on Metal the translation must also match the unchanged upstream kernel word
for word.
"""

import functools
import hashlib
import json
import math
import os
import re
import shutil
import struct
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
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    declared_buffers,
    parse_pin,
    require_pinned_checkout,
    row_major,
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
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    BITS,
    DTYPES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    ENTRIES as QMV_FAST_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    EXCLUDED_KERNELS,
    GATHER_QMM_ENTRY_COUNT,
    GROUP_SIZES,
    GUARD_COUNT,
    INDEX_BOUND,
    NUM_SIMDGROUPS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    OPENGL_INDEX_RANGE_ASSERTIONS as QMV_FAST_INDEX_RANGE_ASSERTIONS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    POISON,
    QMM_N_ENTRY_COUNT,
    QMM_T_ENTRY_COUNT,
    QMM_T_SPLITK_ENTRY_COUNT,
    QMV_HEADERS,
    QMV_QUAD_ENTRY_COUNT,
    QMV_UNALIGNED_ENTRY_COUNT,
    QMV_WIDE_ENTRY_COUNT,
    QUANTIZE_ENTRY_COUNT,
    QVM_ENTRY_COUNT,
    QVM_SPLIT_K_ENTRY_COUNT,
    ROWS_PER_GROUP,
    SENTINEL,
    SIMD_SIZE,
    WORD_POISON,
    WORKGROUP,
    _checked,
    _elem_to_loc,
    _exact_dyadic,
    _f32,
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
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QMV_WIDE"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_qmv_wide.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "affine_qmv_wide"
# instantiate_quantized_all_wide's tiles; affine entries use k_lanes 8.
VECS_PER_TG = (2, 3, 4, 5)
K_LANES = 8
# instantiate_quantized_wide_wrap instantiates batched 0, then 1.
VARIANTS = ("batch_0", "batch_1")

# qmv_wide_impl's constants: results_per_simdgroup rows per simdgroup, each
# group decoded in sub-chunks of eight values.
RESULTS_PER_SIMDGROUP = SIMD_SIZE // K_LANES
SUB = 8
assert NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP == ROWS_PER_GROUP

# affine_qmv_wide's buffer parameters, without [[buffer(i)]] attributes, in
# declaration order as quantized.cpp binds them.
PARAMETERS = (
    ("w", 0),
    ("scales", 1),
    ("biases", 2),
    ("x", 3),
    ("y", 4),
    ("in_vec_size", 5),
    ("out_vec_size", 6),
    ("M", 7),
    ("x_batch_ndims", 8),
    ("x_shape", 9),
    ("x_strides", 10),
    ("w_batch_ndims", 11),
    ("w_shape", 12),
    ("w_strides", 13),
    ("s_strides", 14),
    ("b_strides", 15),
)
# qmv_wide binds only w, scales, biases, x, y, K, N and M when unbatched:
# add_strides_and_shapes skips the batch parameters.
UNBATCHED_BOUND = 8
SCALAR_TYPES = {
    "in_vec_size": "int32",
    "out_vec_size": "int32",
    "M": "int32",
    "x_batch_ndims": "int32",
    "w_batch_ndims": "int32",
}
ARRAY_TYPES = {
    "x_shape": "int32",
    "w_shape": "int32",
    "x_strides": "int64",
    "w_strides": "int64",
    "s_strides": "int64",
    "b_strides": "int64",
}

CASES = {
    "batch_0": ("tile", "tiles"),
    "batch_1": ("stacked", "broadcast", "shared"),
}
# Expressions OpenGL may evaluate in 32 bits: adjust_matrix_offsets' batch
# offsets, as for qmv_fast. The model shows each names an element inside its
# buffer.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "batch_0": (),
    "batch_1": QMV_FAST_INDEX_RANGE_ASSERTIONS["batch_1"],
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for vecs in VECS_PER_TG:
                    for variant in VARIANTS:
                        entry = (
                            f"{KERNEL}_{itype}_gs_{group_size}_b_{bits}"
                            f"_nv_{vecs}_kl_{K_LANES}_{variant}"
                        )
                        yield entry, {
                            "variant": variant,
                            "itype": itype,
                            "dtype": dtype,
                            "groupSize": group_size,
                            "bits": bits,
                            "vecsPerTg": vecs,
                        }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    """The translation of one entry, as the gate and contract run it."""
    # simd_shuffle_down: software subgroups give DirectX and OpenGL MLX's 32
    # lanes, and DirectX keeps a lane's own value past the simdgroup's end.
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    assertions = (
        OPENGL_INDEX_RANGE_ASSERTIONS[INFO[entry]["variant"]]
        if target == "opengl"
        else ()
    )
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: WORKGROUP}},
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
# Dispatch and cases
# ---------------------------------------------------------------------------

# get_qmv_batch_limit on the gen-15 and gen-16 devices use_qmv_wide admits
# for affine weights, with K and N at most 2048: M below it reaches
# dispatch_qmv, and qmv_wide from M = 2.
VECTOR_LIMIT = 13
# dispatch_qmv routes these to qmv_quad when bits is a power of two.
QUAD_K = (64, 128)


def _vecs_per_tg(m):
    """qmv_wide's tile: the fewest tiles of at most five, then the smallest."""
    n_tiles = (m + 4) // 5
    return (m + n_tiles - 1) // n_tiles


# The M below the vector limit that qmv_wide gives each tile.
TILE_ROWS = {
    vecs: tuple(m for m in range(2, VECTOR_LIMIT) if _vecs_per_tg(m) == vecs)
    for vecs in VECS_PER_TG
}


def _tiled_rows(vecs):
    """The most rows of ``TILE_ROWS``, preferring a partial last tile."""
    rows = TILE_ROWS[vecs]
    return max([m for m in rows if m % vecs] or rows)


# (N, groups per row): N = 12 leaves the last row group partial, and nine
# groups give lane 0 of each row two of them.
SHAPES = {
    "tile": (8, 9),
    "tiles": (12, 3),
    "stacked": (8, 9),
    "broadcast": (12, 3),
    "shared": (8, 3),
}
TILED_CASES = ("tiles", "shared")


def _rows(case, vecs):
    return _tiled_rows(vecs) if case in TILED_CASES else vecs


def _layout(entry, case):
    """quantized.cpp's shapes and strides for one case.

    ``batches`` gives, per ``tid.z``, the x element offset and the physical
    weight matrix the arrays are built for; the kernel model derives the same
    from the bound parameters.
    """
    info = INFO[entry]
    variant, bits, group_size = info["variant"], info["bits"], info["groupSize"]
    vecs = info["vecsPerTg"]
    assert case in CASES[variant], (entry, case)
    m = _rows(case, vecs)
    n, groups = SHAPES[case]
    k = groups * group_size
    words = k * bits // 32
    matrix = m * k
    layout = {"M": m, "N": n, "K": k, "groups": groups, "words": words}
    if case in CASES["batch_0"]:
        # add_strides_and_shapes would bind these unbatched shapes; it is
        # skipped when unbatched and the kernel never reads them.
        params = {
            "x_batch_ndims": 0,
            "x_shape": [m, k],
            "x_strides": [k, 1],
            "w_batch_ndims": 0,
            "w_shape": [n, words],
            "w_strides": [words, 1],
            "s_strides": [groups, 1],
            "b_strides": [groups, 1],
        }
        batches = [{"x": 0, "w": 0}]
        x_rows, matrices = m, 1
    elif case == "stacked":
        params = {
            "x_batch_ndims": 1,
            "x_shape": [2, m, k],
            "x_strides": [matrix, k, 1],
            "w_batch_ndims": 1,
            "w_shape": [2, n, words],
            "w_strides": [n * words, words, 1],
            "s_strides": [n * groups, groups, 1],
            "b_strides": [n * groups, groups, 1],
        }
        batches = [{"x": z * matrix, "w": z} for z in range(2)]
        x_rows, matrices = 2 * m, 2
    elif case == "shared":
        # Every other matrix of a strided activation view against one weight
        # matrix: x is not row contiguous, so B > 1 with w_batch_ndims 0.
        params = {
            "x_batch_ndims": 1,
            "x_shape": [3, m, k],
            "x_strides": [2 * matrix, k, 1],
            "w_batch_ndims": 0,
            "w_shape": [n, words],
            "w_strides": [words, 1],
            "s_strides": [groups, 1],
            "b_strides": [groups, 1],
        }
        batches = [{"x": 2 * z * matrix, "w": 0} for z in range(3)]
        x_rows, matrices = 6 * m, 1
    else:
        # A (2, 2) batch: activations with transposed batch strides and
        # weights broadcast along the second batch dimension.
        params = {
            "x_batch_ndims": 2,
            "x_shape": [2, 2, m, k],
            "x_strides": [matrix, 2 * matrix, k, 1],
            "w_batch_ndims": 2,
            "w_shape": [2, 2, n, words],
            "w_strides": [n * words, 0, words, 1],
            "s_strides": [n * groups, 0, groups, 1],
            "b_strides": [n * groups, 0, groups, 1],
        }
        batches = [
            {"x": (z % 2) * 2 * matrix + (z // 2) * matrix, "w": z // 2}
            for z in range(4)
        ]
        x_rows, matrices = 4 * m, 2
    for z, batch in enumerate(batches):
        batch["y"] = z * m * n
    layout.update(
        params=params,
        batches=batches,
        xRows=x_rows,
        matrices=matrices,
        grid=(-(-m // vecs), -(-n // ROWS_PER_GROUP), len(batches)),
    )
    return layout


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    info = INFO[entry]
    dtype, bits, group_size = info["dtype"], info["bits"], info["groupSize"]
    n_bins = (1 << bits) - 1
    layout = _layout(entry, case)
    m, n, k = layout["M"], layout["N"], layout["K"]
    groups = layout["groups"]
    draws = _draws("quantized-qmv-wide", entry, case)
    used_rows = {
        batch["x"] + row * k for batch in layout["batches"] for row in range(m)
    }
    x, inputs = [POISON] * (layout["xRows"] * k), []
    for start in sorted(used_rows):
        x[start : start + k] = [_signed(draws, 4) / 4 for _ in range(k)]
        inputs += x[start : start + k]
    codes, scales, biases = [], [], []
    for _ in range(layout["matrices"] * n):
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
    assert len(w) == layout["matrices"] * n * layout["words"]
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
        "M": ("int32", [m]),
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
    assert set(values) == {name for name, _ in PARAMETERS}, entry
    data = {
        "layout": layout,
        "codes": codes,
        "values": values,
        "initial": {"y": initial},
        "kinds": {"y": dtype},
        "grid": layout["grid"],
    }
    data["expected"], data["totals"], data["budget"] = _direct(entry, data)
    return data


def _direct(entry, data):
    """Each output as a dequantized dot product, its float32 total and budget.

    Every dequantized weight is ``(q - a) * scale``, at most 255 in
    magnitude, and the budget bounds every partial sum of products, whatever
    its order: below 2^19, every multiple of 2^-5 it can reach is exact in
    float32.
    """
    info = INFO[entry]
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
                line = batch["w"] * n + col
                codes = data["codes"][line]
                total, bound = 0.0, 0.0
                for index, value in enumerate(vector):
                    group = line * groups + index // group_size
                    weight = scales[group] * codes[index] + biases[group]
                    assert abs(weight) <= 255 and _exact_dyadic(weight, 8)
                    total += weight * value
                    bound += abs(weight * value)
                assert _exact_dyadic(total) and bound < 2**19, (entry, total, bound)
                budget = max(budget, bound)
                index = batch["y"] + row * n + col
                totals[index] = total
                expected[index] = _to_type(dtype, total)
    return {"y": expected}, totals, budget


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _decode(bits, w):
    """dequantize's codes from one run's bytes (3, 5 and 6 bits)."""
    if bits == 3:
        return (
            w[0] & 0x7,
            (w[0] & 0x38) >> 3,
            ((w[0] & 0xC0) >> 6) + ((w[1] & 0x1) << 2),
            (w[1] & 0xE) >> 1,
            (w[1] & 0x70) >> 4,
            ((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1),
            (w[2] & 0x1C) >> 2,
            (w[2] & 0xE0) >> 5,
        )
    if bits == 5:
        return (
            w[0] & 0x1F,
            ((w[0] & 0xE0) >> 5) + ((w[1] & 0x3) << 3),
            (w[1] & 0x7C) >> 2,
            ((w[1] & 0x80) >> 7) + ((w[2] & 0xF) << 1),
            ((w[2] & 0xF0) >> 4) + ((w[3] & 0x1) << 4),
            (w[3] & 0x3E) >> 1,
            ((w[3] & 0xC0) >> 6) + ((w[4] & 0x7) << 2),
            (w[4] & 0xF8) >> 3,
        )
    return (
        w[0] & 0x3F,
        ((w[0] >> 6) & 0x03) + ((w[1] & 0x0F) << 2),
        ((w[1] >> 4) & 0x0F) + ((w[2] & 0x03) << 4),
        (w[2] >> 2) & 0x3F,
    )


def _dequantize(bits, byte, base, scale, bias, mutation):
    """dequantize<float, 8, bits> into w_dq, expression for expression."""
    s = scale
    b = 0.0 if mutation == "bias" else bias
    # U w_dq[sub] starts uninitialized; only a mutation leaves values unset.
    out = [0.0] * SUB
    if bits == 2:
        sc = (s, s / 4.0, s / 16.0, s / 64.0)
        for i in range(SUB // 4):
            w = byte(base + i)
            out[4 * i] = sc[0] * (w & 0x03) + b
            out[4 * i + 1] = sc[1] * (w & 0x0C) + b
            out[4 * i + 2] = sc[2] * (w & 0x30) + b
            out[4 * i + 3] = sc[3] * (w & 0xC0) + b
    elif bits == 4:
        sc = (s, s / 16.0)
        for i in range(SUB // 2):
            w = byte(base + i)
            out[2 * i] = sc[0] * (w & 0x0F) + b
            out[2 * i + 1] = sc[1] * (w & 0xF0) + b
    elif bits == 8:
        for i in range(SUB):
            out[i] = s * byte(base + i) + b
    else:
        # 3, 5 and 6 bits advance w_local by run * i and w by width * i each
        # iteration, cumulatively, as the kernel does.
        run, width = {3: (8, 3), 5: (8, 5), 6: (4, 3)}[bits]
        local = offset = 0
        for i in range(SUB // run):
            if mutation != "rebase":
                local += run * i
                offset += width * i
            codes = _decode(bits, [byte(base + offset + j) for j in range(width)])
            for j, code in enumerate(codes):
                out[local + j] = code * s + b
    if mutation == "order":
        out.reverse()
    # static_cast<U> with U = float: every value is exact.
    return [_checked(value) for value in out]


MUTATIONS = {
    "bias": "drops dequantize's bias",
    "order": "pairs each sub-chunk's inputs with its codes in reverse",
    "rebase": "drops dequantize's w_local and w steps",
    "stride": "gives each lane only its first group",
    "vectors": "streams the tile's first vector for every vector",
    "rows": "ignores each lane's row within its simdgroup",
    "ladder": "sums all 32 lanes instead of each row's 8",
    "batch": "ignores the weight batch offsets",
}


def _mutation_applies(entry, mutation):
    info = INFO[entry]
    if mutation == "rebase":
        # Only six bits takes a second, rebased run per sub-chunk.
        return info["bits"] == 6
    if mutation == "batch":
        return info["variant"] == "batch_1"
    return True


def _adjust(storage, z, mutation, records):
    """adjust_matrix_offsets for ``tid.z``: x, w, scales, biases and y bases."""

    def scalar(name):
        return storage[name][0]

    if scalar("x_batch_ndims") == 1:
        x_base = z * storage["x_strides"][0]
        records.append(("x_idx * x_strides[0]", "x", x_base))
    else:
        (x_base,) = _elem_to_loc(
            z, storage["x_shape"], (storage["x_strides"],), scalar("x_batch_ndims")
        )
    strides = (storage["w_strides"], storage["s_strides"], storage["b_strides"])
    if scalar("w_batch_ndims") == 1:
        bases = [z * values[0] for values in strides]
        records += [
            (f"w_idx * {name}[0]", buffer, value)
            for name, buffer, value in zip(
                ("w_strides", "s_strides", "b_strides"),
                ("w", "scales", "biases"),
                bases,
            )
        ]
    else:
        bases = _elem_to_loc(z, storage["w_shape"], strides, scalar("w_batch_ndims"))
        records += [
            (f"idx.{axis}", buffer, value)
            for axis, buffer, value in zip("xyz", ("w", "scales", "biases"), bases)
        ]
    if mutation == "batch":
        bases = [0, 0, 0]
    # qmv_wide passes out_vec_size * M as the output stride.
    return (x_base, *bases, z * scalar("out_vec_size") * scalar("M"))


def _ladder(partials, mutation):
    """k_lanes 8's simd_shuffle_down ladder over 4, 2 and 1 lanes.

    A lane whose source is past the simdgroup keeps its own value.
    """
    if mutation == "ladder":
        total = 0.0
        for value in partials:
            total = _checked(total + value)
        return [total] * SIMD_SIZE
    values = list(partials)
    for delta in (4, 2, 1):
        values = [
            _checked(
                values[lane]
                + values[lane + delta if lane + delta < SIMD_SIZE else lane]
            )
            for lane in range(SIMD_SIZE)
        ]
    return values


def _simulate(entry, case, mutation=None):
    """The kernel lane by lane; returns outputs, stores, reads and offsets."""
    if mutation is None:
        return _baseline(entry, case)
    return _model(entry, case, mutation)


@functools.lru_cache(maxsize=None)
def _baseline(entry, case):
    return _model(entry, case, None)


def _model(entry, case, mutation):
    info = INFO[entry]
    dtype, bits, group_size = info["dtype"], info["bits"], info["groupSize"]
    vecs = info["vecsPerTg"]
    data = _case_data(entry, case)
    storage = {name: values for name, (_, values) in data["values"].items()}
    in_vec_size, out_vec_size = storage["in_vec_size"][0], storage["out_vec_size"][0]
    rows = storage["M"][0]
    in_vec_size_w = in_vec_size * bits // 8
    in_vec_size_g = in_vec_size // group_size
    reads = {name: set() for name in storage}
    outputs = {"y": list(data["initial"]["y"])}
    stores, records = Counter(), []
    x_values = storage["x"]

    def load(name, index):
        assert 0 <= index < len(storage[name]) - GUARD_COUNT, (name, index)
        reads[name].add(index)
        return storage[name][index]

    def byte(index):
        word = load("w", index // 4)
        return (word >> (8 * (index % 4))) & 0xFF

    def run(start):
        """xc[0 .. 7]: one sub-chunk of a vector, as float."""
        assert 0 <= start and start + SUB <= len(x_values) - GUARD_COUNT, start
        reads["x"].update(range(start, start + SUB))
        return x_values[start : start + SUB]

    grid_x, grid_y, grid_z = data["grid"]
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if info["variant"] == "batch_1":
            bases = _adjust(storage, z, mutation, records)
        x_base, w_base, s_base, b_base, y_base = bases
        for tid_x in range(grid_x):
            vec0 = tid_x * vecs
            # xv[v] = x + min(vec0 + v, M - 1) * in_vec_size.
            xv = [
                x_base
                + min(vec0 + (0 if mutation == "vectors" else v), rows - 1)
                * in_vec_size
                for v in range(vecs)
            ]
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    lanes = []
                    for lane in range(SIMD_SIZE):
                        k_lane = lane % K_LANES
                        sg_row = 0 if mutation == "rows" else lane // K_LANES
                        out_row = (
                            tid_y * ROWS_PER_GROUP
                            + RESULTS_PER_SIMDGROUP * simd_gid
                            + sg_row
                        )
                        row = min(out_row, out_vec_size - 1)
                        wrow = w_base * 4 + row * in_vec_size_w
                        srow = s_base + row * in_vec_size_g
                        brow = b_base + row * in_vec_size_g
                        result = [0.0] * vecs
                        groups = range(k_lane, in_vec_size_g, K_LANES)
                        if mutation == "stride":
                            groups = groups[:1]
                        for g in groups:
                            scale = load("scales", srow + g)
                            bias = load("biases", brow + g)
                            for sc in range(group_size // SUB):
                                k0 = g * group_size + sc * SUB
                                w_dq = _dequantize(
                                    bits,
                                    byte,
                                    wrow + k0 * bits // 8,
                                    scale,
                                    bias,
                                    mutation,
                                )
                                for v in range(vecs):
                                    acc = 0.0
                                    for value, weight in zip(run(xv[v] + k0), w_dq):
                                        acc += value * weight
                                    result[v] = _checked(result[v] + _checked(acc))
                        lanes.append((out_row, result))
                    reduced = [
                        _ladder([result[v] for _, result in lanes], mutation)
                        for v in range(vecs)
                    ]
                    for lane, (out_row, _) in enumerate(lanes):
                        if lane % K_LANES or out_row >= out_vec_size:
                            continue
                        for v in range(vecs):
                            if vec0 + v >= rows:
                                continue
                            index = y_base + (vec0 + v) * out_vec_size + out_row
                            assert 0 <= index < len(outputs["y"]), index
                            stores["y", index] += 1
                            outputs["y"][index] = _to_type(dtype, reduced[v][lane])
    return {"outputs": outputs, "stores": stores, "reads": reads, "records": records}


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_WIDE_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_WIDE_SHARD_COUNT"
# An odd count interleaves both variants and every tile into each shard.
CI_SHARD_COUNT = 9


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"quantized qmv_wide shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


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
            "qmv_wide_impl lane by lane after adjust_matrix_offsets: "
            "dequantize's eight-value sub-chunks, products with each streamed "
            "vector, the simd_shuffle_down ladder over each row's eight lanes "
            "and the partial-tile guards of k_lane 0's static_cast stores, "
            "against direct dequantized dot products"
        ),
        "packing": "least-significant-bit-first code stream per weight row",
        "comparison": "exact words",
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(M / vecs_per_tg), ceil(N / 8), B)",
        "kLanes": K_LANES,
        "cases": {
            variant: {
                case: {
                    "M": {str(vecs): _rows(case, vecs) for vecs in VECS_PER_TG},
                    "N": SHAPES[case][0],
                    "groups": SHAPES[case][1],
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
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + len(ENTRIES)
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
        "qmvFastGateEntryCount": len(QMV_FAST_ENTRIES),
        "qmvUnalignedGateEntryCount": QMV_UNALIGNED_ENTRY_COUNT,
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
        "kind": "crosstl-mlx-current-tree-quantized-qmv-wide-contract",
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
        "artifacts": artifacts,
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(),
    }


def test_current_mlx_quantized_qmv_wide_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qmv_wide_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qmv_wide.current-tree.translation.json" in readme
    assert len(ENTRIES) == 432 == len(set(ENTRIES)) == QMV_WIDE_ENTRY_COUNT
    assert Counter(info["variant"] for info in INFO.values()) == {
        "batch_0": 216,
        "batch_1": 216,
    }
    assert Counter(info["vecsPerTg"] for info in INFO.values()) == {
        vecs: 108 for vecs in VECS_PER_TG
    }
    assert ENTRIES[:3] == (
        "affine_qmv_wide_float_gs_128_b_2_nv_2_kl_8_batch_0",
        "affine_qmv_wide_float_gs_128_b_2_nv_2_kl_8_batch_1",
        "affine_qmv_wide_float_gs_128_b_2_nv_3_kl_8_batch_0",
    )
    assert ENTRIES[-1] == "affine_qmv_wide_bfloat16_t_gs_32_b_8_nv_5_kl_8_batch_1"
    assert KERNEL not in EXCLUDED_KERNELS
    assert not set(ENTRIES) & set(QMV_FAST_ENTRIES)
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + len(ENTRIES)
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


def test_quantized_qmv_wide_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {48}
    for shard in shards:
        assert Counter(INFO[entry]["variant"] for entry in shard) == {
            "batch_0": 24,
            "batch_1": 24,
        }
        assert Counter(INFO[entry]["vecsPerTg"] for entry in shard) == {
            vecs: 12 for vecs in VECS_PER_TG
        }
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_quantized_qmv_wide_tiles_follow_qmv_wide():
    assert TILE_ROWS == {2: (2,), 3: (3, 6), 4: (4, 7, 8, 11, 12), 5: (5, 9, 10)}
    assert {vecs: _tiled_rows(vecs) for vecs in VECS_PER_TG} == {
        2: 2,
        3: 6,
        4: 11,
        5: 9,
    }
    # Tiles 4 and 5 also end their last vector tile early, and two cases end
    # their last row group early.
    for vecs in (4, 5):
        assert _tiled_rows(vecs) % vecs
    assert all(SHAPES[case][0] % ROWS_PER_GROUP for case in ("tiles", "broadcast"))


@pytest.mark.parametrize("entry", ENTRIES[::5])
def test_quantized_qmv_wide_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    vecs, bits = info["vecsPerTg"], info["bits"]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        # QuantizedMatmul::eval_gpu takes M below the vector limit to
        # dispatch_qmv; K and N stay within get_qmv_batch_limit's first band.
        assert 2 <= m < VECTOR_LIMIT and k <= 2048 and n <= 2048
        # dispatch_qmv picks qmv_quad first for K of 64 or 128.
        assert k not in QUAD_K or bits & (bits - 1)
        # qmv_wide: vecs_per_tg from M, rows_per_tg 8 and its grid.
        assert _vecs_per_tg(m) == vecs
        assert data["grid"] == (-(-m // vecs), -(-n // ROWS_PER_GROUP), batches)
        # qmv_wide names the kernel _batch_1 exactly when B > 1.
        assert (batches > 1) == (info["variant"] == "batch_1")
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        assert values["M"][1] == [m]
        params = layout["params"]
        # add_strides_and_shapes: x.ndim() - 2 and w.ndim() - 2 batch dims.
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]
        # eval_gpu folds x's batch into M when w is a matrix and x is row
        # contiguous; every batched case is neither.
        contiguous = params["x_strides"] == row_major(params["x_shape"])
        if info["variant"] == "batch_0":
            assert params["w_batch_ndims"] == 0 and contiguous
        else:
            assert params["w_batch_ndims"] > 0 or not contiguous


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qmv_wide_model_matches_dot_products(entry):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, case)
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
def test_quantized_qmv_wide_inputs_keep_float32_exact(dtype):
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


# Every bit width, tile, variant and element type, at the smallest group.
MUTATION_ENTRIES = tuple(e for e in ENTRIES if INFO[e]["groupSize"] == 32)


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in MUTATION_ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    ],
)
def test_quantized_qmv_wide_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        changed |= (
            _simulate(entry, case, mutation)["outputs"]
            != _case_data(entry, case)["expected"]
        )
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize(
    "entry", [entry for entry in ENTRIES if INFO[entry]["variant"] == "batch_1"]
)
def test_quantized_qmv_wide_opengl_assertions_name_buffer_elements(entry):
    variant = INFO[entry]["variant"]
    used = set()
    for case in CASES[variant]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _simulate(entry, case)["records"]:
            used.add(expression)
            assert 0 <= value < lengths[buffer] - GUARD_COUNT, (
                case,
                expression,
                buffer,
                value,
            )
            assert value <= INDEX_BOUND
    # Every precondition is reached, and none is missing.
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[variant])


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qmv_wide_configs_scope_target_options(tmp_path, target):
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
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "quantized qmv_wide", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # ENTRIES follow quantized.metal's own instantiation macros.
    text = source.decode("utf-8")
    wide = _macro(text, "instantiate_quantized_all_wide")
    assert re.findall(
        r"instantiate_quantized_wide_wrap\((\w+), type, group_size, bits, (\d), (\d)\)",
        wide,
    ) == [(KERNEL, str(vecs), str(K_LANES)) for vecs in VECS_PER_TG]
    wrap = _macro(text, "instantiate_quantized_wide_wrap")
    assert re.findall(r"k_lanes, (\d)\)", wrap) == ["0", "1"]
    name = _macro(text, "instantiate_quantized_wide")
    assert (
        '#name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg '
        '"_kl_" #k_lanes "_batch_" #batched' in name
    )
    funcs = _macro(text, "instantiate_quantized_funcs")
    assert "instantiate_quantized_all_wide(type, group_size, bits)" in funcs
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
    assert declared_buffers(header, KERNEL) == PARAMETERS
    for required in (
        "constexpr int sub = 8;",
        "const int row = min(out_row, out_vec_size - 1);",
        "xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;",
        "for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {",
        "dequantize<U, sub, bits>(wc, scale, bias, w_dq);",
        "if (k_lane == 0 && out_row < out_vec_size) {",
        "out_vec_size * M,",
    ):
        assert required in header, required
    # The dispatch the cases follow.
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) {",
        "if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) {",
        'return mode != "affine" || d.get_architecture_gen() >= 15;',
        "int n_tiles = (M + 4) / 5; // ceil(M / 5); tile size caps at 5",
        "int vecs_per_tg = (M + n_tiles - 1) / n_tiles;",
        'int k_lanes = mode == "affine" ? 8 : 16;',
        "int rows_per_tg = (32 / k_lanes) * num_simdgroups;",
        "MTL::Size group_dims(32, num_simdgroups, 1);",
        "(M + vecs_per_tg - 1) / vecs_per_tg,",
        "(N + rows_per_tg - 1) / rows_per_tg,",
        'batched ? "_batch_1" : "_batch_0"',
        "compute_encoder.set_bytes(M, c++);",
        "add_strides_and_shapes(compute_encoder, !batched, x, w, scales, biases, c);",
        "bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;",
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
    ):
        assert required in host, required
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qmv-wide-metal-reference")
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


def _translate(root, work, target, entry, record):
    """Translate one entry and check it against its pinned artifact."""
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
    assert execution["workgroupSize"] == list(WORKGROUP)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    return _prepare_native_package(translation, work)


def _request(descriptor, package, target, data):
    """Bind every reflected binding by source parameter and build one dispatch."""
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
    assert output is not None and bound == {name for name, _ in PARAMETERS}
    expected = _runtime_value(target, data["kinds"]["y"], data["expected"]["y"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: expected},
        {"workgroupCount": list(data["grid"]), "workgroupSize": list(WORKGROUP)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, len(observed)


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw buffers in declaration order."""
    runner, library = reference
    parameters = PARAMETERS
    if INFO[entry]["variant"] == "batch_0":
        # qmv_wide binds only buffers 0-7 when unbatched.
        parameters = parameters[:UNBATCHED_BOUND]
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
                "workgroupSize": list(WORKGROUP),
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


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_quantized_qmv_wide_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-qmv-wide-", dir=root) as path:
        work = Path(path)
        try:
            descriptor, package = _translate(root, work, target, entry, record)
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES[INFO[entry]["variant"]]:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, data
                    )
                    words = _execute(
                        executor, work, case, request, {"y": output}, observed
                    )["y"]["values"]
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work, metal_reference, entry, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        assert_identical(case, words, upstream_words)
                        upstream = {"device": device, "identicalToTranslation": True}
                    expected = _runtime_value(
                        target, data["kinds"]["y"], data["expected"]["y"]
                    )["values"]
                    assert_identical(case, words, expected)
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


def test_ci_requires_native_quantized_qmv_wide_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized wide matrix-vector native parity",
        family="quantized-qmv-wide",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv_wide.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv_wide.py",
    )
