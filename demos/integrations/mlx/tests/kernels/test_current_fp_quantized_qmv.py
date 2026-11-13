"""Prove fp_quantized.metal's fp_qmv and fp_gather_qmv entries natively.

``fp_quantized.metal`` instantiates ``fp_qmv`` with and without batching and
``fp_gather_qmv``, each without and with a global scale (``_hgs``), for nvfp4
(groups of 16 E2M1 codes with E4M3 scales), mxfp8 (groups of 32 E4M3 codes
with E8M0 scales) and mxfp4 (groups of 32 E2M1 codes with E8M0 scales) in
float, bfloat16_t and float16_t: 54 of its 518 entries. Each is translated
and pinned per target, then runs ``quantized.cpp``'s ``qmv`` or
``gather_qmv`` dispatch: threadgroups of ``(32, 2, 1)`` threads, two
simdgroups of four output rows each, over a grid of ``(M, ceil(N / 8), B)``.

Unbatched entries take each of ``fp_qmv_impl``'s paths; batched and gather
entries add both branches of each ``adjust_matrix_offsets`` overload and the
per-expert global scale. Inputs are quarters in ``[-1, 1]``, every E2M1
code, E4M3 codes from subnormal to 15 and scales near one, so only the
global scale's product and ``static_cast<T>`` round. A lane-level model
must reach a direct dequantized dot product; each target must produce its
words and leave every guard untouched, and on Metal the translation must
also match the unchanged upstream kernel word for word.
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
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    declared_buffers,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    DTYPES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    ENTRIES as QUANTIZE_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    F4E2M1_MAX,
    F8E4M3_MAX,
    HEADERS,
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
    UNIT_ENTRY_COUNT,
    _div,
    _e2m1_decode,
    _e4m3_decode,
    _e8m0_decode,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _pack_bitstream,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    ARRAY_TYPES,
    BROADCAST_RHS,
    EXPERT_RHS,
    NUM_SIMDGROUPS,
    QMV_HEADERS,
    RESULTS_PER_SIMDGROUP,
    ROWS_PER_GROUP,
    SCALAR_TYPES,
    SIMD_SIZE,
    UNBATCHED_BOUND,
    WORKGROUP,
    _elem_to_loc,
    _f32,
    _request,
    _upstream,
    layout_for,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _below,
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FP_QUANTIZED_QMV"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/fp_quantized_qmv.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# adjust_matrix_offsets' general branches use steel/utils.h's
# elem_to_loc_broadcast.
STEEL_UTILS = "mlx/backend/metal/kernels/steel/utils.h"
QMV_FP_HEADERS = {**HEADERS, STEEL_UTILS: QMV_HEADERS[STEEL_UTILS]}

# (mode, group size, bits) in instantiate_quantized_types' order.
MODES = (("nvfp4", 16, 4), ("mxfp8", 32, 8), ("mxfp4", 32, 4))
# Per type and mode, instantiate_quantized_all_batched's qmv wrap instantiates
# fp_qmv with batched 1, then 0, before instantiate_quantized_all_single's
# fp_gather_qmv; each comes without, then with, the _hgs global scale.
VARIANTS = ("batch_1", "batch_0", "gather")
KERNELS = {"batch_1": "fp_qmv", "batch_0": "fp_qmv", "gather": "fp_gather_qmv"}
# The 36 fp_quantize, fp_dequantize and fp_quantize_dequantize entries have
# their own gate, test_current_fp_quantized.py.
QUANTIZE_ENTRY_COUNT = len(QUANTIZE_ENTRIES)
# Kernels of fp_quantized.metal no gate covers yet.
EXCLUDED_KERNELS = (
    "fp_qmv_fast",
    "fp_qvm",
    "fp_qmm_n",
    "fp_gather_qmv_fast",
    "fp_gather_qvm",
    "fp_gather_qmm_n",
    "fp_qmv_quad",
    "fp_qmv_wide",
    "fp_qvm_split_k",
    "fp_qmm_t_splitk",
    "fp_gather_qmm_t",
    "fp_qmm_t",
    "fp_gather_qmm_rhs",
)

# fp_qmv and fp_gather_qmv take their buffers in declaration order.
_BATCH_BUFFERS = ("x_batch_ndims", "x_shape", "x_strides", "w_batch_ndims")
_BATCH_BUFFERS += ("w_shape", "w_strides", "s_strides")
QMV_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("w", "scales", "global_scale", "x", "y", "in_vec_size", "out_vec_size")
        + _BATCH_BUFFERS
    )
)
GATHER_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("w", "scales", "global_scale", "x", "lhs_indices", "rhs_indices", "y")
        + ("in_vec_size", "out_vec_size")
        + _BATCH_BUFFERS
        + ("batch_ndims", "batch_shape", "lhs_strides", "rhs_strides")
    )
)

# Unbatched entries take each of fp_qmv_impl's paths; batched and gather
# entries combine them with adjust_matrix_offsets' layouts.
CASES = {
    "batch_0": ("narrow", "short", "ragged", "deep"),
    "batch_1": ("stacked", "broadcast", "shared"),
    "gather": ("experts", "broadcast"),
}
# (M, N, whole blocks, extra groups): K starts at blocks * block_size +
# groups * group_size. Without a global scale, dispatch_qmv sends M >= 2 to
# qmv_wide, so fp_qmv runs one vector.
SHAPES = {
    # Five rows: the guarded branch, the second simdgroup on one row.
    "narrow": (1, 5, 1, 1),
    # Three rows: the second simdgroup returns before simd_sum.
    "short": (2, 3, 0, 3),
    # Thirteen rows: the last tile moves back four rows minus the remainder.
    "ragged": (2, 13, 2, 1),
    # Whole tiles, and every lane's tail a whole block.
    "deep": (1, 16, 3, 0),
    "stacked": (1, 6, 1, 1),
    "broadcast": (2, 13, 1, 2),
    # Eleven rows: the last tile's second simdgroup returns and its first
    # moves back.
    "shared": (1, 11, 2, 1),
    "experts": (1, 3, 1, 1),
}
LAYOUTS = {
    "narrow": "unbatched",
    "short": "unbatched",
    "ragged": "unbatched",
    "deep": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
    "experts": "experts",
}
# K values dispatch_qmv sends to qmv_quad without a global scale.
QUAD_K = (64, 128)
# Global scales: inv_scale_enc = g / (F8E4M3_MAX * F4E2M1_MAX) is 3000 / 2688
# (inexact, so the product rounds), 4, 2 or 1 / 2. fp_qmv reads one for
# every batch; fp_gather_qmv one per expert.
GLOBAL_SCALES = (3000.0, 10752.0, 5376.0, 1344.0)
CASE_GLOBAL_SCALES = {
    "narrow": 3000.0,
    "short": 10752.0,
    "ragged": 3000.0,
    "deep": 5376.0,
    "stacked": 3000.0,
    "broadcast": 10752.0,
    "shared": 3000.0,
}
# E4M3 exponent fields of the mxfp8 codes: subnormals through 2^2 * 1.875.
E4M3_CODE_FIELDS = tuple(range(10))
# Scale bytes: E4M3 2^-2 through 3.75 for nvfp4, E8M0 2^-3 through 2 for mx.
NV_SCALE_FIELDS = (5, 6, 7, 8)
MX_SCALE_BYTES = (124, 125, 126, 127, 128)
GUARD_COUNT = 8
SENTINEL = -3.25
# Exactly representable poison for every element no case may read.
POISON = 1.5
WORD_POISON = 0xA5A5A5A5
BYTE_POISON = 0xA5
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
        "idx.x",
        "idx.y",
    ),
    "gather": (
        "tid.z * lhs_strides[0]",
        "tid.z * rhs_strides[0]",
        "idx.x",
        "idx.y",
        "x_idx * x_strides[0]",
        "w_idx * w_strides[0]",
        "w_idx * s_strides[0]",
    ),
}


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for variant in VARIANTS:
                for global_scale in (False, True):
                    name = "gather_qmv" if variant == "gather" else "qmv"
                    entry = f"{mode}_{name}_{itype}_gs_{group_size}_b_{bits}"
                    if variant != "gather":
                        entry += f"_{variant}"
                    if global_scale:
                        entry += "_hgs"
                    yield entry, {
                        "kernel": KERNELS[variant],
                        "variant": variant,
                        "mode": mode,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                        "globalScale": global_scale,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _options(entry):
    """Per-target options: software subgroups for simd_sum, and correctly
    rounded division for the global scale's ``g / 2688``."""
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    if INFO[entry]["globalScale"]:
        for target in options.values():
            target["binary32_division_profile"] = "rne-flush"
    return options


def _config(directory, target, entry):
    """The translation of one entry, as the gate and contract run it."""
    variant = INFO[entry]["variant"]
    assertions = OPENGL_INDEX_RANGE_ASSERTIONS[variant] if target == "opengl" else ()
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
                "target_options": _options(entry),
            }
        },
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits, group_size):
    """fp_qmv_impl's constants: one 32-bit pack per lane."""
    # get_pack_factor<32, bits>() and get_bytes_per_pack<32>().
    pack_factor = 32 // bits
    return {
        "packFactor": pack_factor,
        "bytesPerPack": 4,
        "valuesPerThread": pack_factor,
        "blockSize": pack_factor * SIMD_SIZE,
        "scaleStepPerThread": group_size // pack_factor,
        # qmv_fast_k_alignment(bits).
        "fastAlignment": pack_factor * 2 * SIMD_SIZE,
    }


def _routes_to_qmv(info, m, n, k):
    """Whether quantized.cpp dispatches fp_qmv_impl for this product."""
    geometry = _geometry(info["bits"], info["groupSize"])
    if n % ROWS_PER_GROUP == 0 and k % geometry["fastAlignment"] == 0:
        return False
    if info["variant"] == "gather" or info["globalScale"]:
        # gather_qmv has no qmv_quad or qmv_wide route, and dispatch_qmv
        # takes neither with a global scale.
        return True
    return k not in QUAD_K and m == 1


def _shape(entry, case):
    """``(M, N, K)``: K grows by a group until quantized.cpp picks fp_qmv."""
    info = INFO[entry]
    m, n, blocks, groups = SHAPES[case]
    if not (info["globalScale"] or info["variant"] == "gather"):
        m = 1
    geometry = _geometry(info["bits"], info["groupSize"])
    k = blocks * geometry["blockSize"] + groups * info["groupSize"]
    while not _routes_to_qmv(info, m, n, k):
        k += info["groupSize"]
    return m, n, k


def _layout(entry, case):
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    layout = layout_for(info, LAYOUTS[case], *_shape(entry, case))
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    return layout


def _parameters(variant):
    return GATHER_PARAMETERS if variant == "gather" else QMV_PARAMETERS


def _weight_code(draws, bits):
    if bits == 4:
        return _below(draws, 16)
    field = E4M3_CODE_FIELDS[_below(draws, len(E4M3_CODE_FIELDS))]
    return _below(draws, 2) << 7 | field << 3 | _below(draws, 8)


def _scale_byte(draws, mode):
    if mode == "nvfp4":
        field = NV_SCALE_FIELDS[_below(draws, len(NV_SCALE_FIELDS))]
        return field << 3 | _below(draws, 8)
    return MX_SCALE_BYTES[_below(draws, len(MX_SCALE_BYTES))]


def _scale_value(mode, byte):
    """dequantize_scale<float, group_size>: E4M3 for groups of 16, else E8M0."""
    return _e4m3_decode(byte) if mode == "nvfp4" else _e8m0_decode(byte)


def _code_value(bits, byte):
    """Dequantize<bits>: E4M3 of the byte, or E2M1 of its low nibble."""
    return _e4m3_decode(byte) if bits == 8 else _e2m1_decode(byte)


def _experts(layout):
    """The number of global scales fp_gather_qmv may read: one per expert."""
    params = layout["params"]
    return math.prod(params["w_shape"][: params["w_batch_ndims"]])


def _batch_expert(case, z):
    """The rhs index, so the expert and its global scale, of batch ``z``."""
    return EXPERT_RHS[z] if case == "experts" else BROADCAST_RHS[z // 2]


def _inverse_scale(global_scale):
    return _div(global_scale, F8E4M3_MAX * F4E2M1_MAX)


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    info = INFO[entry]
    dtype, bits, mode = info["dtype"], info["bits"], info["mode"]
    layout = _layout(entry, case)
    m, n, k = layout["M"], layout["N"], layout["K"]
    draws = _draws("fp-quantized-qmv", entry, case)
    used_rows = {
        batch["x"] + row * k for batch in layout["batches"] for row in range(m)
    }
    x = [POISON] * (layout["xRows"] * k)
    for start in sorted(used_rows):
        x[start : start + k] = [_signed(draws, 4) / 4 for _ in range(k)]
    codes, scales = [], []
    for _ in range(layout["matrices"] * layout["rows"]):
        codes.append([_weight_code(draws, bits) for _ in range(k)])
        scales += [_scale_byte(draws, mode) for _ in range(layout["groups"])]
    w = []
    for row in codes:
        payload = _pack_bitstream(row, bits)
        w += list(struct.unpack(f"<{len(payload) // 4}I", bytes(payload)))
    assert len(w) == layout["matrices"] * layout["rows"] * layout["words"]
    for value in x:
        assert _to_type(dtype, value) == value, (entry, value)
    if not info["globalScale"]:
        global_scale = [POISON]
    elif info["variant"] == "gather":
        global_scale = list(GLOBAL_SCALES[: _experts(layout)])
    else:
        global_scale = [CASE_GLOBAL_SCALES[case]]
    total = len(layout["batches"]) * m * n
    initial = [SENTINEL] * (total + GUARD_COUNT)
    params = layout["params"]
    values = {
        "w": ("uint32", w + [WORD_POISON] * GUARD_COUNT),
        "scales": ("uint8", scales + [BYTE_POISON] * GUARD_COUNT),
        "global_scale": ("float32", global_scale + [POISON] * GUARD_COUNT),
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
        "case": case,
        "layout": layout,
        "codes": codes,
        "values": values,
        "initial": {"y": initial},
        "kinds": {"y": dtype},
        "grid": layout["grid"],
    }
    data["expected"], data["totals"], data["budget"] = _direct(info, data)
    return data


def _fraction_bits(value):
    """The fewest binary fraction digits that hold ``value`` exactly."""
    denominator = Fraction(value).denominator
    assert denominator & (denominator - 1) == 0, value
    return denominator.bit_length() - 1


def _direct(info, data):
    """Each output as a dequantized dot product, its float32 total and budget.

    The budget bounds the kernel's float32 arithmetic whatever its order:
    every product is a multiple of ``2^-f`` and the products' magnitudes sum
    below ``2^(24 - f)``, so every partial sum is exact. It is the largest
    such sum, in units of ``2^(24 - f)``.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    global_scale = data["values"]["global_scale"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    for z, batch in enumerate(layout["batches"]):
        factor = None
        if info["globalScale"]:
            expert = (
                _batch_expert(data["case"], z) if info["variant"] == "gather" else 0
            )
            factor = _inverse_scale(global_scale[expert])
        for row in range(m):
            vector = x[batch["x"] + row * k : batch["x"] + (row + 1) * k]
            for col in range(n):
                line = batch["w"] * n + col
                total, bound, digits = 0.0, 0.0, 0
                for index, value in enumerate(vector):
                    scale = _scale_value(
                        mode, scales[line * groups + index // group_size]
                    )
                    term = value * _code_value(bits, data["codes"][line][index]) * scale
                    total += term
                    bound += abs(term)
                    digits = max(digits, _fraction_bits(term))
                share = bound / 2.0 ** (24 - digits)
                assert share < 1 and _f32(total) == total, (info, total, bound)
                budget = max(budget, share)
                index = batch["y"] + row * n + col
                totals[index] = total
                if factor is not None:
                    total = _f32(total * factor)
                expected[index] = _to_type(dtype, total)
    return {"y": expected}, totals, budget


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    "scale-step": "gives every lane its block's first scale",
    "order": "pairs each lane's inputs with its codes in reverse",
    "rows": "ignores the simdgroup's row offset",
    "batch": "ignores the weight batch offsets",
    "tail": "drops the safe tail after the whole blocks",
    "sign": "decodes every code without its sign bit",
    "global-scale": "drops the global scale's factor",
    "expert-scale": "reads the first expert's global scale for every expert",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "batch":
        return info["variant"] != "batch_0"
    if mutation == "global-scale":
        return info["globalScale"]
    if mutation == "expert-scale":
        return info["globalScale"] and info["variant"] == "gather"
    return True


def _checked(value):
    """A float32 intermediate the exactness argument covers."""
    assert _f32(value) == value, value
    return value


def _qdot(bits, byte, base, x_thread, scale, mutation):
    """qdot over one lane's pack, expression for expression."""
    # The sign bit of the E4M3 byte or of the E2M1 low nibble.
    sign = 0xFF if mutation != "sign" else 0x7F if bits == 8 else 0xF7
    accum = 0.0
    if bits == 4:
        # qdot reads the pack as little-endian uint16_t words and hands
        # Dequantize<4> the low byte of each shift; fp4_e2m1 keeps its low
        # nibble.
        for i in range(len(x_thread) // 4):
            ws = byte(base + 2 * i) | byte(base + 2 * i + 1) << 8
            terms = [
                x_thread[4 * i + j] * _code_value(4, (ws >> (4 * j)) & 0xFF & sign)
                for j in range(4)
            ]
            partial = _checked(_checked(terms[0] + terms[1]) + terms[2])
            accum = _checked(accum + _checked(partial + terms[3]))
    else:
        for i, value in enumerate(x_thread):
            accum = _checked(accum + value * _code_value(8, byte(base + i) & sign))
    return _checked(scale * accum)


def _adjust(info, storage, z, mutation, records):
    """adjust_matrix_offsets for ``tid.z``: x, w, scales, global scale, y."""

    def scalar(name):
        return storage[name][0]

    if info["variant"] == "gather":
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
    strides = (storage["w_strides"], storage["s_strides"])
    if scalar("w_batch_ndims") == 1:
        bases = [w_idx * values[0] for values in strides]
        records += [
            ("w_idx * w_strides[0]", "w", bases[0]),
            ("w_idx * s_strides[0]", "scales", bases[1]),
        ]
    else:
        bases = _elem_to_loc(
            w_idx, storage["w_shape"], strides, scalar("w_batch_ndims")
        )
        records += [("idx.x", "w", bases[0]), ("idx.y", "scales", bases[1])]
    if mutation == "batch":
        bases = [0, 0]
    # The gather overload moves global_scale to the expert's; fp_qmv's
    # batched overload leaves it.
    expert = 0
    if info["variant"] == "gather" and info["globalScale"]:
        expert = 0 if mutation == "expert-scale" else w_idx
    rows = storage["x_shape"][scalar("x_batch_ndims")]
    return (x_base, *bases, expert, z * scalar("out_vec_size") * rows)


def _lane(context, lane, first, rows, mutation):
    """One lane's per-row partial sums over the whole blocks and the tail."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    bits, group_size, mode = info["bits"], info["groupSize"], info["mode"]
    x_base, w_base, s_base = context["bases"]
    per_thread, block = geometry["valuesPerThread"], geometry["blockSize"]
    in_vec_size, in_vec_size_w, in_vec_size_g = context["sizes"]
    step = geometry["scaleStepPerThread"]
    group = 0 if mutation == "scale-step" else lane // step
    ws = w_base * 4 + first * in_vec_size_w + lane * geometry["bytesPerPack"]
    scales = s_base + first * in_vec_size_g + group
    x = x_base + context["tid_x"] * in_vec_size + lane * per_thread
    result = [0.0] * RESULTS_PER_SIMDGROUP

    def accumulate(count):
        # load_vector and load_vector_safe convert T to float exactly.
        values = [load("x", x + i) for i in range(count)]
        if mutation == "order":
            values.reverse()
        for row in rows:
            scale = _scale_value(mode, load("scales", scales + row * in_vec_size_g))
            result[row] = _checked(
                result[row]
                + _qdot(
                    bits,
                    context["byte"],
                    ws + row * in_vec_size_w,
                    values,
                    scale,
                    mutation,
                )
            )

    k = iterations = 0
    while k < in_vec_size - block:
        accumulate(per_thread)
        ws += block * geometry["bytesPerPack"] // geometry["packFactor"]
        scales += block // group_size
        x += block
        k += block
        iterations += 1
    remaining = min(max(in_vec_size - k - lane * per_thread, 0), per_thread)
    # K is a multiple of group_size, so of every pack factor: a lane's tail
    # is whole or empty, and load_vector_safe never zero-fills.
    assert remaining in (0, per_thread), remaining
    if remaining and mutation != "tail":
        accumulate(remaining)
    return result, iterations, remaining > 0


def _simulate(entry, data, mutation=None):
    """fp_qmv_impl lane by lane; returns outputs, stores, reads, offsets, paths."""
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"])
    storage = {name: values for name, (_, values) in data["values"].items()}
    in_vec_size, out_vec_size = storage["in_vec_size"][0], storage["out_vec_size"][0]
    reads = {name: set() for name in storage}
    outputs = {"y": list(data["initial"]["y"])}
    stores, records, paths = Counter(), [], set()

    def load(name, index):
        assert 0 <= index < len(storage[name]) - GUARD_COUNT, (name, index)
        reads[name].add(index)
        return storage[name][index]

    def byte(index):
        word = load("w", index // 4)
        return (word >> (8 * (index % 4))) & 0xFF

    guarded = out_vec_size < NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP
    grid_x, grid_y, grid_z = data["grid"]
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if info["variant"] != "batch_0":
            bases = _adjust(info, storage, z, mutation, records)
        expert, y_base = bases[3], bases[4]
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:3],
                "tid_x": tid_x,
                "sizes": (
                    in_vec_size,
                    in_vec_size * geometry["bytesPerPack"] // geometry["packFactor"],
                    in_vec_size // info["groupSize"],
                ),
            }
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    out_row = tid_y * ROWS_PER_GROUP
                    if mutation != "rows":
                        out_row += simd_gid * RESULTS_PER_SIMDGROUP
                    if out_row >= out_vec_size:
                        # A simdgroup-uniform return before simd_sum.
                        paths.add("exit")
                        continue
                    factor = None
                    if info["globalScale"]:
                        factor = _inverse_scale(load("global_scale", expert))
                        if mutation == "global-scale":
                            factor = None
                    if guarded:
                        first = out_row
                        count = min(RESULTS_PER_SIMDGROUP, out_vec_size - out_row)
                        paths.add("guarded")
                    else:
                        first = min(out_vec_size - RESULTS_PER_SIMDGROUP, out_row)
                        count = RESULTS_PER_SIMDGROUP
                        paths.add("moved" if first < out_row else "whole")
                    rows = range(count)
                    lanes, tails = [], 0
                    for lane in range(SIMD_SIZE):
                        result, iterations, tail = _lane(
                            context, lane, first, rows, mutation
                        )
                        lanes.append(result)
                        tails += tail
                    paths.add("blocks" if iterations else "tail-only")
                    paths.add("full-tail" if tails == SIMD_SIZE else "partial-tail")
                    for row in rows:
                        # simd_sum: exact in any order; lane 0 stores
                        # static_cast<T>(sum * inv_scale_enc).
                        total = 0.0
                        for result in lanes:
                            total = _checked(total + result[row])
                        if factor is not None:
                            total = _f32(total * factor)
                        index = y_base + tid_x * out_vec_size + first + row
                        assert 0 <= index < len(outputs["y"]) - GUARD_COUNT, index
                        value = _to_type(info["dtype"], total)
                        if stores["y", index]:
                            # A moved tile stores the value already there.
                            assert outputs["y"][index] == value, index
                        stores["y", index] += 1
                        outputs["y"][index] = value
    return {
        "outputs": outputs,
        "stores": stores,
        "reads": reads,
        "records": records,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_FP_QUANTIZED_QMV_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_FP_QUANTIZED_QMV_SHARD_COUNT"
CI_SHARD_COUNT = 2


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"fp quantized qmv shard {index} is outside [0, {count})")
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
            "fp_qmv_impl lane by lane after adjust_matrix_offsets: whole blocks "
            "through load_vector and qdot, the lane's tail through "
            "load_vector_safe and qdot or qdot_safe, Dequantize's E2M1 and "
            "E4M3 decoding, dequantize_scale's E4M3 and E8M0 scales, the "
            "guarded rows, the early return and the moved-back last tile, "
            "simd_sum, the global scale and lane 0's static_cast stores, "
            "against direct dequantized dot products"
        ),
        "packing": "least-significant-bit-first code stream per weight row",
        "comparison": "exact words",
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / 8), B)",
        "k": (
            "blocks * 32 * pack_factor + groups * group_size, plus group_size "
            "until quantized.cpp picks fp_qmv over fp_qmv_fast and qmv_quad"
        ),
        "m": "1 without a global scale outside gather, which dispatch_qmv "
        "otherwise sends to qmv_wide",
        "cases": {
            variant: {
                case: dict(zip(("M", "N", "blocks", "groups"), SHAPES[case]))
                for case in cases
            }
            for variant, cases in CASES.items()
        },
        "inputs": (
            "x in quarters of [-1, 1]; every E2M1 code; E4M3 codes of either "
            "sign with exponent fields 0 to 9; nvfp4 scales E4M3 with "
            "exponent fields 5 to 8, mx scales E8M0 bytes 124 to 128"
        ),
        "globalScales": {
            "qmv": dict(CASE_GLOBAL_SCALES),
            "gatherPerExpert": list(GLOBAL_SCALES),
        },
        "exactness": (
            "every product is a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f); only the global scale's product and static_cast<T> "
            "round, to nearest even"
        ),
        "guards": {"count": GUARD_COUNT, "sentinel": SENTINEL},
        "softwareSubgroups": {
            "width": SIMD_SIZE,
            "directxShuffleOutOfRange": "self",
        },
        "globalScaleDivisionProfile": "rne-flush",
        "openglIndexRangeAssertions": {
            variant: {
                "expressions": list(expressions),
                "minimum": 0,
                "maximum": INDEX_BOUND,
            }
            for variant, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
        },
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": "unchanged fp_quantized.metal, word for word",
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        "quantizeGateEntryCount": QUANTIZE_ENTRY_COUNT,
        "excludedEntries": {
            "kernels": list(EXCLUDED_KERNELS),
            "count": UNIT_ENTRY_COUNT - QUANTIZE_ENTRY_COUNT - len(ENTRIES),
            "reason": (
                "the fast, quad, wide and split-K matrix-vector, the "
                "vector-matrix and the matrix-matrix kernels are not gated yet"
            ),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmv-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(QMV_FP_HEADERS),
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


def test_current_mlx_fp_quantized_qmv_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            record = parse_pin(pin)
            assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            assert record["sizeBytes"] > 0


def test_fp_quantized_qmv_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmv.current-tree.translation.json" in readme
    assert len(ENTRIES) == 54 == len(set(ENTRIES))
    assert Counter(info["variant"] for info in INFO.values()) == dict.fromkeys(
        VARIANTS, 18
    )
    assert Counter(info["globalScale"] for info in INFO.values()) == {
        False: 27,
        True: 27,
    }
    assert ENTRIES[:6] == (
        "nvfp4_qmv_float_gs_16_b_4_batch_1",
        "nvfp4_qmv_float_gs_16_b_4_batch_1_hgs",
        "nvfp4_qmv_float_gs_16_b_4_batch_0",
        "nvfp4_qmv_float_gs_16_b_4_batch_0_hgs",
        "nvfp4_gather_qmv_float_gs_16_b_4",
        "nvfp4_gather_qmv_float_gs_16_b_4_hgs",
    )
    assert ENTRIES[-1] == "mxfp4_gather_qmv_float16_t_gs_32_b_4_hgs"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert not set(ENTRIES) & set(QUANTIZE_ENTRIES)


def test_fp_quantized_qmv_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {27}
    for shard in shards:
        variants = Counter(INFO[entry]["variant"] for entry in shard)
        assert variants == dict.fromkeys(VARIANTS, 9)
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"])
    assert geometry["packFactor"] * info["bits"] == 8 * geometry["bytesPerPack"]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert _routes_to_qmv(info, m, n, k)
        assert k % info["groupSize"] == 0
        assert data["grid"] == (m, -(-n // ROWS_PER_GROUP), batches)
        # qmv's and gather_qmv's kernel names for this product.
        hgs = "_hgs" if info["globalScale"] else ""
        stem = f"{info['mode']}_{{}}_{info['itype']}_gs_{info['groupSize']}"
        stem += f"_b_{info['bits']}"
        if info["variant"] == "gather":
            assert entry == stem.format("gather_qmv") + hgs
        else:
            suffix = "_batch_1" if batches > 1 else "_batch_0"
            assert entry == stem.format("qmv") + suffix + hgs
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        # add_strides_and_shapes: x.ndim() - 2 and w.ndim() - 2 batch dims,
        # and no bias strides.
        assert "b_strides" not in params
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]
        if info["globalScale"] and info["variant"] == "gather":
            assert len(values["global_scale"][1]) == _experts(layout) + GUARD_COUNT
            assert max(layout["rhs"]) < _experts(layout)


def test_fp_quantized_qmv_cases_reach_every_path():
    paths = {variant: set() for variant in VARIANTS}
    for entry in ENTRIES:
        variant = INFO[entry]["variant"]
        for case in CASES[variant]:
            paths[variant] |= _simulate(entry, _case_data(entry, case))["paths"]
    every = {"exit", "guarded", "moved", "whole", "blocks", "tail-only"}
    every |= {"full-tail", "partial-tail"}
    assert paths["batch_0"] == every
    assert {"moved", "exit", "blocks", "partial-tail"} <= paths["batch_1"]
    assert {"guarded", "moved", "exit"} <= paths["gather"]


def test_fp_quantized_qmv_inputs_reach_every_code():
    for entry in ENTRIES:
        info = INFO[entry]
        codes = Counter()
        for case in CASES[info["variant"]]:
            for row in _case_data(entry, case)["codes"]:
                codes.update(row)
        if info["bits"] == 4:
            assert set(codes) == set(range(16)), entry
        else:
            fields = {(code >> 3) & 15 for code in codes}
            assert fields == set(E4M3_CODE_FIELDS), entry
            assert {code >> 7 for code in codes} == {0, 1}, entry
            assert not {0x7F, 0xFF} & set(codes), entry


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_model_matches_dot_products(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert all(run["stores"]["y", index] >= 1 for index in range(written))
        assert len(run["stores"]) == written
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word and scale of the used matrices is read, and no
        # guard or unused activation row is.
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
        assert run["reads"]["scales"] == {
            (matrix * n + col) * groups + group
            for matrix in matrices
            for col in range(n)
            for group in range(groups)
        }
        if not info["globalScale"]:
            assert run["reads"]["global_scale"] == set()
        elif info["variant"] == "gather":
            assert run["reads"]["global_scale"] == {
                _batch_expert(case, z) for z in range(len(layout["batches"]))
            }
        else:
            assert run["reads"]["global_scale"] == {0}


def test_fp_quantized_qmv_mutations_apply_where_they_are_defects():
    applied = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    )
    assert applied["batch"] == 36 and applied["global-scale"] == 27
    assert applied["expert-scale"] == 9
    for mutation in ("scale-step", "order", "rows", "tail", "sign"):
        assert applied[mutation] == len(ENTRIES), mutation


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmv_inputs_keep_float32_exact(dtype):
    rounded = 0
    for entry in ENTRIES:
        info = INFO[entry]
        if info["dtype"] != dtype:
            continue
        for case in CASES[info["variant"]]:
            data = _case_data(entry, case)
            assert data["budget"] < 1
            for name, (kind, values) in data["values"].items():
                if kind == dtype:
                    assert all(_to_type(dtype, value) == value for value in values)
            for total, value in zip(data["totals"], data["expected"]["y"]):
                assert _f32(total) == total and math.isfinite(value)
                if value != total and not info["globalScale"]:
                    rounded += 1
    if dtype == "float32":
        # Without a global scale nothing rounds.
        assert rounded == 0
    else:
        assert rounded > 0


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    ],
)
def test_fp_quantized_qmv_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_opengl_assertions_name_buffer_elements(entry):
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
def test_fp_quantized_qmv_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES:
        info = INFO[entry]
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        division = (
            {"binary32_division_profile": "rne-flush"} if info["globalScale"] else {}
        )
        assert options == {
            "directx": {
                "software_subgroup_width": SIMD_SIZE,
                "relative_wave_shuffle_out_of_range": "self",
                **division,
            },
            "opengl": {"software_subgroup_width": SIMD_SIZE, **division},
        }
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "opengl":
            assert expressions == list(OPENGL_INDEX_RANGE_ASSERTIONS[info["variant"]])
        else:
            assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmv", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_FP_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # ENTRIES follow fp_quantized.metal's own instantiation macros.
    text = source.decode("utf-8")
    types = re.findall(r"^instantiate_quantized_types\((\w+)\)", text, re.MULTILINE)
    assert types == [itype for itype, _ in DTYPES]
    modes = re.findall(
        r"instantiate_quantized_modes\(type, (\w+), (\d+), (\d+)\)",
        _macro(text, "instantiate_quantized_types"),
    )
    assert modes == [(mode, str(size), str(bits)) for mode, size, bits in MODES]
    groups = _macro(text, "instantiate_quantized_modes")
    assert groups.index("all_batched") < groups.index("all_single")
    batched = re.findall(
        r"instantiate_quantized_batched_wrap\((\w+),",
        _macro(text, "instantiate_quantized_all_batched"),
    )
    assert batched[1] == "qmv"
    wrap = _macro(text, "instantiate_quantized_batched_wrap")
    assert re.findall(r"type, (\d), group_size", wrap) == ["1", "0"]
    single = re.findall(
        r"instantiate_quantized\(mode, (\w+),",
        _macro(text, "instantiate_quantized_all_single"),
    )
    assert single[1] == "gather_qmv"
    # Each instantiation adds its _hgs twin after it.
    for macro in ("instantiate_quantized", "instantiate_quantized_batched"):
        body = _macro(text, macro)
        assert body.count("instantiate_kernel(") == 2
        assert body.index('"_hgs"') > body.index("instantiate_kernel(", 1)
    header = (root / "mlx/backend/metal/kernels/fp_quantized.h").read_text(
        encoding="utf-8"
    )
    assert declared_buffers(header, "fp_qmv") == QMV_PARAMETERS
    assert declared_buffers(header, "fp_gather_qmv") == GATHER_PARAMETERS
    impl = header[header.index("METAL_FUNC void fp_qmv_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int packs_per_thread = 1;",
        "constexpr int pack_factor = get_pack_factor<32, bits>();",
        "constexpr int bytes_per_pack = get_bytes_per_pack<32>();",
        "const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);",
        "if (out_row >= out_vec_size) {",
        "inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);",
        "if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {",
        "for (; k < in_vec_size - block_size; k += block_size) {",
        "load_vector_safe<T, U, values_per_thread>(x, x_thread, remaining);",
        "qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, remaining);",
        "y[row] = static_cast<T>(result[row] * inv_scale_enc);",
    ):
        assert required in impl, required
    assert "global_scale += w_idx;" in header
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "int bk = 32;",
        "MTL::Size group_dims(bk, 2, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        "bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;",
        "return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;",
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) {",
        "if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) {",
        'return mode != "affine" || d.get_architecture_gen() >= 15;',
        'mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),',
        'mode + (fast ? "_qmv_fast_" : "_qmv_"),',
        'B > 1 ? "_batch_1" : "_batch_0"',
        'global_scale ? "_hgs" : "");',
        "compute_encoder.set_input_array(*global_scale, 2);",
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
    ):
        assert required in host, required
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-qmv-metal-reference")
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
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
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


def _bound_parameters(descriptor, entry):
    """The kernel parameters the translation reflects.

    An entry without a global scale never reads ``global_scale``, so its
    translation may drop the binding; every other parameter stays.
    """
    parameters = _parameters(INFO[entry]["variant"])
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    if INFO[entry]["globalScale"] or "global_scale" in bound:
        return parameters
    return tuple(item for item in parameters if item[0] != "global_scale")


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_fp_quantized_qmv_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-fpqmv-", dir=root) as path:
        work = Path(path)
        try:
            descriptor, package = _translate(root, work, target, entry, record)
            parameters = _bound_parameters(descriptor, entry)
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES[variant]:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, variant, data, parameters
                    )
                    words = _execute(
                        executor, work, case, request, {"y": output}, observed
                    )["y"]["values"]
                    upstream = None
                    if target == "metal":
                        reference = (work, metal_reference, entry, variant, case)
                        device, upstream_words = _upstream(
                            *reference, data, _parameters(variant)
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        assert_identical(case, words, upstream_words)
                        upstream = {"device": device, "identicalToTranslation": True}
                    expected = _runtime_value(
                        target, data["kinds"]["y"], data["expected"]["y"]
                    )["values"]
                    assert_identical(case, words, expected)
                    layout = data["layout"]
                    evidence.append(
                        {
                            "case": case,
                            "grid": list(data["grid"]),
                            "M": layout["M"],
                            "N": layout["N"],
                            "K": layout["K"],
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


def test_ci_requires_native_fp_quantized_qmv_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX fp quantized matrix-vector native parity",
        family="fp-quantized-qmv",
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv.py",
    )
