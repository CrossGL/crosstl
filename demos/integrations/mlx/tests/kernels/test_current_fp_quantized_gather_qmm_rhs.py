"""Prove fp_quantized.metal's fp_gather_qmm_rhs entries natively.

``quantized.cpp``'s ``GatherQMM`` sends a product to ``gather_qmm_rhs`` when
every batch is one activation row (``M == 1``), there are at least 16
batches, the rhs indices are sorted and every expert averages at least four
of them (``B / E >= 4``); ``QQMatmul``'s gather path does the same for its
bfloat16 nvfp4 products. On devices without NAX kernels ``gather_qmm_rhs``
runs ``fp_gather_qmm_rhs``: the batches become ``M = B`` rows of one matrix,
``gather_mm_offsets`` gives each expert's first row, and ``(32, 2, 1)``
threadgroups cover ``16`` by ``32`` output tiles of a
``(ceil(N / 32), min(M, ceil(M / 16) + E - 1), 1)`` grid. ``align_N`` and
``align_K``, function constants 201 and 202, say whether ``N`` and ``K`` are
multiples of 32. ``fp_quantized.metal`` instantiates the kernel for nvfp4,
mxfp8 and mxfp4 in float, bfloat16_t and float16_t, with transposed (``nt``)
and untransposed (``nn``) weights, each without and with one global scale per
expert: 36 of its 518 entries. Each is translated, with ``align_N`` and
``align_K`` frozen, once per pair the host reaches: quantization makes the
quantized dimension (``K`` for ``nt``, ``N`` for ``nn``) a multiple of the
group size, so groups of 32 fix its constant. That gives 96 translations per
target, each pinned and run on two cases.

Each threadgroup first finds its row tile with ``schedule_row_tile``: lanes
take 32 experts at a time, count each expert's tiles of 16 rows and the
prefix sum names the expert, its first row and its valid rows; a threadgroup
past the last tile returns. The ``experts`` case spreads 40 rows over five
experts, two of them empty and two spanning two tiles. The ``chunks`` case
spreads 136 rows over 34 experts, five of them non-empty, so the tiles of the
last expert are found only in the second chunk of 32. Both cases follow the
host's routing and have idle threadgroups. Their ``N`` and ``K`` reach every
branch: whole tiles, partial rows, partial columns and both, the partial
block past the whole blocks of ``K``, and the tails where a weight loader,
which bounds only its rows, reads past ``K`` or ``N`` into the next row or
the guards. The inputs keep every float32 product and partial sum exact; the
per-expert global scales make the bfloat16 and float16 E2M1 weights round.
The tile-level model must reach direct dot products of the dequantized
weights, storing every output once. Each target must produce the model's
words and leave every guard untouched; on Metal the translation must also
match the unchanged upstream kernel, run with the same function constants,
word for word.
"""

import functools
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.gather_mm_layout import MAX_GROUPS
from demos.integrations.mlx.tests.kernels import fp_gather_qmm_rhs_model as model
from demos.integrations.mlx.tests.kernels import fp_gather_qmm_rhs_sources as sources
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    BYTE_POISON,
    GLOBAL_SCALES,
    GUARD_COUNT,
    INDEX_POISON,
    MODES,
    POISON,
    SENTINEL,
    WORD_POISON,
    _code_value,
    _fraction_bits,
    _inverse_scale,
    _scale_byte,
    _scale_value,
    _weight_code,
    translate_entry,
)
from demos.integrations.mlx.tests.kernels.fp_quantized_scope import (
    EXCLUDED_KERNELS,
    GATE_ENTRY_COUNTS,
    UNIT_ENTRY_COUNT,
    shared_scope,
)
from demos.integrations.mlx.tests.kernels.gemm_fused_model import Loader
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import padded
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
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmm_t import (
    HELPER_ELEMENTS,
    QMM_FP_HEADERS,
    _target_options,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _pack_bitstream,
    _to_type,
    _words,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    K_LIMIT,
    SIGNED_MAX,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    SIMD_SIZE,
    _f32,
    _request,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FP_QUANTIZED_QMM_RHS"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_FP_QUANTIZED_QMM_RHS_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_FP_QUANTIZED_QMM_RHS_SHARD_COUNT"
CI_SHARD_COUNT = 2
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/fp_quantized_gather_qmm_rhs.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = sources.KERNEL
BM, BN, BK, WM, WN = model.BM, model.BN, model.BK, model.WM, model.WN
TILE = f"bm_{BM}_bn_{BN}_bk_{BK}_wm_{WM}_wn_{WN}"
# gather_qmm_rhs's group_dims(32, wn, wm).
WORKGROUP = (SIMD_SIZE, WN, WM)
# The kernel's buffers, as gather_qmm_rhs binds them: the global scale in the
# biases' slot, then offsets, y and the four sizes.
PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("x", "w", "scales", "global_scale", "offsets", "y", "M", "N", "K")
        + ("num_groups",)
    )
)
CONSTANTS = ("align_N", "align_K")
CONSTANT_IDS = sources.CONSTANT_IDS
# utils.h's schedule_row_tile, gather_mm_offsets and the host's routing.
SCHEDULE_SOURCE = "mlx/backend/metal/kernels/utils.h"
OFFSETS_SOURCE = "mlx/backend/metal/kernels/gather_mm_offsets.metal"
OFFSETS_SHA256 = "e2b3839ba16c384d65cad76f7426da38f7c11931f5e55bba75c6f70a5f7a98d4"
CASES = ("experts", "chunks")
# Rows per expert, sorted as gather_qmm_rhs requires: B rows over E experts.
COUNTS = {
    # Two tiles of expert 0, the second of one row; an empty expert; three
    # rows; two tiles of expert 3, the second of four rows; an empty last
    # expert, whose rows end at M.
    "experts": (17, 0, 3, 20, 0),
    # 34 experts: 18, 1, 16 and 33 rows in the first chunk of 32, then 68
    # rows of expert 33, whose five tiles only the second chunk finds.
    "chunks": tuple(
        {0: 18, 5: 1, 17: 16, 31: 33, 33: 68}.get(expert, 0) for expert in range(34)
    ),
}
# (aligned, unaligned) N: two column tiles, the second partial when
# unaligned, or one; nn needs a multiple of nvfp4's 16.
N_SHAPES = {"experts": (64, 48), "chunks": (32, 16)}
# Aligned K and the unaligned K of nt, a multiple of nvfp4's 16, and of nn:
# one block and a tail, or the tail alone; nn's tails end inside a vector.
K_SHAPES = {
    "experts": {"aligned": 64, "nt": 48, "nn": 44},
    "chunks": {"aligned": 32, "nt": 16, "nn": 12},
}
# Per-expert global scales, by expert modulo four. inv_scale_enc = g / 2688
# is inexact for 3000, so bfloat16 and float16 E2M1 weights round; float32
# and mxfp8 entries take 3024 first, whose inv_scale_enc of 1.125 is exact.
EXACT_GLOBAL_SCALES = (3024.0,) + GLOBAL_SCALES[1:]


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for name, transpose in (("nt", True), ("nn", False)):
                stem = f"{mode}_gather_qmm_rhs_{name}_{itype}_gs_{group_size}_b_{bits}"
                for hgs in (False, True):
                    yield f"{stem}_{TILE}" + ("_hgs" if hgs else ""), {
                        "kernel": KERNEL,
                        "transpose": transpose,
                        "mode": mode,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                        "globalScale": hgs,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _constants(info):
    """The ``(align_N, align_K)`` pairs the host reaches for one entry.

    Quantization makes the quantized dimension, ``K`` transposed and ``N``
    otherwise, a multiple of the group size, so groups of 32 align it.
    """
    for align_n in (True, False):
        for align_k in (True, False):
            quantized = align_k if info["transpose"] else align_n
            if info["groupSize"] % BK == 0 and not quantized:
                continue
            yield {"align_N": align_n, "align_K": align_k}


def _unit_name(entry, constants):
    """gather_qmm_rhs's hash_name: kname, then each constant as t or n."""
    return entry + "".join(
        f"_{name}_{'t' if constants[name] else 'n'}" for name in CONSTANTS
    )


UNITS = {
    _unit_name(entry, constants): (entry, constants)
    for entry in ENTRIES
    for constants in _constants(INFO[entry])
}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _x_loader(itype):
    """The activation BlockLoader<T, BM, BK, BK_padded, 1, 64>."""
    ld = padded(itype)
    name = f"BlockLoader_{itype}_{BM}_{BK}_{ld}_1_{model.THREADS}__load_unsafe"
    return Loader(name, BM, BK, ld, model.THREADS)


def _workgroup_access_assertions(entry):
    """The x loader's ``dst`` stays inside the slots BlockLoader writes."""
    loader = _x_loader(INFO[entry]["itype"])
    slots = [slot for lane in loader.slots() for slot in lane]
    return (
        {
            "source": SOURCE,
            "entryPoint": entry,
            "function": loader.function,
            "parameter": "self.dst",
            "minimum": min(slots),
            "maximum": max(slots),
        },
    )


def _global_scale_family(info):
    """The OpenGL assertion family of an entry's global scale reads.

    ``crosstl_resource_load_float`` reads the expert's global scale, the same
    helper as a float entry's elements; ``None`` when the entry reads none.
    """
    if not info["globalScale"]:
        return None
    return "elements" if info["dtype"] == "float32" else "global-scales"


def _opengl_index_items(entry):
    """``(expression, scope, maximum, family)`` of each OpenGL assertion.

    ``family`` names what the model shows the expression indexes: the
    activations the x loader reads whole, elements a resource helper reads
    or writes, weight and scale bytes, expert offsets, global scales and
    threadgroup tile slots. GLSL indexes buffers with 32-bit integers, so
    each buffer index is asserted to fit one.
    """
    info = INFO[entry]
    element = HELPER_ELEMENTS[info["dtype"]]
    items = [
        (f"source.offset + {index}", None, SIGNED_MAX, "x-vector")
        for index in range(model.X_READS)
    ]
    accesses = [
        (f"load_{element}", "elements"),
        (f"store_{element}", "y"),
        ("load_uint8", "bytes"),
        ("load_int", "offsets"),
    ]
    if _global_scale_family(info) == "global-scales":
        accesses.append(("load_float", "global-scales"))
    for access, family in accesses:
        scope = {"function": f"crosstl_resource_{access}"}
        items.append(("reference.offset + index", scope, SIGNED_MAX, family))
    ld = padded(info["itype"])
    for tile, rows in (("Xs", BM), ("Ws", BN)):
        scope = {"indexed": f"crosstl_workgroup_{tile}"}
        items.append(
            ("reference.offset + index", scope, (rows - 1) * ld + BK - 1, tile)
        )
    return items


def _config(directory, target, unit):
    """One unit's translation: frozen constants, loader and index assertions.

    ``offsets`` is immutable and ``num_groups`` lies in ``[1, MAX_GROUPS]``,
    so the schedule's early return is uniform over each threadgroup.
    """
    entry, constants = UNITS[unit]
    index_assertions = ()
    if target == "opengl":
        index_assertions = tuple(
            {
                "source": SOURCE,
                "expression": expression,
                **(scope or {}),
                "minimum": 0,
                "maximum": maximum,
            }
            for expression, scope, maximum, _ in _opengl_index_items(entry)
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
        specialization_constants=dict(constants),
        freeze_specialization_constants=True,
        workgroup_access_assertions=(
            _workgroup_access_assertions(entry) if target != "metal" else ()
        ),
        index_range_assertions=index_assertions,
        immutable_storage=[
            {"source": SOURCE, "entryPoint": entry, "parameter": "offsets"}
        ],
        metadata_preconditions=[
            {
                "source": SOURCE,
                "entryPoint": entry,
                "parameter": "num_groups",
                "minimum": 1,
                "maximum": MAX_GROUPS,
            },
            {
                "source": SOURCE,
                "entryPoint": entry,
                "parameter": "K",
                "minimum": 0,
                "maximum": K_LIMIT,
            },
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "target_options": _target_options(INFO[entry]),
            }
        },
    )


def _bound_parameters(descriptor, info):
    """The kernel parameters the translation reflects.

    An entry without a global scale never reads ``global_scale``, so its
    translation may drop the binding; every other parameter stays.
    """
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    if info["globalScale"] or "global_scale" in bound:
        return PARAMETERS
    return tuple(item for item in PARAMETERS if item[0] != "global_scale")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _shape(unit, case):
    """``(N, K)`` of one case."""
    entry, constants = UNITS[unit]
    aligned_n, unaligned_n = N_SHAPES[case]
    n = aligned_n if constants["align_N"] else unaligned_n
    shapes = K_SHAPES[case]
    if constants["align_K"]:
        return n, shapes["aligned"]
    return n, shapes["nt" if INFO[entry]["transpose"] else "nn"]


def _offsets(counts):
    """gather_mm_offsets: each expert's first row, the indices below it."""
    offsets, total = [], 0
    for count in counts:
        offsets.append(total)
        total += count
    return offsets


def _experts(counts):
    """The expert of each row, sorted as gather_qmm_rhs requires."""
    return [expert for expert, count in enumerate(counts) for _ in range(count)]


def _row_tiles(counts):
    """Each row tile ``(row, expert, rows)``, in schedule order."""
    tiles = []
    for expert, (start, count) in enumerate(zip(_offsets(counts), counts)):
        for first in range(0, count, BM):
            tiles.append((start + first, expert, min(BM, count - first)))
    return tiles


def _grid(counts, n):
    """gather_qmm_rhs's grid_dims for ``M = B`` rows over ``E`` experts."""
    m, experts = sum(counts), len(counts)
    return (-(-n // BN), min(m, -(-m // BM) + experts - 1), 1)


def _rounds_weights(info):
    """Whether the element type rounds the global-scale weights."""
    return info["dtype"] != "float32" and info["bits"] == 4


def _global_scales(info):
    return GLOBAL_SCALES if _rounds_weights(info) else EXACT_GLOBAL_SCALES


def _direct(info, data):
    """Each output as a dot product of the kernel's dequantized weights.

    Every weight is ``T(f32(f32(T(scale) * inv_scale_enc) * code))`` with the
    expert's ``inv_scale_enc``, as the loader computes it. The budget bounds
    the float32 arithmetic whatever its order: every product is a multiple
    of ``2^-f`` and the products' magnitudes sum below ``2^(24 - f)``, so
    every partial sum is exact. ``rounded`` counts the weights the element
    type rounds.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    n, k, groups = layout["N"], layout["K"], layout["groups"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    global_scale = data["values"]["global_scale"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget, rounded = 0.0, 0
    for row, expert in enumerate(_experts(layout["counts"])):
        inverse = 1.0
        if info["globalScale"]:
            inverse = _inverse_scale(global_scale[expert])
        vector = x[row * k : (row + 1) * k]
        for col in range(n):
            total, bound, digits = 0.0, 0.0, 0
            for index, value in enumerate(vector):
                # N rows of K codes transposed, else K rows of N codes.
                if info["transpose"]:
                    line, position = expert * n + col, index
                else:
                    line, position = expert * k + index, col
                scale = _scale_value(
                    mode, scales[line * groups + position // group_size]
                )
                scale = _f32(_to_type(dtype, scale) * inverse)
                product = _f32(scale * _code_value(bits, data["codes"][line][position]))
                weight = _to_type(dtype, product)
                rounded += weight != product
                term = value * weight
                total += term
                bound += abs(term)
                digits = max(digits, _fraction_bits(term))
            share = bound / 2.0 ** (24 - digits)
            assert share < 1 and _f32(total) == total, (info, total, bound)
            budget = max(budget, share)
            totals[row * n + col] = total
            expected[row * n + col] = _to_type(dtype, total)
    return {"y": expected}, totals, budget, rounded


@functools.lru_cache(maxsize=None)
def _case_data(unit, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    entry, constants = UNITS[unit]
    info = INFO[entry]
    dtype, bits, mode = info["dtype"], info["bits"], info["mode"]
    counts = COUNTS[case]
    n, k = _shape(unit, case)
    m, experts = sum(counts), len(counts)
    rows, width = (n, k) if info["transpose"] else (k, n)
    groups = width // info["groupSize"]
    draws = _draws("fp-quantized-gather-qmm-rhs", entry, case, n, k)
    x = [_signed(draws, 4) / 4 for _ in range(m * k)]
    codes, scales, w = [], [], []
    for _ in range(experts * rows):
        codes.append([_weight_code(draws, bits) for _ in range(width)])
        scales += [_scale_byte(draws, mode) for _ in range(groups)]
        payload = _pack_bitstream(codes[-1], bits)
        w += list(struct.unpack(f"<{len(payload) // 4}I", bytes(payload)))
    for value in x:
        assert _to_type(dtype, value) == value, (entry, value)
    global_scale = [POISON]
    if info["globalScale"]:
        cycle = _global_scales(info)
        global_scale = [cycle[expert % len(cycle)] for expert in range(experts)]
    initial = [SENTINEL] * (m * n + GUARD_COUNT)
    values = {
        "x": (dtype, x + [POISON] * GUARD_COUNT),
        "w": ("uint32", w + [WORD_POISON] * GUARD_COUNT),
        "scales": ("uint8", scales + [BYTE_POISON] * GUARD_COUNT),
        "global_scale": ("float32", global_scale + [POISON] * GUARD_COUNT),
        "offsets": ("int32", _offsets(counts) + [INDEX_POISON] * GUARD_COUNT),
        "y": (dtype, initial),
        "M": ("int32", [m]),
        "N": ("int32", [n]),
        "K": ("int32", [k]),
        "num_groups": ("int32", [experts]),
    }
    assert set(values) == {name for name, _ in PARAMETERS}, entry
    data = {
        "case": case,
        "layout": {
            "counts": counts,
            "M": m,
            "N": n,
            "K": k,
            "rows": rows,
            "words": width * bits // 32,
            "groups": groups,
        },
        "codes": codes,
        "values": values,
        "initial": {"y": initial},
        "kinds": {"y": dtype},
        "grid": _grid(counts, n),
    }
    data["expected"], data["totals"], data["budget"], data["rounded"] = _direct(
        info, data
    )
    return data


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _simulate(unit, data, mutation=None):
    entry, constants = UNITS[unit]
    return model.simulate(INFO[entry], constants, data, mutation)


def _tail(unit, case):
    """Whether the case's K leaves a partial block, which only nt overreads."""
    return _shape(unit, case)[1] % BK != 0


def _partial_columns(unit, case):
    """Whether the case's N leaves a partial column tile."""
    return _shape(unit, case)[0] % BN != 0


def _mutation_applies(unit, mutation):
    """Whether ``mutation`` is a defect for ``unit`` at all."""
    entry, constants = UNITS[unit]
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "tail":
        return not constants["align_K"]
    if mutation == "tail-zero":
        # The nt weight loader dequantizes past K, which only zeroed
        # activations cancel; nn's loader zeroes its rows past K itself.
        return info["transpose"] and not constants["align_K"]
    if mutation in ("global-scale", "expert-scale"):
        return info["globalScale"]
    if mutation == "fold-scale":
        # Scaling each weight or each sum differs only where weights round.
        return info["globalScale"] and _rounds_weights(info)
    return True


def _expected_paths(unit, case):
    """The branches and stores a case's row and column tiles reach."""
    _, constants = UNITS[unit]
    n, _ = _shape(unit, case)
    counts = COUNTS[case]
    paths = {"idle"} if _grid(counts, n)[1] > len(_row_tiles(counts)) else set()
    if not constants["align_K"]:
        paths.add("finalize")
    for _, _, rows in _row_tiles(counts):
        for column in range(0, n, BN):
            columns = BN if constants["align_N"] else min(BN, n - column)
            whole_rows, whole_columns = rows == BM, columns == BN
            if whole_rows and whole_columns:
                paths |= {"aligned", "store-whole"}
                continue
            paths.add("store-safe")
            if whole_columns:
                paths.add("rows")
            elif whole_rows:
                paths.add("columns")
            else:
                paths.add("both")
    return paths


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _selected_units():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return tuple(UNITS)
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"fp gather_qmm_rhs shard {index} is outside [0, {count})")
    # Alternating units within each entry, from a start that steps per
    # entry, sends each entry's units and its _hgs twin's to both shards.
    selected, position = [], {}
    for unit, (entry, _) in UNITS.items():
        within = position.setdefault(entry, [0])
        if (ENTRIES.index(entry) + within[0]) % count == index:
            selected.append(unit)
        within[0] += 1
    return tuple(selected)


SELECTED_UNITS = _selected_units()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, unit):
    return parse_pin(_contract()["artifacts"][unit][target])


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        len(_case_data(unit, case)["expected"]["y"]) for unit in UNITS for case in CASES
    )


def _expected_validation_contract():
    return {
        "model": (
            "schedule_row_tile over offsets 32 experts at a time, then the "
            "kernel's aligned, partial-row, partial-column or partial branch "
            "with BlockLoader, QuantizedBlockLoader and float32 products per "
            "block of K and the partial block when align_K is false, against "
            "direct dot products of the dequantized weights"
        ),
        "tile": {"BM": BM, "BN": BN, "BK": BK, "WM": WM, "WN": WN},
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), min(M, ceil(M / 16) + E - 1), 1)",
        "functionConstants": {
            "ids": CONSTANT_IDS,
            "frozen": "align_N = N % 32 == 0 and align_K = K % 32 == 0",
            "reached": (
                "groups of 32 align the quantized dimension, K for nt and N for "
                "nn; groups of 16 reach all four pairs"
            ),
            "unit": "gather_qmm_rhs's hash_name: kname_align_N_<t|n>_align_K_<t|n>",
        },
        "routing": (
            "GatherQMM with M == 1, B >= 16, sorted rhs indices and B / E >= 4, "
            "or QQMatmul's bfloat16 nvfp4 gather with K % 32 == 0, on devices "
            "without NAX kernels; _hgs with one global scale per expert"
        ),
        "cases": {
            case: {
                "rowsPerExpert": list(COUNTS[case]),
                "N": dict(zip(("aligned", "unaligned"), N_SHAPES[case])),
                "K": dict(K_SHAPES[case]),
            }
            for case in CASES
        },
        "inputs": (
            "activations in [-1, 1] in steps of 1/4, E2M1 or E4M3 codes and "
            "E4M3 or E8M0 scales"
        ),
        "globalScales": {
            "perExpertModulo4": list(GLOBAL_SCALES),
            "float32AndMxfp8PerExpertModulo4": list(EXACT_GLOBAL_SCALES),
        },
        "exactness": (
            "every product is a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f); only the bfloat16 and float16 E2M1 global-scale "
            "weights and static_cast<T> round, to nearest even"
        ),
        "comparison": "exact words",
        "guards": (
            f"{GUARD_COUNT} guards after every buffer; y's must stay untouched "
            "and only the weight loaders' row tails may read the others"
        ),
        "softwareSubgroups": SIMD_SIZE,
        "cooperativeMatrices": "software 8x8 fragments, tile_4x4_row_pair",
        "globalScaleDivisionProfile": "rne-flush on DirectX and OpenGL for _hgs",
        "workgroupAccessAssertions": "the x loader's dst slots",
        "openglIndexRangeAssertions": (
            "x vectors, resource helper elements, bytes, offsets and global "
            "scales fit 32-bit indices; tile slots stay in their tiles"
        ),
        "immutableStorage": ["offsets"],
        "metadataPreconditions": {"num_groups": [1, MAX_GROUPS], "K": [0, K_LIMIT]},
        "metalUpstreamComparison": (
            "the unchanged upstream kernel with the same function constants, "
            "word for word"
        ),
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("gatherQmmRhs"),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "constantConfigurationCount": len(UNITS),
        "artifactCountByTarget": len(UNITS),
        "dispatchCountByTarget": len(UNITS) * len(CASES),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-gather-qmm-rhs-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(QMM_FP_HEADERS),
            "hostSource": {"path": HOST_SOURCE, "sha256": HOST_SHA256},
            "offsetsSource": {"path": OFFSETS_SOURCE, "sha256": OFFSETS_SHA256},
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


def test_current_mlx_fp_quantized_gather_qmm_rhs_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(UNITS)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            record = parse_pin(pin)
            assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            assert record["sizeBytes"] > 0


def test_fp_quantized_gather_qmm_rhs_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_gather_qmm_rhs.current-tree.translation.json" in readme
    assert len(ENTRIES) == 36 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["gatherQmmRhs"]
    assert sum(GATE_ENTRY_COUNTS.values()) == UNIT_ENTRY_COUNT
    assert EXCLUDED_KERNELS == ()
    assert ENTRIES[:4] == (
        f"nvfp4_gather_qmm_rhs_nt_float_gs_16_b_4_{TILE}",
        f"nvfp4_gather_qmm_rhs_nt_float_gs_16_b_4_{TILE}_hgs",
        f"nvfp4_gather_qmm_rhs_nn_float_gs_16_b_4_{TILE}",
        f"nvfp4_gather_qmm_rhs_nn_float_gs_16_b_4_{TILE}_hgs",
    )
    assert ENTRIES[-1] == f"mxfp4_gather_qmm_rhs_nn_float16_t_gs_32_b_4_{TILE}_hgs"
    assert len(UNITS) == 96
    assert Counter(
        (INFO[entry]["groupSize"], INFO[entry]["transpose"], tuple(constants.values()))
        for entry, constants in UNITS.values()
    ) == {
        **{
            (16, transpose, pair): 6
            for transpose in (True, False)
            for pair in ((True, True), (True, False), (False, True), (False, False))
        },
        **{(32, True, (True, True)): 12, (32, True, (False, True)): 12},
        **{(32, False, (True, True)): 12, (32, False, (True, False)): 12},
    }
    assert list(UNITS)[:4] == [
        f"{ENTRIES[0]}_align_N_{n}_align_K_{k}" for n, k in ("tt", "tn", "nt", "nn")
    ]


def test_fp_quantized_gather_qmm_rhs_shards_partition_the_units(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_units())
    assert sorted(unit for shard in shards for unit in shard) == sorted(UNITS)
    assert [len(shard) for shard in shards] == [48, 48]
    for shard in shards:
        # Each shard runs every kernel, mode, element type, global scale and
        # constant pair, and some unit of every entry.
        for key in ("transpose", "mode", "dtype", "globalScale"):
            counts = Counter(INFO[UNITS[unit][0]][key] for unit in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
        pairs = {tuple(UNITS[unit][1].values()) for unit in shard}
        assert len(pairs) == 4
        assert {UNITS[unit][0] for unit in shard} == set(ENTRIES)
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_units()


@pytest.mark.parametrize("unit", list(UNITS)[::5])
def test_fp_quantized_gather_qmm_rhs_cases_follow_mlx_dispatch(unit):
    entry, constants = UNITS[unit]
    info = INFO[entry]
    for case in CASES:
        data = _case_data(unit, case)
        layout, values = data["layout"], data["values"]
        counts = layout["counts"]
        m, n, k, experts = layout["M"], layout["N"], layout["K"], len(counts)
        # GatherQMM's route: one row per batch, B >= 16, B / E >= 4.
        assert m >= 16 and m // experts >= 4 and m == sum(counts)
        # The frozen constants are the host's for this shape.
        assert constants == {"align_N": n % BN == 0, "align_K": k % BK == 0}
        quantized = k if info["transpose"] else n
        assert quantized % info["groupSize"] == 0
        assert layout["groups"] == quantized // info["groupSize"]
        assert layout["rows"] == (n if info["transpose"] else k)
        assert data["grid"] == (
            (n + BN - 1) // BN,
            min(m, (m + BM - 1) // BM + experts - 1),
            1,
        )
        # gather_mm_offsets' lower bounds of the sorted rhs indices.
        indices = _experts(counts)
        offsets = values["offsets"][1][:-GUARD_COUNT]
        assert offsets == [
            sum(1 for index in indices if index < g) for g in range(experts)
        ]
        assert indices == sorted(indices) and len(indices) == m
        assert values["M"][1] == [m] and values["num_groups"][1] == [experts]
        assert values["N"][1] == [n] and values["K"][1] == [k]
        assert 1 <= experts <= MAX_GROUPS and 0 <= k <= K_LIMIT
        assert (
            len(values["w"][1])
            == experts * layout["rows"] * layout["words"] + GUARD_COUNT
        )
        assert len(values["y"][1]) == m * n + GUARD_COUNT
        scales = values["global_scale"][1][:-GUARD_COUNT]
        if info["globalScale"]:
            assert len(scales) == experts and len(set(scales)) == 4
        else:
            assert scales == [POISON]


def test_fp_quantized_gather_qmm_rhs_schedule_matches_the_row_tiles():
    for case in CASES:
        counts = COUNTS[case]
        offsets, m = _offsets(counts), sum(counts)
        tiles = _row_tiles(counts)
        grid_y = _grid(counts, BN)[1]
        assert len(tiles) < grid_y, case
        for tile in range(grid_y):
            owner = model.schedule(offsets.__getitem__, len(counts), m, tile)
            assert owner == (tiles[tile] if tile < len(tiles) else None), (case, tile)
        # Empty experts, partial tiles and multi-tile experts in both cases.
        assert 0 in counts and any(rows < BM for _, _, rows in tiles)
        assert len({expert for _, expert, _ in tiles}) < len(tiles)
    # The chunks case's last expert lies in the second chunk of 32 lanes.
    assert {expert for _, expert, _ in _row_tiles(COUNTS["chunks"])} >= {31, 33}
    assert COUNTS["experts"][-1] == 0 and len(COUNTS["chunks"]) > SIMD_SIZE


def test_fp_quantized_gather_qmm_rhs_cases_reach_every_path():
    reached = {}
    for unit, (entry, constants) in UNITS.items():
        key = (INFO[entry]["transpose"], tuple(constants.values()))
        for case in CASES:
            reached.setdefault(key, set()).update(_expected_paths(unit, case))
    base = {"idle", "store-whole", "store-safe", "aligned", "rows"}
    every = base | {"columns", "both"}
    for (transpose, (align_n, align_k)), paths in reached.items():
        expected = every if not align_n else base
        assert paths == expected | ({"finalize"} if not align_k else set())
    assert len(reached) == 8


@pytest.mark.parametrize("unit", SELECTED_UNITS[::2])
def test_fp_quantized_gather_qmm_rhs_model_matches_dot_products(unit):
    entry, constants = UNITS[unit]
    info = INFO[entry]
    for case in CASES:
        data = _case_data(unit, case)
        run = _simulate(unit, data)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        counts = layout["counts"]
        assert run["outputs"] == data["expected"], (unit, case)
        assert run["paths"] == _expected_paths(unit, case), (unit, case)
        tiles = _row_tiles(counts)
        assert run["schedule"] == tiles + [None] * (data["grid"][1] - len(tiles))
        written = m * n
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every activation is read, and only the experts with rows are.
        assert run["reads"]["x"] == set(range(m * k))
        used = sorted({expert for _, expert, _ in tiles})
        rows, words, groups = layout["rows"], layout["words"], layout["groups"]
        assert run["reads"]["w"] == {
            (expert * rows + row) * words + word
            for expert in used
            for row in range(rows)
            for word in range(words)
        }
        assert run["reads"]["scales"] == {
            (expert * rows + row) * groups + group
            for expert in used
            for row in range(rows)
            for group in range(groups)
        }
        assert run["reads"]["global_scale"] == (
            set(used) if info["globalScale"] else set()
        )
        # A threadgroup reads both bounds of every expert its lanes take.
        assert run["reads"]["offsets"] == set(range(len(counts)))
        # A transposed tail, or an untransposed partial column tile, makes
        # some threads read the next row's first bytes and scale: words and
        # scales of a row past its expert's last, or the guards.
        over = _tail(unit, case) if info["transpose"] else _partial_columns(unit, case)
        if not over:
            assert run["overreads"] == {"w": set(), "scales": set()}
            continue
        limit_w, limit_s = len(data["values"]["w"][1]), len(data["values"]["scales"][1])
        assert run["overreads"]["w"] and run["overreads"]["scales"]
        assert all(index < limit_w for index in run["overreads"]["w"])
        assert all(index < limit_s for index in run["overreads"]["scales"])
        heads = {
            (expert * rows + row + 1) * groups for expert in used for row in range(rows)
        }
        if info["transpose"]:
            assert run["overreads"]["scales"] <= heads
        else:
            # nn steps whole groups past N: the next row's first groups.
            assert all(
                (index % groups) < groups for index in run["overreads"]["scales"]
            )


def test_fp_quantized_gather_qmm_rhs_model_derives_the_loader_assertions():
    for unit in list(UNITS)[:8:3]:
        entry, _ = UNITS[unit]
        run = _simulate(unit, _case_data(unit, CASES[0]))
        (assertion,) = _workgroup_access_assertions(entry)
        loader = _x_loader(INFO[entry]["itype"])
        slots = {slot for lane in loader.slots() for slot in lane}
        assert {slot for _, slot in run["records"]["Xs"]} == slots
        assert (assertion["minimum"], assertion["maximum"]) == (min(slots), max(slots))
        assert assertion["function"] == loader.function
        assert assertion["entryPoint"] == entry


def test_fp_quantized_gather_qmm_rhs_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for unit in UNITS
        for mutation in model.MUTATIONS
        if not _mutation_applies(unit, mutation)
    )
    assert skipped == {
        "order": 24,
        "tail": 60,
        "tail-zero": 84,
        "global-scale": 48,
        "expert-scale": 48,
        "fold-scale": 72,
    }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_gather_qmm_rhs_inputs_keep_float32_exact(dtype):
    rounded = ties = weights = 0
    for unit, (entry, _) in UNITS.items():
        info = INFO[entry]
        if info["dtype"] != dtype:
            continue
        for case in CASES:
            data = _case_data(unit, case)
            assert data["budget"] < 1
            # Only the global scale makes a loader round a weight.
            assert (data["rounded"] > 0) == (
                info["globalScale"] and _rounds_weights(info)
            )
            weights += data["rounded"]
            for name, (kind, values) in data["values"].items():
                if kind == dtype:
                    assert all(_to_type(dtype, value) == value for value in values)
            for total, value in zip(data["totals"], data["expected"]["y"]):
                assert _f32(total) == total and value == value
                if value != total:
                    rounded += 1
                    neighbour = 2 * total - value
                    ties += _to_type(dtype, neighbour) == neighbour
    if dtype == "float32":
        assert rounded == weights == 0
    else:
        assert rounded > 0 and ties > 0 and weights > 0, (rounded, ties, weights)


def _mutation_cases():
    return [
        (unit, mutation)
        for unit in SELECTED_UNITS
        for mutation in model.MUTATIONS
        if _mutation_applies(unit, mutation)
    ]


@pytest.mark.parametrize("unit,mutation", _mutation_cases())
def test_fp_quantized_gather_qmm_rhs_expectations_reject_defects(unit, mutation):
    for case in CASES:
        data = _case_data(unit, case)
        if _simulate(unit, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{unit}: {model.MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("unit", SELECTED_UNITS[::4])
def test_fp_quantized_gather_qmm_rhs_opengl_assertions_name_buffer_elements(unit):
    entry, _ = UNITS[unit]
    items = _opengl_index_items(entry)
    seen = {family: set() for _, _, _, family in items}
    for case in CASES:
        data = _case_data(unit, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        records = _simulate(unit, data)["records"]
        for _, _, maximum, family in items:
            for buffer, index in records[family]:
                seen[family].add(buffer)
                if family == "bytes":
                    # The weight loaders' row tails may read into the guards.
                    limit = lengths[buffer] * (4 if buffer == "w" else 1)
                elif family in ("Xs", "Ws"):
                    limit = maximum + 1
                else:
                    limit = lengths[buffer] - GUARD_COUNT
                assert 0 <= index < limit <= SIGNED_MAX + 1, (case, family, index)
    expected = {
        "x-vector": {"x"},
        "elements": {"x"},
        "y": {"y"},
        "bytes": {"w", "scales"},
        "offsets": {"offsets"},
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }
    family = _global_scale_family(INFO[entry])
    if family is not None:
        expected.setdefault(family, set()).add("global_scale")
    assert seen == expected


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_gather_qmm_rhs_configs_scope_target_options(tmp_path, target):
    for unit in list(UNITS)[:8]:
        entry, constants = UNITS[unit]
        info = INFO[entry]
        config = _config(tmp_path / "work", target, unit)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        assert config.specialization_constants == constants
        assert config.freeze_specialization_constants is True
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            assert options[backend]["software_subgroup_width"] == SIMD_SIZE
            profile = options[backend].get("binary32_division_profile")
            assert profile == ("rne-flush" if info["globalScale"] else None)
        assert [
            (item.parameter, item.minimum, item.maximum)
            for item in config.metadata_preconditions
        ] == [("num_groups", 1, MAX_GROUPS), ("K", 0, K_LIMIT)]
        assert [item.parameter for item in config.immutable_storage] == ["offsets"]
        assertions = config.workgroup_access_assertions
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "metal":
            assert assertions == () and expressions == []
        else:
            assert [item.entry_point for item in assertions] == [entry]
            assert [item.parameter for item in assertions] == ["self.dst"]
            if target == "opengl":
                assert expressions == [item[0] for item in _opengl_index_items(entry)]
            else:
                assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized gather_qmm_rhs", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    pins = {**QMM_FP_HEADERS, HOST_SOURCE: HOST_SHA256, OFFSETS_SOURCE: OFFSETS_SHA256}
    for header, digest in pins.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    sources.check_instantiation(source.decode("utf-8"))
    kernels = root / "mlx/backend/metal/kernels"
    sources.check_kernel(
        (kernels / "fp_quantized.h").read_text(encoding="utf-8"), PARAMETERS
    )
    sources.check_schedule((root / SCHEDULE_SOURCE).read_text(encoding="utf-8"))
    sources.check_loops((kernels / "quantized_utils.h").read_text(encoding="utf-8"))
    sources.check_host((root / HOST_SOURCE).read_text(encoding="utf-8"))
    sources.check_offsets((root / OFFSETS_SOURCE).read_text(encoding="utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-gather-qmm-rhs-metal-reference")
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


def _upstream(work, reference, entry, constants, label, data):
    """The unchanged upstream kernel on raw ``[[buffer(i)]]`` storage.

    The function constants take the frozen values, as gather_qmm_rhs sets
    them through MTLFunctionConstantValues.
    """
    runner, library = reference
    files, payloads = [], {}
    for name, index in PARAMETERS:
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
                "boolConstants": [
                    {"index": CONSTANT_IDS[name], "value": constants[name]}
                    for name in CONSTANTS
                ],
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    words = None
    for name, index in PARAMETERS:
        raw = (readback / f"buffer-{index}.bin").read_bytes()
        if name == "y":
            words = _words(data["kinds"]["y"], raw)
        else:
            assert raw == payloads[name], name
    return device, words


def _run_case(work, target, unit, case, executor, translated, reference):
    """Dispatch one case on the translation and, on Metal, upstream."""
    descriptor, package = translated
    entry, constants = UNITS[unit]
    data = _case_data(unit, case)
    request, output, observed = _request(
        descriptor,
        package,
        target,
        "gather",
        data,
        parameters=_bound_parameters(descriptor, INFO[entry]),
        workgroup=WORKGROUP,
    )
    words = _execute(executor, work, case, request, {"y": output}, observed)["y"][
        "values"
    ]
    upstream = None
    if target == "metal":
        device, upstream_words = _upstream(
            work, reference, entry, constants, case, data
        )
        # The translation reproduces the unchanged upstream kernel word for
        # word on the same device.
        assert_identical(case, words, upstream_words)
        upstream = {"device": device, "identicalToTranslation": True}
    expected = _runtime_value(target, data["kinds"]["y"], data["expected"]["y"])
    assert_identical(case, words, expected["values"])
    layout = data["layout"]
    return {
        "case": case,
        "grid": list(data["grid"]),
        "M": layout["M"],
        "N": layout["N"],
        "K": layout["K"],
        "experts": len(layout["counts"]),
        "checkedElementCount": len(data["expected"]["y"]),
        "observedInputCount": observed,
        "upstream": upstream,
    }


@pytest.mark.parametrize("unit", SELECTED_UNITS)
def test_current_mlx_fp_quantized_gather_qmm_rhs_native_parity(
    current_mlx, metal_reference, tmp_path, unit
):
    root, target = current_mlx
    record = _record(target, unit)
    with tempfile.TemporaryDirectory(prefix=".current-fpgqmmrhs-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, unit)
            translated = translate_entry(
                root, work, target, config, record, workgroup=WORKGROUP
            )
            descriptor = translated[0]
            constants = descriptor.get("specializationConstants", [])
            assert all(item.get("frozen") is True for item in constants), constants
            executor = _executor(target)
            try:
                evidence = [
                    _run_case(
                        work, target, unit, case, executor, translated, metal_reference
                    )
                    for case in CASES
                ]
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "unit": unit,
                        "constants": UNITS[unit][1],
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


def test_ci_requires_native_fp_quantized_gather_qmm_rhs_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX fp quantized gather_qmm_rhs native parity",
        family="fp-quantized-qmm-rhs",
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_gather_qmm_rhs.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_gather_qmm_rhs.py",
    )
