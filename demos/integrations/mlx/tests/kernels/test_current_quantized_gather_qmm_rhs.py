"""Prove quantized.metal's affine_gather_qmm_rhs entries natively.

``quantized.cpp``'s ``GatherQMM`` sends a product to ``gather_qmm_rhs`` when
every batch is one activation row (``M == 1``), there are at least 16
batches, the rhs indices are sorted and every expert averages at least four
of them (``B / E >= 4``). On devices without NAX kernels, or for
untransposed weights, ``gather_qmm_rhs`` runs ``affine_gather_qmm_rhs``: the
batches become ``M = B`` rows of one matrix, ``gather_mm_offsets`` gives each
expert's first row, and ``(32, 2, 1)`` threadgroups cover ``16`` by ``32``
output tiles of a ``(ceil(N / 32), min(M, ceil(M / 16) + E - 1), 1)`` grid.
``align_N`` and ``align_K``, function constants 201 and 202, say whether
``N`` and ``K`` are multiples of 32. ``quantized.metal`` instantiates the
kernel for float, float16_t and bfloat16_t, group sizes 128, 64 and 32 and
2, 3, 4, 5, 6 and 8 bits, with transposed (``nt``) and untransposed (``nn``)
weights: 108 of its 2,052 entries. Quantization makes the quantized
dimension, ``K`` for ``nt`` and ``N`` for ``nn``, a multiple of the group
size and so of 32, which fixes that constant; each entry is translated, with
``align_N`` and ``align_K`` frozen, once per value of the other. That gives
216 translations per target, each pinned and run on two cases.

The row tiles, cases and branches are the fp rhs gate's: the ``experts``
case spreads 40 rows over five experts, two of them empty and two spanning
two tiles, and the ``chunks`` case 136 rows over 34 experts, so the tiles of
the last expert are found only in the second chunk of 32. ``nt`` reaches
partial column tiles and ``nn`` the partial block of ``K``; because the
quantized dimension is aligned, no weight loader thread reads past its row.
The inputs are the qmv_fast gate's, which keep every dequantized weight
exact in the element type and every float32 accumulator exact, so only the
final ``static_cast<T>`` rounds. The tile-level model must reach direct dot
products of the dequantized weights, storing every output once. Each target
must produce the model's words and leave every guard untouched; on Metal the
translation must also match the unchanged upstream kernel, run with the same
function constants, word for word.
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
from operator import mul
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig
from demos.integrations.mlx.portable_host.gather_mm_layout import MAX_GROUPS
from demos.integrations.mlx.tests.kernels import affine_gather_qmm_rhs_model as model
from demos.integrations.mlx.tests.kernels import (
    affine_gather_qmm_rhs_sources as sources,
)
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import translate_entry
from demos.integrations.mlx.tests.kernels.gemm_fused_model import Loader
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import padded
from demos.integrations.mlx.tests.kernels.quantized_scope import (
    EXCLUDED_KERNELS,
    GATE_ENTRY_COUNTS,
    UNIT_ENTRY_COUNT,
    shared_scope,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_gather_qmm_rhs import (
    CASES,
    COUNTS,
    OFFSETS_SHA256,
    OFFSETS_SOURCE,
    SCHEDULE_SOURCE,
    _experts,
    _grid,
    _offsets,
    _row_tiles,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmm_t import (
    HELPER_ELEMENTS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    BITS,
    DTYPES,
    GROUP_SIZES,
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
    _pack_bitstream,
    _to_type,
    _words,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    K_LIMIT,
    QMM_HEADERS,
    SIGNED_MAX,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    GUARD_COUNT,
    INDEX_POISON,
    POISON,
    SENTINEL,
    SIMD_SIZE,
    WORD_POISON,
    _exact_dyadic,
    _f32,
    _request,
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
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_GATHER_QMM_RHS"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_GATHER_QMM_RHS_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_GATHER_QMM_RHS_SHARD_COUNT"
# Seven is coprime with the two kernels, three element types and three
# group sizes, so every shard runs each of them.
CI_SHARD_COUNT = 7
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_gather_qmm_rhs.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = sources.KERNEL
BM, BN, BK, WM, WN = model.BM, model.BN, model.BK, model.WM, model.WN
TILE = f"bm_{BM}_bn_{BN}_bk_{BK}_wm_{WM}_wn_{WN}"
# gather_qmm_rhs's group_dims(32, wn, wm).
WORKGROUP = (SIMD_SIZE, WN, WM)
# The kernel's [[buffer(i)]] parameters, as gather_qmm_rhs binds them.
PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("x", "w", "scales", "biases", "offsets", "y", "M", "N", "K", "num_groups")
    )
)
CONSTANTS = ("align_N", "align_K")
CONSTANT_IDS = sources.CONSTANT_IDS
# N of nt, aligned and unaligned: two column tiles, the second partial when
# unaligned, or one. Its K is NT_K_GROUPS groups.
NT_N_SHAPES = {"experts": (64, 48), "chunks": (32, 16)}
NT_K_GROUPS = {"experts": 2, "chunks": 1}
# N of nn is NN_N_GROUPS groups; its K, aligned and unaligned: two blocks or
# one block and a tail, one block or the tail alone.
NN_N_GROUPS = {"experts": 2, "chunks": 1}
NN_K_SHAPES = {"experts": (64, 44), "chunks": (32, 12)}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for name, transpose in (("nt", True), ("nn", False)):
                    stem = f"{KERNEL}_{name}_{itype}_gs_{group_size}_b_{bits}"
                    yield f"{stem}_{TILE}", {
                        "kernel": KERNEL,
                        "transpose": transpose,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _constants(info):
    """The ``(align_N, align_K)`` pairs the host reaches for one entry.

    Quantization makes the quantized dimension, ``K`` transposed and ``N``
    otherwise, a multiple of the group size, which is a multiple of 32.
    """
    assert info["groupSize"] % BK == 0
    for align_n in (True, False):
        for align_k in (True, False):
            if not (align_k if info["transpose"] else align_n):
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


def _opengl_index_items(entry):
    """``(expression, scope, maximum, family)`` of each OpenGL assertion.

    ``family`` names what the model shows the expression indexes: the
    activations the x loader reads whole, elements a resource helper reads
    or writes, among them the scales and biases, weight bytes, expert
    offsets and threadgroup tile slots. GLSL indexes buffers with 32-bit
    integers, so each buffer index is asserted to fit one.
    """
    info = INFO[entry]
    element = HELPER_ELEMENTS[info["dtype"]]
    items = [
        (f"source.offset + {index}", None, SIGNED_MAX, "x-vector")
        for index in range(model.X_READS)
    ]
    for access, family in (
        (f"load_{element}", "elements"),
        (f"store_{element}", "y"),
        ("load_uint8", "bytes"),
        ("load_int", "offsets"),
    ):
        scope = {"function": f"crosstl_resource_{access}"}
        items.append(("reference.offset + index", scope, SIGNED_MAX, family))
    ld = padded(info["itype"])
    for tile, rows in (("Xs", BM), ("Ws", BN)):
        scope = {"indexed": f"crosstl_workgroup_{tile}"}
        items.append(
            ("reference.offset + index", scope, (rows - 1) * ld + BK - 1, tile)
        )
    return items


def _target_options():
    """Software subgroups give DirectX and OpenGL MLX's 32 lanes, and the
    cooperative-matrix lowering MLX's BlockMMA fragments."""
    return {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
            "cooperative_matrix_software_lowering": True,
        },
        "opengl": {
            "software_subgroup_width": SIMD_SIZE,
            "cooperative_matrix_software_lowering": True,
        },
    }


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
                "target_options": _target_options(),
            }
        },
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _shape(unit, case):
    """``(N, K)`` of one case."""
    entry, constants = UNITS[unit]
    info = INFO[entry]
    if info["transpose"]:
        aligned, unaligned = NT_N_SHAPES[case]
        n = aligned if constants["align_N"] else unaligned
        return n, NT_K_GROUPS[case] * info["groupSize"]
    aligned, unaligned = NN_K_SHAPES[case]
    k = aligned if constants["align_K"] else unaligned
    return NN_N_GROUPS[case] * info["groupSize"], k


def _dequantized(info, data):
    """Each expert's weights as the loader dequantizes them, ``[line][code]``."""
    layout = data["layout"]
    groups, group_size = layout["groups"], info["groupSize"]
    scales = data["values"]["scales"][1]
    biases = data["values"]["biases"][1]
    weights = []
    for line, codes in enumerate(data["codes"]):
        row = []
        for position, code in enumerate(codes):
            group = line * groups + position // group_size
            # static_cast<T>(code * float(scale) + float(bias)), exact in T.
            weight = _f32(code * scales[group] + biases[group])
            assert _to_type(info["dtype"], weight) == weight, (line, position)
            row.append(weight)
        weights.append(row)
    return weights


def _direct(info, data):
    """Each output as a dot product of the dequantized weights.

    The budget bounds the magnitude of every partial sum of the kernel's
    float32 arithmetic, whatever its order: below 2^19, every multiple of
    2^-5 it can reach is exact in float32.
    """
    layout = data["layout"]
    n, k = layout["N"], layout["K"]
    x = data["values"]["x"][1]
    weights = _dequantized(info, data)
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    columns = {}
    for row, expert in enumerate(_experts(layout["counts"])):
        if expert not in columns:
            # Each output column's K weights: a row nt, a column nn.
            if info["transpose"]:
                columns[expert] = weights[expert * n : (expert + 1) * n]
            else:
                lines = weights[expert * k : (expert + 1) * k]
                columns[expert] = [list(column) for column in zip(*lines)]
        vector = x[row * k : (row + 1) * k]
        magnitudes = [abs(value) for value in vector]
        for col, column in enumerate(columns[expert]):
            total = sum(map(mul, vector, column))
            bound = sum(map(mul, magnitudes, map(abs, column)))
            assert _exact_dyadic(total) and bound < 2**19, (info, total, bound)
            budget = max(budget, bound)
            totals[row * n + col] = total
            expected[row * n + col] = _to_type(info["dtype"], total)
    return {"y": expected}, totals, budget


@functools.lru_cache(maxsize=None)
def _case_data(unit, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    entry, constants = UNITS[unit]
    info = INFO[entry]
    dtype, bits, group_size = info["dtype"], info["bits"], info["groupSize"]
    n_bins = (1 << bits) - 1
    counts = COUNTS[case]
    n, k = _shape(unit, case)
    m, experts = sum(counts), len(counts)
    rows, width = (n, k) if info["transpose"] else (k, n)
    groups = width // group_size
    draws = _draws("quantized-gather-qmm-rhs", entry, case, n, k)
    x = [_signed(draws, 4) / 4 for _ in range(m * k)]
    codes, scales, biases, w = [], [], [], []
    for _ in range(experts * rows):
        row = []
        for _ in range(groups):
            # qmv_fast's groups: every code, with 0 and the top code present.
            values = [_below(draws, n_bins + 1) for _ in range(group_size)]
            low = _below(draws, group_size)
            values[low] = 0
            values[(low + 1 + _below(draws, group_size - 1)) % group_size] = n_bins
            row += values
            scale = 2.0 ** -_below(draws, 4) * (1 if _below(draws, 2) else -1)
            scales.append(scale)
            biases.append(-_below(draws, n_bins + 1) * scale)
        codes.append(row)
        payload = _pack_bitstream(row, bits)
        w += list(struct.unpack(f"<{len(payload) // 4}I", bytes(payload)))
    # Every element lies on the quarter, 2^-3 and 2^-5 grids the exactness
    # argument needs and is exact in the element type.
    assert all(_exact_dyadic(value, 4) and abs(value) <= 1 for value in x)
    assert all(abs(scale) in (1.0, 0.5, 0.25, 0.125) for scale in scales)
    assert all(_exact_dyadic(bias, 8) for bias in biases)
    for value in x + scales + biases:
        assert _to_type(dtype, value) == value, (entry, value)
    initial = [SENTINEL] * (m * n + GUARD_COUNT)
    values = {
        "x": (dtype, x + [POISON] * GUARD_COUNT),
        "w": ("uint32", w + [WORD_POISON] * GUARD_COUNT),
        "scales": (dtype, scales + [POISON] * GUARD_COUNT),
        "biases": (dtype, biases + [POISON] * GUARD_COUNT),
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
    data["expected"], data["totals"], data["budget"] = _direct(info, data)
    return data


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _simulate(unit, data, mutation=None):
    entry, constants = UNITS[unit]
    return model.simulate(INFO[entry], constants, data, mutation)


def _mutation_applies(unit, mutation):
    """Whether ``mutation`` is a defect for ``unit`` at all."""
    entry, constants = UNITS[unit]
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "group-steps":
        # Only a transposed loader whose group spans several blocks waits.
        return info["transpose"] and info["groupSize"] > BK
    if mutation == "tail":
        return not constants["align_K"]
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
        raise RuntimeError(
            f"affine gather_qmm_rhs shard {index} is outside [0, {count})"
        )
    return tuple(
        unit for position, unit in enumerate(UNITS) if position % count == index
    )


SELECTED_UNITS = _selected_units()
# Every third unit: each entry's units come in pairs and each element type's
# in fours, so the stride reaches every kind of unit.
SAMPLED_UNITS = SELECTED_UNITS[::3]


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, unit):
    return parse_pin(_contract()["artifacts"][unit][target])


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        sum(COUNTS[case]) * _shape(unit, case)[0] + GUARD_COUNT
        for unit in UNITS
        for case in CASES
    )


def _expected_validation_contract():
    return {
        "model": (
            "schedule_row_tile over offsets 32 experts at a time, then the "
            "kernel's aligned, partial-row, partial-column or partial branch "
            "with BlockLoader, the affine QuantizedBlockLoader and float32 "
            "products per block of K and the partial block when align_K is "
            "false, against direct dot products of the dequantized weights"
        ),
        "tile": {"BM": BM, "BN": BN, "BK": BK, "WM": WM, "WN": WN},
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), min(M, ceil(M / 16) + E - 1), 1)",
        "functionConstants": {
            "ids": CONSTANT_IDS,
            "frozen": "align_N = N % 32 == 0 and align_K = K % 32 == 0",
            "reached": (
                "groups of 32 or more align the quantized dimension, K for nt "
                "and N for nn"
            ),
            "unit": "gather_qmm_rhs's hash_name: kname_align_N_<t|n>_align_K_<t|n>",
        },
        "routing": (
            "GatherQMM with M == 1, B >= 16, sorted rhs indices and B / E >= 4; "
            "nt on devices without NAX kernels"
        ),
        "cases": {
            case: {
                "rowsPerExpert": list(COUNTS[case]),
                "nt": {
                    "N": dict(zip(("aligned", "unaligned"), NT_N_SHAPES[case])),
                    "KGroups": NT_K_GROUPS[case],
                },
                "nn": {
                    "NGroups": NN_N_GROUPS[case],
                    "K": dict(zip(("aligned", "unaligned"), NN_K_SHAPES[case])),
                },
            }
            for case in CASES
        },
        "inputs": (
            "activations in [-1, 1] in steps of 1/4, every code of each group "
            "with 0 and the top code, scales of +-2^-j for j < 4 and biases of "
            "-c * scale"
        ),
        "exactness": (
            "every dequantized weight is exact in the element type and every "
            "partial sum a multiple of 2^-5 below 2^19; only static_cast<T> "
            "rounds, to nearest even"
        ),
        "comparison": "exact words",
        "guards": (
            f"{GUARD_COUNT} guards after every buffer; y's must stay untouched "
            "and no loader reads them"
        ),
        "softwareSubgroups": SIMD_SIZE,
        "cooperativeMatrices": "software 8x8 fragments, tile_4x4_row_pair",
        "workgroupAccessAssertions": "the x loader's dst slots",
        "openglIndexRangeAssertions": (
            "x vectors, resource helper elements, weight bytes and offsets fit "
            "32-bit indices; tile slots stay in their tiles"
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
        "kind": "crosstl-mlx-current-tree-quantized-gather-qmm-rhs-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(QMM_HEADERS),
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


def test_current_mlx_quantized_gather_qmm_rhs_contract_is_exact():
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


def test_quantized_gather_qmm_rhs_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_gather_qmm_rhs.current-tree.translation.json" in readme
    assert len(ENTRIES) == 108 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["gatherQmmRhs"]
    assert sum(GATE_ENTRY_COUNTS.values()) == UNIT_ENTRY_COUNT
    assert EXCLUDED_KERNELS == ()
    assert ENTRIES[:4] == (
        f"{KERNEL}_nt_float_gs_128_b_2_{TILE}",
        f"{KERNEL}_nn_float_gs_128_b_2_{TILE}",
        f"{KERNEL}_nt_float16_t_gs_128_b_2_{TILE}",
        f"{KERNEL}_nn_float16_t_gs_128_b_2_{TILE}",
    )
    assert ENTRIES[-1] == f"{KERNEL}_nn_bfloat16_t_gs_32_b_8_{TILE}"
    assert len(UNITS) == 216
    assert Counter(
        (INFO[entry]["transpose"], tuple(constants.values()))
        for entry, constants in UNITS.values()
    ) == {
        (True, (True, True)): 54,
        (True, (False, True)): 54,
        (False, (True, True)): 54,
        (False, (True, False)): 54,
    }
    assert list(UNITS)[:4] == [
        f"{ENTRIES[0]}_align_N_t_align_K_t",
        f"{ENTRIES[0]}_align_N_n_align_K_t",
        f"{ENTRIES[1]}_align_N_t_align_K_t",
        f"{ENTRIES[1]}_align_N_t_align_K_n",
    ]


def test_quantized_gather_qmm_rhs_shards_partition_the_units(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_units())
    assert sorted(unit for shard in shards for unit in shard) == sorted(UNITS)
    assert sorted(len(shard) for shard in shards) == [30] + [31] * 6
    for shard in shards:
        # Each shard runs every kernel, element type, group size, width and
        # constant pair.
        for key in ("transpose", "dtype", "groupSize", "bits"):
            counts = Counter(INFO[UNITS[unit][0]][key] for unit in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
        pairs = {
            (INFO[UNITS[unit][0]]["transpose"], tuple(UNITS[unit][1].values()))
            for unit in shard
        }
        assert len(pairs) == 4
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_units()


@pytest.mark.parametrize("unit", list(UNITS)[::5])
def test_quantized_gather_qmm_rhs_cases_follow_mlx_dispatch(unit):
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
        groups = experts * layout["rows"] * layout["groups"]
        assert len(values["scales"][1]) == len(values["biases"][1])
        assert len(values["scales"][1]) == groups + GUARD_COUNT
        assert len(values["y"][1]) == m * n + GUARD_COUNT


def test_quantized_gather_qmm_rhs_cases_reach_every_path():
    reached = {}
    for unit, (entry, constants) in UNITS.items():
        key = (INFO[entry]["transpose"], tuple(constants.values()))
        for case in CASES:
            reached.setdefault(key, set()).update(_expected_paths(unit, case))
    base = {"idle", "store-whole", "store-safe", "aligned", "rows"}
    assert reached == {
        (True, (True, True)): base,
        (True, (False, True)): base | {"columns", "both"},
        (False, (True, True)): base,
        (False, (True, False)): base | {"finalize"},
    }


@pytest.mark.parametrize("unit", SAMPLED_UNITS)
def test_quantized_gather_qmm_rhs_model_matches_dot_products(unit):
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
        for name in ("scales", "biases"):
            assert run["reads"][name] == {
                (expert * rows + row) * groups + group
                for expert in used
                for row in range(rows)
                for group in range(groups)
            }
        # A threadgroup reads both bounds of every expert its lanes take.
        assert run["reads"]["offsets"] == set(range(len(counts)))


def test_quantized_gather_qmm_rhs_model_derives_the_loader_assertions():
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


def test_quantized_gather_qmm_rhs_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for unit in UNITS
        for mutation in model.MUTATIONS
        if not _mutation_applies(unit, mutation)
    )
    assert skipped == {"order": 36, "group-steps": 144, "tail": 162}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_gather_qmm_rhs_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
    for unit, (entry, _) in UNITS.items():
        info = INFO[entry]
        if info["dtype"] != dtype:
            continue
        for case in CASES:
            data = _case_data(unit, case)
            assert data["budget"] < 2**19
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
        assert rounded == 0
    else:
        assert rounded > 0 and ties > 0, (rounded, ties)


def _mutation_cases():
    return [
        (unit, mutation)
        for unit in SAMPLED_UNITS
        for mutation in model.MUTATIONS
        if _mutation_applies(unit, mutation)
    ]


@pytest.mark.parametrize("unit,mutation", _mutation_cases())
def test_quantized_gather_qmm_rhs_expectations_reject_defects(unit, mutation):
    for case in CASES:
        data = _case_data(unit, case)
        if _simulate(unit, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{unit}: {model.MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("unit", SELECTED_UNITS[::9])
def test_quantized_gather_qmm_rhs_opengl_assertions_name_buffer_elements(unit):
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
                    limit = (lengths[buffer] - GUARD_COUNT) * 4
                elif family in ("Xs", "Ws"):
                    limit = maximum + 1
                else:
                    limit = lengths[buffer] - GUARD_COUNT
                assert 0 <= index < limit <= SIGNED_MAX + 1, (case, family, index)
    assert seen == {
        "x-vector": {"x"},
        "elements": {"x", "scales", "biases"},
        "y": {"y"},
        "bytes": {"w"},
        "offsets": {"offsets"},
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_gather_qmm_rhs_configs_scope_target_options(tmp_path, target):
    for unit in list(UNITS)[:8]:
        entry, constants = UNITS[unit]
        config = _config(tmp_path / "work", target, unit)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        assert config.specialization_constants == constants
        assert config.freeze_specialization_constants is True
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            assert options[backend]["software_subgroup_width"] == SIMD_SIZE
            assert "binary32_division_profile" not in options[backend]
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
        REQUIRE_ENV, "quantized gather_qmm_rhs", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    pins = {**QMM_HEADERS, HOST_SOURCE: HOST_SHA256, OFFSETS_SOURCE: OFFSETS_SHA256}
    for header, digest in pins.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    sources.check_instantiation(source.decode("utf-8"))
    kernels = root / "mlx/backend/metal/kernels"
    sources.check_kernel(
        (kernels / "quantized.h").read_text(encoding="utf-8"), PARAMETERS
    )
    sources.check_schedule((root / SCHEDULE_SOURCE).read_text(encoding="utf-8"))
    sources.check_loops((kernels / "quantized_utils.h").read_text(encoding="utf-8"))
    sources.check_affine_host((root / HOST_SOURCE).read_text(encoding="utf-8"))
    sources.check_offsets((root / OFFSETS_SOURCE).read_text(encoding="utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-gather-qmm-rhs-metal-reference")
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
        parameters=PARAMETERS,
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
def test_current_mlx_quantized_gather_qmm_rhs_native_parity(
    current_mlx, metal_reference, tmp_path, unit
):
    root, target = current_mlx
    record = _record(target, unit)
    with tempfile.TemporaryDirectory(prefix=".current-gqmmrhs-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, unit)
            translated = translate_entry(
                root,
                work,
                target,
                config,
                record,
                workgroup=WORKGROUP,
                source_sha256=SOURCE_SHA256,
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


def test_ci_requires_native_quantized_gather_qmm_rhs_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized gather_qmm_rhs native parity",
        family="quantized-gather-qmm-rhs",
        count=CI_SHARD_COUNT,
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        f"{SHARD_INDEX_ENV}: ${{{{ matrix.shard }}}}",
        f'{SHARD_COUNT_ENV}: "{CI_SHARD_COUNT}"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/${{ matrix.group }}",
        "--junitxml=mlx-current-results/${{ matrix.group }}.xml",
        "demos/integrations/mlx/tests/kernels/test_current_quantized_gather_qmm_rhs.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_gather_qmm_rhs.py",
    )
