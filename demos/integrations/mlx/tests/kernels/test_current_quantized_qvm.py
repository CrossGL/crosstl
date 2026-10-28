"""Natively prove quantized.metal's 162 affine qvm vector-matrix kernels.

``quantized.cpp``'s ``qvm`` and ``gather_qvm`` multiply vectors by weights
that are not transposed, ``K`` rows of ``N`` codes quantized along ``N``: for
``M < 4``, and for ``qvm`` while ``K < 1024``, they pick ``affine_qvm`` and
``affine_gather_qvm``. ``quantized.metal`` instantiates both for float,
float16_t and bfloat16_t, group sizes 128, 64 and 32 and 2, 3, 4, 5, 6 and 8
bits, ``affine_qvm`` with and without batching: 162 of its 2,052 entries.
Each is translated and pinned per target, then runs ``(32, 2, 1)``
threadgroups over an ``(M, ceil(N / 64), B)`` grid.

``qvm_impl`` gives each simdgroup 32 output columns and each lane one input
row per block of 32: the lane reads its row's ``vec_w`` record of 32 codes
through a private byte view and ``qouter`` adds ``x * (scale * w + bias)``
into all 32 results. A ``K`` that is a multiple of 32 runs whole blocks;
otherwise the blocks after the first run whole and a last pass reads only
where ``K`` reaches the lane, zeroing ``x``, the scale and the bias of the
other lanes. A simdgroup whose first column lies past ``N`` returns before
``simd_sum``, which group size 32 reaches with an odd number of groups. The
cases reach every path, with the inputs of the ``qmv_fast`` gate, so float32
holds every intermediate exactly and only the final ``static_cast<T>``
rounds. A model replays each lane and must reach direct dequantized dot
products, storing every column once. Each target must produce the model's
words and leave every guard untouched; on Metal the translation must also
match the unchanged upstream kernel word for word.
"""

import functools
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
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
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
    UNIT_ENTRY_COUNT,
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    BITS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    CASES as QMV_FAST_CASES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import DTYPES
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    ENTRIES as QMV_FAST_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    EXCLUDED_KERNELS,
    GATHER_PARAMETERS,
    GATHER_QMM_ENTRY_COUNT,
    GROUP_SIZES,
    GUARD_COUNT,
    INDEX_BOUND,
    NUM_SIMDGROUPS,
    OPENGL_INDEX_RANGE_ASSERTIONS,
    QMM_N_ENTRY_COUNT,
    QMM_T_ENTRY_COUNT,
    QMM_T_SPLITK_ENTRY_COUNT,
    QMV_HEADERS,
    QMV_PARAMETERS,
    QMV_QUAD_ENTRY_COUNT,
    QMV_UNALIGNED_ENTRY_COUNT,
    QMV_WIDE_ENTRY_COUNT,
    QUANTIZE_ENTRY_COUNT,
    QVM_ENTRY_COUNT,
    QVM_SPLIT_K_ENTRY_COUNT,
    SIMD_SIZE,
    UNBATCHED_BOUND,
    VARIANTS,
    WORKGROUP,
    _adjust,
    _checked,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _config as _qmv_config,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _expected_validation_contract as _qmv_fast_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _f32,
    _request,
    _translate,
    _upstream,
    case_data_for,
    layout_for,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QVM"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_qvm.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNELS = {
    "gather": "affine_gather_qvm",
    "batch_1": "affine_qvm",
    "batch_0": "affine_qvm",
}
# qvm_impl's 32 output columns per simdgroup, and the input rows of a block.
COLUMNS = 32
BLOCK = SIMD_SIZE

# Unbatched entries take each of qvm_impl's paths; batched and gather entries
# combine them with adjust_matrix_offsets' layouts.
CASES = {
    "batch_0": ("tail", "blocks", "ragged"),
    "batch_1": ("stacked", "broadcast", "shared"),
    "gather": ("experts", "broadcast"),
}
# (M, N in groups, K): N is a multiple of group_size, as quantize requires.
SHAPES = {
    # K below one block: only the last pass, on 13 of the 32 lanes.
    "tail": (1, 1, 13),
    # Whole blocks of 32 input rows.
    "blocks": (3, 2, 64),
    # Two blocks, then nine lanes of the last pass.
    "ragged": (2, 3, 73),
    "stacked": (1, 1, 45),
    "broadcast": (2, 2, 32),
    "shared": (1, 3, 71),
    "experts": (1, 2, 50),
}
GATHER_SHAPES = {"broadcast": (2, 3, 64)}
LAYOUTS = {
    "tail": "unbatched",
    "blocks": "unbatched",
    "ragged": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
    "experts": "experts",
}
# quantized.cpp's qvm routes K >= 1024 to qvm_split_k and M >= 4 to qmm;
# gather_qvm leaves M >= 4 to gather_qmm and sorted M == 1, B >= 16 batches
# to gather_qmm_rhs.
SPLIT_K = 1024
VECTOR_LIMIT = 4


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


def _config(directory, target, entry):
    """The translation of one entry, with the qmv_fast gate's options."""
    return _qmv_config(directory, target, entry, INFO[entry]["variant"])


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits):
    """qvm_impl's constants: one 32-code record per lane for every width."""
    # get_pack_factor<bits, 32>() and get_bytes_per_pack<bits>().
    pack_factor = 8 if bits in (3, 5) else 4 if bits == 6 else 32 // bits
    power_of_2_bits = bits & (bits - 1) == 0
    bytes_per_pack = 1 if power_of_2_bits else 5 if bits == 5 else 3
    tn = 32 // pack_factor
    # W_T: uint32_t for power-of-two widths, uint8_t otherwise.
    width = 4 if power_of_2_bits else 1
    return {
        "packFactor": pack_factor,
        "bytesPerPack": bytes_per_pack,
        "tn": tn,
        "wordBytes": width,
        "recordBytes": tn * bytes_per_pack * width,
        "columns": tn * pack_factor,
    }


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    info = INFO[entry]
    shapes = GATHER_SHAPES if info["variant"] == "gather" else {}
    m, groups, k = shapes.get(case, SHAPES[case])
    return m, groups * info["groupSize"], k


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    layout = layout_for(info, LAYOUTS[case], *_shape(entry, case), transposed=True)
    return case_data_for(info, entry, case, layout)


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    "bias": "drops the bias term",
    "order": "pairs each pack's codes with its columns in reverse",
    "columns": "ignores the simdgroup's column offset",
    "group": "gives every column the first group's scale and bias",
    "scale-step": "keeps the first block's scales and biases",
    "tail": "drops the last pass after the whole blocks",
    "batch": "ignores the weight batch offsets",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    return mutation != "batch" or INFO[entry]["variant"] != "batch_0"


def _unpack(bits, byte, base):
    """qouter's 32 codes of one record as ``(masked value, scale divisor)``.

    Two- and four-bit codes keep their in-byte position, which qouter
    divides out of the scale; the other widths shift the code down.
    """
    codes = []
    if bits == 2:
        for i in range(8):
            w = byte(base + i)
            codes += [(w & 0x03, 1), (w & 0x0C, 4), (w & 0x30, 16), (w & 0xC0, 64)]
    elif bits == 3:
        for i in range(4):
            w0, w1, w2 = (byte(base + 3 * i + j) for j in range(3))
            codes += [
                (w0 & 0x7, 1),
                ((w0 & 0x38) >> 3, 1),
                (((w0 & 0xC0) >> 6) + ((w1 & 0x1) << 2), 1),
                ((w1 & 0xE) >> 1, 1),
                ((w1 & 0x70) >> 4, 1),
                (((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1), 1),
                ((w2 & 0x1C) >> 2, 1),
                ((w2 & 0xE0) >> 5, 1),
            ]
    elif bits == 4:
        for i in range(16):
            w = byte(base + i)
            codes += [(w & 0x0F, 1), (w & 0xF0, 16)]
    elif bits == 5:
        for i in range(4):
            w0, w1, w2, w3, w4 = (byte(base + 5 * i + j) for j in range(5))
            codes += [
                (w0 & 0x1F, 1),
                (((w0 & 0xE0) >> 5) + ((w1 & 0x3) << 3), 1),
                ((w1 & 0x7C) >> 2, 1),
                (((w1 & 0x80) >> 7) + ((w2 & 0xF) << 1), 1),
                (((w2 & 0xF0) >> 4) + ((w3 & 0x1) << 4), 1),
                ((w3 & 0x3E) >> 1, 1),
                (((w3 & 0xC0) >> 6) + ((w4 & 0x7) << 2), 1),
                ((w4 & 0xF8) >> 3, 1),
            ]
    elif bits == 6:
        for i in range(8):
            w0, w1, w2 = (byte(base + 3 * i + j) for j in range(3))
            codes += [
                (w0 & 0x3F, 1),
                (((w0 >> 6) & 0x03) + ((w1 & 0x0F) << 2), 1),
                (((w1 >> 4) & 0x0F) + ((w2 & 0x03) << 4), 1),
                ((w2 >> 2) & 0x3F, 1),
            ]
    else:
        codes = [(byte(base + i), 1) for i in range(32)]
    assert len(codes) == COLUMNS, bits
    return codes


def _qouter(codes, x, scale, bias, result, mutation):
    """``result[i] += x * (scale * w[i] + bias)``, expression for expression."""
    for index, (masked, divisor) in enumerate(codes):
        # scale / 4.0f, / 16.0f and / 64.0f of a power of two are exact.
        part = scale / divisor
        assert _f32(part) == part, part
        term = _checked(part * masked)
        if mutation != "bias":
            term = _checked(term + bias)
        result[index] = _checked(result[index] + _checked(x * term))


def _lane(context, lane, out_col, mutation):
    """One lane's 32 partial sums: the whole blocks, then the last pass."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    group_size = info["groupSize"]
    x_base, w_base, s_base, b_base = context["bases"]
    in_vec_size, out_vec_size_w, out_vec_size_g = context["sizes"]
    pack_factor, width = geometry["packFactor"], geometry["wordBytes"]
    group = 0 if mutation == "group" else out_col // group_size
    # ws counts W_T elements from w, which adjust_matrix_offsets moved in words.
    ws = out_col * geometry["bytesPerPack"] // pack_factor + lane * out_vec_size_w
    scales = s_base + group + lane * out_vec_size_g
    biases = b_base + group + lane * out_vec_size_g
    # x moves by in_vec_stride rows, which only qvm_split_k sets apart.
    x = x_base + context["tid_x"] * context["stride"] + lane
    result = [0.0] * COLUMNS
    w_local = [(0, 1)] * COLUMNS

    def record():
        codes = _unpack(info["bits"], context["byte"], w_base * 4 + ws * width)
        if mutation == "order":
            codes = [
                codes[start + pack_factor - 1 - i]
                for start in range(0, COLUMNS, pack_factor)
                for i in range(pack_factor)
            ]
        return codes

    def block():
        nonlocal ws, scales, biases, x, w_local
        x_local = load("x", x)
        scale, bias = load("scales", scales), load("biases", biases)
        w_local = record()
        _qouter(w_local, x_local, scale, bias, result, mutation)
        x += BLOCK
        if mutation != "scale-step":
            scales += BLOCK * out_vec_size_g
            biases += BLOCK * out_vec_size_g
        ws += BLOCK * out_vec_size_w

    remaining = in_vec_size % BLOCK
    if remaining == 0:
        for _ in range(0, in_vec_size, BLOCK):
            block()
        return result, False
    for _ in range(BLOCK, in_vec_size, BLOCK):
        block()
    reads = lane < remaining and mutation != "tail"
    if reads:
        x_local = load("x", x)
        scale, bias = load("scales", scales), load("biases", biases)
        w_local = record()
    else:
        # The other lanes keep their last record, or none, and add zeros.
        x_local = scale = bias = 0.0
    _qouter(w_local, x_local, scale, bias, result, mutation)
    return result, reads


def _simulate(entry, data, mutation=None, info=None, sizes=None):
    """qvm_impl lane by lane; returns outputs, stores, reads, offsets and paths.

    ``sizes(storage, z, mutation)`` gives ``tid.z``'s ``in_vec_size`` and
    ``in_vec_stride``, which qvm passes as its ``in_vec_size`` twice.
    """
    info = info or INFO[entry]
    geometry = _geometry(info["bits"])
    storage = {name: values for name, (_, values) in data["values"].items()}
    out_vec_size = storage["out_vec_size"][0]
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

    grid_x, grid_y, grid_z = data["grid"]
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if info["variant"] != "batch_0":
            bases = _adjust(info["variant"], storage, z, mutation, records)
        y_base = bases[4]
        in_vec_size = stride = storage["in_vec_size"][0]
        if sizes is not None:
            in_vec_size, stride = sizes(storage, z, mutation)
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:4],
                "tid_x": tid_x,
                "stride": stride,
                "sizes": (
                    in_vec_size,
                    out_vec_size * geometry["bytesPerPack"] // geometry["packFactor"],
                    out_vec_size // info["groupSize"],
                ),
            }
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    tile = tid_y * NUM_SIMDGROUPS
                    if mutation != "columns":
                        tile += simd_gid
                    out_col = geometry["columns"] * tile
                    if out_col >= out_vec_size:
                        # A simdgroup-uniform return before simd_sum.
                        paths.add("exit")
                        continue
                    lanes, tails = [], 0
                    for lane in range(SIMD_SIZE):
                        result, tail = _lane(context, lane, out_col, mutation)
                        lanes.append(result)
                        tails += tail
                    if in_vec_size % BLOCK == 0:
                        paths.add("whole")
                    else:
                        paths.add("blocks-tail" if in_vec_size > BLOCK else "tail-only")
                        assert tails == in_vec_size % BLOCK or mutation == "tail"
                    for column in range(COLUMNS):
                        # simd_sum: exact in any order; lane 0 stores T(sum).
                        total = 0.0
                        for result in lanes:
                            total = _checked(total + result[column])
                        index = y_base + tid_x * out_vec_size + out_col + column
                        assert 0 <= index < len(outputs["y"]) - GUARD_COUNT, index
                        stores["y", index] += 1
                        outputs["y"][index] = _to_type(info["dtype"], total)
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

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QVM_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QVM_SHARD_COUNT"
CI_SHARD_COUNT = 4


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"qvm shard {index} is outside [0, {count})")
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


def _case_shapes(variant):
    shapes = {**SHAPES, **(GATHER_SHAPES if variant == "gather" else {})}
    return {
        case: dict(zip(("M", "groups", "K"), shapes[case])) for case in CASES[variant]
    }


def _expected_validation_contract():
    fast = _qmv_fast_validation_contract()
    return {
        "model": (
            "qvm_impl lane by lane after adjust_matrix_offsets: whole blocks "
            "and the last pass through the vec_w byte view and qouter, the "
            "zeroed lanes past K, the early return, simd_sum and lane 0's "
            "static_cast stores, against direct dequantized dot products"
        ),
        "packing": fast["packing"],
        "weights": "K rows of N codes, quantized along N",
        "comparison": fast["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / 64), B)",
        "n": "groups * group_size",
        "cases": {variant: _case_shapes(variant) for variant in CASES},
        "inputs": fast["inputs"],
        "exactness": fast["exactness"],
        "guards": fast["guards"],
        "softwareSubgroups": fast["softwareSubgroups"],
        "openglIndexRangeAssertions": fast["openglIndexRangeAssertions"],
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": fast["metalUpstreamComparison"],
    }


def _expected_scope():
    covered = (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + len(ENTRIES)
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
        "qmvWideGateEntryCount": QMV_WIDE_ENTRY_COUNT,
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
        "kind": "crosstl-mlx-current-tree-quantized-qvm-contract",
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


def test_current_mlx_quantized_qvm_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qvm_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qvm.current-tree.translation.json" in readme
    assert len(ENTRIES) == 162 == len(set(ENTRIES)) == QVM_ENTRY_COUNT
    assert Counter(info["variant"] for info in INFO.values()) == {
        "gather": 54,
        "batch_1": 54,
        "batch_0": 54,
    }
    assert ENTRIES[:3] == (
        "affine_gather_qvm_float_gs_128_b_2",
        "affine_qvm_float_gs_128_b_2_batch_1",
        "affine_qvm_float_gs_128_b_2_batch_0",
    )
    assert ENTRIES[-1] == "affine_qvm_bfloat16_t_gs_32_b_8_batch_0"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert not set(ENTRIES) & set(QMV_FAST_ENTRIES)
    assert set(CASES) == set(QMV_FAST_CASES) and set(SHAPES) == set(LAYOUTS)
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + len(ENTRIES)
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + GATHER_QMM_ENTRY_COUNT
        + QMV_QUAD_ENTRY_COUNT
        + 108
        == UNIT_ENTRY_COUNT
    )


def test_quantized_qvm_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [41, 41, 40, 40]
    for shard in shards:
        counts = Counter(INFO[entry]["variant"] for entry in shard)
        assert set(counts) == set(VARIANTS)
        assert max(counts.values()) - min(counts.values()) <= 1, counts
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_quantized_qvm_geometry_follows_qvm_impl():
    for bits in BITS:
        geometry = _geometry(bits)
        # Every width packs 32 codes, 4 * bits bytes, into one vec_w record.
        assert geometry["columns"] == COLUMNS
        assert geometry["recordBytes"] == 4 * bits
        # The record's offset of 32 columns is a whole number of W_T elements.
        offset = COLUMNS * geometry["bytesPerPack"] / geometry["packFactor"]
        assert offset.is_integer() and offset * geometry["wordBytes"] == 4 * bits


@pytest.mark.parametrize("entry", ENTRIES[::7])
def test_quantized_qvm_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n, k) == _shape(entry, case) and n % info["groupSize"] == 0
        # QuantizedMatmul and GatherQMM keep M below the vector limit.
        assert m < VECTOR_LIMIT
        if info["variant"] != "gather":
            # qvm, not qvm_split_k, and _batch_1 exactly when B > 1.
            assert k < SPLIT_K
            assert (batches > 1) == (info["variant"] == "batch_1")
        else:
            # gather_qmm_rhs needs M == 1 and B >= 16.
            assert batches < 16
        assert data["grid"] == (m, -(-n // (NUM_SIMDGROUPS * COLUMNS)), batches)
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        # (K, N) weights: K rows of N * bits / 32 words and N / group_size groups.
        assert params["w_shape"][-2:] == [k, n * info["bits"] // 32]
        assert layout["words"] == n * info["bits"] // 32
        assert layout["groups"] == n // info["groupSize"]
        assert params["s_strides"][-2:] == [layout["groups"], 1]


def test_quantized_qvm_cases_reach_every_path():
    paths = {variant: set() for variant in VARIANTS}
    exits = Counter()
    for entry in ENTRIES:
        for case in CASES[INFO[entry]["variant"]]:
            run = _simulate(entry, _case_data(entry, case))
            paths[INFO[entry]["variant"]] |= run["paths"]
            exits[INFO[entry]["groupSize"]] += "exit" in run["paths"]
    every = {"exit", "whole", "blocks-tail", "tail-only"}
    # Only unbatched entries run a K below one block.
    assert paths["batch_0"] == every
    assert paths["batch_1"] == paths["gather"] == every - {"tail-only"}
    # Only group size 32 allows an N that is not a multiple of 64.
    assert set(exits) == set(GROUP_SIZES) and exits[128] == exits[64] == 0
    assert exits[32] > 0


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qvm_model_matches_dot_products(entry):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        layout = data["layout"]
        m, k = layout["M"], layout["K"]
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word, scale and bias of the used matrices is read, and
        # no guard or unused activation row is.
        rows = {
            batch["x"] + row * k + index
            for batch in layout["batches"]
            for row in range(m)
            for index in range(k)
        }
        assert run["reads"]["x"] == rows
        matrices = {batch["w"] for batch in layout["batches"]}
        words, groups = layout["words"], layout["groups"]
        assert run["reads"]["w"] == {
            (matrix * k + row) * words + word
            for matrix in matrices
            for row in range(k)
            for word in range(words)
        }
        for name in ("scales", "biases"):
            assert run["reads"][name] == {
                (matrix * k + row) * groups + group
                for matrix in matrices
                for row in range(k)
                for group in range(groups)
            }


def test_quantized_qvm_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {"batch": 54}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qvm_inputs_keep_float32_exact(dtype):
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
                assert _f32(total) == total and value == value
                if value != total:
                    rounded += 1
                    neighbour = 2 * total - value
                    ties += _to_type(dtype, neighbour) == neighbour
    if dtype == "float32":
        assert rounded == 0
    else:
        assert rounded > 0 and ties > 0, (rounded, ties)


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    ],
)
def test_quantized_qvm_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qvm_opengl_assertions_name_buffer_elements(entry):
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
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[variant])


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qvm_configs_scope_target_options(tmp_path, target):
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
        REQUIRE_ENV, "quantized qvm", TARGETS, MLX_COMMIT
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
    assert single.index("affine_gather_qvm") == 4
    batched = re.findall(
        r"instantiate_quantized_batched_wrap\((\w+),",
        _macro(text, "instantiate_quantized_all_batched"),
    )
    assert batched[2] == "affine_qvm"
    wrap = _macro(text, "instantiate_quantized_batched_wrap")
    assert re.findall(r"bits, (\d)\)", wrap) == ["1", "0"]
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
        ("affine_qvm", QMV_PARAMETERS),
        ("affine_gather_qvm", GATHER_PARAMETERS),
    ):
        match = re.search(
            rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", header, re.DOTALL
        )
        assert match, kernel
        found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
        assert tuple((name, int(index)) for name, index in found) == parameters
    impl = header[header.index("METAL_FUNC void qvm_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int num_simdgroups = 2;",
        "constexpr int tn = 32 / pack_factor;",
        "constexpr int block_size = SIMD_SIZE;",
        "int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);",
        "if (out_col >= out_vec_size) {",
        "int remaining = in_vec_size % block_size;",
        "for (int i = block_size; i < in_vec_size; i += block_size) {",
        "if (static_cast<int>(simd_lid) < remaining) {",
        "result[k] = simd_sum(result[k]);",
    ):
        assert required in impl, required
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "int bn = std::min(group_size, 32) * num_simdgroups;",
        "MTL::Size group_dims(bk, num_simdgroups, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        'mode + "_qvm_",',
        'mode + "_gather_qvm_",',
        'B > 1 ? "_batch_1" : "_batch_0",',
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);",
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
        "if (K < 1024) {",
        "if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {",
    ):
        assert required in host, required
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qvm-metal-reference")
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


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_quantized_qvm_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-qvm-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = _translate(root, work, target, config, record)
            executor = _executor(target)
            evidence = []
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


def test_ci_requires_native_quantized_qvm_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized vector-matrix native parity",
        family="quantized-qvm",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qvm.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qvm.py",
    )
