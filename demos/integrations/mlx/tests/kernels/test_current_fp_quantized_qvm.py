"""Prove fp_quantized.metal's fp_qvm and fp_gather_qvm entries natively.

``quantized.cpp``'s ``qvm`` and ``gather_qvm`` multiply vectors by weights
that are not transposed, ``K`` rows of ``N`` codes quantized along ``N``:
``QuantizedMatmul`` picks ``qvm`` for ``M < 4`` and ``K < 1024``, and
``GatherQMM`` picks ``gather_qvm`` for ``M < 4`` outside ``gather_qmm_rhs``.
``fp_quantized.metal`` instantiates ``fp_qvm`` with and without batching and
``fp_gather_qvm``, each without and with a global scale (``_hgs``), for
nvfp4, mxfp8 and mxfp4 in float, bfloat16_t and float16_t: 54 of its 518
entries. Each is translated and pinned per target, then runs ``(32, 2, 1)``
threadgroups over an ``(M, ceil(N / (2 * group_size)), B)`` grid.

``fp_qvm_impl`` gives each simdgroup one group of output columns and each
lane one input row per block of 32: the lane reads its row's ``vec_w`` record
of ``group_size`` codes through a private byte view, and ``qouter`` adds
``x * scale * w`` into every result. A ``K`` that is a multiple of 32 runs
whole blocks; otherwise the blocks after the first run whole and a last pass
reads only where ``K`` reaches the lane, zeroing ``x`` and the scale of the
other lanes. A simdgroup whose first column lies past ``N`` returns before
``simd_sum``. ``QuantizedMatmul`` passes ``qvm`` no global scale, so the
``_hgs`` ``fp_qvm`` entries run with the global scale ``qvm`` itself binds at
buffer 2; ``gather_qvm`` takes ``GatherQMM``'s, one per expert. Inputs share
``fp_qmv_model``'s exactness argument, so only the global scale's product and
``static_cast<T>`` round. A lane-level model must reach direct dequantized
dot products, storing every column once; each target must produce its words
and leave every guard untouched, and on Metal the translation must also match
the unchanged upstream kernel word for word.
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
    declared_buffers,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    E4M3_CODE_FIELDS,
    GATHER_PARAMETERS,
    GLOBAL_SCALES,
    GUARD_COUNT,
    INDEX_BOUND,
    MODES,
    OPENGL_INDEX_RANGE_ASSERTIONS,
    QMV_PARAMETERS,
    _adjust,
    _batch_expert,
    _checked,
    _code_value,
    _experts,
    _inverse_scale,
    _parameters,
    _scale_value,
    bound_parameters,
    case_data_for,
    config_for,
    translate_entry,
)
from demos.integrations.mlx.tests.kernels.fp_quantized_scope import (
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
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    DTYPES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    ENTRIES as QUANTIZE_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    CI_SHARD_COUNT,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    ENTRIES as QMV_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    QMV_FP_HEADERS,
    REQUIRE_ENV,
    SHARD_COUNT_ENV,
    SHARD_INDEX_ENV,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    _expected_validation_contract as _qmv_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv_fast import (
    ENTRIES as QMV_FAST_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    NUM_SIMDGROUPS,
    SIMD_SIZE,
    WORKGROUP,
    _f32,
    _request,
    _upstream,
    layout_for,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/fp_quantized_qvm.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# Per type and mode, instantiate_quantized_all_batched's qvm wrap instantiates
# fp_qvm with batched 1, then 0, before instantiate_quantized_all_single's
# fp_gather_qvm; each comes without, then with, the _hgs global scale.
VARIANTS = ("batch_1", "batch_0", "gather")
KERNELS = {"batch_1": "fp_qvm", "batch_0": "fp_qvm", "gather": "fp_gather_qvm"}
# The input rows of one block: one per lane.
BLOCK = SIMD_SIZE

# Unbatched entries take each of fp_qvm_impl's paths; batched and gather
# entries combine them with adjust_matrix_offsets' layouts.
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
# For a K below one block, the lanes past K pass qouter a vec_w they never
# loaded, times zeros. Every E2M1 pattern is finite, so that adds zeros; an
# E4M3 pattern may be NaN, so mxfp8 cases keep K at least one block.
E4M3_SHORT_K_STEP = BLOCK
CASE_GLOBAL_SCALES = {
    "tail": 3000.0,
    "blocks": 10752.0,
    "ragged": 3000.0,
    "stacked": 5376.0,
    "broadcast": 3000.0,
    "shared": 10752.0,
}
# quantized.cpp's QuantizedMatmul routes K >= 1024 to qvm_split_k, and both
# QuantizedMatmul and GatherQMM route M >= 4 to qmm; GatherQMM sends sorted
# M == 1, B >= 16 batches to gather_qmm_rhs.
SPLIT_K = 1024
VECTOR_LIMIT = 4


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for variant in VARIANTS:
                for global_scale in (False, True):
                    name = "gather_qvm" if variant == "gather" else "qvm"
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


def _config(directory, target, entry):
    return config_for(directory, target, entry, INFO[entry])


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits, group_size):
    """fp_qvm_impl's constants: one record of a group's codes per lane."""
    # get_pack_factor<32, bits>(), and get_bytes_per_pack() with its default
    # wsize of 8: the record is tn uint32_t words.
    pack_factor = 32 // bits
    bytes_per_pack = 1
    tn = group_size // pack_factor
    return {
        "packFactor": pack_factor,
        "bytesPerPack": bytes_per_pack,
        "tn": tn,
        "recordWords": tn * bytes_per_pack,
        "columns": tn * pack_factor,
    }


def _tile(info):
    """qvm's and gather_qvm's bn: the columns of one threadgroup."""
    return min(info["groupSize"], 32) * NUM_SIMDGROUPS


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    info = INFO[entry]
    shapes = GATHER_SHAPES if info["variant"] == "gather" else {}
    m, groups, k = shapes.get(case, SHAPES[case])
    if info["bits"] == 8 and k < BLOCK:
        k += E4M3_SHORT_K_STEP
    return m, groups * info["groupSize"], k


def _layout(entry, case):
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, LAYOUTS[case], m, n, k, transposed=True)
    # fp_quantized.h's kernels have no biases, and bn follows the group size.
    del layout["params"]["b_strides"]
    layout["grid"] = (m, -(-n // _tile(info)), layout["grid"][2])
    return layout


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    layout, scale = _layout(entry, case), CASE_GLOBAL_SCALES.get(case)
    return case_data_for(INFO[entry], entry, case, layout, "fp-quantized-qvm", scale)


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    "order": "pairs each word's codes with its columns in reverse",
    "columns": "ignores the simdgroup's column offset",
    "group": "gives every column the first group's scale",
    "lane-row": "reads every lane's codes from its block's first row",
    "scale-step": "keeps the first block's scales",
    "tail": "drops the last pass after the whole blocks",
    "batch": "ignores the weight batch offsets",
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


def _unpack(bits, columns, byte, base, mutation):
    """qouter's codes of one record, in column order, through the byte view."""
    # The sign bit of the E4M3 byte or of the E2M1 nibble.
    sign = 0xFF if mutation != "sign" else 0x7F if bits == 8 else 0xF7
    codes = []
    if bits == 4:
        # Dequantize<4> of w[i] keeps its low nibble; w[i] >> 4 is the high.
        for i in range(columns // 2):
            value = byte(base + i)
            codes += [value & 0x0F & sign, value >> 4 & sign]
    else:
        codes = [byte(base + i) & sign for i in range(columns)]
    if mutation == "order":
        per_word = 32 // bits
        codes = [
            codes[start + per_word - 1 - i]
            for start in range(0, columns, per_word)
            for i in range(per_word)
        ]
    return codes


def _qouter(bits, codes, x, scale, result):
    """``result[i] += x * scale * Dequantize<bits>(w[i])``, in that order."""
    part = _checked(x * scale)
    for index, code in enumerate(codes):
        term = _checked(part * _code_value(bits, code))
        result[index] = _checked(result[index] + term)


def _lane(context, lane, out_col, mutation):
    """One lane's partial sums: the whole blocks, then the last pass."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    bits, group_size, mode = info["bits"], info["groupSize"], info["mode"]
    columns = geometry["columns"]
    x_base, w_base, s_base = context["bases"]
    in_vec_size, out_vec_size_w, out_vec_size_g = context["sizes"]
    group = 0 if mutation == "group" else out_col // group_size
    row = 0 if mutation == "lane-row" else lane
    # ws counts uint32_t words from w, which adjust_matrix_offsets moved.
    ws = w_base + out_col * geometry["bytesPerPack"] // geometry["packFactor"]
    ws += row * out_vec_size_w
    scales = s_base + group + lane * out_vec_size_g
    # x moves by in_vec_stride rows, which only fp_qvm_split_k sets apart.
    x = x_base + context["tid_x"] * context["stride"] + lane
    result = [0.0] * columns
    w_local = None

    def block():
        nonlocal ws, scales, x, w_local
        x_local = load("x", x)
        scale = _scale_value(mode, load("scales", scales))
        w_local = _unpack(bits, columns, context["byte"], ws * 4, mutation)
        _qouter(bits, w_local, x_local, scale, result)
        x += BLOCK
        if mutation != "scale-step":
            scales += BLOCK * out_vec_size_g
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
        scale = _scale_value(mode, load("scales", scales))
        w_local = _unpack(bits, columns, context["byte"], ws * 4, mutation)
    else:
        # The other lanes keep their last record and add zeros. A lane with
        # no record adds zeros only because every E2M1 pattern is finite.
        x_local = scale = 0.0
        if w_local is None:
            assert bits == 4, info
            w_local = [0] * columns
    _qouter(bits, w_local, x_local, scale, result)
    return result, reads


def _simulate(entry, data, mutation=None, info=None, sizes=None):
    """fp_qvm_impl lane by lane; returns outputs, stores, reads, offsets, paths.

    ``sizes(storage, z, mutation)`` gives ``tid.z``'s ``in_vec_size`` and
    ``in_vec_stride``, which fp_qvm and fp_gather_qvm pass as ``in_vec_size``
    twice.
    """
    info = info or INFO[entry]
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

    columns = geometry["columns"]
    grid_x, grid_y, grid_z = data["grid"]
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if info["variant"] != "batch_0":
            bases = _adjust(info, storage, z, mutation, records)
        expert, y_base = bases[3], bases[4]
        size = stride = in_vec_size
        if sizes is not None:
            size, stride = sizes(storage, z, mutation)
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:3],
                "tid_x": tid_x,
                "stride": stride,
                "sizes": (
                    size,
                    out_vec_size * geometry["bytesPerPack"] // geometry["packFactor"],
                    out_vec_size // info["groupSize"],
                ),
            }
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    tile = tid_y * NUM_SIMDGROUPS
                    if mutation != "columns":
                        tile += simd_gid
                    out_col = columns * tile
                    if out_col >= out_vec_size:
                        # A simdgroup-uniform return before simd_sum, and
                        # before the global scale is read.
                        paths.add("exit")
                        continue
                    factor = None
                    if info["globalScale"]:
                        factor = _inverse_scale(load("global_scale", expert))
                        if mutation == "global-scale":
                            factor = None
                    lanes, tails = [], 0
                    for lane in range(SIMD_SIZE):
                        result, tail = _lane(context, lane, out_col, mutation)
                        lanes.append(result)
                        tails += tail
                    if size % BLOCK == 0:
                        paths.add("whole")
                    else:
                        paths.add("blocks-tail" if size > BLOCK else "tail-only")
                        assert tails == size % BLOCK or mutation == "tail"
                    for column in range(columns):
                        # simd_sum: exact in any order; lane 0 stores
                        # static_cast<T>(sum * inv_scale_enc).
                        total = 0.0
                        for result in lanes:
                            total = _checked(total + result[column])
                        if factor is not None:
                            total = _f32(total * factor)
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


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"fp quantized qvm shard {index} is outside [0, {count})")
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
    qmv = _qmv_validation_contract()
    return {
        "model": (
            "fp_qvm_impl lane by lane after adjust_matrix_offsets: whole blocks "
            "and the last pass through the vec_w byte view and qouter, "
            "Dequantize's E2M1 and E4M3 decoding, dequantize_scale's E4M3 and "
            "E8M0 scales, the zeroed lanes past K, the early return, simd_sum, "
            "the global scale and lane 0's static_cast stores, against direct "
            "dequantized dot products"
        ),
        "packing": qmv["packing"],
        "weights": "K rows of N codes, quantized along N",
        "comparison": qmv["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / (2 * group_size)), B)",
        "n": "groups * group_size",
        "cases": {variant: _case_shapes(variant) for variant in CASES},
        "mxfp8ShortK": (
            f"K + {E4M3_SHORT_K_STEP} for a K below one block: the lanes past K "
            "would decode a vec_w they never loaded"
        ),
        "inputs": qmv["inputs"],
        "globalScales": {
            "qvm": dict(CASE_GLOBAL_SCALES),
            "gatherPerExpert": list(GLOBAL_SCALES),
        },
        "hostGlobalScale": (
            "QuantizedMatmul passes qvm none, so _hgs fp_qvm entries bind it "
            "at buffer 2 as qvm does; gather_qvm passes GatherQMM's"
        ),
        "exactness": qmv["exactness"],
        "guards": qmv["guards"],
        "softwareSubgroups": qmv["softwareSubgroups"],
        "globalScaleDivisionProfile": qmv["globalScaleDivisionProfile"],
        "openglIndexRangeAssertions": qmv["openglIndexRangeAssertions"],
        "unbatchedUpstreamBuffers": qmv["unbatchedUpstreamBuffers"],
        "metalUpstreamComparison": qmv["metalUpstreamComparison"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qvm"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qvm-contract",
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


def test_current_mlx_fp_quantized_qvm_contract_is_exact():
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


def test_fp_quantized_qvm_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qvm.current-tree.translation.json" in readme
    assert len(ENTRIES) == 54 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qvm"]
    assert Counter(info["variant"] for info in INFO.values()) == dict.fromkeys(
        VARIANTS, 18
    )
    assert Counter(info["globalScale"] for info in INFO.values()) == {
        False: 27,
        True: 27,
    }
    assert ENTRIES[:6] == (
        "nvfp4_qvm_float_gs_16_b_4_batch_1",
        "nvfp4_qvm_float_gs_16_b_4_batch_1_hgs",
        "nvfp4_qvm_float_gs_16_b_4_batch_0",
        "nvfp4_qvm_float_gs_16_b_4_batch_0_hgs",
        "nvfp4_gather_qvm_float_gs_16_b_4",
        "nvfp4_gather_qvm_float_gs_16_b_4_hgs",
    )
    assert ENTRIES[-1] == "mxfp4_gather_qvm_float16_t_gs_32_b_4_hgs"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    others = set(QUANTIZE_ENTRIES) | set(QMV_ENTRIES) | set(QMV_FAST_ENTRIES)
    assert not set(ENTRIES) & others


def test_fp_quantized_qvm_shards_partition_the_entries(monkeypatch):
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


def test_fp_quantized_qvm_geometry_follows_fp_qvm_impl():
    for mode, group_size, bits in MODES:
        geometry = _geometry(bits, group_size)
        # A lane's record holds one group of codes in whole uint32_t words.
        assert geometry["columns"] == group_size
        assert geometry["recordWords"] * 32 == group_size * bits, mode
        # qvm's bn covers the two simdgroups' groups.
        assert _tile({"groupSize": group_size}) == NUM_SIMDGROUPS * group_size


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qvm_cases_follow_mlx_dispatch(entry):
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
        if info["bits"] == 8:
            assert k >= BLOCK
        assert data["grid"] == (m, -(-n // _tile(info)), batches)
        # qvm's and gather_qvm's kernel names for this product.
        hgs = "_hgs" if info["globalScale"] else ""
        stem = f"{info['mode']}_{{}}_{info['itype']}_gs_{info['groupSize']}"
        stem += f"_b_{info['bits']}"
        if info["variant"] == "gather":
            assert entry == stem.format("gather_qvm") + hgs
        else:
            suffix = "_batch_1" if batches > 1 else "_batch_0"
            assert entry == stem.format("qvm") + suffix + hgs
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        assert "b_strides" not in params
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        # (K, N) weights: K rows of N * bits / 32 words and N / group_size groups.
        assert params["w_shape"][-2:] == [k, n * info["bits"] // 32]
        assert layout["words"] == n * info["bits"] // 32
        assert layout["groups"] == n // info["groupSize"]
        assert params["s_strides"][-2:] == [layout["groups"], 1]
        if info["globalScale"] and info["variant"] == "gather":
            assert len(values["global_scale"][1]) == _experts(layout) + GUARD_COUNT
            assert max(layout["rhs"]) < _experts(layout)


def test_fp_quantized_qvm_cases_reach_every_path():
    paths = {(variant, bits): set() for variant in VARIANTS for bits in (4, 8)}
    exits = Counter()
    for entry in ENTRIES:
        info = INFO[entry]
        for case in CASES[info["variant"]]:
            run = _simulate(entry, _case_data(entry, case))
            paths[info["variant"], info["bits"]] |= run["paths"]
            exits[info["groupSize"]] += "exit" in run["paths"]
    every = {"exit", "whole", "blocks-tail", "tail-only"}
    # Only unbatched E2M1 entries run a K below one block.
    assert paths["batch_0", 4] == every
    assert paths["batch_0", 8] == every - {"tail-only"}
    for variant in ("batch_1", "gather"):
        for bits in (4, 8):
            assert paths[variant, bits] == every - {"tail-only"}, variant
    # An odd number of groups returns the last tile's second simdgroup.
    assert exits[16] > 0 and exits[32] > 0


def test_fp_quantized_qvm_inputs_reach_every_code():
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
def test_fp_quantized_qvm_model_matches_dot_products(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        layout = data["layout"]
        m, k = layout["M"], layout["K"]
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word and scale of the used matrices is read, and no
        # guard or unused activation row is.
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
        assert run["reads"]["scales"] == {
            (matrix * k + row) * groups + group
            for matrix in matrices
            for row in range(k)
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


def test_fp_quantized_qvm_mutations_apply_where_they_are_defects():
    applied = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    )
    assert applied["batch"] == 36 and applied["global-scale"] == 27
    assert applied["expert-scale"] == 9
    for mutation in ("order", "columns", "group", "lane-row", "scale-step"):
        assert applied[mutation] == len(ENTRIES), mutation
    assert applied["tail"] == applied["sign"] == len(ENTRIES)


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qvm_inputs_keep_float32_exact(dtype):
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
                assert _f32(total) == total and value == value
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
def test_fp_quantized_qvm_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qvm_opengl_assertions_name_buffer_elements(entry):
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
def test_fp_quantized_qvm_configs_scope_target_options(tmp_path, target):
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


def _check_instantiation(text):
    """ENTRIES follow fp_quantized.metal's own instantiation macros."""
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
    assert batched[2] == "qvm"
    wrap = _macro(text, "instantiate_quantized_batched_wrap")
    assert re.findall(r"type, (\d), group_size", wrap) == ["1", "0"]
    single = re.findall(
        r"instantiate_quantized\(mode, (\w+),",
        _macro(text, "instantiate_quantized_all_single"),
    )
    assert single[2] == "gather_qvm"
    # Each instantiation adds its _hgs twin after it.
    for macro in ("instantiate_quantized", "instantiate_quantized_batched"):
        body = _macro(text, macro)
        assert body.count("instantiate_kernel(") == 2
        assert body.index('"_hgs"') > body.index("instantiate_kernel(", 1)


def _check_kernel(header):
    assert declared_buffers(header, "fp_qvm") == QMV_PARAMETERS
    assert declared_buffers(header, "fp_gather_qvm") == GATHER_PARAMETERS
    assert "template <int wsize = 8>\ninline constexpr short get_bytes_per_pack()" in (
        header
    )
    qouter = header[header.index("inline void qouter(") :]
    qouter = qouter[: qouter.index("\n}\n")]
    for required in (
        "result[2 * i] += x * scale * Dequantize<4>{}(w[i]);",
        "result[2 * i + 1] += x * scale * Dequantize<4>{}(w[i] >> 4);",
        "result[i] += x * scale * Dequantize<8>{}(w[i]);",
    ):
        assert required in qouter, required
    impl = header[header.index("METAL_FUNC void fp_qvm_impl(") :]
    impl = impl[: impl.index("template <", 1)]
    for required in (
        "constexpr int num_simdgroups = 2;",
        "constexpr int pack_factor = get_pack_factor<32, bits>();",
        "constexpr int bytes_per_pack = get_bytes_per_pack();",
        "constexpr int tn = group_size / pack_factor;",
        "constexpr int block_size = SIMD_SIZE;",
        "W_T wi[tn * bytes_per_pack];",
        "const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;",
        "int out_col = pack_factor * tn * (tid.y * num_simdgroups + simd_gid);",
        "ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;",
        "scales += out_col / group_size + simd_lid * out_vec_size_g;",
        "x += tid.x * in_vec_stride + simd_lid;",
        "if (out_col >= out_vec_size) {",
        "inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);",
        "int remaining = in_vec_size % block_size;",
        "for (int i = block_size; i < in_vec_size; i += block_size) {",
        "if (static_cast<int>(simd_lid) < remaining) {",
        "result[k] = simd_sum(result[k]);",
        "y[k] = static_cast<T>(result[k] * inv_scale_enc);",
    ):
        assert required in impl, required
    # The early return comes before the global scale is read.
    assert impl.index("if (out_col >= out_vec_size) {") < impl.index("*global_scale")
    for kernel, overload in (
        ("fp_qvm", "adjust_matrix_offsets(\n        x,\n        w,\n        scales,\n"),
        ("fp_gather_qvm", "adjust_matrix_offsets<has_global_scale>("),
    ):
        body = header[header.index(f"[[kernel]] void {kernel}(") :]
        body = body[: body.index("\n}\n")]
        assert overload in body, kernel
        assert "fp_qvm_impl<T, group_size, bits, has_global_scale>(" in body
    assert "global_scale += w_idx;" in header


def _check_host(host):
    for function in ("void qvm(", "void gather_qvm("):
        body = host[host.index(function) :]
        body = body[: body.index("\n}\n")]
        for required in (
            "int bn = std::min(group_size, 32) * num_simdgroups;",
            "MTL::Size group_dims(bk, num_simdgroups, 1);",
            "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
            'global_scale ? "_hgs" : "");',
            "compute_encoder.set_input_array(*global_scale, 2);",
        ):
            assert required in body, (function, required)
    for required in (
        'mode + "_qvm_",',
        'mode + "_gather_qvm_",',
        'B > 1 ? "_batch_1" : "_batch_0",',
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);",
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
        "if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {",
        # QuantizedMatmul hands qvm no global scale.
        "  // Run of the mill qvm\n  if (K < 1024) {\n    qvm(x,\n        w,\n"
        "        scales,\n        biases,\n        std::nullopt,",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qvm", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_FP_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    _check_instantiation(source.decode("utf-8"))
    kernels = root / "mlx/backend/metal/kernels/fp_quantized.h"
    _check_kernel(kernels.read_text(encoding="utf-8"))
    _check_host((root / HOST_SOURCE).read_text(encoding="utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-qvm-metal-reference")
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
def test_current_mlx_fp_quantized_qvm_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-fpqvm-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = translate_entry(root, work, target, config, record)
            parameters = bound_parameters(descriptor, INFO[entry])
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


def test_ci_requires_native_fp_quantized_qvm_parity():
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
        "pytest -q -n auto",
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qvm.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qvm.py",
    )
