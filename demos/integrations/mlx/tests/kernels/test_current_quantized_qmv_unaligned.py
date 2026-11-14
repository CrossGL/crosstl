"""Natively prove quantized.metal's 162 unaligned affine qmv kernels.

``quantized.cpp``'s ``qmv`` and ``gather_qmv`` pick ``affine_qmv`` and
``affine_gather_qmv`` instead of their ``_fast`` forms whenever ``N`` is not a
multiple of 8 or ``K`` is not a multiple of ``qmv_fast_k_alignment(bits)``.
``quantized.metal`` instantiates both for float, float16_t and bfloat16_t,
group sizes 128, 64 and 32 and 2, 3, 4, 5, 6 and 8 bits, ``affine_qmv`` with
and without batching: 162 of its 2,052 entries. Each is translated and
pinned per target, then runs the same ``(32, 2, 1)`` threadgroups over a
``(M, ceil(N / 8), B)`` grid.

``qmv_impl`` reads one pack per lane and splits ``K`` into whole blocks and a
tail that ``load_vector_safe`` and ``qdot_safe`` read only where it reaches
the lane. Fewer than eight output rows take its guarded branch, where a
simdgroup past the last row returns before ``simd_sum``; otherwise the last
tile is moved back to end on row ``N - 1`` and recomputes rows another
simdgroup stores. The cases reach every one of these paths, whole tiles
with a misaligned ``K`` and both branches of ``adjust_matrix_offsets``,
with the inputs of the ``qmv_fast`` gate, so float32 holds every
intermediate exactly and only the final ``static_cast<T>`` rounds. A model
replays each lane and must reach direct dequantized dot products, storing
every row at least once and moved rows twice with the same value. Each
target must produce the model's words and leave every guard untouched; on
Metal the translation must also match the unchanged upstream kernel word for
word.
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
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    MUTATIONS as QMV_FAST_MUTATIONS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
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
    RESULTS_PER_SIMDGROUP,
    ROWS_PER_GROUP,
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
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import _f32
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _geometry as _qmv_fast_geometry,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _load_vector,
    _qdot,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QMV_UNALIGNED"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_qmv_unaligned.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNELS = {
    "gather": "affine_gather_qmv",
    "batch_1": "affine_qmv",
    "batch_0": "affine_qmv",
}

# Unbatched entries take each of qmv_impl's paths; batched and gather entries
# combine them with adjust_matrix_offsets' layouts.
CASES = {
    "batch_0": ("narrow", "short", "ragged", "deep"),
    "batch_1": ("stacked", "broadcast", "shared"),
    "gather": ("experts", "broadcast"),
}
# (M, N, whole blocks, extra groups): K starts at blocks * block_size +
# groups * group_size; layouts as the qmv_fast gate builds them.
SHAPES = {
    # Five rows: the guarded branch, the second simdgroup on one row.
    "narrow": (1, 5, 1, 1),
    # Three rows: the second simdgroup returns before simd_sum.
    "short": (2, 3, 0, 3),
    # Thirteen rows: the last tile moves back four rows minus the remainder.
    "ragged": (2, 13, 2, 1),
    # Whole tiles, with K off qmv_fast_k_alignment.
    "deep": (1, 16, 1, 1),
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
# K values dispatch_qmv sends to qmv_quad for power-of-two widths.
QUAD_K = (64, 128)


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


def _geometry(bits, group_size):
    """qmv_impl's constants: one pack per lane for every width."""
    fast = _qmv_fast_geometry(bits, group_size)
    pack_factor = fast["packFactor"]
    return {
        "packFactor": pack_factor,
        "bytesPerPack": fast["bytesPerPack"],
        "valuesPerThread": pack_factor,
        "blockSize": pack_factor * SIMD_SIZE,
        "scaleStepPerThread": group_size // pack_factor,
        # qmv_fast_k_alignment(bits).
        "fastAlignment": pack_factor * (1 if bits == 2 else 2) * SIMD_SIZE,
    }


def _routes_to_qmv(info, n, k):
    """Whether quantized.cpp dispatches ``qmv_impl`` for an ``N x K`` weight."""
    geometry = _geometry(info["bits"], info["groupSize"])
    if n % ROWS_PER_GROUP == 0 and k % geometry["fastAlignment"] == 0:
        return False
    power_of_2_bits = info["bits"] & (info["bits"] - 1) == 0
    # gather_qmv has no qmv_quad route.
    return info["variant"] == "gather" or not (k in QUAD_K and power_of_2_bits)


def _shape(entry, case):
    """``(M, N, K)``: K grows by a group until quantized.cpp picks qmv."""
    info = INFO[entry]
    m, n, blocks, groups = SHAPES[case]
    geometry = _geometry(info["bits"], info["groupSize"])
    k = blocks * geometry["blockSize"] + groups * info["groupSize"]
    while not _routes_to_qmv(info, n, k):
        k += info["groupSize"]
    return m, n, k


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    layout = layout_for(info, LAYOUTS[case], *_shape(entry, case))
    return case_data_for(info, entry, case, layout)


def _moved_rows(n):
    """Rows the moved-back last tile stores a second time, per output row block.

    With ``r = N - base`` rows in the last tile at ``base``: for ``r <= 4``
    its second simdgroup returns and its first starts at ``N - 4``, inside
    the previous tile's second simdgroup; otherwise the second simdgroup
    starts at ``N - 4``, inside the first.
    """
    if n < ROWS_PER_GROUP or n % ROWS_PER_GROUP == 0:
        return set()
    base = n - n % ROWS_PER_GROUP
    end = base if n - base <= RESULTS_PER_SIMDGROUP else base + RESULTS_PER_SIMDGROUP
    return set(range(n - RESULTS_PER_SIMDGROUP, end))


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    **QMV_FAST_MUTATIONS,
    "tail": "drops the safe tail after the whole blocks",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "batch":
        return info["variant"] != "batch_0"
    if mutation == "scale-step":
        # With one pack of 4 per lane and groups of 128, a block is one group.
        step = _geometry(info["bits"], info["groupSize"])["scaleStepPerThread"]
        return step < SIMD_SIZE
    return True


def _lane(context, lane, first, rows, mutation):
    """One lane's per-row partial sums over the whole blocks and the tail."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    dtype, bits, group_size = info["dtype"], info["bits"], info["groupSize"]
    x_base, w_base, s_base, b_base = context["bases"]
    per_thread, block = geometry["valuesPerThread"], geometry["blockSize"]
    in_vec_size, in_vec_size_w, in_vec_size_g = context["sizes"]
    step = geometry["scaleStepPerThread"]
    group = 0 if mutation == "scale-step" else lane // step
    ws = w_base * 4 + first * in_vec_size_w + lane * geometry["bytesPerPack"]
    scales = s_base + first * in_vec_size_g + group
    biases = b_base + first * in_vec_size_g + group
    x = x_base + context["tid_x"] * in_vec_size + lane * per_thread
    result = [0.0] * RESULTS_PER_SIMDGROUP

    def accumulate(count):
        values = [load("x", x + i) for i in range(count)]
        if mutation == "order":
            pack = geometry["packFactor"]
            values = [
                values[start + pack - 1 - i]
                for start in range(0, count, pack)
                for i in range(pack)
            ]
        x_thread, total = _load_vector(dtype, bits, values)
        for row in rows:
            scale = load("scales", scales + row * in_vec_size_g)
            bias = load("biases", biases + row * in_vec_size_g)
            result[row] = _checked(
                result[row]
                + _qdot(
                    bits,
                    context["byte"],
                    ws + row * in_vec_size_w,
                    x_thread,
                    scale,
                    bias,
                    total,
                    mutation,
                )
            )

    k = iterations = 0
    while k < in_vec_size - block:
        accumulate(per_thread)
        ws += block * geometry["bytesPerPack"] // geometry["packFactor"]
        scales += block // group_size
        biases += block // group_size
        x += block
        k += block
        iterations += 1
    remaining = min(max(in_vec_size - k - lane * per_thread, 0), per_thread)
    # K is a multiple of group_size, so of 32 and every pack factor: a lane's
    # tail is whole or empty, and load_vector_safe never zero-fills.
    assert remaining in (0, per_thread), remaining
    if remaining and mutation != "tail":
        accumulate(remaining)
    return result, iterations, remaining > 0


def _simulate(entry, data, mutation=None):
    """qmv_impl lane by lane; returns outputs, stores, reads, offsets and paths."""
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
            bases = _adjust(info["variant"], storage, z, mutation, records)
        y_base = bases[4]
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:4],
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
                        # simd_sum: exact in any order; lane 0 stores T(sum).
                        total = 0.0
                        for result in lanes:
                            total = _checked(total + result[row])
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

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_UNALIGNED_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_UNALIGNED_SHARD_COUNT"
CI_SHARD_COUNT = 4


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"unaligned qmv shard {index} is outside [0, {count})")
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
    fast = _qmv_fast_validation_contract()
    return {
        "model": (
            "qmv_impl lane by lane after adjust_matrix_offsets: whole blocks "
            "through load_vector and qdot, the lane's tail through "
            "load_vector_safe and qdot_safe, the guarded rows, the early "
            "return and the moved-back last tile, simd_sum and lane 0's "
            "static_cast stores, against direct dequantized dot products"
        ),
        "packing": fast["packing"],
        "comparison": fast["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / 8), B)",
        "k": (
            "blocks * 32 * pack_factor + groups * group_size, plus group_size "
            "until quantized.cpp picks qmv over qmv_fast and qmv_quad"
        ),
        "cases": {
            variant: {
                case: dict(zip(("M", "N", "blocks", "groups"), SHAPES[case]))
                for case in cases
            }
            for variant, cases in CASES.items()
        },
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
        + len(ENTRIES)
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
        "qmvFastGateEntryCount": len(QMV_FAST_ENTRIES),
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
        "kind": "crosstl-mlx-current-tree-quantized-qmv-unaligned-contract",
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


def test_current_mlx_quantized_qmv_unaligned_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qmv_unaligned_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qmv_unaligned.current-tree.translation.json" in readme
    assert len(ENTRIES) == 162 == len(set(ENTRIES)) == QMV_UNALIGNED_ENTRY_COUNT
    assert Counter(info["variant"] for info in INFO.values()) == {
        "gather": 54,
        "batch_1": 54,
        "batch_0": 54,
    }
    assert ENTRIES[:3] == (
        "affine_gather_qmv_float_gs_128_b_2",
        "affine_qmv_float_gs_128_b_2_batch_1",
        "affine_qmv_float_gs_128_b_2_batch_0",
    )
    assert ENTRIES[-1] == "affine_qmv_bfloat16_t_gs_32_b_8_batch_0"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert not set(ENTRIES) & set(QMV_FAST_ENTRIES)
    assert set(CASES) == set(QMV_FAST_CASES) and set(SHAPES) == set(LAYOUTS)
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + len(ENTRIES)
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


def test_quantized_qmv_unaligned_shards_partition_the_entries(monkeypatch):
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


@pytest.mark.parametrize("entry", ENTRIES[::7])
def test_quantized_qmv_unaligned_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"])
    fast = _qmv_fast_geometry(info["bits"], info["groupSize"])
    # qmv_fast_impl's K step, and qmv_impl's packs_per_thread = 1.
    assert geometry["fastAlignment"] == fast["blockSize"]
    assert geometry["blockSize"] * (1 if info["bits"] == 2 else 2) == fast["blockSize"]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n) == SHAPES[case][:2] and k % info["groupSize"] == 0
        # Not qmv_fast: N % bn != 0 or K % qmv_fast_k_alignment(bits) != 0.
        assert n % ROWS_PER_GROUP or k % geometry["fastAlignment"]
        if info["variant"] != "gather":
            # dispatch_qmv's qmv_quad route, and _batch_1 exactly when B > 1.
            assert k not in QUAD_K or info["bits"] in (3, 5, 6)
            assert (batches > 1) == (info["variant"] == "batch_1")
        else:
            # gather_qmm_rhs needs B >= 16; M < 4 keeps gather_qmm away.
            assert batches < 16 and m < 4
        assert data["grid"] == (m, -(-n // ROWS_PER_GROUP), batches)
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]


def test_quantized_qmv_unaligned_cases_reach_every_path():
    paths = {variant: set() for variant in VARIANTS}
    for entry in ENTRIES:
        for case in CASES[INFO[entry]["variant"]]:
            run = _simulate(entry, _case_data(entry, case))
            paths[INFO[entry]["variant"]] |= run["paths"]
    every = {
        "exit",
        "guarded",
        "moved",
        "whole",
        "blocks",
        "tail-only",
        "full-tail",
        "partial-tail",
    }
    # Only unbatched entries run a K below one block.
    assert paths["batch_0"] == every
    assert paths["batch_1"] == paths["gather"] == every - {"tail-only"}


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qmv_unaligned_model_matches_dot_products(entry):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        layout = data["layout"]
        m, n = layout["M"], layout["N"]
        twice = _moved_rows(n)
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        for index in range(written):
            row = index % n
            assert run["stores"]["y", index] == 1 + (row in twice), (case, index)
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word, scale and bias of the used matrices is read, and
        # no guard or unused activation row is.
        rows = {
            batch["x"] + row * layout["K"] + index
            for batch in layout["batches"]
            for row in range(m)
            for index in range(layout["K"])
        }
        assert run["reads"]["x"] == rows
        matrices = {batch["w"] for batch in layout["batches"]}
        words, groups = layout["words"], layout["groups"]
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


def test_quantized_qmv_unaligned_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    # 54 unbatched entries, and 2 widths x 3 types x 3 variants at group 128.
    assert skipped == {"batch": 54, "scale-step": 18}


def test_quantized_qmv_unaligned_moved_rows_follow_the_tiles():
    assert _moved_rows(5) == _moved_rows(16) == set()
    assert _moved_rows(13) == {9, 10, 11}
    assert _moved_rows(11) == {7}
    assert _moved_rows(12) == set() and _moved_rows(9) == {5, 6, 7}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qmv_unaligned_inputs_keep_float32_exact(dtype):
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
def test_quantized_qmv_unaligned_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_qmv_unaligned_opengl_assertions_name_buffer_elements(entry):
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
def test_quantized_qmv_unaligned_configs_scope_target_options(tmp_path, target):
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
        REQUIRE_ENV, "unaligned quantized qmv", TARGETS, MLX_COMMIT
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
    assert single.index("affine_gather_qmv") == 3
    batched = re.findall(
        r"instantiate_quantized_batched_wrap\((\w+),",
        _macro(text, "instantiate_quantized_all_batched"),
    )
    assert batched[1] == "affine_qmv"
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
        ("affine_qmv", QMV_PARAMETERS),
        ("affine_gather_qmv", GATHER_PARAMETERS),
    ):
        match = re.search(
            rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", header, re.DOTALL
        )
        assert match, kernel
        found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
        assert tuple((name, int(index)) for name, index in found) == parameters
    impl = header[header.index("METAL_FUNC void qmv_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int packs_per_thread = 1;",
        "const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);",
        "if (out_row >= out_vec_size) {",
        "if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {",
        "for (; k < in_vec_size - block_size; k += block_size) {",
        "U sum = load_vector_safe<T, U, values_per_thread, bits>(",
        "result[row] += qdot_safe<U, values_per_thread, bits>(",
    ):
        assert required in impl, required
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "int bk = 32;",
        "MTL::Size group_dims(bk, 2, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        "bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;",
        "return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;",
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) {",
        "if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) {",
        'mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),',
        'B > 1 ? "_batch_1" : "_batch_0"',
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
    ):
        assert required in host, required
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qmv-unaligned-metal-reference")
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
def test_current_mlx_quantized_qmv_unaligned_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-qmvu-", dir=root) as path:
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


def test_ci_requires_native_quantized_qmv_unaligned_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX unaligned quantized matrix-vector native parity",
        family="quantized-qmv-unaligned",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv_unaligned.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv_unaligned.py",
    )
