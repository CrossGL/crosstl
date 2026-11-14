"""Natively prove quantized.metal's 108 affine qvm split-K kernels.

``quantized.cpp``'s ``qvm`` routes vectors with ``K`` of 1,024 or more
against weights that are not transposed, ``K`` rows of ``N`` codes, to
``qvm_split_k``. It splits ``K`` into ``split_k`` parts of
``split_D = ceil(K / split_k)`` rows, the last of
``final_block_size = K - (split_k - 1) * split_D``, adds a batch dimension of
``split_k`` to the activation, weight, scale and bias views and lets each
``tid.z`` write one part's partial products to an intermediate that a
reduction then sums. ``quantized.metal`` instantiates ``affine_qvm_split_k``
for float, float16_t and bfloat16_t, group sizes 128, 64 and 32, 2, 3, 4, 5,
6 and 8 bits and ``split_k`` 8 and 32: 108 of its 2,052 entries. Each is
translated and pinned per target, then runs ``(32, 2, 1)`` threadgroups over
an ``(M, ceil(N / 64), B * split_k)`` grid.

The kernel runs ``qvm_impl`` with the part's size as ``in_vec_size`` and the
whole ``K`` as the activation row stride, so the qvm gate's lane-level model
replays it part by part. The cases bind the host's own views, split sizes
and final block sizes; they reach a part made of whole blocks, blocks and a
last pass, a last pass alone and a final part of another size, with one or
more rows, batched weights and broadcast weights. ``quantized.cpp`` picks
``split_k`` 32 only above ``K = 8,192``; those entries run the same formulas
at smaller ``K``, which the kernel cannot tell apart. Each target must
produce the model's words and leave every guard untouched; on Metal the
translation must also match the unchanged upstream kernel word for word.
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
    ARRAY_TYPES,
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
    SCALAR_TYPES,
    SENTINEL,
    SIMD_SIZE,
    WORKGROUP,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _config as _qmv_config,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _exact_dyadic,
    _f32,
    _request,
    _translate,
    _upstream,
    case_data_for,
    layout_for,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qvm import (
    BLOCK,
    COLUMNS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qvm import (
    ENTRIES as QVM_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qvm import (
    MUTATIONS as QVM_MUTATIONS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qvm import (
    _expected_validation_contract as _qvm_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qvm import (
    _simulate as _qvm_simulate,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QVM_SPLIT_K"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/"
    "quantized_qvm_split_k.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "affine_qvm_split_k"
SPLIT_KS = (8, 32)
# The kernel's buffers: qvm's batched parameters and the final part's size.
PARAMETERS = (*QMV_PARAMETERS, ("final_block_size", 15))
# quantized.cpp's thresholds: qvm below K = 1024, split_k 32 above 8192.
SPLIT_K_THRESHOLD = 1024
WIDE_SPLIT_THRESHOLD = 8192
VECTOR_LIMIT = 4

# (M, N in groups, K) and the activation and weight layout of each case.
SHAPES = {
    # A K quantized.cpp routes here: four blocks and a last pass per part.
    "routed": (1, 1, 1100),
    # Parts of one whole block.
    "whole": (1, 2, 256),
    # Parts of a block and a three-lane last pass, then a whole final part.
    "ragged": (2, 3, 277),
    # Parts of a last pass alone, 13 and then 9 lanes.
    "tail": (1, 1, 100),
    "stacked": (1, 1, 300),
    "broadcast": (1, 1, 200),
    # One whole block per part.
    "whole-32": (1, 1, 1024),
    # A block and a last pass, then a final last pass of 17 lanes.
    "ragged-32": (2, 1, 1040),
    "stacked-32": (1, 1, 315),
    "broadcast-32": (2, 2, 95),
}
LAYOUTS = {
    "routed": "unbatched",
    "whole": "unbatched",
    "ragged": "unbatched",
    "tail": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "whole-32": "unbatched",
    "ragged-32": "unbatched",
    "stacked-32": "stacked",
    "broadcast-32": "broadcast",
}
# Cheaper cases first: mutations stop at the first case that exposes them.
CASES = {
    8: ("tail", "whole", "stacked", "broadcast", "ragged", "routed"),
    32: ("broadcast-32", "stacked-32", "whole-32", "ragged-32"),
}
MUTATIONS = {
    **QVM_MUTATIONS,
    "final": "gives every part the final part's size",
    "stride": "steps activation rows by the part's size, not by K",
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for split_k in SPLIT_KS:
                    entry = f"{KERNEL}_{itype}_gs_{group_size}_b_{bits}_spk_{split_k}"
                    yield entry, {
                        "kernel": KERNEL,
                        "variant": "split_k",
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                        "splitK": split_k,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    """The translation of one entry, with the batched qmv_fast gate's options."""
    return _qmv_config(directory, target, entry, "batch_1")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _split_sizes(k, split_k):
    """quantized.cpp's ``(split_D, final_block_size)``."""
    split_d = -(-k // split_k)
    return split_d, k - (split_k - 1) * split_d


def _split_params(params, split_k, split_d, words, groups):
    """qvm_split_k's views: a split_k batch dimension before the matrices."""
    x_shape, x_strides = list(params["x_shape"]), list(params["x_strides"])
    x_ndim = len(x_shape)
    x_shape.insert(x_ndim - 2, split_k)
    x_shape[-1] //= split_k
    x_strides.insert(x_ndim - 2, split_d)
    # quantized.cpp overwrites the row stride, which the kernel never reads.
    x_strides[x_ndim - 1] = split_d
    w_shape = list(params["w_shape"])
    w_ndim = len(w_shape)
    w_shape.insert(w_ndim - 2, split_k)
    w_shape[w_ndim - 1] //= split_k

    def strides(name, width):
        values = list(params[name])
        values.insert(len(values) - 2, split_d * width)
        return values

    return {
        "x_batch_ndims": x_ndim - 1,
        "x_shape": x_shape,
        "x_strides": x_strides,
        "w_batch_ndims": w_ndim - 1,
        "w_shape": w_shape,
        "w_strides": strides("w_strides", words),
        "s_strides": strides("s_strides", groups),
        "b_strides": strides("b_strides", groups),
    }


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The qvm inputs of one case, bound through qvm_split_k's views."""
    info = INFO[entry]
    assert case in CASES[info["splitK"]], (entry, case)
    m, groups, k = SHAPES[case]
    n = groups * info["groupSize"]
    batched = {**info, "variant": "batch_1"}
    layout = layout_for(batched, LAYOUTS[case], m, n, k, transposed=True)
    data = case_data_for(batched, entry, case, layout)
    split_k = info["splitK"]
    split_d, final = _split_sizes(k, split_k)
    assert 0 < final <= split_d, (case, split_k)
    params = _split_params(
        layout["params"], split_k, split_d, layout["words"], layout["groups"]
    )
    parts = len(layout["batches"]) * split_k
    initial = [SENTINEL] * (parts * m * n + GUARD_COUNT)
    values = dict(data["values"])
    values.update(
        {
            "y": (info["dtype"], initial),
            "in_vec_size": ("int32", [split_d]),
            "final_block_size": ("int32", [final]),
        }
    )
    for name, value in params.items():
        types = SCALAR_TYPES if name in SCALAR_TYPES else ARRAY_TYPES
        values[name] = (types[name], [value] if isinstance(value, int) else value)
    assert set(values) == {name for name, _ in PARAMETERS}, entry
    split = {
        **data,
        "layout": {
            **layout,
            "params": params,
            "splitK": split_k,
            "splitD": split_d,
            "finalBlockSize": final,
            "grid": (m, layout["grid"][1], parts),
        },
        "values": values,
        "initial": {"y": initial},
        "grid": (m, layout["grid"][1], parts),
    }
    split["expected"], split["totals"], split["budget"] = _partials(info, split)
    return split


def _partials(info, data):
    """Each part's dequantized partial dot products, as the intermediate holds."""
    dtype, group_size = info["dtype"], info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    split_k, split_d = layout["splitK"], layout["splitD"]
    x = data["values"]["x"][1]
    scales, biases = data["values"]["scales"][1], data["values"]["biases"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    for b, batch in enumerate(layout["batches"]):
        for part in range(split_k):
            start = part * split_d
            stop = k if part == split_k - 1 else start + split_d
            z = b * split_k + part
            for row in range(m):
                vector = x[batch["x"] + row * k : batch["x"] + (row + 1) * k]
                for col in range(n):
                    total, bound = 0.0, 0.0
                    for index in range(start, stop):
                        line = batch["w"] * k + index
                        code = data["codes"][line][col]
                        group = line * groups + col // group_size
                        scale, bias = scales[group], biases[group]
                        total += (scale * code + bias) * vector[index]
                        bound += abs(vector[index]) * (abs(scale) * code + abs(bias))
                    assert _exact_dyadic(total) and bound < 2**19, (info, total)
                    budget = max(budget, bound)
                    index = (z * m + row) * n + col
                    totals[index] = total
                    expected[index] = _to_type(dtype, total)
    return {"y": expected}, totals, budget


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all: all of them are."""
    return mutation in MUTATIONS


def _sizes(split_k):
    """qvm_split_k's ``in_vec_size_adj`` and ``in_vec_stride`` for ``tid.z``."""

    def sizes(storage, z, mutation):
        split_d, final = storage["in_vec_size"][0], storage["final_block_size"][0]
        size = final if z % split_k == split_k - 1 else split_d
        if mutation == "final":
            size = final
        stride = (split_k - 1) * split_d + final
        if mutation == "stride":
            stride = split_d
        return size, stride

    return sizes


def _simulate(entry, data, mutation=None):
    """The kernel lane by lane: qvm_impl on each tid.z's part."""
    info = INFO[entry]
    qvm_mutation = mutation if mutation in QVM_MUTATIONS else None
    sizes = _sizes(info["splitK"])

    def split_sizes(storage, z, _mutation):
        return sizes(storage, z, mutation)

    return _qvm_simulate(entry, data, qvm_mutation, info=info, sizes=split_sizes)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QVM_SPLIT_K_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QVM_SPLIT_K_SHARD_COUNT"
CI_SHARD_COUNT = 4


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"qvm split-K shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


# A shard runs the per-entry model, defect and parity tests of its own
# entries; the four shards of each platform cover every entry between them.
SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


def _dispatch_count():
    return sum(len(CASES[INFO[entry]["splitK"]]) for entry in ENTRIES)


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        len(_case_data(entry, case)["expected"]["y"])
        for entry in ENTRIES
        for case in CASES[INFO[entry]["splitK"]]
    )


def _case_shapes(split_k):
    cases = {}
    for case in CASES[split_k]:
        m, groups, k = SHAPES[case]
        split_d, final = _split_sizes(k, split_k)
        cases[case] = {
            "M": m,
            "groups": groups,
            "K": k,
            "layout": LAYOUTS[case],
            "splitD": split_d,
            "finalBlockSize": final,
        }
    return cases


def _expected_validation_contract():
    qvm = _qvm_validation_contract()
    return {
        "model": (
            "qvm_impl lane by lane on each tid.z's part after "
            "adjust_matrix_offsets over the split views: split_D rows, or "
            "final_block_size for the last part, with K as the row stride, "
            "against direct dequantized partial dot products"
        ),
        "packing": qvm["packing"],
        "weights": qvm["weights"],
        "split": (
            "split_D = ceil(K / split_k), final_block_size = K - (split_k - 1) "
            "* split_D; quantized.cpp's views add a split_k batch dimension"
        ),
        "comparison": qvm["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / 64), B * split_k)",
        "n": qvm["n"],
        "cases": {str(split_k): _case_shapes(split_k) for split_k in SPLIT_KS},
        "inputs": qvm["inputs"],
        "exactness": qvm["exactness"],
        "guards": qvm["guards"],
        "softwareSubgroups": qvm["softwareSubgroups"],
        "openglIndexRangeAssertions": qvm["openglIndexRangeAssertions"],
        "metalUpstreamComparison": qvm["metalUpstreamComparison"],
    }


def _expected_scope():
    covered = (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + len(ENTRIES)
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
        "qvmGateEntryCount": QVM_ENTRY_COUNT,
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
        "kind": "crosstl-mlx-current-tree-quantized-qvm-split-k-contract",
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


def test_current_mlx_quantized_qvm_split_k_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qvm_split_k_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qvm_split_k.current-tree.translation.json" in readme
    assert len(ENTRIES) == 108 == len(set(ENTRIES)) == QVM_SPLIT_K_ENTRY_COUNT
    assert Counter(info["splitK"] for info in INFO.values()) == {8: 54, 32: 54}
    assert ENTRIES[:2] == (
        "affine_qvm_split_k_float_gs_128_b_2_spk_8",
        "affine_qvm_split_k_float_gs_128_b_2_spk_32",
    )
    assert ENTRIES[-1] == "affine_qvm_split_k_bfloat16_t_gs_32_b_8_spk_32"
    assert KERNEL not in EXCLUDED_KERNELS
    assert not set(ENTRIES) & (set(QMV_FAST_ENTRIES) | set(QVM_ENTRIES))
    assert set(SHAPES) == set(LAYOUTS) == {c for cases in CASES.values() for c in cases}
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + len(ENTRIES)
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + GATHER_QMM_ENTRY_COUNT
        + QMV_QUAD_ENTRY_COUNT
        + 108
        == UNIT_ENTRY_COUNT
    )


def test_quantized_qvm_split_k_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [27, 27, 27, 27]
    # The two split sizes alternate, so the even shards run split_k 8 and the
    # odd shards split_k 32, each over half of the dtypes, groups and bits.
    for index, shard in enumerate(shards):
        counts = Counter(INFO[entry]["splitK"] for entry in shard)
        assert counts == {SPLIT_KS[index % 2]: 27}, counts
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_quantized_qvm_split_k_cases_follow_mlx_dispatch():
    for split_k in SPLIT_KS:
        for case in CASES[split_k]:
            m, groups, k = SHAPES[case]
            split_d, final = _split_sizes(k, split_k)
            # The last part is never empty and never longer than the others.
            assert 0 < final <= split_d
            assert (split_k - 1) * split_d + final == k
            assert m < VECTOR_LIMIT
    # The routed case is one quantized.cpp sends to split_k 8 itself.
    assert SPLIT_K_THRESHOLD <= SHAPES["routed"][2] <= WIDE_SPLIT_THRESHOLD
    assert all(SHAPES[case][2] <= WIDE_SPLIT_THRESHOLD for case in CASES[32])


@pytest.mark.parametrize("entry", ENTRIES[::9])
def test_quantized_qvm_split_k_views_follow_quantized_cpp(entry):
    info = INFO[entry]
    split_k = info["splitK"]
    for case in CASES[split_k]:
        data = _case_data(entry, case)
        layout = data["layout"]
        params = layout["params"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        split_d, words = layout["splitD"], layout["words"]
        assert n % info["groupSize"] == 0
        assert data["grid"] == (
            m,
            -(-n // (NUM_SIMDGROUPS * COLUMNS)),
            batches * split_k,
        )
        assert len(data["expected"]["y"]) == batches * split_k * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [split_d]
        assert values["out_vec_size"][1] == [n]
        assert values["final_block_size"][1] == [layout["finalBlockSize"]]
        # The split dimension precedes the matrices of every view.
        assert params["x_shape"][-3:] == [split_k, m, k // split_k]
        assert params["x_strides"][-3:] == [split_d, split_d, 1]
        assert params["w_shape"][-3:] == [split_k, k // split_k, words]
        assert params["w_strides"][-3:] == [split_d * words, words, 1]
        assert params["s_strides"][-3:] == [
            split_d * layout["groups"],
            layout["groups"],
            1,
        ]
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2


def _expected_paths(split_k, group_size, case):
    """The qvm_impl paths a case's parts take, from its shape alone.

    A part of ``size`` rows runs whole blocks when ``size`` is a multiple of
    the block, blocks and a last pass when it is longer, and a last pass
    alone otherwise. A simdgroup whose first column is at or past ``N``
    returns early.
    """
    m, groups, k = SHAPES[case]
    n = groups * group_size
    split_d, final = _split_sizes(k, split_k)
    paths = set()
    for size in {split_d, final}:
        if size % BLOCK == 0:
            paths.add("whole")
        else:
            paths.add("blocks-tail" if size > BLOCK else "tail-only")
    tiles = -(-n // (NUM_SIMDGROUPS * COLUMNS)) * NUM_SIMDGROUPS
    if COLUMNS * (tiles - 1) >= n:
        paths.add("exit")
    return paths


def test_quantized_qvm_split_k_cases_reach_every_path():
    # The model test checks each case's paths against this table.
    every = {"exit", "whole", "blocks-tail", "tail-only"}
    for split_k in SPLIT_KS:
        reached = set()
        for group_size in GROUP_SIZES:
            paths = set().union(
                *(_expected_paths(split_k, group_size, case) for case in CASES[split_k])
            )
            # Only group size 32 allows an N that is not a multiple of 64.
            assert ("exit" in paths) == (group_size == 32), (split_k, group_size)
            reached |= paths
        assert reached == every, split_k


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_quantized_qvm_split_k_model_matches_partial_products(entry):
    info = INFO[entry]
    for case in CASES[info["splitK"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        assert run["paths"] == _expected_paths(
            info["splitK"], info["groupSize"], case
        ), (entry, case)
        layout = data["layout"]
        m, k = layout["M"], layout["K"]
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        # The parts read every row of the used activations and matrices once
        # between them, and no guard or unused activation row.
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


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qvm_split_k_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
    for entry in ENTRIES:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in CASES[INFO[entry]["splitK"]]:
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
    [(entry, mutation) for entry in SELECTED_ENTRIES for mutation in MUTATIONS],
)
def test_quantized_qvm_split_k_expectations_reject_defects(entry, mutation):
    for case in CASES[INFO[entry]["splitK"]]:
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: {MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_quantized_qvm_split_k_opengl_assertions_name_buffer_elements(entry):
    used = set()
    for case in CASES[INFO[entry]["splitK"]]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _simulate(entry, data)["records"]:
            used.add(expression)
            assert 0 <= value < lengths[buffer] - GUARD_COUNT, (case, expression)
            assert value <= INDEX_BOUND
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS["batch_1"])


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qvm_split_k_configs_scope_target_options(tmp_path, target):
    entry = ENTRIES[0]
    config = _config(tmp_path / "work", target, entry)
    assert config.entry_points == {SOURCE: (entry,)}
    (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
    assert tuple(int(value) for value in rule) == WORKGROUP
    options = config.source_options["metal"]["target_options"]
    assert options["directx"]["software_subgroup_width"] == SIMD_SIZE
    assert options["opengl"]["software_subgroup_width"] == SIMD_SIZE
    expressions = [item.expression for item in config.index_range_assertions]
    if target == "opengl":
        assert expressions == list(OPENGL_INDEX_RANGE_ASSERTIONS["batch_1"])
    else:
        assert expressions == []


def test_quantized_qvm_split_k_blocks_follow_qvm_impl():
    # The model reuses qvm_impl's blocks of one input row per lane.
    assert BLOCK == SIMD_SIZE == 32 and COLUMNS == 32


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "quantized qvm split-K", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # ENTRIES follow quantized.metal's own instantiation macros.
    text = source.decode("utf-8")
    splits = _macro(text, "instantiate_quantized_all_splitk")
    assert re.findall(
        r"instantiate_quantized_split_k\((\w+), type, group_size, bits, (\d+)\)",
        splits,
    ) == [(KERNEL, str(split_k)) for split_k in SPLIT_KS]
    funcs = _macro(text, "instantiate_quantized_funcs")
    assert "instantiate_quantized_all_splitk(type, group_size, bits)" in funcs
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
    match = re.search(rf"\]\]\s*void\s+{KERNEL}\s*\((.*?)\)\s*\{{", header, re.DOTALL)
    assert match, KERNEL
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
    assert tuple((name, int(index)) for name, index in found) == PARAMETERS
    body = header[match.end() : header.index("template", match.end())]
    for required in (
        "int M = x_shape[x_batch_ndims];",
        "out_vec_size * M,",
        "tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;",
        "int in_vec_stride = (split_k - 1) * in_vec_size + final_block_size;",
        "in_vec_size_adj,",
        "in_vec_stride,",
    ):
        assert required in body, required
    impl = header[header.index("METAL_FUNC void qvm_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    assert "x += tid.x * in_vec_stride + simd_lid;" in impl
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    start = host.index("void qvm_split_k(")
    host_split = host[start : host.index("void qvm(", start)]
    for required in (
        "int split_k = K > 8192 ? 32 : 8;",
        "int split_D = (K + split_k - 1) / split_k;",
        "B *= split_k;",
        "MTL::Size group_dims = MTL::Size(bk, num_simdgroups, 1);",
        "MTL::Size grid_dims = MTL::Size(M, (N + bn - 1) / bn, B);",
        "x_shape.insert(x_shape.end() - 2, split_k);",
        "x_shape.back() /= split_k;",
        "x_strides.insert(x_strides.end() - 2, split_D);",
        "x_strides[x_ndim - 1] = split_D;",
        "w_shape.insert(w_shape.end() - 2, split_k);",
        "w_shape[w.ndim() - 1] /= split_k;",
        "w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));",
        "s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));",
        "b_strides.insert(b_strides.end() - 2, split_D * biases->shape(-1));",
        "int final_block_size = K - (split_k - 1) * split_D;",
        "compute_encoder.set_bytes(split_D, c++);",
        "compute_encoder.set_bytes(final_block_size, c++);",
    ):
        assert required in host_split, required
    assert "if (K < 1024) {" in host
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qvm-split-k-metal-reference")
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
def test_current_mlx_quantized_qvm_split_k_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-qvm-split-k-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = _translate(root, work, target, config, record)
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES[INFO[entry]["splitK"]]:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, "batch_1", data, PARAMETERS
                    )
                    words = _execute(
                        executor, work, case, request, {"y": output}, observed
                    )["y"]["values"]
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work,
                            metal_reference,
                            entry,
                            "batch_1",
                            case,
                            data,
                            PARAMETERS,
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
                            "splitD": layout["splitD"],
                            "finalBlockSize": layout["finalBlockSize"],
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


def test_ci_requires_native_quantized_qvm_split_k_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized split-K vector-matrix native parity",
        family="quantized-qvm-split-k",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qvm_split_k.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qvm_split_k.py",
    )
