"""Natively prove quantized.metal's 162 affine gather_qmm matrix-matrix kernels.

``quantized.cpp``'s ``GatherQMM`` multiplies, per output batch, the
activation matrix ``lhs_indices`` selects by the weight matrix
``rhs_indices`` selects. On devices without NAX kernels, with ``M`` at or
above the vector limit (33 for transposed weights up to 2,048 by 2,048, 4
otherwise) and no sorted single-row batch for ``gather_qmm_rhs``, its
``gather_qmm`` picks ``affine_gather_qmm_t`` for transposed weights and
``affine_gather_qmm_n`` otherwise. ``quantized.metal`` instantiates both for
float, float16_t and bfloat16_t, group sizes 128, 64 and 32 and 2, 3, 4, 5, 6
and 8 bits, ``affine_gather_qmm_t`` with aligned and unaligned ``N``: 162 of
its 2,052 entries. Each is translated and pinned per target, then runs
``(32, 2, 2)`` threadgroups over a ``(ceil(N / 32), ceil(M / 32), B)`` grid.

Both kernels first run ``adjust_matrix_offsets``: ``tid.z``'s lhs and rhs
indices, read through ``lhs_strides`` and ``rhs_strides`` or, for several
batch dimensions, ``elem_to_loc_broadcast`` over ``batch_shape``, select the
activation matrix and the weight, scale and bias matrices, and ``y`` moves to
``tid.z``'s ``M`` by ``N`` slice. ``affine_gather_qmm_t`` then runs
``qmm_t_impl`` and ``affine_gather_qmm_n`` ``qmm_n_impl``, which
``test_current_quantized_qmm_t.py`` and ``test_current_quantized_qmm_n.py``
gate. The cases take the gather_qmv and gather_qvm gates' two index layouts:
interleaved one-dimensional indices into two activation and three weight
matrices, and a ``(2, 2)`` batch whose rhs indices broadcast along its second
dimension. Their shapes reach every load and store path of both
implementations with the inputs of the ``qmv_fast`` gate, which keep every
dequantized weight exact in the element type and every float32 accumulator
exact, so only the final ``static_cast<T>`` rounds. The tile-level models of
both implementations replay the gathered offsets and must reach direct
dequantized dot products, storing every output once. Each target must produce
the models' words and leave every guard untouched; on Metal the translation
must also match the unchanged upstream kernel word for word.
"""

import functools
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels import quantized_qmm_n_model as n_model
from demos.integrations.mlx.tests.kernels import quantized_qmm_t_model as t_model
from demos.integrations.mlx.tests.kernels import test_current_quantized_qmm_n as n_gate
from demos.integrations.mlx.tests.kernels import test_current_quantized_qmm_t as t_gate
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import BK, BM, BN
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
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    DIMENSION_LIMIT,
    K_LIMIT,
    QMM_HEADERS,
    SIGNED_MAX,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    VECTOR_LIMIT as TRANSPOSED_VECTOR_LIMIT,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    WORKGROUP,
    qmm_config,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    BITS,
    BROADCAST_LHS,
    BROADCAST_RHS,
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
    EXPERT_LHS,
    EXPERT_RHS,
    GATHER_PARAMETERS,
    GATHER_QMM_ENTRY_COUNT,
    GROUP_SIZES,
    GUARD_COUNT,
    QMM_N_ENTRY_COUNT,
    QMM_T_ENTRY_COUNT,
    QMM_T_SPLITK_ENTRY_COUNT,
    QMV_QUAD_ENTRY_COUNT,
    QMV_UNALIGNED_ENTRY_COUNT,
    QMV_WIDE_ENTRY_COUNT,
    QUANTIZE_ENTRY_COUNT,
    QVM_ENTRY_COUNT,
    QVM_SPLIT_K_ENTRY_COUNT,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_GATHER_QMM"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_gather_qmm.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNELS = {True: "affine_gather_qmm_t", False: "affine_gather_qmm_n"}
# Per (type, group size, bits), instantiate_quantized_all_single's
# affine_gather_qmm_n precedes instantiate_quantized_all_aligned's two
# affine_gather_qmm_t entries, aligned N first.
VARIANTS = (
    ("", False, True),
    ("_alN_true", True, True),
    ("_alN_false", True, False),
)
# gather_qmv's buffers with K, N and M in place of its two vector sizes, so
# every later buffer moves up one. gather_qmm binds all of them.
PARAMETERS = (
    GATHER_PARAMETERS[:7]
    + (("K", 7), ("N", 8), ("M", 9))
    + tuple((name, index + 1) for name, index in GATHER_PARAMETERS[9:])
)
VECTOR_LIMITS = {True: TRANSPOSED_VECTOR_LIMIT, False: n_gate.VECTOR_LIMIT}
CASES = ("experts", "broadcast")
# Transposed weights: (M, aligned N, unaligned N, K in groups).
T_SHAPES = {
    # Two row tiles, the second of 8 rows; two column tiles, the second of 8
    # columns when N is unaligned; two groups of K.
    "experts": (40, 64, 40, 2),
    # A row tile of one row; one column tile, of 8 columns when unaligned.
    "broadcast": (33, 32, 8, 1),
}
# Untransposed weights: (M, columns, K); N is the least common multiple of
# the columns and the group size.
N_SHAPES = {
    # Two row tiles, the second of one row; a block, then 8 rows of K.
    "experts": (33, 64, 40),
    # Two row tiles, the second of 8 rows; two whole blocks of K.
    "broadcast": (40, 64, 64),
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for suffix, transpose, aligned in VARIANTS:
                    kernel = KERNELS[transpose]
                    entry = f"{kernel}_{itype}_gs_{group_size}_b_{bits}{suffix}"
                    yield entry, {
                        "kernel": kernel,
                        "variant": "gather",
                        "transpose": transpose,
                        "aligned": aligned,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _sibling(entry):
    """The qmm gate and batched entry that share ``entry``'s tile code."""
    info = INFO[entry]
    stem = f"{info['itype']}_gs_{info['groupSize']}_b_{info['bits']}"
    if info["transpose"]:
        return t_gate, f"affine_qmm_t_{stem}_alN_true_batch_1"
    return n_gate, f"affine_qmm_n_{stem}_batch_1"


def _workgroup_access_assertions(entry):
    """The x loader's ``dst`` stays inside the slots BlockLoader writes."""
    gate, sibling = _sibling(entry)
    return tuple(
        {**assertion, "entryPoint": entry}
        for assertion in gate._workgroup_access_assertions(sibling)
    )


def _opengl_index_items(entry):
    """``(expression, scope, maximum, family)`` of each OpenGL assertion.

    The sibling qmm gate's families, and the lhs and rhs index elements
    ``crosstl_resource_load_uint`` reads: GLSL indexes buffers with 32-bit
    integers, so each index is asserted to fit one.
    """
    gate, sibling = _sibling(entry)
    items = list(gate._opengl_index_items(sibling))
    tiles = [item for item in items if item[3] in ("Xs", "Ws")]
    items = [item for item in items if item not in tiles]
    items.append(
        (
            "reference.offset + index",
            {"function": "crosstl_resource_load_uint"},
            SIGNED_MAX,
            "indices",
        )
    )
    return items + tiles


def _opengl_index_assertions(entry):
    return tuple(
        {
            "source": SOURCE,
            "expression": expression,
            **(scope or {}),
            "minimum": 0,
            "maximum": maximum,
        }
        for expression, scope, maximum, _ in _opengl_index_items(entry)
    )


def _config(directory, target, entry):
    """The translation of one entry, with the qmm gates' options."""
    return qmm_config(
        directory,
        target,
        entry,
        _workgroup_access_assertions(entry) if target != "metal" else (),
        _opengl_index_assertions(entry) if target == "opengl" else (),
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    info = INFO[entry]
    if info["transpose"]:
        m, aligned_n, unaligned_n, groups = T_SHAPES[case]
        n = aligned_n if info["aligned"] else unaligned_n
        return m, n, groups * info["groupSize"]
    m, columns, k = N_SHAPES[case]
    return m, math.lcm(columns, info["groupSize"]), k


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The qmv_fast gate's inputs in a gather layout, bound as gather_qmm's."""
    info = INFO[entry]
    assert case in CASES, (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, case, m, n, k, transposed=not info["transpose"])
    data = case_data_for(info, entry, case, layout)
    values = dict(data["values"])
    values["K"] = values.pop("in_vec_size")
    values["N"] = values.pop("out_vec_size")
    values["M"] = ("int32", [m])
    assert set(values) == {name for name, _ in PARAMETERS}, entry
    grid = (-(-n // BN), -(-m // BM), len(layout["batches"]))
    return {**data, "layout": {**layout, "grid": grid}, "values": values, "grid": grid}


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

GATHER_MUTATION = {"indices": "uses tid.z in place of the gathered indices"}


def _mutations(entry):
    model = t_model if INFO[entry]["transpose"] else n_model
    return {**model.MUTATIONS, **GATHER_MUTATION}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "group-steps":
        return info["groupSize"] > BK
    return True


def _simulate(entry, data, mutation=None):
    """The qmm_t or qmm_n model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    model = t_model if INFO[entry]["transpose"] else n_model
    return model.simulate(INFO[entry], data, families, mutation)


def _expected_paths(entry, case):
    info = INFO[entry]
    m, n, k = _shape(entry, case)
    if info["transpose"]:
        return t_model.expected_paths(info["aligned"], m, n)
    return n_model.expected_paths(m, k)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_GATHER_QMM_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_GATHER_QMM_SHARD_COUNT"
# Seven is coprime with the three variants and three element types.
CI_SHARD_COUNT = 7


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"gather_qmm shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    total = 0
    for entry in ENTRIES:
        for case in CASES:
            m, n, _ = _shape(entry, case)
            total += len(_case_data(entry, case)["layout"]["batches"]) * m * n
            total += GUARD_COUNT
    return total


def _case_shapes(transpose, aligned):
    shapes = {}
    for case in CASES:
        if transpose:
            m, aligned_n, unaligned_n, groups = T_SHAPES[case]
            n, k = (aligned_n if aligned else unaligned_n), f"{groups} * group_size"
        else:
            m, columns, k = N_SHAPES[case]
            n = f"lcm({columns}, group_size)"
        shapes[case] = {"M": m, "N": n, "K": k, "layout": case}
    return shapes


def _expected_validation_contract():
    qmm_t = t_gate._expected_validation_contract()
    return {
        "model": (
            "adjust_matrix_offsets' gathered x, w, scale, bias and y offsets, "
            "then qmm_t_impl's or qmm_n_impl's tile model, against direct "
            "dequantized dot products"
        ),
        "packing": qmm_t["packing"],
        "weights": {
            "affine_gather_qmm_t": "N rows of K codes, quantized along K",
            "affine_gather_qmm_n": "K rows of N codes, quantized along N",
        },
        "comparison": qmm_t["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), ceil(M / 32), B)",
        "tile": qmm_t["tile"],
        "routing": (
            "GatherQMM with M at or above the vector limit, 33 for transposed "
            "weights and 4 otherwise, and no sorted M == 1 batch for "
            "gather_qmm_rhs; alN_true exactly when N is a multiple of 32; on "
            "devices without NAX kernels"
        ),
        "indices": {
            "experts": {
                "lhs": list(EXPERT_LHS),
                "lhsStride": 2,
                "rhs": list(EXPERT_RHS),
            },
            "broadcast": {
                "lhs": list(BROADCAST_LHS),
                "rhs": list(BROADCAST_RHS),
                "batchShape": [2, 2],
            },
        },
        "cases": {
            KERNELS[transpose] + suffix: _case_shapes(transpose, aligned)
            for suffix, transpose, aligned in VARIANTS
        },
        "inputs": qmm_t["inputs"],
        "exactness": qmm_t["exactness"],
        "guards": qmm_t["guards"],
        "softwareSubgroups": qmm_t["softwareSubgroups"],
        "cooperativeMatrices": qmm_t["cooperativeMatrices"],
        "workgroupAccessAssertions": qmm_t["workgroupAccessAssertions"],
        "openglIndexRangeAssertions": (
            qmm_t["openglIndexRangeAssertions"]
            + "; lhs and rhs index elements fit them too"
        ),
        "metadataPreconditions": {"K": [0, K_LIMIT]},
        "metalUpstreamComparison": qmm_t["metalUpstreamComparison"],
    }


def _expected_scope():
    covered = (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + len(ENTRIES)
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
        "qvmSplitKGateEntryCount": QVM_SPLIT_K_ENTRY_COUNT,
        "qmmTGateEntryCount": QMM_T_ENTRY_COUNT,
        "qmmNGateEntryCount": QMM_N_ENTRY_COUNT,
        "qmmTSplitKGateEntryCount": QMM_T_SPLITK_ENTRY_COUNT,
        "qmvQuadGateEntryCount": QMV_QUAD_ENTRY_COUNT,
        "excludedEntries": {
            "kernels": list(EXCLUDED_KERNELS),
            "count": UNIT_ENTRY_COUNT - covered,
            "reason": "the gather_qmm_rhs matrix-matrix kernels are not gated yet",
        },
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
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
        "kind": "crosstl-mlx-current-tree-quantized-gather-qmm-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(QMM_HEADERS),
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


def test_current_mlx_quantized_gather_qmm_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_gather_qmm_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_gather_qmm.current-tree.translation.json" in readme
    assert len(ENTRIES) == 162 == len(set(ENTRIES)) == GATHER_QMM_ENTRY_COUNT
    assert Counter((info["kernel"], info["aligned"]) for info in INFO.values()) == {
        ("affine_gather_qmm_n", True): 54,
        ("affine_gather_qmm_t", True): 54,
        ("affine_gather_qmm_t", False): 54,
    }
    assert ENTRIES[:3] == (
        "affine_gather_qmm_n_float_gs_128_b_2",
        "affine_gather_qmm_t_float_gs_128_b_2_alN_true",
        "affine_gather_qmm_t_float_gs_128_b_2_alN_false",
    )
    assert ENTRIES[-1] == "affine_gather_qmm_t_bfloat16_t_gs_32_b_8_alN_false"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert set(CASES) == set(QMV_FAST_CASES["gather"])
    assert set(T_SHAPES) == set(N_SHAPES) == set(CASES)
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + len(ENTRIES)
        + QMV_QUAD_ENTRY_COUNT
        + 108
        == UNIT_ENTRY_COUNT
    )


def test_quantized_gather_qmm_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [24] + [23] * 6
    for shard in shards:
        for key in ("kernel", "aligned", "dtype", "groupSize"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_quantized_gather_qmm_parameters_follow_gather_qmv():
    assert [index for _, index in PARAMETERS] == list(range(22))
    names = [name for name, _ in PARAMETERS]
    assert names[:10] == [
        "w",
        "scales",
        "biases",
        "x",
        "lhs_indices",
        "rhs_indices",
        "y",
        "K",
        "N",
        "M",
    ]
    assert names[10:] == [name for name, _ in GATHER_PARAMETERS[9:]]


@pytest.mark.parametrize("entry", ENTRIES[::5])
def test_quantized_gather_qmm_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = layout["batches"]
        assert (m, n, k) == _shape(entry, case)
        # GatherQMM reaches gather_qmm, not gather_qmm_rhs, on every device
        # without NAX kernels.
        assert VECTOR_LIMITS[info["transpose"]] <= m and m != 1
        assert k <= DIMENSION_LIMIT and n <= DIMENSION_LIMIT
        if info["transpose"]:
            assert k % info["groupSize"] == 0
            assert (n % BN == 0) == info["aligned"]
        else:
            # quantize requires N to be a multiple of the group size.
            assert n % info["groupSize"] == 0 and n % BN == 0
        assert (
            data["grid"] == (-(-n // BN), -(-m // BM), len(batches)) == layout["grid"]
        )
        assert len(batches) == 4
        assert len(data["expected"]["y"]) == len(batches) * m * n + GUARD_COUNT
        values = data["values"]
        assert values["K"][1] == [k] and values["N"][1] == [n] and values["M"][1] == [m]
        params = layout["params"]
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert len(params["batch_shape"]) == params["batch_ndims"]
        assert params["x_shape"][-2:] == [m, k]
        if info["transpose"]:
            assert params["w_shape"][-2:] == [n, k * info["bits"] // 32]
            assert layout["groups"] == k // info["groupSize"]
        else:
            assert params["w_shape"][-2:] == [k, n * info["bits"] // 32]
            assert layout["groups"] == n // info["groupSize"]
        assert params["s_strides"][-2:] == [layout["groups"], 1]
        # Each batch gathers a weight matrix the buffer holds, and the
        # offsets are not tid.z's: the indices repeat and reorder matrices.
        assert {batch["w"] for batch in batches} <= set(range(layout["matrices"]))
        assert len({batch["w"] for batch in batches}) > 1
        assert [batch["x"] for batch in batches] != sorted(
            {batch["x"] for batch in batches}
        )
        assert 0 <= k <= K_LIMIT


def test_quantized_gather_qmm_cases_reach_every_path():
    t_every = {
        "x-unsafe/w-unsafe",
        "x-safe/w-unsafe",
        "x-unsafe/w-safe",
        "x-safe/w-safe",
        "store-whole",
        "store-safe",
    }
    n_every = {
        "rows-whole/k-whole",
        "rows-partial/k-whole",
        "rows-whole/k-tail",
        "rows-partial/k-tail",
        "store-whole",
        "store-safe",
    }
    reached = {}
    for entry in ENTRIES:
        key = (INFO[entry]["kernel"], INFO[entry]["aligned"])
        for case in CASES:
            reached.setdefault(key, set()).update(_expected_paths(entry, case))
    assert reached == {
        ("affine_gather_qmm_t", True): t_every - {"x-unsafe/w-safe", "x-safe/w-safe"},
        ("affine_gather_qmm_t", False): t_every,
        ("affine_gather_qmm_n", True): n_every,
    }


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_quantized_gather_qmm_model_matches_dot_products(entry):
    transpose = INFO[entry]["transpose"]
    for case in CASES:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        assert run["paths"] == _expected_paths(entry, case), (entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every gathered activation row, weight word, scale and bias is read,
        # and no guard, unselected matrix or skipped row is.
        assert run["reads"]["x"] == {
            batch["x"] + row * k + index
            for batch in layout["batches"]
            for row in range(m)
            for index in range(k)
        }
        matrices = {batch["w"] for batch in layout["batches"]}
        rows, words, groups = layout["rows"], layout["words"], layout["groups"]
        assert rows == (n if transpose else k)
        assert run["reads"]["w"] == {
            (matrix * rows + row) * words + word
            for matrix in matrices
            for row in range(rows)
            for word in range(words)
        }
        for name in ("scales", "biases"):
            assert run["reads"][name] == {
                (matrix * rows + row) * groups + group
                for matrix in matrices
                for row in range(rows)
                for group in range(groups)
            }


def test_quantized_gather_qmm_model_derives_the_loader_assertions():
    for entry in ENTRIES[:3]:
        run = _simulate(entry, _case_data(entry, CASES[0]))
        (assertion,) = _workgroup_access_assertions(entry)
        gate, _ = _sibling(entry)
        loader = gate._x_loader(INFO[entry]["itype"])
        slots = {slot for lane in loader.slots() for slot in lane}
        assert {slot for _, slot in run["records"]["Xs"]} == slots
        assert (assertion["minimum"], assertion["maximum"]) == (min(slots), max(slots))
        assert assertion["function"] == loader.function
        assert assertion["entryPoint"] == entry


def test_quantized_gather_qmm_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in _mutations(entry)
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {"order": 27, "group-steps": 36}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_gather_qmm_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
    for entry in ENTRIES:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in CASES:
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


def _shard_mutations():
    return [
        (entry, mutation)
        for entry in SELECTED_ENTRIES
        for mutation in _mutations(entry)
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _shard_mutations())
def test_quantized_gather_qmm_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
        if changed:
            break
    assert changed, (entry, mutation, _mutations(entry)[mutation])


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::3])
def test_quantized_gather_qmm_opengl_assertions_name_buffer_elements(entry):
    items = _opengl_index_items(entry)
    seen = {family: set() for _, _, _, family in items}
    for case in CASES:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        records = _simulate(entry, data)["records"]
        for _, _, maximum, family in items:
            for buffer, index in records[family]:
                seen[family].add(buffer)
                if family == "w-bytes":
                    limit = 4 * (lengths["w"] - GUARD_COUNT)
                elif family in ("strides", "indices"):
                    limit = lengths[buffer]
                elif family in ("Xs", "Ws"):
                    limit = maximum + 1
                else:
                    limit = lengths[buffer] - GUARD_COUNT
                assert 0 <= index < limit <= SIGNED_MAX + 1, (case, family, index)
    assert seen == {
        "x-vector": {"x"},
        "elements": {"x", "scales", "biases"},
        "y": {"y"},
        "w-bytes": {"w"},
        "strides": {
            "x_strides",
            "w_strides",
            "s_strides",
            "b_strides",
            "lhs_strides",
            "rhs_strides",
        },
        "indices": {"lhs_indices", "rhs_indices"},
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_gather_qmm_configs_scope_target_options(tmp_path, target):
    for suffix, transpose, _ in VARIANTS:
        entry = next(
            e
            for e in ENTRIES
            if e.startswith(KERNELS[transpose]) and e.endswith(suffix)
        )
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        (precondition,) = config.metadata_preconditions
        assert (precondition.parameter, precondition.maximum) == ("K", K_LIMIT)
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


def _check_upstream_source(root, text):
    """Pin the instantiations, kernel parameters, offsets and host dispatch."""
    single = _macro(text, "instantiate_quantized_all_single")
    found = re.findall(
        r"instantiate_quantized\((\w+), type, group_size, bits\)", single
    )
    assert found[-1] == "affine_gather_qmm_n", found
    aligned = _macro(text, "instantiate_quantized_all_aligned")
    found = re.findall(
        r"instantiate_quantized_aligned\((\w+), type, group_size, bits, (true|false)\)",
        aligned,
    )
    assert found == [("affine_gather_qmm_t", "true"), ("affine_gather_qmm_t", "false")]
    funcs = _macro(text, "instantiate_quantized_funcs")
    order = re.findall(r"(instantiate_quantized_all_\w+)\(", funcs)
    assert order.index("instantiate_quantized_all_single") < order.index(
        "instantiate_quantized_all_aligned"
    )
    header = (root / "mlx/backend/metal/kernels/quantized.h").read_text(
        encoding="utf-8"
    )
    calls = {
        # K_eff is K: the whole reduction.
        "affine_gather_qmm_t": (
            "qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(\n"
            "      w,\n      scales,\n      biases,\n      x,\n      y,\n      Xs,\n"
            "      Ws,\n      K,\n      N,\n      M,\n      K,\n      tid,"
        ),
        "affine_gather_qmm_n": (
            "qmm_n_impl<T, group_size, bits, BM, BK, BN>(\n      w, scales, biases, "
            "x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);"
        ),
    }
    for kernel, call in calls.items():
        match = re.search(
            rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", header, re.DOTALL
        )
        assert match, kernel
        found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
        assert tuple((name, int(index)) for name, index in found) == PARAMETERS
        body = header[match.end() :]
        body = body[: body.index("\n}\n")]
        assert "      y,\n      M * N,\n      batch_ndims," in body, kernel
        assert body.index("adjust_matrix_offsets<T>(") < body.index(call), kernel
    offsets = header[header.rindex("METAL_FUNC void adjust_matrix_offsets(") :]
    offsets = offsets[: offsets.index("\n}\n")]
    for required in (
        "const device uint32_t* lhs_indices,",
        "x_idx = lhs_indices[tid.z * lhs_strides[0]];",
        "w_idx = rhs_indices[tid.z * rhs_strides[0]];",
        "tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);",
        "x_idx = lhs_indices[idx.x];",
        "w_idx = rhs_indices[idx.y];",
        "x += x_idx * x_strides[0];",
        "x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);",
        "w += w_idx * w_strides[0];",
        "w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);",
        "y += tid.z * output_stride;",
    ):
        assert required in offsets, required
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    dispatch = host[host.index("void gather_qmm(") :]
    dispatch = dispatch[: dispatch.index("\n}\n")]
    for required in (
        "MTL::Size group_dims(32, wn, wm);",
        "MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
        "bool aligned = N % 32 == 0;",
        'mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),',
        'transpose ? (aligned ? "_alN_true" : "_alN_false") : "",',
        "compute_encoder.set_input_array(lhs_indices, c++);",
        "compute_encoder.set_bytes(M, c++);",
        "c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);",
        "add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);",
    ):
        assert required in dispatch, required
    route = host[host.index("void GatherQMM::eval_gpu(") :]
    route = route[: route.index("\n}\n")]
    for required in (
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
        "if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {",
        "if (M >= vector_limit) {",
    ):
        assert required in route, required
    assert (
        route.index("gather_qmm_rhs(")
        < route.index("gather_qmm(")
        < route.index("gather_qmv(")
    )


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "quantized gather_qmm", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMM_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    text = source.decode("utf-8")
    # The tile code both implementations share with the qmm gates.
    t_gate._check_upstream_source(root, text)
    n_gate._check_upstream_source(root, text)
    _check_upstream_source(root, text)
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-gather-qmm-metal-reference")
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


def _run_case(work, target, entry, case, executor, translated, reference):
    """Dispatch one case on the translation and, on Metal, upstream."""
    descriptor, package = translated
    data = _case_data(entry, case)
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
            work,
            reference,
            entry,
            "gather",
            case,
            data,
            parameters=PARAMETERS,
            workgroup=WORKGROUP,
        )
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
        "checkedElementCount": len(data["expected"]["y"]),
        "observedInputCount": observed,
        "upstream": upstream,
    }


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_quantized_gather_qmm_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-gather-qmm-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            translated = _translate(
                root, work, target, config, record, workgroup=WORKGROUP
            )
            executor = _executor(target)
            try:
                evidence = [
                    _run_case(
                        work, target, entry, case, executor, translated, metal_reference
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


def test_ci_requires_native_quantized_gather_qmm_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized gather matrix-matrix native parity",
        family="quantized-gather-qmm",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_gather_qmm.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_gather_qmm.py",
    )
