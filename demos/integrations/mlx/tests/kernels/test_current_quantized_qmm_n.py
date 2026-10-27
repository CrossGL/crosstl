"""Natively prove quantized.metal's 108 affine qmm_n matrix-matrix kernels.

``quantized.cpp``'s ``qmm`` multiplies ``M`` activation rows by weights that
are not transposed, ``K`` rows of ``N`` codes quantized along ``N``. Without
transposition the vector limit is 4 and there is no split-K fallback, so for
``M >= 4`` it picks ``affine_qmm_n``, batched for ``B > 1``. ``quantized.metal``
instantiates it for float, float16_t and bfloat16_t, group sizes 128, 64 and
32 and 2, 3, 4, 5, 6 and 8 bits, with and without batching: 108 of its 2,052
entries. Each is translated and pinned per target, then runs ``(32, 2, 2)``
threadgroups over a ``(N / 32, ceil(M / 32), B)`` grid.

``qmm_n_impl`` gives each threadgroup a 32 by 32 output tile. Per block of 32
reduction rows, a ``BlockLoader`` of four-element vectors copies the tile's
activation rows into ``Xs``, and a ``QuantizedBlockLoader`` reducing along its
rows dequantizes 32 weight rows into ``Ws``, stepping the scales a block of
rows at a time; ``BlockMMA``'s four simdgroups multiply them as eight by eight
fragments into float32 accumulators. A partial row tile loads its activations
with ``load_safe``, which zeroes the rows past ``M``, and stores with
``store_result_safe``; a ``K`` that is not a multiple of 32 ends with a pass
that loads only the remaining rows of both tiles. ``N`` is a multiple of the
group size, so every column tile is whole. The cases reach every one of these
paths with the inputs of the ``qmv_fast`` gate, which keep every dequantized
weight exact in the element type and every float32 accumulator exact, so only
the final ``static_cast<T>`` rounds. A tile-level model replays both loaders
and the fragment products and must reach direct dequantized dot products,
storing every output once. Each target must produce the model's words and
leave every guard untouched; on Metal the translation must also match the
unchanged upstream kernel word for word.
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

from demos.integrations.mlx.tests.kernels import quantized_qmm_n_model as model
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_n_model import (
    X_READS,
    X_THREAD_ROWS,
    VectorLoader,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import (
    BK,
    BM,
    BN,
    THREADS,
    WM,
    WN,
    padded,
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
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    HELPER_ELEMENTS,
    K_LIMIT,
    QMM_HEADERS,
    QMM_PARAMETERS,
    SIGNED_MAX,
    UNBATCHED_BOUND,
    WORKGROUP,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    _expected_validation_contract as _qmm_t_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import qmm_config
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
    SIMD_SIZE,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QMM_N"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_qmm_n.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "affine_qmm_n"
# instantiate_quantized_batched_wrap's order: batched 1, then 0.
VARIANTS = ("batch_1", "batch_0")
# QuantizedMatmul::eval_gpu's vector limit without transposition.
VECTOR_LIMIT = 4

CASES = {
    "batch_0": ("tiles", "ragged", "short"),
    "batch_1": ("stacked", "broadcast", "shared"),
}
# (M, columns, K): N is the least common multiple of the columns and the
# group size, as quantize requires N to be a multiple of the group size.
SHAPES = {
    # Two row tiles, the second of 8 rows; two whole blocks of K.
    "tiles": (40, 64, 64),
    # Two row tiles, the second of 4 rows; a block, then 13 rows of K.
    "ragged": (36, 64, 45),
    # At the vector limit: one row tile of 4 rows; only the last pass of K.
    "short": (VECTOR_LIMIT, 128, 13),
    "stacked": (33, 64, 40),
    "broadcast": (8, 64, 32),
    "shared": (32, 64, 64),
}
LAYOUTS = {
    "tiles": "unbatched",
    "ragged": "unbatched",
    "short": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for variant in VARIANTS:
                    entry = f"{KERNEL}_{itype}_gs_{group_size}_b_{bits}_{variant}"
                    yield entry, {
                        "kernel": KERNEL,
                        "variant": variant,
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


def _x_loader(itype):
    """The four-element activation BlockLoader the translation materializes."""
    return VectorLoader(
        f"BlockLoader_{itype}_{BM}_{BK}_{padded(itype)}_1_{THREADS}_1_{X_READS}"
        "__load_unsafe",
        BM,
        BK,
        padded(itype),
        THREADS,
    )


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

    ``family`` names what the model shows the expression indexes: activation
    elements the x loader reads as whole vectors, elements a resource helper
    reads or writes, weight bytes, int64 batch strides, and threadgroup tile
    slots. GLSL indexes buffers with 32-bit integers, so each buffer index is
    asserted to fit one.
    """
    info = INFO[entry]
    element = HELPER_ELEMENTS[info["dtype"]]
    items = [
        (f"source.offset + {index}", None, SIGNED_MAX, "x-vector")
        for index in range(X_READS)
    ]
    for access, family in (
        (f"load_{element}", "elements"),
        (f"store_{element}", "y"),
        ("load_uint8", "w-bytes"),
        ("load_int64_t", "strides"),
    ):
        items.append(
            (
                "reference.offset + index",
                {"function": f"crosstl_resource_{access}"},
                SIGNED_MAX,
                family,
            )
        )
    # Xs is BM rows of BK_padded and Ws BK rows of BN_padded: the same slots.
    last = (BM - 1) * padded(info["itype"]) + BK - 1
    for tile in ("Xs", "Ws"):
        items.append(
            (
                "reference.offset + index",
                {"indexed": f"crosstl_workgroup_{tile}"},
                last,
                tile,
            )
        )
    return items


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
    """The translation of one entry, with the qmm_t gate's options."""
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
    m, columns, k = SHAPES[case]
    return m, math.lcm(columns, INFO[entry]["groupSize"]), k


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The qmv_fast gate's inputs, laid out as qvm's and bound as qmm's."""
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, LAYOUTS[case], m, n, k, transposed=True)
    data = case_data_for(info, entry, case, layout)
    values = dict(data["values"])
    values["K"] = values.pop("in_vec_size")
    values["N"] = values.pop("out_vec_size")
    values["M"] = ("int32", [m])
    assert set(values) == {name for name, _ in QMM_PARAMETERS}, entry
    grid = (n // BN, -(-m // BM), len(layout["batches"]))
    return {**data, "layout": {**layout, "grid": grid}, "values": values, "grid": grid}


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "batch":
        return info["variant"] == "batch_1"
    return True


def _simulate(entry, data, mutation=None):
    """quantized_qmm_n_model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    return model.simulate(INFO[entry], data, families, mutation)


def _expected_paths(entry, case):
    m, _, k = _shape(entry, case)
    return model.expected_paths(m, k)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMM_N_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMM_N_SHARD_COUNT"
# Five is coprime with the two variants and three element types.
CI_SHARD_COUNT = 5


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"qmm_n shard {index} is outside [0, {count})")
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
    total = 0
    for entry in ENTRIES:
        for case in CASES[INFO[entry]["variant"]]:
            m, n, k = _shape(entry, case)
            layout = layout_for(INFO[entry], LAYOUTS[case], m, n, k, transposed=True)
            total += len(layout["batches"]) * m * n + GUARD_COUNT
    return total


def _case_shapes(variant):
    shapes = {}
    for case in CASES[variant]:
        m, columns, k = SHAPES[case]
        shapes[case] = {
            "M": m,
            "N": f"lcm({columns}, group_size)",
            "K": k,
            "layout": LAYOUTS[case],
        }
    return shapes


def _expected_validation_contract():
    qmm_t = _qmm_t_validation_contract()
    return {
        "model": (
            "qmm_n_impl tile by tile after adjust_matrix_offsets: the "
            "four-element BlockLoader and the row-reducing QuantizedBlockLoader "
            "with their safe loads and the last partial block of K, the scales "
            "and biases stepped a block of rows at a time, BlockMMA's float32 "
            "fragment products and the whole or safe stores, against direct "
            "dequantized dot products"
        ),
        "packing": qmm_t["packing"],
        "weights": "K rows of N codes, quantized along N",
        "comparison": qmm_t["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(N / 32, ceil(M / 32), B)",
        "tile": {"BM": BM, "BN": BN, "BK": BK, "WM": WM, "WN": WN},
        "xLoaderReads": X_READS,
        "routing": (
            "untransposed weights with M at or above the vector limit of 4; "
            "batch_1 for B > 1; on devices without NAX kernels"
        ),
        "cases": {variant: _case_shapes(variant) for variant in VARIANTS},
        "inputs": qmm_t["inputs"],
        "exactness": qmm_t["exactness"],
        "guards": qmm_t["guards"],
        "softwareSubgroups": qmm_t["softwareSubgroups"],
        "cooperativeMatrices": qmm_t["cooperativeMatrices"],
        "workgroupAccessAssertions": qmm_t["workgroupAccessAssertions"],
        "openglIndexRangeAssertions": qmm_t["openglIndexRangeAssertions"],
        "metadataPreconditions": {"K": [0, K_LIMIT]},
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
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
        + len(ENTRIES)
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
        "qvmSplitKGateEntryCount": QVM_SPLIT_K_ENTRY_COUNT,
        "qmmTGateEntryCount": QMM_T_ENTRY_COUNT,
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
        "kind": "crosstl-mlx-current-tree-quantized-qmm-n-contract",
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


def test_current_mlx_quantized_qmm_n_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qmm_n_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qmm_n.current-tree.translation.json" in readme
    assert len(ENTRIES) == 108 == len(set(ENTRIES)) == QMM_N_ENTRY_COUNT
    assert Counter(info["variant"] for info in INFO.values()) == {
        "batch_1": 54,
        "batch_0": 54,
    }
    assert ENTRIES[:2] == (
        "affine_qmm_n_float_gs_128_b_2_batch_1",
        "affine_qmm_n_float_gs_128_b_2_batch_0",
    )
    assert ENTRIES[-1] == "affine_qmm_n_bfloat16_t_gs_32_b_8_batch_0"
    assert KERNEL not in EXCLUDED_KERNELS
    assert set(CASES) == {"batch_0", "batch_1"} <= set(QMV_FAST_CASES)
    assert set(SHAPES) == set(LAYOUTS)
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + len(ENTRIES)
        + QMM_T_SPLITK_ENTRY_COUNT
        + GATHER_QMM_ENTRY_COUNT
        + QMV_QUAD_ENTRY_COUNT
        + 108
        == UNIT_ENTRY_COUNT
    )


def test_quantized_qmm_n_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [22] * 3 + [21] * 2
    for shard in shards:
        for key in ("variant", "dtype", "groupSize", "bits"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_quantized_qmm_n_geometry_follows_the_loaders():
    for bits in BITS:
        geometry = model.base.geometry(bits)
        # Every thread dequantizes eight codes of one row: 128 threads fill
        # the 32 x 32 weight tile.
        assert geometry["reads"] * geometry["packFactor"] == BK * BN // THREADS
        assert geometry["packedColumns"] * BK >= THREADS
        for group_size in GROUP_SIZES:
            # One scale per thread and block: its columns share a group.
            assert group_size % BN == 0
    # Each activation thread copies two rows of four elements.
    loader = _x_loader("float")
    assert loader.reads == X_READS and BM // X_THREAD_ROWS == 2
    slots = loader.slots()
    assert len(slots) == THREADS
    assert sorted(slot for lane in slots for slot in lane) == sorted(
        row * padded("float") + column for row in range(BM) for column in range(BK)
    )


def test_quantized_qmm_n_fragments_cover_the_tile_once():
    cells = model.base.fragment_cells()
    assert len(cells) == BM * BN
    assert {(row, column) for _, _, row, column in cells} == {
        (row, column) for row in range(BM) for column in range(BN)
    }


@pytest.mark.parametrize("entry", ENTRIES[::5])
def test_quantized_qmm_n_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n, k) == _shape(entry, case)
        # QuantizedMatmul reaches qmm for M >= 4 without transposition.
        assert m >= VECTOR_LIMIT
        assert n % info["groupSize"] == 0 and n % BN == 0
        assert (batches > 1) == (info["variant"] == "batch_1")
        assert data["grid"] == (n // BN, -(-m // BM), batches)
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["K"][1] == [k] and values["N"][1] == [n] and values["M"][1] == [m]
        params = layout["params"]
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        # (K, N) weights: K rows of N * bits / 32 words and N / group_size groups.
        assert params["w_shape"][-2:] == [k, n * info["bits"] // 32]
        assert layout["words"] == n * info["bits"] // 32
        assert layout["groups"] == n // info["groupSize"]
        assert params["s_strides"][-2:] == [layout["groups"], 1]
        assert 0 <= k <= K_LIMIT


def test_quantized_qmm_n_cases_reach_every_path():
    every = {
        "rows-whole/k-whole",
        "rows-whole/k-tail",
        "rows-partial/k-whole",
        "rows-partial/k-tail",
        "store-whole",
        "store-safe",
    }
    reached = {}
    for entry in ENTRIES:
        for case in CASES[INFO[entry]["variant"]]:
            reached.setdefault(INFO[entry]["variant"], set()).update(
                _expected_paths(entry, case)
            )
    assert reached == {"batch_0": every, "batch_1": every}
    # "short" runs no whole block of K: only the last pass.
    assert SHAPES["short"][2] < BK


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_quantized_qmm_n_model_matches_dot_products(entry):
    for case in CASES[INFO[entry]["variant"]]:
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
        # Every used activation row, weight word, scale and bias is read, and
        # no guard or unused activation row is.
        assert run["reads"]["x"] == {
            batch["x"] + row * k + index
            for batch in layout["batches"]
            for row in range(m)
            for index in range(k)
        }
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


def test_quantized_qmm_n_model_derives_the_loader_assertions():
    for itype, _ in DTYPES:
        entry = next(e for e in ENTRIES if INFO[e]["itype"] == itype)
        run = _simulate(entry, _case_data(entry, CASES[INFO[entry]["variant"]][0]))
        loader = _x_loader(itype)
        slots = {slot for lane in loader.slots() for slot in lane}
        # Each block writes every BlockLoader slot and the tile's Ws slots.
        assert {slot for _, slot in run["records"]["Xs"]} == slots
        assert {slot for _, slot in run["records"]["Ws"]} == slots
        (assertion,) = _workgroup_access_assertions(entry)
        assert (assertion["minimum"], assertion["maximum"]) == (min(slots), max(slots))
        assert assertion["function"] == loader.function


def test_quantized_qmm_n_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in model.MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {"order": 18, "batch": 54}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qmm_n_inputs_keep_float32_exact(dtype):
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


def _shard_mutations():
    return [
        (entry, mutation)
        for entry in SELECTED_ENTRIES
        for mutation in model.MUTATIONS
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _shard_mutations())
def test_quantized_qmm_n_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
        if changed:
            break
    assert changed, (entry, mutation, model.MUTATIONS[mutation])


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::3])
def test_quantized_qmm_n_opengl_assertions_name_buffer_elements(entry):
    items = _opengl_index_items(entry)
    seen = {family: set() for _, _, _, family in items}
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        records = _simulate(entry, data)["records"]
        for _, _, maximum, family in items:
            for buffer, index in records[family]:
                seen[family].add(buffer)
                if family == "w-bytes":
                    limit = 4 * (lengths["w"] - GUARD_COUNT)
                elif family == "strides":
                    limit = lengths[buffer]
                elif family in ("Xs", "Ws"):
                    limit = maximum + 1
                else:
                    limit = lengths[buffer] - GUARD_COUNT
                assert 0 <= index < limit <= SIGNED_MAX + 1, (case, family, index)
    batched = INFO[entry]["variant"] == "batch_1"
    assert seen == {
        "x-vector": {"x"},
        "elements": {"x", "scales", "biases"},
        "y": {"y"},
        "w-bytes": {"w"},
        "strides": (
            {"x_strides", "w_strides", "s_strides", "b_strides"} if batched else set()
        ),
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qmm_n_configs_scope_target_options(tmp_path, target):
    for variant in VARIANTS:
        entry = next(e for e in ENTRIES if e.endswith(variant))
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            assert options[backend]["software_subgroup_width"] == SIMD_SIZE
            assert options[backend]["cooperative_matrix_software_lowering"] is True
        assert options["directx"]["relative_wave_shuffle_out_of_range"] == "self"
        (precondition,) = config.metadata_preconditions
        assert (precondition.parameter, precondition.maximum) == ("K", K_LIMIT)
        assertions = config.workgroup_access_assertions
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "metal":
            assert assertions == () and expressions == []
        else:
            assert [item.function for item in assertions] == [
                _x_loader(INFO[entry]["itype"]).function
            ]
            assert [item.parameter for item in assertions] == ["self.dst"]
            if target == "opengl":
                assert expressions == [item[0] for item in _opengl_index_items(entry)]
            else:
                assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


def _check_upstream_source(root, text):
    """Pin the instantiations, kernel parameters, tile code and host dispatch."""
    batched = re.findall(
        r"instantiate_quantized_batched_wrap\((\w+),",
        _macro(text, "instantiate_quantized_all_batched"),
    )
    assert batched[3] == KERNEL
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
    match = re.search(rf"\]\]\s*void\s+{KERNEL}\s*\((.*?)\)\s*\{{", header, re.DOTALL)
    assert match, KERNEL
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
    assert tuple((name, int(index)) for name, index in found) == QMM_PARAMETERS
    impl = header[header.index("METAL_FUNC void qmm_n_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int WM = 2;",
        "constexpr int WN = 2;",
        "constexpr int BK_padded = (BK + 16 / sizeof(T));",
        "constexpr int BN_padded = (BN + 16 / sizeof(T));",
        "BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;",
        "BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;",
        "const int k_blocks = K / BK;",
        "for (int k = 0; k < K; k += BK) {",
        "loader_x.load_safe(short2(BK, num_els));",
        "loader_x.load_safe(short2(num_k, num_els));",
        "loader_x.load_safe(short2(num_k, BM));",
        "loader_w.load_safe(short2(BN, num_k));",
        "mma_op.store_result_safe(y, N, short2(BN, num_els));",
    ):
        assert required in impl, required
    # The weight loader reduces along its BK rows of BN columns.
    assert re.search(
        r"QuantizedBlockLoader<\s*T,\s*BK,\s*BN,\s*BN_padded,\s*0,\s*"
        r"WM \* WN \* SIMD_SIZE,\s*group_size,\s*bits>;",
        impl,
    )
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "MTL::Size group_dims(32, wn, wm);",
        "MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
        'mode + (transpose ? "_qmm_t_" : "_qmm_n_"),',
        'batched ? "_batch_1" : "_batch_0",',
        'd, kname, "qmm_n", mode, type_string, group_size, bits, batched);',
        "compute_encoder.set_bytes(M, c++);",
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
        "if (transpose_ && B == 1) {",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "quantized qmm_n", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMM_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    _check_upstream_source(root, source.decode("utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qmm-n-metal-reference")
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
    variant = INFO[entry]["variant"]
    data = _case_data(entry, case)
    request, output, observed = _request(
        descriptor,
        package,
        target,
        variant,
        data,
        parameters=QMM_PARAMETERS,
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
            variant,
            case,
            data,
            parameters=QMM_PARAMETERS,
            workgroup=WORKGROUP,
            bound=UNBATCHED_BOUND,
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
def test_current_mlx_quantized_qmm_n_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-qmm-n-", dir=root) as path:
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
                    for case in CASES[INFO[entry]["variant"]]
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


def test_ci_requires_native_quantized_qmm_n_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized matrix-matrix native parity",
        family="quantized-qmm-n",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmm_n.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmm_n.py",
    )
