"""Natively prove quantized.metal's 108 affine qmm_t_splitk kernels.

``quantized.cpp``'s ``QuantizedMatmul`` sends an unbatched product with
transposed weights and ``M`` at or above the vector limit to ``qmm_splitk``
on every device. It aims at about 512 threadgroups: ``split_k`` is 512 over
the output's 32 by 32 tiles, at most ``K / k_align`` with ``k_align`` the
larger of the group size and 32, and lowered until ``split_k * k_align``
divides ``K``. With ``split_k > 1`` it dispatches ``affine_qmm_t_splitk``
over a ``(ceil(N / 32), ceil(M / 32), split_k)`` grid of ``(32, 2, 2)``
threadgroups into a ``(split_k, M, N)`` intermediate, which a strided sum
reduction then folds; otherwise it falls back to ``qmm``, which
``test_current_quantized_qmm_t.py`` gates. ``quantized.metal`` instantiates
it for float, float16_t and bfloat16_t, group sizes 128, 64 and 32 and 2, 3,
4, 5, 6 and 8 bits, with aligned and unaligned ``N``: 108 of its 2,052
entries. Each is translated and pinned per target.

The kernel moves ``x``, the weight bytes, the scales and the biases to its
partition's first reduction column, ``tid.z * k_partition_size``, and ``y``
to its partition's ``M`` by ``N`` slice, then runs ``qmm_t_impl`` over
``k_partition_size`` columns while its rows keep the stride ``K``; the
weights reach ``qmm_t_impl`` as a ``uint32_t`` view of the moved byte
pointer. The cases split ``K`` into two and three partitions with the qmm_t
gate's tiles and inputs, so they reach every load and store path, and each
partition's partial sums are exact in float32 and round once to the element
type. A tile-level model replays ``qmm_t_impl`` per partition and must reach
each partition's direct dequantized partial dot products, storing every
intermediate element once. Each target must produce the model's words and
leave every guard untouched; on Metal the translation must also match the
unchanged upstream kernel word for word.
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

from demos.integrations.mlx.tests.kernels import quantized_qmm_t_model as model
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import (
    BK,
    BM,
    BN,
    THREADS,
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
    DIMENSION_LIMIT,
    HELPER_ELEMENTS,
    K_LIMIT,
    QMM_HEADERS,
    SIGNED_MAX,
    VECTOR_LIMIT,
    WORKGROUP,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    _expected_validation_contract as _qmm_t_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    _split_k,
    _x_loader,
    qmm_config,
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
    QMM_N_ENTRY_COUNT,
    QMM_T_ENTRY_COUNT,
    QMM_T_SPLITK_ENTRY_COUNT,
    QMV_QUAD_ENTRY_COUNT,
    QMV_UNALIGNED_ENTRY_COUNT,
    QMV_WIDE_ENTRY_COUNT,
    QUANTIZE_ENTRY_COUNT,
    QVM_ENTRY_COUNT,
    QVM_SPLIT_K_ENTRY_COUNT,
    SENTINEL,
    SIMD_SIZE,
    _exact_dyadic,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QMM_T_SPLITK"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/"
    "quantized_qmm_t_splitk.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "affine_qmm_t_splitk"
# instantiate_quantized_all_splitk_qmm's order.
VARIANTS = (("alN_true", True), ("alN_false", False))
PARAMETERS = (
    ("w", 0),
    ("scales", 1),
    ("biases", 2),
    ("x", 3),
    ("y", 4),
    ("K", 5),
    ("N", 6),
    ("M", 7),
    ("k_partition_size", 8),
    ("split_k_partition_stride", 9),
)
# qmm_splitk binds every parameter.
BOUND = len(PARAMETERS)
# The K_eff loop runs to k_partition_size; the rows still stride by K.
BOUNDED = ("K", "k_partition_size")

CASES = ("halves", "thirds")
# (M, aligned N, unaligned N, K in groups).
SHAPES = {
    # Two row tiles, the second of 8 rows; two column tiles, the second of 8
    # columns when N is unaligned; two partitions.
    "halves": (40, 64, 40, 2),
    # A row tile of one row, three column tiles and three partitions.
    "thirds": (33, 96, 72, 3),
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for suffix, aligned in VARIANTS:
                    entry = f"{KERNEL}_{itype}_gs_{group_size}_b_{bits}_{suffix}"
                    yield entry, {
                        "kernel": KERNEL,
                        # Unbatched: qmm_splitk runs only for B == 1.
                        "variant": "batch_0",
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
    elements the x loader reads whole, elements a resource helper reads or
    writes, weight bytes, and threadgroup tile slots. GLSL indexes buffers
    with 32-bit integers, so each buffer index is asserted to fit one.
    """
    info = INFO[entry]
    element = HELPER_ELEMENTS[info["dtype"]]
    items = [
        (f"source.offset + {index}", None, SIGNED_MAX, "x-vector")
        for index in range(BK * BM // THREADS)
    ]
    for access, family in (
        (f"load_{element}", "elements"),
        (f"store_{element}", "y"),
        ("load_uint8", "w-bytes"),
    ):
        items.append(
            (
                "reference.offset + index",
                {"function": f"crosstl_resource_{access}"},
                SIGNED_MAX,
                family,
            )
        )
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
    """The translation of one entry, as the gate and contract run it."""
    return qmm_config(
        directory,
        target,
        entry,
        _workgroup_access_assertions(entry) if target != "metal" else (),
        _opengl_index_assertions(entry) if target == "opengl" else (),
        bounded=BOUNDED,
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    info = INFO[entry]
    m, aligned_n, unaligned_n, groups = SHAPES[case]
    return m, aligned_n if info["aligned"] else unaligned_n, groups * info["groupSize"]


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The qmv_fast gate's inputs, bound as qmm_t_splitk's parameters."""
    info = INFO[entry]
    assert case in CASES, (entry, case)
    m, n, k = _shape(entry, case)
    split_k = _split_k(m, n, k, info["groupSize"])
    assert split_k > 1 and k % split_k == 0, (entry, case, split_k)
    layout = layout_for(info, "unbatched", m, n, k)
    data = case_data_for(info, entry, case, layout)
    fast = data["values"]
    initial = [SENTINEL] * (split_k * m * n + GUARD_COUNT)
    values = {
        **{name: fast[name] for name in ("w", "scales", "biases", "x")},
        "y": (info["dtype"], initial),
        "K": fast["in_vec_size"],
        "N": fast["out_vec_size"],
        "M": ("int32", [m]),
        "k_partition_size": ("int32", [k // split_k]),
        "split_k_partition_stride": ("int32", [m * n]),
    }
    assert set(values) == {name for name, _ in PARAMETERS}, entry
    grid = (-(-n // BN), -(-m // BM), split_k)
    split = {
        **data,
        "layout": {
            **layout,
            "grid": grid,
            "splitK": split_k,
            "partitionSize": k // split_k,
        },
        "values": values,
        "initial": {"y": initial},
        "grid": grid,
    }
    split["expected"], split["totals"], split["budget"] = _partials(info, split)
    return split


def _partials(info, data):
    """Each partition's dequantized partial dot products, as ``y`` holds them."""
    dtype, group_size = info["dtype"], info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    size = layout["partitionSize"]
    x = data["values"]["x"][1]
    scales, biases = data["values"]["scales"][1], data["values"]["biases"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    for part in range(layout["splitK"]):
        for row in range(m):
            for col in range(n):
                total, bound = 0.0, 0.0
                for index in range(part * size, (part + 1) * size):
                    # Weight row ``col`` holds output ``col``'s K codes.
                    code = data["codes"][col][index]
                    group = col * groups + index // group_size
                    scale, bias = scales[group], biases[group]
                    value = x[row * k + index]
                    total += (scale * code + bias) * value
                    bound += abs(value) * (abs(scale) * code + abs(bias))
                assert _exact_dyadic(total) and bound < 2**19, (info, total)
                budget = max(budget, bound)
                position = (part * m + row) * n + col
                totals[position] = total
                expected[position] = _to_type(dtype, total)
    return {"y": expected}, totals, budget


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    **model.MUTATIONS,
    "partition-x": "keeps the activations on the first partition",
    "partition-w": "keeps the weight bytes on the first partition",
    "partition-scales": "keeps the scales and biases on the first partition",
    "partition-size": "runs every partition over all of K",
    "partition-stride": "places the partitions N outputs apart",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation in ("scale-step", "batch"):
        # Each partition is one group of one unbatched product.
        return False
    if mutation in ("group-steps", "k-step"):
        # A partition of one block leaves nothing to step past.
        return info["groupSize"] > BK
    return mutation in MUTATIONS


def _offsets(info):
    """affine_qmm_t_splitk's partition offsets for ``tid.z``."""
    geometry = model.geometry(info["bits"])

    def offsets(storage, z, mutation):
        start = z * storage["k_partition_size"][0]
        x = 0 if mutation == "partition-x" else start
        byte = start * geometry["bytesPerPack"] // geometry["packFactor"]
        if mutation == "partition-w":
            byte = 0
        group = 0 if mutation == "partition-scales" else start // info["groupSize"]
        stride = storage["split_k_partition_stride"][0]
        if mutation == "partition-stride":
            stride = storage["N"][0]
        # The uint32_t view of the moved byte pointer stays word aligned.
        assert byte % 4 == 0, byte
        return x, byte // 4, group, group, z * stride

    return offsets


def _tile(context, bases, tid_x, tid_y, mutation):
    """qmm_t_impl for one threadgroup, ``K_eff`` columns of its partition."""
    if mutation == "partition-size":
        context = {**context, "Keff": context["K"]}
    model._tile(context, bases, tid_x, tid_y, mutation)


def _simulate(entry, data, mutation=None):
    """quantized_qmm_t_model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    info = INFO[entry]
    return model.simulate(
        info, data, families, mutation, tile=_tile, offsets=_offsets(info)
    )


def _expected_paths(entry, case):
    m, n, _ = _shape(entry, case)
    return model.expected_paths(INFO[entry]["aligned"], m, n)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMM_T_SPLITK_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMM_T_SPLITK_SHARD_COUNT"
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
        raise RuntimeError(f"qmm_t_splitk shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


def _checked_element_count():
    """Intermediate elements, guards included, every target compares."""
    total = 0
    for entry in ENTRIES:
        for case in CASES:
            m, n, k = _shape(entry, case)
            total += _split_k(m, n, k, INFO[entry]["groupSize"]) * m * n
            total += GUARD_COUNT
    return total


def _case_shapes(aligned):
    shapes = {}
    for case in CASES:
        m, aligned_n, unaligned_n, groups = SHAPES[case]
        shapes[case] = {
            "M": m,
            "N": aligned_n if aligned else unaligned_n,
            "K": f"{groups} * group_size",
            "splitK": groups,
        }
    return shapes


def _expected_validation_contract():
    qmm_t = _qmm_t_validation_contract()
    return {
        "model": (
            "qmm_t_impl tile by tile per partition after affine_qmm_t_splitk's "
            "partition offsets: the BlockLoader and QuantizedBlockLoader "
            "blocks over k_partition_size columns with their safe loads, "
            "BlockMMA's float32 fragment products and the whole or safe "
            "stores into the partition's slice, against direct dequantized "
            "partial dot products"
        ),
        "packing": qmm_t["packing"],
        "weights": qmm_t["weights"],
        "comparison": qmm_t["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), ceil(M / 32), split_k)",
        "tile": qmm_t["tile"],
        "split": (
            "qmm_splitk's split_k: 512 over the output tiles, at most "
            "K / max(group_size, 32), lowered until split_k * max(group_size, "
            "32) divides K; k_partition_size = K / split_k and "
            "split_k_partition_stride = M * N"
        ),
        "routing": (
            "unbatched products with transposed weights, M at or above "
            "get_qmv_batch_limit's largest limit and split_k > 1, on every "
            "device; alN_true exactly when N is a multiple of 32"
        ),
        "cases": {suffix: _case_shapes(aligned) for suffix, aligned in VARIANTS},
        "inputs": qmm_t["inputs"],
        "exactness": (
            "every dequantized weight scale * (code - b) is exact in the "
            "element type; float32 holds every partition's products and "
            "partial sums exactly, so only each partition's final "
            "static_cast<T> rounds"
        ),
        "intermediate": (
            "the (split_k, M, N) partial sums the kernel writes, before "
            "qmm_splitk's strided sum reduction"
        ),
        "guards": qmm_t["guards"],
        "softwareSubgroups": qmm_t["softwareSubgroups"],
        "cooperativeMatrices": qmm_t["cooperativeMatrices"],
        "workgroupAccessAssertions": qmm_t["workgroupAccessAssertions"],
        "openglIndexRangeAssertions": (
            "buffer element and weight byte indices fit GLSL's 32-bit "
            "indices; tile indices stay within the BM x BK slots"
        ),
        "metadataPreconditions": {name: [0, K_LIMIT] for name in BOUNDED},
        "upstreamBuffers": BOUND,
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
        + len(ENTRIES)
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
        "qmmNGateEntryCount": QMM_N_ENTRY_COUNT,
        "gatherQmmGateEntryCount": GATHER_QMM_ENTRY_COUNT,
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
        "kind": "crosstl-mlx-current-tree-quantized-qmm-t-splitk-contract",
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


def test_current_mlx_quantized_qmm_t_splitk_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for pins in artifacts.values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qmm_t_splitk_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qmm_t_splitk.current-tree.translation.json" in readme
    assert len(ENTRIES) == 108 == len(set(ENTRIES)) == QMM_T_SPLITK_ENTRY_COUNT
    assert Counter(info["aligned"] for info in INFO.values()) == {True: 54, False: 54}
    assert ENTRIES[:2] == (
        "affine_qmm_t_splitk_float_gs_128_b_2_alN_true",
        "affine_qmm_t_splitk_float_gs_128_b_2_alN_false",
    )
    assert ENTRIES[-1] == "affine_qmm_t_splitk_bfloat16_t_gs_32_b_8_alN_false"
    assert KERNEL not in EXCLUDED_KERNELS
    assert (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + len(ENTRIES)
        + GATHER_QMM_ENTRY_COUNT
        + QMV_QUAD_ENTRY_COUNT
        + 108
        == UNIT_ENTRY_COUNT
    )


def test_quantized_qmm_t_splitk_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [22] * 3 + [21] * 2
    for shard in shards:
        for key in ("aligned", "dtype", "groupSize"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES[::5])
def test_quantized_qmm_t_splitk_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        assert (m, n, k) == _shape(entry, case)
        # QuantizedMatmul reaches qmm_splitk on every device.
        assert VECTOR_LIMIT <= m and k <= DIMENSION_LIMIT and n <= DIMENSION_LIMIT
        assert (n % BN == 0) == info["aligned"]
        split_k = layout["splitK"]
        # Each partition is a whole number of groups and of BK-wide blocks.
        assert split_k == SHAPES[case][3] > 1
        size = data["values"]["k_partition_size"][1][0]
        assert size * split_k == k and size % info["groupSize"] == size % BK == 0
        assert data["values"]["split_k_partition_stride"][1] == [m * n]
        assert data["grid"] == (-(-n // BN), -(-m // BM), split_k)
        assert len(data["expected"]["y"]) == split_k * m * n + GUARD_COUNT
        values = data["values"]
        assert values["K"][1] == [k] and values["N"][1] == [n] and values["M"][1] == [m]
        assert layout["words"] == k * info["bits"] // 32
        assert layout["groups"] == k // info["groupSize"]
        assert 0 <= size <= k <= K_LIMIT


def test_quantized_qmm_t_splitk_split_follows_qmm_splitk():
    # Fewer tiles split further, up to one group per partition, and a split
    # that does not divide K is lowered until it does.
    assert _split_k(33, 32, 6 * 32, 32) == 6
    assert _split_k(33, 32 * 52, 6 * 128, 128) == 3
    assert _split_k(33, 32 * 52, 5 * 128, 128) == 1
    # Below two groups of k_align columns qmm_splitk falls back to qmm.
    assert _split_k(40, 64, 64, 64) == 1
    assert _split_k(40, 64, 32, 32) == 1


def test_quantized_qmm_t_splitk_cases_reach_every_path():
    every = {
        "x-unsafe/w-unsafe",
        "x-safe/w-unsafe",
        "x-unsafe/w-safe",
        "x-safe/w-safe",
        "store-whole",
        "store-safe",
    }
    reached = {}
    for entry in ENTRIES:
        for case in CASES:
            reached.setdefault(INFO[entry]["aligned"], set()).update(
                _expected_paths(entry, case)
            )
    assert reached == {
        True: every - {"x-unsafe/w-safe", "x-safe/w-safe"},
        False: every,
    }


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_quantized_qmm_t_splitk_model_matches_partial_dot_products(entry):
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
        # The partitions together read every activation, weight word, scale
        # and bias once over, and no guard.
        assert run["reads"]["x"] == set(range(m * k))
        assert run["reads"]["w"] == set(range(n * layout["words"]))
        for name in ("scales", "biases"):
            assert run["reads"][name] == set(range(n * layout["groups"]))


def test_quantized_qmm_t_splitk_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {
        "order": 18,
        "scale-step": 108,
        "batch": 108,
        "group-steps": 36,
        "k-step": 36,
    }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qmm_t_splitk_inputs_keep_float32_exact(dtype):
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
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _shard_mutations())
def test_quantized_qmm_t_splitk_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
        if changed:
            break
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::3])
def test_quantized_qmm_t_splitk_opengl_assertions_name_buffer_elements(entry):
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
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }


def test_quantized_qmm_t_splitk_model_derives_the_loader_assertions():
    for itype, _ in DTYPES:
        entry = next(e for e in ENTRIES if INFO[e]["itype"] == itype)
        run = _simulate(entry, _case_data(entry, CASES[0]))
        loader = _x_loader(itype)
        slots = {slot for lane in loader.slots() for slot in lane}
        assert {slot for _, slot in run["records"]["Xs"]} == slots
        assert {slot for _, slot in run["records"]["Ws"]} == slots
        (assertion,) = _workgroup_access_assertions(entry)
        assert (assertion["minimum"], assertion["maximum"]) == (min(slots), max(slots))
        assert assertion["function"] == loader.function


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qmm_t_splitk_configs_scope_target_options(tmp_path, target):
    for suffix, _ in VARIANTS:
        entry = next(e for e in ENTRIES if e.endswith(suffix))
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            assert options[backend]["software_subgroup_width"] == SIMD_SIZE
            assert options[backend]["cooperative_matrix_software_lowering"] is True
        assert options["directx"]["relative_wave_shuffle_out_of_range"] == "self"
        assert [
            (item.parameter, item.minimum, item.maximum)
            for item in config.metadata_preconditions
        ] == [(name, 0, K_LIMIT) for name in BOUNDED]
        assertions = config.workgroup_access_assertions
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "metal":
            assert assertions == () and expressions == []
        else:
            assert [item.parameter for item in assertions] == ["self.dst"]
            if target == "opengl":
                assert expressions == [item[0] for item in _opengl_index_items(entry)]
            else:
                assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


def _check_upstream_source(root, text):
    """Pin the instantiations, kernel parameters, partition code and host split."""
    found = re.findall(
        r"instantiate_quantized_splitk_qmm\((\w+), type, group_size, bits, "
        r"(true|false)\)",
        _macro(text, "instantiate_quantized_all_splitk_qmm"),
    )
    assert found == [(KERNEL, "true"), (KERNEL, "false")]
    funcs = _macro(text, "instantiate_quantized_funcs")
    assert "instantiate_quantized_all_splitk_qmm(type, group_size, bits)" in funcs
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
    match = re.search(
        rf"\]\]\s*void\s+{KERNEL}\s*\((.*?)\)\s*\{{(.*?)\n\}}", header, re.DOTALL
    )
    assert match, KERNEL
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
    assert tuple((name, int(index)) for name, index in found) == PARAMETERS
    body = match.group(2)
    for required in (
        "const int k_start = tid.z * k_partition_size;",
        "x += k_start;",
        "auto wl = (const device uint8_t*)w;",
        "wl += k_start * bytes_per_pack / pack_factor;",
        "scales += k_start / group_size;",
        "biases += k_start / group_size;",
        "y += tid.z * static_cast<int64_t>(split_k_partition_stride);",
        "qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(",
        "(const device uint32_t*)wl,",
        "k_partition_size,",
    ):
        assert required in body, required
    impl = header[header.index("METAL_FUNC void qmm_t_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    assert "for (int k = 0; k < K_eff; k += BK) {" in impl
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    for required in (
        "if (transpose_ && B == 1) {",
        "int split_k = std::max(1, 512 / current_tgs);",
        "int k_align = group_size > 32 ? group_size : 32;",
        "split_k = std::min(split_k, K / k_align);",
        "while (split_k > 1 && (K % (split_k * k_align) != 0)) {",
        "int k_partition_size = K / split_k;",
        "int split_k_partition_stride = M * N;",
        "MTL::Size group_dims(32, 2, 2);",
        "MTL::Size grid_dims(n_tiles, m_tiles, split_k);",
        'aligned ? "_alN_true" : "_alN_false");',
        "compute_encoder.set_bytes(split_k_partition_stride, c++);",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "quantized qmm_t_splitk", TARGETS, MLX_COMMIT
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
    work = tmp_path_factory.mktemp("quantized-qmm-t-splitk-metal-reference")
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
            variant,
            case,
            data,
            parameters=PARAMETERS,
            workgroup=WORKGROUP,
            bound=BOUND,
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
        "splitK": layout["splitK"],
        "checkedElementCount": len(data["expected"]["y"]),
        "observedInputCount": observed,
        "upstream": upstream,
    }


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_quantized_qmm_t_splitk_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-qmm-t-splitk-", dir=root) as path:
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


def test_ci_requires_native_quantized_qmm_t_splitk_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized split-K matrix-matrix native parity",
        family="quantized-qmm-t-splitk",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmm_t_splitk.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmm_t_splitk.py",
    )
