"""Prove fp_quantized.metal's fp_qmm_t_splitk entries natively.

``quantized.cpp``'s ``QuantizedMatmul`` sends an unbatched product with
transposed weights and ``M`` at or above the vector limit to ``qmm_splitk``
on every device. It aims at about 512 threadgroups: ``split_k`` is 512 over
the output's 32 by 32 tiles, at most ``K / k_align`` with ``k_align`` the
larger of the group size and 32, and lowered until ``split_k * k_align``
divides ``K``. Every fp mode's groups hold at most 32 codes, so each
partition is a whole number of 32-code blocks. With ``split_k > 1`` it
dispatches ``fp_qmm_t_splitk`` over a ``(ceil(N / 32), ceil(M / 32),
split_k)`` grid of ``(32, 2, 2)`` threadgroups into a ``(split_k, M, N)``
intermediate, which a strided sum reduction then folds; otherwise it falls
back to ``qmm``, which ``test_current_fp_quantized_qmm_t.py`` gates.
``fp_quantized.metal`` instantiates it for nvfp4 (groups of 16 E2M1 codes with
E4M3 scales), mxfp8 (groups of 32 E4M3 codes with E8M0 scales) and mxfp4
(groups of 32 E2M1 codes with E8M0 scales) in float, bfloat16_t and
float16_t, with aligned and unaligned ``N``: 18 of its 518 entries. Each is
translated and pinned per target.

The kernel moves ``x``, the weight bytes and the scales to its partition's
first reduction column, ``tid.z * k_partition_size``, and ``y`` to its
partition's ``M`` by ``N`` slice, then runs ``fp_qmm_t_impl`` without a
global scale over ``k_partition_size`` columns while its rows keep the stride
``K``; the weights reach ``fp_qmm_t_impl`` as a ``uint32_t`` view of the moved
byte pointer. Whole blocks leave ``fp_qmm_t_impl`` no ``num_k`` tail. The
cases split ``K`` into two and three partitions of one block and four of two
blocks, with the fp_qmm_t gate's tiles and the fp_qmv gate's inputs, so they
reach every load and store path. Every dequantized weight is exact, and
float32 holds every product and partial sum, so each partition's partial sums
round once to the element type. A tile-level model replays ``fp_qmm_t_impl``
per partition and must reach each partition's direct dequantized partial dot
products, storing every intermediate element once. Each target must produce
the model's words and leave every guard untouched; on Metal the translation
must also match the unchanged upstream kernel word for word.
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

from demos.integrations.mlx.tests.kernels import fp_qmm_t_model as model
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    GUARD_COUNT,
    MODES,
    SENTINEL,
    _code_value,
    _fraction_bits,
    _scale_value,
    case_data_for,
    translate_entry,
)
from demos.integrations.mlx.tests.kernels.fp_quantized_scope import (
    EXCLUDED_KERNELS,
    GATE_ENTRY_COUNTS,
    UNIT_ENTRY_COUNT,
    shared_scope,
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
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    DTYPES,
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmm_t import (
    CI_SHARD_COUNT,
    HELPER_ELEMENTS,
    QMM_FP_HEADERS,
    REQUIRE_ENV,
    SHARD_COUNT_ENV,
    SHARD_INDEX_ENV,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmm_t import (
    _expected_validation_contract as _qmm_t_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmm_t import (
    fp_qmm_config,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm_split_k import (
    _check_ops,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    DIMENSION_LIMIT,
    K_LIMIT,
    SIGNED_MAX,
    VECTOR_LIMIT,
    WORKGROUP,
    _split_k,
    _x_loader,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    SIMD_SIZE,
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
    ROOT / "demos/integrations/mlx/contracts/"
    "fp_quantized_qmm_t_splitk.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "fp_qmm_t_splitk"
# instantiate_quantized_all_splitk's qmm_t_splitk order.
VARIANTS = (("alN_true", True), ("alN_false", False))
PARAMETERS = (
    ("w", 0),
    ("scales", 1),
    ("x", 2),
    ("y", 3),
    ("K", 4),
    ("N", 5),
    ("M", 6),
    ("k_partition_size", 7),
    ("split_k_partition_stride", 8),
)
# qmm_splitk binds every parameter.
BOUND = len(PARAMETERS)
# The K_eff loop runs to k_partition_size; the rows still stride by K.
BOUNDED = ("K", "k_partition_size")

CASES = ("halves", "thirds", "deep")
# (M, aligned N, unaligned N, K, split_k).
SHAPES = {
    # Two row tiles, the second of 8 rows; two column tiles, the second of 8
    # columns when N is unaligned; two partitions of one block.
    "halves": (40, 64, 40, 64, 2),
    # A row tile of one row, three column tiles and three partitions of one
    # block.
    "thirds": (33, 96, 72, 96, 3),
    # One column tile, of 8 columns when N is unaligned, and four partitions
    # of two blocks, as qmm_splitk splits DEEP_HOST_SHAPE.
    "deep": (33, 32, 8, 256, 4),
}
# qmm_splitk splits a product into partitions of several blocks only past
# 512 / tiles blocks: N = 2048 gives 128 tiles and 512 / 128 = 4 splits of
# K = 256, two blocks each, which "deep"'s partitions stand for.
DEEP_HOST_SHAPE = (33, 2048, 256)


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for suffix, aligned in VARIANTS:
                entry = f"{mode}_qmm_t_splitk_{itype}_gs_{group_size}_b_{bits}_{suffix}"
                yield entry, {
                    "kernel": KERNEL,
                    # Unbatched: qmm_splitk runs only for B == 1.
                    "variant": "batch_0",
                    "aligned": aligned,
                    "mode": mode,
                    "itype": itype,
                    "dtype": dtype,
                    "groupSize": group_size,
                    "bits": bits,
                    # fp_qmm_t_splitk passes fp_qmm_t_impl no global scale.
                    "globalScale": False,
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
    writes, weight and scale bytes, and threadgroup tile slots. GLSL indexes
    buffers with 32-bit integers, so each buffer index is asserted to fit one.
    """
    element = HELPER_ELEMENTS[INFO[entry]["dtype"]]
    items = [
        (f"source.offset + {index}", None, SIGNED_MAX, "x-vector")
        for index in range(BK * BM // THREADS)
    ]
    for access, family in (
        (f"load_{element}", "elements"),
        (f"store_{element}", "y"),
        ("load_uint8", "bytes"),
    ):
        scope = {"function": f"crosstl_resource_{access}"}
        items.append(("reference.offset + index", scope, SIGNED_MAX, family))
    last = (BM - 1) * padded(INFO[entry]["itype"]) + BK - 1
    for tile in ("Xs", "Ws"):
        scope = {"indexed": f"crosstl_workgroup_{tile}"}
        items.append(("reference.offset + index", scope, last, tile))
    return items


def _config(directory, target, entry):
    """The translation of one entry, with the fp_qmm_t gate's options."""
    return fp_qmm_config(
        directory,
        target,
        entry,
        INFO[entry],
        _workgroup_access_assertions(entry),
        _opengl_index_items(entry),
        bounded=BOUNDED,
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    m, aligned_n, unaligned_n, k, _ = SHAPES[case]
    return m, aligned_n if INFO[entry]["aligned"] else unaligned_n, k


def _partials(info, data):
    """Each partition's dequantized partial dot products, as ``y`` holds them.

    Every weight is ``T(f32(T(scale) * code))``, as the loader computes it
    without a global scale, and exact. The budget bounds the float32
    arithmetic whatever its order: every product is a multiple of ``2^-f``
    and each partition's products' magnitudes sum below ``2^(24 - f)``, so
    every partial sum is exact. It is the largest such sum, in units of
    ``2^(24 - f)``. ``rounded`` counts the weights the element type rounds.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    size = layout["partitionSize"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget, rounded = 0.0, 0
    for part in range(layout["splitK"]):
        for row in range(m):
            for col in range(n):
                total, bound, digits = 0.0, 0.0, 0
                for index in range(part * size, (part + 1) * size):
                    # Weight row ``col`` holds output ``col``'s K codes.
                    scale = _scale_value(
                        mode, scales[col * groups + index // group_size]
                    )
                    scale = _f32(_to_type(dtype, scale))
                    product = _f32(scale * _code_value(bits, data["codes"][col][index]))
                    weight = _to_type(dtype, product)
                    rounded += weight != product
                    term = x[row * k + index] * weight
                    total += term
                    bound += abs(term)
                    digits = max(digits, _fraction_bits(term))
                share = bound / 2.0 ** (24 - digits)
                assert share < 1 and _f32(total) == total, (info, total, bound)
                budget = max(budget, share)
                position = (part * m + row) * n + col
                totals[position] = total
                expected[position] = _to_type(dtype, total)
    return {"y": expected}, totals, budget, rounded


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The fp_qmv gate's inputs, bound as fp_qmm_t_splitk's parameters."""
    info = INFO[entry]
    assert case in CASES, (entry, case)
    m, n, k = _shape(entry, case)
    split_k = SHAPES[case][4]
    assert split_k > 1 and k % (split_k * BK) == 0, (entry, case, split_k)
    layout = layout_for(info, "unbatched", m, n, k)
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    data = case_data_for(info, entry, case, layout, "fp-quantized-qmm-t-splitk", None)
    plain = data["values"]
    initial = [SENTINEL] * (split_k * m * n + GUARD_COUNT)
    values = {
        **{name: plain[name] for name in ("w", "scales", "x")},
        "y": (info["dtype"], initial),
        "K": plain["in_vec_size"],
        "N": plain["out_vec_size"],
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
    split["expected"], split["totals"], split["budget"], split["rounded"] = _partials(
        info, split
    )
    return split


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    **model.MUTATIONS,
    "partition-x": "keeps the activations on the first partition",
    "partition-w": "keeps the weight bytes on the first partition",
    "partition-scales": "keeps the scales on the first partition",
    "partition-size": "runs every partition over all of K",
    "partition-stride": "places the partitions N outputs apart",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "scale-split":
        # Groups of 32 give every thread of a block the same group.
        return info["groupSize"] < BK
    if mutation in ("batch", "tail", "tail-zero", "global-scale", "fold-scale"):
        # One unbatched product without a global scale, in whole blocks.
        return False
    return mutation in MUTATIONS


def _offsets(info):
    """fp_qmm_t_splitk's partition offsets for ``tid.z``."""
    geometry = model.geometry(info["bits"], info["groupSize"])

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
        return x, byte // 4, group, None, z * stride

    return offsets


def _tile(run, context, bases, tid_x, tid_y, mutation):
    """fp_qmm_t_impl for one threadgroup, ``K_eff`` columns of its partition."""
    if mutation == "partition-size":
        context = {**context, "Keff": context["K"]}
    run(context, bases, tid_x, tid_y, mutation)


def _simulate(entry, data, mutation=None):
    """fp_qmm_t_model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    info = INFO[entry]
    return model.simulate(
        info, data, families, mutation, offsets=_offsets(info), tile=_tile
    )


def _expected_paths(entry, case):
    m, n, _ = _shape(entry, case)
    info = INFO[entry]
    size = SHAPES[case][3] // SHAPES[case][4]
    return model.expected_paths(info["aligned"], m, n, size, info["groupSize"])


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
        raise RuntimeError(f"fp quantized qmm shard {index} is outside [0, {count})")
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
            m, n, _ = _shape(entry, case)
            total += SHAPES[case][4] * m * n + GUARD_COUNT
    return total


def _case_shapes(aligned):
    shapes = {}
    for case in CASES:
        m, aligned_n, unaligned_n, k, split_k = SHAPES[case]
        shapes[case] = {
            "M": m,
            "N": aligned_n if aligned else unaligned_n,
            "K": k,
            "splitK": split_k,
        }
    return shapes


def _expected_validation_contract():
    qmm_t = _qmm_t_validation_contract()
    return {
        "model": (
            "fp_qmm_t_impl tile by tile per partition after fp_qmm_t_splitk's "
            "partition offsets: the BlockLoader and fp QuantizedBlockLoader "
            "blocks over k_partition_size columns with their safe loads, one "
            "scale per thread's eight codes stepped by 32 / group_size per "
            "block, BlockMMA's float32 fragment products and the whole or "
            "safe stores into the partition's slice, against direct "
            "dequantized partial dot products"
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
            "32) divides K; k_partition_size = K / split_k, whole blocks of "
            "32 codes, and split_k_partition_stride = M * N"
        ),
        "routing": (
            "unbatched products with transposed weights, M at or above "
            "get_qmv_batch_limit's largest limit and split_k > 1, on every "
            "device; alN_true exactly when N is a multiple of 32"
        ),
        "cases": {suffix: _case_shapes(aligned) for suffix, aligned in VARIANTS},
        "deepHostShape": dict(zip(("M", "N", "K"), DEEP_HOST_SHAPE)),
        "inputs": qmm_t["inputs"],
        "exactness": (
            "every dequantized weight is exact in the element type and every "
            "product a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f) per partition; only each partition's final "
            "static_cast<T> rounds, to nearest even"
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
            "buffer element and weight and scale byte indices fit GLSL's "
            "32-bit indices; tile indices stay within the BM x BK slots"
        ),
        "metadataPreconditions": {name: [0, K_LIMIT] for name in BOUNDED},
        "upstreamBuffers": BOUND,
        "metalUpstreamComparison": qmm_t["metalUpstreamComparison"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qmmTSplitK"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmm-t-splitk-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(QMM_FP_HEADERS),
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


def test_current_mlx_fp_quantized_qmm_t_splitk_contract_is_exact():
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


def test_fp_quantized_qmm_t_splitk_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmm_t_splitk.current-tree.translation.json" in readme
    assert len(ENTRIES) == 18 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qmmTSplitK"]
    assert Counter(info["aligned"] for info in INFO.values()) == {True: 9, False: 9}
    assert ENTRIES[:2] == (
        "nvfp4_qmm_t_splitk_float_gs_16_b_4_alN_true",
        "nvfp4_qmm_t_splitk_float_gs_16_b_4_alN_false",
    )
    assert ENTRIES[-1] == "mxfp4_qmm_t_splitk_float16_t_gs_32_b_4_alN_false"
    assert KERNEL not in EXCLUDED_KERNELS
    assert set(SHAPES) == set(CASES)


def test_fp_quantized_qmm_t_splitk_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [9, 9]
    for shard in shards:
        # Each shard runs every mode and element type; the alignments
        # alternate, so each shard keeps one.
        for key in ("mode", "dtype"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    assert {INFO[entry]["aligned"] for entry in shards[0]} == {True}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmm_t_splitk_cases_follow_mlx_dispatch(entry):
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
        size = data["values"]["k_partition_size"][1][0]
        # Each partition is a whole number of groups and of BK-wide blocks.
        assert size * split_k == k and size % info["groupSize"] == size % BK == 0
        if case == "deep":
            # The partitions qmm_splitk gives DEEP_HOST_SHAPE.
            host_m, host_n, host_k = DEEP_HOST_SHAPE
            assert (host_m, host_k) == (m, k)
            assert _split_k(host_m, host_n, host_k, info["groupSize"]) == split_k
            assert VECTOR_LIMIT <= host_m and host_n <= DIMENSION_LIMIT
        else:
            assert _split_k(m, n, k, info["groupSize"]) == split_k
        assert data["values"]["split_k_partition_stride"][1] == [m * n]
        assert data["grid"] == (-(-n // BN), -(-m // BM), split_k)
        assert len(data["expected"]["y"]) == split_k * m * n + GUARD_COUNT
        values = data["values"]
        assert values["K"][1] == [k] and values["N"][1] == [n] and values["M"][1] == [m]
        assert layout["words"] == k * info["bits"] // 32
        assert layout["groups"] == k // info["groupSize"]
        assert 0 <= size <= k <= K_LIMIT


def test_fp_quantized_qmm_t_splitk_split_follows_qmm_splitk():
    # Every fp group fits one block, so k_align is 32 and a K of q blocks
    # splits into q partitions while 512 / tiles allows.
    assert _split_k(40, 64, 64, 16) == _split_k(40, 64, 64, 32) == 2
    assert _split_k(33, 96, 96, 32) == 3
    # Past 512 / tiles the largest divisor of q that fits wins: 4 of 8.
    assert _split_k(33, 2048, 256, 16) == 4
    # A prime number of blocks above 512 / tiles leaves no split.
    assert _split_k(33, 2048, 160, 32) == 1
    # Below two blocks qmm_splitk falls back to qmm.
    assert _split_k(40, 64, 32, 16) == 1


def test_fp_quantized_qmm_t_splitk_cases_reach_every_path():
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
    # Whole blocks leave fp_qmm_t_impl no tail.
    assert reached == {
        True: every - {"x-unsafe/w-safe", "x-safe/w-safe"},
        False: every,
    }
    # The partitions hold one block and two.
    assert {SHAPES[case][3] // SHAPES[case][4] // BK for case in CASES} == {1, 2}


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_fp_quantized_qmm_t_splitk_model_matches_partial_dot_products(entry):
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
        # The partitions together read every activation, weight word and
        # scale, no guard, and nothing past a row.
        assert run["reads"]["x"] == set(range(m * k))
        assert run["reads"]["w"] == set(range(n * layout["words"]))
        assert run["reads"]["scales"] == set(range(n * layout["groups"]))
        # fp_qmm_t_splitk takes no global scale.
        assert "global_scale" not in run["reads"]
        assert run["overreads"] == {"w": set(), "scales": set()}


def test_fp_quantized_qmm_t_splitk_model_derives_the_loader_assertions():
    for itype, _ in DTYPES:
        entry = next(e for e in ENTRIES if INFO[e]["itype"] == itype)
        run = _simulate(entry, _case_data(entry, CASES[0]))
        loader = _x_loader(itype)
        slots = {slot for lane in loader.slots() for slot in lane}
        # Each block writes every BlockLoader slot and the tile's Ws slots.
        assert {slot for _, slot in run["records"]["Xs"]} == slots
        assert {slot for _, slot in run["records"]["Ws"]} == slots
        (assertion,) = _workgroup_access_assertions(entry)
        assert (assertion["minimum"], assertion["maximum"]) == (min(slots), max(slots))
        assert assertion["function"] == loader.function


def test_fp_quantized_qmm_t_splitk_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {
        "order": 6,
        "scale-split": 12,
        "batch": 18,
        "tail": 18,
        "tail-zero": 18,
        "global-scale": 18,
        "fold-scale": 18,
    }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmm_t_splitk_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
    for entry in ENTRIES:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in CASES:
            data = _case_data(entry, case)
            assert data["budget"] < 1 and data["rounded"] == 0
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
        (entry, mutation)
        for entry in SELECTED_ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _mutation_cases())
def test_fp_quantized_qmm_t_splitk_expectations_reject_defects(entry, mutation):
    for case in CASES:
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: {MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::2])
def test_fp_quantized_qmm_t_splitk_opengl_assertions_name_buffer_elements(entry):
    items = _opengl_index_items(entry)
    seen = {family: set() for _, _, _, family in items}
    for case in CASES:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        records = _simulate(entry, data)["records"]
        for _, _, maximum, family in items:
            for buffer, index in records[family]:
                seen[family].add(buffer)
                if family == "bytes":
                    words = lengths[buffer] - GUARD_COUNT
                    limit = words * (4 if buffer == "w" else 1)
                elif family in ("Xs", "Ws"):
                    limit = maximum + 1
                else:
                    limit = lengths[buffer] - GUARD_COUNT
                assert 0 <= index < limit <= SIGNED_MAX + 1, (case, family, index)
    assert seen == {
        "x-vector": {"x"},
        "elements": {"x"},
        "y": {"y"},
        "bytes": {"w", "scales"},
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_qmm_t_splitk_configs_scope_target_options(tmp_path, target):
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
            assert "binary32_division_profile" not in options[backend]
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


def _check_instantiation(text):
    """ENTRIES follow fp_quantized.metal's own instantiation macros."""
    types = re.findall(r"^instantiate_quantized_types\((\w+)\)", text, re.MULTILINE)
    assert types == [itype for itype, _ in DTYPES]
    modes = re.findall(
        r"instantiate_quantized_modes\(type, (\w+), (\d+), (\d+)\)",
        _macro(text, "instantiate_quantized_types"),
    )
    assert modes == [(mode, str(size), str(bits)) for mode, size, bits in MODES]
    assert "instantiate_quantized_all_splitk(type, mode, group_size, bits)" in (
        _macro(text, "instantiate_quantized_modes")
    )
    found = re.findall(
        r"instantiate_quantized_aligned\(mode, (\w+), type, (true|false), "
        r"group_size, bits\)",
        _macro(text, "instantiate_quantized_all_splitk"),
    )
    assert found == [("qmm_t_splitk", "true"), ("qmm_t_splitk", "false")]
    body = _macro(text, "instantiate_quantized_aligned")
    assert (
        '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,'
        in (body)
    )
    assert "fp_ ## name," in body


def _check_kernel(header):
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
        "y += tid.z * static_cast<int64_t>(split_k_partition_stride);",
        "fp_qmm_t_impl<T, group_size, bits, aligned_N, false, BM, BK, BN>(",
        "(const device uint32_t*)wl,",
        "      scales,\n      nullptr,\n      x,",
        "      M,\n      k_partition_size,\n      tid,",
    ):
        assert required in body, required
    impl = header[header.index("METAL_FUNC void fp_qmm_t_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "const int K_w = K * bytes_per_pack / pack_factor;",
        "const int K_g = K / group_size;",
        "loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);",
        "const int k_blocks = K_eff / BK;",
        "const short num_k = partial_k ? short(K_eff - k_blocks * BK) : short(0);",
    ):
        assert required in impl, required


def _check_host(host):
    splitk = host[host.index("void qmm_splitk(\n") :]
    splitk = splitk[: splitk.index("\n}\n")]
    for required in (
        "int split_k = std::max(1, 512 / current_tgs);",
        "int k_align = group_size > 32 ? group_size : 32;",
        "split_k = std::min(split_k, K / k_align);",
        "while (split_k > 1 && (K % (split_k * k_align) != 0)) {",
        "int k_partition_size = K / split_k;",
        "int split_k_partition_stride = M * N;",
        "MTL::Size group_dims(32, 2, 2);",
        "MTL::Size grid_dims(n_tiles, m_tiles, split_k);",
        "bool aligned = N % 32 == 0;",
        'mode + "_qmm_t_splitk_",',
        'aligned ? "_alN_true" : "_alN_false");',
        "  if (biases) {\n    compute_encoder.set_input_array(*biases, c++);\n  }",
        "compute_encoder.set_bytes(k_partition_size, c++);",
        "compute_encoder.set_bytes(split_k_partition_stride, c++);",
    ):
        assert required in splitk, required
    for required in (
        "if (transpose_ && B == 1) {",
        "if (D <= 2048 && O <= 2048) {\n      return 33;",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmm_t_splitk", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMM_FP_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    _check_instantiation(source.decode("utf-8"))
    kernels = root / "mlx/backend/metal/kernels/fp_quantized.h"
    _check_kernel(kernels.read_text(encoding="utf-8"))
    _check_host((root / HOST_SOURCE).read_text(encoding="utf-8"))
    _check_ops((root / "mlx/ops.cpp").read_text(encoding="utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-qmm-t-splitk-metal-reference")
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
        "splitK": layout["splitK"],
        "checkedElementCount": len(data["expected"]["y"]),
        "observedInputCount": observed,
        "upstream": upstream,
    }


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_fp_quantized_qmm_t_splitk_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-fpqmmtsk-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            translated = translate_entry(
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


def test_ci_requires_native_fp_quantized_qmm_t_splitk_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX fp quantized matrix-matrix native parity",
        family="fp-quantized-qmm",
        count=CI_SHARD_COUNT,
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        f"{SHARD_INDEX_ENV}: ${{{{ matrix.shard }}}}",
        f'{SHARD_COUNT_ENV}: "{CI_SHARD_COUNT}"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "pytest -q -n auto",
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmm_t_splitk.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmm_t_splitk.py",
    )
