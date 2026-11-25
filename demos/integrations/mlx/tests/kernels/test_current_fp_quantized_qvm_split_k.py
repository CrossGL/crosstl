"""Prove fp_quantized.metal's fp_qvm_split_k entries natively.

``quantized.cpp``'s ``QuantizedMatmul`` routes fewer than four vectors with
``K`` of 1,024 or more against weights that are not transposed, ``K`` rows of
``N`` codes quantized along ``N``, to ``qvm_split_k``. It splits ``K`` into
``split_k`` parts of ``split_D = ceil(K / split_k)`` rows, the last of
``final_block_size = K - (split_k - 1) * split_D``, adds a batch dimension of
``split_k`` to the activation, weight and scale views, and lets each
``tid.z`` write one part's partial products to an intermediate that a
reduction then sums. ``fp_quantized.metal`` instantiates ``fp_qvm_split_k``
with ``split_k`` 8 and 32 for nvfp4, mxfp8 and mxfp4 in float, bfloat16_t and
float16_t: 18 of its 518 entries. Each is translated and pinned per target,
then runs ``(32, 2, 1)`` threadgroups over an
``(M, ceil(N / (2 * group_size)), B * split_k)`` grid.

The kernel runs ``fp_qvm_impl`` without a global scale, with the part's size
as ``in_vec_size`` and the whole ``K`` as the activation row stride, so the
fp_qvm gate's lane-level model replays it part by part. The fp modes have no
biases, so the host binds the kernel's fourteen buffers in declaration order.
The cases bind the host's own views, split sizes and final block sizes; they
reach parts made of whole blocks, of blocks and a last pass, and a final part
of another size, with one or more rows, batched weights, broadcast weights and
a second simdgroup past ``N``. E2M1 entries also run parts below one block:
there the lanes past the part decode a record they never loaded, times zeros,
which only E2M1's finite patterns make harmless, and the host never sends
such parts, because ``K`` of 1,024 or more gives every part at least 121
rows. ``quantized.cpp`` picks ``split_k`` 32 only above ``K = 8,192``; those
entries run the same formulas at smaller ``K``, which the kernel cannot tell
apart. Each target must produce the model's words and leave every guard
untouched; on Metal the translation must also match the unchanged upstream
kernel word for word.
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
    GUARD_COUNT,
    INDEX_BOUND,
    MODES,
    OPENGL_INDEX_RANGE_ASSERTIONS,
    QMV_PARAMETERS,
    SENTINEL,
    _code_value,
    _fraction_bits,
    _scale_value,
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
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    CI_SHARD_COUNT,
    QMV_FP_HEADERS,
    REQUIRE_ENV,
    SHARD_COUNT_ENV,
    SHARD_INDEX_ENV,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    BLOCK,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    ENTRIES as QVM_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    MUTATIONS as QVM_MUTATIONS,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    SPLIT_K,
    VECTOR_LIMIT,
    _check_instantiation,
    _check_kernel,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    _expected_validation_contract as _qvm_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    _simulate as _qvm_simulate,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import _tile
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    ARRAY_TYPES,
    NUM_SIMDGROUPS,
    SCALAR_TYPES,
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
    ROOT / "demos/integrations/mlx/contracts/"
    "fp_quantized_qvm_split_k.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "fp_qvm_split_k"
SPLIT_KS = (8, 32)
# fp_qvm's buffers without the global scale, then the final part's size.
_NAMES = [name for name, _ in QMV_PARAMETERS if name != "global_scale"]
SPLIT_K_PARAMETERS = tuple(
    (name, index) for index, name in enumerate((*_NAMES, "final_block_size"))
)
# quantized.cpp picks split_k 32 only above this K.
WIDE_SPLIT_THRESHOLD = 8192

# (M, N in groups, K) and the activation and weight layout of each case, with
# each case's split_D and final_block_size in the comments.
SHAPES = {
    # Parts of a last pass alone: 13 lanes, then 9.
    "tail": (1, 1, 100),
    # Parts of one whole block.
    "whole": (1, 2, 256),
    # A block and a last pass: 38, then 34.
    "stacked": (1, 1, 300),
    "broadcast": (2, 2, 264),
    # A block and a three-lane last pass, then a whole final part.
    "ragged": (2, 3, 277),
    # A K quantized.cpp routes here: four blocks and a last pass per part.
    "routed": (1, 1, 1100),
    # Parts of a last pass alone: 3 lanes, then 2.
    "tail-32": (2, 3, 95),
    "whole-32": (1, 1, 1024),
    # A block and a last pass, then a whole final part: 33, then 32.
    "ragged-32": (2, 1, 1055),
    "stacked-32": (1, 3, 1056),
    "broadcast-32": (1, 2, 1088),
}
LAYOUTS = {
    "tail": "unbatched",
    "whole": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "ragged": "unbatched",
    "routed": "unbatched",
    "tail-32": "unbatched",
    "whole-32": "unbatched",
    "ragged-32": "unbatched",
    "stacked-32": "stacked",
    "broadcast-32": "broadcast",
}
# Cheaper cases first: mutations stop at the first case that exposes them.
CASES = {
    8: ("tail", "whole", "stacked", "broadcast", "ragged", "routed"),
    32: ("tail-32", "whole-32", "ragged-32", "stacked-32", "broadcast-32"),
}
# Parts below one block: the lanes past the part pass qouter a vec_w they
# never loaded, times zeros. Every E2M1 pattern is finite, so that adds
# zeros; an E4M3 pattern may be NaN, so mxfp8 entries skip these cases.
SHORT_PART_CASES = ("tail", "tail-32")
MUTATIONS = {
    name: text
    for name, text in QVM_MUTATIONS.items()
    # fp_qvm_split_k passes fp_qvm_impl no global scale.
    if name not in ("global-scale", "expert-scale")
}
MUTATIONS["final"] = "gives every part the final part's size"
MUTATIONS["stride"] = "steps activation rows by the part's size, not by K"


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for split_k in SPLIT_KS:
                entry = f"{mode}_qvm_split_k_{itype}_gs_{group_size}_b_{bits}"
                yield entry + f"_spk_{split_k}", {
                    "kernel": KERNEL,
                    # adjust_matrix_offsets' batched overload, as fp_qvm's
                    # batched entries run it.
                    "variant": "batch_1",
                    "mode": mode,
                    "itype": itype,
                    "dtype": dtype,
                    "groupSize": group_size,
                    "bits": bits,
                    "globalScale": False,
                    "splitK": split_k,
                }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    return config_for(directory, target, entry, INFO[entry])


def _cases(entry):
    info = INFO[entry]
    cases = CASES[info["splitK"]]
    if info["bits"] == 8:
        cases = tuple(case for case in cases if case not in SHORT_PART_CASES)
    return cases


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
    }


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """fp_qvm's inputs of one case, bound through qvm_split_k's views."""
    info = INFO[entry]
    assert case in _cases(entry), (entry, case)
    m, groups, k = SHAPES[case]
    n = groups * info["groupSize"]
    layout = layout_for(info, LAYOUTS[case], m, n, k, transposed=True)
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    data = case_data_for(info, entry, case, layout, "fp-quantized-qvm-split-k", None)
    split_k = info["splitK"]
    split_d, final = _split_sizes(k, split_k)
    assert 0 < final <= split_d, (case, split_k)
    params = _split_params(
        layout["params"], split_k, split_d, layout["words"], layout["groups"]
    )
    parts = len(layout["batches"]) * split_k
    grid = (m, -(-n // _tile(info)), parts)
    initial = [SENTINEL] * (parts * m * n + GUARD_COUNT)
    values = {
        name: value for name, value in data["values"].items() if name != "global_scale"
    }
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
    assert set(values) == {name for name, _ in SPLIT_K_PARAMETERS}, entry
    split = {
        **data,
        "layout": {
            **layout,
            "params": params,
            "splitK": split_k,
            "splitD": split_d,
            "finalBlockSize": final,
            "grid": grid,
        },
        "values": values,
        "initial": {"y": initial},
        "grid": grid,
    }
    split["expected"], split["totals"], split["budget"] = _partials(info, split)
    return split


def _partials(info, data):
    """Each part's dequantized partial dot products, as the intermediate holds.

    The budget is fp_qmv_model's: every product is a multiple of ``2^-f`` and
    a part's products' magnitudes sum below ``2^(24 - f)``, so every partial
    sum the kernel forms is exact whatever its order.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    split_k, split_d = layout["splitK"], layout["splitD"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget = 0.0
    for b, batch in enumerate(layout["batches"]):
        for part in range(split_k):
            start = part * split_d
            stop = k if part == split_k - 1 else start + split_d
            z = b * split_k + part
            for row in range(m):
                base = batch["x"] + row * k
                for col in range(n):
                    total, bound, digits = 0.0, 0.0, 0
                    for index in range(start, stop):
                        line = batch["w"] * k + index
                        scale = _scale_value(
                            mode, scales[line * groups + col // group_size]
                        )
                        code = data["codes"][line][col]
                        term = x[base + index] * _code_value(bits, code) * scale
                        total += term
                        bound += abs(term)
                        digits = max(digits, _fraction_bits(term))
                    share = bound / 2.0 ** (24 - digits)
                    assert share < 1 and _f32(total) == total, (info, total, bound)
                    budget = max(budget, share)
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


def _sizes(split_k, mutation):
    """fp_qvm_split_k's ``in_vec_size_adj`` and ``in_vec_stride`` for ``tid.z``."""

    def sizes(storage, z, _mutation):
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
    """The kernel lane by lane: fp_qvm_impl on each tid.z's part."""
    info = INFO[entry]
    qvm_mutation = mutation if mutation in QVM_MUTATIONS else None
    sizes = _sizes(info["splitK"], mutation)
    return _qvm_simulate(entry, data, qvm_mutation, info=info, sizes=sizes)


def _expected_paths(entry, case):
    """The fp_qvm_impl paths a case's parts take, from its shape alone.

    A part of ``size`` rows runs whole blocks when ``size`` is a multiple of
    the block, blocks and a last pass when it is longer, and a last pass
    alone otherwise. A simdgroup whose first column is at or past ``N``
    returns early.
    """
    info = INFO[entry]
    m, groups, k = SHAPES[case]
    split_d, final = _split_sizes(k, info["splitK"])
    paths = set()
    for size in {split_d, final}:
        if size % BLOCK == 0:
            paths.add("whole")
        else:
            paths.add("blocks-tail" if size > BLOCK else "tail-only")
    # Each simdgroup takes one group of columns.
    if groups % NUM_SIMDGROUPS:
        paths.add("exit")
    return paths


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
        raise RuntimeError(
            f"fp quantized qvm_split_k shard {index} is outside [0, {count})"
        )
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


def _dispatch_count():
    return sum(len(_cases(entry)) for entry in ENTRIES)


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        len(_case_data(entry, case)["expected"]["y"])
        for entry in ENTRIES
        for case in _cases(entry)
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
            "fp_qvm_impl lane by lane on each tid.z's part after "
            "adjust_matrix_offsets over the split views: split_D rows, or "
            "final_block_size for the last part, with K as the row stride and "
            "no global scale, against direct dequantized partial dot products"
        ),
        "packing": qvm["packing"],
        "weights": qvm["weights"],
        "split": (
            "split_D = ceil(K / split_k), final_block_size = K - (split_k - 1) "
            "* split_D; quantized.cpp's views add a split_k batch dimension"
        ),
        "hostBuffers": (
            "the fp modes have no biases, so quantized.cpp binds the kernel's "
            "fourteen buffers in declaration order"
        ),
        "comparison": qvm["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / (2 * group_size)), B * split_k)",
        "n": qvm["n"],
        "cases": {str(split_k): _case_shapes(split_k) for split_k in SPLIT_KS},
        "e2m1OnlyCases": list(SHORT_PART_CASES),
        "shortParts": (
            "parts below one block run only on E2M1 entries: the lanes past "
            "the part would decode a vec_w they never loaded, and K >= 1024 "
            "gives every part the host sends at least 121 rows"
        ),
        "inputs": qvm["inputs"],
        "exactness": qvm["exactness"],
        "guards": qvm["guards"],
        "softwareSubgroups": qvm["softwareSubgroups"],
        "openglIndexRangeAssertions": qvm["openglIndexRangeAssertions"],
        "metalUpstreamComparison": qvm["metalUpstreamComparison"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qvmSplitK"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qvm-split-k-contract",
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


def test_current_mlx_fp_quantized_qvm_split_k_contract_is_exact():
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


def test_fp_quantized_qvm_split_k_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qvm_split_k.current-tree.translation.json" in readme
    assert len(ENTRIES) == 18 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qvmSplitK"]
    assert Counter(info["splitK"] for info in INFO.values()) == {8: 9, 32: 9}
    assert ENTRIES[:2] == (
        "nvfp4_qvm_split_k_float_gs_16_b_4_spk_8",
        "nvfp4_qvm_split_k_float_gs_16_b_4_spk_32",
    )
    assert ENTRIES[-1] == "mxfp4_qvm_split_k_float16_t_gs_32_b_4_spk_32"
    assert KERNEL not in EXCLUDED_KERNELS
    assert not set(ENTRIES) & set(QVM_ENTRIES)
    assert set(SHAPES) == set(LAYOUTS) == {c for cases in CASES.values() for c in cases}
    assert [name for name, _ in SPLIT_K_PARAMETERS][-1] == "final_block_size"


def test_fp_quantized_qvm_split_k_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    # The two split sizes alternate, so each shard runs one of them over every
    # mode and element type.
    for index, shard in enumerate(shards):
        assert Counter(INFO[entry]["splitK"] for entry in shard) == {SPLIT_KS[index]: 9}
        assert Counter(INFO[entry]["mode"] for entry in shard) == {
            mode: 3 for mode, _, _ in MODES
        }
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_fp_quantized_qvm_split_k_cases_follow_mlx_dispatch():
    for split_k in SPLIT_KS:
        for case in CASES[split_k]:
            m, groups, k = SHAPES[case]
            split_d, final = _split_sizes(k, split_k)
            # The last part is never empty and never longer than the others.
            assert 0 < final <= split_d
            assert (split_k - 1) * split_d + final == k
            # Only the E2M1-only cases have parts below one block.
            assert (min(split_d, final) < BLOCK) == (case in SHORT_PART_CASES)
            assert m < VECTOR_LIMIT
    # The routed case is one quantized.cpp sends to split_k 8 itself.
    assert SPLIT_K <= SHAPES["routed"][2] <= WIDE_SPLIT_THRESHOLD
    assert all(SHAPES[case][2] <= WIDE_SPLIT_THRESHOLD for case in CASES[32])
    # quantized.cpp never sends a part below one block: K >= 1024 gives
    # split_k 8 parts of at least 121 rows and split_k 32 parts above K = 8192
    # at least 225.
    for split_k, k in ((8, SPLIT_K), (32, WIDE_SPLIT_THRESHOLD + 1)):
        smallest = min(
            min(_split_sizes(size, split_k)) for size in range(k, k + 4 * split_k)
        )
        assert smallest >= 121, (split_k, smallest)
    assert {len(_cases(entry)) for entry in ENTRIES if INFO[entry]["bits"] == 8} == {
        4,
        5,
    }


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qvm_split_k_views_follow_quantized_cpp(entry):
    info = INFO[entry]
    split_k = info["splitK"]
    for case in _cases(entry):
        data = _case_data(entry, case)
        layout = data["layout"]
        params = layout["params"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        split_d, words = layout["splitD"], layout["words"]
        assert n % info["groupSize"] == 0
        assert data["grid"] == (m, -(-n // _tile(info)), batches * split_k)
        assert len(data["expected"]["y"]) == batches * split_k * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [split_d]
        assert values["out_vec_size"][1] == [n]
        assert values["final_block_size"][1] == [layout["finalBlockSize"]]
        assert "b_strides" not in params and "global_scale" not in values
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
        # (K, N) weights: K rows of N * bits / 32 words and N / group_size groups.
        assert words == n * info["bits"] // 32
        assert layout["groups"] == n // info["groupSize"]


def test_fp_quantized_qvm_split_k_cases_reach_every_path():
    every = {"exit", "whole", "blocks-tail", "tail-only"}
    for entry in ENTRIES:
        reached = set().union(*(_expected_paths(entry, c) for c in _cases(entry)))
        # Only E2M1 entries run parts below one block.
        if INFO[entry]["bits"] == 4:
            assert reached == every, entry
        else:
            assert reached == every - {"tail-only"}, entry


def test_fp_quantized_qvm_split_k_inputs_reach_every_code():
    for entry in ENTRIES:
        info = INFO[entry]
        codes = Counter()
        for case in _cases(entry):
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
def test_fp_quantized_qvm_split_k_model_matches_partial_products(entry):
    for case in _cases(entry):
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        assert run["paths"] == _expected_paths(entry, case), (entry, case)
        layout = data["layout"]
        m, k = layout["M"], layout["K"]
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
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
        assert run["reads"]["scales"] == {
            (matrix * k + row) * groups + group
            for matrix in matrices
            for row in range(k)
            for group in range(groups)
        }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qvm_split_k_inputs_keep_float32_exact(dtype):
    rounded = 0
    for entry in ENTRIES:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in _cases(entry):
            data = _case_data(entry, case)
            assert data["budget"] < 1
            for name, (kind, values) in data["values"].items():
                if kind == dtype:
                    assert all(_to_type(dtype, value) == value for value in values)
            for total, value in zip(data["totals"], data["expected"]["y"]):
                assert _f32(total) == total and value == value
                rounded += value != total
    if dtype == "float32":
        assert rounded == 0
    else:
        assert rounded > 0


@pytest.mark.parametrize(
    "entry,mutation",
    [(entry, mutation) for entry in ENTRIES for mutation in MUTATIONS],
)
def test_fp_quantized_qvm_split_k_expectations_reject_defects(entry, mutation):
    for case in _cases(entry):
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: {MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qvm_split_k_opengl_assertions_name_buffer_elements(entry):
    used = set()
    for case in _cases(entry):
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _simulate(entry, data)["records"]:
            used.add(expression)
            assert 0 <= value < lengths[buffer] - GUARD_COUNT, (case, expression)
            assert value <= INDEX_BOUND
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS["batch_1"])


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_qvm_split_k_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES:
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        # No global scale, so no division profile.
        assert options == {
            "directx": {
                "software_subgroup_width": SIMD_SIZE,
                "relative_wave_shuffle_out_of_range": "self",
            },
            "opengl": {"software_subgroup_width": SIMD_SIZE},
        }
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "opengl":
            assert expressions == list(OPENGL_INDEX_RANGE_ASSERTIONS["batch_1"])
        else:
            assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


def _check_split_instantiation(text):
    """ENTRIES follow fp_quantized.metal's split-K instantiation macros."""
    _check_instantiation(text)
    groups = _macro(text, "instantiate_quantized_modes")
    assert "instantiate_quantized_all_splitk(type, mode, group_size, bits)" in groups
    splits = re.findall(
        r"instantiate_quantized_split_k\(mode, qvm_split_k, type, (\d+), "
        r"group_size, bits\)",
        _macro(text, "instantiate_quantized_all_splitk"),
    )
    assert splits == [str(split_k) for split_k in SPLIT_KS]
    name = _macro(text, "instantiate_quantized_split_k")
    for required in (
        '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k',
        "fp_ ## name,",
    ):
        assert required in name, required


def _check_split_kernel(header):
    _check_kernel(header)
    assert declared_buffers(header, KERNEL) == SPLIT_K_PARAMETERS
    body = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "int M = x_shape[x_batch_ndims];",
        "out_vec_size * M,",
        "tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;",
        "int in_vec_stride = (split_k - 1) * in_vec_size + final_block_size;",
        "fp_qvm_impl<T, group_size, bits, false>(",
        "nullptr,",
        "in_vec_size_adj,",
        "in_vec_stride,",
    ):
        assert required in body, required


def _check_split_host(host):
    start = host.index("void qvm_split_k(")
    body = host[start : host.index("void qvm(", start)]
    for required in (
        "int split_k = K > 8192 ? 32 : 8;",
        "int split_D = (K + split_k - 1) / split_k;",
        "B *= split_k;",
        "int bn = std::min(group_size, 32) * num_simdgroups;",
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
        "int final_block_size = K - (split_k - 1) * split_D;",
        'mode + "_qvm_split_k_",',
        "compute_encoder.set_input_array(w, c++);\n"
        "  compute_encoder.set_input_array(scales, c++);\n"
        "  if (biases) {",
        "compute_encoder.set_input_array(x, c++);\n"
        "  compute_encoder.set_output_array(intermediate, c++);\n"
        "  compute_encoder.set_bytes(split_D, c++);\n"
        "  compute_encoder.set_bytes(N, c++);",
        "compute_encoder.set_bytes(final_block_size, c++);",
    ):
        assert required in body, required
    # QuantizedMatmul sends K >= 1024 here, with no global scale.
    for required in (
        "  // Run of the mill qvm\n  if (K < 1024) {",
        "qvm_split_k(\n      x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);",
    ):
        assert required in host, required


def _check_ops(ops):
    """quantized_matmul passes the fp modes x, w and scales only: no biases."""
    body = ops[ops.index("array quantized_matmul(") :]
    body = body[: body.index("\n}\n")]
    assert (
        "  } else {\n    inputs = {x, w, scales};\n  }" in body
    ), "quantized_matmul's fp inputs"


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qvm split-K", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_FP_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    _check_split_instantiation(source.decode("utf-8"))
    kernels = root / "mlx/backend/metal/kernels/fp_quantized.h"
    _check_split_kernel(kernels.read_text(encoding="utf-8"))
    _check_split_host((root / HOST_SOURCE).read_text(encoding="utf-8"))
    _check_ops((root / "mlx/ops.cpp").read_text(encoding="utf-8"))
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-qvm-split-k-metal-reference")
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
def test_current_mlx_fp_quantized_qvm_split_k_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-fpsplitk-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = translate_entry(root, work, target, config, record)
            executor = _executor(target)
            evidence = []
            try:
                for case in _cases(entry):
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, "batch_1", data, SPLIT_K_PARAMETERS
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
                            SPLIT_K_PARAMETERS,
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


def test_ci_requires_native_fp_quantized_qvm_split_k_parity():
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qvm_split_k.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qvm_split_k.py",
    )
