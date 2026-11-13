"""Prove fp_quantized.metal's fp_qmm_n entries natively.

``quantized.cpp``'s ``QuantizedMatmul`` multiplies ``M`` activation rows by
weights that are not transposed, ``K`` rows of ``N`` codes quantized along
``N``. Without transposition the vector limit is 4 and there is no split-K
route, so for ``M >= 4`` it calls ``qmm``, which has no NAX kernel for the fp
modes without transposition and dispatches ``fp_qmm_n``, batched for
``B > 1``. ``fp_quantized.metal`` instantiates ``fp_qmm_n`` for nvfp4 (groups
of 16 E2M1 codes with E4M3 scales), mxfp8 (groups of 32 E4M3 codes with E8M0
scales) and mxfp4 (groups of 32 E2M1 codes with E8M0 scales) in float,
bfloat16_t and float16_t, with and without batching, each also as an
``_hgs`` twin: 36 of its 518 entries. ``fp_qmm_n`` passes ``fp_qmm_n_impl``
no global scale, so a twin computes what its plain entry does; ``qmm`` asserts
that a global scale comes with transposed weights, so the host dispatches only
the plain entries. Each is translated and pinned per target, then runs
``(32, 2, 2)`` threadgroups over a ``(ceil(N / 32), ceil(M / 32), B)`` grid.

``fp_qmm_n_impl`` gives each threadgroup a 32 by 32 output tile. Per block of
32 reduction rows a ``BlockLoader`` of four-element vectors copies the
activations into ``Xs`` and the fp ``QuantizedBlockLoader``, reducing along
its rows, dequantizes 32 weight rows into ``Ws``: each thread reads one scale
for its eight codes and rounds each weight to the element type, and the
scales step a block of rows at a time. ``BlockMMA`` multiplies the tiles in
float32. A ``K`` that is not a multiple of 32 ends in a pass over the
remaining rows of both tiles. Groups of 16 let ``N`` end in a tile of 16
columns; its weight loads bound only rows, so the threads past ``N``
dequantize the next row's first bytes and scale, or the guards after the
last row, into columns ``store_result_safe`` never writes. Inputs are the
fp_qvm gate's: every dequantized weight is exact, and float32 holds every
product and partial sum, so only the final ``static_cast<T>`` rounds. A
tile-level model replays both loaders, the last pass and the fragment
products and must reach direct dot products of the dequantized weights,
storing every output once. Each target must produce the model's words and
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

from demos.integrations.mlx.tests.kernels import fp_qmm_n_model as model
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    declared_buffers,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    GUARD_COUNT,
    MODES,
    case_data_for,
    translate_entry,
)
from demos.integrations.mlx.tests.kernels.fp_quantized_scope import (
    EXCLUDED_KERNELS,
    GATE_ENTRY_COUNTS,
    UNIT_ENTRY_COUNT,
    shared_scope,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_n_model import X_READS
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import (
    BK,
    BM,
    BN,
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
    fp_qmm_config,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmm_t import (
    _expected_validation_contract as _qmm_t_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm_split_k import (
    _check_ops,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_n import (
    VECTOR_LIMIT,
    _x_loader,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    K_LIMIT,
    SIGNED_MAX,
    WORKGROUP,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
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
    / "demos/integrations/mlx/contracts/fp_quantized_qmm_n.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "fp_qmm_n"
# fp_qmm_n's buffers in declaration order: no global scale.
FP_QMM_N_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        (
            "w",
            "scales",
            "x",
            "y",
            "K",
            "N",
            "M",
            "x_batch_ndims",
            "x_shape",
            "x_strides",
            "w_batch_ndims",
            "w_shape",
            "w_strides",
            "s_strides",
        )
    )
)
# qmm binds w, scales, x, y, K, N and M without biases or transposition;
# add_strides_and_shapes skips the batch parameters when B <= 1.
UNBATCHED_BOUND = 7
# instantiate_quantized_batched_wrap's order: batched 1, then 0, each
# instantiate_quantized_batched followed by its _hgs twin.
VARIANTS = (
    ("batch_1", False),
    ("batch_1", True),
    ("batch_0", False),
    ("batch_0", True),
)

CASES = {
    "batch_0": ("tiles", "ragged", "short"),
    "batch_1": ("stacked", "broadcast", "shared"),
}
# (M, mx N, nvfp4 N, K). The mx group of 32 keeps every column tile whole;
# nvfp4's N of 16 more than a multiple of 32 ends in a tile of 16 columns.
SHAPES = {
    # Two row tiles, the second of 8 rows; two whole blocks of K.
    "tiles": (40, 64, 48, 64),
    # Two row tiles, the second of 4 rows; a block, then 13 rows of K.
    "ragged": (36, 64, 80, 45),
    # At the vector limit: one row tile of 4 rows; only the last pass of K;
    # nvfp4's one column tile of 16.
    "short": (VECTOR_LIMIT, 128, 16, 13),
    "stacked": (33, 64, 48, 40),
    "broadcast": (8, 64, 48, 32),
    "shared": (32, 64, 48, 64),
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
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for variant, hgs in VARIANTS:
                stem = f"{mode}_qmm_n_{itype}_gs_{group_size}_b_{bits}_{variant}"
                yield stem + ("_hgs" if hgs else ""), {
                    "kernel": KERNEL,
                    "variant": variant,
                    "mode": mode,
                    "itype": itype,
                    "dtype": dtype,
                    "groupSize": group_size,
                    "bits": bits,
                    "hgs": hgs,
                    # fp_qmm_n_impl never reads a global scale.
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
    elements the x loader reads as whole vectors, elements a resource helper
    reads or writes, weight and scale bytes, int64 batch strides, and
    threadgroup tile slots. GLSL indexes buffers with 32-bit integers, so
    each buffer index is asserted to fit one.
    """
    element = HELPER_ELEMENTS[INFO[entry]["dtype"]]
    items = [
        (f"source.offset + {index}", None, SIGNED_MAX, "x-vector")
        for index in range(X_READS)
    ]
    for access, family in (
        (f"load_{element}", "elements"),
        (f"store_{element}", "y"),
        ("load_uint8", "bytes"),
        ("load_int64_t", "strides"),
    ):
        scope = {"function": f"crosstl_resource_{access}"}
        items.append(("reference.offset + index", scope, SIGNED_MAX, family))
    # Xs is BM rows of BK_padded and Ws BK rows of BN_padded: the same slots.
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
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _column(info):
    return "nvfp4" if info["mode"] == "nvfp4" else "mx"


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    m, mx_n, nvfp4_n, k = SHAPES[case]
    return m, nvfp4_n if _column(INFO[entry]) == "nvfp4" else mx_n, k


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The fp_qvm gate's inputs, laid out as qvm's and bound as fp_qmm_n's."""
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, LAYOUTS[case], m, n, k, transposed=True)
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    grid = (-(-n // BN), -(-m // BM), len(layout["batches"]))
    layout["grid"] = grid
    data = case_data_for(info, entry, case, layout, "fp-quantized-qmm-n", None)
    values = dict(data["values"])
    del values["global_scale"]
    values["K"] = values.pop("in_vec_size")
    values["N"] = values.pop("out_vec_size")
    values["M"] = ("int32", [m])
    assert set(values) == {name for name, _ in FP_QMM_N_PARAMETERS}, entry
    return {**data, "values": values}


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _partial_n(entry, case):
    return INFO[entry]["groupSize"] % BN != 0 and _shape(entry, case)[1] % BN != 0


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "scale-split":
        # Groups of 32 give every thread of a tile the same group.
        return info["groupSize"] < BN
    if mutation == "batch":
        return info["variant"] == "batch_1"
    return True


def _simulate(entry, data, mutation=None):
    """fp_qmm_n_model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    return model.simulate(INFO[entry], data, families, mutation)


def _expected_paths(entry, case):
    m, n, k = _shape(entry, case)
    return model.expected_paths(m, n, k, INFO[entry]["groupSize"])


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


def _expected_validation_contract():
    qmm_t = _qmm_t_validation_contract()
    return {
        "model": (
            "fp_qmm_n_impl tile by tile after adjust_matrix_offsets: the "
            "four-element BlockLoader and the row-reducing fp "
            "QuantizedBlockLoader with their safe loads and the last partial "
            "block of K, one scale per thread's eight codes stepped a block of "
            "rows at a time, the partial column tiles of groups of 16, "
            "BlockMMA's float32 fragment products and the whole or safe "
            "stores, against direct dot products of the dequantized weights"
        ),
        "packing": qmm_t["packing"],
        "weights": "K rows of N codes, quantized along N",
        "comparison": qmm_t["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), ceil(M / 32), B)",
        "tile": {"BM": BM, "BN": BN, "BK": BK, "WM": WM, "WN": WN},
        "xLoaderReads": X_READS,
        "routing": (
            "untransposed weights with M at or above the vector limit of 4; "
            "batch_1 for B > 1; the fp modes have no NAX kernel without "
            "transposition, so on every device; qmm takes a global scale only "
            "with transposed weights, so the host dispatches no _hgs twin, "
            "which passes fp_qmm_n_impl no global scale either"
        ),
        "cases": {
            variant: {
                case: {
                    "M": SHAPES[case][0],
                    "mxN": SHAPES[case][1],
                    "nvfp4N": SHAPES[case][2],
                    "K": SHAPES[case][3],
                    "layout": LAYOUTS[case],
                }
                for case in cases
            }
            for variant, cases in CASES.items()
        },
        "partialN": (
            "groups of 16 let N end in a tile of 16 columns; its weight loads "
            "bound rows only, so the threads past N dequantize the next row's "
            "first bytes and scale, or the guards, into columns "
            "store_result_safe never writes"
        ),
        "inputs": qmm_t["inputs"],
        "exactness": (
            "every product is a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f); every dequantized weight is exact in T, so only "
            "static_cast<T> rounds, to nearest even"
        ),
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
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qmmN"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmm-n-contract",
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


def test_current_mlx_fp_quantized_qmm_n_contract_is_exact():
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


def test_fp_quantized_qmm_n_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmm_n.current-tree.translation.json" in readme
    assert len(ENTRIES) == 36 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qmmN"]
    assert Counter((info["variant"], info["hgs"]) for info in INFO.values()) == {
        ("batch_1", False): 9,
        ("batch_1", True): 9,
        ("batch_0", False): 9,
        ("batch_0", True): 9,
    }
    assert ENTRIES[:4] == (
        "nvfp4_qmm_n_float_gs_16_b_4_batch_1",
        "nvfp4_qmm_n_float_gs_16_b_4_batch_1_hgs",
        "nvfp4_qmm_n_float_gs_16_b_4_batch_0",
        "nvfp4_qmm_n_float_gs_16_b_4_batch_0_hgs",
    )
    assert ENTRIES[-1] == "mxfp4_qmm_n_float16_t_gs_32_b_4_batch_0_hgs"
    assert KERNEL not in EXCLUDED_KERNELS
    assert set(SHAPES) == set(LAYOUTS)
    assert {case for cases in CASES.values() for case in cases} == set(SHAPES)


def test_fp_quantized_qmm_n_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [18, 18]
    for shard in shards:
        # Each shard runs every mode, element type and batching.
        for key in ("mode", "dtype", "variant"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_fp_quantized_qmm_n_geometry_follows_the_fp_block_loader():
    for _, group_size, bits in MODES:
        shape = model.geometry(bits, group_size)
        # Every thread dequantizes eight codes of one group of one weight row:
        # 128 threads fill 32 rows of 32 columns.
        assert shape["reads"] * shape["packFactor"] == BK * BN // 128 == 8
        assert group_size % (shape["reads"] * shape["packFactor"]) == 0
        assert shape["partialN"] == (group_size == 16)


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmm_n_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n, k) == _shape(entry, case)
        # QuantizedMatmul reaches qmm, and qmm fp_qmm_n, on every device.
        assert VECTOR_LIMIT <= m
        # validate_quantized_input: N is whole words and whole groups.
        assert n * info["bits"] % 32 == 0 and n % info["groupSize"] == 0
        assert (batches > 1) == (info["variant"] == "batch_1")
        assert data["grid"] == (-(-n // BN), -(-m // BM), batches)
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["K"][1] == [k] and values["N"][1] == [n] and values["M"][1] == [m]
        params = layout["params"]
        assert params["x_shape"][-2:] == [m, k]
        # (K, N) weights: K rows of N * bits / 32 words and N / group_size groups.
        assert params["w_shape"][-2:] == [k, n * info["bits"] // 32]
        assert layout["words"] == n * info["bits"] // 32
        assert layout["groups"] == n // info["groupSize"]
        assert params["s_strides"][-2:] == [layout["groups"], 1]
        assert 0 <= k <= K_LIMIT


def test_fp_quantized_qmm_n_cases_reach_every_path():
    reached = {}
    for entry in ENTRIES:
        info = INFO[entry]
        key = (info["variant"], _column(info))
        for case in CASES[info["variant"]]:
            reached.setdefault(key, set()).update(_expected_paths(entry, case))
    expected = {}
    for variant in ("batch_0", "batch_1"):
        for column, widths in (("nvfp4", ("partial", "whole")), ("mx", ("whole",))):
            expected[variant, column] = {
                f"rows-{rows}/k-{k}/n-{n}"
                for rows in ("partial", "whole")
                for k in ("tail", "whole")
                for n in widths
            } | {"tail", "store-whole", "store-safe"}
    assert reached == expected


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_fp_quantized_qmm_n_model_matches_dot_products(entry):
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
        # Every used activation row, weight word and scale is read in its row,
        # and no guard or unused activation row is.
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
        assert run["reads"]["scales"] == {
            (matrix * k + row) * groups + group
            for matrix in matrices
            for row in range(k)
            for group in range(groups)
        }
        # The threads past N read the next row's first two words and its
        # first scale, or the guards after the last row.
        partial = _partial_n(entry, case)
        assert run["overreads"]["w"] == {
            (matrix * k + row + 1) * words + word
            for matrix in matrices
            for row in range(k)
            for word in range(2 if partial else 0)
        }
        assert run["overreads"]["scales"] == {
            (matrix * k + row + 1) * groups
            for matrix in matrices
            for row in range(k)
            if partial
        }
        if partial:
            assert (BN - n % BN) * INFO[entry]["bits"] // 8 == 2 * 4


def test_fp_quantized_qmm_n_model_derives_the_loader_assertions():
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


def test_fp_quantized_qmm_n_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in model.MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {"order": 12, "scale-split": 24, "batch": 18}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmm_n_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
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
        for mutation in model.MUTATIONS
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _mutation_cases())
def test_fp_quantized_qmm_n_expectations_reject_defects(entry, mutation):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: {model.MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::2])
def test_fp_quantized_qmm_n_opengl_assertions_name_buffer_elements(entry):
    items = _opengl_index_items(entry)
    seen = {family: set() for _, _, _, family in items}
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        records = _simulate(entry, data)["records"]
        for _, _, maximum, family in items:
            for buffer, index in records[family]:
                seen[family].add(buffer)
                if family == "bytes":
                    # The loads past N may read into the guards.
                    limit = lengths[buffer] * (4 if buffer == "w" else 1)
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
        "elements": {"x"},
        "y": {"y"},
        "bytes": {"w", "scales"},
        "strides": {"x_strides", "w_strides", "s_strides"} if batched else set(),
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_qmm_n_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES[:4]:
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            assert options[backend]["cooperative_matrix_software_lowering"] is True
            # No global scale, so no division profile.
            assert "binary32_division_profile" not in options[backend]
        (precondition,) = config.metadata_preconditions
        assert (precondition.parameter, precondition.maximum) == ("K", K_LIMIT)
        assertions = config.workgroup_access_assertions
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "metal":
            assert assertions == () and expressions == []
        else:
            assert [item.parameter for item in assertions] == ["self.dst"]
            assert [item.function for item in assertions] == [
                _x_loader(INFO[entry]["itype"]).function
            ]
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
    assert "instantiate_quantized_all_batched(type, mode, group_size, bits)" in (
        _macro(text, "instantiate_quantized_modes")
    )
    assert (
        "instantiate_quantized_batched_wrap(qmm_n, type, mode, group_size, bits)" in (
            _macro(text, "instantiate_quantized_all_batched")
        )
    )
    wrap = _macro(text, "instantiate_quantized_batched_wrap")
    assert re.findall(
        r"instantiate_quantized_batched\(mode, name, type, (\d),", wrap
    ) == [
        "1",
        "0",
    ]
    body = _macro(text, "instantiate_quantized_batched")
    plain = (
        '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched,'
    )
    twin = (
        '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" '
        '#batched "_hgs",'
    )
    assert plain in body and twin in body and body.index(plain) < body.index(twin)
    # The twin passes has_global_scale = true after batched.
    assert re.search(r"batched,\s*\\\s*true\)", body)


def _check_kernel(header):
    assert declared_buffers(header, KERNEL) == FP_QMM_N_PARAMETERS
    loader = header[header.index("struct QuantizedBlockLoader {") :]
    loader = loader[: loader.index("\n};\n")]
    for required in (
        "MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();",
        "MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();",
        ": BROWS * src_ld * bytes_per_pack / pack_factor),",
        "group_stride(BROWS* src_ld / group_size),",
        "(bj * pack_factor) / group_size) {",
        "float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc;",
        "if (bi >= src_tile_dim.y) {",
        "scales += group_stride;",
    ):
        assert required in loader, required
    # load_safe bounds the rows of the tile only.
    safe = loader[loader.index("void load_safe(") :]
    safe = safe[: safe.index("void next()")]
    assert "src_tile_dim.x" not in safe
    impl = header[header.index("METAL_FUNC void fp_qmm_n_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int WM = 2;",
        "constexpr int WN = 2;",
        "constexpr int BK_padded = (BK + 16 / sizeof(T));",
        "constexpr int BN_padded = (BN + 16 / sizeof(T));",
        "BlockMMA<T, T, BM, BN, BK, WM, WN, false, false, BK_padded, BN_padded>;",
        "BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 1, 4>;",
        "wl += y_col * bytes_per_pack / pack_factor;",
        "scales += y_col / group_size;",
        "constexpr bool partial_n = group_size % BN != 0;",
        "const short num_outs = partial_n ? short(min(BN, N - y_col)) : short(BN);",
        "if ((K % BK) != 0) {",
        "const int k_blocks = K / BK;",
        "for (int k = 0; k < K; k += BK) {",
        "} else if (num_outs < BN) {",
        "loader_x.load_safe(short2(BK, num_els));",
        "loader_w.load_safe(short2(num_outs, BK));",
        "loader_x.load_safe(short2(num_k, num_els));",
        "loader_x.load_safe(short2(num_k, BM));",
        "loader_w.load_safe(short2(num_outs, num_k));",
        "mma_op.store_result_safe(y, N, short2(num_outs, num_els));",
    ):
        assert required in impl, required
    # The weight loader reduces along its BK rows of BN columns.
    assert re.search(
        r"QuantizedBlockLoader<\s*T,\s*BK,\s*BN,\s*BN_padded,\s*0,\s*"
        r"WM \* WN \* SIMD_SIZE,\s*group_size,\s*bits,\s*has_global_scale>;",
        impl,
    )
    kernel = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    kernel = kernel[: kernel.index("\n}\n")]
    for required in (
        "threadgroup T Ws[BK * BN_padded];",
        "M * N,",
        "fp_qmm_n_impl<T, group_size, bits, false, BM, BK, BN>(",
        "w, scales, nullptr, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);",
    ):
        assert required in kernel, required


def _check_host(host):
    qmm = host[host.index("void qmm(\n") :]
    qmm = qmm[: qmm.index("\n}\n")]
    for required in (
        "(!global_scale || (transpose && !biases)) &&",
        'metal::is_nax_available() && (transpose || mode == "affine");',
        "MTL::Size group_dims(32, wn, wm);",
        "MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
        'mode + (transpose ? "_qmm_t_" : "_qmm_n_"),',
        'batched ? "_batch_1" : "_batch_0",',
        'd, kname, "qmm_n", mode, type_string, group_size, bits, batched);',
        "  if (biases) {\n    compute_encoder.set_input_array(*biases, c++);\n"
        "  } else if (transpose) {",
        "compute_encoder.set_bytes(M, c++);",
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
    ):
        assert required in qmm, required
    for required in (
        "int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;",
        "if (transpose_ && B == 1) {",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmm_n", TARGETS, MLX_COMMIT
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
    work = tmp_path_factory.mktemp("fp-quantized-qmm-n-metal-reference")
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
        parameters=FP_QMM_N_PARAMETERS,
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
            parameters=FP_QMM_N_PARAMETERS,
            workgroup=WORKGROUP,
            bound=UNBATCHED_BOUND,
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
        "checkedElementCount": len(data["expected"]["y"]),
        "observedInputCount": observed,
        "upstream": upstream,
    }


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_fp_quantized_qmm_n_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-fpqmmn-", dir=root) as path:
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


def test_ci_requires_native_fp_quantized_qmm_n_parity():
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
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmm_n.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmm_n.py",
    )
