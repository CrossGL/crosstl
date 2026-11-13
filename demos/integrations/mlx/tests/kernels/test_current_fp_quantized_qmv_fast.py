"""Prove fp_quantized.metal's fp_qmv_fast and fp_gather_qmv_fast entries natively.

``fp_quantized.metal`` instantiates ``fp_qmv_fast`` with and without batching
and ``fp_gather_qmv_fast``, each without and with a global scale (``_hgs``),
for nvfp4, mxfp8 and mxfp4 in float, bfloat16_t and float16_t, and nvfp4's
two-row ``_r_2`` ``fp_qmv_fast`` tiles: 66 of its 518 entries. Each is
translated and pinned per target, then runs ``quantized.cpp``'s ``qmv`` or
``gather_qmv`` dispatch for a product the host sends to the fast kernel:
threadgroups of ``(32, 2, 1)`` threads, two simdgroups of four (``_r_2``: two)
output rows each, over a grid of ``(M, N / 8, B)`` (``_r_2``: ``N / 4``).

``fp_qmv_fast_impl`` has no guard and no tail: each lane reads two packs per
block. Cases take one and several blocks, one and several row tiles, one and
two vectors, and both branches of each ``adjust_matrix_offsets`` overload.
Inputs share ``fp_qmv_model``'s exactness argument, so only the global
scale's product and ``static_cast<T>`` round. A lane-level model must reach a
direct dequantized dot product; each target must produce its words and leave
every guard untouched, and on Metal the translation must also match the
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
    SENTINEL,
    _adjust,
    _batch_expert,
    _checked,
    _experts,
    _inverse_scale,
    _parameters,
    _qdot,
    _scale_value,
    bound_parameters,
    case_data_for,
    config_for,
    translate_entry,
)
from demos.integrations.mlx.tests.kernels.fp_quantized_scope import (
    EXCLUDED_KERNELS,
    GATE_ENTRY_COUNTS,
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
    UNIT_ENTRY_COUNT,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    CI_SHARD_COUNT,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    ENTRIES as QMV_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    QMV_FP_HEADERS,
    QUAD_K,
    REQUIRE_ENV,
    SHARD_COUNT_ENV,
    SHARD_INDEX_ENV,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    NUM_SIMDGROUPS,
    SIMD_SIZE,
    UNBATCHED_BOUND,
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
    / "demos/integrations/mlx/contracts/fp_quantized_qmv_fast.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# Per type and mode, instantiate_quantized_all_batched's qmv_fast wrap
# instantiates fp_qmv_fast with batched 1, then 0, before
# instantiate_quantized_all_single's fp_gather_qmv_fast; after the three
# modes, instantiate_quantized_qmv_fast_r2 adds nvfp4's two-row tiles. Each
# comes without, then with, the _hgs global scale.
VARIANTS = ("batch_1", "batch_0", "gather")
KERNELS = {
    "batch_1": "fp_qmv_fast",
    "batch_0": "fp_qmv_fast",
    "gather": "fp_gather_qmv_fast",
}
# Rows per simdgroup: fp_qmv_fast_impl's results_per_simdgroup.
RESULTS = 4
NARROW_RESULTS = 2
NARROW_MODE = ("nvfp4", 16, 4)
PACKS_PER_THREAD = 2
QUANTIZE_ENTRY_COUNT = len(QUANTIZE_ENTRIES)
QMV_ENTRY_COUNT = len(QMV_ENTRIES)

# (M, N, blocks): K is blocks * block_size, a multiple of
# qmv_fast_k_alignment, and N a multiple of 8, so quantized.cpp picks the
# fast kernel. Without a global scale outside gather, dispatch_qmv sends
# M >= 2 to qmv_wide, so those entries run one vector.
CASES = {
    "batch_0": ("single", "deep"),
    "batch_1": ("stacked", "broadcast", "shared"),
    "gather": ("experts", "broadcast"),
}
SHAPES = {
    # One block and one tile of rows (_r_2: two).
    "single": (1, 8, 1),
    # Three blocks, two tiles, and two vectors with a global scale.
    "deep": (2, 16, 3),
    "stacked": (1, 8, 2),
    "broadcast": (2, 8, 1),
    "shared": (1, 16, 1),
    "experts": (1, 8, 2),
}
LAYOUTS = {
    "single": "unbatched",
    "deep": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
    "experts": "experts",
}
CASE_GLOBAL_SCALES = {
    "single": 3000.0,
    "deep": 10752.0,
    "stacked": 3000.0,
    "broadcast": 5376.0,
    "shared": 1344.0,
}


def _entry_name(mode, variant, itype, group_size, bits, global_scale, results):
    name = "gather_qmv_fast" if variant == "gather" else "qmv_fast"
    entry = f"{mode}_{name}_{itype}_gs_{group_size}_b_{bits}"
    if results == NARROW_RESULTS:
        entry += f"_r_{NARROW_RESULTS}"
    if variant != "gather":
        entry += f"_{variant}"
    return entry + ("_hgs" if global_scale else "")


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""

    def info(mode, group_size, bits, variant, global_scale, results):
        entry = _entry_name(
            mode, variant, itype, group_size, bits, global_scale, results
        )
        return entry, {
            "kernel": KERNELS[variant],
            "variant": variant,
            "mode": mode,
            "itype": itype,
            "dtype": dtype,
            "groupSize": group_size,
            "bits": bits,
            "globalScale": global_scale,
            "results": results,
        }

    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for variant in VARIANTS:
                for global_scale in (False, True):
                    yield info(mode, group_size, bits, variant, global_scale, RESULTS)
        for variant in ("batch_1", "batch_0"):
            for global_scale in (False, True):
                yield info(*NARROW_MODE, variant, global_scale, NARROW_RESULTS)


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    return config_for(directory, target, entry, INFO[entry])


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits, group_size):
    """fp_qmv_fast_impl's constants: two 32-bit packs per lane."""
    pack_factor = 32 // bits
    values = pack_factor * PACKS_PER_THREAD
    return {
        "packFactor": pack_factor,
        "bytesPerPack": 4,
        "valuesPerThread": values,
        "blockSize": values * SIMD_SIZE,
        "scaleStepPerThread": group_size // values,
        # qmv_fast_k_alignment(bits).
        "fastAlignment": pack_factor * (1 if bits == 2 else 2) * SIMD_SIZE,
    }


def _tile(info):
    """Output rows per threadgroup: quantized.cpp's bn."""
    return NUM_SIMDGROUPS * info["results"]


def _routes_to_fast(info, m, n, k):
    """Whether quantized.cpp dispatches fp_qmv_fast_impl for this product."""
    geometry = _geometry(info["bits"], info["groupSize"])
    # fast is decided with bn = 8, before the narrow tile.
    if n % 8 or k % geometry["fastAlignment"]:
        return False
    if info["variant"] == "gather" or info["globalScale"]:
        # gather_qmv has no qmv_quad or qmv_wide route, and dispatch_qmv
        # takes neither with a global scale.
        return True
    return k not in QUAD_K and m == 1


def _shape(entry, case):
    info = INFO[entry]
    m, n, blocks = SHAPES[case]
    if not (info["globalScale"] or info["variant"] == "gather"):
        m = 1
    return m, n, blocks * _geometry(info["bits"], info["groupSize"])["blockSize"]


def _layout(entry, case):
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, LAYOUTS[case], m, n, k)
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    # qmv's grid_dims(M, (N + bn - 1) / bn, B) with the entry's bn.
    layout["grid"] = (m, -(-n // _tile(info)), len(layout["batches"]))
    return layout


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    layout, scale = _layout(entry, case), CASE_GLOBAL_SCALES.get(case)
    label = "fp-quantized-qmv-fast"
    return case_data_for(INFO[entry], entry, case, layout, label, scale)


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    "scale-step": "gives every lane its block's first scale",
    "order": "pairs each lane's inputs with its codes in reverse",
    "pack-stride": "steps each lane one pack, not two, into its row",
    "rows": "ignores the simdgroup's row offset",
    "batch": "ignores the weight batch offsets",
    "last-block": "stops one block early",
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


def _lane(context, lane, out_row, mutation):
    """One lane's per-row partial sums over every block."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    bits, group_size, mode = info["bits"], info["groupSize"], info["mode"]
    x_base, w_base, s_base = context["bases"]
    per_thread, block = geometry["valuesPerThread"], geometry["blockSize"]
    in_vec_size, in_vec_size_w, in_vec_size_g = context["sizes"]
    packs = 1 if mutation == "pack-stride" else PACKS_PER_THREAD
    group = 0 if mutation == "scale-step" else lane // geometry["scaleStepPerThread"]
    ws = w_base * 4 + out_row * in_vec_size_w + lane * packs * geometry["bytesPerPack"]
    scales = s_base + out_row * in_vec_size_g + group
    x = x_base + context["tid_x"] * in_vec_size + lane * per_thread
    result = [0.0] * info["results"]
    end = in_vec_size - block if mutation == "last-block" else in_vec_size
    iterations = 0
    for _ in range(0, end, block):
        # load_vector converts T to float exactly.
        values = [load("x", x + i) for i in range(per_thread)]
        if mutation == "order":
            values.reverse()
        for row in range(info["results"]):
            scale = _scale_value(mode, load("scales", scales + row * in_vec_size_g))
            result[row] = _checked(
                result[row]
                + _qdot(
                    bits,
                    context["byte"],
                    ws + row * in_vec_size_w,
                    values,
                    scale,
                    mutation,
                )
            )
        ws += block * geometry["bytesPerPack"] // geometry["packFactor"]
        scales += block // group_size
        x += block
        iterations += 1
    return result, iterations


def _simulate(entry, data, mutation=None):
    """fp_qmv_fast_impl lane by lane; returns outputs, stores, reads, offsets, paths."""
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

    grid_x, grid_y, grid_z = data["grid"]
    paths.add("tiles" if grid_y > 1 else "one-tile")
    paths.add("vectors" if grid_x > 1 else "one-vector")
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if info["variant"] != "batch_0":
            bases = _adjust(info, storage, z, mutation, records)
        expert, y_base = bases[3], bases[4]
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:3],
                "tid_x": tid_x,
                "sizes": (
                    in_vec_size,
                    in_vec_size * geometry["bytesPerPack"] // geometry["packFactor"],
                    in_vec_size // info["groupSize"],
                ),
            }
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    out_row = tid_y * _tile(info)
                    if mutation != "rows":
                        out_row += simd_gid * info["results"]
                    factor = None
                    if info["globalScale"] and mutation != "global-scale":
                        factor = _inverse_scale(load("global_scale", expert))
                    lanes = [
                        _lane(context, lane, out_row, mutation)
                        for lane in range(SIMD_SIZE)
                    ]
                    (iterations,) = {count for _, count in lanes}
                    paths.add("blocks" if iterations > 1 else "one-block")
                    for row in range(info["results"]):
                        # simd_sum: exact in any order; lane 0 stores
                        # static_cast<T>(sum * inv_scale_enc).
                        total = 0.0
                        for result, _ in lanes:
                            total = _checked(total + result[row])
                        if factor is not None:
                            total = _f32(total * factor)
                        index = y_base + tid_x * out_vec_size + out_row + row
                        assert 0 <= index < len(outputs["y"]) - GUARD_COUNT, index
                        value = _to_type(info["dtype"], total)
                        if stores["y", index]:
                            # Only a defect stores one element twice.
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


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(
            f"fp quantized qmv fast shard {index} is outside [0, {count})"
        )
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
    return {
        "model": (
            "fp_qmv_fast_impl lane by lane after adjust_matrix_offsets: every "
            "block through load_vector and qdot over two packs per lane, "
            "Dequantize's E2M1 and E4M3 decoding, dequantize_scale's E4M3 and "
            "E8M0 scales, simd_sum, the global scale and lane 0's static_cast "
            "stores, against direct dequantized dot products"
        ),
        "packing": "least-significant-bit-first code stream per weight row",
        "comparison": "exact words",
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, N / (2 * results_per_simdgroup), B)",
        "k": "blocks * 64 * pack_factor, a multiple of qmv_fast_k_alignment",
        "m": (
            "1 without a global scale outside gather, which dispatch_qmv "
            "otherwise sends to qmv_wide"
        ),
        "narrowTile": (
            "quantized.cpp picks the _r_2 entries for nvfp4 when N >= 4096 on "
            "generation-17 's' devices; fp_qmv_fast_impl has no branch on N, "
            "so they run the cases' N over the same (M, N / 4, B) grid"
        ),
        "cases": {
            variant: {
                case: dict(zip(("M", "N", "blocks"), SHAPES[case])) for case in cases
            }
            for variant, cases in CASES.items()
        },
        "inputs": (
            "x in quarters of [-1, 1]; every E2M1 code; E4M3 codes of either "
            "sign with exponent fields 0 to 9; nvfp4 scales E4M3 with "
            "exponent fields 5 to 8, mx scales E8M0 bytes 124 to 128"
        ),
        "globalScales": {
            "qmv": dict(CASE_GLOBAL_SCALES),
            "gatherPerExpert": list(GLOBAL_SCALES),
        },
        "exactness": (
            "every product is a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f); only the global scale's product and static_cast<T> "
            "round, to nearest even"
        ),
        "guards": {"count": GUARD_COUNT, "sentinel": SENTINEL},
        "softwareSubgroups": {
            "width": SIMD_SIZE,
            "directxShuffleOutOfRange": "self",
        },
        "globalScaleDivisionProfile": "rne-flush",
        "openglIndexRangeAssertions": {
            variant: {
                "expressions": list(expressions),
                "minimum": 0,
                "maximum": INDEX_BOUND,
            }
            for variant, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
        },
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": "unchanged fp_quantized.metal, word for word",
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qmvFast"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmv-fast-contract",
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


def test_current_mlx_fp_quantized_qmv_fast_contract_is_exact():
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


def test_fp_quantized_qmv_fast_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmv_fast.current-tree.translation.json" in readme
    assert len(ENTRIES) == 66 == len(set(ENTRIES))
    # The other fp_quantized.metal gates count these entries as gated here.
    assert len(ENTRIES) == GATE_ENTRY_COUNTS["qmvFast"]
    assert QUANTIZE_ENTRY_COUNT == GATE_ENTRY_COUNTS["quantize"]
    assert QMV_ENTRY_COUNT == GATE_ENTRY_COUNTS["qmv"]
    assert Counter(info["variant"] for info in INFO.values()) == {
        "batch_1": 24,
        "batch_0": 24,
        "gather": 18,
    }
    assert Counter(info["globalScale"] for info in INFO.values()) == {
        False: 33,
        True: 33,
    }
    narrow = [entry for entry, info in INFO.items() if info["results"] == 2]
    assert len(narrow) == 12
    assert {INFO[entry]["mode"] for entry in narrow} == {"nvfp4"}
    assert ENTRIES[:6] == (
        "nvfp4_qmv_fast_float_gs_16_b_4_batch_1",
        "nvfp4_qmv_fast_float_gs_16_b_4_batch_1_hgs",
        "nvfp4_qmv_fast_float_gs_16_b_4_batch_0",
        "nvfp4_qmv_fast_float_gs_16_b_4_batch_0_hgs",
        "nvfp4_gather_qmv_fast_float_gs_16_b_4",
        "nvfp4_gather_qmv_fast_float_gs_16_b_4_hgs",
    )
    assert ENTRIES[18:22] == (
        "nvfp4_qmv_fast_float_gs_16_b_4_r_2_batch_1",
        "nvfp4_qmv_fast_float_gs_16_b_4_r_2_batch_1_hgs",
        "nvfp4_qmv_fast_float_gs_16_b_4_r_2_batch_0",
        "nvfp4_qmv_fast_float_gs_16_b_4_r_2_batch_0_hgs",
    )
    assert ENTRIES[-1] == "nvfp4_qmv_fast_float16_t_gs_16_b_4_r_2_batch_0_hgs"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert not set(ENTRIES) & (set(QUANTIZE_ENTRIES) | set(QMV_ENTRIES))


def test_fp_quantized_qmv_fast_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {33}
    for shard in shards:
        variants = Counter(INFO[entry]["variant"] for entry in shard)
        assert variants == {"batch_1": 12, "batch_0": 12, "gather": 9}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_fast_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"])
    assert geometry["packFactor"] * info["bits"] == 8 * geometry["bytesPerPack"]
    # The kernel's K step is the host's alignment, and each lane's values
    # stay inside one scale group.
    assert geometry["blockSize"] == geometry["fastAlignment"]
    assert info["groupSize"] % geometry["valuesPerThread"] == 0
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert _routes_to_fast(info, m, n, k)
        assert data["grid"] == (m, n // _tile(info), batches)
        # qmv's and gather_qmv's kernel names for this product; the host
        # adds _r_2 for nvfp4 when N >= 4096 on generation-17 's' devices.
        hgs = "_hgs" if info["globalScale"] else ""
        stem = f"{info['mode']}_{{}}_{info['itype']}_gs_{info['groupSize']}"
        stem += f"_b_{info['bits']}"
        if info["variant"] == "gather":
            assert entry == stem.format("gather_qmv_fast") + hgs
        else:
            narrow = "_r_2" if info["results"] == NARROW_RESULTS else ""
            suffix = "_batch_1" if batches > 1 else "_batch_0"
            assert entry == stem.format("qmv_fast") + narrow + suffix + hgs
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        assert "b_strides" not in params
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]
        if info["globalScale"] and info["variant"] == "gather":
            assert len(values["global_scale"][1]) == _experts(layout) + GUARD_COUNT
            assert max(layout["rhs"]) < _experts(layout)


def test_fp_quantized_qmv_fast_cases_reach_every_path():
    paths = {variant: set() for variant in VARIANTS}
    narrow = set()
    for entry in ENTRIES:
        info = INFO[entry]
        for case in CASES[info["variant"]]:
            reached = _simulate(entry, _case_data(entry, case))["paths"]
            paths[info["variant"]] |= reached
            if info["results"] == NARROW_RESULTS:
                narrow |= reached
    every = {"one-block", "blocks", "one-tile", "tiles", "one-vector", "vectors"}
    assert paths["batch_0"] == every
    assert paths["batch_1"] == every
    assert {"one-block", "blocks", "one-vector", "vectors"} <= paths["gather"]
    assert {"one-block", "blocks", "tiles", "vectors"} <= narrow


def test_fp_quantized_qmv_fast_inputs_reach_every_code():
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
def test_fp_quantized_qmv_fast_model_matches_dot_products(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        written = len(data["expected"]["y"]) - GUARD_COUNT
        # Every output is stored exactly once: the kernel has no guard and
        # no moved-back tile.
        assert run["stores"] == Counter({("y", index): 1 for index in range(written)})
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word and scale of the used matrices is read, and no
        # guard or unused activation row is.
        layout = data["layout"]
        rows = {
            batch["x"] + row * layout["K"] + index
            for batch in layout["batches"]
            for row in range(layout["M"])
            for index in range(layout["K"])
        }
        assert run["reads"]["x"] == rows
        matrices = {batch["w"] for batch in layout["batches"]}
        n, words, groups = layout["N"], layout["words"], layout["groups"]
        assert run["reads"]["w"] == {
            (matrix * n + col) * words + word
            for matrix in matrices
            for col in range(n)
            for word in range(words)
        }
        assert run["reads"]["scales"] == {
            (matrix * n + col) * groups + group
            for matrix in matrices
            for col in range(n)
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


def test_fp_quantized_qmv_fast_mutations_apply_where_they_are_defects():
    applied = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    )
    assert applied["batch"] == 42 and applied["global-scale"] == 33
    assert applied["expert-scale"] == 9
    for mutation in (
        "scale-step",
        "order",
        "pack-stride",
        "rows",
        "last-block",
        "sign",
    ):
        assert applied[mutation] == len(ENTRIES), mutation


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmv_fast_inputs_keep_float32_exact(dtype):
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
                assert _f32(total) == total and math.isfinite(value)
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
def test_fp_quantized_qmv_fast_expectations_reject_defects(entry, mutation):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: no case rejects a model that {MUTATIONS[mutation]}")


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_fast_opengl_assertions_name_buffer_elements(entry):
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
def test_fp_quantized_qmv_fast_configs_scope_target_options(tmp_path, target):
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
    per_type = _macro(text, "instantiate_quantized_types")
    modes = re.findall(
        r"instantiate_quantized_modes\(type, (\w+), (\d+), (\d+)\)", per_type
    )
    assert modes == [(mode, str(size), str(bits)) for mode, size, bits in MODES]
    narrow = "instantiate_quantized_qmv_fast_r2(nvfp4, type, 16, 4)"
    assert per_type.index(narrow) > per_type.index(
        "instantiate_quantized_modes(type, mxfp4"
    )
    groups = _macro(text, "instantiate_quantized_modes")
    assert groups.index("all_batched") < groups.index("all_single")
    batched = re.findall(
        r"instantiate_quantized_batched_wrap\((\w+),",
        _macro(text, "instantiate_quantized_all_batched"),
    )
    assert batched[0] == "qmv_fast"
    wrap = _macro(text, "instantiate_quantized_batched_wrap")
    assert re.findall(r"type, (\d), group_size", wrap) == ["1", "0"]
    single = re.findall(
        r"instantiate_quantized\(mode, (\w+),",
        _macro(text, "instantiate_quantized_all_single"),
    )
    assert single[0] == "gather_qmv_fast"
    rows = re.findall(
        r"instantiate_quantized_qmv_fast\(mode, type, (\d), (\d), group_size, bits\)",
        _macro(text, "instantiate_quantized_qmv_fast_r2"),
    )
    assert rows == [(str(NARROW_RESULTS), "1"), (str(NARROW_RESULTS), "0")]
    fast = _macro(text, "instantiate_quantized_qmv_fast")
    assert '"_r_" #results "_batch_" #batched' in fast
    # Each instantiation adds its _hgs twin after it.
    for macro in (
        "instantiate_quantized",
        "instantiate_quantized_batched",
        "instantiate_quantized_qmv_fast",
    ):
        body = _macro(text, macro)
        assert body.count("instantiate_kernel(") == 2
        assert body.index('"_hgs"') > body.index("instantiate_kernel(", 1)


def _check_kernel(header):
    assert declared_buffers(header, "fp_qmv_fast") == QMV_PARAMETERS
    assert declared_buffers(header, "fp_gather_qmv_fast") == GATHER_PARAMETERS
    assert re.search(
        r"bool has_global_scale = false,\s+int results_per_simdgroup = 4>\s+"
        r"METAL_FUNC void fp_qmv_fast_impl\(",
        header,
    )
    impl = header[header.index("METAL_FUNC void fp_qmv_fast_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int packs_per_thread = 2;",
        "constexpr int num_simdgroups = 2;",
        "constexpr int values_per_thread = pack_factor * packs_per_thread;",
        "constexpr int scale_step_per_thread = group_size / values_per_thread;",
        "ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;",
        "scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;",
        "for (int k = 0; k < in_vec_size; k += block_size) {",
        "U s = dequantize_scale<U, group_size>(sl[0]);",
        "result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);",
        "inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);",
        "y[row] = static_cast<T>(result[row] * inv_scale_enc);",
    ):
        assert required in impl, required
    # No guard, early return or tail.
    for absent in ("load_vector_safe", "qdot_safe", "if (out_row"):
        assert absent not in impl, absent
    gather = header[header.index("[[kernel]] void fp_gather_qmv_fast(") :]
    assert "fp_qmv_fast_impl<T, group_size, bits, has_global_scale>(" in gather


def _check_host(host):
    for required in (
        "int bk = 32;",
        "MTL::Size group_dims(bk, 2, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        "bool fast = N % bn == 0 && K % qmv_fast_k_alignment(bits) == 0;",
        "return get_pack_factor(bits, 32) * (bits == 2 ? 1 : 2) * 32;",
        "bool use_narrow_qmv = fast && N >= 4096 && d.get_architecture_gen() == 17 &&",
        "d.get_architecture().back() == 's' && mode == \"nvfp4\";",
        "int results_per_simdgroup = use_narrow_qmv ? 2 : 4;",
        "bn = 2 * results_per_simdgroup;",
        'use_narrow_qmv ? "_r_2" : "",',
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) {",
        "if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) {",
        'return mode != "affine" || d.get_architecture_gen() >= 15;',
        'mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),',
        'mode + (fast ? "_qmv_fast_" : "_qmv_"),',
        'B > 1 ? "_batch_1" : "_batch_0"',
        'global_scale ? "_hgs" : "");',
        "compute_encoder.set_input_array(*global_scale, 2);",
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmv fast", TARGETS, MLX_COMMIT
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
    work = tmp_path_factory.mktemp("fp-quantized-qmv-fast-metal-reference")
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
def test_current_mlx_fp_quantized_qmv_fast_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-fpqmvf-", dir=root) as path:
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


def test_ci_requires_native_fp_quantized_qmv_fast_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    # The fp_qmv gate's step runs this gate in the same legs and shards.
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv.py",
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv_fast.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv_fast.py",
    )
