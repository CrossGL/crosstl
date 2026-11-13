"""Current-tree MLX fp_quantized.metal fp_qmv_wide native parity.

``quantized.cpp``'s ``dispatch_qmv`` sends a matrix-vector product of two to
``vector_limit - 1`` vectors without a global scale, whose ``K`` is not 64 or
128, to ``qmv_wide``, on every device for the fp modes. ``fp_quantized.metal``
instantiates ``fp_qmv_wide`` for nvfp4, mxfp8 and mxfp4 in float, bfloat16_t
and float16_t, with tiles of 2, 3, 4 and 5 input vectors and sixteen lanes
per output row, with and without batching: 72 of its 518 entries. Each is
translated and pinned per target, then runs ``qmv_wide``'s ``(32, 2, 1)``
threadgroups, two simdgroups of two output rows each, over a
``(ceil(M / vecs_per_tg), ceil(N / 4), B)`` grid, where ``vecs_per_tg`` is
the tile ``qmv_wide`` picks for ``M``.

``fp_qmv_wide_impl`` gives each of a row's sixteen lanes every sixteenth
quant group, dequantizes it once to float4 lanes and dots it against every
vector of the tile, then reduces the row's lanes with a shuffle-down ladder.
Rows past ``N`` and vectors past ``M`` read row ``N - 1`` and vector
``M - 1`` and are never stored, so nothing reads past the used rows.
Unbatched entries run one tile against more groups per row than lanes, and a
tiled matrix whose last row tile, and for tiles of 4 and 5 last vector tile,
is partial; batched entries reach both branches of ``adjust_matrix_offsets``.
Inputs share ``fp_qmv_model``'s exactness argument, so only
``static_cast<T>`` rounds. A lane-level model must reach direct dequantized
dot products, storing every output once; each target must produce its words
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
    GUARD_COUNT,
    INDEX_BOUND,
    MODES,
    OPENGL_INDEX_RANGE_ASSERTIONS,
    QMV_PARAMETERS,
    _adjust,
    _checked,
    _code_value,
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
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv_quad import (
    ENTRIES as QMV_QUAD_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv_quad import (
    VECTOR_LIMIT,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qvm import (
    ENTRIES as QVM_ENTRIES,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _macro,
    _to_type,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    SIMD_SIZE,
    _f32,
    _request,
    _upstream,
    layout_for,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv_wide import (
    _tiled_rows,
    _vecs_per_tg,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/fp_quantized_qmv_wide.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "fp_qmv_wide"

# instantiate_quantized_all_wide's tiles; the fp modes use k_lanes 16.
VECS_PER_TG = (2, 3, 4, 5)
K_LANES = 16
# instantiate_quantized_wide_wrap instantiates batched 0, then 1.
VARIANTS = ("batch_0", "batch_1")
NUM_SIMDGROUPS = 2
RESULTS_PER_SIMDGROUP = SIMD_SIZE // K_LANES
# Output rows per threadgroup, qmv_wide's rows_per_tg.
TILE_ROWS = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS
WORKGROUP = (SIMD_SIZE, NUM_SIMDGROUPS, 1)
# dispatch_qmv routes these K to qmv_quad: 4 and 8 bits are powers of two.
QUAD_K = (64, 128)

# fp_qmv_wide takes fp_qmv's buffers without the global scale, with M after
# out_vec_size.
_NAMES = [name for name, _ in QMV_PARAMETERS if name != "global_scale"]
_NAMES.insert(_NAMES.index("out_vec_size") + 1, "M")
WIDE_PARAMETERS = tuple((name, index) for index, name in enumerate(_NAMES))
# Unbatched, qmv_wide skips add_strides_and_shapes: w, scales, x, y, K, N, M.
UNBATCHED_BOUND = 7

CASES = {
    "batch_0": ("tile", "tiles"),
    "batch_1": ("stacked", "broadcast", "shared"),
}
# (N, groups per row); K is groups * group_size.
SHAPES = {
    # One tile of vectors over two whole row tiles; seventeen groups give
    # lane 0 of each row two of them.
    "tile": (8, 17),
    # The most rows qmv_wide gives the tile below the vector limit; the last
    # row tile's second simdgroup is past N, and three groups leave thirteen
    # lanes of every row idle.
    "tiles": (6, 3),
    "stacked": (6, 17),
    # Five rows: the last row tile keeps one.
    "broadcast": (5, 3),
    "shared": (8, 3),
}
LAYOUTS = {
    "tile": "unbatched",
    "tiles": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
}
TILED_CASES = ("tiles", "shared")


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for vecs in VECS_PER_TG:
                for variant in VARIANTS:
                    entry = (
                        f"{mode}_qmv_wide_{itype}_gs_{group_size}_b_{bits}"
                        f"_nv_{vecs}_kl_{K_LANES}_{variant}"
                    )
                    yield entry, {
                        "kernel": KERNEL,
                        "variant": variant,
                        "mode": mode,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                        "vecsPerTg": vecs,
                        "globalScale": False,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    return config_for(directory, target, entry, INFO[entry], workgroup=WORKGROUP)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _rows(case, vecs):
    return _tiled_rows(vecs) if case in TILED_CASES else vecs


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """fp_qmv_model's inputs over qmv_wide's grid, with M bound."""
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    vecs = info["vecsPerTg"]
    m = _rows(case, vecs)
    n, groups = SHAPES[case]
    layout = layout_for(info, LAYOUTS[case], m, n, groups * info["groupSize"])
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    layout["grid"] = (-(-m // vecs), -(-n // TILE_ROWS), len(layout["batches"]))
    data = case_data_for(info, entry, case, layout, "fp-quantized-qmv-wide", None)
    values = {
        name: value for name, value in data["values"].items() if name != "global_scale"
    }
    values["M"] = ("int32", [m])
    return {**data, "values": values}


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    "order": "pairs each float4's inputs with its codes in reverse",
    "sign": "decodes every code without its sign bit",
    "scale-step": "gives every group its row's first scale",
    "stride": "gives each lane only its first group",
    "vectors": "streams the tile's first vector for every vector",
    "rows": "ignores each lane's row within its simdgroup",
    "ladder": "sums all 32 lanes instead of each row's 16",
    "row-guard": "stores the clamped rows past N",
    "vector-guard": "stores the clamped vectors past M",
    "batch": "ignores the weight batch offsets",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "batch":
        return info["variant"] == "batch_1"
    if mutation == "vector-guard":
        # Only tiles of 4 and 5 leave a partial last vector tile below the
        # vector limit.
        return _tiled_rows(info["vecsPerTg"]) % info["vecsPerTg"] != 0
    return True


def _dequantize_group(info, byte, wg, mutation):
    """One group's w4: float4 lanes of Dequantize, as a flat list."""
    bits, nf4 = info["bits"], info["groupSize"] // 4
    # The sign bit of the E4M3 byte or of the E2M1 low nibble.
    sign = 0xFF if mutation != "sign" else 0x7F if bits == 8 else 0xF7
    values = []
    for i in range(nf4):
        if bits == 4:
            # wq[i] is a little-endian uint16_t; Dequantize<4> takes the low
            # byte of each shift and fp4_e2m1 its low nibble.
            wq = byte(wg + 2 * i) | byte(wg + 2 * i + 1) << 8
            lanes = [_code_value(4, (wq >> (4 * j)) & 0xFF & sign) for j in range(4)]
        else:
            lanes = [_code_value(8, byte(wg + 4 * i + j) & sign) for j in range(4)]
        values += lanes[::-1] if mutation == "order" else lanes
    return values


def _lane(context, out_row, k_lane, mutation):
    """One lane's per-vector partial sums over its strided groups."""
    info, load, byte = context["info"], context["load"], context["byte"]
    _, w_base, s_base = context["bases"]
    in_vec_size_w, in_vec_size_g = context["sizes"]
    group_size, bits = info["groupSize"], info["bits"]
    # Rows past N read row N - 1.
    row = min(out_row, context["outVecSize"] - 1)
    wrow = 4 * w_base + row * in_vec_size_w
    srow = s_base + row * in_vec_size_g
    xv = context["xv"]
    result = [0.0] * len(xv)
    groups = range(k_lane, in_vec_size_g, K_LANES)
    if mutation == "stride":
        groups = groups[:1]
    for g in groups:
        k0 = g * group_size
        scale = load("scales", srow + (0 if mutation == "scale-step" else g))
        s = _scale_value(info["mode"], scale)
        w4 = _dequantize_group(info, byte, wrow + k0 * bits // 8, mutation)
        for v, start in enumerate(xv):
            # float4(xv4[j]) converts T to float exactly.
            x4 = [load("x", start + k0 + i) for i in range(group_size)]
            acc = 0.0
            for j in range(0, group_size, 4):
                dot = 0.0
                for weight, value in zip(w4[j : j + 4], x4[j : j + 4]):
                    dot = _checked(dot + _checked(weight * value))
                acc = _checked(acc + dot)
            result[v] = _checked(result[v] + _checked(s * acc))
    return result, len(groups)


def _ladder(partials, mutation):
    """k_lanes 16's simd_shuffle_down ladder over 8, 4, 2 and 1 lanes.

    A lane whose source is past the simdgroup keeps its own value.
    """
    if mutation == "ladder":
        total = 0.0
        for value in partials:
            total = _checked(total + value)
        return [total] * SIMD_SIZE
    values = list(partials)
    for delta in (8, 4, 2, 1):
        values = [
            _checked(
                values[lane]
                + values[lane + delta if lane + delta < SIMD_SIZE else lane]
            )
            for lane in range(SIMD_SIZE)
        ]
    return values


def _store(context, outputs, stores, simdgroup, vec0, mutation):
    """Lane 0 of each row stores the tile's vectors, guarded by N and M."""
    info, rows, n = context["info"], context["rows"], context["outVecSize"]
    for lane, (out_row, _) in enumerate(simdgroup["lanes"]):
        if lane % K_LANES:
            continue
        if out_row >= n and mutation != "row-guard":
            continue
        for v in range(info["vecsPerTg"]):
            if vec0 + v >= rows and mutation != "vector-guard":
                continue
            index = context["yBase"] + (vec0 + v) * n + out_row
            assert 0 <= index < len(outputs["y"]), index
            stores["y", index] += 1
            outputs["y"][index] = _to_type(info["dtype"], simdgroup["reduced"][v][lane])


def _simulate(entry, data, mutation=None):
    """fp_qmv_wide_impl lane by lane; returns outputs, stores, reads, offsets, paths."""
    info = INFO[entry]
    vecs, group_size = info["vecsPerTg"], info["groupSize"]
    storage = {name: values for name, (_, values) in data["values"].items()}
    in_vec_size, out_vec_size = storage["in_vec_size"][0], storage["out_vec_size"][0]
    rows = storage["M"][0]
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
    for z in range(grid_z):
        bases = (0, 0, 0)
        if info["variant"] == "batch_1":
            bases = _adjust(info, storage, z, mutation, records)[:3]
        context = {
            "info": info,
            "load": load,
            "byte": byte,
            "bases": bases,
            "rows": rows,
            "outVecSize": out_vec_size,
            # adjust_matrix_offsets' output_stride is out_vec_size * M.
            "yBase": z * out_vec_size * rows,
            # in_vec_size_w in bytes, and in_vec_size_g.
            "sizes": (in_vec_size * info["bits"] // 8, in_vec_size // group_size),
        }
        for tid_x in range(grid_x):
            vec0 = tid_x * vecs
            # xv[v] = x + min(vec0 + v, M - 1) * in_vec_size.
            context["xv"] = [
                bases[0]
                + min(vec0 + (0 if mutation == "vectors" else v), rows - 1)
                * in_vec_size
                for v in range(vecs)
            ]
            if vec0 + vecs > rows:
                paths.add("vector-clamp")
            for tid_y in range(grid_y):
                for simd_gid in range(NUM_SIMDGROUPS):
                    lanes = []
                    for lane in range(SIMD_SIZE):
                        k_lane = lane % K_LANES
                        sg_row = 0 if mutation == "rows" else lane // K_LANES
                        out_row = tid_y * TILE_ROWS
                        out_row += RESULTS_PER_SIMDGROUP * simd_gid + sg_row
                        result, count = _lane(context, out_row, k_lane, mutation)
                        paths.add({0: "idle-lane", 1: "one-group"}.get(count, "groups"))
                        if out_row >= out_vec_size:
                            paths.add("row-clamp")
                        lanes.append((out_row, result))
                    if all(out_row >= out_vec_size for out_row, _ in lanes):
                        paths.add("idle-simdgroup")
                    reduced = [
                        _ladder([result[v] for _, result in lanes], mutation)
                        for v in range(vecs)
                    ]
                    simdgroup = {"lanes": lanes, "reduced": reduced}
                    _store(context, outputs, stores, simdgroup, vec0, mutation)
    return {
        "outputs": outputs,
        "stores": stores,
        "reads": reads,
        "records": records,
        "paths": paths,
    }


@functools.lru_cache(maxsize=None)
def _baseline(entry, case):
    return _simulate(entry, _case_data(entry, case))


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
            f"fp quantized qmv_wide shard {index} is outside [0, {count})"
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
    qmv = _qmv_validation_contract()
    return {
        "model": (
            "fp_qmv_wide_impl lane by lane after adjust_matrix_offsets: each "
            "lane's strided quant groups, Dequantize's E2M1 and E4M3 decoding "
            "to float4 lanes, dequantize_scale's E4M3 and E8M0 scales, the "
            "dot against every vector of the tile, the clamped rows and "
            "vectors, the k_lanes 16 simd_shuffle_down ladder and lane 0's "
            "guarded static_cast stores, against direct dequantized dot "
            "products"
        ),
        "packing": qmv["packing"],
        "comparison": qmv["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(M / vecs_per_tg), ceil(N / 4), B)",
        "k": "groups * group_size, never 64 or 128, which go to qmv_quad",
        "m": (
            f"vecs_per_tg, or the most rows below {VECTOR_LIMIT} that qmv_wide "
            "gives the tile"
        ),
        "cases": {
            variant: {
                case: {
                    "M": {str(vecs): _rows(case, vecs) for vecs in VECS_PER_TG},
                    **dict(zip(("N", "groups"), SHAPES[case])),
                }
                for case in cases
            }
            for variant, cases in CASES.items()
        },
        "rowClamp": (
            "rows past N and vectors past M read row N - 1 and vector M - 1 "
            "and are never stored; nothing reads past the used rows"
        ),
        "inputs": qmv["inputs"],
        "exactness": qmv["exactness"],
        "guards": qmv["guards"],
        "softwareSubgroups": qmv["softwareSubgroups"],
        "openglIndexRangeAssertions": qmv["openglIndexRangeAssertions"],
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": qmv["metalUpstreamComparison"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qmvWide"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmv-wide-contract",
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


def test_current_mlx_fp_quantized_qmv_wide_contract_is_exact():
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


def test_fp_quantized_qmv_wide_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmv_wide.current-tree.translation.json" in readme
    assert len(ENTRIES) == 72 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qmvWide"]
    assert Counter(info["variant"] for info in INFO.values()) == dict.fromkeys(
        VARIANTS, 36
    )
    assert Counter(info["vecsPerTg"] for info in INFO.values()) == dict.fromkeys(
        VECS_PER_TG, 18
    )
    assert ENTRIES[:3] == (
        "nvfp4_qmv_wide_float_gs_16_b_4_nv_2_kl_16_batch_0",
        "nvfp4_qmv_wide_float_gs_16_b_4_nv_2_kl_16_batch_1",
        "nvfp4_qmv_wide_float_gs_16_b_4_nv_3_kl_16_batch_0",
    )
    assert ENTRIES[-1] == "mxfp4_qmv_wide_float16_t_gs_32_b_4_nv_5_kl_16_batch_1"
    assert KERNEL not in EXCLUDED_KERNELS
    others = set(QUANTIZE_ENTRIES) | set(QMV_ENTRIES) | set(QMV_FAST_ENTRIES)
    others |= set(QVM_ENTRIES) | set(QMV_QUAD_ENTRIES)
    assert not set(ENTRIES) & others
    names = [name for name, _ in WIDE_PARAMETERS]
    assert names[:7] == ["w", "scales", "x", "y", "in_vec_size", "out_vec_size", "M"]


def test_fp_quantized_qmv_wide_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {36}
    for shard in shards:
        tiles = Counter(INFO[entry]["vecsPerTg"] for entry in shard)
        assert tiles == dict.fromkeys(VECS_PER_TG, 9)
        modes = Counter(INFO[entry]["mode"] for entry in shard)
        assert modes == {mode: 12 for mode, _, _ in MODES}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_fp_quantized_qmv_wide_tiles_follow_qmv_wide():
    # The tile qmv_wide picks for every M it receives.
    tiles = {m: _vecs_per_tg(m) for m in range(2, VECTOR_LIMIT)}
    assert set(tiles.values()) == set(VECS_PER_TG)
    for vecs in VECS_PER_TG:
        m = _tiled_rows(vecs)
        assert tiles[m] == vecs and m == max(
            [row for row, tile in tiles.items() if tile == vecs and row % vecs]
            or [row for row, tile in tiles.items() if tile == vecs]
        )
    assert {vecs: _tiled_rows(vecs) for vecs in VECS_PER_TG} == {
        2: 2,
        3: 6,
        4: 11,
        5: 9,
    }


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_wide_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    vecs = info["vecsPerTg"]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (n, k // info["groupSize"]) == SHAPES[case]
        # dispatch_qmv's qmv_wide route: no global scale, M from 2 below
        # every vector limit, K not a qmv_quad width; qmv_wide's tile.
        assert 2 <= m < VECTOR_LIMIT and k not in QUAD_K
        assert _vecs_per_tg(m) == vecs and k <= 2048 and n <= 2048
        assert (batches > 1) == (info["variant"] == "batch_1")
        assert data["grid"] == (-(-m // vecs), -(-n // TILE_ROWS), batches)
        stem = f"{info['mode']}_qmv_wide_{info['itype']}_gs_{info['groupSize']}"
        suffix = "_batch_1" if batches > 1 else "_batch_0"
        assert entry == f"{stem}_b_{info['bits']}_nv_{vecs}_kl_{K_LANES}{suffix}"
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert set(values) == {name for name, _ in WIDE_PARAMETERS}
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        assert values["M"][1] == [m]
        params = layout["params"]
        assert "b_strides" not in params
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]


def test_fp_quantized_qmv_wide_cases_reach_every_path():
    paths = {(variant, vecs): set() for variant in VARIANTS for vecs in VECS_PER_TG}
    for entry in ENTRIES:
        info = INFO[entry]
        for case in CASES[info["variant"]]:
            paths[info["variant"], info["vecsPerTg"]] |= _baseline(entry, case)["paths"]
    every = {"idle-lane", "one-group", "groups", "row-clamp", "idle-simdgroup"}
    for (variant, vecs), reached in paths.items():
        expected = every | ({"vector-clamp"} if vecs in (4, 5) else set())
        assert reached == expected, (variant, vecs)


def test_fp_quantized_qmv_wide_inputs_reach_every_code():
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
def test_fp_quantized_qmv_wide_model_matches_dot_products(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        run = _baseline(entry, case)
        assert run["outputs"] == data["expected"], (entry, case)
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word and scale of the used matrices and every element
        # of the used activation rows is read; the clamps read nothing else.
        layout = data["layout"]
        n, k = layout["N"], layout["K"]
        assert run["reads"]["x"] == {
            batch["x"] + row * k + index
            for batch in layout["batches"]
            for row in range(layout["M"])
            for index in range(k)
        }
        matrices = {batch["w"] for batch in layout["batches"]}
        words, groups = layout["words"], layout["groups"]
        assert run["reads"]["w"] == {
            (matrix * n + row) * words + word
            for matrix in matrices
            for row in range(n)
            for word in range(words)
        }
        assert run["reads"]["scales"] == {
            (matrix * n + row) * groups + group
            for matrix in matrices
            for row in range(n)
            for group in range(groups)
        }


def test_fp_quantized_qmv_wide_mutations_apply_where_they_are_defects():
    applied = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    )
    assert applied["batch"] == 36 and applied["vector-guard"] == 36
    for mutation in set(MUTATIONS) - {"batch", "vector-guard"}:
        assert applied[mutation] == len(ENTRIES), mutation


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmv_wide_inputs_keep_float32_exact(dtype):
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
                rounded += value != total
    if dtype == "float32":
        assert rounded == 0
    else:
        # Only static_cast<T> rounds.
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
def test_fp_quantized_qmv_wide_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_wide_opengl_assertions_name_buffer_elements(entry):
    variant = INFO[entry]["variant"]
    used = set()
    for case in CASES[variant]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _baseline(entry, case)["records"]:
            used.add(expression)
            assert 0 <= value < lengths[buffer] - GUARD_COUNT, (case, expression)
            assert value <= INDEX_BOUND
    # Every precondition is reached, and none is missing.
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[variant])


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_qmv_wide_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES:
        info = INFO[entry]
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        # Without a global scale there is no division to round.
        assert config.source_options["metal"]["target_options"] == {
            "directx": {
                "software_subgroup_width": SIMD_SIZE,
                "relative_wave_shuffle_out_of_range": "self",
            },
            "opengl": {"software_subgroup_width": SIMD_SIZE},
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
    assert "instantiate_quantized_all_wide(type, mode, group_size, bits)" in groups
    wides = re.findall(
        r"instantiate_quantized_wide_wrap\(mode, (\w+), type, (\d+), (\d+), "
        r"group_size, bits\)",
        _macro(text, "instantiate_quantized_all_wide"),
    )
    assert wides == [("qmv_wide", str(vecs), str(K_LANES)) for vecs in VECS_PER_TG]
    wrap = re.findall(
        r"instantiate_quantized_wide\(mode, name, type, vecs_per_tg, k_lanes, "
        r"group_size, bits, (\d)\)",
        _macro(text, "instantiate_quantized_wide_wrap"),
    )
    assert wrap == [variant[-1] for variant in VARIANTS]
    wide = _macro(text, "instantiate_quantized_wide")
    assert wide.count("instantiate_kernel(") == 1
    assert (
        '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" '
        '#vecs_per_tg "_kl_" #k_lanes "_batch_" #batched'
    ) in wide
    assert "fp_ ## name," in wide


def _check_kernel(header):
    assert declared_buffers(header, KERNEL) == WIDE_PARAMETERS
    impl = header[header.index("METAL_FUNC void fp_qmv_wide_impl(") :]
    impl = impl[: impl.index("template <", 1)]
    for required in (
        "constexpr int num_simdgroups = 2;",
        "constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;",
        "constexpr int nf4 = group_size / 4;",
        "const short k_lane =\n      simd_lid % k_lanes;",
        "const short sg_row = simd_lid / k_lanes;",
        "const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) +\n"
        "      results_per_simdgroup * simd_gid + sg_row;",
        "const int vec0 = tid.x * vecs_per_tg;",
        "const int row = min(out_row, out_vec_size - 1);",
        "const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;",
        "const int in_vec_size_g = in_vec_size / group_size;",
        "xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;",
        "for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {",
        "U s = dequantize_scale<U, group_size>(srow[g]);",
        "const device uint8_t* wg = wrow + k0 * bytes_per_pack / pack_factor;",
        "Dequantize<4>{}(wq[i] >> 12));",
        "Dequantize<8>{}(wg[4 * i + 3]));",
        "acc += dot(w4[j], float4(xv4[j]));",
        "result[v] += s * acc;",
        "result[v] += simd_shuffle_down(result[v], 8);",
        "result[v] += simd_shuffle_down(result[v], 1);",
        "if (k_lane == 0 && out_row < out_vec_size) {",
        "if (vec0 + v < M) {",
        "y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]);",
    ):
        assert required in impl, required
    body = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    body = body[: body.index("\n}\n")]
    assert "if (batched) {" in body
    assert "adjust_matrix_offsets(\n        x,\n        w,\n        scales,\n" in body
    assert "out_vec_size * M," in body
    assert "fp_qmv_wide_impl<T, group_size, bits, vecs_per_tg, k_lanes>(" in body


def _check_host(host):
    body = host[host.index("void qmv_wide(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "int n_tiles = (M + 4) / 5;",
        "int vecs_per_tg = (M + n_tiles - 1) / n_tiles;",
        'int k_lanes = mode == "affine" ? 8 : 16;',
        "constexpr int num_simdgroups = 2;",
        "bool batched = B > 1;",
        "int rows_per_tg = (32 / k_lanes) * num_simdgroups;",
        "MTL::Size group_dims(32, num_simdgroups, 1);",
        "(M + vecs_per_tg - 1) / vecs_per_tg,",
        "(N + rows_per_tg - 1) / rows_per_tg,",
        'mode + "_qmv_wide_",',
        'batched ? "_batch_1" : "_batch_0");',
        "compute_encoder.set_input_array(w, c++);",
        "compute_encoder.set_input_array(scales, c++);",
        "compute_encoder.set_input_array(x, c++);",
        "compute_encoder.set_output_array(out, c++);",
        "compute_encoder.set_bytes(K, c++);",
        "compute_encoder.set_bytes(N, c++);",
        "compute_encoder.set_bytes(M, c++);",
        "add_strides_and_shapes(compute_encoder, !batched, x, w, scales, biases, c);",
    ):
        assert required in body, required
    assert 'return mode != "affine" || d.get_architecture_gen() >= 15;' in host
    route = host[host.index("void dispatch_qmv(") :]
    route = route[: route.index("\n}\n")]
    assert route.index("if ((K == 128 || K == 64) && is_power_of_2(bits)") < (
        route.index("if (M >= 2 && use_qmv_wide(mode, d) && !global_scale) {")
    )


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmv_wide", TARGETS, MLX_COMMIT
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
    work = tmp_path_factory.mktemp("fp-quantized-qmv-wide-metal-reference")
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
def test_current_mlx_fp_quantized_qmv_wide_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-fpwide-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = translate_entry(
                root, work, target, config, record, workgroup=WORKGROUP
            )
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES[variant]:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor,
                        package,
                        target,
                        variant,
                        data,
                        WIDE_PARAMETERS,
                        workgroup=WORKGROUP,
                    )
                    words = _execute(
                        executor, work, case, request, {"y": output}, observed
                    )["y"]["values"]
                    upstream = None
                    if target == "metal":
                        reference = (work, metal_reference, entry, variant, case)
                        device, upstream_words = _upstream(
                            *reference,
                            data,
                            WIDE_PARAMETERS,
                            workgroup=WORKGROUP,
                            bound=UNBATCHED_BOUND,
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


def test_ci_requires_native_fp_quantized_qmv_wide_parity():
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv_wide.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv_wide.py",
    )
