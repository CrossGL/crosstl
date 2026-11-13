"""Prove fp_quantized.metal's fp_qmv_quad entries natively.

``quantized.cpp``'s ``dispatch_qmv`` sends a matrix-vector product whose
``K`` is 64 or 128 and whose width is a power of two, without a global
scale, to ``qmv_quad``. ``fp_quantized.metal`` instantiates ``fp_qmv_quad``
for nvfp4, mxfp8 and mxfp4 in float, bfloat16_t and float16_t, with ``D`` of
64 and 128, with and without batching: 36 of its 518 entries. Both widths, 4
and 8 bits, are powers of two and every group size divides both ``D``, so
the host can dispatch all 36. Each is translated and pinned per target, then
runs ``qmv_quad``'s ``(32, 1, 1)`` threadgroups over an
``(M, ceil(N / 64), B)`` grid.

``fp_qmv_quad_impl`` gives each of a simdgroup's eight quads eight output
rows, eight apart, and each lane of a quad a quarter of ``K``, in one step
per group it spans: two for nvfp4 at ``D`` 128, otherwise one. A row past
``N`` skips its ``qdot`` and its store but still loads its scales, so the
cases pad the scales through the last tile and the model shows those bytes
never reach an output. ``quad_sum`` adds a quad's four partial sums; DirectX
and OpenGL lower it to a software reduction over four consecutive
invocations. Inputs share ``fp_qmv_model``'s exactness argument, so only
``static_cast<T>`` rounds. A lane-level model must reach direct dequantized
dot products, storing every row once; each target must produce its words and
leave every guard untouched, and on Metal the translation must also match
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
    BYTE_POISON,
    E4M3_CODE_FIELDS,
    GUARD_COUNT,
    INDEX_BOUND,
    MODES,
    OPENGL_INDEX_RANGE_ASSERTIONS,
    QMV_PARAMETERS,
    _adjust,
    _checked,
    _qdot,
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
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/fp_quantized_qmv_quad.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "fp_qmv_quad"

# instantiate_quantized_all_quad's (D, batched) pairs, in order.
QUAD_VARIANTS = ((64, 1), (64, 0), (128, 1), (128, 0))
VARIANTS = ("batch_1", "batch_0")
QUAD_SIZE = 4
QUADS_PER_SIMD = SIMD_SIZE // QUAD_SIZE
RESULTS_PER_QUADGROUP = 8
# Output rows per threadgroup, qmv_quad's bn.
TILE_ROWS = QUADS_PER_SIMD * RESULTS_PER_QUADGROUP
WORKGROUP = (SIMD_SIZE, 1, 1)

# fp_qmv_quad takes fp_qmv's buffers without the global scale.
QUAD_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        name for name, _ in QMV_PARAMETERS if name != "global_scale"
    )
)
# With B <= 1 qmv_quad skips add_strides_and_shapes: w, scales, x, y, K, N.
UNBATCHED_BOUND = 6

# Unbatched entries reach whole tiles, a partial last tile and quads with no
# row at all; batched entries combine them with adjust_matrix_offsets'
# layouts.
CASES = {
    "batch_1": ("stacked", "broadcast", "shared"),
    "batch_0": ("tile", "ragged", "short"),
}
# (M, N); K is the entry's D.
SHAPES = {
    # One whole tile: every quad stores all eight of its rows.
    "tile": (1, 64),
    # A whole tile and a last tile of six rows, past N for every quad's
    # second row on.
    "ragged": (2, 70),
    # Five rows: quads 5-7 store nothing.
    "short": (3, 5),
    # Six rows of two stacked batches: quads 6 and 7 store nothing.
    "stacked": (1, 6),
    "broadcast": (2, 13),
    # One tile and eight rows of the next, against one weight matrix.
    "shared": (1, 72),
}
LAYOUTS = {
    "tile": "unbatched",
    "ragged": "unbatched",
    "short": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
}
# QuantizedMatmul's smallest get_qmv_batch_limit for K and N of at most 2048.
VECTOR_LIMIT = 13


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for d, batched in QUAD_VARIANTS:
                entry = (
                    f"{mode}_qmv_quad_{itype}_gs_{group_size}_b_{bits}"
                    f"_d_{d}_batch_{batched}"
                )
                yield entry, {
                    "kernel": KERNEL,
                    "variant": f"batch_{batched}",
                    "mode": mode,
                    "itype": itype,
                    "dtype": dtype,
                    "groupSize": group_size,
                    "bits": bits,
                    "D": d,
                    "globalScale": False,
                }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    return config_for(directory, target, entry, INFO[entry], workgroup=WORKGROUP)


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits, group_size, d):
    """fp_qmv_quad_impl's constants."""
    pack_factor = 32 // bits
    values_per_thread = d // QUAD_SIZE
    steps = 1 if values_per_thread < group_size else values_per_thread // group_size
    values_per_step = values_per_thread // steps
    return {
        "packFactor": pack_factor,
        "valuesPerThread": values_per_thread,
        "stepsPerThread": steps,
        "valuesPerStep": values_per_step,
        "packsPerThread": values_per_thread // pack_factor,
        "packsPerStep": values_per_step // pack_factor,
    }


def _lane_groups(info, quad_lid):
    """The scale groups one lane of a quad reads, step by step."""
    geometry = _geometry(info["bits"], info["groupSize"], info["D"])
    first = quad_lid * geometry["valuesPerThread"] // info["groupSize"]
    return [first + step for step in range(geometry["stepsPerThread"])]


def _overread(layout):
    """Scale bytes the rows past ``N`` of each last tile load."""
    n, groups = layout["N"], layout["groups"]
    rows = layout["grid"][1] * TILE_ROWS
    return {
        (batch["w"] * n + row) * groups + group
        for batch in layout["batches"]
        for row in range(n, rows)
        for group in range(groups)
    }


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """fp_qmv_model's inputs over qmv_quad's grid, padded for the overreads."""
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n = SHAPES[case]
    layout = layout_for(info, LAYOUTS[case], m, n, info["D"])
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    layout["grid"] = (m, -(-n // TILE_ROWS), len(layout["batches"]))
    data = case_data_for(info, entry, case, layout, "fp-quantized-qmv-quad", None)
    values = {
        name: value for name, value in data["values"].items() if name != "global_scale"
    }
    used = layout["matrices"] * n * layout["groups"]
    padding = max(0, max(_overread(layout), default=-1) + 1 - used)
    kind, storage = values["scales"]
    assert len(storage) == used + GUARD_COUNT, (entry, case)
    # The padding sits before the guards, so every guard stays past any read.
    storage = storage[:used] + [BYTE_POISON] * padding + storage[used:]
    values["scales"] = (kind, storage)
    return {**data, "values": values, "padding": padding, "usedScales": used}


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    "order": "pairs each pack's inputs with its codes in reverse",
    "rows": "ignores the quad's row offset",
    "row-step": "reads every row's codes from the quad's first row",
    "scale-step": "gives every lane its row's first scale",
    "group-step": "keeps a lane's first scale for its second step",
    "quad": "stores lane 0's partial sum instead of the quad_sum",
    "batch": "ignores the weight batch offsets",
    "sign": "decodes every code without its sign bit",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"], info["D"])
    if mutation == "batch":
        return info["variant"] != "batch_0"
    if mutation == "scale-step":
        # A lane reads its own group only when a quad spans several.
        return info["D"] > info["groupSize"]
    if mutation == "group-step":
        return geometry["stepsPerThread"] > 1
    return True


def _quad(context, out_row, mutation):
    """One quad's four lanes: per-row partial sums, loading every row's scales."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    x_base, w_base, s_base = context["bases"]
    in_vec_size, in_vec_size_w, in_vec_size_g = context["sizes"]
    per_thread, pack_factor = geometry["valuesPerThread"], geometry["packFactor"]
    per_step = geometry["valuesPerStep"]
    lanes = []
    for quad_lid in range(QUAD_SIZE):
        ws = 4 * (w_base + out_row * in_vec_size_w)
        ws += 4 * quad_lid * geometry["packsPerThread"]
        group = 0 if mutation == "scale-step" else _lane_groups(info, quad_lid)[0]
        scales = s_base + out_row * in_vec_size_g + group
        x = x_base + context["tid_x"] * in_vec_size + quad_lid * per_thread
        # load_vector converts T to float exactly.
        values = [load("x", x + i) for i in range(per_thread)]
        if mutation == "order":
            values = [
                values[start + pack_factor - 1 - i]
                for start in range(0, per_thread, pack_factor)
                for i in range(pack_factor)
            ]
        result = [0.0] * RESULTS_PER_QUADGROUP
        for row in range(RESULTS_PER_QUADGROUP):
            step = row * QUADS_PER_SIMD
            wl = ws if mutation == "row-step" else ws + 4 * step * in_vec_size_w
            sl = scales + step * in_vec_size_g
            for k in range(geometry["stepsPerThread"]):
                # Each step loads its scale before the row guard.
                scale = _scale_value(info["mode"], load("scales", sl))
                if step + out_row < context["outVecSize"]:
                    result[row] = _checked(
                        result[row]
                        + _qdot(
                            info["bits"],
                            context["byte"],
                            wl,
                            values[k * per_step : (k + 1) * per_step],
                            scale,
                            mutation,
                        )
                    )
                if mutation != "group-step":
                    sl += 1
                wl += 4 * geometry["packsPerStep"]
        lanes.append(result)
    return lanes


def _simulate(entry, data, mutation=None):
    """fp_qmv_quad_impl lane by lane; returns outputs, stores, reads, offsets, paths."""
    info = INFO[entry]
    geometry = _geometry(info["bits"], info["groupSize"], info["D"])
    storage = {name: values for name, (_, values) in data["values"].items()}
    in_vec_size, out_vec_size = storage["in_vec_size"][0], storage["out_vec_size"][0]
    assert in_vec_size == info["D"], entry
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
        bases = (0, 0, 0, 0, 0)
        if info["variant"] != "batch_0":
            bases = _adjust(info, storage, z, mutation, records)
        y_base = bases[4]
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:3],
                "tid_x": tid_x,
                "outVecSize": out_vec_size,
                "sizes": (
                    in_vec_size,
                    in_vec_size // geometry["packFactor"],
                    in_vec_size // info["groupSize"],
                ),
            }
            for tid_y in range(grid_y):
                for quad_gid in range(QUADS_PER_SIMD):
                    out_row = tid_y * TILE_ROWS
                    if mutation != "rows":
                        out_row += quad_gid
                    lanes = _quad(context, out_row, mutation)
                    rows = [
                        row
                        for row in range(RESULTS_PER_QUADGROUP)
                        if row * QUADS_PER_SIMD + out_row < out_vec_size
                    ]
                    if not rows:
                        paths.add("idle")
                    elif len(rows) < RESULTS_PER_QUADGROUP:
                        paths.add("partial")
                    else:
                        paths.add("whole")
                    for row in rows:
                        # quad_sum: exact in any order; lane 0 stores
                        # static_cast<T>.
                        total = 0.0
                        for result in lanes[:1] if mutation == "quad" else lanes:
                            total = _checked(total + result[row])
                        index = y_base + tid_x * out_vec_size
                        index += out_row + row * QUADS_PER_SIMD
                        assert 0 <= index < len(outputs["y"]) - GUARD_COUNT, index
                        stores["y", index] += 1
                        outputs["y"][index] = _to_type(info["dtype"], total)
    if data["padding"]:
        paths.add("overread")
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
            f"fp quantized qmv_quad shard {index} is outside [0, {count})"
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
            "fp_qmv_quad_impl lane by lane after adjust_matrix_offsets: "
            "load_vector and qdot per step, Dequantize's E2M1 and E4M3 "
            "decoding, dequantize_scale's E4M3 and E8M0 scales loaded before "
            "each row guard, quad_sum and lane 0's static_cast stores, against "
            "direct dequantized dot products"
        ),
        "packing": qmv["packing"],
        "comparison": qmv["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / 64), B)",
        "k": "the entry's D",
        "cases": {
            variant: {case: dict(zip(("M", "N"), SHAPES[case])) for case in cases}
            for variant, cases in CASES.items()
        },
        "scaleOverread": (
            "rows past N load their scales before the row guard; the cases pad "
            "the scales through the last tile, before the guards"
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
        **shared_scope("qmvQuad"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmv-quad-contract",
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


def test_current_mlx_fp_quantized_qmv_quad_contract_is_exact():
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


def test_fp_quantized_qmv_quad_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmv_quad.current-tree.translation.json" in readme
    assert len(ENTRIES) == 36 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qmvQuad"]
    assert Counter(info["variant"] for info in INFO.values()) == dict.fromkeys(
        VARIANTS, 18
    )
    assert Counter(info["D"] for info in INFO.values()) == {64: 18, 128: 18}
    assert ENTRIES[:4] == (
        "nvfp4_qmv_quad_float_gs_16_b_4_d_64_batch_1",
        "nvfp4_qmv_quad_float_gs_16_b_4_d_64_batch_0",
        "nvfp4_qmv_quad_float_gs_16_b_4_d_128_batch_1",
        "nvfp4_qmv_quad_float_gs_16_b_4_d_128_batch_0",
    )
    assert ENTRIES[-1] == "mxfp4_qmv_quad_float16_t_gs_32_b_4_d_128_batch_0"
    assert KERNEL not in EXCLUDED_KERNELS
    others = set(QUANTIZE_ENTRIES) | set(QMV_ENTRIES) | set(QMV_FAST_ENTRIES)
    assert not set(ENTRIES) & (others | set(QVM_ENTRIES))


def test_fp_quantized_qmv_quad_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert {len(shard) for shard in shards} == {18}
    for shard in shards:
        assert Counter(INFO[entry]["D"] for entry in shard) == {64: 9, 128: 9}
        modes = Counter(INFO[entry]["mode"] for entry in shard)
        assert modes == {mode: 6 for mode, _, _ in MODES}
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_fp_quantized_qmv_quad_geometry_follows_fp_qmv_quad_impl():
    steps = {}
    for mode, group_size, bits in MODES:
        for d, _ in QUAD_VARIANTS:
            info = {"mode": mode, "groupSize": group_size, "bits": bits, "D": d}
            geometry = _geometry(bits, group_size, d)
            assert geometry["packsPerThread"] * geometry["packFactor"] == d // 4
            assert geometry["stepsPerThread"] * geometry["valuesPerStep"] == d // 4
            # The quad's four lanes read every group of a row exactly once.
            groups = [
                group
                for quad_lid in range(QUAD_SIZE)
                for group in _lane_groups(info, quad_lid)
            ]
            assert sorted(set(groups)) == list(range(d // group_size)), (mode, d)
            steps[mode, d] = geometry["stepsPerThread"]
    # Only nvfp4 at D 128 spans two groups per lane.
    assert {key for key, count in steps.items() if count > 1} == {("nvfp4", 128)}


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_quad_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n) == SHAPES[case] and k == info["D"]
        # dispatch_qmv's qmv_quad route; M stays below every vector limit.
        assert k in (64, 128) and info["bits"] in (4, 8) and m < VECTOR_LIMIT
        assert k % info["groupSize"] == 0
        assert (batches > 1) == (info["variant"] == "batch_1")
        assert data["grid"] == (m, -(-n // TILE_ROWS), batches)
        stem = f"{info['mode']}_qmv_quad_{info['itype']}_gs_{info['groupSize']}"
        suffix = "_batch_1" if batches > 1 else "_batch_0"
        assert entry == f"{stem}_b_{info['bits']}_d_{k}{suffix}"
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert set(values) == {name for name, _ in QUAD_PARAMETERS}
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        assert "b_strides" not in params
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]
        scales = values["scales"][1]
        used, padding = data["usedScales"], data["padding"]
        assert len(scales) == used + padding + GUARD_COUNT
        assert scales[used:] == [BYTE_POISON] * (padding + GUARD_COUNT)


def test_fp_quantized_qmv_quad_cases_reach_every_path():
    paths = {variant: set() for variant in VARIANTS}
    for entry in ENTRIES:
        for case in CASES[INFO[entry]["variant"]]:
            run = _simulate(entry, _case_data(entry, case))
            paths[INFO[entry]["variant"]] |= run["paths"]
    assert paths["batch_0"] == {"whole", "partial", "idle", "overread"}
    assert paths["batch_1"] == {"whole", "partial", "idle", "overread"}


def test_fp_quantized_qmv_quad_inputs_reach_every_code():
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
def test_fp_quantized_qmv_quad_model_matches_dot_products(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word and scale of the used matrices is read, the rows
        # past N read only their scales, and no guard or unused activation
        # row is read.
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
        used = {
            (matrix * n + row) * groups + group
            for matrix in matrices
            for row in range(n)
            for group in range(groups)
        }
        assert run["reads"]["scales"] == used | _overread(layout)
        assert max(run["reads"]["scales"]) < data["usedScales"] + data["padding"]


def test_fp_quantized_qmv_quad_mutations_apply_where_they_are_defects():
    applied = Counter(
        mutation
        for entry in ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    )
    assert applied["batch"] == 18 and applied["group-step"] == 6
    for mutation in ("order", "rows", "row-step", "scale-step", "quad", "sign"):
        assert applied[mutation] == len(ENTRIES), mutation


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmv_quad_inputs_keep_float32_exact(dtype):
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
def test_fp_quantized_qmv_quad_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmv_quad_opengl_assertions_name_buffer_elements(entry):
    variant = INFO[entry]["variant"]
    used = set()
    for case in CASES[variant]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _simulate(entry, data)["records"]:
            used.add(expression)
            assert 0 <= value < lengths[buffer] - GUARD_COUNT, (case, expression)
            assert value <= INDEX_BOUND
    # Every precondition is reached, and none is missing.
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[variant])


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_qmv_quad_configs_scope_target_options(tmp_path, target):
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
    assert "instantiate_quantized_all_quad(type, mode, group_size, bits)" in groups
    quads = re.findall(
        r"instantiate_quantized_quad\(mode, (\w+), type, (\d+), (\d), group_size, bits\)",
        _macro(text, "instantiate_quantized_all_quad"),
    )
    assert quads == [("qmv_quad", str(d), str(b)) for d, b in QUAD_VARIANTS]
    quad = _macro(text, "instantiate_quantized_quad")
    assert quad.count("instantiate_kernel(") == 1
    assert '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D' in quad


def _check_kernel(header):
    assert declared_buffers(header, KERNEL) == QUAD_PARAMETERS
    assert "MLX_MTL_CONST int QUAD_SIZE = 4;" in header
    impl = header[header.index("METAL_FUNC void fp_qmv_quad_impl(") :]
    impl = impl[: impl.index("template <", 1)]
    for required in (
        "constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;",
        "constexpr int pack_factor = get_pack_factor<32, bits>();",
        "constexpr int values_per_thread = D / QUAD_SIZE;",
        "values_per_thread < group_size ? 1 : values_per_thread / group_size;",
        "constexpr int values_per_step = values_per_thread / steps_per_thread;",
        "constexpr int packs_per_thread = values_per_thread / pack_factor;",
        "constexpr int packs_per_step = values_per_step / pack_factor;",
        "constexpr int results_per_quadgroup = 8;",
        "const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;",
        "w += out_row * in_vec_size_w + quad_lid * packs_per_thread;",
        "out_row * in_vec_size_g + (quad_lid * values_per_thread) / group_size;",
        "x += tid.x * in_vec_size + quad_lid * values_per_thread;",
        "load_vector<T, U, values_per_thread>(x, x_thread);",
        "U s = dequantize_scale<U, group_size>(sl[0]);",
        "if (row * quads_per_simd + out_row < out_vec_size) {",
        "wl, x_thread + k * values_per_step, s);",
        "wl += (sizeof(uint32_t) / sizeof(uint8_t)) * packs_per_step;",
        "result[row] = quad_sum(result[row]);",
        "if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {",
        "y[row * quads_per_simd] = static_cast<T>(result[row]);",
    ):
        assert required in impl, required
    # The scale load precedes the row guard.
    assert impl.index("U s = dequantize_scale") < impl.index(
        "if (row * quads_per_simd + out_row < out_vec_size) {"
    )
    body = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    body = body[: body.index("\n}\n")]
    assert "if (batched) {" in body
    assert "adjust_matrix_offsets(\n        x,\n        w,\n        scales,\n" in body
    assert "out_vec_size * M," in body
    assert "fp_qmv_quad_impl<T, group_size, bits, D>(" in body


def _check_host(host):
    body = host[host.index("void qmv_quad(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "constexpr int quads_per_simd = 8;",
        "constexpr int results_per_quadgroup = 8;",
        "int bn = quads_per_simd * results_per_quadgroup;",
        "MTL::Size group_dims(simdgroup_size, 1, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        'mode + "_qmv_quad_",',
        '"_d_",',
        'B > 1 ? "_batch_1" : "_batch_0");',
        "compute_encoder.set_input_array(w, c++);",
        "compute_encoder.set_input_array(scales, c++);",
        "compute_encoder.set_input_array(x, c++);",
        "compute_encoder.set_output_array(out, c++);",
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);",
    ):
        assert required in body, required
    assert (
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) {"
        "\n    qmv_quad(" in host
    )


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmv_quad", TARGETS, MLX_COMMIT
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
    work = tmp_path_factory.mktemp("fp-quantized-qmv-quad-metal-reference")
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
def test_current_mlx_fp_quantized_qmv_quad_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    variant = INFO[entry]["variant"]
    with tempfile.TemporaryDirectory(prefix=".current-fpquad-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            descriptor, package = translate_entry(root, work, target, config, record)
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
                        QUAD_PARAMETERS,
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
                            QUAD_PARAMETERS,
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


def test_ci_requires_native_fp_quantized_qmv_quad_parity():
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv_quad.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmv_quad.py",
    )
