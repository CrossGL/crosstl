"""Natively prove quantized.metal's 216 affine_qmv_quad kernels.

``quantized.cpp``'s ``dispatch_qmv`` sends every matrix-vector product whose
``K`` is 64 or 128 and whose width is a power of two to ``qmv_quad``.
``quantized.metal`` instantiates ``affine_qmv_quad`` for float, float16_t and
bfloat16_t, group sizes 128, 64 and 32, 2, 3, 4, 5, 6 and 8 bits and ``D`` of
64 and 128, with and without batching: 216 of its 2,052 entries. Each is
translated, pinned and validated per target, except where the upstream body
is undefined. For 3, 5 and 6 bits, ``qmv_quad_impl``'s ``32 / bits`` pack
factor misplaces each lane's codes and ``qdot``'s cumulative ``x_thread``
steps run past the lane's ``D / 4`` values once a lane takes three or more
steps: 3 and 5 bits at ``D`` 128 and 6 bits at both widths, 72 entries.
``dispatch_qmv`` routes only power-of-two widths there. DirectX and OpenGL
refuse those 72 with their private-pointer bounds diagnostic, which the gate
derives from the same pointer steps; Metal keeps the upstream arithmetic.
The 90 entries the host can dispatch, power-of-two widths whose group
size divides ``D``, then run ``qmv_quad``'s ``(32, 1, 1)`` threadgroups over
an ``(M, ceil(N / 64), B)`` grid. The other 126 have no runtime case.
``quantize`` requires ``K`` to be a multiple of the group size, so no weight
reaches ``D`` 64 with groups of 128.

``qmv_quad_impl`` gives each of a simdgroup's eight quads eight output rows,
eight apart, and each lane of a quad a quarter of ``K``. A row past ``N``
skips its ``qdot`` and its store but still loads its scale and bias, so the
cases pad both buffers through the last tile and the model shows those values
never reach an output. ``quad_sum`` adds a quad's four partial sums; DirectX
and OpenGL lower it to a software reduction over four consecutive
invocations, the grouping of Metal's ``quadgroup_index_in_threadgroup``. The
cases reuse the qmv_fast gate's inputs, so float32 holds every intermediate
exactly and only the final ``static_cast<T>`` rounds. A model replays each
lane and must reach direct dequantized dot products, storing every row once.
Each target must produce the model's words and leave every guard untouched;
on Metal the translation must also match the unchanged upstream kernel word
for word.
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

from crosstl.project import translate_project
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
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    MUTATIONS as QMV_FAST_MUTATIONS,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    OPENGL_INDEX_RANGE_ASSERTIONS,
    POISON,
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
    SIMD_SIZE,
    UNBATCHED_BOUND,
    _adjust,
    _checked,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _config as _qmv_config,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _expected_validation_contract as _qmv_fast_validation_contract,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    _f32,
    _load_vector,
    _qdot,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED_QMV_QUAD"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/quantized_qmv_quad.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "affine_qmv_quad"
# instantiate_quantized_all_quad's (D, batched) pairs, in order.
QUAD_VARIANTS = ((64, 1), (64, 0), (128, 1), (128, 0))
VARIANTS = ("batch_1", "batch_0")
QUAD_SIZE = 4
QUADS_PER_SIMD = SIMD_SIZE // QUAD_SIZE
RESULTS_PER_QUADGROUP = 8
# Output rows per threadgroup, quantized.cpp's bn.
TILE_ROWS = QUADS_PER_SIMD * RESULTS_PER_QUADGROUP
WORKGROUP = (SIMD_SIZE, 1, 1)

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


def _dispatched(bits, group_size, d):
    """Whether the host can send a weight to this instantiation at all."""
    return bits & (bits - 1) == 0 and d % group_size == 0


# qdot's 3-, 5- and 6-bit loops advance x_thread by ``step * i`` before
# iteration i reads its ``step`` values, so the offsets accumulate.
QDOT_CUMULATIVE_STEPS = {3: 8, 5: 8, 6: 4}
# DirectX and OpenGL refuse a private pointer view that reaches past its
# backing array; Metal keeps the upstream pointer arithmetic.
REFUSAL_TARGETS = ("opengl", "directx")
REFUSAL_CODES = {
    "opengl": "project.translate.opengl-private-pointer-unsupported",
    "directx": "project.translate.directx-private-pointer-unsupported",
}
REFUSAL_CAPABILITIES = {
    "opengl": "opengl.private-pointer-parameter-lowering",
    "directx": "directx.private-pointer-parameter-lowering",
}


def _x_thread_reach(bits, d):
    """``(elements qdot reads, extent)`` of one lane's ``x_thread``.

    Replays qdot's pointer steps rather than a closed form.
    """
    extent = d // QUAD_SIZE
    step = QDOT_CUMULATIVE_STEPS.get(bits)
    if step is None:
        return extent, extent
    offset = reach = 0
    for i in range(extent // step):
        offset += step * i
        reach = max(reach, offset + step)
    return reach, extent


def _overruns(bits, d):
    reach, extent = _x_thread_reach(bits, d)
    return reach > extent


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for d, batched in QUAD_VARIANTS:
                    entry = (
                        f"{KERNEL}_{itype}_gs_{group_size}_b_{bits}"
                        f"_d_{d}_batch_{batched}"
                    )
                    yield entry, {
                        "kernel": KERNEL,
                        "variant": f"batch_{batched}",
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                        "D": d,
                        "dispatched": _dispatched(bits, group_size, d),
                        "overruns": _overruns(bits, d),
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)
RUNTIME_ENTRIES = tuple(entry for entry in ENTRIES if INFO[entry]["dispatched"])
OVERRUN_ENTRIES = tuple(entry for entry in ENTRIES if INFO[entry]["overruns"])


def _targets(entry):
    """The targets that must translate ``entry``; the others refuse it."""
    if INFO[entry]["overruns"]:
        return tuple(target for target in TARGETS if target not in REFUSAL_TARGETS)
    return TARGETS


def _refusal(target, entry):
    """The diagnostic ``target`` must raise for an overrunning ``entry``."""
    info = INFO[entry]
    reach, extent = _x_thread_reach(info["bits"], info["D"])
    view = f"qdot_float_{extent}_{info['bits']}.x_thread"
    if target == "directx":
        need = f"DirectX private pointer view '{view}' requires elements 0 through"
        need = f"{need} {reach - 1}"
    else:
        need = f"OpenGL private pointer view '{view}' requires at least {reach}"
        need = f"{need} elements"
    return {
        "code": REFUSAL_CODES[target],
        "message": f"{need}, but backing array 'x_thread' has extent {extent}",
        "missingCapabilities": [REFUSAL_CAPABILITIES[target]],
        "privatePointer": {
            "function": view.split(".")[0],
            "parameter": "x_thread",
            "reason": "view-out-of-bounds",
        },
    }


def _config(directory, target, entry):
    """The translation of one entry, with the qmv_fast gate's options."""
    return _qmv_config(
        directory, target, entry, INFO[entry]["variant"], workgroup=WORKGROUP
    )


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _geometry(bits, group_size, d):
    """qmv_quad_impl's constants for a power-of-two width."""
    pack_factor = 32 // bits
    values_per_thread = d // QUAD_SIZE
    return {
        "packFactor": pack_factor,
        "valuesPerThread": values_per_thread,
        "packsPerThread": values_per_thread // pack_factor,
        "scaleStepPerThread": group_size // values_per_thread,
    }


def _overread(layout):
    """Scale and bias elements the rows past ``N`` of each last tile load."""
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
    """The qmv_fast gate's inputs over qmv_quad's grid, padded for overreads."""
    info = INFO[entry]
    assert info["dispatched"] and case in CASES[info["variant"]], (entry, case)
    m, n = SHAPES[case]
    layout = layout_for(info, LAYOUTS[case], m, n, info["D"])
    layout["grid"] = (m, -(-n // TILE_ROWS), len(layout["batches"]))
    data = case_data_for(info, entry, case, layout)
    used = layout["matrices"] * n * layout["groups"]
    padding = max(0, max(_overread(layout), default=-1) + 1 - used)
    values = dict(data["values"])
    for name in ("scales", "biases"):
        kind, storage = values[name]
        assert len(storage) == used + GUARD_COUNT, (entry, case, name)
        values[name] = (kind, storage + [POISON] * padding)
    return {**data, "values": values, "padding": padding}


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

MUTATIONS = {
    **{
        name: description
        for name, description in QMV_FAST_MUTATIONS.items()
        if name != "rows"
    },
    "rows": "ignores the quad's row offset",
    "quad": "stores lane 0's partial sum instead of the quad_sum",
}


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "batch":
        return info["variant"] != "batch_0"
    if mutation == "scale-step":
        # A lane reads its own group only when a quad spans several.
        return info["D"] > info["groupSize"]
    return True


def _quad(context, out_row, mutation):
    """One quad's four lanes: per-row partial sums and the rows they load."""
    info, geometry, load = context["info"], context["geometry"], context["load"]
    x_base, w_base, s_base, b_base = context["bases"]
    in_vec_size, in_vec_size_w, in_vec_size_g = context["sizes"]
    out_vec_size = context["outVecSize"]
    per_thread, pack_factor = geometry["valuesPerThread"], geometry["packFactor"]
    lanes = []
    for quad_lid in range(QUAD_SIZE):
        ws = 4 * (w_base + out_row * in_vec_size_w)
        ws += 4 * quad_lid * geometry["packsPerThread"]
        group = 0
        if mutation != "scale-step":
            group = quad_lid // geometry["scaleStepPerThread"]
        scales = s_base + out_row * in_vec_size_g + group
        biases = b_base + out_row * in_vec_size_g + group
        x = x_base + context["tid_x"] * in_vec_size + quad_lid * per_thread
        values = [load("x", x + i) for i in range(per_thread)]
        if mutation == "order":
            values = [
                values[start + pack_factor - 1 - i]
                for start in range(0, per_thread, pack_factor)
                for i in range(pack_factor)
            ]
        x_thread, total = _load_vector(info["dtype"], info["bits"], values)
        result = [0.0] * RESULTS_PER_QUADGROUP
        for row in range(RESULTS_PER_QUADGROUP):
            step = row * QUADS_PER_SIMD
            # Each row loads its scale and bias before the row guard.
            scale = load("scales", scales + step * in_vec_size_g)
            bias = load("biases", biases + step * in_vec_size_g)
            if step + out_row < out_vec_size:
                result[row] = _checked(
                    result[row]
                    + _qdot(
                        info["bits"],
                        context["byte"],
                        ws + 4 * step * in_vec_size_w,
                        x_thread,
                        scale,
                        bias,
                        total,
                        mutation,
                    )
                )
        lanes.append(result)
    return lanes


def _simulate(entry, data, mutation=None):
    """qmv_quad_impl lane by lane; returns outputs, stores, reads, offsets, paths."""
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
            bases = _adjust(info["variant"], storage, z, mutation, records)
        y_base = bases[4]
        for tid_x in range(grid_x):
            context = {
                "info": info,
                "geometry": geometry,
                "load": load,
                "byte": byte,
                "bases": bases[:4],
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
                    for row in range(RESULTS_PER_QUADGROUP):
                        # quad_sum: exact in any order; every lane gets it.
                        total = 0.0
                        for result in lanes[:1] if mutation == "quad" else lanes:
                            total = _checked(total + result[row])
                        if row not in rows:
                            continue
                        index = y_base + tid_x * out_vec_size
                        index += out_row + row * QUADS_PER_SIMD
                        assert 0 <= index < len(outputs["y"]) - GUARD_COUNT, index
                        stores["y", index] += 1
                        outputs["y"][index] = _to_type(info["dtype"], total)
    if data.get("padding"):
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

SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_QUAD_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_QUANTIZED_QMV_QUAD_SHARD_COUNT"
CI_SHARD_COUNT = 3


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"qmv_quad shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


def _dispatch_count():
    return sum(len(CASES[INFO[entry]["variant"]]) for entry in RUNTIME_ENTRIES)


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        len(_case_data(entry, case)["expected"]["y"])
        for entry in RUNTIME_ENTRIES
        for case in CASES[INFO[entry]["variant"]]
    )


def _expected_validation_contract():
    fast = _qmv_fast_validation_contract()
    return {
        "model": (
            "qmv_quad_impl lane by lane after adjust_matrix_offsets: each "
            "quad's eight rows eight apart, every lane's quarter of K through "
            "load_vector and qdot, the row guard after each row's scale and "
            "bias loads, quad_sum and lane 0's static_cast stores, against "
            "direct dequantized dot products"
        ),
        "packing": fast["packing"],
        "comparison": fast["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(M, ceil(N / 64), B)",
        "k": "D",
        "runtimeEntries": (
            "power-of-two widths whose group size divides D; quantize needs K "
            "to be a multiple of the group size, and dispatch_qmv sends no "
            "other width to qmv_quad"
        ),
        "cases": {
            variant: {case: dict(zip(("M", "N"), SHAPES[case])) for case in cases}
            for variant, cases in CASES.items()
        },
        "inputs": fast["inputs"],
        "overreadPadding": (
            "scales and biases continue with poison through the last tile's "
            "rows, which load them before the row guard"
        ),
        "exactness": fast["exactness"],
        "guards": fast["guards"],
        "softwareSubgroups": {
            **fast["softwareSubgroups"],
            "quadSum": "four consecutive invocations, added in lane order",
        },
        "openglIndexRangeAssertions": {
            variant: fast["openglIndexRangeAssertions"][variant] for variant in VARIANTS
        },
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": fast["metalUpstreamComparison"],
    }


def _covered_entry_count():
    return (
        QUANTIZE_ENTRY_COUNT
        + len(QMV_FAST_ENTRIES)
        + QMV_UNALIGNED_ENTRY_COUNT
        + QMV_WIDE_ENTRY_COUNT
        + QVM_ENTRY_COUNT
        + QVM_SPLIT_K_ENTRY_COUNT
        + QMM_T_ENTRY_COUNT
        + QMM_N_ENTRY_COUNT
        + QMM_T_SPLITK_ENTRY_COUNT
        + GATHER_QMM_ENTRY_COUNT
        + len(ENTRIES)
    )


def _expected_scope():
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
        "gatherQmmGateEntryCount": GATHER_QMM_ENTRY_COUNT,
        "excludedEntries": {
            "kernels": list(EXCLUDED_KERNELS),
            "count": UNIT_ENTRY_COUNT - _covered_entry_count(),
            "reason": "the gather_qmm_rhs matrix-matrix kernels are not gated yet",
        },
        "runtimeCoveredEntryCount": len(RUNTIME_ENTRIES),
        "artifactCountByTarget": {
            target: sum(target in _targets(entry) for entry in ENTRIES)
            for target in TARGETS
        },
        "refusedEntryCountByTarget": {
            target: len(OVERRUN_ENTRIES) for target in REFUSAL_TARGETS
        },
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
        "kind": "crosstl-mlx-current-tree-quantized-qmv-quad-contract",
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
        "runtimeEntries": list(RUNTIME_ENTRIES),
        "artifacts": artifacts,
        "refusals": {
            "entries": list(OVERRUN_ENTRIES),
            "targets": list(REFUSAL_TARGETS),
            "codes": dict(REFUSAL_CODES),
            "reason": (
                "qdot's 3-, 5- and 6-bit loops advance x_thread cumulatively, so "
                "with D / 4 values per lane they read past the lane's x_thread "
                "array; out-of-bounds pointer arithmetic is undefined in the C++14 "
                "the Metal Shading Language builds on ([expr.add]), and no host "
                "path dispatches these instantiations"
            ),
        },
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(),
    }


def test_current_mlx_quantized_qmv_quad_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    artifacts = contract["artifacts"]
    assert list(artifacts) == list(ENTRIES)
    for entry, pins in artifacts.items():
        assert list(pins) == list(_targets(entry))
        for pin in pins.values():
            assert re.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*", pin)


def test_quantized_qmv_quad_refusals_are_the_overrunning_qdots():
    """Exactly the 3-, 5- and 6-bit qdots that run past x_thread are refused."""
    assert {(INFO[e]["bits"], INFO[e]["D"]) for e in OVERRUN_ENTRIES} == {
        (3, 128),
        (5, 128),
        (6, 64),
        (6, 128),
    }
    assert len(OVERRUN_ENTRIES) == 72
    assert not any(INFO[entry]["dispatched"] for entry in OVERRUN_ENTRIES)
    assert _x_thread_reach(3, 128) == (56, 32)
    assert _x_thread_reach(6, 128) == (116, 32)
    assert _x_thread_reach(6, 64) == (28, 16)
    # Two iterations stay inside the array: offsets 0 and step.
    assert _x_thread_reach(3, 64) == _x_thread_reach(5, 64) == (16, 16)
    for bits in (2, 4, 8):
        assert _x_thread_reach(bits, 128) == (32, 32)
    entry = "affine_qmv_quad_float_gs_128_b_6_d_64_batch_1"
    assert _refusal("directx", entry)["message"] == (
        "DirectX private pointer view 'qdot_float_16_6.x_thread' requires "
        "elements 0 through 27, but backing array 'x_thread' has extent 16"
    )
    assert _refusal("opengl", entry)["message"] == (
        "OpenGL private pointer view 'qdot_float_16_6.x_thread' requires at "
        "least 28 elements, but backing array 'x_thread' has extent 16"
    )


def test_quantized_qmv_quad_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "quantized_qmv_quad.current-tree.translation.json" in readme
    assert len(ENTRIES) == 216 == len(set(ENTRIES)) == QMV_QUAD_ENTRY_COUNT
    assert Counter(info["variant"] for info in INFO.values()) == {
        "batch_1": 108,
        "batch_0": 108,
    }
    assert ENTRIES[:4] == (
        "affine_qmv_quad_float_gs_128_b_2_d_64_batch_1",
        "affine_qmv_quad_float_gs_128_b_2_d_64_batch_0",
        "affine_qmv_quad_float_gs_128_b_2_d_128_batch_1",
        "affine_qmv_quad_float_gs_128_b_2_d_128_batch_0",
    )
    assert ENTRIES[-1] == "affine_qmv_quad_bfloat16_t_gs_32_b_8_d_128_batch_0"
    assert KERNEL not in EXCLUDED_KERNELS
    assert not set(ENTRIES) & set(QMV_FAST_ENTRIES)
    assert set(SHAPES) == set(LAYOUTS) == {c for cs in CASES.values() for c in cs}
    # The rest of quantized.metal: affine_gather_qmm_rhs_nt and _nn.
    assert _covered_entry_count() + 108 == UNIT_ENTRY_COUNT


def test_quantized_qmv_quad_runtime_entries_are_the_dispatchable_ones():
    assert len(RUNTIME_ENTRIES) == 90
    skipped = Counter(
        "width" if INFO[entry]["bits"] in (3, 5, 6) else "group"
        for entry in ENTRIES
        if entry not in RUNTIME_ENTRIES
    )
    # Three widths x 3 group sizes x 3 types x 4 variants, and groups of 128
    # at D 64 for the other three widths, types and both batchings.
    assert skipped == {"width": 108, "group": 18}
    for entry in RUNTIME_ENTRIES:
        info = INFO[entry]
        geometry = _geometry(info["bits"], info["groupSize"], info["D"])
        # A quad covers its D codes in whole packs and whole groups per lane
        # or lanes per group.
        assert geometry["packsPerThread"] * geometry["packFactor"] * QUAD_SIZE == (
            info["D"]
        )
        assert geometry["scaleStepPerThread"] in (1, 2, 4), entry


def test_quantized_qmv_quad_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [72, 72, 72]
    for shard in shards:
        counts = Counter(INFO[entry]["variant"] for entry in shard)
        assert counts == {"batch_1": 36, "batch_0": 36}, counts
        assert sum(INFO[entry]["dispatched"] for entry in shard) == 30
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", RUNTIME_ENTRIES[::5])
def test_quantized_qmv_quad_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n) == SHAPES[case] and k == info["D"]
        # dispatch_qmv's qmv_quad route; M stays below every vector limit.
        assert k in (64, 128) and info["bits"] in (2, 4, 8) and m < 4
        assert k % info["groupSize"] == 0
        assert (batches > 1) == (info["variant"] == "batch_1")
        assert data["grid"] == (m, -(-n // TILE_ROWS), batches)
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["in_vec_size"][1] == [k] and values["out_vec_size"][1] == [n]
        params = layout["params"]
        assert len(params["x_shape"]) == params["x_batch_ndims"] + 2
        assert len(params["w_shape"]) == params["w_batch_ndims"] + 2
        assert params["x_shape"][-2:] == [m, k]
        assert params["w_shape"][-2:] == [n, layout["words"]]


def test_quantized_qmv_quad_cases_reach_every_path():
    paths = {variant: set() for variant in VARIANTS}
    for entry in RUNTIME_ENTRIES:
        for case in CASES[INFO[entry]["variant"]]:
            run = _simulate(entry, _case_data(entry, case))
            paths[INFO[entry]["variant"]] |= run["paths"]
    every = {"whole", "partial", "idle", "overread"}
    assert paths == {"batch_1": every, "batch_0": every}


@pytest.mark.parametrize("entry", RUNTIME_ENTRIES)
def test_quantized_qmv_quad_model_matches_dot_products(entry):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        assert run["outputs"] == data["expected"], (entry, case)
        layout = data["layout"]
        m, n = layout["M"], layout["N"]
        written = len(data["expected"]["y"]) - GUARD_COUNT
        assert len(run["stores"]) == written
        assert all(run["stores"]["y", index] == 1 for index in range(written))
        assert run["outputs"]["y"][written:] == data["initial"]["y"][written:]
        # Every weight word of the used matrices is read, every used
        # activation row, and the scales and biases of every used row and of
        # the last tile's rows past N; no guard is.
        assert run["reads"]["x"] == {
            batch["x"] + row * layout["K"] + index
            for batch in layout["batches"]
            for row in range(m)
            for index in range(layout["K"])
        }
        matrices = {batch["w"] for batch in layout["batches"]}
        words, groups = layout["words"], layout["groups"]
        assert run["reads"]["w"] == {
            (matrix * n + col) * words + word
            for matrix in matrices
            for col in range(n)
            for word in range(words)
        }
        used = {
            (matrix * n + col) * groups + group
            for matrix in matrices
            for col in range(n)
            for group in range(groups)
        }
        overread = _overread(layout)
        assert data["padding"] == max(
            0, max(overread, default=-1) + 1 - layout["matrices"] * n * groups
        )
        for name in ("scales", "biases"):
            assert run["reads"][name] == used | overread, (case, name)


def test_quantized_qmv_quad_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in RUNTIME_ENTRIES
        for mutation in MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    # 45 unbatched entries, and one group per quad at D 64 with groups of 64
    # and at D 128 with groups of 128: 2 x 3 widths x 3 types x 2 batchings.
    assert skipped == {"batch": 45, "scale-step": 36}


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_qmv_quad_inputs_keep_float32_exact(dtype):
    rounded = ties = 0
    for entry in RUNTIME_ENTRIES:
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


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in RUNTIME_ENTRIES
        for mutation in MUTATIONS
        if _mutation_applies(entry, mutation)
    ],
)
def test_quantized_qmv_quad_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[mutation])


@pytest.mark.parametrize("entry", RUNTIME_ENTRIES)
def test_quantized_qmv_quad_opengl_assertions_name_buffer_elements(entry):
    variant = INFO[entry]["variant"]
    used = set()
    for case in CASES[variant]:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, buffer, value in _simulate(entry, data)["records"]:
            used.add(expression)
            assert 0 <= value < lengths[buffer] - GUARD_COUNT, (case, expression)
            assert value <= INDEX_BOUND
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[variant])


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_qmv_quad_configs_scope_target_options(tmp_path, target):
    for variant in VARIANTS:
        entry = next(e for e in ENTRIES if INFO[e]["variant"] == variant)
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        assert options["directx"]["software_subgroup_width"] == SIMD_SIZE
        assert options["directx"]["relative_wave_shuffle_out_of_range"] == "self"
        assert options["opengl"]["software_subgroup_width"] == SIMD_SIZE
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "opengl":
            assert expressions == list(OPENGL_INDEX_RANGE_ASSERTIONS[variant])
        else:
            assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "quantized qmv_quad", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMV_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # ENTRIES follow quantized.metal's own instantiation macros.
    text = source.decode("utf-8")
    quads = re.findall(
        r"instantiate_quantized_quad\((\w+), type, group_size, bits, (\d+), (\d)\)",
        _macro(text, "instantiate_quantized_all_quad"),
    )
    assert quads == [(KERNEL, str(d), str(b)) for d, b in QUAD_VARIANTS]
    funcs = _macro(text, "instantiate_quantized_funcs")
    assert "instantiate_quantized_all_quad(type, group_size, bits)" in funcs
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
    assert match
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
    assert tuple((name, int(index)) for name, index in found) == QMV_PARAMETERS
    impl = header[header.index("METAL_FUNC void qmv_quad_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;",
        "constexpr int pack_factor = 32 / bits;",
        "constexpr int values_per_thread = D / QUAD_SIZE;",
        "constexpr int packs_per_thread = values_per_thread / pack_factor;",
        "constexpr int scale_step_per_thread = group_size / values_per_thread;",
        "constexpr int results_per_quadgroup = 8;",
        "const int out_row = tid.y * quads_per_simd * results_per_quadgroup + quad_gid;",
        "w += out_row * in_vec_size_w + quad_lid * packs_per_thread;",
        "scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;",
        "x += tid.x * in_vec_size + quad_lid * values_per_thread;",
        "U s = sl[0];",
        "if (row * quads_per_simd + out_row < out_vec_size) {",
        "result[row] = quad_sum(result[row]);",
        "if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {",
    ):
        assert required in impl, required
    assert "MLX_MTL_CONST int QUAD_SIZE = 4;" in header
    host = (root / HOST_SOURCE).read_text(encoding="utf-8")
    quad_host = host[host.index("void qmv_quad(") :]
    quad_host = quad_host[: quad_host.index("\nvoid ", 1)]
    for required in (
        "constexpr int quads_per_simd = 8;",
        "constexpr int results_per_quadgroup = 8;",
        "int simdgroup_size = 32;",
        "MTL::Size group_dims(simdgroup_size, 1, 1);",
        "MTL::Size grid_dims(M, (N + bn - 1) / bn, B);",
        'mode + "_qmv_quad_",',
        '"_d_",',
        'B > 1 ? "_batch_1" : "_batch_0"',
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);",
    ):
        assert required in quad_host, required
    assert (
        "if ((K == 128 || K == 64) && is_power_of_2(bits) && !global_scale) {" in host
    )
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-qmv-quad-metal-reference")
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


def _run_cases(work, target, entry, descriptor, package, reference):
    """Every case of a dispatchable entry on the native backend."""
    variant = INFO[entry]["variant"]
    executor = _executor(target)
    evidence = []
    try:
        for case in CASES[variant]:
            data = _case_data(entry, case)
            request, output, observed = _request(
                descriptor, package, target, variant, data, workgroup=WORKGROUP
            )
            words = _execute(executor, work, case, request, {"y": output}, observed)[
                "y"
            ]["values"]
            upstream = None
            if target == "metal":
                device, upstream_words = _upstream(
                    work, reference, entry, variant, case, data, workgroup=WORKGROUP
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
                    "overreadPadding": data["padding"],
                    "checkedElementCount": len(data["expected"]["y"]),
                    "observedInputCount": observed,
                    "upstream": upstream,
                }
            )
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if callable(close):
            close()
    return evidence


def _refuse(work, target, config, entry):
    """Translate an overrunning entry and require its fail-closed diagnostic."""
    translation = translate_project(
        config, format_output=False, validate=True, run_toolchains=False
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    expected = _refusal(target, entry)
    # Validation reports the failed artifact once more, with the same cause.
    diagnostic, follow_up = report["diagnostics"]
    assert follow_up["code"] == "project.validate.failed-artifact"
    assert follow_up["severity"] == "error" and follow_up["target"] == target
    assert follow_up["message"].endswith(expected["message"])
    assert diagnostic["severity"] == "error"
    assert diagnostic["target"] == target
    for key in ("code", "message", "missingCapabilities"):
        assert diagnostic[key] == expected[key], key
    assert diagnostic["details"]["privatePointer"] == expected["privatePointer"]
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    (artifact,) = report["artifacts"]
    assert artifact["status"] == "failed"
    assert "generatedHash" not in artifact
    assert not list((work / "out").rglob("*.*"))
    return expected


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_current_mlx_quantized_qmv_quad_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    with tempfile.TemporaryDirectory(prefix=".current-qmvq-", dir=root) as path:
        work = Path(path)
        try:
            config = _config(work, target, entry)
            if target not in _targets(entry):
                refusal = _refuse(work, target, config, entry)
                (work / "refusal.json").write_text(
                    json.dumps(
                        {"commit": MLX_COMMIT, "entry": entry, **refusal}, indent=2
                    ),
                    encoding="utf-8",
                )
                return
            record = _record(target, entry)
            descriptor, package = _translate(
                root, work, target, config, record, workgroup=WORKGROUP
            )
            evidence = []
            if INFO[entry]["dispatched"]:
                evidence = _run_cases(
                    work, target, entry, descriptor, package, metal_reference
                )
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "dispatched": INFO[entry]["dispatched"],
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


def test_ci_requires_native_quantized_qmv_quad_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_shard_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_shard_step(
        workflow,
        "Prove current MLX quantized quad matrix-vector native parity",
        family="quantized-qmv-quad",
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
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv_quad.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_quantized_qmv_quad.py",
    )
