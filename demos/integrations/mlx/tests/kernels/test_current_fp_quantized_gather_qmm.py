"""Prove fp_quantized.metal's fp_gather_qmm_t and fp_gather_qmm_n entries natively.

``quantized.cpp``'s ``GatherQMM`` multiplies, per output batch, the
activation matrix ``lhs_indices`` selects by the weight matrix
``rhs_indices`` selects. On devices without NAX kernels, with ``M`` at or
above the vector limit (33 for transposed weights up to 2,048 by 2,048, 4
otherwise) and no sorted single-row batch for ``gather_qmm_rhs``, its
``gather_qmm`` dispatches ``fp_gather_qmm_t`` for transposed weights and
``fp_gather_qmm_n`` otherwise, the ``_hgs`` entry when the weights come with
one global scale per expert. ``fp_quantized.metal`` instantiates both for
nvfp4 (groups of 16 E2M1 codes with E4M3 scales), mxfp8 (groups of 32 E4M3
codes with E8M0 scales) and mxfp4 (groups of 32 E2M1 codes with E8M0 scales)
in float, bfloat16_t and float16_t, ``fp_gather_qmm_t`` with aligned and
unaligned ``N``, each without and with a global scale: 54 of its 518 entries.
``ops.cpp`` admits a global scale for nvfp4 only, so the host never
dispatches the mx ``_hgs`` entries; they are instantiated and run all the
same. Each entry is translated and pinned per target, then runs ``(32, 2, 2)``
threadgroups over a ``(ceil(N / 32), ceil(M / 32), B)`` grid.

Both kernels first run the gather ``adjust_matrix_offsets``: ``tid.z``'s lhs
and rhs indices, read through ``lhs_strides`` and ``rhs_strides`` or, for
several batch dimensions, ``elem_to_loc_broadcast`` over ``batch_shape``,
select the activation matrix and the weight and scale matrices,
``global_scale`` moves to the expert's and ``y`` to ``tid.z``'s ``M`` by
``N`` slice. ``fp_gather_qmm_t`` then runs ``fp_qmm_t_impl`` over the whole
reduction and ``fp_gather_qmm_n`` ``fp_qmm_n_impl``, which
``test_current_fp_quantized_qmm_t.py`` and
``test_current_fp_quantized_qmm_n.py`` gate. Unlike ``fp_qmm_n``,
``fp_gather_qmm_n`` passes its global scale on, so both loaders fold the
expert's ``inv_scale_enc`` into each group scale and round every dequantized
weight to the element type. The cases take the fp_gather_qmv gate's two index
layouts: interleaved one-dimensional indices into two activation and three
weight matrices, and a ``(2, 2)`` batch whose rhs indices broadcast along its
second dimension. Their shapes reach every load, tail and store path of both
implementations with the fp_qmv gate's inputs, which keep every float32
product and partial sum exact. The per-expert global scales make the E2M1
weights round in bfloat16 and float16; rounded float32 or E4M3 weights would
need more bits than float32 sums hold, so float32 and mxfp8 entries take
global scales whose ``inv_scale_enc`` is exact. The tile-level models of both implementations
replay the gathered offsets and must reach direct dot products of the
dequantized weights, storing every output once. Each target must produce the
models' words and leave every guard untouched; on Metal the translation must
also match the unchanged upstream kernel word for word.
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

from demos.integrations.mlx.tests.kernels import fp_qmm_n_model as n_model
from demos.integrations.mlx.tests.kernels import fp_qmm_t_model as t_model
from demos.integrations.mlx.tests.kernels import (
    test_current_fp_quantized_qmm_n as n_gate,
)
from demos.integrations.mlx.tests.kernels import (
    test_current_fp_quantized_qmm_t as t_gate,
)
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    declared_buffers,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    GATHER_PARAMETERS,
    GLOBAL_SCALES,
    GUARD_COUNT,
    MODES,
    POISON,
    _batch_expert,
    _code_value,
    _fraction_bits,
    _inverse_scale,
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
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import BK, BM, BN
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
    QMM_FP_HEADERS,
    REQUIRE_ENV,
    SHARD_COUNT_ENV,
    SHARD_INDEX_ENV,
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
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    VECTOR_LIMIT as TRANSPOSED_VECTOR_LIMIT,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmm_t import (
    WORKGROUP,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    BROADCAST_LHS,
    BROADCAST_RHS,
    EXPERT_LHS,
    EXPERT_RHS,
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
    / "demos/integrations/mlx/contracts/fp_quantized_gather_qmm.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNELS = {True: "fp_gather_qmm_t", False: "fp_gather_qmm_n"}
# Per (type, mode), instantiate_quantized_all_single's fp_gather_qmm_n
# precedes instantiate_quantized_all_aligned's two fp_gather_qmm_t entries,
# aligned N first; each entry is followed by its _hgs twin.
VARIANTS = (
    ("", False, True),
    ("_alN_true", True, True),
    ("_alN_false", True, False),
)
# fp_gather_qmv's buffers with K, N and M in place of its two vector sizes,
# so every later buffer moves up one. gather_qmm binds all of them.
PARAMETERS = (
    GATHER_PARAMETERS[:7]
    + (("K", 7), ("N", 8), ("M", 9))
    + tuple((name, index + 1) for name, index in GATHER_PARAMETERS[9:])
)
VECTOR_LIMITS = {True: TRANSPOSED_VECTOR_LIMIT, False: n_gate.VECTOR_LIMIT}
CASES = ("experts", "broadcast")
# Transposed weights: (M, aligned N, unaligned N, nvfp4 K, mx K).
T_SHAPES = {
    # Two row tiles, the second of 8 rows; two column tiles, the second of 8
    # columns when N is unaligned; nvfp4 ends one block in a tail of 16.
    "experts": (40, 64, 40, 48, 64),
    # A row tile of one row; one column tile, of 8 columns when unaligned;
    # nvfp4's tail alone.
    "broadcast": (33, 32, 8, 16, 32),
}
# Untransposed weights: (M, nvfp4 N, mx N, K). nvfp4's N of 16 more than a
# multiple of 32 ends in a tile of 16 columns.
N_SHAPES = {
    # Two row tiles, the second of one row; a block, then 8 rows of K.
    "experts": (33, 48, 64, 40),
    # Two row tiles, the second of 8 rows; two whole blocks of K.
    "broadcast": (40, 48, 64, 64),
}
# Per-expert global scales. inv_scale_enc = g / 2688 is inexact for 3000, so
# bfloat16 and float16 E2M1 weights round. Rounded float32 or E4M3 weights
# would carry more fraction bits than float32 sums hold exactly, so float32
# and mxfp8 entries take 3024 first, whose inv_scale_enc of 1.125 is exact.
EXACT_GLOBAL_SCALES = (3024.0,) + GLOBAL_SCALES[1:]


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for suffix, transpose, aligned in VARIANTS:
                kernel = KERNELS[transpose]
                stem = f"{mode}_{kernel[3:]}_{itype}_gs_{group_size}_b_{bits}{suffix}"
                for hgs in (False, True):
                    yield stem + ("_hgs" if hgs else ""), {
                        "kernel": kernel,
                        "variant": "gather",
                        "transpose": transpose,
                        "aligned": aligned,
                        "mode": mode,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                        "globalScale": hgs,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _sibling(entry):
    """The fp qmm gate and batched entry that share ``entry``'s tile code."""
    info = INFO[entry]
    stem = f"{info['mode']}_qmm_{'t' if info['transpose'] else 'n'}_{info['itype']}"
    stem += f"_gs_{info['groupSize']}_b_{info['bits']}"
    if info["transpose"]:
        return t_gate, f"{stem}_alN_true_batch_1"
    return n_gate, f"{stem}_batch_1"


def _workgroup_access_assertions(entry):
    """The x loader's ``dst`` stays inside the slots BlockLoader writes."""
    gate, sibling = _sibling(entry)
    return tuple(
        {**assertion, "entryPoint": entry}
        for assertion in gate._workgroup_access_assertions(sibling)
    )


def _opengl_index_items(entry):
    """``(expression, scope, maximum, family)`` of each OpenGL assertion.

    The sibling fp qmm gate's families, the lhs and rhs index elements
    ``crosstl_resource_load_uint`` reads and, for an _hgs entry with half or
    bfloat elements, the expert's global scale ``crosstl_resource_load_float``
    reads: GLSL indexes buffers with 32-bit integers, so each index is
    asserted to fit one. A float entry's ``elements`` assertion already
    covers that helper.
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
    if _global_scale_family(entry) == "global-scales":
        items.append(
            (
                "reference.offset + index",
                {"function": "crosstl_resource_load_float"},
                SIGNED_MAX,
                "global-scales",
            )
        )
    return items + tiles


def _global_scale_family(entry):
    """The OpenGL assertion family of an entry's global scale reads.

    The expert's global scale sits at the gathered rhs index, so its read
    index is unproven. ``crosstl_resource_load_float`` reads it, the same
    helper as a float entry's elements; ``None`` when the entry reads none.
    """
    info = INFO[entry]
    if not info["globalScale"]:
        return None
    return "elements" if info["dtype"] == "float32" else "global-scales"


def _config(directory, target, entry):
    """The translation of one entry, with the fp qmm gates' options."""
    return fp_qmm_config(
        directory,
        target,
        entry,
        INFO[entry],
        _workgroup_access_assertions(entry),
        _opengl_index_items(entry),
    )


def _bound_parameters(descriptor, info):
    """The kernel parameters the translation reflects.

    An entry without a global scale never reads ``global_scale``, so its
    translation may drop the binding; every other parameter stays.
    """
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    if info["globalScale"] or "global_scale" in bound:
        return PARAMETERS
    return tuple(item for item in PARAMETERS if item[0] != "global_scale")


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    info = INFO[entry]
    nvfp4 = info["mode"] == "nvfp4"
    if info["transpose"]:
        m, aligned_n, unaligned_n, nvfp4_k, mx_k = T_SHAPES[case]
        n = aligned_n if info["aligned"] else unaligned_n
        return m, n, nvfp4_k if nvfp4 else mx_k
    m, nvfp4_n, mx_n, k = N_SHAPES[case]
    return m, nvfp4_n if nvfp4 else mx_n, k


def _rounds_weights(info):
    """Whether the element type rounds the global-scale weights."""
    return info["dtype"] != "float32" and info["bits"] == 4


def _global_scales(info):
    return GLOBAL_SCALES if _rounds_weights(info) else EXACT_GLOBAL_SCALES


def _direct(info, data):
    """Each output as a dot product of the kernel's dequantized weights.

    Every weight is ``T(f32(f32(T(scale) * inv_scale_enc) * code))`` with the
    expert's ``inv_scale_enc``, as both loaders compute it. The budget bounds
    the float32 arithmetic whatever its order: every product is a multiple
    of ``2^-f`` and the products' magnitudes sum below ``2^(24 - f)``, so
    every partial sum is exact. It is the largest such sum, in units of
    ``2^(24 - f)``. ``rounded`` counts the weights the element type rounds.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    global_scale = data["values"]["global_scale"][1]
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget, rounded = 0.0, 0
    for z, batch in enumerate(layout["batches"]):
        inverse = 1.0
        if info["globalScale"]:
            inverse = _inverse_scale(global_scale[_batch_expert(data["case"], z)])
        for row in range(m):
            vector = x[batch["x"] + row * k : batch["x"] + (row + 1) * k]
            for col in range(n):
                total, bound, digits = 0.0, 0.0, 0
                for index, value in enumerate(vector):
                    # N rows of K codes, or K rows of N codes untransposed.
                    if info["transpose"]:
                        line, position = batch["w"] * n + col, index
                    else:
                        line, position = batch["w"] * k + index, col
                    scale = _scale_value(
                        mode, scales[line * groups + position // group_size]
                    )
                    scale = _f32(_to_type(dtype, scale) * inverse)
                    product = _f32(
                        scale * _code_value(bits, data["codes"][line][position])
                    )
                    weight = _to_type(dtype, product)
                    rounded += weight != product
                    term = value * weight
                    total += term
                    bound += abs(term)
                    digits = max(digits, _fraction_bits(term))
                share = bound / 2.0 ** (24 - digits)
                assert share < 1 and _f32(total) == total, (info, total, bound)
                budget = max(budget, share)
                index = batch["y"] + row * n + col
                totals[index] = total
                expected[index] = _to_type(dtype, total)
    return {"y": expected}, totals, budget, rounded


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """The fp_gather_qmv gate's inputs in a gather layout, bound as gather_qmm's."""
    info = INFO[entry]
    assert case in CASES, (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, case, m, n, k, transposed=not info["transpose"])
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    grid = (-(-n // BN), -(-m // BM), len(layout["batches"]))
    layout["grid"] = grid
    data = case_data_for(info, entry, case, layout, "fp-quantized-gather-qmm", None)
    values = dict(data["values"])
    if info["globalScale"]:
        experts = len(values["global_scale"][1]) - GUARD_COUNT
        scales = list(_global_scales(info)[:experts])
        values["global_scale"] = ("float32", scales + [POISON] * GUARD_COUNT)
    values["K"] = values.pop("in_vec_size")
    values["N"] = values.pop("out_vec_size")
    values["M"] = ("int32", [m])
    assert set(values) == {name for name, _ in PARAMETERS}, entry
    data = {**data, "values": values}
    data["expected"], data["totals"], data["budget"], data["rounded"] = _direct(
        info, data
    )
    return data


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------

GATHER_MUTATIONS = {
    "indices": "uses tid.z in place of the gathered indices",
    "expert-scale": "reads the first expert's global scale for every expert",
}
# fp_qmm_n_model leaves these to fp_qmm_t_model's names: fp_qmm_n passes its
# implementation no global scale, fp_gather_qmm_n the expert's.
N_GLOBAL_SCALE_MUTATIONS = {
    name: t_model.MUTATIONS[name] for name in ("global-scale", "fold-scale")
}


def _mutations(entry):
    if INFO[entry]["transpose"]:
        return {**t_model.MUTATIONS, **GATHER_MUTATIONS}
    return {**n_model.MUTATIONS, **N_GLOBAL_SCALE_MUTATIONS, **GATHER_MUTATIONS}


def _has_tail(entry, case):
    """fp_qmm_t_impl's num_k tail: groups of 16 and K % 32 == 16."""
    info = INFO[entry]
    k = _shape(entry, case)[2]
    return info["transpose"] and info["groupSize"] % BK != 0 and k % BK != 0


def _partial_n(entry, case):
    """fp_qmm_n_impl's partial column tile: groups of 16 and N % 32 == 16."""
    info = INFO[entry]
    n = _shape(entry, case)[1]
    return not info["transpose"] and info["groupSize"] % BN != 0 and n % BN != 0


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "scale-split":
        # Groups of 32 give every thread of a block or tile the same group.
        return info["groupSize"] < BK
    if mutation in ("tail", "tail-zero"):
        return any(_has_tail(entry, case) for case in CASES)
    if mutation in ("global-scale", "expert-scale"):
        return info["globalScale"]
    if mutation == "fold-scale":
        # Scaling each weight or each sum differs only where weights round:
        # an exact inv_scale_enc rounds none.
        return info["globalScale"] and _rounds_weights(info)
    return True


def _simulate(entry, data, mutation=None):
    """The fp_qmm_t or fp_qmm_n model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    model = t_model if INFO[entry]["transpose"] else n_model
    run = model.simulate(INFO[entry], data, families, mutation)
    family = _global_scale_family(entry)
    if family in run["records"]:
        run["records"][family] |= {
            ("global_scale", index) for index in run["reads"]["global_scale"]
        }
    return run


def _expected_paths(entry, case):
    info = INFO[entry]
    m, n, k = _shape(entry, case)
    if info["transpose"]:
        return t_model.expected_paths(info["aligned"], m, n, k, info["groupSize"])
    return n_model.expected_paths(m, n, k, info["groupSize"])


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
    # Every entry precedes its _hgs twin: stepping one place further per pair
    # sends both twins to different shards and each shard both kinds.
    return tuple(
        entry
        for position, entry in enumerate(ENTRIES)
        if (position + position // 2) % count == index
    )


SELECTED_ENTRIES = _selected_entries()


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _record(target, entry):
    return parse_pin(_contract()["artifacts"][entry][target])


def _checked_element_count():
    """Output elements, guards included, every target compares."""
    return sum(
        len(_case_data(entry, case)["expected"]["y"])
        for entry in ENTRIES
        for case in CASES
    )


def _case_shapes(transpose, aligned):
    shapes = {}
    for case in CASES:
        if transpose:
            m, aligned_n, unaligned_n, nvfp4_k, mx_k = T_SHAPES[case]
            n = aligned_n if aligned else unaligned_n
            shapes[case] = {"M": m, "N": n, "K": {"nvfp4": nvfp4_k, "mx": mx_k}}
        else:
            m, nvfp4_n, mx_n, k = N_SHAPES[case]
            shapes[case] = {"M": m, "N": {"nvfp4": nvfp4_n, "mx": mx_n}, "K": k}
        shapes[case]["layout"] = case
    return shapes


def _expected_validation_contract():
    qmm_t = t_gate._expected_validation_contract()
    qmm_n = n_gate._expected_validation_contract()
    return {
        "model": (
            "the gather adjust_matrix_offsets' x, w word, scale, expert and y "
            "offsets, then fp_qmm_t_impl's or fp_qmm_n_impl's tile model with "
            "the expert's global scale folded into each weight, against "
            "direct dot products of the dequantized weights"
        ),
        "packing": qmm_t["packing"],
        "weights": {
            "fp_gather_qmm_t": qmm_t["weights"],
            "fp_gather_qmm_n": qmm_n["weights"],
        },
        "comparison": qmm_t["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), ceil(M / 32), B)",
        "tile": qmm_t["tile"],
        "routing": (
            "GatherQMM with M at or above the vector limit, 33 for transposed "
            "weights and 4 otherwise, and no sorted M == 1 batch for "
            "gather_qmm_rhs; alN_true exactly when N is a multiple of 32; _hgs "
            "with one global scale per expert, which ops.cpp admits for nvfp4 "
            "only, so the mx _hgs entries run although the host never "
            "dispatches them; on devices without NAX kernels"
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
        "partialK": qmm_t["partialK"],
        "partialN": qmm_n["partialN"],
        "inputs": qmm_t["inputs"],
        "globalScales": {
            "perExpert": list(GLOBAL_SCALES),
            "float32AndMxfp8PerExpert": list(EXACT_GLOBAL_SCALES),
        },
        "exactness": (
            "every product is a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f); only the bfloat16 and float16 E2M1 global-scale "
            "weights and static_cast<T> round, to nearest even"
        ),
        "guards": qmm_t["guards"],
        "softwareSubgroups": qmm_t["softwareSubgroups"],
        "cooperativeMatrices": qmm_t["cooperativeMatrices"],
        "globalScaleDivisionProfile": qmm_t["globalScaleDivisionProfile"],
        "workgroupAccessAssertions": qmm_t["workgroupAccessAssertions"],
        "openglIndexRangeAssertions": (
            qmm_t["openglIndexRangeAssertions"]
            + "; lhs and rhs index elements fit them too"
        ),
        "metadataPreconditions": {"K": [0, K_LIMIT]},
        "metalUpstreamComparison": qmm_t["metalUpstreamComparison"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("gatherQmm"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-gather-qmm-contract",
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


def test_current_mlx_fp_quantized_gather_qmm_contract_is_exact():
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


def test_fp_quantized_gather_qmm_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_gather_qmm.current-tree.translation.json" in readme
    assert len(ENTRIES) == 54 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["gatherQmm"]
    assert Counter(
        (info["kernel"], info["aligned"], info["globalScale"]) for info in INFO.values()
    ) == {
        ("fp_gather_qmm_n", True, False): 9,
        ("fp_gather_qmm_n", True, True): 9,
        ("fp_gather_qmm_t", True, False): 9,
        ("fp_gather_qmm_t", True, True): 9,
        ("fp_gather_qmm_t", False, False): 9,
        ("fp_gather_qmm_t", False, True): 9,
    }
    assert ENTRIES[:6] == (
        "nvfp4_gather_qmm_n_float_gs_16_b_4",
        "nvfp4_gather_qmm_n_float_gs_16_b_4_hgs",
        "nvfp4_gather_qmm_t_float_gs_16_b_4_alN_true",
        "nvfp4_gather_qmm_t_float_gs_16_b_4_alN_true_hgs",
        "nvfp4_gather_qmm_t_float_gs_16_b_4_alN_false",
        "nvfp4_gather_qmm_t_float_gs_16_b_4_alN_false_hgs",
    )
    assert ENTRIES[-1] == "mxfp4_gather_qmm_t_float16_t_gs_32_b_4_alN_false_hgs"
    assert not set(KERNELS.values()) & set(EXCLUDED_KERNELS)
    assert set(T_SHAPES) == set(N_SHAPES) == set(CASES)
    for entry in ENTRIES:
        gate, sibling = _sibling(entry)
        assert sibling in gate.INFO, (entry, sibling)


def test_fp_quantized_gather_qmm_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [27, 27]
    for shard in shards:
        # Each shard runs every kernel, mode, element type and global scale.
        for key in ("kernel", "mode", "dtype", "globalScale"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_fp_quantized_gather_qmm_parameters_follow_fp_gather_qmv():
    assert [index for _, index in PARAMETERS] == list(range(21))
    names = [name for name, _ in PARAMETERS]
    assert names[:10] == [
        "w",
        "scales",
        "global_scale",
        "x",
        "lhs_indices",
        "rhs_indices",
        "y",
        "K",
        "N",
        "M",
    ]
    assert names[10:] == [name for name, _ in GATHER_PARAMETERS[9:]]


@pytest.mark.parametrize("entry", ENTRIES[::3])
def test_fp_quantized_gather_qmm_cases_follow_mlx_dispatch(entry):
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
            assert layout["params"]["w_shape"][-2:] == [n, k * info["bits"] // 32]
            assert layout["groups"] == k // info["groupSize"]
        else:
            # quantize requires N to be a multiple of the group size.
            assert n % info["groupSize"] == 0
            assert layout["params"]["w_shape"][-2:] == [k, n * info["bits"] // 32]
            assert layout["groups"] == n // info["groupSize"]
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
        assert params["s_strides"][-2:] == [layout["groups"], 1]
        # One global scale per expert: the batch dimensions of w.
        experts = math.prod(params["w_shape"][: params["w_batch_ndims"]])
        scales = values["global_scale"][1][:-GUARD_COUNT]
        if info["globalScale"]:
            assert scales == list(_global_scales(info)[:experts])
            assert {_batch_expert(case, z) for z in range(4)} <= set(range(experts))
        else:
            assert scales == [POISON]
        # Each batch gathers a weight matrix the buffer holds, and the
        # offsets are not tid.z's: the indices repeat and reorder matrices.
        assert {batch["w"] for batch in batches} <= set(range(layout["matrices"]))
        assert len({batch["w"] for batch in batches}) > 1
        assert [batch["x"] for batch in batches] != sorted(
            {batch["x"] for batch in batches}
        )
        assert 0 <= k <= K_LIMIT


def test_fp_quantized_gather_qmm_cases_reach_every_path():
    t_every = {
        "x-unsafe/w-unsafe",
        "x-safe/w-unsafe",
        "x-unsafe/w-safe",
        "x-safe/w-safe",
        "store-whole",
        "store-safe",
    }
    t_aligned = t_every - {"x-unsafe/w-safe", "x-safe/w-safe"}
    n_every = {
        "rows-whole/k-whole/n-whole",
        "rows-partial/k-whole/n-whole",
        "rows-whole/k-tail/n-whole",
        "rows-partial/k-tail/n-whole",
        "store-whole",
        "store-safe",
        "tail",
    }
    n_partial = {
        path.replace("n-whole", "n-partial") for path in n_every if "/n-" in path
    }
    reached = {}
    for entry in ENTRIES:
        info = INFO[entry]
        nvfp4 = info["mode"] == "nvfp4"
        key = (info["kernel"], info["aligned"], nvfp4)
        for case in CASES:
            reached.setdefault(key, set()).update(_expected_paths(entry, case))
    assert reached == {
        ("fp_gather_qmm_t", True, True): t_aligned | {"tail"},
        ("fp_gather_qmm_t", True, False): t_aligned,
        ("fp_gather_qmm_t", False, True): t_every | {"tail"},
        ("fp_gather_qmm_t", False, False): t_every,
        ("fp_gather_qmm_n", True, True): n_every | n_partial,
        ("fp_gather_qmm_n", True, False): n_every,
    }
    # nvfp4's transposed cases end in a tail alone and after one block.
    assert {T_SHAPES[case][3] // BK for case in CASES} == {0, 1}


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_fp_quantized_gather_qmm_model_matches_dot_products(entry):
    info = INFO[entry]
    transpose = info["transpose"]
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
        # Every gathered activation row, weight word and scale is read in its
        # row, and no guard, unselected matrix or skipped row is.
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
        assert run["reads"]["scales"] == {
            (matrix * rows + row) * groups + group
            for matrix in matrices
            for row in range(rows)
            for group in range(groups)
        }
        experts = {_batch_expert(case, z) for z in range(len(layout["batches"]))}
        assert run["reads"]["global_scale"] == (
            experts if info["globalScale"] else set()
        )
        # The tail's threads past K, or fp_qmm_n's past N, read the next
        # row's first two words and its first scale, or the guards after the
        # last row.
        over = _has_tail(entry, case) or _partial_n(entry, case)
        assert run["overreads"]["w"] == {
            (matrix * rows + row + 1) * words + word
            for matrix in matrices
            for row in range(rows)
            for word in range(2 if over else 0)
        }
        assert run["overreads"]["scales"] == {
            (matrix * rows + row + 1) * groups
            for matrix in matrices
            for row in range(rows)
            if over
        }


def test_fp_quantized_gather_qmm_model_derives_the_loader_assertions():
    for entry in ENTRIES[:6:2]:
        run = _simulate(entry, _case_data(entry, CASES[0]))
        (assertion,) = _workgroup_access_assertions(entry)
        gate, _ = _sibling(entry)
        loader = gate._x_loader(INFO[entry]["itype"])
        slots = {slot for lane in loader.slots() for slot in lane}
        assert {slot for _, slot in run["records"]["Xs"]} == slots
        assert (assertion["minimum"], assertion["maximum"]) == (min(slots), max(slots))
        assert assertion["function"] == loader.function
        assert assertion["entryPoint"] == entry


def test_fp_quantized_gather_qmm_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in _mutations(entry)
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {
        "order": 18,
        "scale-split": 36,
        "tail": 24,
        "tail-zero": 24,
        "global-scale": 27,
        "expert-scale": 27,
        "fold-scale": 42,
    }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_gather_qmm_inputs_keep_float32_exact(dtype):
    rounded = ties = weights = 0
    for entry in ENTRIES:
        info = INFO[entry]
        if info["dtype"] != dtype:
            continue
        for case in CASES:
            data = _case_data(entry, case)
            assert data["budget"] < 1
            # Only the global scale makes a loader round a weight.
            assert (data["rounded"] > 0) == (
                info["globalScale"] and _rounds_weights(info)
            )
            weights += data["rounded"]
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
        assert rounded == weights == 0
    else:
        assert rounded > 0 and ties > 0 and weights > 0, (rounded, ties, weights)


def _mutation_cases():
    return [
        (entry, mutation)
        for entry in SELECTED_ENTRIES
        for mutation in _mutations(entry)
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _mutation_cases())
def test_fp_quantized_gather_qmm_expectations_reject_defects(entry, mutation):
    for case in CASES:
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: {_mutations(entry)[mutation]} goes unnoticed")


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::3])
def test_fp_quantized_gather_qmm_opengl_assertions_name_buffer_elements(entry):
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
                    # The tail and the partial column tile may read into the
                    # guards.
                    limit = lengths[buffer] * (4 if buffer == "w" else 1)
                elif family in ("strides", "indices"):
                    limit = lengths[buffer]
                elif family in ("Xs", "Ws"):
                    limit = maximum + 1
                else:
                    limit = lengths[buffer] - GUARD_COUNT
                assert 0 <= index < limit <= SIGNED_MAX + 1, (case, family, index)
    expected = {
        "x-vector": {"x"},
        "elements": {"x"},
        "y": {"y"},
        "bytes": {"w", "scales"},
        "strides": {
            "x_strides",
            "w_strides",
            "s_strides",
            "lhs_strides",
            "rhs_strides",
        },
        "indices": {"lhs_indices", "rhs_indices"},
        "Xs": {"Xs"},
        "Ws": {"Ws"},
    }
    family = _global_scale_family(entry)
    if family is not None:
        expected.setdefault(family, set()).add("global_scale")
    assert seen == expected


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_gather_qmm_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES[:6]:
        info = INFO[entry]
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            profile = options[backend].get("binary32_division_profile")
            assert profile == ("rne-flush" if info["globalScale"] else None)
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


def _check_instantiation(text):
    """ENTRIES follow fp_quantized.metal's own instantiation macros."""
    types = re.findall(r"^instantiate_quantized_types\((\w+)\)", text, re.MULTILINE)
    assert types == [itype for itype, _ in DTYPES]
    modes = re.findall(
        r"instantiate_quantized_modes\(type, (\w+), (\d+), (\d+)\)",
        _macro(text, "instantiate_quantized_types"),
    )
    assert modes == [(mode, str(size), str(bits)) for mode, size, bits in MODES]
    order = re.findall(
        r"(instantiate_quantized_all_\w+)\(",
        _macro(text, "instantiate_quantized_modes"),
    )
    assert order.index("instantiate_quantized_all_single") < order.index(
        "instantiate_quantized_all_aligned"
    )
    single = re.findall(
        r"instantiate_quantized\(mode, (\w+), type, group_size, bits\)",
        _macro(text, "instantiate_quantized_all_single"),
    )
    assert single[-1] == "gather_qmm_n", single
    aligned = re.findall(
        r"instantiate_quantized_aligned_hgs\(mode, (\w+), type, (true|false), "
        r"group_size, bits\)",
        _macro(text, "instantiate_quantized_all_aligned"),
    )
    assert aligned == [("gather_qmm_t", "true"), ("gather_qmm_t", "false")]
    # Both macros give each entry an _hgs twin with has_global_scale = true.
    for macro in ("instantiate_quantized", "instantiate_quantized_aligned_hgs"):
        body = _macro(text, macro)
        assert '"_hgs",' in body and body.rstrip().endswith("true)"), macro
    assert '#bits "_alN_" #aligned,' in _macro(text, "instantiate_quantized_aligned")


def _check_kernel(header):
    """The kernels' buffers, gathered offsets and implementation calls."""
    calls = {
        # K_eff is K: the whole reduction.
        "fp_gather_qmm_t": (
            "fp_qmm_t_impl<T, group_size, bits, aligned_N, has_global_scale, BM, "
            "BK, BN>(\n      w,\n      scales,\n      global_scale,\n      x,\n"
            "      y,\n      Xs,\n      Ws,\n      K,\n      N,\n      M,\n"
            "      K,\n      tid,"
        ),
        "fp_gather_qmm_n": (
            "fp_qmm_n_impl<T, group_size, bits, has_global_scale, BM, BK, BN>(\n"
            "      w,\n      scales,\n      global_scale,\n      x,\n      y,\n"
            "      Xs,\n      Ws,\n      K,\n      N,\n      M,\n      tid,"
        ),
    }
    for kernel, call in calls.items():
        assert declared_buffers(header, kernel) == PARAMETERS, kernel
        body = header[header.index(f"[[kernel]] void {kernel}(") :]
        body = body[: body.index("\n}\n")]
        assert "      y,\n      M * N,\n      batch_ndims," in body, kernel
        assert body.index("adjust_matrix_offsets<has_global_scale>(") < body.index(
            call
        ), kernel
    # The gather overload, after the batched one fp_qmm_t and fp_qmm_n use.
    offsets = header[
        header.index(
            "template <bool has_global_scale, typename T>\n"
            "METAL_FUNC void adjust_matrix_offsets(\n"
        ) :
    ]
    offsets = offsets[: offsets.index("\n}\n")]
    for required in (
        "const device float*& global_scale,",
        "const device uint32_t* lhs_indices,",
        "x_idx = lhs_indices[tid.z * lhs_strides[0]];",
        "w_idx = rhs_indices[tid.z * rhs_strides[0]];",
        "tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);",
        "x_idx = lhs_indices[idx.x];",
        "w_idx = rhs_indices[idx.y];",
        "x += x_idx * x_strides[0];",
        "x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);",
        "w += w_idx * w_strides[0];",
        "scales += w_idx * s_strides[0];",
        "w_idx, w_shape, w_strides, s_strides, w_batch_ndims);",
        "if constexpr (has_global_scale) {\n    global_scale += w_idx;\n  }",
        "y += tid.z * output_stride;",
    ):
        assert required in offsets, required


def _check_host(host):
    dispatch = host[host.index("void gather_qmm(") :]
    dispatch = dispatch[: dispatch.index("\n}\n")]
    for required in (
        "MTL::Size group_dims(32, wn, wm);",
        "MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
        "bool aligned = N % 32 == 0;",
        'mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),',
        'transpose ? (aligned ? "_alN_true" : "_alN_false") : "",',
        'global_scale ? "_hgs" : "");',
        "  } else if (global_scale) {\n"
        "    compute_encoder.set_input_array(*global_scale, 2);\n  }",
        "compute_encoder.set_input_array(lhs_indices, c++);",
        "compute_encoder.set_bytes(M, c++);",
        "c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);",
        "add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);",
    ):
        assert required in dispatch, required
    route = host[host.index("void GatherQMM::eval_gpu(") :]
    route = route[: route.index("\n}\n")]
    for required in (
        "  } else if (inputs.size() == 6) {\n"
        "    global_scale = ensure_row_contiguous(inputs[3], d, s);",
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


def _check_gather_ops(ops):
    """gather_qmm admits one float32 global scale per expert, nvfp4 only."""
    body = ops[ops.index("array gather_qmm(") :]
    body = body[: body.index("\n}\n")]
    for required in (
        "if (qmode != QuantizationMode::Nvfp4) {",
        "Shape expected(w.shape().begin(), w.shape().end() - 2);",
        "inputs.push_back(*global_scale);",
    ):
        assert required in body, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized gather_qmm", TARGETS, MLX_COMMIT
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**QMM_FP_HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    _check_instantiation(source.decode("utf-8"))
    header = (root / "mlx/backend/metal/kernels/fp_quantized.h").read_text(
        encoding="utf-8"
    )
    # The tile code both implementations share with the fp qmm gates.
    t_gate._check_kernel(header)
    n_gate._check_kernel(header)
    _check_kernel(header)
    _check_host((root / HOST_SOURCE).read_text(encoding="utf-8"))
    ops = (root / "mlx/ops.cpp").read_text(encoding="utf-8")
    _check_ops(ops)
    _check_gather_ops(ops)
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-gather-qmm-metal-reference")
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
        parameters=_bound_parameters(descriptor, INFO[entry]),
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
def test_current_mlx_fp_quantized_gather_qmm_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-fpgqmm-", dir=root) as path:
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


def test_ci_requires_native_fp_quantized_gather_qmm_parity():
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
        "--basetemp=mlx-current-results/${{ matrix.group }}",
        "--junitxml=mlx-current-results/${{ matrix.group }}.xml",
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_gather_qmm.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_gather_qmm.py",
    )
