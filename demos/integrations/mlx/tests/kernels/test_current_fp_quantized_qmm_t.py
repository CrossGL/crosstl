"""Prove fp_quantized.metal's fp_qmm_t entries natively.

``quantized.cpp``'s ``QuantizedMatmul`` multiplies ``M`` activation rows by
transposed weights, ``N`` rows of ``K`` codes quantized along ``K``. With
``M`` at or above the vector limit it sends batches (``B > 1``) to ``qmm``
and single matrices to ``qmm_splitk``, which falls back to ``qmm`` when ``K``
leaves no split; on devices without NAX kernels ``qmm`` dispatches
``fp_qmm_t``. ``DualQuantizedMatmul`` sends bfloat16 activations against
2-D nvfp4 weights with global scales and ``K`` a multiple of 32 to ``qmm``
with the weights' global scale. ``fp_quantized.metal`` instantiates
``fp_qmm_t`` for nvfp4 (groups of 16 E2M1 codes with E4M3 scales), mxfp8
(groups of 32 E4M3 codes with E8M0 scales) and mxfp4 (groups of 32 E2M1
codes with E8M0 scales) in float, bfloat16_t and float16_t, aligned and
unaligned ``N``, with and without batching, and the two unbatched nvfp4
bfloat16_t entries with a global scale (``_hgs``): 38 of its 518 entries.
Each is translated and pinned per target, then runs ``(32, 2, 2)``
threadgroups over a ``(ceil(N / 32), ceil(M / 32), B)`` grid.

``fp_qmm_t_impl`` gives each threadgroup a 32 by 32 output tile. Per block of
32 codes a ``BlockLoader`` copies the activations into ``Xs`` and the fp
``QuantizedBlockLoader`` dequantizes the weights into ``Ws``: each thread
reads one scale for its eight codes, an nvfp4 block spans two groups, and the
scales step by ``32 / group_size`` per block. ``BlockMMA`` multiplies the
tiles in float32. Groups of 16 leave ``K`` a partial block of 16 codes
whenever ``K`` is an odd number of groups; ``fp_qmm_t_impl`` then loads a
``num_k`` tail whose activation columns past ``K`` are zero, while its weight
loader bounds only rows, so the threads past ``K`` dequantize the next row's
first bytes and scale, or the guards after the last row. E2M1 codes and the
inputs' E4M3 scales are finite, so the zeros cancel them. A global scale
folds ``inv_scale_enc`` into each group scale and rounds every dequantized
weight to bfloat16. Inputs are the fp_qmv gate's: every other dequantized
weight is exact, and float32 holds every product and partial sum, so only
those weights and the final ``static_cast<T>`` round. A tile-level model
replays both loaders, the tail and the fragment products and must reach
direct dot products of the dequantized weights, storing every output once.
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

from crosstl.project import ProjectConfig
from demos.integrations.mlx.tests.kernels import fp_qmm_t_model as model
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
    QMV_PARAMETERS,
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
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized import (
    DTYPES,
    HOST_SHA256,
    HOST_SOURCE,
    SOURCE,
    SOURCE_SHA256,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    QMV_FP_HEADERS,
)
from demos.integrations.mlx.tests.kernels.test_current_fp_quantized_qmv import (
    _expected_validation_contract as _qmv_validation_contract,
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
    QMM_HEADERS,
    SIGNED_MAX,
    STEEL,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FP_QUANTIZED_QMM"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_FP_QUANTIZED_QMM_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_FP_QUANTIZED_QMM_SHARD_COUNT"
CI_SHARD_COUNT = 2
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/fp_quantized_qmm_t.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
KERNEL = "fp_qmm_t"
# fp_qmm_t's steel/gemm loader, fragments and transforms.
QMM_FP_HEADERS = {
    **QMV_FP_HEADERS,
    **{path: digest for path, digest in QMM_HEADERS.items() if path.startswith(STEEL)},
}
# fp_qmm_t takes fp_qmv's buffers with K, N and M for its two sizes.
FP_QMM_PARAMETERS = tuple(
    (name, index)
    for index, name in enumerate(
        ("w", "scales", "global_scale", "x", "y", "K", "N", "M")
        + tuple(name for name, _ in QMV_PARAMETERS[7:])
    )
)
# qmm binds w, scales, the global scale's slot, x, y, K, N and M;
# add_strides_and_shapes skips the batch parameters when B <= 1.
UNBATCHED_BOUND = 8
# instantiate_quantized_all_aligned's qmm_t order: aligned N with batching,
# then without, then unaligned N.
VARIANTS = (
    ("alN_true_batch_1", True, True),
    ("alN_true_batch_0", True, False),
    ("alN_false_batch_1", False, True),
    ("alN_false_batch_0", False, False),
)
# instantiate_quantized_qmm_hgs(bfloat16_t): unbatched nvfp4 with a global
# scale, aligned and unaligned N.
HGS_VARIANTS = (("alN_true_batch_0", True), ("alN_false_batch_0", False))
HGS_TYPE = "bfloat16_t"

CASES = {
    "batch_0": ("tiles", "wide", "deep"),
    "batch_1": ("stacked", "broadcast", "shared"),
}
# (M, aligned N, unaligned N).
SHAPES = {
    # Two row tiles, the second of 8 rows; two column tiles, the second of 8
    # columns when N is unaligned.
    "tiles": (40, 64, 40),
    # A row tile of one row, and three column tiles.
    "wide": (33, 96, 72),
    # One column tile, of 8 columns when N is unaligned.
    "deep": (33, 32, 8),
    "stacked": (40, 32, 40),
    "broadcast": (33, 32, 8),
    "shared": (34, 64, 40),
}
# K per case for nvfp4, for the mx modes and with a global scale. nvfp4's
# odd numbers of groups end in a partial block: 16 codes alone, after one
# block or after two. Unbatched mx products take one block, where qmm_splitk
# finds no split; "deep" runs three, which the host reaches only for K = 32p,
# p a prime above 512 / tiles. The global-scale route requires K % 32 == 0.
K_BY_CASE = {
    "tiles": {"nvfp4": 48, "mx": 32, "hgs": 64},
    "wide": {"nvfp4": 16, "mx": 32, "hgs": 32},
    "deep": {"nvfp4": 80, "mx": 96, "hgs": 96},
    "stacked": {"nvfp4": 80, "mx": 64},
    "broadcast": {"nvfp4": 16, "mx": 32},
    "shared": {"nvfp4": 48, "mx": 64},
}
LAYOUTS = {
    "tiles": "unbatched",
    "wide": "unbatched",
    "deep": "unbatched",
    "stacked": "stacked",
    "broadcast": "broadcast",
    "shared": "shared",
}
# The global scales of the _hgs entries: g / 2688 is inexact for 3000, so
# every scale and weight rounds, and 4 for 10752.
CASE_GLOBAL_SCALES = {"tiles": 3000.0, "wide": 10752.0, "deep": 3000.0}
# A host-reachable unbatched mx product of several blocks: N = 2048 gives
# 128 tiles and 512 / 128 = 4 splits, none of which divides K = 160.
DEEP_HOST_SHAPE = (33, 2048, 160)


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""

    def info(itype, dtype, mode, group_size, bits, suffix, aligned, batched, hgs):
        entry = f"{mode}_qmm_t_{itype}_gs_{group_size}_b_{bits}_{suffix}"
        return entry + ("_hgs" if hgs else ""), {
            "kernel": KERNEL,
            "variant": "batch_1" if batched else "batch_0",
            "aligned": aligned,
            "mode": mode,
            "itype": itype,
            "dtype": dtype,
            "groupSize": group_size,
            "bits": bits,
            "globalScale": hgs,
        }

    for itype, dtype in DTYPES:
        for mode, group_size, bits in MODES:
            for suffix, aligned, batched in VARIANTS:
                yield info(
                    itype,
                    dtype,
                    mode,
                    group_size,
                    bits,
                    suffix,
                    aligned,
                    batched,
                    False,
                )
    dtype = dict(DTYPES)[HGS_TYPE]
    for suffix, aligned in HGS_VARIANTS:
        yield info(HGS_TYPE, dtype, "nvfp4", 16, 4, suffix, aligned, False, True)


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


# Element types of the resource helpers OpenGL may index in 32 bits.
HELPER_ELEMENTS = {"float32": "float", "float16": "float16", "bfloat16": "bfloat16"}


def _opengl_index_items(entry):
    """``(expression, scope, maximum, family)`` of each OpenGL assertion.

    ``family`` names what the model shows the expression indexes: activation
    elements the x loader reads whole, elements a resource helper reads or
    writes, weight and scale bytes, int64 batch strides, and threadgroup tile
    slots. GLSL indexes buffers with 32-bit integers, so each buffer index is
    asserted to fit one.
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
        ("load_int64_t", "strides"),
    ):
        scope = {"function": f"crosstl_resource_{access}"}
        items.append(("reference.offset + index", scope, SIGNED_MAX, family))
    last = (BM - 1) * padded(INFO[entry]["itype"]) + BK - 1
    for tile in ("Xs", "Ws"):
        scope = {"indexed": f"crosstl_workgroup_{tile}"}
        items.append(("reference.offset + index", scope, last, tile))
    return items


def _target_options(info):
    """Software subgroups and fragments; a division profile for g / 2688."""
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
            "cooperative_matrix_software_lowering": True,
        },
        "opengl": {
            "software_subgroup_width": SIMD_SIZE,
            "cooperative_matrix_software_lowering": True,
        },
    }
    if info["globalScale"]:
        for target in options.values():
            target["binary32_division_profile"] = "rne-flush"
    return options


def _config(directory, target, entry):
    """The translation of one entry, as the gate and contract run it."""
    info = INFO[entry]
    index_assertions = ()
    if target == "opengl":
        index_assertions = tuple(
            {
                "source": SOURCE,
                "expression": expression,
                **(scope or {}),
                "minimum": 0,
                "maximum": maximum,
            }
            for expression, scope, maximum, _ in _opengl_index_items(entry)
        )
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: WORKGROUP}},
        workgroup_access_assertions=(
            _workgroup_access_assertions(entry) if target != "metal" else ()
        ),
        index_range_assertions=index_assertions,
        metadata_preconditions=[
            dict(
                source=SOURCE,
                entryPoint=entry,
                parameter="K",
                minimum=0,
                maximum=K_LIMIT,
            )
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "target_options": _target_options(info),
            }
        },
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
        return FP_QMM_PARAMETERS
    return tuple(item for item in FP_QMM_PARAMETERS if item[0] != "global_scale")


# ---------------------------------------------------------------------------
# Geometry and cases
# ---------------------------------------------------------------------------


def _column(info):
    if info["globalScale"]:
        return "hgs"
    return "nvfp4" if info["mode"] == "nvfp4" else "mx"


def _shape(entry, case):
    """``(M, N, K)`` of one case."""
    info = INFO[entry]
    m, aligned_n, unaligned_n = SHAPES[case]
    n = aligned_n if info["aligned"] else unaligned_n
    return m, n, K_BY_CASE[case][_column(info)]


def _direct(info, data):
    """Each output as a dot product of the kernel's dequantized weights.

    Every weight is ``T(f32(f32(T(scale) * inv_scale_enc) * code))``, as the
    loader computes it. The budget bounds the float32 arithmetic whatever its
    order: every product is a multiple of ``2^-f`` and the products'
    magnitudes sum below ``2^(24 - f)``, so every partial sum is exact. It is
    the largest such sum, in units of ``2^(24 - f)``. ``rounded`` counts the
    weights the element type rounds.
    """
    dtype, mode, bits = info["dtype"], info["mode"], info["bits"]
    group_size = info["groupSize"]
    layout = data["layout"]
    m, n, k, groups = layout["M"], layout["N"], layout["K"], layout["groups"]
    x = data["values"]["x"][1]
    scales = data["values"]["scales"][1]
    inverse = 1.0
    if info["globalScale"]:
        inverse = _inverse_scale(data["values"]["global_scale"][1][0])
    expected = list(data["initial"]["y"])
    totals = [None] * (len(expected) - GUARD_COUNT)
    budget, rounded = 0.0, 0
    for batch in layout["batches"]:
        for row in range(m):
            vector = x[batch["x"] + row * k : batch["x"] + (row + 1) * k]
            for col in range(n):
                line = batch["w"] * n + col
                total, bound, digits = 0.0, 0.0, 0
                for index, value in enumerate(vector):
                    scale = _scale_value(
                        mode, scales[line * groups + index // group_size]
                    )
                    scale = _f32(_to_type(dtype, scale) * inverse)
                    product = _f32(
                        scale * _code_value(bits, data["codes"][line][index])
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
    """The fp_qmv gate's inputs, bound as fp_qmm_t's parameters."""
    info = INFO[entry]
    assert case in CASES[info["variant"]], (entry, case)
    m, n, k = _shape(entry, case)
    layout = layout_for(info, LAYOUTS[case], m, n, k)
    # fp_quantized.h's kernels have no biases.
    del layout["params"]["b_strides"]
    scale = CASE_GLOBAL_SCALES.get(case)
    data = case_data_for(info, entry, case, layout, "fp-quantized-qmm-t", scale)
    values = dict(data["values"])
    values["K"] = values.pop("in_vec_size")
    values["N"] = values.pop("out_vec_size")
    values["M"] = ("int32", [m])
    assert set(values) == {name for name, _ in FP_QMM_PARAMETERS}, entry
    grid = (-(-n // BN), -(-m // BM), len(layout["batches"]))
    data = {**data, "layout": {**layout, "grid": grid}, "values": values, "grid": grid}
    data["expected"], data["totals"], data["budget"], data["rounded"] = _direct(
        info, data
    )
    return data


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


def _has_tail(entry, case):
    return INFO[entry]["groupSize"] % BK != 0 and _shape(entry, case)[2] % BK != 0


def _mutation_applies(entry, mutation):
    """Whether ``mutation`` is a defect for ``entry`` at all."""
    info = INFO[entry]
    if mutation == "order":
        # One code per byte: a pack has nothing to reverse.
        return info["bits"] != 8
    if mutation == "scale-split":
        # Groups of 32 give every thread of a block the same group.
        return info["groupSize"] < BK
    if mutation == "batch":
        return info["variant"] == "batch_1"
    if mutation in ("tail", "tail-zero"):
        return any(_has_tail(entry, case) for case in CASES[info["variant"]])
    if mutation in ("global-scale", "fold-scale"):
        return info["globalScale"]
    return True


def _simulate(entry, data, mutation=None):
    """fp_qmm_t_model's replay of one entry's dispatch."""
    families = {family for _, _, _, family in _opengl_index_items(entry)}
    return model.simulate(INFO[entry], data, families, mutation)


def _expected_paths(entry, case):
    m, n, k = _shape(entry, case)
    info = INFO[entry]
    return model.expected_paths(info["aligned"], m, n, k, info["groupSize"])


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
            layout = layout_for(INFO[entry], LAYOUTS[case], m, n, k)
            total += len(layout["batches"]) * m * n + GUARD_COUNT
    return total


def _expected_validation_contract():
    qmv = _qmv_validation_contract()
    return {
        "model": (
            "fp_qmm_t_impl tile by tile after adjust_matrix_offsets: the "
            "BlockLoader and fp QuantizedBlockLoader blocks with their safe "
            "loads, one scale per thread's eight codes stepped by "
            "32 / group_size per block, the num_k tail past the whole blocks, "
            "the global scale folded into each weight, BlockMMA's float32 "
            "fragment products and the whole or safe stores, against direct "
            "dot products of the dequantized weights"
        ),
        "packing": qmv["packing"],
        "weights": "N rows of K codes, quantized along K",
        "comparison": qmv["comparison"],
        "workgroupSize": list(WORKGROUP),
        "grid": "(ceil(N / 32), ceil(M / 32), B)",
        "tile": {"BM": BM, "BN": BN, "BK": BK, "WM": WM, "WN": WN},
        "routing": (
            "M at or above get_qmv_batch_limit's largest limit; batch_1 for "
            "B > 1, batch_0 where qmm_splitk finds no split; _hgs from "
            "DualQuantizedMatmul for bfloat16 activations, 2-D nvfp4 weights "
            "and K % 32 == 0; alN_true exactly when N is a multiple of 32; on "
            "devices without NAX kernels"
        ),
        "cases": {
            variant: {
                case: {
                    "M": SHAPES[case][0],
                    "alignedN": SHAPES[case][1],
                    "unalignedN": SHAPES[case][2],
                    "K": dict(K_BY_CASE[case]),
                    "layout": LAYOUTS[case],
                }
                for case in cases
            }
            for variant, cases in CASES.items()
        },
        "deepHostShape": dict(zip(("M", "N", "K"), DEEP_HOST_SHAPE)),
        "partialK": (
            "groups of 16 leave K % 32 == 16 a num_k tail of 16 codes; its "
            "weight loader bounds rows only, so threads past K dequantize the "
            "next row's first bytes and scale, or the guards, times zeroed "
            "activations"
        ),
        "inputs": qmv["inputs"],
        "globalScales": dict(CASE_GLOBAL_SCALES),
        "exactness": (
            "every product is a multiple of 2^-f whose magnitudes sum below "
            "2^(24 - f); only the global-scale weights and static_cast<T> "
            "round, to nearest even"
        ),
        "guards": qmv["guards"],
        "softwareSubgroups": qmv["softwareSubgroups"],
        "cooperativeMatrices": (
            "DirectX and OpenGL lower simdgroup_matrix to software fragments "
            "with BaseMMAFrag::get_coord's tile_4x4_row_pair lane mapping"
        ),
        "globalScaleDivisionProfile": qmv["globalScaleDivisionProfile"],
        "workgroupAccessAssertions": (
            "the activation BlockLoader's dst stays within the slots "
            "load_unsafe writes"
        ),
        "openglIndexRangeAssertions": (
            "buffer element, weight and scale byte and int64 stride indices "
            "fit GLSL's 32-bit indices; tile indices stay within the BM x BK "
            "slots"
        ),
        "metadataPreconditions": {"K": [0, K_LIMIT]},
        "unbatchedUpstreamBuffers": UNBATCHED_BOUND,
        "metalUpstreamComparison": qmv["metalUpstreamComparison"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        **shared_scope("qmmT"),
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
        "kind": "crosstl-mlx-current-tree-fp-quantized-qmm-t-contract",
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


def test_current_mlx_fp_quantized_qmm_t_contract_is_exact():
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


def test_fp_quantized_qmm_t_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fp_quantized_qmm_t.current-tree.translation.json" in readme
    assert len(ENTRIES) == 38 == len(set(ENTRIES)) == GATE_ENTRY_COUNTS["qmmT"]
    assert Counter(
        (info["variant"], info["aligned"], info["globalScale"])
        for info in INFO.values()
    ) == {
        ("batch_1", True, False): 9,
        ("batch_0", True, False): 9,
        ("batch_1", False, False): 9,
        ("batch_0", False, False): 9,
        ("batch_0", True, True): 1,
        ("batch_0", False, True): 1,
    }
    assert ENTRIES[:2] == (
        "nvfp4_qmm_t_float_gs_16_b_4_alN_true_batch_1",
        "nvfp4_qmm_t_float_gs_16_b_4_alN_true_batch_0",
    )
    assert ENTRIES[-3:] == (
        "mxfp4_qmm_t_float16_t_gs_32_b_4_alN_false_batch_0",
        "nvfp4_qmm_t_bfloat16_t_gs_16_b_4_alN_true_batch_0_hgs",
        "nvfp4_qmm_t_bfloat16_t_gs_16_b_4_alN_false_batch_0_hgs",
    )
    assert KERNEL not in EXCLUDED_KERNELS
    assert set(SHAPES) == set(LAYOUTS) == set(K_BY_CASE)
    assert {case for cases in CASES.values() for case in cases} == set(SHAPES)
    assert set(CASE_GLOBAL_SCALES) == set(CASES["batch_0"])


def test_fp_quantized_qmm_t_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry for shard in shards for entry in shard) == sorted(ENTRIES)
    assert [len(shard) for shard in shards] == [19, 19]
    for shard in shards:
        # Each shard runs every mode, element type and alignment.
        for key in ("mode", "dtype", "aligned"):
            counts = Counter(INFO[entry][key] for entry in shard)
            assert set(counts) == {info[key] for info in INFO.values()}, key
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


def test_fp_quantized_qmm_t_geometry_follows_the_fp_block_loader():
    for _, group_size, bits in MODES:
        shape = model.geometry(bits, group_size)
        # Every thread dequantizes eight codes of one group: 128 threads fill
        # 32 x 32.
        assert shape["reads"] * shape["packFactor"] == BK * BN // THREADS == 8
        assert group_size % (shape["reads"] * shape["packFactor"]) == 0
        assert shape["groupSteps"] == 1
        assert shape["scaleStep"] == BK // group_size
        assert shape["partialK"] == (group_size == 16)


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_qmm_t_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
        data = _case_data(entry, case)
        layout = data["layout"]
        m, n, k = layout["M"], layout["N"], layout["K"]
        batches = len(layout["batches"])
        assert (m, n, k) == _shape(entry, case)
        # QuantizedMatmul and DualQuantizedMatmul reach qmm on every device
        # without NAX kernels.
        assert VECTOR_LIMIT <= m and k <= DIMENSION_LIMIT and n <= DIMENSION_LIMIT
        assert k % info["groupSize"] == 0
        assert (n % BN == 0) == info["aligned"]
        assert (batches > 1) == (info["variant"] == "batch_1")
        if info["globalScale"]:
            # DualQuantizedMatmul calls qmm itself, for K % 32 == 0 only.
            assert k % 32 == 0 and info["dtype"] == "bfloat16" and batches == 1
        elif batches == 1 and (case, _column(info)) != ("deep", "mx"):
            # qmm_splitk falls back to qmm.
            assert _split_k(m, n, k, info["groupSize"]) <= 1
        assert data["grid"] == (-(-n // BN), -(-m // BM), batches)
        assert len(data["expected"]["y"]) == batches * m * n + GUARD_COUNT
        values = data["values"]
        assert values["K"][1] == [k] and values["N"][1] == [n] and values["M"][1] == [m]
        params = layout["params"]
        assert params["x_shape"][-2:] == [m, k]
        # (N, K) weights: N rows of K * bits / 32 words and K / group_size groups.
        assert params["w_shape"][-2:] == [n, k * info["bits"] // 32]
        assert layout["words"] == k * info["bits"] // 32
        assert layout["groups"] == k // info["groupSize"]
        assert params["s_strides"][-2:] == [layout["groups"], 1]
        assert 0 <= k <= K_LIMIT
    # "deep"'s blocks stand for a host-reachable unbatched mx product.
    m, n, k = DEEP_HOST_SHAPE
    assert _split_k(m, n, k, 32) == 1 and k // BK > 1
    assert VECTOR_LIMIT <= m and n <= DIMENSION_LIMIT and k <= DIMENSION_LIMIT


def test_fp_quantized_qmm_t_cases_reach_every_path():
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
        info = INFO[entry]
        key = (info["variant"], info["aligned"], _column(info))
        for case in CASES[info["variant"]]:
            reached.setdefault(key, set()).update(_expected_paths(entry, case))
    aligned = every - {"x-unsafe/w-safe", "x-safe/w-safe"}
    expected = {}
    for variant in ("batch_0", "batch_1"):
        for column in ("nvfp4", "mx"):
            for is_aligned in (True, False):
                paths = aligned if is_aligned else every
                if column == "nvfp4":
                    paths = paths | {"tail"}
                expected[variant, is_aligned, column] = paths
    expected["batch_0", True, "hgs"] = aligned
    expected["batch_0", False, "hgs"] = every
    assert reached == expected
    # nvfp4 cases end in a tail alone, after one block and after two.
    nvfp4 = {K_BY_CASE[case]["nvfp4"] // BK for case in K_BY_CASE}
    assert nvfp4 == {0, 1, 2}


@pytest.mark.parametrize("entry", SELECTED_ENTRIES)
def test_fp_quantized_qmm_t_model_matches_dot_products(entry):
    info = INFO[entry]
    for case in CASES[info["variant"]]:
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
        assert run["reads"]["global_scale"] == ({0} if info["globalScale"] else set())
        # The tail's threads past K read the next row's first two words and
        # its first scale, or the guards after the last row.
        tail = _has_tail(entry, case)
        assert run["overreads"]["w"] == {
            (matrix * n + row + 1) * words + word
            for matrix in matrices
            for row in range(n)
            for word in range(2 if tail else 0)
        }
        assert run["overreads"]["scales"] == {
            (matrix * n + row + 1) * groups
            for matrix in matrices
            for row in range(n)
            if tail
        }
        if tail:
            assert (BK - k % BK) * info["bits"] // 8 == 2 * 4


def test_fp_quantized_qmm_t_model_derives_the_loader_assertions():
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


def test_fp_quantized_qmm_t_mutations_apply_where_they_are_defects():
    skipped = Counter(
        mutation
        for entry in ENTRIES
        for mutation in model.MUTATIONS
        if not _mutation_applies(entry, mutation)
    )
    assert skipped == {
        "order": 12,
        "scale-split": 24,
        "batch": 20,
        "tail": 26,
        "tail-zero": 26,
        "global-scale": 36,
        "fold-scale": 36,
    }


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_qmm_t_inputs_keep_float32_exact(dtype):
    rounded = ties = weights = 0
    for entry in ENTRIES:
        info = INFO[entry]
        if info["dtype"] != dtype:
            continue
        for case in CASES[info["variant"]]:
            data = _case_data(entry, case)
            assert data["budget"] < 1
            # Only the global scale makes the loader round a weight.
            assert (data["rounded"] > 0) == (
                info["globalScale"] and CASE_GLOBAL_SCALES[case] == 3000.0
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
        assert rounded > 0 and ties > 0, (rounded, ties)
    assert (weights > 0) == (dtype == "bfloat16")


def _mutation_cases():
    return [
        (entry, mutation)
        for entry in SELECTED_ENTRIES
        for mutation in model.MUTATIONS
        if _mutation_applies(entry, mutation)
    ]


@pytest.mark.parametrize("entry,mutation", _mutation_cases())
def test_fp_quantized_qmm_t_expectations_reject_defects(entry, mutation):
    for case in CASES[INFO[entry]["variant"]]:
        data = _case_data(entry, case)
        if _simulate(entry, data, mutation)["outputs"] != data["expected"]:
            return
    pytest.fail(f"{entry}: {model.MUTATIONS[mutation]} goes unnoticed")


@pytest.mark.parametrize("entry", SELECTED_ENTRIES[::2])
def test_fp_quantized_qmm_t_opengl_assertions_name_buffer_elements(entry):
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
                    # The tail may read into the guards.
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
def test_fp_quantized_qmm_t_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES[:4] + ENTRIES[-2:]:
        info = INFO[entry]
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        for backend in ("directx", "opengl"):
            assert options[backend]["software_subgroup_width"] == SIMD_SIZE
            assert options[backend]["cooperative_matrix_software_lowering"] is True
            profile = options[backend].get("binary32_division_profile")
            assert profile == ("rne-flush" if info["globalScale"] else None)
        assert options["directx"]["relative_wave_shuffle_out_of_range"] == "self"
        (precondition,) = config.metadata_preconditions
        assert (precondition.parameter, precondition.maximum) == ("K", K_LIMIT)
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
    assert "instantiate_quantized_all_aligned(type, mode, group_size, bits)" in (
        _macro(text, "instantiate_quantized_modes")
    )
    found = re.findall(
        r"instantiate_quantized_aligned_batched\(mode, (\w+), type, (true|false), "
        r"(\d), group_size, bits\)",
        _macro(text, "instantiate_quantized_all_aligned"),
    )
    assert found == [
        ("qmm_t", "true", "1"),
        ("qmm_t", "true", "0"),
        ("qmm_t", "false", "1"),
        ("qmm_t", "false", "0"),
    ]
    hgs = re.findall(
        r"instantiate_quantized_aligned_batched_hgs\((\w+), (\w+), type, "
        r"(true|false), (\d), (\d+), (\d+)\)",
        _macro(text, "instantiate_quantized_qmm_hgs"),
    )
    assert hgs == [
        ("nvfp4", "qmm_t", "true", "0", "16", "4"),
        ("nvfp4", "qmm_t", "false", "0", "16", "4"),
    ]
    assert re.findall(
        r"^instantiate_quantized_qmm_hgs\((\w+)\)", text, re.MULTILINE
    ) == [HGS_TYPE]
    for macro, suffix in (
        ("instantiate_quantized_aligned_batched", '"_batch_" #batched,'),
        ("instantiate_quantized_aligned_batched_hgs", '"_batch_" #batched "_hgs",'),
    ):
        body = _macro(text, macro)
        assert (
            '#mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_"' in body
        )
        assert suffix in body, macro


def _check_kernel(header):
    assert declared_buffers(header, KERNEL) == FP_QMM_PARAMETERS
    loader = header[header.index("struct QuantizedBlockLoader {") :]
    loader = loader[: loader.index("\n};\n")]
    for required in (
        "MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();",
        "MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();",
        "MLX_MTL_CONST short group_steps = group_size < BCOLS ? 1 : group_size / BCOLS;",
        "MLX_MTL_CONST short scale_step = group_size < BCOLS ? BCOLS / group_size : 1;",
        "(bj * pack_factor) / group_size) {",
        "inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);",
        "float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc;",
        "if (bi >= src_tile_dim.y) {",
        "scales += scale_step;",
    ):
        assert required in loader, required
    # load_safe bounds the rows of the tile only.
    safe = loader[loader.index("void load_safe(") :]
    safe = safe[: safe.index("void next()")]
    assert "src_tile_dim.x" not in safe
    dequantize = header[header.index("inline void dequantize(uint8_t w,") :]
    dequantize = dequantize[: dequantize.index("\n}\n")]
    for required in (
        "w_local[0] = static_cast<U>(scale * Dequantize<4, float>{}(w));",
        "w_local[1] = static_cast<U>(scale * Dequantize<4, float>{}(w >> 4));",
        "w_local[0] = static_cast<U>(scale * Dequantize<8, float>{}(w));",
    ):
        assert required in dequantize, required
    impl = header[header.index("METAL_FUNC void fp_qmm_t_impl(") :]
    impl = impl[: impl.index("METAL_FUNC", 1)]
    for required in (
        "constexpr int WM = 2;",
        "constexpr int WN = 2;",
        "constexpr int BK_padded = (BK + 16 / sizeof(T));",
        "BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;",
        "mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;",
        "constexpr bool partial_k = group_size % BK != 0;",
        "const int k_blocks = K_eff / BK;",
        "const short num_k = partial_k ? short(K_eff - k_blocks * BK) : short(0);",
        "loader_x.load_safe(short2(BK, num_els));",
        "loader_w.load_safe(short2(BK, num_outs));",
        "if (!aligned_N && num_outs < BN) {",
        "loader_x.load_safe(short2(num_k, num_els));",
        "loader_x.load_safe(short2(num_k, BM));",
        "loader_w.load_safe(short2(num_k, num_outs));",
        "mma_op.store_result_safe(y, N, short2(num_outs, num_els));",
    ):
        assert required in impl, required
    kernel = header[header.index(f"[[kernel]] void {KERNEL}(") :]
    kernel = kernel[: kernel.index("\n}\n")]
    for required in (
        "M * N,",
        "fp_qmm_t_impl<T, group_size, bits, aligned_N, has_global_scale, BM, BK, BN>(",
        "      M,\n      K,\n      tid,",
    ):
        assert required in kernel, required


def _check_host(host):
    qmm = host[host.index("void qmm(\n") :]
    qmm = qmm[: qmm.index("\n}\n")]
    for required in (
        "MTL::Size group_dims(32, wn, wm);",
        "MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);",
        "bool aligned = N % 32 == 0;",
        'transpose ? (aligned ? "_alN_true" : "_alN_false") : "",',
        'batched ? "_batch_1" : "_batch_0",',
        'global_scale ? "_hgs" : "");',
        "  } else if (transpose) {\n    if (global_scale) {\n"
        "      compute_encoder.set_input_array(*global_scale, c);\n    }\n    c++;",
        "compute_encoder.set_bytes(M, c++);",
        "add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);",
    ):
        assert required in qmm, required
    for required in (
        "int k_align = group_size > 32 ? group_size : 32;",
        "split_k = std::min(split_k, K / k_align);",
        "while (split_k > 1 && (K % (split_k * k_align) != 0)) {",
        "int split_k = std::max(1, 512 / current_tgs);",
        "if (transpose_ && B == 1) {",
        "if (D <= 2048 && O <= 2048) {\n      return 33;",
        "x.dtype() == bfloat16 && K % 32 == 0 &&\n"
        "      M >= get_qmv_batch_limit(K, N, d)) {",
    ):
        assert required in host, required


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(
        REQUIRE_ENV, "fp quantized qmm_t", TARGETS, MLX_COMMIT
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
    work = tmp_path_factory.mktemp("fp-quantized-qmm-t-metal-reference")
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
    info = INFO[entry]
    variant = info["variant"]
    data = _case_data(entry, case)
    request, output, observed = _request(
        descriptor,
        package,
        target,
        variant,
        data,
        parameters=_bound_parameters(descriptor, info),
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
            parameters=FP_QMM_PARAMETERS,
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
def test_current_mlx_fp_quantized_qmm_t_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _record(target, entry)
    with tempfile.TemporaryDirectory(prefix=".current-fpqmmt-", dir=root) as path:
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


def test_ci_requires_native_fp_quantized_qmm_t_parity():
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
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmm_t.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized_qmm_t.py",
    )
