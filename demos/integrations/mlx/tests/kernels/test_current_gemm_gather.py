"""Execute the pinned MLX gathered GEMM entries natively.

``steel_gemm_gather.metal`` instantiates ``gather_mm`` for three element
types, five tile shapes and four transpositions, and ``gather_mm_rhs`` for
one shape and two transpositions. This gate translates every ``gather_mm``
entry with three frozen function-constant configurations and dispatches it
with the buffers ``matmul.cpp`` binds: operand batches selected through
``lhs_indices`` and ``rhs_indices``, including repeated, reordered, strided
and broadcast indices over multi-dimensional operand batches. Every output
word, guards included, must equal an exact reference; on Metal it must also
equal the original source. ``gather_mm_rhs`` derives its row tile from
``offsets`` with subgroup scans and returns early on their reduction, which
DirectX and OpenGL can lower only once reductions of values equal in every
subgroup count as workgroup-uniform (#2546), so it is not covered.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    pack_storage_records,
    translate_project,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeConstantBinding,
    RuntimeExecutionState,
    RuntimeSpecializationConstant,
)
from crosstl.translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused
from demos.integrations.mlx.tests.kernels import gemm_gather_model as model
from demos.integrations.mlx.tests.kernels.test_current_gemm import (
    GATE_TOOLS,
    GUARDS,
    HEADER_HASHES,
    HELPER_ELEMENTS,
    MLX_COMMIT,
    ORIGINAL_FLAGS,
    PLATFORM_TARGETS,
    SIGNED_MAX,
    _exact,
    _loader_access_assertions,
    _operand,
    _storage,
    _write_json,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.metal"
SOURCE_SHA256 = "191402a9a6a18b6cfd5e74309c15e3c748dd59f2f609b9c8324ea514b872d7ed"
GATHER_HEADERS = {
    **HEADER_HASHES,
    "kernels/steel_gemm_gather.h": (
        "317e226d116fd64752fc44527c8a846c9947e5d89ade2abf2a3bdb70e1de7907"
    ),
    # steel/utils.h's elem_to_loc_broadcast and utils.h's elem_to_loc.
    "../utils.h": "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a",
    "../../utils.h": "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM_GATHER"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_GEMM_GATHER_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_GEMM_GATHER_SHARD_COUNT"
# Seven is coprime with the three variants, three types, four
# transpositions and five shapes, so every shard mixes all of them.
CI_SHARD_COUNT = 7
CONSTANT_IDS = {"has_batch": 10, "align_M": 200, "align_N": 201, "align_K": 202}
ENTRIES = tuple(entry for entry in model.entries() if not entry.rhs)
VARIANTS = ("plain", "batched", "aligned")
# (M, N, K, layout, padding, extraWorkgroups). ``single`` dispatches one
# product (no output batch), ``vector`` a one-dimensional index batch with a
# strided lhs index buffer, and ``grid`` a two-dimensional index batch with
# rhs indices broadcast along its first axis. Padding widens every leading
# dimension and batch stride; extra workgroups exercise the early return.
CASES = {
    "plain": ((4, 5, 7, "single", 0, 0), (33, 35, 17, "vector", 3, 1)),
    "batched": ((70, 69, 41, "grid", 1, 1),),
}
# Operand batch shapes. A uses two batch dimensions so ``elem_to_loc``
# decomposes the gathered index; B uses one. Each batch shifts the operand
# pattern by its flat index, and the shifts stay below every pattern period,
# so distinct batches hold distinct matrices.
A_BATCH = (2, 2)
B_BATCH = (3,)
INDICES = {
    "single": {"shape": (), "lhs": (3,), "rhs": (1,)},
    # lhs_indices is read with stride 2, so its odd words are never read.
    "vector": {"shape": (3,), "lhs": (3, 0, 3), "rhs": (2, 2, 0)},
    # rhs_indices broadcasts along the first index axis.
    "grid": {"shape": (2, 2), "lhs": (2, 0, 1, 3), "rhs": (2, 1)},
}
PAIR_COUNTS = {"single": 1, "vector": 3, "grid": 4}
UNREAD_INDEX = 0x7FFFFFF0
SCOPE = (
    "Every float32, float16 and bfloat16 steel_gemm_gather gather_mm entry "
    "with three frozen constant configurations; gather_mm_rhs and the "
    "upstream suite are not covered."
)
TARGET_OPTIONS = {
    "directx": {
        "cooperative_matrix_software_lowering": True,
        "software_subgroup_width": 32,
        "relative_wave_shuffle_out_of_range": "self",
    },
    "opengl": {
        "cooperative_matrix_software_lowering": True,
        "software_subgroup_width": 32,
    },
}
BINDINGS = (
    "A",
    "B",
    "lhs_indices",
    "rhs_indices",
    "C",
    "params",
    "indices_shape",
    "lhs_strides",
    "rhs_strides",
    "batch_ndim_a",
    "batch_shape_a",
    "batch_strides_a",
    "batch_ndim_b",
    "batch_shape_b",
    "batch_strides_b",
)


def _frozen_values(variant):
    assert variant in VARIANTS
    aligned = variant == "aligned"
    return {
        "has_batch": variant == "batched",
        "align_M": aligned,
        "align_N": aligned,
        "align_K": aligned,
    }


def _cases(entry, variant):
    if variant == "aligned":
        # Whole tiles: one tile, then 64x64x32, which every tile divides.
        return (
            (entry.bm, entry.bn, entry.bk, "single", 0, 0),
            (64, 64, 32, "vector", 3, 1),
        )
    return CASES[variant]


PAIRS = tuple((entry, variant) for entry in ENTRIES for variant in VARIANTS)


def _selected_pairs():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return PAIRS
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"gather GEMM shard {index} is outside [0, {count})")
    return PAIRS[index::count]


SELECTED_PAIRS = _selected_pairs()


def _pair_id(pair):
    entry, variant = pair
    return f"{entry.name}-{variant}"


def _index_assertions(entry):
    """GLSL index ranges: MLX buffers stay below 2^31 elements.

    A loader reads ``reads`` adjacent elements past its source offset; the
    index buffers are read as ``uint`` and the batch strides as ``int64_t``.
    The threadgroup views reach only the slots their loaders write.
    """
    element = HELPER_ELEMENTS[entry.dtype]
    reads = max(loader.reads for loader in entry.loaders)
    device = [
        {"expression": f"source.offset + {offset}", "maximum": SIGNED_MAX}
        for offset in range(reads)
    ] + [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}",
            "maximum": SIGNED_MAX,
        }
        for access in sorted(
            {f"load_{element}", f"store_{element}", "load_int64_t", "load_uint"}
        )
    ]
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{tile}",
            "maximum": loader["maximum"],
        }
        for tile, loader in zip(("As", "Bs"), _loader_access_assertions(entry, SOURCE))
    ]
    return tuple(
        {"source": SOURCE, **item, "minimum": 0} for item in device + workgroup
    )


def _config(root, output, target, entry, variant):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (entry.name,)},
        specialization_constants=_frozen_values(variant),
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: entry.workgroup_size},
        workgroup_access_assertions=(
            _loader_access_assertions(entry, SOURCE) if target != "metal" else ()
        ),
        index_range_assertions=(_index_assertions(entry) if target == "opengl" else ()),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                "target_options": TARGET_OPTIONS,
            }
        },
    )


def _contiguous_strides(shape, inner, padding):
    """Row-major strides over ``shape`` whose innermost step is ``inner``,
    each widened by ``padding`` so batches are not adjacent."""
    strides = []
    step = inner
    for extent in reversed(shape):
        strides.append(step)
        step = step * extent + padding
    return list(reversed(strides)), step


def _elem_to_loc(elem, shape, strides):
    """``utils.h``'s ``elem_to_loc`` over the leading ``len(shape)`` axes."""
    loc = 0
    for extent, stride in zip(reversed(shape), reversed(strides[: len(shape)])):
        if elem <= 0:
            break
        loc += (elem % extent) * stride
        elem //= extent
    return loc


def _operand_layout(entry, m, n, k, padding):
    a_rows, a_columns = (k, m) if entry.transpose_a else (m, k)
    b_rows, b_columns = (n, k) if entry.transpose_b else (k, n)
    lda, ldb = a_columns + padding, b_columns + padding
    a_strides, a_size = _contiguous_strides(A_BATCH, a_rows * lda + padding, padding)
    b_strides, b_size = _contiguous_strides(B_BATCH, b_rows * ldb + padding, padding)
    return {
        "lda": lda,
        "ldb": ldb,
        "a_shape": [*A_BATCH, a_rows, a_columns],
        "a_strides": [*a_strides, lda, 1],
        "a_size": a_size,
        "b_shape": [*B_BATCH, b_rows, b_columns],
        "b_strides": [*b_strides, ldb, 1],
        "b_size": b_size,
    }


def _a_index(entry, layout, base, row, inner):
    if entry.transpose_a:
        return base + inner * layout["lda"] + row
    return base + row * layout["lda"] + inner


def _b_index(entry, layout, base, inner, column):
    if entry.transpose_b:
        return base + column * layout["ldb"] + inner
    return base + inner * layout["ldb"] + column


def _index_buffers(name):
    """The index buffers and their shape and strides as ``matmul.cpp`` binds
    them, and each output batch's gathered ``(A batch, B batch)`` pair.

    ``batch_stride_a`` and ``batch_stride_b`` are the index arrays' leading
    strides; a dispatch without output batches binds a placeholder shape.
    """
    spec = INDICES[name]
    shape, lhs, rhs = spec["shape"], spec["lhs"], spec["rhs"]
    if name == "single":
        pairs = [(lhs[0], rhs[0])]
        return dict(
            lhs=[*lhs],
            rhs=[*rhs],
            shape=[1],
            lhs_strides=[0],
            rhs_strides=[0],
            batch_ndim=0,
            pairs=pairs,
        )
    if name == "vector":
        lhs_buffer = []
        for value in lhs:
            lhs_buffer += [value, UNREAD_INDEX]
        pairs = [(lhs[z], rhs[z]) for z in range(shape[0])]
        return dict(
            lhs=lhs_buffer,
            rhs=[*rhs],
            shape=[*shape],
            lhs_strides=[2],
            rhs_strides=[1],
            batch_ndim=1,
            pairs=pairs,
        )
    lhs_strides, rhs_strides = [shape[1], 1], [0, 1]
    pairs = []
    for z in range(shape[0] * shape[1]):
        position = (z // shape[1], z % shape[1])
        loc_a = sum(p * s for p, s in zip(position, lhs_strides))
        loc_b = sum(p * s for p, s in zip(position, rhs_strides))
        pairs.append((lhs[loc_a], rhs[loc_b]))
    return dict(
        lhs=[*lhs],
        rhs=[*rhs],
        shape=[*shape],
        lhs_strides=lhs_strides,
        rhs_strides=rhs_strides,
        batch_ndim=2,
        pairs=pairs,
    )


def _case_data(entry, case):
    """Resources, the expected output and the dispatch geometry for ``case``."""
    m, n, k, indices, padding, extra = case
    dtype, guard = entry.dtype, GUARDS[entry.dtype]
    layout = _operand_layout(entry, m, n, k, padding)
    a = [guard] * (layout["a_size"] + 8)
    b = [guard] * (layout["b_size"] + 8)
    a_bases = [
        _elem_to_loc(index, A_BATCH, layout["a_strides"])
        for index in range(A_BATCH[0] * A_BATCH[1])
    ]
    b_bases = [
        _elem_to_loc(index, B_BATCH, layout["b_strides"]) for index in range(B_BATCH[0])
    ]
    for batch, base in enumerate(a_bases):
        for row in range(m):
            for inner in range(k):
                a[_a_index(entry, layout, base, row, inner)] = _operand(
                    dtype, "A", row * k + inner + batch
                )
    for batch, base in enumerate(b_bases):
        for inner in range(k):
            for column in range(n):
                b[_b_index(entry, layout, base, inner, column)] = _operand(
                    dtype, "B", inner * n + column + batch
                )
    gathered = _index_buffers(indices)
    pairs = gathered["pairs"]
    ldd = n + padding
    batch_stride_d = m * ldd + padding
    expected = [guard] * (len(pairs) * batch_stride_d + 8)
    for z, (index_a, index_b) in enumerate(pairs):
        for row in range(m):
            for column in range(n):
                total = sum(
                    a[_a_index(entry, layout, a_bases[index_a], row, inner)]
                    * b[_b_index(entry, layout, b_bases[index_b], inner, column)]
                    for inner in range(k)
                )
                # The float accumulator and the output cast round nothing.
                assert _exact(dtype, total), (entry.name, case, total)
                expected[z * batch_stride_d + row * ldd + column] = total
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=layout["lda"],
        ldb=layout["ldb"],
        ldd=ldd,
        tiles_n=-(-n // entry.bn),
        tiles_m=-(-m // entry.bm),
        batch_stride_a=gathered["lhs_strides"][0],
        batch_stride_b=gathered["rhs_strides"][0],
        batch_stride_d=batch_stride_d,
        swizzle_log=0,
        gemm_k_iterations_aligned=k // entry.bk,
        batch_ndim=gathered["batch_ndim"],
    )
    resources = dict(
        A=("element", a),
        B=("element", b),
        lhs_indices=("uint32", gathered["lhs"]),
        rhs_indices=("uint32", gathered["rhs"]),
        C=("element", [guard] * len(expected)),
        indices_shape=("int32", gathered["shape"]),
        lhs_strides=("int64", gathered["lhs_strides"]),
        rhs_strides=("int64", gathered["rhs_strides"]),
        batch_ndim_a=("int32", [len(A_BATCH)]),
        batch_shape_a=("int32", layout["a_shape"]),
        batch_strides_a=("int64", layout["a_strides"]),
        batch_ndim_b=("int32", [len(B_BATCH)]),
        batch_shape_b=("int32", layout["b_shape"]),
        batch_strides_b=("int64", layout["b_strides"]),
    )
    geometry = {
        "workgroupCount": [
            params["tiles_n"] + extra,
            params["tiles_m"] + extra,
            len(pairs),
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    return resources, params, expected, geometry


def _dispatch_values(bindings, entry, case, target):
    resources, params, expected, geometry = _case_data(entry, case)
    inputs = {}
    for name, binding in bindings.items():
        if name == "params":
            words = pack_storage_records(binding["scalarLayout"], [params])
            value = {"dtype": "uint32", "shape": [len(words)], "values": words}
        else:
            dtype, values = resources[name]
            if dtype == "element":
                value = _storage(target, binding, entry.dtype, values)
            else:
                value = {"dtype": dtype, "shape": [len(values)], "values": values}
        inputs[binding["name"]] = value
    outputs = {
        bindings["C"]["name"]: _storage(target, bindings["C"], entry.dtype, expected)
    }
    return inputs, outputs, geometry


def test_gather_model_names_every_entry():
    entries = model.entries()
    assert len(entries) == 3 * (2 + 5 * 4)
    assert len(model.BY_NAME) == len(entries)
    assert len(ENTRIES) == 60
    assert Counter(entry.dtype for entry in ENTRIES) == {
        dtype: 20 for dtype, _, _ in model.TYPES
    }
    assert Counter(entry.transpose for entry in ENTRIES) == {
        name: 15 for name, _, _ in fused.TRANSPOSES
    }
    assert {entry.name for entry in entries if entry.rhs} == {
        f"steel_gather_mm_rhs_{t}_{d}_{d}_bm16_bn64_bk16_wm1_wn2"
        for t in ("nn", "nt")
        for d, _, _ in model.TYPES
    }


def test_gather_host_names_resolve():
    """``matmul.cpp`` spells ``steel_gather_mm_`` names the way the model does."""
    entry = model.BY_NAME["steel_gather_mm_tn_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2"]
    assert (entry.transpose_a, entry.transpose_b) == (True, False)
    assert entry.workgroup_size == (32, 2, 2)


def test_gather_shards_partition_the_pairs(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_pairs())
    assert sorted(map(_pair_id, (p for shard in shards for p in shard))) == sorted(
        map(_pair_id, PAIRS)
    )
    assert {len(shard) for shard in shards} == {25, 26}
    for shard in shards:
        assert {variant for _, variant in shard} == set(VARIANTS)
        assert {entry.dtype for entry, _ in shard} == {d for d, _, _ in model.TYPES}
        assert {entry.transpose for entry, _ in shard} == {
            name for name, _, _ in fused.TRANSPOSES
        }
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_pairs()
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_pairs()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_gather_tile_reads_cover_exactly_the_loader_slots(entry):
    a_slots, b_slots = entry.tile_reads()
    loader_a, loader_b = entry.loaders
    for slots, loader in ((a_slots, loader_a), (b_slots, loader_b)):
        written = {slot for lane in loader.slots() for slot in lane}
        assert slots == written


def test_gather_index_buffers_follow_the_kernel():
    """Each layout gathers distinct, repeated and reordered operand batches."""
    for name, spec in INDICES.items():
        gathered = _index_buffers(name)
        pairs = gathered["pairs"]
        assert len(pairs) == PAIR_COUNTS[name]
        assert gathered["batch_ndim"] == len(spec["shape"])
        assert all(0 <= a < A_BATCH[0] * A_BATCH[1] for a, _ in pairs)
        assert all(0 <= b < B_BATCH[0] for _, b in pairs)
        assert len(set(pairs)) == len(pairs)
    vector = _index_buffers("vector")
    # Without has_batch the kernel reads lhs_indices[batch_stride_a * z].
    stride_a, stride_b = vector["lhs_strides"][0], vector["rhs_strides"][0]
    assert [vector["lhs"][stride_a * z] for z in range(3)] == [3, 0, 3]
    assert [vector["rhs"][stride_b * z] for z in range(3)] == [2, 2, 0]
    assert vector["lhs"][1::2] == [UNREAD_INDEX] * 3
    # With has_batch, elem_to_loc_broadcast walks the 2x2 index grid.
    assert _index_buffers("grid")["pairs"] == [(2, 2), (0, 1), (1, 2), (3, 1)]
    # Batch 3 of A decomposes over both batch axes.
    layout = _operand_layout(ENTRIES[0], 4, 5, 7, 1)
    assert _elem_to_loc(3, A_BATCH, layout["a_strides"]) == (
        layout["a_strides"][0] + layout["a_strides"][1]
    )


@pytest.mark.parametrize("transpose", [name for name, _, _ in fused.TRANSPOSES])
@pytest.mark.parametrize("dtype", [name for name, _, _ in model.TYPES])
def test_gather_references_are_exact_and_guarded(dtype, transpose):
    """Every reference value is exact, every unwritten slot keeps its guard and
    gathered batches give distinct products."""
    entry = model.BY_NAME[
        f"steel_gather_mm_{transpose}_{dtype}_{dtype}_bm32_bn64_bk16_wm1_wn2"
    ]
    guard = GUARDS[dtype]
    assert _exact(dtype, guard)
    for variant in VARIANTS:
        for case in _cases(entry, variant):
            m, n, k, indices, padding, extra = case
            resources, params, expected, geometry = _case_data(entry, case)
            batches = geometry["workgroupCount"][2]
            assert geometry["workgroupSize"] == [32, 2, 1]
            assert geometry["workgroupCount"][:2] == [
                -(-n // 64) + extra,
                -(-m // 32) + extra,
            ]
            live = {
                z * params["batch_stride_d"] + row * params["ldd"] + column
                for z in range(batches)
                for row in range(m)
                for column in range(n)
            }
            assert all(
                (expected[index] == guard) == (index not in live)
                for index in range(len(expected))
            ), (variant, case)
            for name in ("A", "B"):
                assert all(_exact(dtype, value) for value in resources[name][1])
            products = [
                tuple(
                    expected[z * params["batch_stride_d"] + row * params["ldd"] + col]
                    for row in range(m)
                    for col in range(n)
                )
                for z in range(batches)
            ]
            pairs = _index_buffers(indices)["pairs"]
            # Equal gathered pairs give equal products; distinct pairs differ.
            for left in range(batches):
                for right in range(batches):
                    if m * n > 1:
                        assert (products[left] == products[right]) == (
                            pairs[left] == pairs[right]
                        ), (variant, case, left, right)


def test_gather_transposed_layouts_hold_the_same_matrix():
    case = (33, 35, 17, "vector", 3, 1)
    products = {}
    for name, _, _ in fused.TRANSPOSES:
        entry = model.BY_NAME[
            f"steel_gather_mm_{name}_float32_float32_bm32_bn32_bk16_wm2_wn2"
        ]
        _, params, expected, _ = _case_data(entry, case)
        products[name] = [
            expected[z * params["batch_stride_d"] + row * params["ldd"] + col]
            for z in range(3)
            for row in range(33)
            for col in range(35)
        ]
    assert len({tuple(values) for values in products.values()}) == 1
    assert len(set(products["nn"])) > 16


@pytest.fixture(scope="module")
def gather_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned gather GEMM native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("gather-original")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    git = ["git", "-C", str(root)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, timeout=30)
    assert head.strip() == MLX_COMMIT
    status = subprocess.check_output(
        [
            *git,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    )
    assert not status.strip()
    source = root / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    for name, digest in GATHER_HEADERS.items():
        header = root / "mlx/backend/metal/kernels/steel/gemm" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest, name
    register_default_sources()
    discovery = SOURCE_REGISTRY.get("metal").discover_entry_points(
        source.read_text(encoding="utf-8"),
        file_path=str(source),
        include_paths=[str(root)],
    )
    assert discovery.status == ENTRY_DISCOVERY_AVAILABLE
    assert [entry.name for entry in discovery.entries] == [
        entry.name for entry in model.entries()
    ]
    original = None
    if target == "metal":
        run(
            ["xcrun", "--sdk", "macosx", "metal", "--version"],
            tmp_path,
            "metal-version",
        )
        original = compile_metal(
            source, root, tmp_path / "original", flags=ORIGINAL_FLAGS
        )
    else:
        tool = GATE_TOOLS[target][0]
        run([tool, "--version"], tmp_path, f"{tool}-version")
    return root, original, target


def _original_outputs(request, source, module, work, specializations, name):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        constants = {
            constant: NativeRuntimeConstantBinding(
                name=constant,
                constant=RuntimeSpecializationConstant(
                    name=constant,
                    constant_id=identifier,
                    dtype="bool",
                    value=specializations[constant],
                    kind="function-constant",
                ),
                value=specializations[constant],
                source="original-source-specialization",
            )
            for constant, identifier in CONSTANT_IDS.items()
        }
        control = replace(
            native,
            artifact_path=source,
            module_path=module,
            entry_point=name,
            loaded_artifact=None,
            constants=constants,
        )
        actual = executor.runtime_adapter.runtime.dispatch(None, state, control)
        module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
        assert state.details["metalRuntime"]["librarySHA256"] == module_hash
        _write_json(
            work / "original-evidence.json",
            {
                "outputs": actual,
                "sourceCommit": MLX_COMMIT,
                "sourceSha256": SOURCE_SHA256,
                "moduleSha256": module_hash,
                "compileFlags": ORIGINAL_FLAGS,
            },
        )
        return actual
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


@pytest.mark.parametrize("pair", SELECTED_PAIRS, ids=_pair_id)
def test_current_gemm_gather_executes(tmp_path, gather_source, pair):
    entry, variant = pair
    root, original, target = gather_source
    specializations = _frozen_values(variant)
    results = []
    with tempfile.TemporaryDirectory(prefix=".current-gather-", dir=root) as directory:
        work = Path(directory)
        try:
            report = translate_project(
                _config(root, work.name, target, entry, variant), format_output=False
            )
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert payload["project"]["indexRangeAssertions"] == (
                list(_index_assertions(entry)) if target == "opengl" else []
            )
            descriptor, package = _prepare_native_package(report, tmp_path)
            constants = descriptor["specializationConstants"]
            assert {item["name"]: item["id"] for item in constants} == CONSTANT_IDS
            assert all(
                item["frozen"] is True
                and item["value"] is specializations[item["name"]]
                for item in constants
            )
            bindings = {
                item["provenance"]["sourceResource"]["parameter"]: item
                for item in descriptor["bindings"]
            }
            assert len(bindings) == len(descriptor["bindings"])
            assert set(bindings) == set(BINDINGS), sorted(bindings)
            assert bindings["params"]["scalarLayout"]["elementStrideBytes"] == 72
            for case in _cases(entry, variant):
                case_work = tmp_path / ("case-" + "-".join(map(str, case)))
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(
                    bindings, entry, case, target
                )
                _write_json(
                    case_work / "inputs.json",
                    {"inputs": inputs, "outputs": outputs, "geometry": geometry},
                )
                request = build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    inputs,
                    outputs,
                    geometry,
                    expected_target=target,
                )
                assert not request.execution_plan.diagnostics
                assert not request.adapter_contract.specialization_constants
                _execute(
                    request,
                    outputs,
                    case_work,
                    validate=(
                        (
                            lambda path, destination, target: compile_metal(
                                path,
                                root,
                                destination,
                                flags=("-std=metal3.1", "-fno-fast-math"),
                            )
                        )
                        if target == "metal"
                        else None
                    ),
                )
                generated = json.loads((case_work / "evidence.json").read_text())
                record = generated["records"]["generated"]
                if target == "metal":
                    assert (
                        _original_outputs(
                            request,
                            root / SOURCE,
                            original,
                            case_work,
                            specializations,
                            entry.name,
                        )
                        == record["outputs"]
                    )
                assert not record["request"]["constants"]
                assert record["outputs"] == outputs
                results.append(
                    {
                        "shape": list(case[:3]),
                        "indices": case[3],
                        "padding": case[4],
                        "extraWorkgroups": case[5],
                        "equalOriginalMetal": True if target == "metal" else None,
                        "equalExactReference": True,
                    }
                )
        finally:
            shutil.copytree(work, tmp_path / "translation", dirs_exist_ok=True)
    _write_json(
        tmp_path / "parity.json",
        {
            "sourceCommit": MLX_COMMIT,
            "target": target,
            "entryPoint": entry.name,
            "variant": variant,
            "specializations": specializations,
            "sourceSha256": SOURCE_SHA256,
            "cases": results,
            "complete": len(results) == len(_cases(entry, variant)),
            "scope": SCOPE,
        },
    )
