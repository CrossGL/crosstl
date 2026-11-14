"""Execute the pinned MLX block-masked GEMM entries natively.

``steel_gemm_masked.metal`` instantiates ``block_masked_gemm`` for three
element types, two tile shapes, four transpositions, four alignment pairs and
eight pairings of a boolean, multiplicative or absent output mask with an
operand mask: 768 entries. This gate translates every entry and dispatches it
with the buffers ``BlockMaskedMM::eval_gpu`` binds: ``GEMMParams``, the batch
shape, the batch strides of every input in input order, and the column and
row strides of each mask, without and with one- and two-dimensional batches
in which operands and masks broadcast. Mask layouts are row- and
column-major, with padding, and unread mask words hold values no mask uses.
Every output word, guards included, must equal an exact reference; on Metal
it must also equal the original source. The upstream suite is not covered.
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
from functools import lru_cache
from operator import mul
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    pack_storage_records,
    translate_project,
)
from crosstl.project.runtime_verification import RuntimeExecutionState
from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused
from demos.integrations.mlx.tests.kernels import gemm_masked_model as model
from demos.integrations.mlx.tests.kernels.test_current_gemm import (
    GATE_TOOLS,
    GUARDS,
    HEADER_HASHES,
    MLX_COMMIT,
    ORIGINAL_FLAGS,
    PLATFORM_TARGETS,
    SIGNED_MAX,
    _exact,
    _loader_access_assertions,
    _opengl_index_assertions,
    _operand,
    _storage,
    _write_json,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _runtime_value,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_masked.metal"
SOURCE_SHA256 = "6cd1102cdcff715226f095869069e5b84eebd3b7965bd5eb7f4a0f130d47354f"
MASKED_HEADERS = {
    **HEADER_HASHES,
    "kernels/steel_gemm_masked.h": (
        "f6fffe176321a9c4691825e7ddd5f4df09fab06bb1730699fca09a4bb2ebea2f"
    ),
    # steel/utils.h's elem_to_loc_broadcast and utils.h's elem_to_loc.
    "../utils.h": "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a",
    "../../utils.h": "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf",
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM_MASKED"
SHARD_INDEX_ENV = "CROSTL_MLX_CURRENT_GEMM_MASKED_SHARD_INDEX"
SHARD_COUNT_ENV = "CROSTL_MLX_CURRENT_GEMM_MASKED_SHARD_COUNT"
# Thirteen is coprime with the eight mask pairings, four alignments, four
# transpositions, two shapes and three types, so every shard mixes them all.
CI_SHARD_COUNT = 13
ENTRIES = model.entries()
MASK_PARAMETERS = ("out_mask", "lhs_mask", "rhs_mask")
BINDINGS = (
    "A",
    "B",
    "D",
    "params",
    "batch_shape",
    "batch_strides",
    *MASK_PARAMETERS,
    "mask_strides",
)
# Each case: output batch shape, the batch shape of each input (1 marks a
# broadcast axis), every mask's layout, the operand padding and the extra
# threadgroups that exercise the early return. ``single`` binds no batch, as
# ``eval_gpu`` does for two-dimensional outputs; ``vector`` takes the
# one-dimensional batch path and ``grid`` the ``elem_to_loc`` path.
CASES = {
    "single": {
        "batch": (),
        "inputs": {"A": (), "B": (), "out": (), "lhs": (), "rhs": ()},
        "layouts": {"out": ("row", 1), "lhs": ("row", 2), "rhs": ("col", 1)},
        "padding": 3,
        "extra": 1,
    },
    "vector": {
        "batch": (3,),
        "inputs": {"A": (3,), "B": (1,), "out": (3,), "lhs": (1,), "rhs": (3,)},
        "layouts": {"out": ("col", 0), "lhs": ("row", 0), "rhs": ("row", 2)},
        "padding": 1,
        "extra": 0,
    },
    "grid": {
        "batch": (2, 2),
        "inputs": {
            "A": (2, 2),
            "B": (1, 2),
            "out": (2, 1),
            "lhs": (2, 2),
            "rhs": (1, 2),
        },
        "layouts": {"out": ("col", 1), "lhs": ("col", 0), "rhs": ("row", 1)},
        "padding": 0,
        "extra": 1,
    },
}
# Unread mask words: 3 is no multiplicative mask value; boolean storage can
# only hold true or false.
MASK_POISON = 3
# An absent mask is a one-byte ``_NoMask`` the kernel never reads; the gate
# still binds a small buffer for it.
NOMASK_COUNT = 4
SCOPE = (
    "Every float32, float16 and bfloat16 steel_gemm_masked block_masked_gemm "
    "entry with unbatched, one-dimensional and broadcast two-dimensional "
    "batches; the upstream suite is not covered."
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


def _selected_entries():
    index, count = os.environ.get(SHARD_INDEX_ENV), os.environ.get(SHARD_COUNT_ENV)
    if index is None and count is None:
        return ENTRIES
    if index is None or count is None:
        raise RuntimeError(f"{SHARD_INDEX_ENV} and {SHARD_COUNT_ENV} go together")
    index, count = int(index), int(count)
    if not 0 <= index < count:
        raise RuntimeError(f"masked GEMM shard {index} is outside [0, {count})")
    return ENTRIES[index::count]


SELECTED_ENTRIES = _selected_entries()


def _shape(entry, name):
    """``(M, N, K)`` for a case, as ``eval_gpu`` would select ``entry``.

    MN-unaligned entries get partial tiles; ``vector`` keeps M whole so only N
    is partial. K-unaligned entries get a K tail inside the last mask block.
    """
    block = entry.block
    if name == "single":
        m, n, k = 2 * block - 7, block + 5, 2 * block - 5
        aligned = 2 * block, 2 * block, 2 * block - 16
    elif name == "vector":
        m, n, k = 2 * block, 2 * block - 3, block + 11
        aligned = block, 2 * block, block + 16
    else:
        m, n, k = block - 3, 2 * block - 9, block - 5
        aligned = block, 2 * block, block
    if entry.mn_aligned:
        m, n = aligned[:2]
    if entry.k_aligned:
        k = aligned[2]
    return m, n, k


def _index_assertions(entry):
    """GLSL index ranges: MLX buffers stay below 2^31 elements.

    The device and threadgroup views are the GEMM loaders' (see
    ``_opengl_index_assertions``); masks are read through views of their own
    element type, which is the operand type for multiplicative masks.
    """
    kinds = {entry.mask_kind("out"), entry.mask_kind("op")}
    masks = [
        {
            "source": SOURCE,
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_load_{element}",
            "maximum": SIGNED_MAX,
            "minimum": 0,
        }
        for kind, element in (("bool", "bool"), ("nomask", "_NoMask"))
        if kind in kinds
    ]
    return (*_opengl_index_assertions(entry, SOURCE), *masks)


def _config(root, output, target, entry):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (entry.name,)},
        workgroup_size_rules={SOURCE: entry.workgroup_size},
        # eval_gpu binds the masks as inputs; nothing writes them while the
        # kernel runs, which lets the subgroup proofs read them as uniform.
        immutable_storage=[
            {"source": SOURCE, "entryPoint": entry.name, "parameter": name}
            for name in MASK_PARAMETERS
        ],
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


def _flat(shape, position):
    """Row-major flat index of ``position`` in ``shape``, broadcasting 1s."""
    index = 0
    for extent, coordinate in zip(shape, position):
        index = index * extent + (coordinate if extent > 1 else 0)
    return index


def _positions(shape):
    """Every position of ``shape`` in row-major order."""
    positions = [()]
    for extent in shape:
        positions = [p + (i,) for p in positions for i in range(extent)]
    return positions


def _batch_strides(shape, size):
    """Strides of contiguous ``size``-element matrices over ``shape``; a
    broadcast axis (extent 1 in a batched shape) has stride 0."""
    strides, step = [], size
    for extent in reversed(shape):
        strides.append(step if extent > 1 else 0)
        step *= extent
    return list(reversed(strides))


def _count(shape):
    total = 1
    for extent in shape:
        total *= extent
    return total


def _matrix_layout(rows, columns, layout):
    """``(row stride, column stride, size)`` of a padded mask matrix."""
    order, padding = layout
    if order == "row":
        return columns + padding, 1, rows * (columns + padding)
    return 1, rows + padding, columns * (rows + padding)


def _mask_value(which, kind, instance, row, column):
    """A deterministic mask value; 0 or false masks its block."""
    if which == "out":
        selector = row + 2 * column + instance
        if kind == "bool":
            return selector % 3 != 1
        return (1, 0, 2, -1)[selector % 4]
    if which == "lhs":
        selector = 2 * row + column + instance
        if kind == "bool":
            return selector % 3 != 2
        return (2, 1, 0, -1, 1)[selector % 5]
    selector = row + 3 * column + instance
    if kind == "bool":
        return selector % 4 != 3
    return (1, -1, 2, 0)[selector % 4]


@lru_cache(maxsize=None)
def _partials(dtype, m, n, k, block, a_batch, b_batch):
    """``A @ B`` restricted to each ``block``-wide K block, per output word."""
    a = [
        [_operand(dtype, "A", row * k + inner + a_batch) for inner in range(k)]
        for row in range(m)
    ]
    b = [
        [_operand(dtype, "B", inner * n + column + b_batch) for inner in range(k)]
        for column in range(n)
    ]
    starts = range(0, k, block)
    return tuple(
        tuple(
            tuple(
                sum(
                    map(
                        mul,
                        a[row][start : start + block],
                        b[column][start : start + block],
                    )
                )
                for start in starts
            )
            for column in range(n)
        )
        for row in range(m)
    )


def _masked(entry, partials, m, n, out_mask, lhs_mask, rhs_mask):
    """``model.product`` from per-K-block partial products."""
    block = entry.block
    scale_operands = entry.mask_kind("op") == "scale"
    scale_output = entry.mask_kind("out") == "scale"
    result = {}
    for row in range(m):
        for column in range(n):
            tile_row, tile_column = row // block, column // block
            if out_mask is not None and not out_mask(tile_row, tile_column):
                result[row, column] = 0
                continue
            # The accumulators start at +0, so a negative output scale of an
            # empty sum stores -0.
            total = 0.0
            for k_block, partial in enumerate(partials[row][column]):
                if lhs_mask is not None:
                    lhs = lhs_mask(tile_row, k_block)
                    rhs = rhs_mask(k_block, tile_column)
                    if not (lhs and rhs):
                        continue
                    if scale_operands:
                        partial = partial * lhs * rhs
                total += partial
            if scale_output:
                total *= out_mask(tile_row, tile_column)
            result[row, column] = total
    return result


def _case_data(entry, name):
    """Resources, ``GEMMParams``, the expected output and the dispatch grid."""
    case = CASES[name]
    m, n, k = _shape(entry, name)
    dtype, guard, block = entry.dtype, GUARDS[entry.dtype], entry.block
    batch, shapes, padding = case["batch"], case["inputs"], case["padding"]
    tiles_m, tiles_n, tiles_k = -(-m // block), -(-n // block), -(-k // block)
    a_rows, a_columns = (k, m) if entry.transpose_a else (m, k)
    b_rows, b_columns = (n, k) if entry.transpose_b else (k, n)
    lda, ldb = a_columns + padding, b_columns + padding
    a_size, b_size = a_rows * lda + padding, b_rows * ldb + padding
    a = [guard] * (_count(shapes["A"]) * a_size + 8)
    b = [guard] * (_count(shapes["B"]) * b_size + 8)
    for index in range(_count(shapes["A"])):
        for row in range(m):
            for inner in range(k):
                offset = inner * lda + row if entry.transpose_a else row * lda + inner
                a[index * a_size + offset] = _operand(
                    dtype, "A", row * k + inner + index
                )
    for index in range(_count(shapes["B"])):
        for inner in range(k):
            for column in range(n):
                offset = (
                    column * ldb + inner if entry.transpose_b else inner * ldb + column
                )
                b[index * b_size + offset] = _operand(
                    dtype, "B", inner * n + column + index
                )
    masks = {}
    present = []
    if entry.mask_kind("out") != "nomask":
        present.append(("out", entry.mask_kind("out"), tiles_m, tiles_n))
    if entry.mask_kind("op") != "nomask":
        present.append(("lhs", entry.mask_kind("op"), tiles_m, tiles_k))
        present.append(("rhs", entry.mask_kind("op"), tiles_k, tiles_n))
    mask_strides = []
    for which, kind, rows, columns in present:
        row_stride, column_stride, size = _matrix_layout(
            rows, columns, case["layouts"][which]
        )
        instances = _count(shapes[which])
        storage = [True if kind == "bool" else MASK_POISON] * (instances * size + 2)
        for instance in range(instances):
            for row in range(rows):
                for column in range(columns):
                    storage[
                        instance * size + row * row_stride + column * column_stride
                    ] = _mask_value(which, kind, instance, row, column)
        masks[which] = (kind, storage, row_stride, column_stride, size)
        # eval_gpu pushes the last stride, then the second to last.
        mask_strides += [column_stride, row_stride]
    expected = [guard] * (max(1, _count(batch)) * m * n + 8)
    for z, position in enumerate(_positions(batch)):
        a_index = _flat(shapes["A"], position)
        b_index = _flat(shapes["B"], position)
        partials = _partials(dtype, m, n, k, block, a_index, b_index)

        def reader(which, position=position):
            if which not in masks:
                return None
            _, storage, row_stride, column_stride, size = masks[which]
            base = _flat(shapes[which], position) * size
            return lambda row, column: storage[
                base + row * row_stride + column * column_stride
            ]

        values = _masked(
            entry, partials, m, n, reader("out"), reader("lhs"), reader("rhs")
        )
        for (row, column), value in values.items():
            # The float accumulator, the scales and the output cast round
            # nothing.
            assert _exact(dtype, value), (entry.name, name, value)
            expected[z * m * n + row * n + column] = value
    if batch:
        inputs = ["A", "B", *(which for which, *_ in present)]
        sizes = {"A": a_size, "B": b_size}
        sizes.update({which: masks[which][4] for which in masks})
        strides = {
            which: _batch_strides(shapes[which], sizes[which]) for which in inputs
        }
        batch_strides = [stride for which in inputs for stride in strides[which]]
        batch_shape = list(batch)
        stride_a, stride_b = strides["A"][-1], strides["B"][-1]
    else:
        batch_strides = [0] * (2 + len(present))
        batch_shape, stride_a, stride_b = [1], 0, 0
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=lda,
        ldb=ldb,
        ldd=n,
        tiles_n=tiles_n,
        tiles_m=tiles_m,
        batch_stride_a=stride_a,
        batch_stride_b=stride_b,
        batch_stride_d=m * n,
        swizzle_log=0,
        gemm_k_iterations_aligned=k // entry.gemm.bk,
        batch_ndim=len(batch_shape),
    )
    resources = dict(
        A=("element", a),
        B=("element", b),
        D=("element", [guard] * len(expected)),
        batch_shape=("int32", batch_shape),
        batch_strides=("int64", batch_strides),
        mask_strides=("int32", mask_strides),
    )
    for which, parameter in zip(("out", "lhs", "rhs"), MASK_PARAMETERS):
        if which in masks:
            kind, storage = masks[which][:2]
            resources[parameter] = ("bool_" if kind == "bool" else "element", storage)
        else:
            resources[parameter] = ("int8", [0] * NOMASK_COUNT)
    geometry = {
        "workgroupCount": [
            tiles_n + case["extra"],
            tiles_m + case["extra"],
            max(1, _count(batch)),
        ],
        "workgroupSize": list(entry.workgroup_size),
    }
    return resources, params, expected, geometry


def _dispatch_values(bindings, entry, name, target):
    resources, params, expected, geometry = _case_data(entry, name)
    inputs = {}
    for parameter, binding in bindings.items():
        if parameter == "params":
            layout = binding["scalarLayout"]
            assert layout["elementStrideBytes"] == 72, layout
            words = pack_storage_records(layout, [params])
            value = {"dtype": "uint32", "shape": [len(words)], "values": words}
        else:
            dtype, values = resources[parameter]
            if dtype == "element":
                value = _storage(target, binding, entry.dtype, values)
            else:
                value = _runtime_value(target, dtype, values)
                assert binding["scalarLayout"]["elementType"] == value["dtype"], (
                    parameter,
                    binding["scalarLayout"],
                )
        inputs[binding["name"]] = value
    outputs = {
        bindings["D"]["name"]: _storage(target, bindings["D"], entry.dtype, expected)
    }
    return inputs, outputs, geometry


def test_masked_model_names_every_entry():
    assert len(ENTRIES) == 3 * 2 * 4 * 4 * 8
    assert len(model.BY_NAME) == len(ENTRIES)
    assert Counter(entry.dtype for entry in ENTRIES) == {
        dtype: 256 for dtype, _, _ in model.TYPES
    }
    assert Counter((entry.out_mask, entry.op_mask) for entry in ENTRIES) == {
        masks: 96 for masks in model.MASKS
    }
    assert ("nomask", "nomask") not in model.MASKS


def test_masked_host_names_resolve():
    """``eval_gpu`` spells kernel names the way the model does."""
    entry = model.BY_NAME[
        "steel_gemm_block_outmask_bool__opmask_float16_tn_float16_float16"
        "_bm64_bn64_bk16_wm2_wn2_MN_naligned_K_taligned"
    ]
    assert (entry.transpose_a, entry.transpose_b) == (True, False)
    assert (entry.mn_aligned, entry.k_aligned) == (False, True)
    assert (entry.mask_kind("out"), entry.mask_kind("op")) == ("bool", "scale")
    assert entry.workgroup_size == (32, 2, 2)


def test_masked_shards_partition_the_entries(monkeypatch):
    shards = []
    for index in range(CI_SHARD_COUNT):
        monkeypatch.setenv(SHARD_INDEX_ENV, str(index))
        monkeypatch.setenv(SHARD_COUNT_ENV, str(CI_SHARD_COUNT))
        shards.append(_selected_entries())
    assert sorted(entry.name for shard in shards for entry in shard) == sorted(
        entry.name for entry in ENTRIES
    )
    assert {len(shard) for shard in shards} == {59, 60}
    for shard in shards:
        assert {(e.out_mask, e.op_mask) for e in shard} == set(model.MASKS)
        assert {(e.mn_aligned, e.k_aligned) for e in shard} == {
            (mn, k) for _, mn, _, k in model.ALIGNMENTS
        }
        assert {entry.transpose for entry in shard} == {
            name for name, _, _ in fused.TRANSPOSES
        }
        assert {entry.dtype for entry in shard} == {d for d, _, _ in model.TYPES}
    monkeypatch.setenv(SHARD_INDEX_ENV, str(CI_SHARD_COUNT))
    with pytest.raises(RuntimeError):
        _selected_entries()
    monkeypatch.delenv(SHARD_COUNT_ENV)
    with pytest.raises(RuntimeError):
        _selected_entries()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_masked_shapes_follow_the_host_selection(entry):
    """Each case's shape selects ``entry`` the way ``eval_gpu`` does, and
    spans several output tiles or K mask blocks."""
    gemm = entry.gemm
    tiles = 0
    for name in CASES:
        m, n, k = _shape(entry, name)
        assert (m % gemm.bm == 0 and n % gemm.bn == 0) == entry.mn_aligned
        assert (k % gemm.bk == 0) == entry.k_aligned
        tiles = max(tiles, -(-m // entry.block) * -(-n // entry.block))
        if not entry.k_aligned:
            # The K tail stays in the mask block the kernel reads for it.
            assert (k // gemm.bk * gemm.bk) // entry.block == (k - 1) // entry.block
    assert tiles >= 4


@pytest.mark.parametrize("transpose", [name for name, _, _ in fused.TRANSPOSES])
@pytest.mark.parametrize("masks", model.MASKS, ids="-".join)
def test_masked_partials_reach_the_model_product(masks, transpose):
    """The gate's block-partial reference equals ``model.product``."""
    out_mask, op_mask = masks
    entry = next(
        e
        for e in ENTRIES
        if (e.out_mask, e.op_mask) == masks
        and e.transpose == transpose
        and e.dtype == "float32"
        and e.block == 32
        and not e.mn_aligned
        and not e.k_aligned
    )
    m, n, k = 37, 45, 75
    out = (
        (lambda r, c: _mask_value("out", entry.mask_kind("out"), 0, r, c))
        if (out_mask != "nomask")
        else None
    )
    lhs = rhs = None
    if op_mask != "nomask":
        kind = entry.mask_kind("op")
        lhs = lambda r, c: _mask_value("lhs", kind, 0, r, c)  # noqa: E731
        rhs = lambda r, c: _mask_value("rhs", kind, 0, r, c)  # noqa: E731
    expected = model.product(
        entry,
        m,
        n,
        k,
        lambda row, inner: _operand("float32", "A", row * k + inner),
        lambda inner, column: _operand("float32", "B", inner * n + column),
        out,
        lhs,
        rhs,
    )
    partials = _partials("float32", m, n, k, 32, 0, 0)
    assert _masked(entry, partials, m, n, out, lhs, rhs) == expected


# Exactness depends on the type, the shapes (block and alignments) and the
# masks, not on the transposition.
REFERENCE_ENTRIES = tuple(entry for entry in ENTRIES if entry.transpose == "nn")


@pytest.mark.parametrize("entry", REFERENCE_ENTRIES, ids=lambda entry: entry.name)
def test_masked_references_are_exact_and_take_every_mask_path(entry):
    """Every reference is exact, unwritten words keep their guard, and the
    cases mask output tiles, skip K blocks and scale both."""
    guard = GUARDS[entry.dtype]
    assert _exact(entry.dtype, guard)
    values = {"out": [], "lhs": [], "rhs": []}
    for name, case in CASES.items():
        resources, params, expected, geometry = _case_data(entry, name)
        m, n, k = _shape(entry, name)
        batches = max(1, _count(case["batch"]))
        assert len(expected) == batches * m * n + 8
        assert expected[batches * m * n :] == [guard] * 8
        assert guard not in expected[: batches * m * n]
        assert geometry["workgroupCount"][2] == batches
        assert params["batch_ndim"] == max(1, len(case["batch"]))
        present = (entry.out_mask != "nomask") + 2 * (entry.op_mask != "nomask")
        assert len(resources["batch_strides"][1]) == params["batch_ndim"] * (
            2 + present
        )
        assert len(resources["mask_strides"][1]) == 2 * present
        for parameter in MASK_PARAMETERS:
            kind, storage = resources[parameter]
            if kind == "element":
                assert all(_exact(entry.dtype, value) for value in storage)
        tiles = {
            "out": (-(-m // entry.block), -(-n // entry.block)),
            "lhs": (-(-m // entry.block), -(-k // entry.block)),
            "rhs": (-(-k // entry.block), -(-n // entry.block)),
        }
        for which, (rows, columns) in tiles.items():
            kind = entry.mask_kind("out" if which == "out" else "op")
            values[which] += [
                _mask_value(which, kind, instance, row, column)
                for instance in range(_count(case["inputs"][which]))
                for row in range(rows)
                for column in range(columns)
            ]
    for which, kind in (
        ("out", entry.mask_kind("out")),
        ("lhs", entry.mask_kind("op")),
        ("rhs", entry.mask_kind("op")),
    ):
        if kind == "nomask":
            continue
        assert any(not value for value in values[which]), which
        assert any(value for value in values[which]), which
        scales = {value for value in values[which] if value not in (0, 1)}
        assert bool(scales) == (kind == "scale"), which


def test_masked_layouts_follow_eval_gpu():
    """Batch strides follow the inputs in order, broadcast axes have stride
    0, and each mask pushes its column stride and then its row stride."""
    entry = model.BY_NAME[
        "steel_gemm_block_outmask_float32_opmask_float32_nn_float32_float32"
        "_bm32_bn32_bk16_wm2_wn2_MN_naligned_K_naligned"
    ]
    resources, params, expected, geometry = _case_data(entry, "grid")
    assert (params["M"], params["N"], params["K"]) == (29, 55, 27)
    assert params["batch_ndim"] == 2 and resources["batch_shape"][1] == [2, 2]
    a_size, b_size = 29 * 27, 27 * 55
    # One mask block row; two output tile columns; one K block. The output
    # mask is column-major with one padding word, lhs column-major and rhs
    # row-major with one padding word.
    out_size, lhs_size, rhs_size = 2 * (1 + 1), 1, 1 * (2 + 1)
    assert resources["batch_strides"][1] == [
        2 * a_size,
        a_size,
        0,
        b_size,
        out_size,
        0,
        2 * lhs_size,
        lhs_size,
        0,
        rhs_size,
    ]
    assert resources["mask_strides"][1] == [2, 1, 1, 1, 1, 3]
    # Words 1 and 3 of each output mask matrix are padding nobody reads.
    out_storage = resources["out_mask"][1]
    assert [out_storage[i] for i in (1, 3, 5, 7)] == [MASK_POISON] * 4
    assert params["batch_stride_a"] == a_size and params["batch_stride_b"] == b_size
    assert geometry["workgroupCount"] == [3, 2, 4]
    vector, vector_params, _, _ = _case_data(entry, "vector")
    # The one-dimensional path reads batch_stride_a and batch_stride_b; B and
    # the lhs mask broadcast.
    assert vector_params["batch_stride_b"] == 0
    assert vector["batch_strides"][1][1] == 0 and vector["batch_strides"][1][3] == 0
    single, single_params, _, _ = _case_data(entry, "single")
    assert single["batch_shape"][1] == [1] and single_params["batch_ndim"] == 1
    assert single["batch_strides"][1] == [0] * 5


@pytest.fixture(scope="module")
def masked_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned masked GEMM native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    tmp_path = tmp_path_factory.mktemp("masked-original")
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
    for name, digest in MASKED_HEADERS.items():
        header = root / "mlx/backend/metal/kernels/steel/gemm" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest, name
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


def _original_outputs(request, source, module, work, name):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        control = replace(
            native,
            artifact_path=source,
            module_path=module,
            entry_point=name,
            loaded_artifact=None,
            constants={},
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


@pytest.mark.parametrize("entry", SELECTED_ENTRIES, ids=lambda entry: entry.name)
def test_current_gemm_masked_executes(tmp_path, masked_source, entry):
    root, original, target = masked_source
    results = []
    with tempfile.TemporaryDirectory(prefix=".current-masked-", dir=root) as directory:
        work = Path(directory)
        try:
            report = translate_project(
                _config(root, work.name, target, entry), format_output=False
            )
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert payload["project"]["indexRangeAssertions"] == (
                list(_index_assertions(entry)) if target == "opengl" else []
            )
            descriptor, package = _prepare_native_package(report, tmp_path)
            assert not descriptor.get("specializationConstants")
            bindings = {
                item["provenance"]["sourceResource"]["parameter"]: item
                for item in descriptor["bindings"]
            }
            assert len(bindings) == len(descriptor["bindings"])
            assert set(bindings) == set(BINDINGS), sorted(bindings)
            for name in CASES:
                case_work = tmp_path / f"case-{name}"
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(
                    bindings, entry, name, target
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
                            request, root / SOURCE, original, case_work, entry.name
                        )
                        == record["outputs"]
                    )
                assert record["outputs"] == outputs
                results.append(
                    {
                        "case": name,
                        "shape": list(_shape(entry, name)),
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
            "sourceSha256": SOURCE_SHA256,
            "cases": results,
            "complete": len(results) == len(CASES),
            "scope": SCOPE,
        },
    )
