"""Native parity of the 528 pinned MLX masked GEMV kernels.

``gemv_masked.metal`` instantiates ``gemv_masked`` and ``gemv_t_masked`` for
float32, float16 and bfloat16 over the tile shapes ``BlockMaskedMM`` selects,
with every pairing of a boolean, multiplicative or absent output mask and
operand mask except no masks at all, each with one (``nc0``) and two
(``nc1``) batch dimensions.

Every entry runs three cases on ``matmul.cpp``'s threadgroups of ``(32, bn,
bm)`` threads and grids of ``(ceil(outputs / outputs per group), 1,
batches)``: a short one with fewer inputs than one block iteration, a ragged
one with three output and input mask blocks, partial tails and leftover
inputs, and an aligned one. A model of both kernels replays each thread's
block position, early exit, tail shift, output mask, operand masks and scales,
iteration and leftover loads, SIMD-group shuffle-down sums and guarded stores,
and must reach the masked product computed directly: an output block whose
mask is false or zero is zero, a multiplicative output mask scales its block,
and an input block whose matrix or vector mask is false or zero is skipped
while multiplicative operand masks scale it. Inputs and mask scales are small
integers: float32 accumulates their products exactly in any order, and every
result is exact in the element type, so each target must produce the model's
words.
"""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_gemv import (
    POISON,
    PRECISION,
    RESULT_BOUNDS,
    SENTINEL,
    SIGNED_MAX,
    SIMD_SIZE,
    _assert_identical,
    _elem_to_loc,
    _exact,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _below,
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/gemv_masked.metal"
SOURCE_SHA256 = "c044719fbf8c8de8b5ec37527be9537a60fdd81220a3415fcda1dc1c4097b249"
HEADERS = {
    "mlx/backend/metal/kernels/gemv_masked.h": (
        "af153cb6241e02c6d9d61c3b25718fb54108450311ad5c1f4ecca53c89cca717"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
    "mlx/backend/metal/kernels/steel/utils.h": (
        "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a"
    ),
    "mlx/backend/metal/kernels/bf16.h": (
        "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb"
    ),
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMV_MASKED"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/gemv_masked.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# gemv_masked.metal's element types, tile shapes (bm, bn, sm, sn, tm, tn) and
# (output mask, operand mask) pairs, in instantiation order. "T" names the
# element type: a multiplicative mask.
DTYPES = (("float32", "float"), ("float16", "half"), ("bfloat16", "bfloat16_t"))
GEMV_BLOCKS = (
    (2, 1, 4, 8, 1, 4),
    (2, 1, 4, 8, 4, 4),
    (2, 1, 2, 16, 1, 4),
    (2, 1, 2, 16, 4, 4),
    (4, 1, 2, 16, 4, 4),
)
GEMV_T_BLOCKS = (
    (1, 1, 8, 4, 4, 1),
    (1, 2, 8, 4, 4, 4),
    (1, 1, 8, 4, 8, 1),
    (1, 1, 8, 4, 8, 4),
    (1, 2, 8, 4, 8, 4),
    (1, 4, 8, 4, 8, 4),
)
MASKS = (
    ("bool_", "bool_"),
    ("T", "T"),
    ("bool_", "T"),
    ("T", "bool_"),
    ("nomask", "T"),
    ("nomask", "bool_"),
    ("bool_", "nomask"),
    ("T", "nomask"),
)
KERNELS = {"gemv": "gemv_masked", "gemv_t": "gemv_t_masked"}

# [[buffer(i)]] parameters of both kernels.
PARAMETERS = (
    ("mat", 0),
    ("in_vec", 1),
    ("out_vec", 3),
    ("in_vec_size", 4),
    ("out_vec_size", 5),
    ("marix_ld", 6),
    ("batch_ndim", 9),
    ("batch_shape", 10),
    ("vector_batch_stride", 11),
    ("matrix_batch_stride", 12),
    ("out_mask", 20),
    ("mat_mask", 21),
    ("vec_mask", 22),
    ("mask_strides", 23),
    ("mask_batch_strides", 24),
)
SCALAR_TYPES = {
    "in_vec_size": "int32",
    "out_vec_size": "int32",
    "marix_ld": "int32",
    "batch_ndim": "int32",
}
ARRAY_TYPES = {
    "batch_shape": "int32",
    "vector_batch_stride": "int64",
    "matrix_batch_stride": "int64",
    "mask_strides": "int32",
    "mask_batch_strides": "int64",
}

CASES = ("short", "ragged", "aligned")
GUARD_COUNT = 16
ROW_PAD = 3
VECTOR_GAP = 5
MATRIX_GAP = 7
MASK_GAP = 2
# Multiplicative mask values; 0 masks a block. Every product is exact.
SCALES = (1, -1, 2)
# Unread mask elements hold a value no mask uses.
MASK_POISON = 3
# An absent mask still binds a small buffer that is never read.
NOMASK_COUNT = 4
# The gemv kernel reads output mask blocks at mask_strides[1]; gemv_t ignores
# its output mask strides and the stride of the operand mask it does not use.
GEMV_OUT_MASK_STRIDES = (1, 2)
GEMV_T_OUT_MASK_STRIDES = (1, 1)
UNUSED_VECTOR_MASK_STRIDE = 3

# Expressions OpenGL may evaluate in 32 bits: the batch prologue's pointer
# offsets. The model shows each names an element inside its buffer.
OPENGL_INDEX_RANGE_ASSERTIONS = (
    "tid.z * buffer_load(vector_batch_stride, 0)",
    "tid.z * buffer_load(matrix_batch_stride, 0)",
    "tid.z * buffer_load(mask_batch_strides, 0)",
    "tid.z * buffer_load(mask_batch_strides, batch_ndim)",
    "batch_offsets.x",
    "batch_offsets.y",
)


def _tile_name(block):
    bm, bn, sm, sn, tm, tn = block
    return f"bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}"


def _mask_name(mask, dtype):
    return dtype if mask == "T" else mask


def _instantiations():
    """``(entry, info)`` in gemv_masked.metal's instantiation order."""
    for kind, blocks in (("gemv", GEMV_BLOCKS), ("gemv_t", GEMV_T_BLOCKS)):
        for dtype, _ in DTYPES:
            for block in blocks:
                bm, bn, sm, sn, tm, tn = block
                for nc in (0, 1):
                    for out_mask, op_mask in MASKS:
                        name = (
                            f"{kind}_outmask_{_mask_name(out_mask, dtype)}"
                            f"_opmask_{_mask_name(op_mask, dtype)}_{dtype}"
                            f"_{_tile_name(block)}_nc{nc}"
                        )
                        yield name, {
                            "kind": kind,
                            "dtype": dtype,
                            "bm": bm,
                            "bn": bn,
                            "sm": sm,
                            "sn": sn,
                            "tm": tm,
                            "tn": tn,
                            "nc": bool(nc),
                            "out_mask": out_mask,
                            "op_mask": op_mask,
                        }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)
GEMV_ENTRIES = tuple(entry for entry in ENTRIES if INFO[entry]["kind"] == "gemv")
GEMV_T_ENTRIES = tuple(entry for entry in ENTRIES if INFO[entry]["kind"] == "gemv_t")


def _workgroup(entry):
    info = INFO[entry]
    return [SIMD_SIZE, info["bn"], info["bm"]]


def _config(directory, target, entry):
    """The translation of one entry, as the gate and its contract run it."""
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: tuple(_workgroup(entry))}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (
                OPENGL_INDEX_RANGE_ASSERTIONS if target == "opengl" else ()
            )
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 16384,
                "target_options": {
                    "directx": {
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {"software_subgroup_width": SIMD_SIZE},
                },
            }
        },
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _blocks(info):
    """``(outputs per group, inputs per iteration, TOut, TIn, mask block)``.

    ``GEMVKernel`` indexes its masks by ``out_row / blockN`` (``tid.x`` when
    ``blockN == blockM``) and by iteration, and ``GEMVTKernel`` by
    ``out_col / blockM`` (``tid.x`` when equal) and by iteration, so one
    mask block covers one input iteration and that many outputs.
    """
    threads_m = info["bm"] * info["sm"]
    threads_n = info["bn"] * info["sn"]
    block_m, block_n = threads_m * info["tm"], threads_n * info["tn"]
    if info["kind"] == "gemv":
        assert block_n >= block_m  # static_assert in GEMVKernel
        return block_m, block_n, info["tm"], info["tn"], block_n
    assert block_m >= block_n
    return block_n, block_m, info["tn"], info["tm"], block_m


def _shape(entry, case):
    """``(out_vec_size, in_vec_size, marix_ld)``; ``gemv_t`` rows are inputs.

    Ragged outputs leave ``TOut + 1`` outputs in the last mask block, so the
    tail block shifts inwards without crossing a mask block.
    """
    info = INFO[entry]
    _, per_iteration, t_out, t_in, mask_block = _blocks(info)
    out_size = {
        "short": t_out + 2,
        "ragged": 2 * mask_block + t_out + 1,
        "aligned": 2 * mask_block,
    }[case]
    if info["kind"] == "gemv":
        in_size = {
            "short": per_iteration - t_in - 1,
            "ragged": 2 * per_iteration + t_in + 1,
            "aligned": 2 * per_iteration,
        }[case]
        row = in_size
    else:
        in_size = {
            "short": per_iteration - 1,
            "ragged": 2 * per_iteration + 3,
            "aligned": 2 * per_iteration,
        }[case]
        row = out_size
    return out_size, in_size, row + (0 if case == "aligned" else ROW_PAD)


def _mask_blocks(entry, case):
    """``(output mask blocks, input mask blocks)``."""
    info = INFO[entry]
    _, per_iteration, _, _, mask_block = _blocks(info)
    out_size, in_size, _ = _shape(entry, case)
    return -(-out_size // mask_block), -(-in_size // per_iteration)


def _mask_strides(entry, case):
    """The ``mask_strides`` words ``BlockMaskedMM`` would pass, by mask."""
    info = INFO[entry]
    _, inputs = _mask_blocks(entry, case)
    pitch = inputs + 1
    if info["kind"] == "gemv":
        out = GEMV_OUT_MASK_STRIDES
        # mat_mask_strides[0] steps inputs, [1] outputs; the vector steps [1].
        mat = (1, pitch)
        vec = (UNUSED_VECTOR_MASK_STRIDE, 1)
    else:
        out = GEMV_T_OUT_MASK_STRIDES
        # mat_mask_strides[0] steps outputs, [1] inputs; the vector steps [0].
        mat = (pitch, 1)
        vec = (1, UNUSED_VECTOR_MASK_STRIDE)
    return {"out": out, "mat": mat, "vec": vec}


def _mask_sizes(entry, case):
    """Elements of one output, matrix and vector mask, gaps included."""
    outputs, inputs = _mask_blocks(entry, case)
    strides = _mask_strides(entry, case)
    out_pitch = strides["out"][1] if INFO[entry]["kind"] == "gemv" else 1
    return {
        "out": outputs * out_pitch + MASK_GAP,
        "mat": outputs * (inputs + 1) + MASK_GAP,
        "vec": inputs * 1 + MASK_GAP,
    }


def _batching(entry, case):
    """Batch metadata and per-batch element offsets of every batched buffer."""
    info = INFO[entry]
    out_size, in_size, ld = _shape(entry, case)
    rows = out_size if info["kind"] == "gemv" else in_size
    vs = in_size + VECTOR_GAP
    ms = rows * ld + MATRIX_GAP
    sizes = _mask_sizes(entry, case)
    os_, mms, vms = sizes["out"], sizes["mat"], sizes["vec"]
    if info["nc"]:
        shape = [2, 2]
        strides = {
            "in_vec": [0, vs],
            "mat": [ms, 2 * ms],
            "out_mask": [2 * os_, os_],
            "mat_mask": [mms, 0],
            "vec_mask": [0, vms],
        }
    else:
        shape = [2]
        strides = {
            "in_vec": [vs],
            "mat": [ms],
            "out_mask": [os_],
            "mat_mask": [mms],
            "vec_mask": [vms],
        }
    has_out = info["out_mask"] != "nomask"
    has_op = info["op_mask"] != "nomask"
    count = shape[0] * (shape[1] if len(shape) > 1 else 1)
    names = ("in_vec", "mat", "out_mask", "mat_mask", "vec_mask")
    batches = []
    for z in range(count):
        if info["nc"]:
            offsets = _elem_to_loc(z, shape, tuple(strides[name] for name in names))
        else:
            offsets = [z * strides[name][0] for name in names]
        batch = dict(zip(names, offsets))
        if not has_out:
            batch["out_mask"] = 0
        if not has_op:
            batch["mat_mask"] = batch["vec_mask"] = 0
        batches.append(batch)
    mask_batch_strides = []
    if has_out:
        mask_batch_strides += strides["out_mask"]
    if has_op:
        mask_batch_strides += strides["mat_mask"] + strides["vec_mask"]
    mask_strides = []
    layout = _mask_strides(entry, case)
    if has_out:
        mask_strides += layout["out"]
    if has_op:
        mask_strides += layout["mat"] + layout["vec"]
    metadata = {
        "batch_ndim": len(shape),
        "batch_shape": shape,
        "vector_batch_stride": strides["in_vec"],
        "matrix_batch_stride": strides["mat"],
        "mask_strides": mask_strides,
        "mask_batch_strides": mask_batch_strides,
    }
    return metadata, batches


def _grid(entry, case):
    per_group = _blocks(INFO[entry])[0]
    out_size = _shape(entry, case)[0]
    return [-(-out_size // per_group), 1, len(_batching(entry, case)[1])]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _scale(mask, value):
    """The factor a mask value applies: 0 masks, a multiplicative one scales."""
    if mask == "nomask":
        return 1
    if mask == "bool_":
        return 1 if value else 0
    return value


def _mask_values(mask, draws, count, forced):
    """``count`` mask values; ``forced`` maps positions to active (or not)."""
    values = []
    for index in range(count):
        active = forced.get(index)
        if active is None:
            active = _below(draws, 4) != 0
        if mask == "bool_":
            values.append(bool(active))
        else:
            values.append(SCALES[_below(draws, len(SCALES))] if active else 0)
    return values


def _row_values(dtype, draws, coefficients, bound):
    """A matrix row whose dot product with ``coefficients`` is nonzero when any
    coefficient is and at most ``bound`` in magnitude: steps of a walk flip sign
    whenever the running sum would leave the bound."""
    values, total = [], 0
    for coefficient in coefficients:
        value = _signed(draws, 2)
        if abs(total + value * coefficient) > bound:
            value = -value
        values.append(value)
        total += value * coefficient
    if total == 0 and any(coefficients):
        last = max(k for k, coefficient in enumerate(coefficients) if coefficient)
        values[last] = 1 if values[last] == 0 else -values[last]
    return values


def _case_data(entry, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    info = INFO[entry]
    dtype = info["dtype"]
    out_size, in_size, ld = _shape(entry, case)
    _, per_iteration, _, _, mask_block = _blocks(info)
    outputs, inputs = _mask_blocks(entry, case)
    layout = _mask_strides(entry, case)
    sizes = _mask_sizes(entry, case)
    metadata, batches = _batching(entry, case)
    draws = _draws("gemv_masked", entry, case)
    rows = out_size if info["kind"] == "gemv" else in_size
    vector_length = max(b["in_vec"] for b in batches) + in_size + VECTOR_GAP
    matrix_length = max(b["mat"] for b in batches) + rows * ld + MATRIX_GAP
    in_vec = [POISON[k % len(POISON)] for k in range(vector_length)]
    mat = [POISON[k % len(POISON)] for k in range(matrix_length)]
    vectors = {}
    for batch in batches:
        if batch["in_vec"] not in vectors:
            values = [
                (1 + _below(draws, 2)) * (1 if _below(draws, 2) else -1)
                for _ in range(in_size)
            ]
            in_vec[batch["in_vec"] : batch["in_vec"] + in_size] = values
            vectors[batch["in_vec"]] = values

    def mask_storage(mask, kind, offsets, count, element):
        """Mask buffer storage and each instance's logical values."""
        if mask == "nomask":
            return [0] * NOMASK_COUNT, {}
        length = max(offsets) + sizes[kind]
        storage = [MASK_POISON if mask != "bool_" else True] * length
        instances = {}
        for number, offset in enumerate(sorted(set(offsets))):
            values = _mask_values(mask, draws, count, forced(kind, number))
            for index, value in enumerate(values):
                storage[offset + element(index)] = value
            instances[offset] = values
        return storage, instances

    def forced(kind, number):
        """Positions forced on or off so every case takes each mask path.

        Input block 0 stays active in every operand mask, so each unmasked
        output has a nonzero dot product; with several input blocks one is
        skipped. With several output blocks the first instance masks one and
        keeps another; one block alternates between instances.
        """
        if kind == "out":
            if outputs > 1 and number == 0:
                return {0: True, outputs - 1: False}
            return {0: number % 2 == 0}
        if kind == "vec":
            positions = {0: True}
            if inputs > 1 and number == 0:
                positions[1] = False
            return positions
        positions = {output * inputs: True for output in range(outputs)}
        if inputs > 1 and number == 0:
            positions[inputs - 1] = False
        return positions

    out_pitch = layout["out"][1] if info["kind"] == "gemv" else 1
    out_mask, out_masks = mask_storage(
        info["out_mask"],
        "out",
        [b["out_mask"] for b in batches],
        outputs,
        lambda index: index * out_pitch,
    )
    if info["kind"] == "gemv":
        # mat_mask[output * strides[1] + input * strides[0]]
        mat_element = lambda index: (  # noqa: E731
            (index // inputs) * layout["mat"][1] + (index % inputs) * layout["mat"][0]
        )
        vec_element = lambda index: index * layout["vec"][1]  # noqa: E731
    else:
        mat_element = lambda index: (  # noqa: E731
            (index // inputs) * layout["mat"][0] + (index % inputs) * layout["mat"][1]
        )
        vec_element = lambda index: index * layout["vec"][0]  # noqa: E731
    mat_mask, mat_masks = mask_storage(
        info["op_mask"],
        "mat",
        [b["mat_mask"] for b in batches],
        outputs * inputs,
        mat_element,
    )
    vec_mask, vec_masks = mask_storage(
        info["op_mask"],
        "vec",
        [b["vec_mask"] for b in batches],
        inputs,
        vec_element,
    )
    bound = RESULT_BOUNDS[dtype]
    total = len(batches) * out_size + GUARD_COUNT
    initial = [SENTINEL] * total
    expected = list(initial)
    built = set()
    for z, batch in enumerate(batches):
        assert batch["mat"] not in built, (entry, case)
        built.add(batch["mat"])
        vector = vectors[batch["in_vec"]]
        for out in range(out_size):
            block = out // mask_block
            out_scale = (
                1
                if info["out_mask"] == "nomask"
                else _scale(info["out_mask"], out_masks[batch["out_mask"]][block])
            )
            coefficients = []
            for k in range(in_size):
                factor = 1
                if info["op_mask"] != "nomask":
                    position = k // per_iteration
                    factor = _scale(
                        info["op_mask"],
                        mat_masks[batch["mat_mask"]][block * inputs + position],
                    ) * _scale(info["op_mask"], vec_masks[batch["vec_mask"]][position])
                    if info["op_mask"] == "bool_":
                        factor = int(factor != 0)
                coefficients.append(vector[k] * factor)
            line = _row_values(
                dtype, draws, coefficients, bound // max(1, abs(out_scale))
            )
            for k, value in enumerate(line):
                if info["kind"] == "gemv":
                    mat[batch["mat"] + out * ld + k] = value
                else:
                    mat[batch["mat"] + k * ld + out] = value
            if out_scale == 0:
                value = 0
            else:
                value = out_scale * sum(a * b for a, b in zip(line, coefficients))
                assert value != 0, (entry, case, z, out)
            assert _exact(dtype, value), (entry, case, z, out, value)
            expected[z * out_size + out] = value
    values = {
        "mat": (dtype, mat),
        "in_vec": (dtype, in_vec),
        "out_vec": (dtype, initial),
        "in_vec_size": ("int32", [in_size]),
        "out_vec_size": ("int32", [out_size]),
        "marix_ld": ("int32", [ld]),
        "batch_ndim": ("int32", [metadata["batch_ndim"]]),
        "batch_shape": ("int32", list(metadata["batch_shape"])),
        "vector_batch_stride": ("int64", list(metadata["vector_batch_stride"])),
        "matrix_batch_stride": ("int64", list(metadata["matrix_batch_stride"])),
        "out_mask": (_mask_dtype(info["out_mask"], dtype), out_mask),
        "mat_mask": (_mask_dtype(info["op_mask"], dtype), mat_mask),
        "vec_mask": (_mask_dtype(info["op_mask"], dtype), vec_mask),
        "mask_strides": ("int32", list(metadata["mask_strides"])),
        "mask_batch_strides": ("int64", list(metadata["mask_batch_strides"])),
    }
    assert set(values) == {name for name, _ in PARAMETERS}
    return {
        "values": values,
        "initial": initial,
        "expected": expected,
        "grid": _grid(entry, case),
        "shape": {"outVecSize": out_size, "inVecSize": in_size, "matrixLd": ld},
        "batches": batches,
        "masks": {"out": out_masks, "mat": mat_masks, "vec": vec_masks},
        "written": len(batches) * out_size,
        "size": total,
    }


def _mask_dtype(mask, dtype):
    # An absent mask binds _NoMask, a one-byte struct.
    return {"nomask": "int8", "bool_": "bool_"}.get(mask, dtype)


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


class _Context:
    """One batch of one case: buffers, offsets and the model's records."""

    def __init__(self, entry, data, batch, mutation):
        info = INFO[entry]
        self.info = info
        self.storage = {name: values for name, (_, values) in data["values"].items()}
        self.out_size = data["shape"]["outVecSize"]
        self.in_size = data["shape"]["inVecSize"]
        self.ld = data["shape"]["matrixLd"]
        self.batch = batch
        self.mutation = mutation
        strides = data["values"]["mask_strides"][1]
        has_out = info["out_mask"] != "nomask"
        self.out_strides = strides[:2] if has_out else None
        rest = strides[2:] if has_out else strides
        self.mat_strides = rest[:2] if info["op_mask"] != "nomask" else None
        self.vec_strides = rest[2:4] if info["op_mask"] != "nomask" else None
        self.accesses = {
            name: [] for name in ("mat", "in_vec", "out_mask", "mat_mask", "vec_mask")
        }
        self.paths = set()

    def load(self, name, index):
        self.accesses[name].append(index)
        values = self.storage[name]
        if self.mutation is not None and not 0 <= index < len(values):
            return POISON[0]  # a defective kernel reads past its buffer
        return values[index]

    def load_safe(self, name, base, offset, size, count):
        """GEMVKernel::load_safe: whole when it fits, else element-guarded."""
        if offset + count <= size:
            return [self.load(name, base + offset + k) for k in range(count)]
        self.paths.add("partial-load")
        return [
            self.load(name, base + offset + k) if offset + k < size else 0
            for k in range(count)
        ]

    def output_mask(self, block, offset):
        """``(stored zeros?, out_scale)`` of ``out_mask[offset]``."""
        mask = self.info["out_mask"]
        if mask == "nomask":
            return False, 1
        value = self.load("out_mask", self.batch["out_mask"] + offset)
        if self.mutation == "outmask":
            value = True if mask == "bool_" else 1
        if not value:
            self.paths.add("output-mask-zero")
            return True, 0
        if mask == "bool_" or self.mutation == "scale":
            return False, 1
        self.paths.add("output-mask-scale")
        return False, value

    def operand_mask(self, mat_offset, vec_offset):
        """``(block taken?, block_scale)`` of one input block."""
        mask = self.info["op_mask"]
        if mask == "nomask":
            return True, 1
        mat = self.load("mat_mask", self.batch["mat_mask"] + mat_offset)
        vec = self.load("vec_mask", self.batch["vec_mask"] + vec_offset)
        if self.mutation == "opmask":
            mat = vec = True if mask == "bool_" else 1
        if not (mat and vec):
            self.paths.add("operand-mask-skip")
            return False, 0
        if mask == "bool_" or self.mutation == "scale":
            return True, 1
        self.paths.add("operand-mask-scale")
        return True, mat * vec


def _gemv_group(ctx, gx):
    """Masked GEMVKernel::run for threadgroup ``gx``: ``(row, value)`` stores."""
    info = ctx.info
    bm_, bn_, sm, sn, tm_, tn_ = (info[k] for k in ("bm", "bn", "sm", "sn", "tm", "tn"))
    assert bn_ == 1  # gemv_masked.metal never instantiates a threadgroup reduction
    block_m, block_n = bm_ * sm * tm_, bn_ * sn * tn_
    rows, size, ld = ctx.out_size, ctx.in_size, ctx.ld
    n_iter = size // block_n
    leftover = size - block_n * n_iter
    lanes, stores = {}, []
    for simd_gid in range(bm_ * bn_):
        for lane in range(SIMD_SIZE):
            thr_m = lane // sn if sn != 32 else 0
            thr_n = lane % sn if sn != 32 else lane
            simd_m = sm * simd_gid
            bm = (simd_m + thr_m) * tm_
            bn = thr_n * tn_
            out_row = gx * block_m + bm
            if out_row >= rows:
                ctx.paths.add("early-exit")
                continue
            if out_row + tm_ > rows and ctx.mutation != "shift":
                ctx.paths.add("tail-shift")
                out_row = rows - tm_
            m_block = out_row // block_n if block_n > block_m else gx
            out_offset = m_block * ctx.out_strides[1] if ctx.out_strides else 0
            zeros, out_scale = ctx.output_mask(m_block, out_offset)
            if zeros:
                if thr_n == 0:
                    stores += [(out_row + t, 0) for t in range(tm_)]
                continue
            mat_offset = m_block * ctx.mat_strides[1] if ctx.mat_strides else 0
            vec_offset = 0
            row_base = ctx.batch["mat"] + out_row * ld
            result = [0] * tm_
            for _ in range(n_iter):
                ctx.paths.add("iterations")
                taken, scale = ctx.operand_mask(mat_offset, vec_offset)
                if taken:
                    coeff = [
                        ctx.load("in_vec", ctx.batch["in_vec"] + bn + k) * scale
                        for k in range(tn_)
                    ]
                    for t in range(tm_):
                        for k in range(tn_):
                            element = ctx.load("mat", row_base + t * ld + bn + k)
                            result[t] += element * coeff[k]
                bn += block_n
                if ctx.mat_strides:
                    mat_offset += ctx.mat_strides[0]
                    vec_offset += ctx.vec_strides[1]
            if leftover > 0 and ctx.mutation != "leftover":
                ctx.paths.add("leftover")
                taken, scale = ctx.operand_mask(mat_offset, vec_offset)
                if taken:
                    coeff = [
                        value * scale
                        for value in ctx.load_safe(
                            "in_vec", ctx.batch["in_vec"], bn, size, tn_
                        )
                    ]
                    for t in range(tm_):
                        inter = ctx.load_safe("mat", row_base + t * ld, bn, size, tn_)
                        for k in range(tn_):
                            result[t] += inter[k] * coeff[k]
            lanes[simd_gid, lane] = {
                "thr_n": thr_n,
                "out_row": out_row,
                "result": [value * out_scale for value in result],
            }
    # simd_shuffle_down(sn = SN / 2 .. 1) leaves each row group's sum in its
    # thrN == 0 lane; the group's lanes share the row block, its masks and
    # exits.
    for (simd_gid, lane), state in sorted(lanes.items()):
        if state["thr_n"] != 0:
            continue
        group = [lanes.get((simd_gid, lane + k)) for k in range(sn)]
        assert all(item and item["out_row"] == state["out_row"] for item in group)
        for t in range(tm_):
            stores.append(
                (state["out_row"] + t, sum(item["result"][t] for item in group))
            )
    return stores


def _gemv_t_group(ctx, gx):
    """Masked GEMVTKernel::run for threadgroup ``gx``: ``(column, value)`` stores."""
    info = ctx.info
    bm_, bn_, sm, sn, tm_, tn_ = (info[k] for k in ("bm", "bn", "sm", "sn", "tm", "tn"))
    assert bm_ == 1  # gemv_masked.metal never instantiates a threadgroup reduction
    block_m, block_n = bm_ * sm * tm_, bn_ * sn * tn_
    columns, size, ld = ctx.out_size, ctx.in_size, ctx.ld
    n_iter = size // block_m
    leftover = size - block_m * n_iter
    lanes, stores = {}, []
    for simd_gid in range(bm_ * bn_):
        for lane in range(SIMD_SIZE):
            thr_m = lane // sn if sn != 32 else 0
            thr_n = lane % sn if sn != 32 else lane
            sg_m = simd_gid // bn_ if bn_ != 1 else simd_gid
            sg_n = simd_gid % bn_ if bn_ != 1 else 0
            cm = sm * sg_m + thr_m
            cn = sn * sg_n + thr_n
            bm, bn = cm * tm_, cn * tn_
            out_col = gx * block_n + bn
            n_block = out_col // block_m if block_m > block_n else gx
            zeros, out_scale = ctx.output_mask(n_block, n_block)
            if zeros:
                if cm == 0 and out_col < columns:
                    stores += [
                        (out_col + k, 0) for k in range(tn_) if out_col + k < columns
                    ]
                continue
            mat_offset = n_block * ctx.mat_strides[0] if ctx.mat_strides else 0
            vec_offset = 0
            result = [0] * tn_
            if out_col < columns:
                if out_col + tn_ > columns and ctx.mutation != "shift":
                    ctx.paths.add("tail-shift")
                    out_col = columns - tn_
                for _ in range(n_iter):
                    # The row loop's threadgroup_barrier runs here.
                    ctx.paths.add("iterations")
                    taken, scale = ctx.operand_mask(mat_offset, vec_offset)
                    if taken:
                        coeff = [
                            ctx.load("in_vec", ctx.batch["in_vec"] + bm + t) * scale
                            for t in range(tm_)
                        ]
                        for t in range(tm_):
                            for k in range(tn_):
                                element = ctx.load(
                                    "mat",
                                    ctx.batch["mat"] + (bm + t) * ld + out_col + k,
                                )
                                result[k] += coeff[t] * element
                    bm += block_m
                    if ctx.mat_strides:
                        mat_offset += ctx.mat_strides[1]
                        vec_offset += ctx.vec_strides[0]
                if leftover > 0 and ctx.mutation != "leftover":
                    ctx.paths.add("leftover")
                    taken, scale = ctx.operand_mask(mat_offset, vec_offset)
                    if taken:
                        t = 0
                        while t < tm_ and bm + t < size:
                            coeff = ctx.load("in_vec", ctx.batch["in_vec"] + bm + t)
                            coeff *= scale
                            for k in range(tn_):
                                element = ctx.load(
                                    "mat",
                                    ctx.batch["mat"] + (bm + t) * ld + out_col + k,
                                )
                                result[k] += coeff * element
                            t += 1
                        if t < tm_:
                            ctx.paths.add("partial-load")
            else:
                ctx.paths.add("divergent-skip")
            lanes[simd_gid, lane] = {
                "thr_m": thr_m,
                "cm": cm,
                "out_col": out_col,
                "result": [value * out_scale for value in result],
            }
    for (simd_gid, lane), state in sorted(lanes.items()):
        if state["thr_m"] != 0:
            continue
        # simd_shuffle_down(SN * sm) sums the lanes of one column block.
        group = [lanes.get((simd_gid, lane + sn * m)) for m in range(sm)]
        assert all(item and item["out_col"] == state["out_col"] for item in group)
        if state["cm"] == 0 and state["out_col"] < columns:
            for k in range(tn_):
                stores.append(
                    (state["out_col"] + k, sum(item["result"][k] for item in group))
                )
    return stores


def _simulate(entry, case, data, mutation=None):
    """Replay every threadgroup: output words, buffer accesses and paths."""
    info = INFO[entry]
    output = list(data["initial"])
    stored = {}
    accesses = {
        name: [] for name in ("mat", "in_vec", "out_mask", "mat_mask", "vec_mask")
    }
    accesses["out_vec"] = []
    paths = set()
    group = _gemv_group if info["kind"] == "gemv" else _gemv_t_group
    batches = data["batches"]
    if mutation == "batch":
        batches = [batches[0]] * len(batches)
    elif mutation == "maskbatch":
        first = batches[0]
        batches = [
            dict(batch, **{name: first[name] for name in ("out_mask", "mat_mask")})
            for batch in batches
        ]
    for z, batch in enumerate(batches):
        ctx = _Context(entry, data, batch, mutation)
        for gx in range(data["grid"][0]):
            for index, value in group(ctx, gx):
                position = z * ctx.out_size + index
                accesses["out_vec"].append(position)
                if mutation is None and position in stored:
                    # Shifted tail blocks store the same words twice.
                    assert stored[position] == value, (entry, case, position)
                stored[position] = value
                if 0 <= position < len(output):
                    output[position] = value
        for name, items in ctx.accesses.items():
            accesses[name] += items
        paths |= ctx.paths
    return output, accesses, paths


def _expected_paths(entry):
    info = INFO[entry]
    paths = {"iterations", "leftover", "partial-load"}
    if _blocks(info)[2] > 1:
        # Blocks of one row or column never overhang the output.
        paths.add("tail-shift")
    paths.add("early-exit" if info["kind"] == "gemv" else "divergent-skip")
    if info["out_mask"] != "nomask":
        paths.add("output-mask-zero")
        if info["out_mask"] == "T":
            paths.add("output-mask-scale")
    if info["op_mask"] != "nomask":
        paths.add("operand-mask-skip")
        if info["op_mask"] == "T":
            paths.add("operand-mask-scale")
    return paths


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _output_count(entry, case):
    return len(_batching(entry, case)[1]) * _shape(entry, case)[0] + GUARD_COUNT


def _element_count():
    return sum(_output_count(entry, case) for entry in ENTRIES for case in CASES)


def _expected_validation_contract():
    return {
        "model": (
            "masked GEMVKernel::run and GEMVTKernel::run thread by thread: block "
            "positions, early exits, tail shifts, output and operand masks and "
            "scales, iteration and leftover loads, SIMD-group shuffle-down sums "
            "and guarded stores"
        ),
        "comparison": "exact words",
        "workgroupSize": "[32, bn, bm]",
        "grid": "[ceil(out_vec_size / outputs per threadgroup), 1, batches]",
        "cases": {
            "gemv": {
                "maskBlock": "BN * SN * TN",
                "outVecSize": {
                    "short": "TM + 2",
                    "ragged": "2 * BN * SN * TN + TM + 1",
                    "aligned": "2 * BN * SN * TN",
                },
                "inVecSize": {
                    "short": "BN * SN * TN - TN - 1",
                    "ragged": "2 * BN * SN * TN + TN + 1",
                    "aligned": "2 * BN * SN * TN",
                },
            },
            "gemv_t": {
                "maskBlock": "BM * SM * TM",
                "outVecSize": {
                    "short": "TN + 2",
                    "ragged": "2 * BM * SM * TM + TN + 1",
                    "aligned": "2 * BM * SM * TM",
                },
                "inVecSize": {
                    "short": "BM * SM * TM - 1",
                    "ragged": "2 * BM * SM * TM + 3",
                    "aligned": "2 * BM * SM * TM",
                },
            },
            "matrixLdPadding": {"short": ROW_PAD, "ragged": ROW_PAD, "aligned": 0},
        },
        "batching": {
            "contiguous": {"batchShape": [2], "strides": "one per batch"},
            "nonContiguous": {
                "batchShape": [2, 2],
                "vectorStrides": "[0, vector]",
                "matrixStrides": "[matrix, 2 * matrix]",
                "outputMaskStrides": "[2 * mask, mask]",
                "matrixMaskStrides": "[mask, 0]",
                "vectorMaskStrides": "[0, mask]",
            },
        },
        "masks": {
            "maskStrides": {
                "gemv": {
                    "output": list(GEMV_OUT_MASK_STRIDES),
                    "matrix": "[1, input blocks + 1]",
                    "vector": [UNUSED_VECTOR_MASK_STRIDE, 1],
                },
                "gemv_t": {
                    "output": list(GEMV_T_OUT_MASK_STRIDES),
                    "matrix": "[input blocks + 1, 1]",
                    "vector": [1, UNUSED_VECTOR_MASK_STRIDE],
                },
            },
            "multiplicativeScales": list(SCALES),
            "poison": MASK_POISON,
            "coverage": (
                "input block 0 stays active; with several input blocks one is "
                "skipped and with several output blocks one is masked"
            ),
            "absentMaskElements": NOMASK_COUNT,
        },
        "inputs": {
            "vectors": "nonzero integers in [-2, 2]",
            "matrices": (
                "integers in [-2, 2] whose masked dot product with the batch's "
                "vector is nonzero and at most the element type's bound over "
                "the output scale"
            ),
            "resultBounds": RESULT_BOUNDS,
            "poison": list(POISON),
        },
        "guards": {
            "output": "every element starts at the sentinel, which no result equals",
            "sentinel": SENTINEL,
            "trailingOutputGuards": GUARD_COUNT,
        },
        "openglIndexRangeAssertions": [
            {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS
        ],
        "softwareSubgroupWidth": {"targets": ["opengl", "directx"], "width": SIMD_SIZE},
        "directx": {"relativeWaveShuffleOutOfRange": "self"},
        "elementTypes": {
            target: {
                dtype: _element_type(target, dtype)
                for dtype in (*(name for name, _ in DTYPES), "bool_", "int8")
            }
            for target in TARGETS
        },
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "excludedEntries": {"count": 0},
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "checkedElementCountByTarget": _element_count(),
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_gemv_masked_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-gemv-masked-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "headers": HEADERS,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
        assert len({record["sha256"] for record in artifacts.values()}) == len(ENTRIES)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_gemv_masked_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "gemv_masked.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 528
    assert len(GEMV_ENTRIES) == 240 and len(GEMV_T_ENTRIES) == 288
    assert ENTRIES[0] == (
        "gemv_outmask_bool__opmask_bool__float32_bm2_bn1_sm4_sn8_tm1_tn4_nc0"
    )
    assert ENTRIES[-1] == (
        "gemv_t_outmask_bfloat16_opmask_nomask_bfloat16_bm1_bn4_sm8_sn4_tm8_tn4_nc1"
    )
    for entry, info in INFO.items():
        # matmul.cpp: "gemv[_t]_outmask_{}_opmask_{}_{type}_bm{}_bn{}_sm{}_sn{}
        # _tm{}_tn{}_nc{}", the masks named "nomask", "bool_" or the type.
        name = (
            f"{info['kind']}_outmask_{_mask_name(info['out_mask'], info['dtype'])}"
            f"_opmask_{_mask_name(info['op_mask'], info['dtype'])}_{info['dtype']}"
            f"_bm{info['bm']}_bn{info['bn']}_sm{info['sm']}_sn{info['sn']}"
            f"_tm{info['tm']}_tn{info['tn']}_nc{int(info['nc'])}"
        )
        assert entry == name
        assert info["sm"] * info["sn"] == SIMD_SIZE
        assert (info["out_mask"], info["op_mask"]) != ("nomask", "nomask")


@pytest.mark.parametrize("entry", ENTRIES[::7])
def test_gemv_masked_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    per_group, per_iteration, t_out, t_in, mask_block = _blocks(info)
    # matmul.cpp: n_out_per_tgp = bm * sm * tm (gemv) or bn * sn * tn
    # (gemv_t), group_dims (32, bn, bm), grid (n_tgp, 1, batch_size_out).
    if info["kind"] == "gemv":
        assert per_group == info["bm"] * info["sm"] * info["tm"]
    else:
        assert per_group == info["bn"] * info["sn"] * info["tn"]
    assert _workgroup(entry) == [32, info["bn"], info["bm"]]
    for case in CASES:
        out_size, in_size, ld = _shape(entry, case)
        grid = _grid(entry, case)
        assert grid[0] == (out_size + per_group - 1) // per_group and grid[1] == 1
        # The tail block shifts inwards within its mask block.
        assert out_size >= t_out and in_size >= 1
        tail = out_size % mask_block
        assert tail == 0 or tail >= t_out or out_size % t_out == 0
        rows_long = in_size if info["kind"] == "gemv" else out_size
        assert ld >= rows_long
        metadata, batches = _batching(entry, case)
        assert grid[2] == len(batches) == 2 * (2 if info["nc"] else 1)
        assert metadata["batch_ndim"] == (2 if info["nc"] else 1)
        has_out = info["out_mask"] != "nomask"
        has_op = info["op_mask"] != "nomask"
        assert len(metadata["mask_strides"]) == 2 * has_out + 4 * has_op
        assert len(metadata["mask_batch_strides"]) == metadata["batch_ndim"] * (
            has_out + 2 * has_op
        )
    assert _shape(entry, "short")[1] < per_iteration
    assert _shape(entry, "aligned")[1] % per_iteration == 0
    assert _shape(entry, "ragged")[1] % per_iteration != 0
    assert _mask_blocks(entry, "ragged") == (3, 3)


MODEL_ENTRIES = ENTRIES[::4]


@pytest.mark.parametrize("entry", MODEL_ENTRIES)
def test_gemv_masked_model_matches_masked_products(entry):
    """The thread-level model stores each direct masked product, and only those."""
    paths = set()
    for case in CASES:
        data = _case_data(entry, case)
        output, accesses, seen = _simulate(entry, case, data)
        paths |= seen
        assert output == data["expected"], (entry, case)
        assert sorted(set(accesses["out_vec"])) == list(range(data["written"]))
    assert paths >= _expected_paths(entry), sorted(_expected_paths(entry) - paths)


@pytest.mark.parametrize("dtype", [dtype for dtype, _ in DTYPES])
def test_gemv_masked_inputs_keep_every_result_exact(dtype):
    """Results, guards, poison and scales are exact; computed results are nonzero."""
    bits = PRECISION[dtype]
    assert 2 * RESULT_BOUNDS[dtype] + 7 < 2**bits
    for value in (SENTINEL, *POISON, *SCALES, MASK_POISON):
        assert _exact(dtype, value)
    for entry in [entry for entry in ENTRIES if INFO[entry]["dtype"] == dtype][::6]:
        for case in CASES:
            data = _case_data(entry, case)
            written = data["expected"][: data["written"]]
            assert all(_exact(dtype, value) for value in written)
            assert SENTINEL not in written
            assert data["expected"][data["written"] :] == [SENTINEL] * GUARD_COUNT
            if INFO[entry]["out_mask"] == "nomask":
                assert 0 not in written
            # Products and their sums stay far inside float32's 24 bits.
            assert 64 * data["shape"]["inVecSize"] < 2**24


@pytest.mark.parametrize(
    "entry", [entry for entry in ENTRIES if INFO[entry]["dtype"] == "bfloat16"][::3]
)
def test_gemv_masked_model_justifies_opengl_index_assertions(entry):
    """Every load and store the asserted offsets form lies inside its buffer."""
    for case in CASES:
        data = _case_data(entry, case)
        _, accesses, _ = _simulate(entry, case, data)
        for name, indices in accesses.items():
            if not indices:
                continue
            length = len(data["values"][name][1])
            assert 0 <= min(indices) and max(indices) < length <= SIGNED_MAX, (
                case,
                name,
            )
        metadata, batches = _batching(entry, case)
        assert min(value for batch in batches for value in batch.values()) >= 0
        for name in (
            "vector_batch_stride",
            "matrix_batch_stride",
            "mask_batch_strides",
        ):
            assert min(metadata[name], default=0) >= 0


MUTATIONS = {
    "outmask": "ignores the output mask",
    "opmask": "ignores the operand masks",
    "scale": "drops multiplicative mask scales",
    "leftover": "drops the leftover inputs",
    "shift": "does not shift the tail block inwards",
    "batch": "addresses every batch as the first",
    "maskbatch": "addresses every batch's masks as the first's",
}


@pytest.mark.parametrize(
    "entry,mutation",
    (
        (
            "gemv_outmask_bool__opmask_nomask_float32_bm2_bn1_sm4_sn8_tm4_tn4_nc0",
            "outmask",
        ),
        (
            "gemv_outmask_nomask_opmask_bool__float16_bm2_bn1_sm2_sn16_tm1_tn4_nc1",
            "opmask",
        ),
        (
            "gemv_outmask_float32_opmask_float32_float32_bm4_bn1_sm2_sn16_tm4_tn4_nc0",
            "scale",
        ),
        (
            "gemv_outmask_bool__opmask_bfloat16_bfloat16_bm2_bn1_sm4_sn8_tm4_tn4_nc1",
            "leftover",
        ),
        (
            "gemv_outmask_float16_opmask_bool__float16_bm2_bn1_sm2_sn16_tm4_tn4_nc0",
            "shift",
        ),
        (
            "gemv_outmask_float32_opmask_bool__float32_bm2_bn1_sm4_sn8_tm1_tn4_nc1",
            "batch",
        ),
        (
            "gemv_outmask_bool__opmask_bool__float32_bm2_bn1_sm4_sn8_tm1_tn4_nc1",
            "maskbatch",
        ),
        (
            "gemv_t_outmask_float16_opmask_nomask_float16_bm1_bn1_sm8_sn4_tm4_tn1_nc0",
            "outmask",
        ),
        (
            "gemv_t_outmask_nomask_opmask_float32_float32_bm1_bn2_sm8_sn4_tm8_tn4_nc1",
            "opmask",
        ),
        (
            "gemv_t_outmask_bfloat16_opmask_bool__bfloat16_bm1_bn4_sm8_sn4_tm8_tn4_nc0",
            "scale",
        ),
        (
            "gemv_t_outmask_bool__opmask_float32_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc1",
            "leftover",
        ),
        (
            "gemv_t_outmask_float32_opmask_float32_float32_bm1_bn1_sm8_sn4_tm8_tn4_nc0",
            "shift",
        ),
        (
            "gemv_t_outmask_nomask_opmask_bool__float16_bm1_bn2_sm8_sn4_tm8_tn4_nc1",
            "batch",
        ),
        (
            "gemv_t_outmask_float32_opmask_float32_float32_bm1_bn1_sm8_sn4_tm4_tn1_nc1",
            "maskbatch",
        ),
    ),
)
def test_gemv_masked_expectations_reject_defects(entry, mutation):
    assert mutation in MUTATIONS
    exposed = False
    for case in CASES:
        data = _case_data(entry, case)
        output, _, _ = _simulate(entry, case, data, mutation)
        exposed |= output != data["expected"]
    assert exposed, (entry, mutation)


@pytest.mark.parametrize("target", TARGETS)
def test_gemv_masked_configs_scope_opengl_index_assertions(tmp_path, target):
    for entry in ENTRIES[::11]:
        config = _config(tmp_path / "work", target, entry)
        assertions = [item.to_json() for item in config.index_range_assertions]
        if target != "opengl":
            assert assertions == []
            continue
        assert [item["expression"] for item in assertions] == list(
            OPENGL_INDEX_RANGE_ASSERTIONS
        )
        assert {(item["minimum"], item["maximum"]) for item in assertions} == {
            (0, SIGNED_MAX)
        }


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------


def _source_tables(text):
    """Tile shapes, element types and mask pairs of gemv_masked.metal's macros."""

    def blocks(helper):
        rows = re.findall(rf"\b{helper}\(name, itype,\s*([\d,\s]+?)\)", text)
        return tuple(tuple(int(part) for part in row.split(",")) for row in rows)

    def types(macro):
        return tuple(
            re.findall(rf"^{macro}\((\w+), (\w+)\);", text, flags=re.MULTILINE)
        )

    def masks(helper):
        pairs = re.findall(
            rf"{helper}\((\w+), (\w+), (\w+), (\w+), name, itype, bm, bn, sm, sn, tm, tn, nc\)",
            text,
        )
        named = {"name": "T", "bool_": "bool_", "nomask": "nomask"}
        return tuple((named[out], named[op]) for out, _, op, _ in pairs)

    return {
        "gemv": blocks("instantiate_gemv"),
        "gemv_t": blocks("instantiate_gemv_t"),
        "types": (types("instantiate_gemv_blocks"), types("instantiate_gemv_t_blocks")),
        "masks": (masks("instantiate_gemv_helper"), masks("instantiate_gemv_t_helper")),
        "nc": tuple(
            re.findall(
                r"instantiate_gemv(?:_t)?_base\(name, itype, bm, bn, sm, sn, tm, tn, (\d)\)",
                text,
            )
        ),
    }


def _kernel_parameters(text, kernel):
    match = re.search(rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", text, re.DOTALL)
    assert match, kernel
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", match.group(1))
    return tuple((name, int(index)) for name, index in found)


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native masked GEMV gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in HEADERS.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # The tables above are gemv_masked.metal's own instantiation lists, and
    # PARAMETERS gemv_masked.h's [[buffer(i)]] signatures.
    tables = _source_tables(source.decode("utf-8"))
    assert tables["gemv"] == GEMV_BLOCKS
    assert tables["gemv_t"] == GEMV_T_BLOCKS
    assert tables["types"] == (DTYPES, DTYPES)
    assert tables["masks"] == (MASKS, MASKS)
    assert tables["nc"] == ("0", "1", "0", "1")
    header = (root / "mlx/backend/metal/kernels/gemv_masked.h").read_text(
        encoding="utf-8"
    )
    for kernel in KERNELS.values():
        assert _kernel_parameters(header, kernel) == PARAMETERS
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("gemv-masked-metal-reference")
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


def _translate(root, work, target, entry):
    """Translate one entry and check it against its pinned artifact."""
    record = _contract()["artifacts"][target][entry]
    translation = translate_project(
        _config(work, target, entry),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    validation = validate_project_report(work / "report.json")
    assert validation["success"] is True, validation
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    assert report["summary"]["failedCount"] == 0
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": record["sha256"],
    }
    assert artifact["generatedSizeBytes"] == record["sizeBytes"]
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == _workgroup(entry)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    descriptor, package = _prepare_native_package(translation, work)
    return descriptor, package, record


def _request(descriptor, package, target, entry, data):
    """Bind every reflected binding by source parameter and build one dispatch."""
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    dtype = INFO[entry]["dtype"]
    values = {
        name: _runtime_value(target, kind, storage)
        for name, (kind, storage) in data["values"].items()
    }
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        source = binding.get("provenance", {}).get("sourceResource")
        assert source is not None, binding
        name = source["parameter"]
        value = values[name]
        layout = binding["scalarLayout"]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
            assert value["encoding"] == FLOAT16_BITS
        inputs[binding["name"]] = value
        if name == "out_vec":
            assert binding["access"] == "read_write", binding
            output = binding["name"]
        else:
            observed.append(binding["name"])
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
    }
    assert output is not None and {"mat", "in_vec", "out_vec"} <= bound <= set(values)
    expected = _runtime_value(target, dtype, data["expected"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: expected},
        {"workgroupCount": data["grid"], "workgroupSize": _workgroup(entry)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, len(observed)


def _check(label, target, entry, actual, data):
    expected = _runtime_value(target, INFO[entry]["dtype"], data["expected"])["values"]
    _assert_identical(label, actual, expected)


# The upstream runner binds buffers by position; indices the kernels do not
# declare get a small zero buffer.
UNBOUND_BYTES = 16


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    info = INFO[entry]
    by_index = {index: name for name, index in PARAMETERS}
    payloads = []
    for index in range(max(by_index) + 1):
        name = by_index.get(index)
        if name is None:
            payloads.append(bytes(UNBOUND_BYTES))
        else:
            payloads.append(_metal_bytes(*data["values"][name]))
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": data["grid"],
                "workgroupSize": _workgroup(entry),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    output = dict(PARAMETERS)["out_vec"]
    for index, payload in enumerate(payloads):
        if index != output:
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    raw = (readback / f"buffer-{output}.bin").read_bytes()
    if info["dtype"] in {"float16", "bfloat16"}:
        words = list(struct.unpack(f"<{len(raw) // 2}H", raw))
    else:
        words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return device, words


def _native_parity(root, target, reference, tmp_path, entry):
    with tempfile.TemporaryDirectory(
        prefix=".current-gemv-masked-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            descriptor, package, record = _translate(root, work, target, entry)
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES:
                    data = _case_data(entry, case)
                    request, output, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    words = _execute(
                        executor, work, case, request, {"out": output}, observed
                    )["out"]["values"]
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work, reference, entry, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        _assert_identical(case, words, upstream_words)
                        upstream = {"device": device, "identicalToTranslation": True}
                    _check(case, target, entry, words, data)
                    evidence.append(
                        {
                            "case": case,
                            **data["shape"],
                            "grid": data["grid"],
                            "batchCount": len(data["batches"]),
                            "writtenElementCount": data["written"],
                            "checkedElementCount": data["size"],
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


@pytest.mark.parametrize("entry", GEMV_ENTRIES)
def test_current_mlx_gemv_masked_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    _native_parity(root, target, metal_reference, tmp_path, entry)


@pytest.mark.parametrize("entry", GEMV_T_ENTRIES)
def test_current_mlx_gemv_t_masked_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    _native_parity(root, target, metal_reference, tmp_path, entry)


def test_ci_requires_native_gemv_masked_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    module = "demos/integrations/mlx/tests/kernels/test_current_gemv_masked.py"
    # Two legs share the module: the first runs everything but the gemv_t
    # native tests, the second only those, so each test runs exactly once.
    selections = {
        "gemv-masked": (
            "Prove current MLX masked GEMV native parity",
            '-k "not test_current_mlx_gemv_t_masked_native_parity"',
        ),
        "gemv-t-masked": (
            "Prove current MLX masked GEMV-T native parity",
            "-k test_current_mlx_gemv_t_masked_native_parity",
        ),
    }
    for group, (name, selection) in selections.items():
        step = current_parity_step(workflow, name, group=group)
        for required in (
            f'{REQUIRE_ENV}: "1"',
            "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
            "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
            "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
            "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
            "pytest -q -n auto",
            f"--basetemp=mlx-current-results/{group}",
            f"--junitxml=mlx-current-results/{group}.xml",
            selection,
            module,
        ):
            assert required in step, (group, required)
        assert "continue-on-error" not in step
    # ``-k`` matches substrings of test names: only the gemv_t native test
    # holds the second leg's expression.
    tests = {name for name in globals() if name.startswith("test_")}
    assert {
        name for name in tests if "test_current_mlx_gemv_t_masked_native_parity" in name
    } == {"test_current_mlx_gemv_t_masked_native_parity"}
    assert "test_current_mlx_gemv_masked_native_parity" in tests
    assert_workflow_triggers(workflow, module)
