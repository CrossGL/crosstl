"""Prove MLX's current wide GEMV kernels natively on every target.

``gemv.metal`` instantiates ``gemv_wide`` and ``gemv_wide_gather`` for
float16 and bfloat16 with two to five vectors per threadgroup and 16 or 32
K lanes per row: 32 entries. ``gemv_wide`` reads function constants 0
(``gemv_wide_has_batch``) and 1 (``gemv_wide_do_axpby``), and ``matmul.cpp``
selects all four combinations; ``gemv_wide_gather`` reads constant 1, which
``gather_mm_wide`` always sets to false. Each of the 80 specializations is
translated and pinned per target: Metal keeps the function constants and
OpenGL keeps them as specialization constants the dispatch sets, while
DirectX bakes each combination into its artifact.

Every specialization runs three cases on ``matmul.cpp``'s threadgroups of
``(32, k_lanes / 8, 1)`` threads and grids of ``(columns, ceil(N / 4),
batches)``: a short one whose second vector chunk is partial, whose last row
block is clamped and whose K needs only the leftover loop, a ragged one whose
two grid columns walk three chunks round-robin (the ``chunk += tgpg.x``
stride) over unrolled blocks plus leftover loads, and an aligned one. A model
replays each lane's clamped rows and vectors, unrolled and leftover loads and
shuffle-down sums and must reach a direct product. Inputs are small integers:
float32 accumulates them exactly in any order, and every result, ``alpha``
and ``beta * bias`` term is exact in the element type, so each target must
produce the model's words. On Metal the translation must also reproduce the
unchanged upstream kernel word for word.
"""

import functools
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
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
    ALPHA,
    BETA,
    GATHER_LAYOUTS,
    GUARD_COUNT,
    HEADERS,
    POISON,
    RESULT_BOUNDS,
    SENTINEL,
    SIGNED_MAX,
    SIMD_SIZE,
    SOURCE,
    SOURCE_SHA256,
    _assert_identical,
    _bias_values,
    _elem_to_loc,
    _exact,
    _vector_values,
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

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMV_WIDE"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/gemv_wide.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# gemv.metal's wide instantiations, in order: instantiate_gemv_wide(name,
# itype, kl) for each (element type, K lanes) row, each over these vectors per
# threadgroup.
KERNELS = ("gemv_wide", "gemv_wide_gather")
DTYPES = (("float16", "half"), ("bfloat16", "bfloat16_t"))
K_LANES = (16, 32)
VECS_PER_TG = (2, 3, 4, 5)
TYPE_ROWS = tuple((dtype, itype, kl) for kl in K_LANES for dtype, itype in DTYPES)
# GemvWide's unrolled loads per lane and rows per threadgroup:
# (32 / k_lanes rows per simdgroup) * (k_lanes / 8 simdgroups).
UNROLL = 8
ROWS_PER_GROUP = 4
CONSTANT_IDS = {"gemv_wide_has_batch": 0, "gemv_wide_do_axpby": 1}

# [[buffer(i)]] parameters of the kernels, as (name, index).
WIDE_PARAMETERS = (
    ("mat", 0),
    ("in_vec", 1),
    ("bias", 2),
    ("out_vec", 3),
    ("in_vec_size", 4),
    ("out_vec_size", 5),
    ("M", 6),
    ("matrix_ld", 7),
    ("vector_ld", 8),
    ("alpha", 9),
    ("beta", 10),
    ("batch_ndim", 11),
    ("batch_shape", 12),
    ("vector_batch_stride", 13),
    ("matrix_batch_stride", 14),
    ("bias_batch_stride", 15),
    ("bias_ld", 16),
    ("bias_fd", 17),
)
GATHER_PARAMETERS = (
    ("mat", 0),
    ("in_vec", 1),
    ("bias", 2),
    ("out_vec", 3),
    ("in_vec_size", 4),
    ("out_vec_size", 5),
    ("M", 6),
    ("matrix_ld", 7),
    ("vector_ld", 8),
    ("batch_ndim", 11),
    ("batch_shape", 12),
    ("index_batch_strides", 13),
    ("vector_batch_ndim", 14),
    ("vector_batch_shape", 15),
    ("vector_batch_stride", 16),
    ("matrix_batch_ndim", 17),
    ("matrix_batch_shape", 18),
    ("matrix_batch_stride", 19),
    ("vec_indices", 20),
    ("mat_indices", 21),
)
SCALAR_TYPES = {
    "in_vec_size": "int32",
    "out_vec_size": "int32",
    "M": "int32",
    "matrix_ld": "int32",
    "vector_ld": "int32",
    "alpha": "float32",
    "beta": "float32",
    "batch_ndim": "int32",
    "bias_ld": "int32",
    "bias_fd": "int32",
    "vector_batch_ndim": "int32",
    "matrix_batch_ndim": "int32",
}
ARRAY_TYPES = {
    "batch_shape": "int32",
    "vector_batch_shape": "int32",
    "matrix_batch_shape": "int32",
    "vector_batch_stride": "int64",
    "matrix_batch_stride": "int64",
    "bias_batch_stride": "int64",
    "index_batch_strides": "int64",
    "vec_indices": "uint32",
    "mat_indices": "uint32",
}

CASES = ("short", "ragged", "aligned")
# Elements between rows, vectors and batches that no case may read; batch
# bases stay multiples of eight elements, as gemv_wide_config requires.
ROW_PAD = 4
VECTOR_PAD = 8
GAP = 7
BIAS_GAP = 3
# (bias_ld, bias_fd): a broadcast row, a strided and a contiguous bias.
BIAS_LAYOUTS = {"short": (0, 1), "ragged": "strided", "aligned": "contiguous"}
FLOAT32_EXACT = 2**24

# Expressions OpenGL may evaluate in 32 bits: the output, matrix row and
# bias offsets of GemvWide::run, its vec4 loads, and the batch and gather
# prologues. The model shows each names an element inside its buffer.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "run": (
        "int64(tid.z) * M * out_vec_size",
        "mat + int64(row) * matrix_ld",
        "int64(row) * matrix_ld",
        "w4 + base + i * {kl} + k_lane",
        "w4 + idx",
        "x4 + x_off4[v] + base + i * {kl} + k_lane",
        "x4 + x_off4[v] + idx",
        "int64(vec0 + v) * bias_ld + out_row * bias_fd",
        "out_idx",
        "out_vec + out_idx",
    ),
    "batch": (
        "tid.z * buffer_load(vector_batch_stride, 0)",
        "in_vec + tid.z * buffer_load(vector_batch_stride, 0)",
        "tid.z * buffer_load(matrix_batch_stride, 0)",
        "mat + tid.z * buffer_load(matrix_batch_stride, 0)",
        "tid.z * buffer_load(bias_batch_stride, 0)",
        "bias + tid.z * buffer_load(bias_batch_stride, 0)",
    ),
    "gather": (
        "buffer_load(index_batch_strides, 0) * tid.z",
        "buffer_load(index_batch_strides, batch_ndim) * tid.z",
        "batch_offsets.x",
        "batch_offsets.y",
        "indx_vec * buffer_load(vector_batch_stride, 0)",
        "in_vec + indx_vec * buffer_load(vector_batch_stride, 0)",
        "indx_mat * buffer_load(matrix_batch_stride, 0)",
        "mat + indx_mat * buffer_load(matrix_batch_stride, 0)",
    ),
}


def _instantiations():
    """``(entry, info)`` in gemv.metal's instantiation order."""
    for kernel in KERNELS:
        for dtype, _, kl in TYPE_ROWS:
            for nv in VECS_PER_TG:
                yield f"{kernel}_{dtype}_nv{nv}_kl{kl}", {
                    "kernel": kernel,
                    "gather": kernel == "gemv_wide_gather",
                    "dtype": dtype,
                    "nv": nv,
                    "kl": kl,
                }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _combinations(entry):
    """Function constants ``matmul.cpp`` sets for the entry."""
    if INFO[entry]["gather"]:
        return ({"gemv_wide_do_axpby": False},)
    return tuple(
        {"gemv_wide_has_batch": has_batch, "gemv_wide_do_axpby": do_axpby}
        for has_batch in (False, True)
        for do_axpby in (False, True)
    )


def _variant_key(constants):
    """The pipeline hash suffix ``matmul.cpp`` gives the specialization."""
    axpby = int(constants["gemv_wide_do_axpby"])
    if "gemv_wide_has_batch" in constants:
        return f"nc{int(constants['gemv_wide_has_batch'])}_axpby{axpby}"
    return f"axpby{axpby}"


VARIANTS = tuple(
    (entry, _variant_key(constants), constants)
    for entry in ENTRIES
    for constants in _combinations(entry)
)


def _workgroup(entry):
    return [SIMD_SIZE, INFO[entry]["kl"] // 8, 1]


def _index_assertions(entry):
    info = INFO[entry]
    run = tuple(
        expression.format(kl=info["kl"])
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS["run"]
    )
    return run + OPENGL_INDEX_RANGE_ASSERTIONS["gather" if info["gather"] else "batch"]


def _config(directory, target, entry, constants):
    """The translation of one specialization, as the gate and contract run it."""
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: tuple(_workgroup(entry))}},
        specialization_constants={
            str(CONSTANT_IDS[name]): value for name, value in constants.items()
        },
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (_index_assertions(entry) if target == "opengl" else ())
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


def _round8(value):
    return -(-value // 8) * 8


def _shape(entry, case):
    """``(M, N, K, grid columns, matrix_ld, vector_ld)`` of one case."""
    info = INFO[entry]
    nv, kl = info["nv"], info["kl"]
    block = 4 * kl * UNROLL
    if case == "short":
        k = 4 * (kl - 3)
        return nv + 1, 6, k, 1, k + ROW_PAD, k + VECTOR_PAD
    if case == "ragged":
        k = block + 4 * (kl + 5)
        return 2 * nv + 1, 11, k, 2, k + ROW_PAD, k + 2 * ROW_PAD
    k = 2 * block
    return 2 * nv, 8, k, 2, k, k


def _bias_strides(entry, case):
    m, n = _shape(entry, case)[:2]
    layout = BIAS_LAYOUTS[case]
    if layout == "strided":
        return 2 * n + 3, 2
    if layout == "contiguous":
        return n, 1
    return layout


def _batching(entry, constants, case):
    """Batch metadata and per-batch ``(vector, matrix, bias)`` element offsets."""
    info = INFO[entry]
    m, n, k, _, ld, vld = _shape(entry, case)
    vs = _round8(m * vld + GAP)
    ms = _round8(n * ld + GAP)
    if info["gather"]:
        layout = GATHER_LAYOUTS[case]
        shape = layout["batch_shape"]
        ndim = len(shape)
        strides = layout["index_batch_strides"]
        vector_shape = layout["vector_batch_shape"]
        matrix_shape = layout["matrix_batch_shape"]
        vector_strides = [vs] if len(vector_shape) == 1 else [vs, 2 * vs]
        matrix_strides = {1: [ms], 2: [ms, 3 * ms]}[len(matrix_shape)]
        if matrix_shape == [1, 2]:
            # A unit outer extent: elem_to_loc stops once the index is spent.
            matrix_strides = [5 * ms, ms]
        count = 1
        for extent in shape:
            count *= extent
        batches = []
        for z in range(count):
            if ndim > 1:
                loc_v, loc_m = _elem_to_loc(z, shape, (strides[:ndim], strides[ndim:]))
            else:
                loc_v, loc_m = strides[0] * z, strides[ndim] * z
            vector = layout["vec_indices"][loc_v]
            matrix = layout["mat_indices"][loc_m]
            if len(vector_shape) > 1:
                (v_offset,) = _elem_to_loc(vector, vector_shape, (vector_strides,))
            else:
                v_offset = vector * vector_strides[0]
            if len(matrix_shape) > 1:
                (m_offset,) = _elem_to_loc(matrix, matrix_shape, (matrix_strides,))
            else:
                m_offset = matrix * matrix_strides[0]
            batches.append({"in_vec": v_offset, "mat": m_offset})
        vectors = 1
        for extent in vector_shape:
            vectors *= extent
        matrices = 1
        for extent in matrix_shape:
            matrices *= extent
        # gather_mm_wide passes whole shapes and strides: the batch prefix
        # the kernel reads, then out's [M, N], a's [M, K] with strides
        # [lda, 1] and the transposed b's [K, N] with strides [1, ldb].
        metadata = {
            "batch_ndim": ndim,
            "batch_shape": shape + [m, n],
            "index_batch_strides": strides,
            "vector_batch_ndim": len(vector_shape),
            "vector_batch_shape": vector_shape + [m, k],
            "vector_batch_stride": vector_strides + [vld, 1],
            "matrix_batch_ndim": len(matrix_shape),
            "matrix_batch_shape": matrix_shape + [k, n],
            "matrix_batch_stride": matrix_strides + [1, ld],
            "vec_indices": layout["vec_indices"],
            "mat_indices": layout["mat_indices"],
        }
        offsets = {
            "in_vec": [
                (
                    _elem_to_loc(v, vector_shape, (vector_strides,))[0]
                    if len(vector_shape) > 1
                    else v * vs
                )
                for v in range(vectors)
            ],
            "mat": [
                (
                    _elem_to_loc(index, matrix_shape, (matrix_strides,))[0]
                    if len(matrix_shape) > 1
                    else index * ms
                )
                for index in range(matrices)
            ],
        }
        return metadata, batches, offsets
    bias_ld, bias_fd = _bias_strides(entry, case)
    bs = (m - 1) * bias_ld + (n - 1) * bias_fd + 1 + BIAS_GAP
    if constants["gemv_wide_has_batch"]:
        # A non-contiguous [2, 2] batch: the vector broadcasts over the outer
        # extent and the matrices and biases are permuted.
        shape = [2, 2]
        vector_strides, matrix_strides, bias_strides = (
            [0, vs],
            [ms, 2 * ms],
            [2 * bs, bs],
        )
    else:
        shape = [2]
        vector_strides, matrix_strides, bias_strides = [vs], [ms], [bs]
    count = shape[0] * (shape[1] if len(shape) > 1 else 1)
    batches = []
    for z in range(count):
        if len(shape) > 1:
            v_offset, m_offset, b_offset = _elem_to_loc(
                z, shape, (vector_strides, matrix_strides, bias_strides)
            )
        else:
            v_offset, m_offset, b_offset = (
                z * vector_strides[0],
                z * matrix_strides[0],
                z * bias_strides[0],
            )
        batches.append({"in_vec": v_offset, "mat": m_offset, "bias": b_offset})
    metadata = {
        "batch_ndim": len(shape),
        "batch_shape": shape,
        "vector_batch_stride": vector_strides,
        "matrix_batch_stride": matrix_strides,
        "bias_batch_stride": bias_strides,
        "bias_ld": bias_ld,
        "bias_fd": bias_fd,
    }
    offsets = {
        "in_vec": sorted({batch["in_vec"] for batch in batches}),
        "mat": [batch["mat"] for batch in batches],
    }
    return metadata, batches, offsets


def _grid(entry, constants, case):
    m, n, _, columns = _shape(entry, case)[:4]
    return [columns, -(-n // ROWS_PER_GROUP), len(_batching(entry, constants, case)[1])]


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _row(draws, vectors, bound):
    """A matrix row whose dot product with every vector is nonzero and at
    most ``bound`` in magnitude: each element is the first of a rotated
    ``2, 1, -1, -2`` that keeps every running sum inside the bound, else 0."""
    totals = [0] * len(vectors)
    row = []
    choices = (2, 1, -1, -2)
    for k in range(len(vectors[0])):
        start = _below(draws, len(choices))
        chosen = 0
        for step in range(len(choices)):
            value = choices[(start + step) % len(choices)]
            if all(
                abs(total + value * vector[k]) <= bound
                for total, vector in zip(totals, vectors)
            ):
                chosen = value
                break
        row.append(chosen)
        totals = [total + chosen * vector[k] for total, vector in zip(totals, vectors)]
    for index in range(len(vectors)):
        if totals[index]:
            continue
        for k in range(len(row)):
            moved = False
            for delta in (1, -1):
                value = row[k] + delta
                if abs(value) > 2:
                    continue
                trial = [t + delta * vector[k] for t, vector in zip(totals, vectors)]
                if all(abs(t) <= bound for t in trial) and all(
                    trial[j] for j in range(len(vectors)) if totals[j] or j == index
                ):
                    row[k], totals, moved = value, trial, True
                    break
            if moved:
                break
        assert totals[index], "no element can make the dot product nonzero"
    assert all(totals) and all(abs(total) <= bound for total in totals)
    return row


def _case_data(entry, constants, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    return _cached_case_data(entry, _variant_key(constants), case)


@functools.lru_cache(maxsize=None)
def _cached_case_data(entry, key, case):
    (constants,) = (c for e, k, c in VARIANTS if e == entry and k == key)
    info = INFO[entry]
    dtype = info["dtype"]
    do_axpby = constants["gemv_wide_do_axpby"]
    m, n, k, _, ld, vld = _shape(entry, case)
    metadata, batches, offsets = _batching(entry, constants, case)
    draws = _draws("gemv_wide", entry, _variant_key(constants), case)
    vector_length = max(offsets["in_vec"]) + m * vld + GAP
    matrix_length = max(offsets["mat"]) + n * ld + GAP
    in_vec = [POISON[index % len(POISON)] for index in range(vector_length)]
    mat = [POISON[index % len(POISON)] for index in range(matrix_length)]
    vectors = {}
    for offset in offsets["in_vec"]:
        rows = []
        for vector in range(m):
            values = _vector_values(dtype, draws, k)
            start = offset + vector * vld
            in_vec[start : start + k] = values
            rows.append(values)
        vectors[offset] = rows
    built = set()
    bound = RESULT_BOUNDS[dtype]
    for batch in batches:
        # Each batch's matrix holds rows whose dot products with that batch's
        # vectors stay exact.
        assert batch["mat"] not in built, (entry, case)
        built.add(batch["mat"])
        for row in range(n):
            start = batch["mat"] + row * ld
            mat[start : start + k] = _row(draws, vectors[batch["in_vec"]], bound)
    for offset in offsets["mat"]:
        if offset not in built:
            for row in range(n):
                start = offset + row * ld
                mat[start : start + k] = [_signed(draws, 2) for _ in range(k)]
    if do_axpby:
        bias_length = max(batch["bias"] for batch in batches) + (
            (m - 1) * metadata["bias_ld"] + (n - 1) * metadata["bias_fd"] + 1 + BIAS_GAP
        )
        bias = _bias_values(dtype, draws, bias_length)
    else:
        bias = [POISON[index % len(POISON)] for index in range(GUARD_COUNT)]
    total = len(batches) * m * n + GUARD_COUNT
    initial = [SENTINEL] * total
    expected = list(initial)
    for z, batch in enumerate(batches):
        for vector in range(m):
            for row in range(n):
                value = sum(
                    mat[batch["mat"] + row * ld + index]
                    * in_vec[batch["in_vec"] + vector * vld + index]
                    for index in range(k)
                )
                if do_axpby:
                    term = bias[
                        batch["bias"]
                        + vector * metadata["bias_ld"]
                        + row * metadata["bias_fd"]
                    ]
                    value = ALPHA * value + BETA * term
                assert value and _exact(dtype, value), (entry, case, z, vector, row)
                expected[(z * m + vector) * n + row] = value
    parameters = dict(GATHER_PARAMETERS if info["gather"] else WIDE_PARAMETERS)
    scalars = {
        "in_vec_size": k,
        "out_vec_size": n,
        "M": m,
        "matrix_ld": ld,
        "vector_ld": vld,
        **{name: value for name, value in metadata.items() if name in SCALAR_TYPES},
    }
    arrays = {
        name: list(value) for name, value in metadata.items() if name in ARRAY_TYPES
    }
    if not info["gather"]:
        scalars["alpha"], scalars["beta"] = (ALPHA, BETA) if do_axpby else (1.0, 0.0)
        if not do_axpby:
            # matmul.cpp binds no bias, scalars or strides without axpby; any
            # value works.
            scalars["bias_ld"] = scalars["bias_fd"] = 0
            arrays["bias_batch_stride"] = [0] * len(arrays["bias_batch_stride"])
    values = {
        "mat": (dtype, mat),
        "in_vec": (dtype, in_vec),
        "bias": (dtype, bias),
        "out_vec": (dtype, initial),
        **{name: (SCALAR_TYPES[name], [value]) for name, value in scalars.items()},
        **{name: (ARRAY_TYPES[name], value) for name, value in arrays.items()},
    }
    assert set(values) == set(parameters), set(values) ^ set(parameters)
    return {
        "values": values,
        "initial": initial,
        "expected": expected,
        "grid": _grid(entry, constants, case),
        "shape": {"M": m, "N": n, "K": k, "matrixLd": ld, "vectorLd": vld},
        "batches": batches,
        "written": len(batches) * m * n,
        "size": total,
    }


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


MUTATIONS = {
    "leftover": "drops the leftover vec4 loads",
    "chunks": "walks only each column's first chunk",
    "lanes": "reduces only half of each row's K lanes",
    "batch": "addresses every batch as the first",
    "bias": "drops beta * bias",
}


def _simulate(entry, constants, case, data, mutation=None):
    """GemvWide::run lane by lane; returns outputs, accesses and paths."""
    info = INFO[entry]
    nv, kl = info["nv"], info["kl"]
    m, n, k, columns, ld, vld = _shape(entry, case)
    do_axpby = constants["gemv_wide_do_axpby"] and not info["gather"]
    storage = {name: values for name, (_, values) in data["values"].items()}
    bias_ld = storage["bias_ld"][0] if do_axpby else 0
    bias_fd = storage["bias_fd"][0] if do_axpby else 0
    accesses = {name: [] for name in ("mat", "in_vec", "bias")}
    paths = set()
    output = list(data["initial"])
    stores = Counter()

    def load(name, index):
        accesses[name].append(index)
        values = storage[name]
        if not 0 <= index < len(values):
            return POISON[0]  # a defective kernel reads past its buffer
        return values[index]

    def dot4(mat_base, w_index, vec_base, x_index):
        total = 0
        for lane in range(4):
            total += load("mat", mat_base + 4 * w_index + lane) * load(
                "in_vec", vec_base + 4 * x_index + lane
            )
        return total

    n_v4 = k // 4
    n_main = n_v4 - n_v4 % (kl * UNROLL)
    n_chunks = -(-m // nv)
    rows_per_simdgroup = SIMD_SIZE // kl
    groups = -(-n // ROWS_PER_GROUP)
    first = data["batches"][0]
    for z, batch in enumerate(data["batches"]):
        place = first if mutation == "batch" else batch
        for column in range(columns):
            for group in range(groups):
                chunk = column
                while chunk < n_chunks:
                    vec0 = chunk * nv
                    if vec0 + nv > m:
                        paths.add("partial-chunk")
                    x_off4 = [min(vec0 + v, m - 1) * (vld // 4) for v in range(nv)]
                    for simdgroup in range(kl // 8):
                        for sg_row in range(rows_per_simdgroup):
                            out_row = (
                                group * ROWS_PER_GROUP
                                + rows_per_simdgroup * simdgroup
                                + sg_row
                            )
                            row = min(out_row, n - 1)
                            if row != out_row:
                                paths.add("clamped-row")
                            mat_base = place["mat"] + row * ld
                            lanes = []
                            for k_lane in range(kl):
                                result = [0] * nv
                                for base in range(0, n_main, kl * UNROLL):
                                    paths.add("unrolled")
                                    for v in range(nv):
                                        acc = 0
                                        for i in range(UNROLL):
                                            acc += dot4(
                                                mat_base,
                                                base + i * kl + k_lane,
                                                place["in_vec"],
                                                x_off4[v] + base + i * kl + k_lane,
                                            )
                                        result[v] += acc
                                if mutation != "leftover":
                                    for idx in range(n_main + k_lane, n_v4, kl):
                                        paths.add("leftover")
                                        for v in range(nv):
                                            result[v] += dot4(
                                                mat_base,
                                                idx,
                                                place["in_vec"],
                                                x_off4[v] + idx,
                                            )
                                assert all(
                                    abs(value) < FLOAT32_EXACT for value in result
                                )
                                lanes.append(result)
                            # Halving shuffle-down sums: lane 0 of the row
                            # gathers its k_lanes lanes.
                            offset = kl // (4 if mutation == "lanes" else 2)
                            while offset >= 1:
                                lanes = [
                                    [
                                        value
                                        + (
                                            lanes[lane + offset][v]
                                            if lane + offset < kl
                                            else 0
                                        )
                                        for v, value in enumerate(lanes[lane])
                                    ]
                                    for lane in range(kl)
                                ]
                                offset //= 2
                            if out_row >= n:
                                continue
                            for v in range(nv):
                                if vec0 + v >= m:
                                    continue
                                value = lanes[0][v]
                                if do_axpby:
                                    term = load(
                                        "bias",
                                        place["bias"]
                                        + (vec0 + v) * bias_ld
                                        + out_row * bias_fd,
                                    )
                                    if mutation == "bias":
                                        term = 0
                                    value = ALPHA * value + BETA * term
                                index = (z * m + vec0 + v) * n + out_row
                                stores[index] += 1
                                output[index] = value
                    if mutation == "chunks":
                        break
                    if chunk + columns < n_chunks:
                        paths.add("strided-chunk")
                    chunk += columns
    if mutation is None:
        assert set(stores.values()) == {1} and len(stores) == data["written"]
    return output, accesses, paths


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _output_count(entry, constants, case):
    m, n = _shape(entry, case)[:2]
    return len(_batching(entry, constants, case)[1]) * m * n + GUARD_COUNT


def _element_count():
    return sum(
        _output_count(entry, constants, case)
        for entry, _, constants in VARIANTS
        for case in CASES
    )


def _expected_validation_contract():
    return {
        "model": (
            "GemvWide::run lane by lane: clamped rows and vectors, the "
            "round-robin chunk loop, unrolled and leftover vec4 loads, "
            "halving shuffle-down sums and guarded stores"
        ),
        "comparison": "exact words",
        "workgroupSize": "[32, k_lanes / 8, 1]",
        "grid": "[columns, ceil(N / 4), batches]",
        "functionConstants": {
            "ids": CONSTANT_IDS,
            "gemv_wide": "every (has_batch, do_axpby) combination",
            "gemv_wide_gather": "do_axpby = false, as gather_mm_wide sets it",
        },
        "cases": {
            "M": {"short": "nv + 1", "ragged": "2 * nv + 1", "aligned": "2 * nv"},
            "N": {"short": 6, "ragged": 11, "aligned": 8},
            "K": {
                "short": "4 * (k_lanes - 3)",
                "ragged": "4 * (8 * k_lanes + k_lanes + 5)",
                "aligned": "4 * 16 * k_lanes",
            },
            "columns": {"short": 1, "ragged": 2, "aligned": 2},
            "matrixLdPadding": {"short": ROW_PAD, "ragged": ROW_PAD, "aligned": 0},
            "vectorLdPadding": {
                "short": VECTOR_PAD,
                "ragged": 2 * ROW_PAD,
                "aligned": 0,
            },
        },
        "batching": {
            "contiguous": {"batchShape": [2], "strides": "one per batch"},
            "nonContiguous": {
                "batchShape": [2, 2],
                "vectorStrides": "[0, vector]",
                "matrixStrides": "[matrix, 2 * matrix]",
                "biasStrides": "[2 * bias, bias]",
            },
            "gather": GATHER_LAYOUTS,
            "batchAlignment": 8,
        },
        "axpby": {
            "alpha": ALPHA,
            "beta": BETA,
            "biasStrides": {
                "short": [0, 1],
                "ragged": "[2 * N + 3, 2]",
                "aligned": "[N, 1]",
            },
        },
        "inputs": {
            "vectors": "nonzero integers in [-2, 2]",
            "matrices": (
                "integers in [-2, 2] whose dot product with each of the "
                "batch's vectors is nonzero and at most the element type's bound"
            ),
            "resultBounds": {key: RESULT_BOUNDS[key] for key, _ in DTYPES},
            "bias": "nonzero integers 2 mod 4, so beta * bias is odd",
            "poison": list(POISON),
        },
        "guards": {
            "output": "every element starts at the sentinel, which no result equals",
            "sentinel": SENTINEL,
            "trailingOutputGuards": GUARD_COUNT,
        },
        "openglIndexRangeAssertions": {
            group: [
                {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                for expression in expressions
            ]
            for group, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
        },
        "softwareSubgroupWidth": {"targets": ["opengl", "directx"], "width": SIMD_SIZE},
        "directx": {"relativeWaveShuffleOutOfRange": "self"},
        "elementTypes": {
            target: {dtype: _element_type(target, dtype) for dtype, _ in DTYPES}
            for target in TARGETS
        },
        "dispatchCountByTarget": len(VARIANTS) * len(CASES),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "specializationCount": len(VARIANTS),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(VARIANTS),
        "dispatchCountByTarget": len(VARIANTS) * len(CASES),
        "checkedElementCountByTarget": _element_count(),
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def expected_contract(artifacts=None):
    """Return the contract this module pins, for review and regeneration."""
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-gemv-wide-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": HEADERS,
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        "artifacts": {
            target: {
                entry: {
                    key: artifacts[target][entry][key]
                    for item, key, _ in VARIANTS
                    if item == entry
                }
                for entry in ENTRIES
            }
            for target in TARGETS
        },
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(),
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_current_mlx_gemv_wide_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract(contract["artifacts"])
    for target in TARGETS:
        records = [
            contract["artifacts"][target][entry][key] for entry, key, _ in VARIANTS
        ]
        for record in records:
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
        hashes = {record["sha256"] for record in records}
        if target == "directx":
            # DirectX bakes each function-constant combination.
            assert len(hashes) == len(VARIANTS)
        else:
            # Metal and OpenGL keep the constants, so one artifact serves
            # every specialization of an entry.
            assert len(hashes) == len(ENTRIES)
            for entry in ENTRIES:
                assert (
                    len(
                        {
                            record["sha256"]
                            for record in contract["artifacts"][target][entry].values()
                        }
                    )
                    == 1
                )


def test_gemv_wide_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "gemv_wide.current-tree.translation.json" in readme
    assert len(ENTRIES) == 32 and len(VARIANTS) == 80
    assert len({(entry, key) for entry, key, _ in VARIANTS}) == len(VARIANTS)
    assert {key for _, key, _ in VARIANTS} == {
        "nc0_axpby0",
        "nc0_axpby1",
        "nc1_axpby0",
        "nc1_axpby1",
        "axpby0",
    }
    from demos.integrations.mlx.tests.kernels import test_current_gemv as gemv

    assert gemv.WIDE_ENTRY_COUNT == len(ENTRIES)
    assert not set(ENTRIES) & set(gemv.ENTRIES)


@pytest.mark.parametrize("entry", ENTRIES)
def test_gemv_wide_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    for constants in _combinations(entry):
        for case in CASES:
            m, n, k, columns, ld, vld = _shape(entry, case)
            # gemv_wide_config: K % 4 == 0, 4-aligned leading dimensions and
            # 8-aligned batch bases.
            assert m > 1 and n > 1 and k % 4 == 0
            assert ld % 4 == 0 and vld % 4 == 0 and ld >= k and vld >= k
            metadata, batches, offsets = _batching(entry, constants, case)
            for batch in batches:
                assert batch["mat"] % 8 == 0 and batch["in_vec"] % 8 == 0
            for name, ndim in (
                ("vector_batch_stride", "vector_batch_ndim"),
                ("matrix_batch_stride", "matrix_batch_ndim"),
            ):
                # Only the batch prefix: gather_mm_wide appends a's and b's
                # own [lda, 1] and [1, ldb].
                width = metadata[ndim if info["gather"] else "batch_ndim"]
                strides = metadata[name][:width]
                assert all(stride % 4 == 0 for stride in strides), (name, strides)
            grid = _grid(entry, constants, case)
            assert grid[1] * ROWS_PER_GROUP >= n > (grid[1] - 1) * ROWS_PER_GROUP
            n_chunks = -(-m // info["nv"])
            assert columns <= n_chunks
    short = _shape(entry, "short")
    ragged = _shape(entry, "ragged")
    kl = info["kl"]
    # The short K never fills an unrolled block; the ragged one fills one
    # block and leaves leftover loads for every lane, some twice.
    assert short[2] // 4 < kl * UNROLL
    assert ragged[2] // 4 - kl * UNROLL > kl


MODEL_VARIANTS = tuple(
    (entry, key, constants)
    for entry, key, constants in VARIANTS
    if key in {"nc1_axpby1", "axpby0"}
    and (INFO[entry]["nv"] in {2, 5} or INFO[entry]["dtype"] == "float16")
)


@pytest.mark.parametrize(
    "entry,key,constants",
    MODEL_VARIANTS,
    ids=[f"{e}-{k}" for e, k, _ in MODEL_VARIANTS],
)
def test_gemv_wide_model_matches_dot_products(entry, key, constants):
    paths = set()
    for case in CASES:
        data = _case_data(entry, constants, case)
        output, accesses, used = _simulate(entry, constants, case, data)
        assert output == data["expected"], (case, key)
        paths |= used
        # Every load the asserted offsets form lies inside its buffer.
        for name, indices in accesses.items():
            if indices:
                length = len(data["values"][name][1])
                assert 0 <= min(indices) and max(indices) < length <= SIGNED_MAX
    assert paths == {
        "partial-chunk",
        "clamped-row",
        "unrolled",
        "leftover",
        "strided-chunk",
    }, paths


@pytest.mark.parametrize("dtype", [dtype for dtype, _ in DTYPES])
def test_gemv_wide_inputs_keep_every_result_exact(dtype):
    for entry, _, constants in VARIANTS:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in CASES:
            data = _case_data(entry, constants, case)
            written = data["expected"][: data["written"]]
            assert SENTINEL not in written and 0 not in written
            assert all(_exact(dtype, value) for value in written)
            assert data["expected"][data["written"] :] == [SENTINEL] * GUARD_COUNT


@pytest.mark.parametrize(
    "entry,key,mutation",
    (
        ("gemv_wide_float16_nv3_kl16", "nc1_axpby1", "leftover"),
        ("gemv_wide_bfloat16_nv2_kl32", "nc0_axpby0", "chunks"),
        ("gemv_wide_float16_nv5_kl32", "nc0_axpby1", "lanes"),
        ("gemv_wide_bfloat16_nv4_kl16", "nc1_axpby0", "batch"),
        ("gemv_wide_float16_nv2_kl16", "nc1_axpby1", "bias"),
        ("gemv_wide_gather_bfloat16_nv3_kl32", "axpby0", "batch"),
        ("gemv_wide_gather_float16_nv4_kl16", "axpby0", "chunks"),
    ),
)
def test_gemv_wide_expectations_reject_defects(entry, key, mutation):
    assert mutation in MUTATIONS
    (constants,) = (c for e, k, c in VARIANTS if e == entry and k == key)
    exposed = False
    for case in CASES:
        data = _case_data(entry, constants, case)
        output, _, _ = _simulate(entry, constants, case, data, mutation)
        exposed |= output != data["expected"]
    assert exposed, (entry, key, mutation)


@pytest.mark.parametrize("target", TARGETS)
def test_gemv_wide_configs_scope_target_options(tmp_path, target):
    for entry, key, constants in VARIANTS[::7]:
        config = _config(tmp_path / "work", target, entry, constants)
        assertions = [item.to_json() for item in config.index_range_assertions]
        assert config.specialization_constants == {
            str(CONSTANT_IDS[name]): value for name, value in constants.items()
        }
        if target != "opengl":
            assert assertions == []
            continue
        assert [item["expression"] for item in assertions] == list(
            _index_assertions(entry)
        )
        assert f"i * {INFO[entry]['kl']}" in " ".join(_index_assertions(entry))
        assert {(item["minimum"], item["maximum"]) for item in assertions} == {
            (0, SIGNED_MAX)
        }


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------


def _source_tables(text):
    """Element types, K lanes and vector counts of gemv.metal's wide macros."""

    def rows(macro):
        return tuple(
            (dtype, itype, int(kl))
            for dtype, itype, kl in re.findall(
                rf"^{macro}\((\w+), (\w+), (\d+)\);", text, flags=re.MULTILINE
            )
        )

    def vectors(helper):
        return tuple(
            int(nv) for nv in re.findall(rf"\b{helper}\(name, itype, (\d+), kl\)", text)
        )

    return {
        "gemv_wide": rows("instantiate_gemv_wide"),
        "gemv_wide_gather": rows("instantiate_gemv_wide_gather"),
        "gemv_wide_vectors": vectors("instantiate_gemv_wide_helper"),
        "gemv_wide_gather_vectors": vectors("instantiate_gemv_wide_gather_helper"),
    }


def _kernel_parameters(text, kernel):
    match = re.search(rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", text, re.DOTALL)
    assert match, kernel
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
    return tuple((name, int(index)) for name, index in found)


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native wide GEMV gate")
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
    # The tables above are gemv.metal's own instantiation lists, and the
    # parameter tuples gemv.h's [[buffer(i)]] signatures.
    tables = _source_tables(source.decode("utf-8"))
    assert tables["gemv_wide"] == tables["gemv_wide_gather"] == TYPE_ROWS
    assert tables["gemv_wide_vectors"] == tables["gemv_wide_gather_vectors"]
    assert tables["gemv_wide_vectors"] == VECS_PER_TG
    header = (root / "mlx/backend/metal/kernels/gemv.h").read_text(encoding="utf-8")
    assert _kernel_parameters(header, "gemv_wide") == WIDE_PARAMETERS
    assert _kernel_parameters(header, "gemv_wide_gather") == GATHER_PARAMETERS
    for name, identifier in CONSTANT_IDS.items():
        assert f"constant bool {name} [[function_constant({identifier})]];" in header
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("gemv-wide-metal-reference")
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


def _translate(root, work, target, entry, constants, record):
    """Translate one specialization and check it against its pinned artifact."""
    translation = translate_project(
        _config(work, target, entry, constants),
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
    return descriptor, package


def _request(descriptor, package, target, entry, data):
    """Bind every reflected binding by source parameter and build one dispatch."""
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    dtype = INFO[entry]["dtype"]
    values = {
        name: _runtime_value(target, kind, storage)
        for name, (kind, storage) in data["values"].items()
    }
    inputs, observed, output, generated = {}, [], None, []
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            # HLSL has no threadgroups_per_grid builtin: the chunk loop's
            # tgpg.x comes from a constant buffer the loader fills from the
            # dispatch.
            assert provenance["executionInput"]["valueSource"] == (
                "dispatch.workgroupCount"
            ), binding
            generated.append(binding["name"])
            continue
        source = provenance.get("sourceResource")
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
    assert len(generated) == (1 if target == "directx" else 0), generated
    bound = {
        binding["provenance"]["sourceResource"]["parameter"]
        for binding in descriptor["bindings"]
        if binding["name"] not in generated
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
    derived = [
        item.value.values
        for item in request.execution_plan.resource_bindings
        if item.value is not None and "executionInput" in item.value.metadata
    ]
    assert derived == [list(data["grid"])] * len(generated)
    return request, output, len(observed)


def _upstream(work, reference, entry, constants, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    info = INFO[entry]
    parameters = dict(GATHER_PARAMETERS if info["gather"] else WIDE_PARAMETERS)
    by_index = {index: name for name, index in parameters.items()}
    files, payloads = [], {}
    for index in range(max(by_index) + 1):
        name = by_index.get(index)
        # gemv_wide_gather declares no buffers 9 and 10.
        payload = _metal_bytes(*data["values"][name]) if name is not None else bytes(4)
        payloads[index] = payload
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
                "boolConstants": [
                    {"index": CONSTANT_IDS[name], "value": value}
                    for name, value in constants.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    output = parameters["out_vec"]
    for index, payload in payloads.items():
        if index != output:
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    raw = (readback / f"buffer-{output}.bin").read_bytes()
    return device, list(struct.unpack(f"<{len(raw) // 2}H", raw))


@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_current_mlx_gemv_wide_native_parity(
    current_mlx, metal_reference, tmp_path, entry, key, constants
):
    root, target = current_mlx
    record = _contract()["artifacts"][target][entry][key]
    with tempfile.TemporaryDirectory(prefix=".current-gemv-wide-", dir=root) as path:
        work = Path(path)
        try:
            descriptor, package = _translate(
                root, work, target, entry, constants, record
            )
            if target != "directx":
                assert {
                    int(item["id"]): item["value"]
                    for item in descriptor["specializationConstants"]
                } == {CONSTANT_IDS[name]: value for name, value in constants.items()}
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES:
                    data = _case_data(entry, constants, case)
                    request, output, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    words = _execute(
                        executor, work, case, request, {"out": output}, observed
                    )["out"]["values"]
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work, metal_reference, entry, constants, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        _assert_identical(case, words, upstream_words)
                        upstream = {"device": device, "identicalToTranslation": True}
                    expected = _runtime_value(
                        target, INFO[entry]["dtype"], data["expected"]
                    )["values"]
                    _assert_identical(case, words, expected)
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
                        "specialization": key,
                        "constants": constants,
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


def test_ci_requires_native_gemv_wide_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX wide GEMV native parity", group="gemv"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/gemv-wide",
        "--junitxml=mlx-current-results/gemv-wide.xml",
        "demos/integrations/mlx/tests/kernels/test_current_gemv_wide.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_gemv_wide.py"
    )
