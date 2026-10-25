"""Storage and launch contracts for MLX's gathered steel GEMM kernels.

``GatherMM::eval_gpu`` in ``gather_mm.inc`` routes gathered matrix products as
matmul.cpp does on a device without NAX and without ``gemv_wide``:

* one output row with sorted right indices: ``gather_mm_rhs``, which first runs
  ``gather_mm_offsets`` to find each group's first row, then
  ``steel_gather_mm_rhs`` over every row of the broadcast left operand;
* one output row or column otherwise: ``gather_mv`` (``gather_mv_layout``);
* everything else: ``gather_mm`` with ``GEMM_TPARAM_MACRO``'s medium-device
  64x64x16 tile and 2x2 simdgroups (``steel_gather_mm``).

The entry names carry the function constants as matmul.cpp's hash names do.
Each plan reads the uint32 indices, or the int32 group offsets, the kernel
will read and proves every gathered operand lies inside its buffer.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host import (
    gather_mv_layout,
    gemm_layout,
    gemv_layout,
)
from demos.integrations.mlx.portable_host.gather_mv_layout import location
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.tests.kernels import gemm_gather_model as model

SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.metal"
OFFSETS_SOURCE = "mlx/backend/metal/kernels/gather_mm_offsets.metal"
OFFSETS_ENTRY = "gather_mm_offsets"
SIMD_SIZE = 32
MAX_ELEMENTS = gemv_layout.MAX_ELEMENTS
SIGNED_MAX = gemv_layout.SIGNED_MAX
MAX_GROUPS = gemv_layout.MAX_GROUPS
MAX_BATCH_DIMENSIONS = gemv_layout.MAX_BATCH_DIMENSIONS
DTYPES = gemv_layout.DTYPES
# gather_mm's GEMM_TPARAM_MACRO tile on a medium device, and gather_mm_rhs's.
HOST_TILE = gemm_layout.HOST_TILE
RHS_TILE = (16, 64, 16, 1, 2)
CONSTANTS = ("has_batch", "align_M", "align_N", "align_K")
RHS_CONSTANTS = ("align_N", "align_K")
CONSTANT_IDS = {"has_batch": 10, "align_M": 200, "align_N": 201, "align_K": 202}
PATTERN = re.compile(
    r"(steel_gather_mm_(nn|nt|tn|tt)_(float32|float16|bfloat16)_\3"
    r"_bm(\d+)_bn(\d+)_bk(\d+)_wm(\d+)_wn(\d+))"
    r"_has_batch_([tn])_align_M_([tn])_align_N_([tn])_align_K_([tn])"
)
RHS_PATTERN = re.compile(
    r"(steel_gather_mm_rhs_(nn|nt)_(float32|float16|bfloat16)_\3"
    r"_bm(\d+)_bn(\d+)_bk(\d+)_wm(\d+)_wn(\d+))_align_N_([tn])_align_K_([tn])"
)
ENTRY_PREFIXES = ("steel_gather_mm_",)
# [[buffer(i)]] parameters of gather_mm, gather_mm_rhs and gather_mm_offsets,
# in index order. matmul.cpp binds every one of them.
PARAMETERS = (
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
RHS_PARAMETERS = ("A", "B", "offsets", "C", "params", "num_groups")
OFFSETS_PARAMETERS = ("indices", "offsets", "M")
MATRIX_PARAMETERS = ("A", "B", "C")
INDEX_PARAMETERS = ("lhs_indices", "rhs_indices")
PARAMS_FIELDS = gemm_layout.PARAMS_FIELDS
TYPES = {
    "lhs_indices": "uint32",
    "rhs_indices": "uint32",
    "indices": "uint32",
    "offsets": "int32",
    "params": "int64",
    "indices_shape": "int32",
    "lhs_strides": "int64",
    "rhs_strides": "int64",
    "batch_ndim_a": "int32",
    "batch_shape_a": "int32",
    "batch_strides_a": "int64",
    "batch_ndim_b": "int32",
    "batch_shape_b": "int32",
    "batch_strides_b": "int64",
    "num_groups": "int32",
    "M": "int32",
}
SCALARS = ("batch_ndim_a", "batch_ndim_b", "num_groups", "M")
LIMITS = {
    "params": len(PARAMS_FIELDS),
    "indices_shape": MAX_BATCH_DIMENSIONS,
    "lhs_strides": MAX_BATCH_DIMENSIONS,
    "rhs_strides": MAX_BATCH_DIMENSIONS,
    "batch_shape_a": MAX_BATCH_DIMENSIONS + 2,
    "batch_strides_a": MAX_BATCH_DIMENSIONS + 2,
    "batch_shape_b": MAX_BATCH_DIMENSIONS + 2,
    "batch_strides_b": MAX_BATCH_DIMENSIONS + 2,
    "offsets": MAX_GROUPS,
}
INTEGER_TYPES = {
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint32": ctypes.c_uint32,
}
OUTPUTS = {"mm": "C", "rhs": "C", "offsets": "offsets"}


def _flag(value):
    return "t" if value else "n"


def kind(entry):
    """``mm``, ``rhs`` or ``offsets``."""
    if entry == OFFSETS_ENTRY:
        return "offsets"
    if isinstance(entry, str) and entry.startswith("steel_gather_mm_rhs_"):
        return "rhs"
    return "mm"


def signature(entry):
    """The source entry, transposition, element type and constants of an entry."""
    if entry == OFFSETS_ENTRY:
        return {"kind": "offsets", "source": OFFSETS_ENTRY, "constants": {}}
    rhs = kind(entry) == "rhs"
    match = (
        (RHS_PATTERN if rhs else PATTERN).fullmatch(entry)
        if isinstance(entry, str)
        else None
    )
    if match is None:
        raise ValueError("Unsupported gathered steel GEMM entry")
    base, transpose, dtype, *rest = match.groups()
    tile, flags = tuple(int(value) for value in rest[:5]), rest[5:]
    model_entry = model.BY_NAME.get(base)
    if model_entry is None or tile != (RHS_TILE if rhs else HOST_TILE):
        raise ValueError("Gathered steel GEMM entry is not the host's tile")
    names = RHS_CONSTANTS if rhs else CONSTANTS
    bm, bn, bk, wm, wn = tile
    return {
        "kind": "rhs" if rhs else "mm",
        "source": base,
        "model": model_entry,
        "transposeA": transpose[0] == "t",
        "transposeB": transpose[1] == "t",
        "dtype": dtype,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "wm": wm,
        "wn": wn,
        "constants": {name: flag == "t" for name, flag in zip(names, flags)},
    }


def source(entry):
    return OFFSETS_SOURCE if kind(entry) == "offsets" else SOURCE


def host_parameters(entry):
    return {"mm": PARAMETERS, "rhs": RHS_PARAMETERS, "offsets": OFFSETS_PARAMETERS}[
        kind(entry)
    ]


def workgroup_size(entry):
    info = signature(entry)
    if info["kind"] == "offsets":
        return (1, 1, 1)
    return (SIMD_SIZE, info["wn"], info["wm"])


def select(dtype, transpose_a, transpose_b, m, n, k, batch_ndim):
    """gather_mm's hash name for one product."""
    if dtype not in DTYPES:
        raise ValueError("Gathered products support float32, float16 and bfloat16")
    bm, bn, bk, wm, wn = HOST_TILE
    transpose = _flag(transpose_a) + _flag(transpose_b)
    return (
        f"steel_gather_mm_{transpose}_{dtype}_{dtype}"
        f"_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_has_batch_{_flag(batch_ndim > 1)}"
        f"_align_M_{_flag(m % bm == 0)}_align_N_{_flag(n % bn == 0)}"
        f"_align_K_{_flag(k % bk == 0)}"
    )


def select_rhs(dtype, transpose_b, n, k):
    """gather_mm_rhs's hash name for one product."""
    if dtype not in DTYPES:
        raise ValueError("Gathered products support float32, float16 and bfloat16")
    bm, bn, bk, wm, wn = RHS_TILE
    return (
        f"steel_gather_mm_rhs_n{_flag(transpose_b)}_{dtype}_{dtype}"
        f"_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"
        f"_align_N_{_flag(n % bn == 0)}_align_K_{_flag(k % bk == 0)}"
    )


def rhs_tiles(m, num_groups):
    """gather_mm_rhs's tiles_m: every group may start a partial tile."""
    return min(m, -(-m // RHS_TILE[0]) + num_groups - 1)


def parameter_type(entry, name):
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    return TYPES[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters, views)``: element counts of the matrix,
    index and offset buffers, the other parameters' values (``params`` as a
    field map), and zero-copy views of the index and offset buffers.
    """
    names = host_parameters(entry)
    if set(buffers) != set(names):
        raise ValueError("Gathered steel GEMM buffers do not match the entry")
    output = OUTPUTS[kind(entry)]
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, views, spans = {}, {}, {}, {}
    for name in names:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        count = buffer.count
        limit = 1 if name in SCALARS else LIMITS.get(name, MAX_ELEMENTS)
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == output)
            or type(count) is not int
            or not 1 <= count <= limit
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError(
                "Gathered steel GEMM buffer storage or direction does not match"
            )
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in MATRIX_PARAMETERS or name == output:
            counts[name] = count
            continue
        values = (INTEGER_TYPES[dtype] * count).from_address(buffer.data)
        if name in (*INDEX_PARAMETERS, "indices", "offsets"):
            counts[name], views[name] = count, values
            continue
        parameters[name] = values[0] if name in SCALARS else list(values)
    if "params" in parameters:
        parameters["params"] = dict(zip(PARAMS_FIELDS, parameters["params"]))
        if len(parameters["params"]) != len(PARAMS_FIELDS):
            raise ValueError("Gathered steel GEMM parameters do not match GEMMParams")
    low, high = spans[output]
    if any(
        other != output and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Gathered steel GEMM output overlaps another buffer")
    return counts, parameters, views


def _check_launch(entry, execution, count):
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Gathered steel GEMM launch geometry is missing")
    groups = list(execution["workgroupCount"])
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or groups != list(count)
        or not all(type(value) is int and 1 <= value <= MAX_GROUPS for value in groups)
    ):
        raise ValueError("Gathered steel GEMM launch does not match MLX's grid")


def _operand_extents(info, params):
    m, n, k, lda, ldb = (params[name] for name in ("M", "N", "K", "lda", "ldb"))
    return {
        "A": (k - 1) * lda + m if info["transposeA"] else (m - 1) * lda + k,
        "B": (n - 1) * ldb + k if info["transposeB"] else (k - 1) * ldb + n,
    }


def _check_common_params(info, params):
    if not isinstance(params, dict) or set(params) != set(PARAMS_FIELDS):
        raise ValueError("Gathered steel GEMM parameters are missing")
    if any(type(value) is not int for value in params.values()):
        raise ValueError("Gathered steel GEMM parameters must be integers")
    m, n, k = params["M"], params["N"], params["K"]
    minor_a = m if info["transposeA"] else k
    minor_b = k if info["transposeB"] else n
    if (
        not 1 <= min(m, n, k) <= max(m, n, k) <= MAX_ELEMENTS
        or params["ldd"] != n
        or params["tiles_n"] != -(-n // info["bn"])
        or params["swizzle_log"] != 0
        or params["gemm_k_iterations_aligned"] != k // info["bk"]
        or not minor_a <= params["lda"] <= MAX_ELEMENTS
        or not minor_b <= params["ldb"] <= MAX_ELEMENTS
    ):
        raise ValueError("Gathered steel GEMM parameters differ from matmul.cpp's")
    return m, n, k


def _check_mm_vectors(parameters):
    params = parameters["params"]
    ndim = params["batch_ndim"]
    shape = parameters["indices_shape"]
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    ndim_a, ndim_b = parameters["batch_ndim_a"], parameters["batch_ndim_b"]
    shape_a, strides_a = parameters["batch_shape_a"], parameters["batch_strides_a"]
    shape_b, strides_b = parameters["batch_shape_b"], parameters["batch_strides_b"]
    # Scalar indices bind one-element vectors the kernel never reads.
    padded = ([1], [0], [0])
    if (
        not 0 <= ndim <= MAX_BATCH_DIMENSIONS
        or not 0 <= ndim_a <= MAX_BATCH_DIMENSIONS
        or not 0 <= ndim_b <= MAX_BATCH_DIMENSIONS
        or (ndim and not len(shape) == len(lhs) == len(rhs) == ndim)
        or (not ndim and (shape, lhs, rhs) != padded)
        or not len(shape_a) == len(strides_a) == ndim_a + 2
        or not len(shape_b) == len(strides_b) == ndim_b + 2
        or any(extent < 1 for extent in (*shape, *shape_a, *shape_b))
        or any(stride < 0 for stride in (*lhs, *rhs, *strides_a, *strides_b))
    ):
        raise ValueError("Gathered steel GEMM batch vectors are invalid")
    return ndim, shape if ndim else []


def _gathered(name, views, counts, locations):
    values = []
    for offset in locations:
        if not 0 <= offset < counts[name]:
            raise ValueError(f"Gathered steel GEMM {name} read exceeds its buffer")
        values.append(views[name][offset])
    return values


def _check_operand(name, indices, ndim, shape, strides, extent, count):
    batches = math.prod(shape[:ndim])
    for index in set(indices):
        if ndim and index >= batches:
            raise ValueError(f"Gathered steel GEMM {name} index is out of range")
        if location(index, shape[:ndim], strides[:ndim]) + extent > count:
            raise ValueError(f"Gathered steel GEMM {name} exceeds its buffer")


def _plan_mm(entry, info, counts, parameters, views, threads, execution):
    params = parameters.get("params")
    m, n, k = _check_common_params(info, params)
    ndim, shape = _check_mm_vectors(parameters)
    lhs, rhs = parameters["lhs_strides"], parameters["rhs_strides"]
    expected = {
        "tiles_m": -(-m // info["bm"]),
        "batch_stride_a": lhs[0] if ndim else 0,
        "batch_stride_b": rhs[0] if ndim else 0,
        "batch_stride_d": m * n,
    }
    if any(params[name] != value for name, value in expected.items()):
        raise ValueError("Gathered steel GEMM parameters differ from matmul.cpp's")
    batches = math.prod(shape)
    _check_launch(entry, execution, (params["tiles_n"], params["tiles_m"], batches))
    if select(info["dtype"], info["transposeA"], info["transposeB"], m, n, k, ndim) != (
        entry
    ):
        raise ValueError("Gathered steel GEMM entry differs from MLX's routing")
    if threads != batches * m * n or counts["C"] != threads:
        raise ValueError("Gathered steel GEMM output extent differs")
    if ndim > 1:
        locations = [
            (location(z, shape, lhs), location(z, shape, rhs)) for z in range(batches)
        ]
    else:
        locations = [
            (params["batch_stride_a"] * z, params["batch_stride_b"] * z)
            for z in range(batches)
        ]
    lhs_values = _gathered("lhs_indices", views, counts, [x for x, _ in locations])
    rhs_values = _gathered("rhs_indices", views, counts, [y for _, y in locations])
    extents = _operand_extents(info, params)
    for name, values, suffix in (("A", lhs_values, "a"), ("B", rhs_values, "b")):
        _check_operand(
            name,
            values,
            parameters[f"batch_ndim_{suffix}"],
            parameters[f"batch_shape_{suffix}"],
            parameters[f"batch_strides_{suffix}"],
            extents[name],
            counts[name],
        )
    return {
        "kind": "mm",
        "source": info["source"],
        "M": m,
        "N": n,
        "K": k,
        "lda": params["lda"],
        "ldb": params["ldb"],
        "batches": batches,
        "constants": info["constants"],
        "lhsIndices": lhs_values,
        "rhsIndices": rhs_values,
    }


def _plan_rhs(entry, info, counts, parameters, views, threads, execution):
    params = parameters.get("params")
    m, n, k = _check_common_params(info, params)
    groups = parameters.get("num_groups")
    if type(groups) is not int or not 1 <= groups <= MAX_GROUPS:
        raise ValueError("Gathered steel GEMM group count is invalid")
    expected = {
        "lda": k,
        "tiles_m": rhs_tiles(m, groups),
        "batch_stride_a": 0,
        "batch_stride_d": 0,
        "batch_ndim": 0,
    }
    if any(params[name] != value for name, value in expected.items()) or (
        counts["offsets"] != groups
    ):
        raise ValueError("Gathered steel GEMM parameters differ from matmul.cpp's")
    _check_launch(entry, execution, (params["tiles_n"], params["tiles_m"], 1))
    if select_rhs(info["dtype"], info["transposeB"], n, k) != entry:
        raise ValueError("Gathered steel GEMM entry differs from MLX's routing")
    if threads != m * n or counts["C"] != threads:
        raise ValueError("Gathered steel GEMM output extent differs")
    offsets = list(views["offsets"])
    if any(
        not 0 <= start <= end <= m for start, end in zip(offsets, offsets[1:] + [m])
    ):
        raise ValueError("Gathered steel GEMM group offsets are not sorted rows")
    extents = _operand_extents(info, params)
    if extents["A"] > counts["A"]:
        raise ValueError("Gathered steel GEMM A exceeds its buffer")
    stride_b = params["batch_stride_b"]
    if stride_b < 0:
        raise ValueError("Gathered steel GEMM B batch stride is negative")
    for group, (start, end) in enumerate(zip(offsets, offsets[1:] + [m])):
        if start < end and group * stride_b + extents["B"] > counts["B"]:
            raise ValueError("Gathered steel GEMM B exceeds its buffer")
    return {
        "kind": "rhs",
        "source": info["source"],
        "M": m,
        "N": n,
        "K": k,
        "ldb": params["ldb"],
        "groups": groups,
        "constants": info["constants"],
        "offsets": offsets,
    }


def _plan_offsets(entry, counts, parameters, views, threads, execution):
    m = parameters.get("M")
    groups = counts["offsets"]
    if type(m) is not int or not 1 <= m <= counts["indices"]:
        raise ValueError("Gathered group offsets read past their indices")
    _check_launch(entry, execution, (groups, 1, 1))
    if threads != groups:
        raise ValueError("Gathered group offsets output extent differs")
    return {"kind": "offsets", "M": m, "groups": groups}


def plan(entry, counts, parameters, views, threads, execution):
    """Validate one launch against MLX's routing and return its metadata."""
    info = signature(entry)
    if info["kind"] == "offsets":
        return _plan_offsets(entry, counts, parameters, views, threads, execution)
    planner = _plan_rhs if info["kind"] == "rhs" else _plan_mm
    return planner(entry, info, counts, parameters, views, threads, execution)


def offsets_reference(indices, groups):
    """gather_mm_offsets: each group's first row in sorted ``indices``."""
    result = []
    for group in range(groups):
        low, high = 0, len(indices)
        while low < high:
            middle = (low + high) // 2
            if indices[middle] < group:
                low = middle + 1
            else:
                high = middle
        result.append(low)
    return result


def loader_access_assertions(entry):
    """The threadgroup slots each ``BlockLoader::load_unsafe`` writes."""
    return gemm_layout.model_loader_assertions(signature(entry)["model"], SOURCE)


def index_assertions(entry):
    """OpenGL ranges for the 64-bit pointer offsets the kernel forms.

    As ``test_current_gemm_gather`` asserts them: device views stay below
    2^31 elements, the index buffers are read as ``uint`` and the batch
    strides as ``int64_t``, and each threadgroup view reaches only the slots
    its loader writes.
    """
    info = signature(entry)
    element = {"float32": "float", "float16": "float16", "bfloat16": "bfloat16"}[
        info["dtype"]
    ]
    reads = max(loader.reads for loader in info["model"].loaders)
    accesses = {f"load_{element}", f"store_{element}", "load_int64_t", "load_uint"}
    if info["kind"] == "rhs":
        accesses = {f"load_{element}", f"store_{element}", "load_int"}
    device = [
        {"expression": f"source.offset + {offset}", "maximum": SIGNED_MAX}
        for offset in range(reads)
    ] + [
        {
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}",
            "maximum": SIGNED_MAX,
        }
        for access in sorted(accesses)
    ]
    workgroup = [
        {
            "expression": "reference.offset + index",
            "indexed": f"crosstl_workgroup_{tile}",
            "maximum": loader["maximum"],
        }
        for tile, loader in zip(("As", "Bs"), loader_access_assertions(entry))
    ]
    return tuple(
        {"source": SOURCE, **item, "minimum": 0} for item in device + workgroup
    )


def _row_contiguous(shape, strides):
    expected = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent != 1 and stride != expected:
            return False
        expected *= extent
    return True


def _check_transpose(shape, strides, is_vector):
    """gather_mm.inc's gemv_operand: matmul.cpp check_transpose that also
    copies views with negative strides. Returns ``(transposed, ld, copied)``."""
    if min(strides) >= 0:
        stx, sty = strides[-2], strides[-1]
        if sty == 1 and (not is_vector or stx == shape[-1]):
            return False, stx, False
        if stx == 1 and (not is_vector or sty == shape[-2]):
            return True, sty, False
    return False, shape[-1], True


def _batch_contiguous(shape, strides):
    """matmul.cpp ensure_batch_contiguous: ``(transposed, ld, copied)``."""
    if min(strides) >= 0 and _row_contiguous(shape, strides):
        return False, strides[-2], False
    rc = min(strides) >= 0 and all(
        strides[i + 1] * shape[i] == strides[i] for i in range(len(shape) - 3)
    )
    if rc:
        stx, sty, k, n = strides[-2], strides[-1], shape[-2], shape[-1]
        if sty == 1 and (n != 1 or stx == n):
            return False, stx, False
        if stx == 1 and (n != 1 or sty == k):
            return True, sty, False
    return False, shape[-1], True


def route(dtype, a_shape, a_strides, b_shape, b_strides, index_shape, right_sorted):
    """GatherMM::eval_gpu's route for nonempty operands and dense indices.

    Returns the kind, the entries in launch order and the operand copies the
    host makes first, each one translated copy dispatch.
    """
    m, k, n = a_shape[-2], a_shape[-1], b_shape[-1]
    if b_shape[-2] != k:
        raise ValueError("Gathered product operand shapes differ")
    if m == 1 and right_sorted:
        rows = math.prod(index_shape)
        broadcast = math.prod(a_shape[:-2]) != rows
        a_copy = broadcast or not _row_contiguous(a_shape, a_strides)
        transposed, ldb, b_copy = _batch_contiguous(b_shape, b_strides)
        return {
            "kind": "rhs",
            "entries": [OFFSETS_ENTRY, select_rhs(dtype, transposed, n, k)],
            "M": rows,
            "N": n,
            "K": k,
            "ldb": ldb,
            "copies": {"a": a_copy, "b": b_copy},
        }
    if m == 1 or n == 1:
        is_mv = m != 1
        mat = (a_shape, a_strides) if is_mv else (b_shape, b_strides)
        vec = (b_shape, b_strides) if is_mv else (a_shape, a_strides)
        out = m if is_mv else n
        transposed, ld, mat_copy = _check_transpose(*mat, out == 1)
        vec_copy = _check_transpose(*vec, True)[2]
        transposed = (not is_mv) != transposed
        return {
            "kind": "mv",
            "entries": [gather_mv_layout.select(dtype, transposed, k, out)],
            "K": k,
            "outVecSize": out,
            "matrixLd": ld,
            "copies": {
                "a": mat_copy if is_mv else vec_copy,
                "b": vec_copy if is_mv else mat_copy,
            },
        }
    transpose_a, lda, a_copy = _check_transpose(a_shape, a_strides, False)
    transpose_b, ldb, b_copy = _check_transpose(b_shape, b_strides, False)
    return {
        "kind": "mm",
        "entries": [select(dtype, transpose_a, transpose_b, m, n, k, len(index_shape))],
        "M": m,
        "N": n,
        "K": k,
        "lda": lda,
        "ldb": ldb,
        "copies": {"a": a_copy, "b": b_copy},
    }
