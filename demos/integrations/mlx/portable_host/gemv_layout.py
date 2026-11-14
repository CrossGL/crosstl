"""Storage and launch contracts for MLX's floating-point matrix-vector kernels.

``Matmul::eval_gpu`` in ``backend.cpp`` routes products with one output row or
column as matmul.cpp does: ``check_transpose`` on both operands,
``collapse_batches``, the fold of a contiguous batch into ``M``, then
``gemv_axbpy``'s choice of ``gemv`` or ``gemv_t`` and its tile shape. MLX takes
``gemv_wide`` only from Apple GPU generation 15, which the portable host never
reports, and ``Matmul`` never asks for ``axpby``. Single dot products, which
MLX sends to ``dot_product`` and an all-reduce, fail with an explicit error;
``gemm_layout`` describes the matrix-matrix route.
"""

import ctypes
import math
import re

from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES

SOURCE = "mlx/backend/metal/kernels/gemv.metal"
SIMD_SIZE = 32
# Every buffer the host binds holds at most this many elements, so each
# pointer offset the OpenGL translation asserts stays below 2^31 - 1.
MAX_ELEMENTS = 2**28 - 1
SIGNED_MAX = 2**31 - 1
MAX_GROUPS = 65535
MAX_BATCH_DIMENSIONS = 16
DTYPES = ("float32", "float16", "bfloat16")
PATTERN = re.compile(
    r"(gemv|gemv_t)_(float32|float16|bfloat16)_bm(\d+)_bn(\d+)_sm(\d+)_sn(\d+)"
    r"_tm(\d+)_tn(\d+)_nc([01])_axpby0"
)
ENTRY_PREFIXES = ("gemv_",)
# gemv.metal's instantiated tiles (bm, bn, sm, sn, tm, tn).
BLOCKS = {
    "gemv": (
        (1, 8, 1, 32, 4, 4),
        (1, 8, 1, 32, 1, 4),
        (1, 1, 8, 4, 4, 4),
        (1, 1, 8, 4, 1, 4),
        (4, 1, 1, 32, 1, 4),
        (4, 1, 1, 32, 4, 4),
        (8, 1, 1, 32, 4, 4),
    ),
    "gemv_t": (
        (1, 2, 8, 4, 4, 1),
        (1, 2, 8, 4, 4, 4),
        (1, 4, 8, 4, 4, 4),
        (1, 16, 8, 4, 4, 4),
        (1, 16, 4, 8, 4, 4),
    ),
}
# [[buffer(i)]] parameters of gemv.metal's kernels, in index order.
PARAMETERS = (
    "mat",
    "in_vec",
    "bias",
    "out_vec",
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "alpha",
    "beta",
    "batch_ndim",
    "batch_shape",
    "vector_batch_stride",
    "matrix_batch_stride",
    "bias_batch_stride",
    "bias_stride",
)
# What gemv_axbpy binds when it does not compute axpby.
HOST_PARAMETERS = (
    "mat",
    "in_vec",
    "out_vec",
    "in_vec_size",
    "out_vec_size",
    "marix_ld",
    "batch_ndim",
    "batch_shape",
    "vector_batch_stride",
    "matrix_batch_stride",
)
MATRIX_PARAMETERS = ("mat", "in_vec", "out_vec")
SCALAR_PARAMETERS = ("in_vec_size", "out_vec_size", "marix_ld", "batch_ndim")
VECTOR_TYPES = {
    "batch_shape": "int32",
    "vector_batch_stride": "int64",
    "matrix_batch_stride": "int64",
}
INTEGER_TYPES = {"int32": ctypes.c_int32, "int64": ctypes.c_int64}
# Expressions OpenGL may evaluate in 32 bits: the row and column loads and
# the batch offsets. Each names an element offset of a bound buffer.
INDEX_RANGE_ASSERTIONS = {
    "gemv": (
        "uint64(out_row) * matrix_ld",
        "mat + uint64(out_row) * matrix_ld",
        "src + src_offset + tn",
    ),
    "gemv_t": (
        "uint64(bm + tm) * marix_ld + out_col + tn",
        "mat + uint64(bm + tm) * marix_ld + out_col + tn",
    ),
    "batch": (
        "tid.z * buffer_load(vector_batch_stride, 0)",
        "in_vec + tid.z * buffer_load(vector_batch_stride, 0)",
        "tid.z * buffer_load(matrix_batch_stride, 0)",
        "mat + tid.z * buffer_load(matrix_batch_stride, 0)",
        "tid.z * buffer_load(bias_batch_stride, 0)",
        "bias + tid.z * buffer_load(bias_batch_stride, 0)",
    ),
}


def signature(entry):
    """The kernel, element type, tile and batch variant of an entry."""
    match = PATTERN.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unsupported matrix-vector entry")
    kind, dtype, *tile, nc = match.groups()
    block = tuple(int(value) for value in tile)
    if block not in BLOCKS[kind]:
        raise ValueError("Matrix-vector entry is not instantiated")
    bm, bn, sm, sn, tm, tn = block
    return {
        "kind": kind,
        "dtype": dtype,
        "bm": bm,
        "bn": bn,
        "sm": sm,
        "sn": sn,
        "tm": tm,
        "tn": tn,
        "nc": nc == "1",
    }


def workgroup_size(entry):
    info = signature(entry)
    return (SIMD_SIZE, info["bn"], info["bm"])


def outputs_per_group(entry):
    """gemv_axbpy's n_out_per_tgp."""
    info = signature(entry)
    if info["kind"] == "gemv_t":
        return info["bn"] * info["sn"] * info["tn"]
    return info["bm"] * info["sm"] * info["tm"]


def index_assertions(entry):
    return INDEX_RANGE_ASSERTIONS[signature(entry)["kind"]] + (
        INDEX_RANGE_ASSERTIONS["batch"]
    )


def select(dtype, transpose_mat, in_size, out_size, batch_ndim):
    """gemv_axbpy's kernel name for one product without axpby."""
    if dtype not in DTYPES:
        raise ValueError("Matrix-vector products support float32, float16 and bfloat16")
    if transpose_mat:
        sm, sn = (4, 8) if in_size >= 8192 and out_size >= 2048 else (8, 4)
        bn = 16 if out_size >= 2048 else 4 if out_size >= 512 else 2
        bm, tm, tn = 1, 4, 1 if out_size < 4 else 4
        kind = "gemv_t"
    else:
        bm, bn, sm, sn, tn = 8 if out_size >= 4096 else 4, 1, 1, 32, 4
        if in_size <= 64:
            bm, sm, sn = 1, 8, 4
        elif in_size >= 16 * out_size:
            bm, bn = 1, 8
        tm = 1 if out_size < 4 else 4
        kind = "gemv"
    return (
        f"{kind}_{dtype}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}"
        f"_nc{int(batch_ndim != 1)}_axpby0"
    )


def collapse_contiguous_dims(shape, strides, size_cap=SIGNED_MAX):
    """backend/common/utils.cpp collapse_contiguous_dims over stride vectors."""
    groups, current = [], []
    size = shape[0] if shape else 0
    for axis, extent in enumerate(shape):
        if axis:
            size *= extent
            if size > size_cap or any(
                values[axis] * extent != values[axis - 1] for values in strides
            ):
                groups.append(current)
                current, size = [], extent
        if extent != 1:
            current.append(axis)
    groups.append(current)
    out_shape, out_strides = [], [[] for _ in strides]
    for group in groups:
        if not group:
            continue
        out_shape.append(math.prod(shape[axis] for axis in group))
        for values, collapsed in zip(strides, out_strides):
            collapsed.append(values[group[-1]])
    if shape and not out_shape:
        out_shape = [1]
        out_strides = [[0] for _ in strides]
    return out_shape, out_strides


def _check_transpose(shape, strides, is_vector):
    """matmul.cpp check_transpose: ``(transposed, ld, copied)``."""
    stx, sty = strides[-2], strides[-1]
    if sty == 1 and (not is_vector or stx == shape[-1]):
        return False, stx, False
    if stx == 1 and (not is_vector or sty == shape[-2]):
        return True, sty, False
    return False, shape[-1], True


def _row_strides(shape):
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.insert(0, stride)
        stride *= extent
    return strides


def operands(a_shape, a_strides, b_shape, b_strides):
    """Matmul::eval_gpu's operand preparation, before it picks a kernel.

    ``a`` is ``(..., M, K)`` and ``b`` ``(..., K, N)`` as ops.cpp matmul hands
    them over. Returns ``check_transpose`` for both operands, the collapsed
    batches, the fold of a contiguous batch into ``M`` and whether MLX would
    take its dot-product route.
    """
    if len(a_shape) != len(b_shape) or len(a_shape) < 2:
        raise ValueError("Matrix product operand ranks differ")
    m, k, n = a_shape[-2], a_shape[-1], b_shape[-1]
    if b_shape[-2] != k or list(a_shape[:-2]) != list(b_shape[:-2]):
        raise ValueError("Matrix product operand shapes differ")
    a_transposed, lda, a_copy = _check_transpose(a_shape, a_strides, m == 1)
    b_transposed, ldb, b_copy = _check_transpose(b_shape, b_strides, n == 1)
    # The host also copies views with negative strides, which it never binds.
    if min(a_strides) < 0:
        a_transposed, lda, a_copy = False, k, True
    if min(b_strides) < 0:
        b_transposed, ldb, b_copy = False, n, True
    a_strides = _row_strides(a_shape) if a_copy else list(a_strides)
    b_strides = _row_strides(b_shape) if b_copy else list(b_strides)
    if len(a_shape) == 2:
        batch_shape, a_batch, b_batch = [1], [0], [0]
    else:
        batch_shape, (a_batch, b_batch) = collapse_contiguous_dims(
            list(a_shape[:-2]), [a_strides[:-2], b_strides[:-2]]
        )
    batches = math.prod(a_shape[:-2])
    if (
        batches > 1
        and not a_transposed
        and len(batch_shape) == 1
        and a_strides[-2] == k
        and a_batch[-1] == m * k
        and b_batch[-1] == 0
    ):
        m, batches = m * batch_shape[-1], 1
        batch_shape, a_batch, b_batch = [1], [0], [0]
    row_contiguous = all(
        extent == 1 or stride == expected
        for shape, strides in ((a_shape, a_strides), (b_shape, b_strides))
        for extent, stride, expected in zip(shape, strides, _row_strides(shape))
    )
    return {
        "M": m,
        "N": n,
        "K": k,
        "transposeA": a_transposed,
        "transposeB": b_transposed,
        "lda": lda,
        "ldb": ldb,
        "batches": batches,
        "batchShape": batch_shape,
        "aBatchStride": a_batch,
        "bBatchStride": b_batch,
        "dot": m == 1 and n == 1 and batches == 1 and row_contiguous,
        "copies": {"a": a_copy, "b": b_copy},
    }


def route(dtype, a_shape, a_strides, b_shape, b_strides):
    """Matmul::eval_gpu's matrix-vector route for operands of equal rank.

    Returns the entry with the values gemv_axbpy binds; raises for the routes
    that take other kernels.
    """
    prepared = operands(a_shape, a_strides, b_shape, b_strides)
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    if prepared["dot"]:
        raise ValueError("Single dot products are not routed")
    if min(m, n) != 1:
        raise ValueError("Matrix-matrix products take the steel GEMM route")
    is_b_matrix = n != 1
    a_batch, b_batch = prepared["aBatchStride"], prepared["bBatchStride"]
    transpose_mat = (
        not prepared["transposeB"] if is_b_matrix else prepared["transposeA"]
    )
    out_size = n if is_b_matrix else m
    batch_shape = prepared["batchShape"]
    return {
        "entry": select(dtype, transpose_mat, k, out_size, len(batch_shape)),
        "M": m,
        "N": n,
        "K": k,
        "batches": prepared["batches"],
        "outVecSize": out_size,
        "matrixLd": prepared["ldb"] if is_b_matrix else prepared["lda"],
        "batchShape": batch_shape,
        "vectorBatchStride": a_batch if is_b_matrix else b_batch,
        "matrixBatchStride": b_batch if is_b_matrix else a_batch,
        "copies": prepared["copies"],
    }


def parameter_type(entry, name):
    if name in MATRIX_PARAMETERS:
        return signature(entry)["dtype"]
    if name in SCALAR_PARAMETERS:
        return "int32"
    return VECTOR_TYPES[name]


def read_storage(entry, buffers):
    """Check native buffer identity and read the integer parameters.

    Returns ``(counts, parameters)``: element counts of the matrix buffers and
    the integer values of the scalar and vector parameters.
    """
    signature(entry)
    if set(buffers) != set(HOST_PARAMETERS):
        raise ValueError("Matrix-vector buffers do not match the entry")
    address_limit = 1 << (8 * ctypes.sizeof(ctypes.c_void_p))
    counts, parameters, spans = {}, {}, {}
    for name in HOST_PARAMETERS:
        buffer, dtype = buffers[name], parameter_type(entry, name)
        itemsize = ITEM_SIZES.get(dtype) or ctypes.sizeof(INTEGER_TYPES[dtype])
        count = buffer.count
        scalar = name in SCALAR_PARAMETERS
        if (
            not buffer.data
            or buffer.data % itemsize
            or buffer.dtype != dtype.encode("ascii")
            or buffer.output != int(name == "out_vec")
            or type(count) is not int
            or not 1 <= count <= (1 if scalar else MAX_ELEMENTS)
            or (name in VECTOR_TYPES and count > MAX_BATCH_DIMENSIONS)
            or buffer.data + count * itemsize > address_limit
        ):
            raise ValueError("Matrix-vector buffer storage or direction does not match")
        spans[name] = (buffer.data, buffer.data + count * itemsize)
        if name in MATRIX_PARAMETERS:
            counts[name] = count
            continue
        values = list(
            ctypes.cast(
                buffer.data, ctypes.POINTER(INTEGER_TYPES[dtype] * count)
            ).contents
        )
        parameters[name] = values[0] if scalar else values
    low, high = spans["out_vec"]
    if any(
        other != "out_vec" and max(low, start) < min(high, end)
        for other, (start, end) in spans.items()
    ):
        raise ValueError("Matrix-vector output overlaps another buffer")
    return counts, parameters


def _location(index, shape, strides):
    """utils.h elem_to_loc."""
    location = 0
    for extent, stride in zip(reversed(shape), reversed(strides)):
        location += (index % extent) * stride
        index //= extent
    return location


def plan(entry, counts, parameters, threads, execution):
    """Validate one launch against MLX's routing and return its metadata.

    The returned ``inert`` values fill the bindings gemv_axbpy leaves unbound
    without axpby: the kernels never read them.
    """
    info = signature(entry)
    k = parameters.get("in_vec_size")
    out = parameters.get("out_vec_size")
    ld = parameters.get("marix_ld")
    ndim = parameters.get("batch_ndim")
    shape = parameters.get("batch_shape")
    vector_strides = parameters.get("vector_batch_stride")
    matrix_strides = parameters.get("matrix_batch_stride")
    if (
        any(type(value) is not int for value in (k, out, ld, ndim))
        or not 1 <= min(k, out) <= max(k, out) <= MAX_ELEMENTS
        or not 0 <= ld <= MAX_ELEMENTS
        or not 1 <= ndim <= MAX_BATCH_DIMENSIONS
        or not len(shape) == len(vector_strides) == len(matrix_strides) == ndim
    ):
        raise ValueError("Matrix-vector dimensions are invalid")
    if info["nc"] != (ndim != 1):
        raise ValueError("Matrix-vector batch variant differs from MLX's routing")
    if any(extent < 1 for extent in shape) or any(
        stride < 0 for stride in (*vector_strides, *matrix_strides)
    ):
        raise ValueError("Matrix-vector batch strides must be nonnegative")
    if not isinstance(execution, dict) or set(execution) != {
        "workgroupCount",
        "workgroupSize",
    }:
        raise ValueError("Matrix-vector launch geometry is missing")
    tiles, rows, batches = execution["workgroupCount"]
    if (
        list(execution["workgroupSize"]) != list(workgroup_size(entry))
        or any(type(value) is not int for value in (tiles, rows, batches))
        or rows != 1
        or tiles != -(-out // outputs_per_group(entry))
        or not 1 <= tiles <= MAX_GROUPS
        or not 1 <= batches <= MAX_GROUPS
        or math.prod(shape) != batches
    ):
        raise ValueError("Matrix-vector launch does not match MLX's grid")
    transpose_mat = info["kind"] == "gemv_t"
    if select(info["dtype"], transpose_mat, k, out, ndim) != entry:
        raise ValueError("Matrix-vector entry differs from MLX's routing")
    if threads != out * batches or counts["out_vec"] != threads:
        raise ValueError("Matrix-vector output extent differs")
    if any(count > MAX_ELEMENTS for count in counts.values()):
        raise ValueError("Matrix-vector buffers exceed the index bound")
    # gemv reads out rows of K elements, gemv_t K rows of out elements.
    extents = {
        "in_vec": k,
        "mat": (out - 1) * ld + k if not transpose_mat else (k - 1) * ld + out,
    }
    for name, strides in (("in_vec", vector_strides), ("mat", matrix_strides)):
        end = max(_location(z, shape, strides) for z in range(batches))
        if end + extents[name] > counts[name]:
            raise ValueError(f"Matrix-vector {name} batch exceeds its buffer")
    return {
        "kind": info["kind"],
        "K": k,
        "outVecSize": out,
        "matrixLd": ld,
        "batches": batches,
        "inert": {
            "alpha": 1.0,
            "beta": 0.0,
            "bias_batch_stride": [0] * ndim,
            "bias_stride": 0,
        },
    }
