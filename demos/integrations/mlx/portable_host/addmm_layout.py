"""Operand preparation and routing of MLX's ``AddMM`` primitive.

``AddMM::eval_gpu`` in ``backend.cpp`` follows matmul.cpp. An empty output is
only allocated. A product with ``K == 0`` copies ``C`` when ``beta == 1`` and
otherwise multiplies it by ``beta``. Every other product runs
``check_transpose`` on ``A`` and ``B``, ``collapse_batches`` over the batch
axes of all three operands and the fold of a contiguous batch into ``M``, then
takes ``gemv_axbpy`` for one output row or column and ``steel_matmul_axpby``
otherwise, with ``alpha`` and ``beta``. MLX takes ``gemv_wide`` only from
Apple GPU generation 15, which the portable host never reports.

``C`` binds as MLX passes it. The host copies it only when one of its strides
is negative, because no binding admits a negative offset. Each route lists the
values ``backend.cpp`` binds, which the gemv, steel GEMM and split-K layouts
check again when the product is launched.
"""

import math

from demos.integrations.mlx.portable_host import (
    gemm_layout,
    gemm_splitk_layout,
    gemv_layout,
)

DTYPES = gemv_layout.DTYPES


def _row_strides(shape):
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.insert(0, stride)
        stride *= extent
    return strides


def operands(a_shape, a_strides, b_shape, b_strides, c_shape, c_strides):
    """AddMM::eval_gpu's operand preparation, before it picks a kernel.

    ``a`` is ``(..., M, K)``, ``b`` ``(..., K, N)`` and ``c`` ``(..., M, N)``
    as ops.cpp addmm hands them over.
    """
    if not len(a_shape) == len(b_shape) == len(c_shape) >= 2:
        raise ValueError("AddMM operand ranks differ")
    m, k, n = a_shape[-2], a_shape[-1], b_shape[-1]
    if (
        b_shape[-2] != k
        or list(a_shape[:-2]) != list(b_shape[:-2])
        or list(c_shape) != [*a_shape[:-1], n]
        or len(c_strides) != len(c_shape)
    ):
        raise ValueError("AddMM operand shapes differ")
    check = gemv_layout._check_transpose
    a_transposed, lda, a_copy = check(a_shape, a_strides, m == 1)
    b_transposed, ldb, b_copy = check(b_shape, b_strides, n == 1)
    # The host also copies views with negative strides, which it never binds.
    if min(a_strides) < 0:
        a_transposed, lda, a_copy = False, k, True
    if min(b_strides) < 0:
        b_transposed, ldb, b_copy = False, n, True
    c_copy = min(c_strides) < 0
    a_strides = _row_strides(a_shape) if a_copy else list(a_strides)
    b_strides = _row_strides(b_shape) if b_copy else list(b_strides)
    c_strides = _row_strides(c_shape) if c_copy else list(c_strides)
    if len(a_shape) == 2:
        batch_shape, a_batch, b_batch, c_batch = [1], [0], [0], [0]
    else:
        batch_shape, (a_batch, b_batch, c_batch) = gemv_layout.collapse_contiguous_dims(
            list(a_shape[:-2]), [a_strides[:-2], b_strides[:-2], c_strides[:-2]]
        )
    batches = math.prod(a_shape[:-2])
    if (
        batches > 1
        and not a_transposed
        and len(batch_shape) == 1
        and a_strides[-2] == k
        and a_batch[-1] == m * k
        and c_batch[-1] == m * c_strides[-2]
        and b_batch[-1] == 0
    ):
        m, batches = m * batch_shape[-1], 1
        batch_shape, a_batch, b_batch, c_batch = [1], [0], [0], [0]
    return {
        "M": m,
        "N": n,
        "K": k,
        "transposeA": a_transposed,
        "transposeB": b_transposed,
        "lda": lda,
        "ldb": ldb,
        "ldc": c_strides[-2],
        "fdc": c_strides[-1],
        "batches": batches,
        "batchShape": batch_shape,
        "aBatchStride": a_batch,
        "bBatchStride": b_batch,
        "cBatchStride": c_batch,
        "copies": {"a": a_copy, "b": b_copy, "c": c_copy},
    }


def _axpby(alpha, beta):
    """gemv_axbpy's and the split-K accumulation's do_axpby."""
    return alpha != 1.0 or beta != 0.0


def _gemv(dtype, prepared, alpha, beta):
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    is_b_matrix = n != 1
    transpose_mat = (
        not prepared["transposeB"] if is_b_matrix else prepared["transposeA"]
    )
    out_size = n if is_b_matrix else m
    batch_shape = prepared["batchShape"]
    axpby = _axpby(alpha, beta)
    route = {
        "entry": gemv_layout.select(
            dtype, transpose_mat, k, out_size, len(batch_shape), axpby
        ),
        "M": m,
        "N": n,
        "K": k,
        "batches": prepared["batches"],
        "outVecSize": out_size,
        "matrixLd": prepared["ldb"] if is_b_matrix else prepared["lda"],
        "batchShape": batch_shape,
        "vectorBatchStride": (
            prepared["aBatchStride"] if is_b_matrix else prepared["bBatchStride"]
        ),
        "matrixBatchStride": (
            prepared["bBatchStride"] if is_b_matrix else prepared["aBatchStride"]
        ),
        "copies": prepared["copies"],
    }
    if axpby:
        # gemv_axbpy binds C's last stride whichever operand is the matrix.
        route["axpby"] = {"alpha": alpha, "beta": beta, "biasStride": prepared["fdc"]}
        route["biasBatchStride"] = prepared["cBatchStride"]
    return route


def _splitk(dtype, prepared, alpha, beta):
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    axbpy = _axpby(alpha, beta)
    entry, accum = gemm_splitk_layout.select(
        dtype, prepared["transposeA"], prepared["transposeB"], m, n, k, axbpy
    )
    params = gemm_splitk_layout.splitk_params(m, n, k, prepared["lda"], prepared["ldb"])
    partitions = params["split_k_partitions"]
    accum_params = {"k_partitions": partitions, "partition_stride": m * n, "ldd": n}
    if axbpy:
        accum_params.update(
            ldc=prepared["ldc"], fdc=prepared["fdc"], alpha=alpha, beta=beta
        )
    return {
        "entry": entry,
        "accum": accum,
        "M": m,
        "N": n,
        "K": k,
        "batches": 1,
        "partitions": partitions,
        "params": params,
        "accumParams": accum_params,
        "copies": prepared["copies"],
    }


def _gemm(dtype, prepared, alpha, beta):
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    batch_shape = prepared["batchShape"]
    bm, bn, bk, _, _ = gemm_layout.HOST_TILE
    a_batch, b_batch = prepared["aBatchStride"], prepared["bBatchStride"]
    c_batch = prepared["cBatchStride"]
    params = {
        "M": m,
        "N": n,
        "K": k,
        "lda": prepared["lda"],
        "ldb": prepared["ldb"],
        "ldd": n,
        "tiles_n": -(-n // bn),
        "tiles_m": -(-m // bm),
        "batch_stride_a": a_batch[-1],
        "batch_stride_b": b_batch[-1],
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // bk,
        "batch_ndim": len(batch_shape),
    }
    use_out_source, _ = gemm_layout.epilogue(alpha, beta)
    route = {
        "entry": gemm_layout.select(
            dtype,
            prepared["transposeA"],
            prepared["transposeB"],
            m,
            n,
            k,
            len(batch_shape),
            alpha,
            beta,
        ),
        "M": m,
        "N": n,
        "K": k,
        "batches": prepared["batches"],
        "batchShape": batch_shape,
        # steel_matmul_axpby appends C's batch strides with CHECK_AB.
        "batchStrides": list(a_batch) + list(b_batch) + list(c_batch),
        "params": params,
        "copies": prepared["copies"],
    }
    if use_out_source:
        route["addmm"] = {
            "ldc": prepared["ldc"],
            "fdc": prepared["fdc"],
            "batch_stride_c": c_batch[-1],
            "alpha": alpha,
            "beta": beta,
        }
    return route


def route(dtype, a, b, c, alpha, beta):
    """The family and launch values of one AddMM product.

    ``a``, ``b`` and ``c`` are ``(shape, strides)`` pairs. Returns
    ``(family, route)`` with family ``"empty"``, ``"scale"`` (``K == 0``),
    ``"gemv"``, ``"splitk"`` or ``"gemm"``.
    """
    if dtype not in DTYPES:
        raise ValueError("AddMM products support float32, float16 and bfloat16")
    if type(alpha) is not float or type(beta) is not float:
        raise ValueError("AddMM alpha and beta are float32 values")
    (a_shape, a_strides), (b_shape, b_strides), (c_shape, c_strides) = a, b, c
    if math.prod(c_shape) == 0:
        return "empty", {"copies": {}}
    if a_shape[-1] == 0:
        # matmul.cpp copies C as is or multiplies it by beta. The host's
        # elementwise multiplication first copies each operand that is not
        # row-contiguous: the broadcast beta unless C has one element, and C.
        if beta == 1.0:
            return "scale", {"beta": beta, "copy": True, "copies": {}}
        dense = all(
            extent == 1 or stride == expected
            for extent, stride, expected in zip(
                c_shape, c_strides, _row_strides(c_shape)
            )
        )
        copies = {"beta": math.prod(c_shape) > 1, "c": not dense}
        return "scale", {"beta": beta, "copy": False, "copies": copies}
    prepared = operands(a_shape, a_strides, b_shape, b_strides, c_shape, c_strides)
    m, n, k = prepared["M"], prepared["N"], prepared["K"]
    if min(m, n) == 1:
        return "gemv", _gemv(dtype, prepared, alpha, beta)
    if gemm_layout.split_k(m, n, k, prepared["batches"]):
        return "splitk", _splitk(dtype, prepared, alpha, beta)
    return "gemm", _gemm(dtype, prepared, alpha, beta)
