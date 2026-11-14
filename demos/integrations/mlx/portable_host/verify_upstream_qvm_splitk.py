"""Run the unchanged pinned split-K ``test_qvm`` tests on CPU and the native backend.

``test_qvm_splitk`` and ``test_qvm_splitk_multi_row`` multiply vectors and
short row blocks by untransposed affine-quantized matrices whose inner
dimension is at least 1024, and compare each product with ``x @ dequantize(w)``.
Through the portable host the quantized products reach the translated
``affine_qvm_split_k`` kernels, whose partitions are summed by the translated
column reductions MLX's reduce.cpp picks: ``col_reduce_small`` for 8
partitions and ``col_reduce_looped`` for 32. The reference products take
Matmul's gemv, steel GEMM or split-K GEMM routes. The error bound is a
translated whole-array maximum. Every product and column package is translated
before the native worker starts; the maximum is packaged on demand at the
widths the run launches.
"""

import math
import sys
from collections import Counter
from itertools import product

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    gemm_layout,
    gemv_layout,
    matmul_layout,
    quantization_layout,
    upstream_runner,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qvm_splitk"
LABEL = "test_qvm_splitk"
UPSTREAM_TESTS = (
    "test_quantized.TestQuantized.test_qvm_splitk",
    "test_quantized.TestQuantized.test_qvm_splitk_multi_row",
)
# test_qvm_splitk's product(group_size, bits, M, N, B), in the test's order.
GRID = ((128, 64, 32), (2, 4, 8), (128,), (16384,), (1, 3))
# Its closing 1-D vector: group size, bits and N.
VECTOR = (32, 8, 2048)
# test_qvm_splitk_multi_row's product(group_size, bits, N, K, M).
ROWS_GRID = ((64, 32), (4, 8), (128,), (2048, 4096), (2, 3))
REDUCTIONS = ("all_reduce_maxfloat32",)
# The looped column sum of 32 partitions; small sums are packaged by variant.
COLUMN_SUMS = ("col_reduce_looped_1_32_32_reduce_sumfloat32",)
# About 12,000 native dispatches; a loaded M2 Max needed more than 5,400 s.
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 900, "native": 6900}
FAMILIES = ("quantized", "gemv", "gemm", "splitk", "small")
# Every affine, steel and column entry counts, so an unexpected route fails.
PREFIXES = ("affine_", *gemv_layout.ENTRY_PREFIXES, "steel_gemm_", "col_reduce_")


def products():
    """``(group, bits, rows, N, K, batches, x_shape, w_shape)`` per product.

    ``rows``, ``N``, ``K`` and ``batches`` are QuantizedMatmul's ``M``, ``N``,
    ``K`` and batch count; ``x_shape`` and ``w_shape`` the test's operands.
    """
    for group, bits, m, n, b in product(*GRID):
        yield group, bits, 1, m, n, b, (b, 1, n), (b, n, m)
    group, bits, n = VECTOR
    # A 1-D x against 2-D weights folds into one row.
    yield group, bits, 1, n, n, 1, (n,), (n, n)
    for group, bits, n, k, m in product(*ROWS_GRID):
        yield group, bits, m, n, k, 1, (m, k), (k, n)


def reference_operands(x_shape, w_shape):
    """The operands ``x @ w_hat`` gives Matmul.

    ops.cpp's matmul expands a 1-D ``x`` to one row, whose stride is the row
    length; every other operand is row-contiguous as the test creates it.
    """
    a_shape = (1, *x_shape) if len(x_shape) == 1 else x_shape
    return (
        a_shape,
        upstream_runner.row_strides(a_shape),
        w_shape,
        upstream_runner.row_strides(w_shape),
    )


def column_sum(quantized, rows, n):
    """The column reduction reduce.cpp runs over a split-K product's partitions.

    Returns ``(entry, small_variant)``: a small sum's cache key, else ``None``.
    """
    split = matmul_layout.signature(quantized)["splitK"]
    if split >= 32:
        return COLUMN_SUMS[0], None
    entry = "col_reduce_small_1_reduce_sumfloat32"
    workgroup = (*column_small_layout.geometry(rows * n, split), 1)
    return entry, column_small_layout.variant(entry, workgroup)


def routes():
    """Each product's quantized entry, column sum and reference route."""
    for group, bits, rows, n, k, batches, x_shape, w_shape in products():
        quantized = matmul_layout.select(
            "float32", group, bits, rows, n, k, batches, transpose=False
        )
        if "splitK" not in matmul_layout.signature(quantized):
            raise ValueError("test_qvm_splitk products must split K")
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(x_shape, w_shape)
        )
        if sum(route["copies"].values()):
            raise ValueError("test_qvm_splitk's reference operands need no copies")
        yield group, bits, quantized, column_sum(quantized, rows, n), family, route


def expected_dispatches():
    """The affine, column and matrix-product entries the tests dispatch.

    Each product quantizes and dequantizes ``w`` in launches of at most
    65,535 groups, multiplies through QuantizedMatmul's split-K route and its
    partition sum, and forms the reference with Matmul; a split-K reference
    also launches its accumulation.
    """
    counts = Counter()
    for (*_, w_shape), (group, bits, quantized, (column, _), family, route) in zip(
        products(), routes()
    ):
        for operation in ("quantize", "dequantize"):
            counts[
                f"affine_{operation}_float_gs_{group}_b_{bits}"
            ] += quantization_layout.launches(math.prod(w_shape), group)
        counts[quantized] += 1
        counts[column] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries and small-sum variants to translate ahead, by family."""
    entries = {family: set() for family in FAMILIES}
    for _, _, quantized, (_, variant), family, route in routes():
        entries["quantized"].add(quantized)
        if variant is not None:
            entries["small"].add(variant)
        entries[family].add(route["entry"])
        if family == "splitk":
            entries[family].add(route["accum"])
    return {family: sorted(names) for family, names in entries.items()}


def validate_trace(trace, target, count):
    upstream_runner.check_trace(
        trace,
        target,
        count,
        label=LABEL,
        expected=expected_dispatches(),
        prefixes=PREFIXES,
        reductions=REDUCTIONS,
    )


def prepare_packages(root, packages, reductions, jobs):
    """Translate every product and column entry and index the maximum."""
    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
    )

    entries = package_entries()
    builders = {
        "quantized": matmul_packages,
        "gemv": gemv_packages,
        "gemm": gemm_packages,
        "splitk": gemm_splitk_packages,
        "small": column_small_packages,
    }
    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(builders[family], entries[family]) for family in FAMILIES if entries[family]],
        REDUCTIONS,
        columns=COLUMN_SUMS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
