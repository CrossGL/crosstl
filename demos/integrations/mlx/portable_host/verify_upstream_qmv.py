"""Run the unchanged pinned ``test_qmv`` on CPU and the translated native backend.

The test multiplies vectors by transposed affine-quantized matrices and compares
each product with ``x @ swapaxes(dequantize(w), -1, -2)``. Through the portable
host the quantized products reach the translated ``affine_qmv_quad``,
``affine_qmv_fast`` and ``affine_qmv`` kernels. The reference products take all
three of Matmul's routes: batched vectors the translated ``gemv`` kernels, and
the unbatched test's three rows ``steel_gemm_fused`` or, when ``K`` covers the
output width, ``steel_gemm_splitk`` and its accumulation. The error bound is a
translated whole-array maximum. Every product package is translated before the
native worker starts; the maximum is packaged on demand at the widths the run
launches.
"""

import math
import sys
from collections import Counter
from itertools import product

from demos.integrations.mlx.portable_host import (
    gemm_layout,
    gemv_layout,
    matmul_layout,
    quantization_layout,
    upstream_runner,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmv"
LABEL = "test_qmv"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_qmv",)
# test_qmv's product(group_size, bits, M, N, B), in the test's order.
GRID = ((128, 64, 32), (2, 3, 4, 5, 6, 8), (256, 512, 67), (64, 256), (0, 1, 3, 8))
# The unbatched test multiplies three vectors at once.
UNBATCHED_ROWS = 3
REDUCTIONS = ("all_reduce_maxfloat32",)
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 7200}
FAMILIES = ("quantized", "gemv", "gemm", "splitk")
# Every affine and steel entry counts, so an unexpected route fails the trace.
PREFIXES = ("affine_", *gemv_layout.ENTRY_PREFIXES, "steel_gemm_")


def products():
    """``(group, bits, M, N, B, x_shape, w_shape)`` for every compared product."""
    for group, bits, m, n, b in product(*GRID):
        if group > n:
            continue
        x_shape = (UNBATCHED_ROWS, 1, n) if b == 0 else (b, 1, n)
        w_shape = (m, n) if b == 0 else (b, m, n)
        yield group, bits, m, n, b, x_shape, w_shape


def reference_operands(m, n, b, x_shape):
    """The operands ``x @ swapaxes(w_hat, -1, -2)`` gives Matmul.

    ops.cpp's matmul flattens a 3-D ``x`` against 2-D weights into its rows;
    batched operands already share their batch shape. The swapped ``w_hat``
    is a transposed view of the contiguous dequantized weights.
    """
    if b == 0:
        a_shape, b_shape, b_strides = (x_shape[0], n), (n, m), [1, n]
    else:
        a_shape, b_shape, b_strides = x_shape, (b, n, m), [m * n, 1, n]
    return a_shape, upstream_runner.row_strides(a_shape), b_shape, b_strides


def routes():
    """Each product's quantized entry and its reference's Matmul route."""
    for group, bits, m, n, b, x_shape, _ in products():
        rows = UNBATCHED_ROWS if b == 0 else 1
        quantized = matmul_layout.select(
            "float32", group, bits, rows, m, n, max(b, 1), transpose=True
        )
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(m, n, b, x_shape)
        )
        if sum(route["copies"].values()):
            raise ValueError("test_qmv's reference operands need no copies")
        yield group, bits, quantized, family, route


def expected_dispatches():
    """The affine and matrix-product entries the test dispatches, with counts.

    Each product quantizes and dequantizes ``w`` in launches of at most
    65,535 groups, multiplies through QuantizedMatmul's transposed route and
    forms the reference with Matmul; a split-K reference also launches its
    accumulation.
    """
    counts = Counter()
    for (*_, w_shape), (group, bits, quantized, family, route) in zip(
        products(), routes()
    ):
        for operation in ("quantize", "dequantize"):
            counts[
                f"affine_{operation}_float_gs_{group}_b_{bits}"
            ] += quantization_layout.launches(math.prod(w_shape), group)
        counts[quantized] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries to translate ahead, by family, each sorted."""
    entries = {family: set() for family in FAMILIES}
    for _, _, quantized, family, route in routes():
        entries["quantized"].add(quantized)
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
    """Translate every product entry and index the on-demand maximum."""
    from demos.integrations.mlx.portable_host import (
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
    }
    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(builders[family], entries[family]) for family in FAMILIES],
        REDUCTIONS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
