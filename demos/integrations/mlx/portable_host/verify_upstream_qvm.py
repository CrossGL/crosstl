"""Run the unchanged pinned ``test_qvm`` on CPU and the translated native backend.

The test multiplies vectors by untransposed affine-quantized matrices and
compares each product with ``x @ dequantize(w)``. Through the portable host the
quantized products reach the translated ``affine_qvm`` kernels, the reference
products the translated ``gemv_t`` kernel, and the error bound a translated
whole-array maximum. Every package is translated before the native worker
starts; the maximum is packaged on demand at the widths the run launches.
"""

import math
import sys
from collections import Counter
from itertools import product

from demos.integrations.mlx.portable_host import (
    gemv_layout,
    matmul_layout,
    quantization_layout,
    upstream_runner,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qvm"
LABEL = "test_qvm"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_qvm",)
# test_qvm's product(group_size, bits, M, N, B), in the test's order.
GRID = ((128, 64, 32), (2, 3, 4, 5, 6, 8), (32, 128, 256), (128, 256, 67), (0, 1, 3, 8))
REDUCTIONS = ("all_reduce_maxfloat32",)
TIMEOUTS = {"random": 600, "packages": 2400, "cpu": 600, "native": 7200}


def products():
    """``(group, bits, M, N, B, x_shape, w_shape)`` for every compared product."""
    for group, bits, m, n, b in product(*GRID):
        if m < group:
            continue
        x_shape = (1, n) if b == 0 else (b, 1, n)
        w_shape = (n, m) if b == 0 else (b, n, m)
        yield group, bits, m, n, b, x_shape, w_shape


def expected_dispatches():
    """The affine and matrix-vector entries the test dispatches, with counts.

    Each product quantizes and dequantizes ``w`` in launches of at most
    65,535 groups, multiplies through
    QuantizedMatmul's untransposed route and forms the reference with Matmul.
    """
    counts = Counter()
    for group, bits, m, n, b, x_shape, w_shape in products():
        for operation in ("quantize", "dequantize"):
            counts[
                f"affine_{operation}_float_gs_{group}_b_{bits}"
            ] += quantization_layout.launches(math.prod(w_shape), group)
        counts[
            matmul_layout.select(
                "float32", group, bits, 1, m, n, max(b, 1), transpose=False
            )
        ] += 1
        route = gemv_layout.route(
            "float32",
            x_shape,
            upstream_runner.row_strides(x_shape),
            w_shape,
            upstream_runner.row_strides(w_shape),
        )
        counts[route["entry"]] += 1
    return counts


def package_entries():
    """The quantized and floating matrix-vector entries to translate ahead."""
    counts = expected_dispatches()
    return (
        sorted(entry for entry in counts if entry.startswith("affine_qvm_")),
        sorted(entry for entry in counts if entry.startswith("gemv_")),
    )


def validate_trace(trace, target, count):
    upstream_runner.check_trace(
        trace,
        target,
        count,
        label=LABEL,
        expected=expected_dispatches(),
        prefixes=("affine_", "gemv_"),
        reductions=REDUCTIONS,
    )


def prepare_packages(root, packages, reductions, jobs):
    """Translate every matrix-vector entry and index the on-demand maximum."""
    from demos.integrations.mlx.portable_host import gemv_packages, matmul_packages

    qvm, gemv = package_entries()
    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(matmul_packages, qvm), (gemv_packages, gemv)],
        REDUCTIONS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
