"""Run the unchanged pinned ``test_fp_qvm`` on CPU and the translated native backend.

The test multiplies vectors by untransposed mxfp4, nvfp4 and mxfp8 matrices and
compares each product with ``x @ dequantize(w)``. Through the portable host the
quantized products reach the translated ``fp_qvm`` kernels while the inner
dimension is below 1024, and the closing case's 16,384 values reach
``fp_qvm_split_k``, whose 32 partitions the translated ``col_reduce_looped``
sum adds as MLX's reduce.cpp does. The weights are quantized and dequantized
by the translated block quantization kernels. The reference products take
Matmul's routes, and the error bound is a translated whole-array maximum.
Every product and column package is translated before the native worker
starts; the maximum is packaged on demand at the widths the run launches.

``dequantize`` gives these modes bfloat16 weights by default (ops.cpp's
``validate_mode_with_type``), so the reference's Matmul first converts
``w_hat`` to float32 through the translated bfloat16 conversion.
"""

import math
import sys
from collections import Counter
from itertools import product

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    fp_matmul_layout,
    gemm_layout,
    gemv_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_fp_qmv import (
    MODES,
    OPERATORS,
    quantization_entries,
)
from demos.integrations.mlx.portable_host.verify_upstream_qvm_splitk import (
    COLUMN_SUMS,
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_fp_qvm"
LABEL = "test_fp_qvm"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_fp_qvm",)
# test_fp_qvm's product(M, N, B) and its appended split-K case, in the
# test's order; M is the output width and N the inner dimension.
GRID = ((32, 128, 256), (128, 256, 67), (0, 1, 3, 8))
SPLIT = (128, 16384, 0)
REDUCTIONS = ("all_reduce_maxfloat32",)
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 7200}
FAMILIES = ("quantized", "gemv", "gemm", "splitk", "small")
# Every block-quantized, steel and column entry counts, so an unexpected
# route fails the trace.
PREFIXES = (
    *(f"{mode}_" for mode in fp_matmul_layout.MODES),
    *gemv_layout.ENTRY_PREFIXES,
    "steel_gemm_",
    "col_reduce_",
)


def products():
    """``(mode, M, N, B, x_shape, w_shape)`` for every compared product."""
    for m, n, b in (*product(*GRID), SPLIT):
        for mode in MODES:
            x_shape = (1, n) if b == 0 else (b, 1, n)
            w_shape = (n, m) if b == 0 else (b, n, m)
            yield mode, m, n, b, x_shape, w_shape


def column_sum(quantized, rows, n):
    """The column reduction reduce.cpp runs over a split-K product's partitions.

    Returns ``None`` for an unsplit product, else ``(entry, small_variant)``
    with a small sum's cache key or ``None``.
    """
    split = fp_matmul_layout.signature(quantized).get("splitK")
    if split is None:
        return None
    if split >= 32:
        return COLUMN_SUMS[0], None
    entry = "col_reduce_small_1_reduce_sumfloat32"
    workgroup = (*column_small_layout.geometry(rows * n, split), 1)
    return entry, column_small_layout.variant(entry, workgroup)


def routes():
    """Each product's quantized entry, column sum and reference route."""
    for mode, m, n, b, x_shape, w_shape in products():
        quantized = fp_matmul_layout.select(
            mode, "float32", 1, m, n, max(b, 1), transpose=False
        )
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(x_shape, w_shape)
        )
        if sum(route["copies"].values()):
            raise ValueError("test_fp_qvm's reference operands need no copies")
        yield mode, quantized, column_sum(quantized, 1, m), family, route


def expected_dispatches():
    """The block-quantized, column and matrix-product entries the test dispatches.

    Each product quantizes and dequantizes ``w`` in launches of at most
    65,535 groups, multiplies through QuantizedMatmul's untransposed route
    (with its partition sum after a split) and forms the reference with
    Matmul; a split-K reference also launches its accumulation.
    """
    counts = Counter()
    for (*_, w_shape), (mode, quantized, column, family, route) in zip(
        products(), routes()
    ):
        group = fp_matmul_layout.MODES[mode][0]
        for operation in ("quantize", "dequantize"):
            counts[
                quantization_entries(mode, operation)
            ] += quantization_layout.launches(math.prod(w_shape), group)
        counts[quantized] += 1
        if column is not None:
            counts[column[0]] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries and small-sum variants to translate ahead, by family."""
    entries = {family: set() for family in FAMILIES}
    for _, quantized, column, family, route in routes():
        entries["quantized"].add(quantized)
        if column is not None and column[1] is not None:
            entries["small"].add(column[1])
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
        fp_matmul_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
    )

    entries = package_entries()
    builders = {
        "quantized": fp_matmul_packages,
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
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
