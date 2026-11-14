"""Run the unchanged pinned ``test_fp_qmv`` on CPU and the translated native backend.

The test multiplies vectors by transposed mxfp4, nvfp4 and mxfp8 matrices and
compares each product with ``x @ swapaxes(dequantize(w), -1, -2)``. Through the
portable host the quantized products reach the translated ``fp_qmv_wide``
kernels for the unbatched test's three rows and ``fp_qmv_quad``,
``fp_qmv_fast`` and ``fp_qmv`` for single vectors; the weights are quantized
and dequantized by the translated block quantization kernels. The reference
products take Matmul's routes as in ``verify_upstream_qmv``, and the error
bound is a translated whole-array maximum. Every product package is translated
before the native worker starts; the maximum is packaged on demand at the
widths the run launches.

``dequantize`` gives these modes bfloat16 weights by default (ops.cpp's
``validate_mode_with_type``), so the reference casts ``swapaxes(w_hat)`` to
float32 through the translated bfloat16 conversion, keeping its layout as
AsType's vector copy does.
"""

import math
import sys
from collections import Counter
from itertools import product

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    gemm_layout,
    gemv_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmv import (
    UNBATCHED_ROWS,
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_fp_qmv"
LABEL = "test_fp_qmv"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_fp_qmv",)
# test_fp_qmv's product(M, N, B) and modes, in the test's order.
GRID = ((256, 512, 67, 5, 7), (64, 256), (0, 1, 3, 8))
MODES = ("mxfp4", "nvfp4", "mxfp8")
# The closing nvfp4 case: one vector of 48 values against 128 rows.
TAIL = ("nvfp4", 128, 48)
REDUCTIONS = ("all_reduce_maxfloat32",)
# The reference converts the bfloat16 dequantized weights to float32.
OPERATORS = ("bfloat",)
# The default output type of a block-scaled dequantize.
DEQUANTIZED_TYPE = "bfloat16"
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 7200}
FAMILIES = ("quantized", "gemv", "gemm", "splitk")
# Every block-quantized and steel entry counts, so an unexpected route fails
# the trace.
PREFIXES = (
    *(f"{mode}_" for mode in fp_matmul_layout.MODES),
    *gemv_layout.ENTRY_PREFIXES,
    "steel_gemm_",
)


def products():
    """``(mode, M, N, B, x_shape, w_shape)`` for every compared product."""
    for m, n, b in product(*GRID):
        for mode in MODES:
            x_shape = (UNBATCHED_ROWS, 1, n) if b == 0 else (b, 1, n)
            w_shape = (m, n) if b == 0 else (b, m, n)
            yield mode, m, n, b, x_shape, w_shape
    mode, m, n = TAIL
    yield mode, m, n, 0, (1, n), (m, n)


def routes():
    """Each product's quantized entry and its reference's Matmul route."""
    for mode, m, n, b, x_shape, _ in products():
        # QuantizedMatmul flattens a row-contiguous x against 2-D weights.
        rows = math.prod(x_shape[:-1]) if b == 0 else 1
        quantized = fp_matmul_layout.select(
            mode, "float32", rows, m, n, max(b, 1), transpose=True
        )
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(m, n, b, x_shape)
        )
        if sum(route["copies"].values()):
            raise ValueError("test_fp_qmv's reference operands need no copies")
        yield mode, quantized, family, route


def quantization_entries(mode, operation):
    """The entry quantizing float32 ``w``, or dequantizing it to bfloat16."""
    group, bits = fp_matmul_layout.MODES[mode]
    dtype = "float32" if operation == "quantize" else DEQUANTIZED_TYPE
    source = quantization_layout.SOURCE_TYPES[dtype]
    return f"{mode}_{operation}_{source}_gs_{group}_b_{bits}_hgs_false"


def expected_dispatches():
    """The block-quantized and matrix-product entries the test dispatches.

    Each product quantizes and dequantizes ``w`` in launches of at most
    65,535 groups, multiplies through QuantizedMatmul's transposed route and
    forms the reference with Matmul; a split-K reference also launches its
    accumulation.
    """
    counts = Counter()
    for (*_, w_shape), (mode, quantized, family, route) in zip(products(), routes()):
        group = fp_matmul_layout.MODES[mode][0]
        for operation in ("quantize", "dequantize"):
            counts[
                quantization_entries(mode, operation)
            ] += quantization_layout.launches(math.prod(w_shape), group)
        counts[quantized] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries to translate ahead, by family, each sorted."""
    entries = {family: set() for family in FAMILIES}
    for _, quantized, family, route in routes():
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
    }
    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(builders[family], entries[family]) for family in FAMILIES],
        REDUCTIONS,
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
