"""Run the unchanged pinned ``test_fp_qmv_large_output`` on CPU and natively.

The test multiplies one 512-value vector, alone and in a batch of two, by
transposed nvfp4 matrices with 4,096 output rows, and requires
``allclose(y_q, x_t @ swapaxes(w_hat, -1, -2))`` within 1e-2 for bfloat16 and
1e-3 otherwise. It runs in float16, bfloat16 and float32 when the default
device is the GPU, so the translated host runs all three while MLX's CPU runs
float32 alone.

Through the portable host each product reaches the translated
``nvfp4_qmv_fast`` kernel of its element type, batched for the stacked
weights: ``K = 512`` is a multiple of the 4-bit fast alignment and ``N`` of 8.
The float32 weights are quantized once per shape and dequantized to each
element type by the translated block quantization kernels, and each reference
takes Matmul's gemv route in that type. ``allclose`` is a translated chain:
ops.cpp's ``isclose`` subtracts, takes absolute values and compares with
infinity in the element type, converts to float32 against its float32
tolerances, multiplies, adds and compares in float32, combines the infinity
masks with bool logic, and ends in a translated ``and`` reduction.
"""

import math
import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    gemm_layout,
    gemv_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmv import (
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_fp_qmv_large_output"
LABEL = "test_fp_qmv_large_output"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_fp_qmv_large_output",)
MODE = "nvfp4"
# The test's inner and output extents, its batch counts and its GPU dtypes,
# in the test's order.
K, N = 512, 4096
BATCHES = (1, 2)
DTYPES = ("float16", "bfloat16", "float32")
# ``allclose`` ends in an ``and`` over isclose's bool comparison.
REDUCTIONS = ("all_reduce_andbool_",)
# The float16 and bfloat16 casts, arithmetic, comparisons and broadcasts.
OPERATORS = ("half", "half-arithmetic", "bfloat")
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 5400}
FAMILIES = ("quantized", "gemv", "gemm", "splitk")
# Every block-scaled and steel entry counts, so an unexpected route fails the
# trace.
PREFIXES = (
    *(f"{mode}_" for mode in fp_matmul_layout.MODES),
    *gemv_layout.ENTRY_PREFIXES,
    "steel_gemm_",
)


def shapes(batches):
    """``x`` and ``w``'s shapes for ``batches`` stacked products."""
    if batches == 1:
        return (1, K), (N, K)
    return (batches, 1, K), (batches, N, K)


def products():
    """``(dtype, B, x_shape, w_shape)`` for every compared product."""
    for batches in BATCHES:
        x_shape, w_shape = shapes(batches)
        for dtype in DTYPES:
            yield dtype, batches, x_shape, w_shape


def routes():
    """Each product's quantized entry and its reference's Matmul route."""
    for dtype, batches, x_shape, _ in products():
        # One row of x per product: QuantizedMatmul's M is 1 either way.
        quantized = fp_matmul_layout.select(
            MODE, dtype, 1, N, K, batches, transpose=True
        )
        stacked = 0 if batches == 1 else batches
        family, route = gemm_layout.matmul_route(
            dtype, *reference_operands(N, K, stacked, x_shape)
        )
        if sum(route["copies"].values()):
            raise ValueError(f"{LABEL} reference operands need no copies")
        yield quantized, family, route


def quantization_entries(dtype, operation):
    """The entry quantizing float32 ``w``, or dequantizing it to ``dtype``."""
    group, bits = fp_matmul_layout.MODES[MODE]
    source = quantization_layout.SOURCE_TYPES[
        "float32" if operation == "quantize" else dtype
    ]
    return f"{MODE}_{operation}_{source}_gs_{group}_b_{bits}_hgs_false"


def expected_dispatches():
    """The block-scaled and matrix-product entries the test dispatches.

    Each shape's float32 weights quantize once, in launches of at most 65,535
    groups; each element type dequantizes them in as many launches, multiplies
    through QuantizedMatmul's transposed route and forms the reference with
    Matmul, whose split-K form also launches its accumulation.
    """
    counts = Counter()
    group = fp_matmul_layout.MODES[MODE][0]
    for batches in BATCHES:
        _, w_shape = shapes(batches)
        launches = quantization_layout.launches(math.prod(w_shape), group)
        counts[quantization_entries("float32", "quantize")] += launches
    for (dtype, _, _, w_shape), (quantized, family, route) in zip(products(), routes()):
        launches = quantization_layout.launches(math.prod(w_shape), group)
        counts[quantization_entries(dtype, "dequantize")] += launches
        counts[quantized] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries to translate ahead, by family, each sorted."""
    entries = {family: set() for family in FAMILIES}
    for quantized, family, route in routes():
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
    """Translate every product entry and index the on-demand reduction."""
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
        [(builders[family], entries[family]) for family in FAMILIES if entries[family]],
        REDUCTIONS,
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
