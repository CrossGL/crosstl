"""Run the unchanged pinned ``test_qmm_vjp`` and ``test_qmm_jvp``.

Both tests quantize float32 weights as 8-bit, group-64 affine groups, once
``N``-by-``K`` (transposed) and once ``K``-by-``N``, and differentiate
``quantized_matmul`` of a ``(2, 64, K)`` input with respect to that input.
``test_qmm_vjp`` pulls back a ``(2, 64, 1024)`` cotangent of ones, which
QuantizedMatmul::vjp multiplies by the weights with the transposition
flipped; ``test_qmm_jvp`` pushes a ``(2, 64, 128)`` tangent of ones forward,
which QuantizedMatmul::jvp multiplies with the transposition kept. Each test
compares the derivative with the same product written out, through
``allclose``.

Neither test evaluates the primal output, so every quantized product is a
derivative or its explicit counterpart. The weights are 2-D and the inputs
row-contiguous, so QuantizedMatmul folds both batches into ``M = 128`` rows:
past the vector limit, an untransposed product reaches ``affine_qmm_n`` and
a transposed one ``affine_qmm_t_splitk`` with its float32 partition sum. The
comparisons end in a translated whole-array conjunction.
"""

import math
import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    gemv_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes import (
    quantized_operands,
    quantized_route,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmm_vjp"
LABEL = "test_qmm_vjp"
UPSTREAM_TESTS = (
    "test_quantized.TestQuantized.test_qmm_vjp",
    "test_quantized.TestQuantized.test_qmm_jvp",
)
DTYPE = "float32"
GROUP, BITS = 64, 8
# Each test's (M, N, K) and the batch of its inputs.
SHAPES = {"vjp": (64, 1024, 512), "jvp": (64, 128, 128)}
BATCH = 2
# The forward product's transposition, in the tests' order.
TRANSPOSES = (True, False)
# Each derivative is compared with the same product written out.
PRODUCTS_PER_CHECK = 2
REDUCTIONS = ("all_reduce_andbool_",)
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 3600}
FAMILIES = ("qmm", "small")
# Every affine, steel and column entry counts, so an unexpected route, such
# as an evaluated primal or a dense reference, fails the trace.
PREFIXES = ("affine_", *gemv_layout.ENTRY_PREFIXES, "steel_gemm_", "col_reduce_")


def products():
    """``(test, w_shape, x_shape, transpose)`` of each derivative product.

    ``transpose`` is the derivative's own: the vjp flips the forward one, and
    its left operand is the cotangent, whose last axis is the forward ``N``.
    """
    for test, (m, n, k) in SHAPES.items():
        for forward in TRANSPOSES:
            w_shape = (n, k) if forward else (k, n)
            if test == "vjp":
                yield test, w_shape, (BATCH, m, n), not forward
            else:
                yield test, w_shape, (BATCH, m, k), forward


def routes():
    """Each product's weight element count and its quantized route."""
    for _, w_shape, x_shape, transpose in products():
        operands = quantized_operands(w_shape, x_shape, transpose)
        route = quantized_route(
            *operands, transpose, dtype=DTYPE, group=GROUP, bits=BITS
        )
        if route[0] != "qmm":
            raise ValueError(f"{LABEL} products are matrix-matrix products")
        yield math.prod(w_shape), route


def quantize_entry():
    stem = f"{quantization_layout.SOURCE_TYPES[DTYPE]}_gs_{GROUP}_b_{BITS}"
    return f"affine_quantize_{stem}"


def expected_dispatches():
    """The affine and column entries the tests dispatch, with counts.

    Each weight matrix is quantized once, in launches of at most 65,535
    groups, and never dequantized. Its derivative and the explicit product
    each launch the quantized entry, and its partition sum when it splits
    ``K``.
    """
    counts = Counter()
    for elements, (_, entry, column, _) in routes():
        counts[quantize_entry()] += quantization_layout.launches(elements, GROUP)
        counts[entry] += PRODUCTS_PER_CHECK
        if column is not None:
            counts[column] += PRODUCTS_PER_CHECK
    return counts


def package_entries():
    """The entries and small-sum variants to translate ahead, by family."""
    entries = {family: set() for family in FAMILIES}
    for _, (_, entry, column, variant) in routes():
        if column is not None and variant is None:
            raise ValueError(f"{LABEL} partition sums are small passes only")
        entries["qmm"].add(entry)
        if variant is not None:
            entries["small"].add(variant)
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
    """Translate every product and partition sum and index the conjunction."""
    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        qmm_packages,
    )

    entries = package_entries()
    builders = {"qmm": qmm_packages, "small": column_small_packages}
    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(builders[family], entries[family]) for family in FAMILIES if entries[family]],
        REDUCTIONS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
