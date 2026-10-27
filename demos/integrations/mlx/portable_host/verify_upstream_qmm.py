"""Run the unchanged pinned ``test_qmm`` and ``test_qmm_large_dims`` in float16.

``test_qmm`` quantizes ``K``-by-``N`` or ``N``-by-``K`` weights at every group
size (128, 64, 32) and bit width (2, 4, 8) and multiplies them by ``M``-row
matrices, transposed and not, comparing each product with ``x @ w_hat`` or
``x @ w_hat.T``. It computes in float16 when the default device is the GPU, so
the translated host runs it in float16 while MLX's CPU runs it in float32.
Through the portable host the products reach ``affine_qmv``, ``affine_qmv_fast``
and ``affine_qmv_quad`` below the vector limit, ``affine_qmm_t`` and
``affine_qmm_t_splitk`` with its element-type partition sum for transposed
matrices, and ``affine_qmm_n`` for untransposed ones. The references take
Matmul's float16 steel GEMM and split-K routes. The operands are cast to
float16 and the error bound is a float16 difference, absolute value, maximum
and comparison, all translated: the ``half`` and ``half-arithmetic`` families
and a float16 whole-array maximum packaged on demand.

``test_qmm_large_dims`` multiplies group-64, 4-bit transposed weights with
32,840 columns by 16, 32 and 33 rows, and 64 columns by 33,000 rows: output
extents past 2^15, which once wrapped the 16-bit tile edges of MLX's NAX
kernels. Each product takes ``affine_qmm_t`` against a fused steel GEMM
reference, and the 32,840-column weights quantize and dequantize in two
launches of at most 65,535 groups.
"""

import itertools
import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    gemm_layout,
    gemv_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes import (
    quantized_operands,
    quantized_route,
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmm"
LABEL = "test_qmm"
UPSTREAM_TESTS = (
    "test_quantized.TestQuantized.test_qmm",
    "test_quantized.TestQuantized.test_qmm_large_dims",
)
# The type test_qmm computes in on the GPU device.
DTYPE = "float16"
# test_qmm's product axes, in its itertools.product order.
GROUPS = (128, 64, 32)
BITS = (2, 4, 8)
ROWS = (8, 32, 33, 64)
COLUMNS = (128, 256)
DEPTHS = (128, 256)
TRANSPOSES = (True, False)
# test_qmm_large_dims' transposed group-64, 4-bit products at K = 128: outputs
# with more than 2^15 columns, or rows, in its (M, N) order.
LARGE_DIMS = ((16, 32840), (32, 32840), (33, 32840), (33000, 64))
LARGE_GROUP, LARGE_BITS, LARGE_DEPTH = 64, 4, 128
REDUCTIONS = ("all_reduce_maxfloat16",)
OPERATORS = ("half", "half-arithmetic")
TIMEOUTS = {"random": 600, "packages": 5400, "cpu": 600, "native": 5400}
FAMILIES = ("quantized", "qmm", "gemm", "splitk", "small")
# Every affine, steel and column entry counts, so an unexpected route fails.
PREFIXES = ("affine_", *gemv_layout.ENTRY_PREFIXES, "steel_gemm_", "col_reduce_")


def products():
    """``(group, bits, M, N, K, transpose)`` for every compared product."""
    yield from itertools.product(GROUPS, BITS, ROWS, COLUMNS, DEPTHS, TRANSPOSES)
    for m, n in LARGE_DIMS:
        yield LARGE_GROUP, LARGE_BITS, m, n, LARGE_DEPTH, True


def weight_shape(n, k, transpose):
    return (n, k) if transpose else (k, n)


def routes():
    """Each product's quantization stem, quantized route and reference route."""
    for group, bits, m, n, k, transpose in products():
        w_shape = weight_shape(n, k, transpose)
        operands = quantized_operands(w_shape, (m, k), transpose)
        quantized = quantized_route(
            *operands, transpose, dtype=DTYPE, group=group, bits=bits
        )
        family, route = gemm_layout.matmul_route(
            DTYPE, *reference_operands(w_shape, (m, k), transpose)
        )
        if sum(route["copies"].values()):
            raise ValueError("The test_qmm references need no copies")
        stem = f"{quantization_layout.SOURCE_TYPES[DTYPE]}_gs_{group}_b_{bits}"
        yield (stem, n * k, group), quantized, family, route


def expected_dispatches():
    """The affine, column and matrix-product entries ``test_qmm`` dispatches.

    Each product's weights are quantized and dequantized, in launches of at
    most 65,535 groups. Each product multiplies through QuantizedMatmul's
    route, and its partition sum when it splits ``K``, and forms the reference
    with Matmul; a split-K reference also launches its accumulation.
    """
    counts = Counter()
    for (stem, elements, group), (_, entry, column, _), family, route in routes():
        launches = quantization_layout.launches(elements, group)
        for operation in ("quantize", "dequantize"):
            counts[f"affine_{operation}_{stem}"] += launches
        counts[entry] += 1
        if column is not None:
            counts[column] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries and small-sum variants to translate ahead, by family."""
    entries = {family: set() for family in FAMILIES}
    for _, (quantized, entry, column, variant), family, route in routes():
        if column is not None and variant is None:
            raise ValueError("The test_qmm partition sums are small passes only")
        entries[quantized].add(entry)
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
    """Translate every product, the float16 operators and index the maximum."""
    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        gemm_packages,
        gemm_splitk_packages,
        matmul_packages,
        qmm_packages,
    )

    entries = package_entries()
    builders = {
        "quantized": matmul_packages,
        "qmm": qmm_packages,
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
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
