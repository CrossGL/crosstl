"""Run the unchanged pinned float32 quantized-shape tests on CPU and natively.

``test_qmm_shapes``, ``test_small_matrix`` and ``test_non_multiples`` multiply
4-bit, group-64 affine weights by vectors and matrices, transposed and not,
including unaligned, broadcast and batched weights, and compare each product
with ``x @ w_hat`` or its transpose. ``test_throw`` checks QuantizedMatmul's
operand validation and then evaluates one transposed product without a
reference, so its weights are quantized but never dequantized. Through the
portable host the products
reach every QuantizedMatmul route: ``affine_qmv``, ``affine_qmv_fast`` and
``affine_qvm`` below the vector limits, ``affine_qmm_t_splitk`` with its
partition sum for single transposed matrices, and ``affine_qmm_n`` for
untransposed ones. The references take Matmul's gemv and steel GEMM routes. The
error bound is a translated whole-array maximum. Every product package is
translated before the native worker starts; the maximum is packaged on demand at
the widths the run launches.
"""

import math
import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    gemm_layout,
    gemv_layout,
    matmul_layout,
    qmm_layout,
    quantization_layout,
    upstream_runner,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes"
LABEL = "test_qmm_shapes"
UPSTREAM_TESTS = (
    "test_quantized.TestQuantized.test_qmm_shapes",
    "test_quantized.TestQuantized.test_small_matrix",
    "test_quantized.TestQuantized.test_non_multiples",
    "test_quantized.TestQuantized.test_throw",
)
# mx.quantize's defaults, which every product here uses.
GROUP, BITS = 64, 4
# Each quantized weight and the (x shape, transpose) products the tests take
# with it, in the tests' order.
_SMALL = (
    ((3, 1, 256), True),
    ((3, 4, 256), True),
    ((3, 10, 256), True),
    ((3, 1, 8), False),
    ((3, 10, 8), False),
)
WEIGHTS = (
    # test_qmm_shapes
    ((32, 256), (((3, 256), True), ((2, 1, 7, 256), True))),
    ((256, 256), (((3, 256), False), ((2, 1, 7, 256), False))),
    # test_small_matrix
    ((8, 256), _SMALL),
    ((1, 8, 256), _SMALL),
    ((3, 8, 256), _SMALL),
    # test_non_multiples
    (
        (33, 256),
        (((1, 256), True), ((10, 256), True), ((1, 33), False), ((10, 33), False)),
    ),
    (
        (3, 256),
        (((1, 256), True), ((10, 256), True), ((1, 3), False), ((10, 3), False)),
    ),
    ((99, 256), (((129, 256), True),)),
)
# test_throw's one evaluated product, (w shape, x shape, transpose). It is
# quantized and multiplied but has no reference and no error bound.
UNCHECKED = (((32, 512), (10, 512), True),)
REDUCTIONS = ("all_reduce_maxfloat32",)
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 3600}
FAMILIES = ("quantized", "qmm", "gemv", "gemm", "splitk", "small")
# Every affine, steel and column entry counts, so an unexpected route fails.
PREFIXES = ("affine_", *gemv_layout.ENTRY_PREFIXES, "steel_gemm_", "col_reduce_")


def products():
    """``(w_shape, x_shape, transpose)`` for every compared product."""
    for w_shape, uses in WEIGHTS:
        for x_shape, transpose in uses:
            yield w_shape, x_shape, transpose


def quantized_operands(w_shape, x_shape, transpose):
    """QuantizedMatmul::eval_gpu's ``M``, ``N``, ``K`` and batch count.

    ops.cpp broadcasts the batch axes only when both operands have them; a
    row-contiguous ``x`` against 2-D weights folds every row into ``M``.
    """
    k = x_shape[-1]
    n = w_shape[-2] if transpose else w_shape[-1]
    if (w_shape[-1] if transpose else w_shape[-2]) != k:
        raise ValueError("Quantized product operand shapes differ")
    out_size = math.prod(x_shape[:-1]) * n
    if len(w_shape) > 2 and len(x_shape) > 2:
        batch = [max(a, b) for a, b in zip(x_shape[:-2], w_shape[:-2])]
        out_size = math.prod(batch) * x_shape[-2] * n
    m = math.prod(x_shape[:-1]) if len(w_shape) == 2 else x_shape[-2]
    return m, n, k, out_size // m // n


def partition_sums(dtype):
    """The small and looped column sums over ``dtype`` split-K partitions."""
    return (
        f"col_reduce_small_1_reduce_sum{dtype}",
        f"col_reduce_looped_1_32_32_reduce_sum{dtype}",
    )


def quantized_route(
    m, n, k, batches, transpose, *, dtype="float32", group=GROUP, bits=BITS
):
    """The product entry and its partition sum, with a small sum's cache key.

    A 16-bit split-K product sums its partitions in the element type, in the
    small pass below 32 partitions and the looped pass from 32.
    """
    if m < matmul_layout.vector_limit(k, n, transpose):
        entry = matmul_layout.select(dtype, group, bits, m, n, k, batches, transpose)
        if "splitK" in matmul_layout.signature(entry):
            raise ValueError("The quantized upstream tests split no vector product")
        return "quantized", entry, None, None
    entry = qmm_layout.select(dtype, group, bits, m, n, k, batches, transpose)
    if qmm_layout.signature(entry)["kernel"] != "qmm_t_splitk":
        return "qmm", entry, None, None
    split = qmm_layout.split_k(m, n, k, group)
    kind = qmm_layout.partition_sum(m, n, split)
    small, looped = partition_sums(dtype)
    if kind == "looped":
        return "qmm", entry, looped, None
    if kind != "small":
        raise ValueError(f"The quantized upstream tests need no {kind} partition sum")
    workgroup = (*column_small_layout.geometry(m * n, split), 1)
    return "qmm", entry, small, column_small_layout.variant(small, workgroup)


def reference_operands(w_shape, x_shape, transpose):
    """The operands ``x @ w_hat`` or ``x @ swapaxes(w_hat, -1, -2)`` give Matmul.

    The swapped ``w_hat`` is a transposed view of the contiguous dequantized
    weights. ops.cpp's matmul flattens a batched ``x`` against 2-D weights
    into its rows, and otherwise broadcasts both batch shapes, giving a
    single weight matrix a zero batch stride.
    """
    if transpose:
        *batch, n, k = w_shape
        b_shape = (*batch, k, n)
        b_strides = [*upstream_runner.row_strides(w_shape)[:-2], 1, k]
    else:
        b_shape, b_strides = w_shape, upstream_runner.row_strides(w_shape)
    if len(b_shape) == 2:
        a_shape = (math.prod(x_shape[:-1]), x_shape[-1])
        return a_shape, upstream_runner.row_strides(a_shape), b_shape, b_strides
    batch = [max(a, b) for a, b in zip(x_shape[:-2], b_shape[:-2])]
    a_shape = (*batch, *x_shape[-2:])
    b_strides = [
        0 if extent == 1 < size else stride
        for extent, size, stride in zip(b_shape[:-2], batch, b_strides[:-2])
    ] + b_strides[-2:]
    return (
        a_shape,
        upstream_runner.row_strides(a_shape),
        (*batch, *b_shape[-2:]),
        b_strides,
    )


def routes():
    """Each product's quantized family, entry, partition sum and reference route."""
    for w_shape, x_shape, transpose in products():
        operands = quantized_operands(w_shape, x_shape, transpose)
        quantized = quantized_route(*operands, transpose)
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(w_shape, x_shape, transpose)
        )
        if sum(route["copies"].values()):
            raise ValueError("The quantized-shape references need no copies")
        yield quantized, family, route


def unchecked_routes():
    """Each unchecked product's quantized family, entry and partition sum."""
    for w_shape, x_shape, transpose in UNCHECKED:
        yield quantized_route(
            *quantized_operands(w_shape, x_shape, transpose), transpose
        )


def expected_dispatches():
    """The affine, column and matrix-product entries the tests dispatch.

    Each weight is quantized and dequantized once, in launches of at most
    65,535 groups. Each product multiplies through QuantizedMatmul's route,
    and its partition sum when it splits ``K``, and forms the reference with
    Matmul; a split-K reference also launches its accumulation. An unchecked
    product's weights are quantized only, and it launches no reference.
    """
    counts = Counter()
    quantize = f"affine_quantize_float_gs_{GROUP}_b_{BITS}"
    for w_shape, _ in WEIGHTS:
        for operation in ("quantize", "dequantize"):
            counts[
                f"affine_{operation}_float_gs_{GROUP}_b_{BITS}"
            ] += quantization_layout.launches(math.prod(w_shape), GROUP)
    for w_shape, _, _ in UNCHECKED:
        counts[quantize] += quantization_layout.launches(math.prod(w_shape), GROUP)
    for (_, entry, column, _), family, route in routes():
        counts[entry] += 1
        if column is not None:
            counts[column] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    for _, entry, column, _ in unchecked_routes():
        counts[entry] += 1
        if column is not None:
            counts[column] += 1
    return counts


def quantized_routes():
    """Every checked and unchecked product's quantized route."""
    yield from (quantized for quantized, _, _ in routes())
    yield from unchecked_routes()


def column_sums():
    """The static column sum entries the products launch."""
    return tuple(
        sorted(
            {
                column
                for _, _, column, variant in quantized_routes()
                if column and not variant
            }
        )
    )


def package_entries():
    """The entries and small-sum variants to translate ahead, by family."""
    entries = {family: set() for family in FAMILIES}
    for quantized, entry, _, variant in quantized_routes():
        entries[quantized].add(entry)
        if variant is not None:
            entries["small"].add(variant)
    for _, family, route in routes():
        entries[family].add(route["entry"])
        if family == "splitk":
            entries[family].add(route["accum"])
    return {family: sorted(names) for family, names in entries.items()}


COLUMN_SUMS = column_sums()


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
        qmm_packages,
    )

    entries = package_entries()
    builders = {
        "quantized": matmul_packages,
        "qmm": qmm_packages,
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
