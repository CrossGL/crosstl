"""Run the unchanged pinned ``test_fp_qmm_non_multiple_of_32`` on CPU and natively.

The test multiplies 33 or 50 rows by mxfp4, mxfp8 and nvfp4 matrices whose
inner or output dimension leaves a 16-wide tail of a 32-wide tile, transposed
and not, twice each, and requires the two products to agree bit for bit and
to stay within a per-type bound of ``x @ dequantize(w)``. It runs in float32,
float16 and bfloat16 when the default device is the GPU, so the translated
host runs all three while MLX's CPU runs float32 alone.

Through the portable host the products reach the translated block-scaled
matrix-matrix kernels (``fp_qmm_layout``): ``fp_qmm_t`` for transposed
weights whose ``K`` admits no split, ``fp_qmm_n`` for untransposed ones, and
``fp_qmm_t_splitk`` for the mxfp4 and mxfp8 ``K = 1056`` control, whose 33
partitions the translated ``col_reduce_looped`` sum adds in the element type.
The weights are quantized by the translated block quantization kernels and
dequantized to bfloat16 (ops.cpp's ``validate_mode_with_type``), so the
float32 and float16 references convert both operands to float32 through the
translated ``half`` and ``bfloat`` conversions while the bfloat16 reference
multiplies in bfloat16. The equality is a translated comparison and ``and``
reduction; the error bounds are translated subtractions, absolute values and
whole-array maxima in float32 or bfloat16.
"""

import math
import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    fp_qmm_layout,
    gemm_layout,
    gemv_layout,
    qmm_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes import (
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_fp_qmm"
LABEL = "test_fp_qmm_non_multiple_of_32"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_fp_qmm_non_multiple_of_32",)
# _fp_tail_dtypes on the GPU device, in the test's order.
DTYPES = ("float32", "float16", "bfloat16")
# (M, K, N, transpose): the nvfp4 tails, then the mxfp4 and mxfp8 controls.
NVFP4_SHAPES = (
    (50, 1040, 128, True),
    (33, 528, 64, True),
    (50, 512, 1040, False),
    (50, 1040, 1040, False),
    (50, 16, 128, True),
)
CONTROL_SHAPES = ((50, 1056, 128, True), (50, 512, 1056, False))
CONTROL_MODES = ("mxfp4", "mxfp8")
# The default output type of a block-scaled dequantize.
DEQUANTIZED_TYPE = "bfloat16"
# quantized_matmul runs twice per check, for the bitwise agreement.
REPEATS = 2
# The float32 and float16 bounds take float32 maxima, the bfloat16 one a
# bfloat16 maximum; array_equal ends in an ``and`` over the comparison.
REDUCTIONS = ("all_reduce_maxfloat32", "all_reduce_maxbfloat16", "all_reduce_andbool_")
# float16 and bfloat16 conversions, comparisons and arithmetic.
OPERATORS = ("half", "half-arithmetic", "bfloat")
TIMEOUTS = {"random": 600, "packages": 5400, "cpu": 600, "native": 5400}
FAMILIES = ("qmm", "gemv", "gemm", "splitk")
# Every block-scaled, steel and column entry counts, so an unexpected route
# fails the trace.
PREFIXES = (
    *(f"{mode}_" for mode in fp_matmul_layout.MODES),
    *gemv_layout.ENTRY_PREFIXES,
    "steel_gemm_",
    "col_reduce_",
)


def products():
    """``(dtype, mode, M, K, N, transpose)`` for every check, in the test's order."""
    for dtype in DTYPES:
        for m, k, n, transpose in NVFP4_SHAPES:
            yield dtype, "nvfp4", m, k, n, transpose
        for mode in CONTROL_MODES:
            for m, k, n, transpose in CONTROL_SHAPES:
                yield dtype, mode, m, k, n, transpose


def weight_shape(k, n, transpose):
    return (n, k) if transpose else (k, n)


def reference_type(dtype):
    """``x @ w_hat``'s type: ``dtype`` promoted with the bfloat16 ``w_hat``."""
    return "bfloat16" if dtype == "bfloat16" else "float32"


def partition_sum(dtype, mode, m, n, k, quantized):
    """The column sum reduce.cpp runs over a split-K product's partitions.

    Returns ``None`` for an unsplit product. qmm_splitk's 33 partitions here
    take the looped pass.
    """
    if fp_qmm_layout.signature(quantized)["kernel"] != "qmm_t_splitk":
        return None
    split = qmm_layout.split_k(m, n, k, fp_matmul_layout.MODES[mode][0])
    if qmm_layout.partition_sum(m, n, split) != "looped":
        raise ValueError(f"{LABEL} partition sums are looped passes only")
    return f"col_reduce_looped_1_32_32_reduce_sum{dtype}"


def routes():
    """Each check's quantized entry, partition sum and reference route."""
    for dtype, mode, m, k, n, transpose in products():
        quantized = fp_qmm_layout.select(mode, dtype, m, n, k, 1, transpose)
        w_shape = weight_shape(k, n, transpose)
        family, route = gemm_layout.matmul_route(
            reference_type(dtype), *reference_operands(w_shape, (m, k), transpose)
        )
        if sum(route["copies"].values()):
            raise ValueError(f"{LABEL} reference operands need no copies")
        column = partition_sum(dtype, mode, m, n, k, quantized)
        yield mode, quantized, column, family, route


def quantization_entries(mode, dtype, operation):
    """The entry quantizing ``dtype`` weights, or dequantizing them to bfloat16."""
    group, bits = fp_matmul_layout.MODES[mode]
    source = dtype if operation == "quantize" else DEQUANTIZED_TYPE
    source = quantization_layout.SOURCE_TYPES[source]
    return f"{mode}_{operation}_{source}_gs_{group}_b_{bits}_hgs_false"


def expected_dispatches():
    """The block-scaled, column and matrix-product entries the test dispatches.

    Each check quantizes and dequantizes ``w`` in launches of at most 65,535
    groups, multiplies twice through QuantizedMatmul's matrix-matrix route
    (each with its partition sum after a split) and forms the reference with
    Matmul; a split-K reference also launches its accumulation.
    """
    counts = Counter()
    for (dtype, _, _, k, n, transpose), (mode, quantized, column, family, route) in zip(
        products(), routes()
    ):
        group = fp_matmul_layout.MODES[mode][0]
        launches = quantization_layout.launches(
            math.prod(weight_shape(k, n, transpose)), group
        )
        for operation in ("quantize", "dequantize"):
            counts[quantization_entries(mode, dtype, operation)] += launches
        counts[quantized] += REPEATS
        if column is not None:
            counts[column] += REPEATS
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def column_sums():
    """The looped partition sums the split-K products need, sorted."""
    return tuple(sorted({column for *_, column, _, _ in routes() if column}))


def package_entries():
    """The product entries to translate ahead, by family, each sorted."""
    entries = {family: set() for family in FAMILIES}
    for _, quantized, _, family, route in routes():
        entries["qmm"].add(quantized)
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
    """Translate every product and column entry and index the reductions."""
    from demos.integrations.mlx.portable_host import (
        fp_qmm_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
    )

    entries = package_entries()
    builders = {
        "qmm": fp_qmm_packages,
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
        columns=COLUMN_SUMS,
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
