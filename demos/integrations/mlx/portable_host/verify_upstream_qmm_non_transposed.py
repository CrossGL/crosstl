"""Run the unchanged pinned ``test_qmm_non_transposed`` on CPU and natively.

The test multiplies ``M``-row activations by untransposed ``K``-by-``N``
weights, the product a quantized linear layer's backward pass runs, and
requires ``max(abs(y_q - x @ w_hat))`` below 1.5e-3 for the 16-bit types. It
sweeps ``M`` over values that leave partial 32-row tiles for affine group-64
and group-128 4-bit and group-64 8-bit weights and for mxfp4, nvfp4 and mxfp8,
then transformer-sized ``K`` and ``N``, batched activations, 33,000 rows, and
group-32 2-, 4- and 8-bit weights with one to four 64-deep reduction tiles. It
runs in float16 and bfloat16 when the default device is the GPU, so the
translated host runs both while MLX's CPU runs float32 alone.

Through the portable host the products reach the translated ``affine_qvm`` and
block-scaled ``qvm`` kernels below the untransposed vector limit of four rows,
and ``affine_qmm_n`` and the block-scaled ``qmm_n`` kernels from there on;
batched activations against 2-D weights fold into their rows. Each weight
matrix is quantized from the test's type by the translated quantization
kernels. Affine weights dequantize to that type; block-scaled ones to
bfloat16, so a float16 block-scaled reference converts both operands to
float32 through the translated ``half`` and ``bfloat`` conversions. The
references take Matmul's gemv, steel GEMM and split-K routes, and every error
bound ends in a translated whole-array maximum in float16, bfloat16 or float32.
"""

import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    fp_qmm_layout,
    gemm_layout,
    gemv_layout,
    matmul_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes import (
    quantized_operands,
    quantized_route,
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmm_non_transposed"
LABEL = "test_qmm_non_transposed"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_qmm_non_transposed",)
# The test's GPU dtypes and block-scaled modes, in its order.
DTYPES = ("float16", "bfloat16")
MODES = ("mxfp4", "nvfp4", "mxfp8")
# The M sweep: (M, K, N) at K = 512 and N = 1024 for each affine
# (group, bits) and then each mode.
SWEEP_ROWS = (1, 2, 31, 32, 33, 63, 64, 65, 96, 97, 100, 127, 128, 129)
SWEEP_AFFINE = ((64, 4), (128, 4), (64, 8))
SWEEP_DEPTH, SWEEP_WIDTH = 512, 1024
# Transformer-sized (K, N) with group-64 4-bit affine weights at each M. The
# block-scaled checks of this loop reuse the sweep's K and N and the last M.
TRANSFORMER_SHAPES = ((2048, 2048), (512, 2048), (2048, 512), (11008, 2048))
TRANSFORMER_ROWS = (100, 256)
# Batched activations; the block-scaled checks reuse the last M.
BATCHES = ((2,), (2, 3))
BATCH_ROWS = (33, 250)
# M > 2**15; the block-scaled check reuses the last mode of the loop before.
LARGE = (33000, 128, 64)
# K of one to four reduction tiles at N = 128, group-32 affine weights.
DEPTH_ROWS = (8, 33, 65)
DEPTHS = (64, 128, 256)
DEPTH_BITS = (2, 4, 8)
DEPTH_WIDTH, DEPTH_GROUP = 128, 32
DEFAULT_GROUP, DEFAULT_BITS = 64, 4
# The default output type of a block-scaled dequantize.
DEQUANTIZED_TYPE = "bfloat16"
# The error bounds' whole-array maxima: affine float16, every bfloat16 one,
# and float16 against a promoted float32 block-scaled reference.
REDUCTIONS = (
    "all_reduce_maxfloat16",
    "all_reduce_maxbfloat16",
    "all_reduce_maxfloat32",
)
# float16 and bfloat16 random scaling, conversions, arithmetic and comparisons.
OPERATORS = ("half", "half-arithmetic", "bfloat")
# The job's outer deadline bounds these commands' sum.
TIMEOUTS = {"random": 600, "packages": 5400, "cpu": 900, "native": 6000}
FAMILIES = ("quantized", "fp", "qmm", "fpqmm", "gemv", "gemm", "splitk")
# Every quantized and matrix-product entry counts, so an unexpected route
# fails the trace.
PREFIXES = (
    "affine_",
    *(f"{mode}_" for mode in fp_matmul_layout.MODES),
    *gemv_layout.ENTRY_PREFIXES,
    "steel_gemm_",
)


def products():
    """``(dtype, mode, group, bits, M, K, N, batch)`` for every check, in order."""
    for dtype in DTYPES:
        yield from _dtype_products(dtype)


def _fp(dtype, mode, m, k, n, batch=()):
    group, bits = fp_matmul_layout.MODES[mode]
    return dtype, mode, group, bits, m, k, n, batch


def _dtype_products(dtype):
    for m in SWEEP_ROWS:
        for group, bits in SWEEP_AFFINE:
            yield dtype, "affine", group, bits, m, SWEEP_DEPTH, SWEEP_WIDTH, ()
        for mode in MODES:
            yield _fp(dtype, mode, m, SWEEP_DEPTH, SWEEP_WIDTH)
    for k, n in TRANSFORMER_SHAPES:
        for m in TRANSFORMER_ROWS:
            yield dtype, "affine", DEFAULT_GROUP, DEFAULT_BITS, m, k, n, ()
        for mode in MODES:
            yield _fp(dtype, mode, TRANSFORMER_ROWS[-1], SWEEP_DEPTH, SWEEP_WIDTH)
    for batch in BATCHES:
        for m in BATCH_ROWS:
            yield (
                dtype,
                "affine",
                DEFAULT_GROUP,
                DEFAULT_BITS,
                m,
                SWEEP_DEPTH,
                SWEEP_WIDTH,
                batch,
            )
        for mode in MODES:
            yield _fp(dtype, mode, BATCH_ROWS[-1], SWEEP_DEPTH, SWEEP_WIDTH, batch)
    yield dtype, "affine", DEFAULT_GROUP, DEFAULT_BITS, *LARGE, ()
    yield _fp(dtype, MODES[-1], *LARGE)
    for m in DEPTH_ROWS:
        for k in DEPTHS:
            for bits in DEPTH_BITS:
                yield dtype, "affine", DEPTH_GROUP, bits, m, k, DEPTH_WIDTH, ()
            for mode in MODES:
                yield _fp(dtype, mode, m, k, DEPTH_WIDTH)


def reference_type(dtype, mode):
    """``x @ w_hat``'s type: ``dtype`` promoted with the dequantized weights."""
    if mode == "affine" or dtype == DEQUANTIZED_TYPE:
        return dtype
    return "float32"


def quantized_entry(dtype, mode, group, bits, m, n, k, batches):
    """QuantizedMatmul's untransposed route: its family and entry."""
    if mode == "affine":
        family, entry, column, _ = quantized_route(
            m, n, k, batches, False, dtype=dtype, group=group, bits=bits
        )
        if column is not None:
            raise ValueError(f"{LABEL} splits no product")
        return family, entry
    if m < matmul_layout.vector_limit(k, n, False):
        return "fp", fp_matmul_layout.select(mode, dtype, m, n, k, batches, False)
    return "fpqmm", fp_qmm_layout.select(mode, dtype, m, n, k, batches, False)


def routes():
    """Each check's quantized family and entry and its reference's route."""
    for dtype, mode, group, bits, m, k, n, batch in products():
        x_shape, w_shape = (*batch, m, k), (k, n)
        rows, columns, depth, batches = quantized_operands(w_shape, x_shape, False)
        family, entry = quantized_entry(
            dtype, mode, group, bits, rows, columns, depth, batches
        )
        reference, route = gemm_layout.matmul_route(
            reference_type(dtype, mode), *reference_operands(w_shape, x_shape, False)
        )
        if sum(route["copies"].values()):
            raise ValueError(f"{LABEL} reference operands need no copies")
        yield family, entry, reference, route


def quantization_entries(dtype, mode, group, bits, operation):
    """The entry quantizing ``dtype`` weights, or dequantizing them."""
    stem = f"gs_{group}_b_{bits}"
    if mode == "affine":
        return f"affine_{operation}_{quantization_layout.SOURCE_TYPES[dtype]}_{stem}"
    source = dtype if operation == "quantize" else DEQUANTIZED_TYPE
    source = quantization_layout.SOURCE_TYPES[source]
    return f"{mode}_{operation}_{source}_{stem}_hgs_false"


def expected_dispatches():
    """The quantized and matrix-product entries the test dispatches, with counts.

    Each check quantizes and dequantizes its ``K``-by-``N`` weights in
    launches of at most 65,535 groups, multiplies once through
    QuantizedMatmul's untransposed route and forms the reference with Matmul;
    a split-K reference also launches its accumulation.
    """
    counts = Counter()
    for (dtype, mode, group, bits, _, k, n, _), (_, entry, family, route) in zip(
        products(), routes()
    ):
        launches = quantization_layout.launches(k * n, group)
        for operation in ("quantize", "dequantize"):
            counts[
                quantization_entries(dtype, mode, group, bits, operation)
            ] += launches
        counts[entry] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries to translate ahead, by family, each sorted."""
    entries = {family: set() for family in FAMILIES}
    for quantized, entry, family, route in routes():
        entries[quantized].add(entry)
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
    # Each type's bound reaches its own maximum; every one must run natively.
    ran = {record["entry"] for record in trace}
    if not set(REDUCTIONS) <= ran:
        raise RuntimeError(
            f"Upstream {LABEL} did not evaluate every error bound natively"
        )


def prepare_packages(root, packages, reductions, jobs):
    """Translate every product entry and index the on-demand maxima."""
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        fp_qmm_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
        qmm_packages,
    )

    entries = package_entries()
    builders = {
        "quantized": matmul_packages,
        "fp": fp_matmul_packages,
        "qmm": qmm_packages,
        "fpqmm": fp_qmm_packages,
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
