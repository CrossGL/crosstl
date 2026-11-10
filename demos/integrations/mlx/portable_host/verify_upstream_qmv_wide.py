"""Run the unchanged pinned ``test_qmv_wide`` on CPU and the translated native backend.

The test multiplies two to nine rows by transposed matrices of 256 or 67 output
rows, unbatched and as three stacked slabs, and requires
``max(abs(y_q - x @ swapaxes(w_hat, -1, -2))) < 1e-3``. It sweeps every affine
group size and width with ``K`` of 128 and 512, the three block-scaled modes
with the same ``K``, and closes with tiny products of 32 or 64 values.

Through the portable host, which reports no Apple GPU generation, quantized.cpp's
``use_qmv_wide`` keeps affine products off ``affine_qmv_wide`` as on MLX's
devices before generation 15: they reach the translated ``affine_qmv_quad``
for ``K`` of 64 or 128 with a power-of-two width, ``affine_qmv_fast`` for
aligned shapes and ``affine_qmv`` otherwise. The block-scaled products reach
``qmv_quad`` for ``K`` of 64 or 128 and ``qmv_wide`` otherwise, with as many
vectors per threadgroup as quantized.cpp's tiling gives each row count. The
references take Matmul's ``steel_gemm_fused`` or split-K routes, the
block-scaled ones after converting the bfloat16 dequantized weights to
float32, and every error bound ends in a translated whole-array maximum.
"""

import sys
from collections import Counter
from itertools import product

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    gemm_layout,
    gemv_layout,
    matmul_layout,
    quantization_layout,
    upstream_runner,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmv import (
    reference_operands,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmv_wide"
LABEL = "test_qmv_wide"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_qmv_wide",)
# The test's row counts, output widths and slab counts, swept as
# product(Ms, Ns, Bs) inside each quantization setting.
ROWS = (2, 3, 4, 5, 6, 7, 9)
WIDTHS = (256, 67)
SLABS = (1, 3)
# The affine sweep's product(group_size, bits, K) and the block-scaled sweep's
# product(mode, K), in the test's order.
AFFINE = ((32, 64, 128), (2, 3, 4, 5, 6, 8), (128, 512))
BLOCK_SCALED = (("mxfp4", "nvfp4", "mxfp8"), (128, 512))
# The tiny (M, K, N) shapes and (bits, group_size, mode) settings, unbatched.
TINY = ((2, 32, 10), (4, 32, 7), (3, 64, 5), (5, 64, 3))
TINY_SETTINGS = ((4, 32, "affine"), (6, 32, "affine"), (4, 16, "nvfp4"))
REDUCTIONS = ("all_reduce_maxfloat32",)
# The block-scaled references convert the bfloat16 dequantized weights to
# float32.
OPERATORS = ("bfloat",)
# The default output type of a block-scaled dequantize.
DEQUANTIZED_TYPE = "bfloat16"
# The job's outer deadline bounds these commands' sum: 1,188 products, each
# quantized, dequantized, multiplied twice and reduced.
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 8400}
# Affine and block-scaled products translate through different builders.
FAMILIES = ("affine", "fp", "gemv", "gemm", "splitk")
# Every quantized and steel entry counts, so an unexpected route fails the
# trace.
PREFIXES = (
    "affine_",
    *(f"{mode}_" for mode in fp_matmul_layout.MODES),
    *gemv_layout.ENTRY_PREFIXES,
    "steel_gemm_",
)


def products():
    """``(mode, group, bits, M, N, K, B)`` for every compared product, in order.

    ``B`` is 1 for an unbatched ``(M, K)`` input against ``(N, K)`` weights
    and 3 for ``(B, M, K)`` against ``(B, N, K)``.
    """
    for group, bits, k in product(*AFFINE):
        for m, n, b in product(ROWS, WIDTHS, SLABS):
            yield "affine", group, bits, m, n, k, b
    for mode, k in product(*BLOCK_SCALED):
        group, bits = fp_matmul_layout.MODES[mode]
        for m, n, b in product(ROWS, WIDTHS, SLABS):
            yield mode, group, bits, m, n, k, b
    for m, k, n in TINY:
        for bits, group, mode in TINY_SETTINGS:
            yield mode, group, bits, m, n, k, 1


def quantized_entry(mode, group, bits, m, n, k, b):
    """QuantizedMatmul's transposed route for one product."""
    if mode == "affine":
        return matmul_layout.select("float32", group, bits, m, n, k, b, transpose=True)
    if fp_matmul_layout.MODES[mode] != (group, bits):
        raise ValueError(f"{LABEL} uses {mode}'s pinned group size and bits")
    return fp_matmul_layout.select(mode, "float32", m, n, k, b, transpose=True)


def routes():
    """Each product's quantized family and entry and its reference's route."""
    for mode, group, bits, m, n, k, b in products():
        quantized = quantized_entry(mode, group, bits, m, n, k, b)
        # x @ swapaxes(w_hat, -1, -2): an unbatched (M, K) input, or B slabs.
        x_shape = (m, k) if b == 1 else (b, m, k)
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(n, k, 0 if b == 1 else b, x_shape)
        )
        if sum(route["copies"].values()):
            raise ValueError(f"{LABEL} reference operands need no copies")
        yield ("affine" if mode == "affine" else "fp"), quantized, family, route


def quantization_entries(mode, group, bits, operation):
    """The entry quantizing float32 ``w``, or dequantizing it to ``w_hat``."""
    if mode == "affine":
        return f"affine_{operation}_float_gs_{group}_b_{bits}"
    dtype = "float32" if operation == "quantize" else DEQUANTIZED_TYPE
    source = quantization_layout.SOURCE_TYPES[dtype]
    return f"{mode}_{operation}_{source}_gs_{group}_b_{bits}_hgs_false"


def expected_dispatches():
    """The quantized and matrix-product entries the test dispatches, with counts.

    Each product quantizes and dequantizes its float32 ``w`` in launches of at
    most 65,535 groups, multiplies through QuantizedMatmul's transposed route
    and forms the reference with Matmul; a split-K reference also launches
    its accumulation.
    """
    counts = Counter()
    for (mode, group, bits, _, n, k, b), (_, quantized, family, route) in zip(
        products(), routes()
    ):
        for operation in ("quantize", "dequantize"):
            counts[
                quantization_entries(mode, group, bits, operation)
            ] += quantization_layout.launches(b * n * k, group)
        counts[quantized] += 1
        counts[route["entry"]] += 1
        if family == "splitk":
            counts[route["accum"]] += 1
    return counts


def package_entries():
    """The entries to translate ahead, by family, each sorted."""
    entries = {family: set() for family in FAMILIES}
    for quantized_family, quantized, family, route in routes():
        entries[quantized_family].add(quantized)
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
        matmul_packages,
    )

    entries = package_entries()
    builders = {
        "affine": matmul_packages,
        "fp": fp_matmul_packages,
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
