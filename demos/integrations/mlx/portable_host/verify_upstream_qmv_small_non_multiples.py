"""Run the unchanged pinned ``test_qmv_small_non_multiples`` on CPU and natively.

The test multiplies one, two or four 32-value rows by transposed matrices of
three, five, seven or ten output rows, quantized as 4-bit and 6-bit affine
groups of 32 and as nvfp4, and requires
``max(abs(y_q - x @ swapaxes(w_hat, -1, -2))) < 1e-3``.

No output width is a multiple of 8 and ``K = 32`` is neither 64 nor 128, so
through the portable host every affine product reaches the translated
``affine_qmv`` kernel: the host has no Apple GPU generation, so affine
``qmv_wide`` stays unselected as on MLX's devices before generation 15. A single
nvfp4 row reaches ``nvfp4_qmv``, and two or four rows ``nvfp4_qmv_wide`` with
one threadgroup of that many vectors, since QuantizedMatmul passes no global
scale. The single-row references take Matmul's ``gemv`` and the others
``steel_gemm_fused``. The nvfp4 reference converts the bfloat16 dequantized
weights to float32 first, and the error bound ends in a translated whole-array
maximum.
"""

import sys
from collections import Counter

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

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_qmv_small_non_multiples"
LABEL = "test_qmv_small_non_multiples"
UPSTREAM_TESTS = ("test_quantized.TestQuantized.test_qmv_small_non_multiples",)
# The test's (M, K, N) cases and (bits, group_size, mode) settings, in order.
CASES = ((1, 32, 3), (2, 32, 10), (1, 32, 5), (4, 32, 7))
SETTINGS = ((4, 32, "affine"), (6, 32, "affine"), (4, 16, "nvfp4"))
REDUCTIONS = ("all_reduce_maxfloat32",)
# The nvfp4 reference converts the bfloat16 dequantized weights to float32.
OPERATORS = ("bfloat",)
# The default output type of a block-scaled dequantize.
DEQUANTIZED_TYPE = "bfloat16"
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 3600}
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
    """``(M, K, N, bits, group_size, mode)`` for every compared product."""
    for m, k, n in CASES:
        for bits, group, mode in SETTINGS:
            yield m, k, n, bits, group, mode


def quantized_entry(m, k, n, bits, group, mode):
    """QuantizedMatmul's transposed route for one unbatched product."""
    if mode == "affine":
        return matmul_layout.select("float32", group, bits, m, n, k, 1, transpose=True)
    if fp_matmul_layout.MODES[mode] != (group, bits):
        raise ValueError(f"{LABEL} uses {mode}'s pinned group size and bits")
    return fp_matmul_layout.select(mode, "float32", m, n, k, 1, transpose=True)


def routes():
    """Each product's quantized family and entry and its reference's route."""
    for m, k, n, bits, group, mode in products():
        quantized = quantized_entry(m, k, n, bits, group, mode)
        # x is (M, K) against the transposed view of the (N, K) weights.
        family, route = gemm_layout.matmul_route(
            "float32", *reference_operands(n, k, 0, (m, k))
        )
        if sum(route["copies"].values()):
            raise ValueError(f"{LABEL} reference operands need no copies")
        yield ("affine" if mode == "affine" else "fp"), quantized, family, route


def quantization_entries(group, bits, mode, operation):
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
    for (m, k, n, bits, group, mode), (_, quantized, family, route) in zip(
        products(), routes()
    ):
        for operation in ("quantize", "dequantize"):
            counts[
                quantization_entries(group, bits, mode, operation)
            ] += quantization_layout.launches(n * k, group)
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
