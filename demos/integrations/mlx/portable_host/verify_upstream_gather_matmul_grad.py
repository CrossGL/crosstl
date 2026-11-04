"""Run the unchanged pinned gathered quantized matmul tests on CPU and natively.

``test_gather_matmul_grad`` multiplies gathered float32 activations of four
by two 32-by-256 matrices by gathered affine 4-bit weights (group 64) through
``mx.gather_qmm``, sums the products, and compares the sum and its gradient
with respect to the activations against ``mx.gather_mm`` of the dequantized
weights. The forward products reach the translated
``affine_gather_qmm_t`` entry through ``GatherQMM::eval_gpu``; ``mx.grad``
never needs its own forward sum. The gradient takes GatherQMM's vector-Jacobian
product, ``gather_qmm`` of the cotangent with the weights untransposed
(``affine_gather_qmm_n``), scattered back over the left indices. The
references take the host's quantize, dequantize, gather_mm, scatter and
reduction routes. ``test_qmm_fp_type`` checks the output types of mxfp8 and
mxfp4 products it never evaluates, so it launches nothing. Every gathered
dispatch of the native trace must match ``expected_dispatches`` exactly.

Not run yet:

- ``test_gather_qmm`` and ``test_gather_qmm_matrix_path`` multiply by
  mxfp4, mxfp8 and nvfp4 weights at matrix sizes. ``fp_gather_qmm_t`` and
  ``fp_gather_qmm_n`` load their scales through ``dequantize_scale`` from a
  member of ``QuantizedBlockLoader``, and the translation drops that
  conversion operator call on every target (CrossGL/crosstl#2690). Their
  affine and vector products pass natively through this host.
- The sorted tests take ``gather_qmm_rhs``, which the host refuses.
"""

import sys
from collections import Counter

from demos.integrations.mlx.portable_host import gather_qmm_layout, upstream_runner

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_gather_matmul_grad"
LABEL = "test_gather_matmul_grad"
UPSTREAM_TESTS = (
    "test_quantized.TestQuantized.test_gather_matmul_grad",
    "test_quantized.TestQuantized.test_qmm_fp_type",
)
# mx.allclose's mx.all and the products' whole-array sums.
REDUCTIONS = ("all_reduce_andbool_", "all_reduce_sumfloat32")
TIMEOUTS = {"random": 600, "packages": 3600, "cpu": 600, "native": 5400}
PREFIXES = gather_qmm_layout.ENTRY_PREFIXES
# mx.quantize's default group size and bits of each mode.
MODES = {"affine": (64, 4), "nvfp4": (16, 4), "mxfp4": (32, 4), "mxfp8": (32, 8)}
# test_gather_matmul_grad: (M, N, K) of x's 32-by-256 matrices and the
# 32-by-256 weights, transposed.
GRAD_SHAPE = (32, 32, 256)


def expected_dispatches():
    """The gathered launches of test_gather_matmul_grad."""
    group, bits = MODES["affine"]
    m, n, k = GRAD_SHAPE
    forward = gather_qmm_layout.select("affine", "float32", group, bits, m, n, k, True)
    # The cotangent's (32, 32) rows times the untransposed weights.
    backward = gather_qmm_layout.select(
        "affine", "float32", group, bits, m, k, n, False
    )
    return Counter({forward: 1, backward: 1})


def entries():
    return sorted(expected_dispatches())


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
    """Translate every gathered entry the tests reach."""
    from demos.integrations.mlx.portable_host import gather_qmm_packages

    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(gather_qmm_packages, entries())],
        REDUCTIONS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
