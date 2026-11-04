"""Run the unchanged pinned Hadamard tests on CPU and the translated native backend.

``test_hadamard`` sweeps power-of-two and ``28 * 2^k`` transforms in float32,
float16, bfloat16 and integer inputs, and compares 4-row transforms of up to
458,752 values per row with MLX's CPU. ``test_hadamard_grad_vmap`` checks the
transform's vector-Jacobian product and nested ``vmap`` against products with
explicit Sylvester matrices. Every transform reaches the translated
``hadamard_n`` and ``hadamard_m`` JIT entries through ``Hadamard::eval_gpu``;
strided inputs, the casts to and from 16-bit types and the matrix products of
the reference take the host's copy, cast and Matmul routes.

``test_hadamard_m_only`` is skipped without MLX's Metal backend, so
``hadamard_workloads`` runs its ``m``-only transforms, both scales and both
output allocations, plus batches that need more than one launch, before the
unchanged tests in both workers. Every Hadamard dispatch of the native trace
must match ``expected_dispatches`` exactly.
"""

import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    hadamard_layout,
    hadamard_workloads,
    upstream_runner,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_hadamard"
LABEL = "test_hadamard"
UPSTREAM_TESTS = (
    "test_ops.TestOps.test_hadamard",
    "test_ops.TestOps.test_hadamard_grad_vmap",
)
# mx.all(y == x) and the CPU comparison's mx.abs(y1 - y2).max().
REDUCTIONS = ("all_reduce_maxfloat32", "all_reduce_andbool_")
# Strided float16 inputs are copied, and test_hadamard converts them to
# bfloat16 and back.
OPERATORS = ("half", "bfloat")
TIMEOUTS = {"random": 600, "packages": 4200, "cpu": 600, "native": 6600}
# test_hadamard skips itself when CI is set ("too slow in CI"); the workers
# run it unchanged with CI unset.
UNSET_ENVIRONMENT = ("CI",)
PREFIXES = hadamard_layout.ENTRY_PREFIXES
WORKLOADS = hadamard_workloads.run


def validate_workloads(records, native):
    hadamard_workloads.validate(records, native=native)


def _launches(counts, size, dtype, transforms):
    batch = hadamard_layout.batch_limit(size)
    for entry in hadamard_layout.entries(size, dtype):
        counts[entry] += -(-transforms // batch)


def expected_dispatches():
    """The Hadamard launches of the workloads and both unchanged tests."""
    counts = Counter()
    for case in hadamard_workloads.cases():
        counts.update(hadamard_workloads.launches(case))
    # test_hadamard's CPU comparison: 4 rows of m * 2^k for k in 14..16.
    for k in range(14, 17):
        for m in (1, 3, 5, 7):
            _launches(counts, m * 2**k, "float32", 4)
    # Its sweep transforms x[::2], 2 rows; integer inputs become float32, and
    # each float16 case is repeated in bfloat16.
    for dtype in ("float32", "float16", "int32"):
        for m in (1, 28):
            for k in range(1, 14):
                if m > 1 and k > 8:
                    continue
                size = m * 2**k
                _launches(
                    counts, size, "float16" if dtype == "float16" else "float32", 2
                )
                if dtype == "float16":
                    _launches(counts, size, "bfloat16", 2)
    # test_hadamard_grad_vmap: the forward transform and its vector-Jacobian
    # product of one vector, then one n * n batch for each vmapped axis.
    for k in range(2, 8):
        n = 2**k
        _launches(counts, n, "float32", 1)
        _launches(counts, n, "float32", 1)
        for _ in range(3):
            _launches(counts, n, "float32", n * n)
    return counts


def package_entries():
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
    """Translate every Hadamard entry and the 16-bit operator families."""
    from demos.integrations.mlx.portable_host import hadamard_packages

    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(hadamard_packages, package_entries())],
        REDUCTIONS,
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
