"""Run the unchanged pinned gated-delta tests on CPU and the translated backend.

``test_gated_delta_dtypes`` builds one-step recurrences of 32 key and value
heads in bfloat16 and float32 and checks each output's type. It never
evaluates them, so it launches no kernel; the native trace confirms it.
``test_gated_delta_fallback_masked`` compares the masked fallback of two head
shapes no kernel is instantiated for with the unmasked one; both run as
ordinary operations through the host's translated element-wise, reduction,
copy and random kernels.

``test_gated_delta_fallback`` compares the CPU stream with a Torch oracle, so
it reaches no translated kernel. ``test_gated_delta_sequential``,
``test_gated_delta_simdgroup``, ``test_gated_delta_nax`` and
``test_gated_delta_vjp`` are skipped without MLX's Metal backend, so
``gated_delta_workloads`` runs their comparisons in both workers first. Every
gated-delta dispatch of the native trace must match ``expected_dispatches``
exactly.
"""

import sys
from collections import Counter

from demos.integrations.mlx.portable_host import (
    gated_delta_layout,
    gated_delta_workloads,
    reduction_packages,
    upstream_runner,
)

MODULE = "demos.integrations.mlx.portable_host.verify_upstream_gated_delta"
LABEL = "test_gated_delta"
UPSTREAM_TESTS = (
    "test_fast_gated_delta.TestGatedDelta.test_gated_delta_fallback_masked",
    "test_fast_gated_delta.TestGatedDelta.test_gated_delta_dtypes",
)
# mx.allclose's conjunction and the workloads' recorded maximum errors, then
# the float32 row sums of the fallback recurrence, its gradient and
# mx.linalg.norm.
REDUCTIONS = (
    "all_reduce_andbool_",
    "all_reduce_maxfloat32",
    *(prefix + "sumfloat32" for prefix in reduction_packages.ROW_PREFIXES),
)
# The backward fallback sums its float32 gradients over strided axes, such as
# the 33 steps of a broadcast operand, through reduce.metal's looped column
# kernel, or its two-pass kernel for long reductions of few outputs.
COLUMN_SUMS = tuple(
    sorted(
        entry
        for entry in reduction_packages.COLUMN_ENTRIES
        if entry.endswith("_reduce_sumfloat32")
    )
)
# The masked fallback selects its states with mx.where.
OPERATORS = ("selection",)
TIMEOUTS = {"random": 600, "packages": 3000, "cpu": 900, "native": 6900}
PREFIXES = gated_delta_layout.ENTRY_PREFIXES
WORKLOADS = gated_delta_workloads.run


def validate_workloads(records, native):
    gated_delta_workloads.validate(records, native=native)


def _forward_entry(dtype, dims, chunk=None):
    """The forward entry gated_delta_chunk_size selects without NAX."""
    _, hk, hv, steps, _, _ = dims
    if chunk is None:
        chunk = 8 if steps > 8 else 1
    if chunk == 16:
        chunk = 8
    if chunk == 8:
        return gated_delta_layout.entry_name("fused", dtype, hk, hv, 8)
    return gated_delta_layout.entry_name("sequential", dtype, hk, hv)


def expected_dispatches():
    """The gated-delta launches of the workloads and both unchanged tests."""
    counts = Counter()
    dims = gated_delta_workloads.DIMS
    for case in gated_delta_workloads.cases():
        shape = dims[case["dims"]]
        if case["kind"] == "forward":
            counts[_forward_entry("float32", shape, int(case["chunk"]))] += 1
            continue
        _, hk, hv, _, _, _ = shape
        # The fallback reference and both kernel runs each evaluate the
        # forward at the default chunk; each kernel run adds the save pass
        # and the backward.
        counts[_forward_entry("float32", shape)] += 3
        runs = len(gated_delta_workloads.VJP_CHUNKS)
        counts[
            gated_delta_layout.entry_name("sequential", "float32", hk, hv, save=True)
        ] += runs
        counts[gated_delta_layout.entry_name("vjp", "float32", hk, hv)] += runs
    # test_gated_delta_dtypes reads only the output dtypes and never
    # evaluates its two recurrences, so it launches nothing.
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
    """Translate every gated-delta entry and the bfloat16 operator family."""
    from demos.integrations.mlx.portable_host import gated_delta_packages

    upstream_runner.translate_ahead(
        root,
        packages,
        reductions,
        jobs,
        [(gated_delta_packages, package_entries())],
        REDUCTIONS,
        columns=COLUMN_SUMS,
        operators=OPERATORS,
    )


def run(root, packages, output, jobs=3):
    return upstream_runner.run(sys.modules[__name__], root, packages, output, jobs)


if __name__ == "__main__":
    upstream_runner.main(sys.modules[__name__])
