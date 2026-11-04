"""The unchanged gathered quantized matmul tests' runner and its expected launches."""

from demos.integrations.mlx.portable_host import (
    gather_qmm_layout,
    gather_qmm_packages,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_gather_matmul_grad as runner,
)


def test_runner_expects_the_forward_and_gradient_launches():
    counts = runner.expected_dispatches()
    # The forward product of the sum, and the gradient's product of the
    # cotangent with the untransposed weights; mx.grad skips its forward sum.
    assert counts == {
        "affine_gather_qmm_t_float_gs_64_b_4_alN_true": 1,
        "affine_gather_qmm_n_float_gs_64_b_4": 1,
    }
    for entry in counts:
        assert entry.startswith(runner.PREFIXES)
        gather_qmm_layout.signature(entry)
    assert runner.entries() == sorted(counts)
    assert gather_qmm_packages.verification_entries() == sorted(counts)


def test_runner_tracks_the_pinned_tests_and_their_bounds():
    assert runner.UPSTREAM_TESTS == (
        "test_quantized.TestQuantized.test_gather_matmul_grad",
        "test_quantized.TestQuantized.test_qmm_fp_type",
    )
    assert runner.REDUCTIONS == ("all_reduce_andbool_", "all_reduce_sumfloat32")
    assert not hasattr(runner, "OPERATORS")
    assert runner.PREFIXES == gather_qmm_layout.ENTRY_PREFIXES
    # The tests the host cannot run yet name their blocker.
    assert "CrossGL/crosstl#2690" in runner.__doc__
    assert "test_gather_qmm``" in runner.__doc__
    # mx.quantize's defaults for each mode.
    assert runner.MODES == {
        "affine": (64, 4),
        "nvfp4": (16, 4),
        "mxfp4": (32, 4),
        "mxfp8": (32, 8),
    }
