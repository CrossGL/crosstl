"""Native readbacks for unchanged MLX Sign and scalar comparison promotion."""

import os
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_metal_boolean_promotion import _inputs
from tests.test_translator.test_metal_precise_trig import _bits, _float

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_SIGN"


def test_current_mlx_sign_native_loader(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned Sign execution")
    inputs = [_float(word) for word in _inputs()]
    expected = [float((value > 0) - (value < 0)) for value in inputs]
    actual = _run_current_unary(
        tmp_path,
        "Sign",
        inputs,
        expected,
        0.0,
        0.0,
        "operator_call",
    )
    assert [_bits(value) for value in actual] == [_bits(value) for value in expected]


@pytest.mark.parametrize("job_id", ["mlx-metal-porting", "portable-host", "metal-host"])
def test_ci_requires_boolean_promotion(job_id):
    from tools import ci_coverage

    text = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = (
        ci_coverage.workflow_job_text(text, job_id)
        .split("      - name: Validate pinned comparison arithmetic\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_METAL_BOOLEAN_PROMOTION: "1"' in step
    assert "--timeout-seconds 180" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for module in (
        "demos/integrations/mlx/tests/kernels/test_current_unary_sign.py",
        "tests/test_translator/test_metal_boolean_promotion.py",
    ):
        assert module in step
        for event in ("push", "pull_request"):
            assert_paths_covered(
                ci_coverage.workflow_event_path_filters(text, event), module
            )
