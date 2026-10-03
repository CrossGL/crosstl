"""Native readbacks for unchanged MLX Sign and scalar comparison promotion."""

import os
from pathlib import Path

import pytest

from tests.test_translator.test_metal_boolean_promotion import _inputs
from tests.test_translator.test_metal_precise_trig import _bits, _float
from tests.test_translator.test_mlx_current_unary_fma import _run_current_unary

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


@pytest.mark.parametrize(
    "filename",
    [
        "mlx-project-porting.yml",
        "mlx-portable-host.yml",
        "mlx-metal-host.yml",
    ],
)
def test_ci_requires_boolean_promotion(filename):
    from tools import ci_coverage

    text = (
        Path(__file__).resolve().parents[2] / ".github/workflows" / filename
    ).read_text()
    step = text.split("      - name: Validate pinned comparison arithmetic\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_METAL_BOOLEAN_PROMOTION: "1"' in step
    assert "--timeout-seconds 180" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for module in ("test_mlx_current_unary_sign.py", "test_metal_boolean_promotion.py"):
        assert f"tests/test_translator/{module}" in step
        for event in ("push", "pull_request"):
            assert (
                f"tests/test_translator/{module}"
                in ci_coverage.workflow_event_path_filters(text, event)
            )
