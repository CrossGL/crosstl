"""Execute pinned MLX sine/cosine through project packaging and native loading."""

import json
import os
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from tests.test_translator.test_metal_precise_trig import (
    _bits,
    _float,
    _inputs,
    _oracle,
)
from tools import ci_coverage

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_TRIG"


@pytest.mark.parametrize("operation", ["Sin", "Cos"])
def test_current_mlx_precise_trig_native_loader(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned trigonometry execution")
    words = _inputs()
    index = 0 if operation == "Sin" else 1
    expected = [_oracle(word)[index] for word in words]
    actual = _run_current_unary(
        tmp_path,
        operation,
        [_float(word) for word in words],
        [_float(word) for word in expected],
        0,
        1e-6,
        f"metal_precise_{operation.lower()}_float",
    )
    maximum = 0
    for got, want in zip(actual, expected):
        actual_word = _bits(got)
        if want & 0x7FFFFFFF > 0x7F800000:
            assert actual_word & 0x7FFFFFFF > 0x7F800000, "NaN classification"
            continue
        assert actual_word >> 31 == want >> 31, "result sign"
        if want & 0x7FFFFFFF == 0:
            assert actual_word == want, "zero sign"
        error = abs(actual_word - want)
        assert error <= 4, (got, _float(want), error)
        maximum = max(maximum, error)
    (tmp_path / "ulp-audit.json").write_text(
        json.dumps(
            {
                "operation": operation,
                "values": len(words),
                "maxUlpError": maximum,
                "oracle": "160-digit Decimal, correctly rounded binary32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("job_id", ["portable-host", "mlx-metal-porting"])
def test_ci_requires_pinned_trigonometry(job_id):
    text = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = (
        ci_coverage.workflow_job_text(text, job_id)
        .split("      - name: Validate pinned unary arithmetic\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "demos/integrations/mlx/tests/kernels/test_current_unary_trig.py" in step
    assert "--timeout-seconds 300" in step
    assert "--basetemp=" in step and "--junitxml=" in step
