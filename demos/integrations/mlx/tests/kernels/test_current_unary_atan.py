"""Execute unchanged pinned MLX arctangent through the native loader."""

import json
import os
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_metal_precise_atan import _check_word, _inputs, _oracle
from tests.test_translator.test_metal_precise_trig import _bits, _float

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_ATAN"


def test_current_mlx_precise_atan_native_loader(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned atan execution")
    words = _inputs()
    expected = [_oracle(word) for word in words]
    actual = _run_current_unary(
        tmp_path,
        "ArcTan",
        [_float(word) for word in words],
        [_float(word) for word in expected],
        2e-5,
        1e-6,
        "metal_precise_atan_float",
    )
    maximum = max(_check_word(_bits(got), want) for got, want in zip(actual, expected))
    (tmp_path / "ulp-audit.json").write_text(
        json.dumps(
            {
                "operation": "ArcTan",
                "values": len(words),
                "maxUlpError": maximum,
                "oracle": (
                    "100-digit Decimal half-angle atan, correctly rounded binary32"
                ),
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("job_id", ["mlx-metal-porting", "portable-host", "metal-host"])
def test_ci_requires_precise_atan(job_id):
    from tools import ci_coverage

    text = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = (
        ci_coverage.workflow_job_text(text, job_id)
        .split("      - name: Validate pinned arctangent\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_METAL_PRECISE_ATAN: "1"' in step
    assert "demos/integrations/mlx/tests/kernels/test_current_unary_atan.py" in step
    assert "tests/test_translator/test_metal_precise_atan.py" in step
    assert "--timeout-seconds 180" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for event in ("push", "pull_request"):
        for module in (
            "demos/integrations/mlx/tests/kernels/test_current_unary_atan.py",
            "tests/test_translator/test_metal_precise_atan.py",
        ):
            assert_paths_covered(
                ci_coverage.workflow_event_path_filters(text, event), module
            )
