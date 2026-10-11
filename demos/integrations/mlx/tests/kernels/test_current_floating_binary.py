"""Native floating binary checks grouped by compatible translation settings."""

from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels import test_current_additive as additive
from demos.integrations.mlx.tests.kernels import test_current_division as division
from demos.integrations.mlx.tests.kernels import test_current_extrema as extrema
from demos.integrations.mlx.tests.kernels import (
    test_current_multiplication as multiplication,
)
from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    run_binary_cases,
)
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FLOATING_BINARY"
SOURCE_PROFILES = ("division", "multiplication", "half", "comparison", "additive")


def _cases(profile):
    if profile == "half":
        for fixture in (extrema, additive, division, multiplication):
            yield from fixture._cases(("float16",))
    else:
        fixture = {
            "comparison": extrema,
            "additive": additive,
            "division": division,
            "multiplication": multiplication,
        }[profile]
        yield from fixture._cases(("bfloat16", "float32"))


def _compare_for(case):
    # Selection preserves operand bits; arithmetic permits different NaN payloads.
    if case.operation in ("Minimum", "Maximum"):
        return extrema._compare_native
    if case.operation in ("Add", "Subtract", "Divide", "Multiply"):
        return additive._check_words
    raise ValueError(f"No comparison policy for {case.operation}")


@pytest.mark.parametrize("profile", SOURCE_PROFILES)
def test_current_floating_binary_native_parity(
    tmp_path, profile, binary_metal_reference
):
    run_binary_cases(
        tmp_path,
        REQUIRE_ENV,
        _cases(profile),
        request_for=extrema._request,
        guard_for=extrema._guard,
        source_control=binary_metal_reference,
        compare_for=_compare_for,
    )


def test_ci_requires_floating_binary_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_floating_binary.py"
    assert workflow.count(path) == 1
    assert f"{path}::test_current_floating_binary_native_parity" in step
    assert "pytest -q -n auto" in step
    assert "--dist load --maxschedchunk=1" in step
    assert "-vv" in step and "--durations=10" in step
    assert "--timeout-seconds 900" in step
