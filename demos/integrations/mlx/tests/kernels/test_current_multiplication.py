"""Pinned half multiplication through native packages and source controls."""

import math
import struct
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BinaryCase,
    run_binary_cases,
)
from demos.integrations.mlx.tests.kernels.test_current_additive import _check_words
from demos.integrations.mlx.tests.kernels.test_current_division import _pairs
from demos.integrations.mlx.tests.kernels.test_current_extrema import _guard, _request
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_MULTIPLICATION"


def _expected(a, b):
    x, y = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (a, b))
    value = x * y
    if math.isnan(value):
        return 0x7E00
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return ((a ^ b) & 0x8000) | 0x7C00


def _cases():
    pairs = _pairs("float16")
    yield BinaryCase(
        dtype="float16",
        operation="Multiply",
        pairs=pairs,
        expected=[_expected(a, b) for a, b in pairs] + [_guard("float16")] * 8,
        provenance={},
        comparison="exact finite values and zero signs; NaN classification",
    )


@pytest.mark.parametrize(
    "a,b,expected",
    (
        (1, 0x3800, 0),
        (3, 0x3800, 2),
        (0x8003, 0x3800, 0x8002),
        (0x03FF, 0x4000, 0x07FE),
        (0x0400, 0x3800, 0x0200),
        (0x3C01, 0x3E00, 0x3E02),
        (0x3C03, 0x3E00, 0x3E04),
        (0x7BFF, 0x4000, 0x7C00),
        (0xFBFF, 0x4000, 0xFC00),
        (0x8000, 0x3C00, 0x8000),
        (0x8000, 0xBC00, 0),
        (0xFC00, 0xBC00, 0x7C00),
        (0, 0x7C00, 0x7E00),
        (0xFC00, 0, 0x7E00),
        (0x7C01, 0x3C00, 0x7E00),
    ),
)
def test_half_multiplication_reference_rounding_and_exceptional_values(a, b, expected):
    assert _expected(a, b) == expected
    assert _expected(b, a) == expected


def test_half_multiplication_inventory_and_source_profile():
    (case,) = _cases()
    assert case.entry == "vv_Multiplyfloat16"
    assert case.provenance == {}
    assert len(case.pairs) == 5632 and len(case.expected) == 5640
    assert case.expected[-8:] == [_guard("float16")] * 8
    assert all(0 <= word <= 65535 for pair in case.pairs for word in pair)
    assert {a >> 10 & 31 for a, _ in case.pairs} == set(range(32))
    assert {a >> 15 for a, _ in case.pairs} == {0, 1}


def test_current_multiplication_native_parity(tmp_path, binary_metal_reference):
    run_binary_cases(
        tmp_path,
        REQUIRE_ENV,
        _cases(),
        request_for=_request,
        guard_for=_guard,
        source_control=binary_metal_reference,
        compare=_check_words,
    )


def test_ci_requires_multiplication_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_multiplication.py"
    assert workflow.count(path) == 1
    assert f"{path}::test_current_multiplication_native_parity" in step
    assert "pytest -q -n auto" in step and "--dist worksteal" in step
    assert "--timeout-seconds 900" in step
