"""Pinned real power identities, operand semantics and finite accuracy."""

from functools import partial
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BinaryCase,
    run_binary_cases,
)
from demos.integrations.mlx.tests.kernels.test_current_additive import _check_words
from demos.integrations.mlx.tests.kernels.test_current_extrema import _guard, _request
from tests.test_translator.test_metal_power import (
    _finite_pairs,
    _identity_pairs,
    _operand_flush_oracle,
    _operand_flush_pairs,
    _oracle,
)
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_REAL_POWER"
WORKGROUP_WIDTH = 2


def _pairs(dtype, profile):
    if profile == "finite":
        assert dtype == "float32"
        return _finite_pairs()
    if profile == "identity":
        if dtype == "float32":
            return _identity_pairs()
        assert dtype in {"float16", "bfloat16"}
        one = 0x3C00 if dtype == "float16" else 0x3F80
        return [(word, one) for word in range(65536)]
    assert profile == "operands"
    if dtype == "float32":
        return _operand_flush_pairs()
    assert dtype == "bfloat16"
    subnormals = [word | sign for word in range(1, 128) for sign in (0, 0x8000)]
    controls = [
        word | sign
        for word in (
            0,
            0x80,
            0x81,
            0x3F00,
            0x3F7F,
            0x3F80,
            0x3F81,
            0x4000,
            0x4040,
            0x4B00,
            0x4B80,
            0x7F7F,
            0x7F80,
            0x7FC1,
        )
        for sign in (0, 0x8000)
    ]
    return sorted(
        {(a, b) for a in subnormals for b in subnormals + controls}
        | {(a, b) for a in controls for b in subnormals}
    )


def _cases(profile):
    types = {
        "identity": ("float16", "bfloat16", "float32"),
        "operands": ("bfloat16", "float32"),
        "finite": ("float32",),
    }[profile]
    for dtype in types:
        pairs = _pairs(dtype, profile)
        if profile == "identity":
            expected = [a for a, _ in pairs]
            provenance = {}
        elif profile == "operands":
            shift = 16 if dtype == "bfloat16" else 0
            expected = [
                _operand_flush_oracle(a << shift, b << shift) >> shift for a, b in pairs
            ]
            provenance = {"binary32PowerOperandProfile": "flush-subnormals"}
        else:
            expected = [_oracle(a, b) for a, b in pairs]
            provenance = {"binary32PowerAccuracyProfile": "portable-finite"}
        yield BinaryCase(
            dtype=dtype,
            operation="Power",
            pairs=pairs,
            expected=expected + [_guard(dtype)] * 8,
            provenance=provenance,
            comparison=(
                "finite result within 16 binary32 storage steps; exact guards"
                if profile == "finite"
                else "exact identity or selected operand-domain result; NaN classification"
            ),
        )


@pytest.mark.parametrize(
    "profile,count", (("identity", 136704), ("operands", 82848), ("finite", 4982))
)
def test_real_power_inventory_preserves_every_input_and_exact_grid(profile, count):
    cases = list(_cases(profile))
    assert sum(len(case.pairs) for case in cases) == count
    for case in cases:
        assert len(case.expected) == len(case.pairs) + 8
        assert case.expected[-8:] == [_guard(case.dtype)] * 8
        assert len(case.pairs) % WORKGROUP_WIDTH == 0
        assert 0 < len(case.pairs) // WORKGROUP_WIDTH <= 65535
        assert (
            case.provenance
            == {
                "identity": {},
                "operands": {"binary32PowerOperandProfile": "flush-subnormals"},
                "finite": {"binary32PowerAccuracyProfile": "portable-finite"},
            }[profile]
        )
        if profile == "identity" and case.dtype != "float32":
            assert [a for a, _ in case.pairs] == list(range(65536))
        if profile == "operands":
            threshold = 0x80 if case.dtype == "bfloat16" else 0x800000
            mask = 0x7FFF if case.dtype == "bfloat16" else 0x7FFFFFFF
            assert all(
                any(0 < (word & mask) < threshold for word in pair)
                for pair in case.pairs
            )


def _check_finite(actual, expected, dtype, target):
    assert dtype == "float32" and target in {"metal", "opengl", "directx"}
    assert len(actual) == len(expected) and len(expected) > 8
    assert all(type(word) is int and 0 <= word <= 0xFFFFFFFF for word in actual)
    assert actual[-8:] == expected[-8:] == [_guard(dtype)] * 8
    maximum = 0
    for index, (got, want) in enumerate(zip(actual[:-8], expected[:-8])):
        assert 0x800000 <= want < 0x7F800000
        assert 0x800000 <= got < 0x7F800000
        distance = abs(got - want)
        assert distance <= 16, (index, hex(got), hex(want), distance)
        maximum = max(maximum, distance)
    return {
        "checked": len(actual) - 8,
        "guardCount": 8,
        "maximumStorageUlpError": maximum,
    }


def test_real_power_finite_comparison_rejects_inaccurate_or_unwritten_results():
    case = next(_cases("finite"))
    assert (
        _check_finite(case.expected, case.expected, "float32", "opengl")[
            "maximumStorageUlpError"
        ]
        == 0
    )
    index = case.pairs.index((0x3F7FFFFF, 0x4B800000))
    for offset, word in ((index, 0x3E3504F5), (0, 0x7F800000), (-1, 0)):
        changed = list(case.expected)
        changed[offset] = word
        with pytest.raises(AssertionError):
            _check_finite(changed, case.expected, "float32", "opengl")


@pytest.mark.parametrize("profile", ("identity", "operands", "finite"))
@pytest.mark.extended_power
def test_current_real_power_native_parity(tmp_path, profile, binary_metal_reference):
    run_binary_cases(
        tmp_path,
        REQUIRE_ENV,
        _cases(profile),
        request_for=partial(_request, workgroup_width=WORKGROUP_WIDTH),
        guard_for=_guard,
        compare_for=lambda case: _check_finite if profile == "finite" else _check_words,
        source_control=binary_metal_reference,
        workgroup_width=WORKGROUP_WIDTH,
    )


def test_ci_requires_real_power_on_existing_native_targets():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned real power"
    )
    path = "demos/integrations/mlx/tests/kernels/test_current_real_power.py"
    assert "if:" not in step
    assert workflow.count(path) == 1
    assert f'{REQUIRE_ENV}: "1"' in step
    assert f"{path}::test_current_real_power_native_parity" in step
    assert "--timeout-seconds 180" in step
    assert "pytest -q -n auto" in step
    assert 'CROSTL_REQUIRE_METAL_POWER: "1"' in step
    assert "tests/test_translator/test_metal_power.py" in step
    assert "-m extended_power" in step
