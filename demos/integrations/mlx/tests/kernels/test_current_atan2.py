"""Pinned arctangent packages with source controls and precision-specific checks."""

import struct
from functools import partial
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BinaryCase,
    run_binary_cases,
)
from demos.integrations.mlx.tests.kernels.test_current_copy import _widen
from demos.integrations.mlx.tests.kernels.test_current_division import _pairs
from demos.integrations.mlx.tests.kernels.test_current_extrema import (
    TYPES,
    _guard,
    _payload,
    _request,
)
from tests.test_translator.test_metal_division import _bfloat
from tests.test_translator.test_metal_precise_atan2 import _oracle, _profile_word
from tests.test_translator.test_metal_precise_trig import _float
from tools import ci_coverage

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ATAN2"
PROFILE = "flush-subnormals"


def _wide(word, dtype):
    if dtype == "float16":
        return _widen(word)
    return word << 16 if dtype == "bfloat16" else word


def _narrow(word, dtype):
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", _float(word)))[0]
    return _bfloat(word) >> 16 if dtype == "bfloat16" else word


def _reference(a, b, dtype):
    return _oracle(_wide(a, dtype), _wide(b, dtype), PROFILE)


def _cases():
    for dtype in TYPES:
        pairs = _pairs(dtype)
        yield BinaryCase(
            dtype=dtype,
            operation="ArcTan2",
            pairs=pairs,
            expected=[_narrow(_reference(a, b, dtype), dtype) for a, b in pairs]
            + [_guard(dtype)] * 8,
            provenance={"binary32Atan2Profile": PROFILE},
            comparison="source precision bounds; exact axes, zero signs and guards",
        )


def _compare(case, actual, expected, dtype, target):
    assert dtype == case.dtype
    assert expected == _payload(dtype, target, case.expected)["values"]
    assert len(actual) == len(expected)
    width = 32 if target == "opengl" else TYPES[dtype][1]
    assert all(type(word) is int and 0 <= word < 1 << width for word in actual)
    assert actual[-8:] == expected[-8:], "output guards"
    sign = 1 << (TYPES[dtype][1] - 1)
    maximum = nan_differences = 0
    for i, (a, b) in enumerate(case.pairs):
        physical = actual[i]
        got = physical
        if target == "opengl" and dtype != "float32":
            if physical & 0x7FFFFFFF > 0x7F800000:
                got = TYPES[dtype][2] | 1
            else:
                got = _narrow(physical, dtype)
                assert _wide(got, dtype) == physical, "unrounded narrow storage"
        reference = _reference(a, b, dtype)
        want = case.expected[i]
        if reference & 0x7FFFFFFF > 0x7F800000:
            assert got & (sign - 1) > TYPES[dtype][2], "NaN classification"
            nan_differences += physical != expected[i]
            continue
        assert got & sign == want & sign, "result sign"
        magnitudes = [
            _profile_word(_wide(word, dtype), PROFILE) & 0x7FFFFFFF for word in (a, b)
        ]
        if want & (sign - 1) == 0 or any(
            value in (0, 0x7F800000) for value in magnitudes
        ):
            assert got == want, "axis or zero result"
        elif dtype == "bfloat16":
            # The source evaluates in float, then applies bfloat nearest-even narrowing.
            allowed = {_narrow(reference + offset, dtype) for offset in range(-6, 7)}
            assert got in allowed, "float accuracy followed by bfloat rounding"
        else:
            # MSL Tables 8.1/8.3 specify six float ULPs and one half ULP.
            assert abs(got - want) <= (1 if dtype == "float16" else 6), (
                i,
                a,
                b,
                got,
                want,
            )
        maximum = max(maximum, abs(got - want))
    return {"maximumStorageUlpError": maximum, "nanPayloadDifferences": nan_differences}


def test_current_atan2_native_parity(tmp_path, binary_metal_reference):
    run_binary_cases(
        tmp_path,
        REQUIRE_ENV,
        _cases(),
        request_for=_request,
        guard_for=_guard,
        compare_for=lambda case: partial(_compare, case),
        source_control=binary_metal_reference,
    )


def test_atan2_inventory_preserves_types_exponents_and_profile():
    counts = {"float16": 5632, "bfloat16": 12800, "float32": 12800}
    cases = list(_cases())
    assert len(cases) == 3
    assert sum(len(case.pairs) for case in cases) == 31232
    for case in cases:
        assert case.entry == f"vv_ArcTan2{case.dtype}"
        assert len(case.pairs) == counts[case.dtype]
        assert case.provenance == {"binary32Atan2Profile": PROFILE}
        assert len(case.expected) == len(case.pairs) + 8
        assert case.expected[-8:] == [_guard(case.dtype)] * 8


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_atan2_comparison_rejects_changed_values_signs_guards_and_nan(dtype, target):
    one = {"float16": 0x3C00, "bfloat16": 0x3F80, "float32": 0x3F800000}[dtype]
    infinity = TYPES[dtype][2]
    sign = 1 << (TYPES[dtype][1] - 1)
    pairs = [(0, one), (sign, one), (infinity, one), (infinity + 1, one), (one, one)]
    words = [_narrow(_reference(a, b, dtype), dtype) for a, b in pairs]
    case = BinaryCase(dtype, "ArcTan2", pairs, words + [_guard(dtype)] * 8, {}, "test")
    expected = _payload(dtype, target, case.expected)["values"]
    _compare(case, expected, expected, dtype, target)
    for index, value in ((0, sign), (1, 0), (2, 0), (3, 0), (4, words[4] + 16), (5, 0)):
        changed = list(case.expected)
        changed[index] = value
        with pytest.raises(AssertionError):
            _compare(
                case,
                _payload(dtype, target, changed)["values"],
                expected,
                dtype,
                target,
            )
    for actual in (expected[:-1], [False] + expected[1:]):
        with pytest.raises(AssertionError):
            _compare(case, actual, expected, dtype, target)
    if target == "opengl" and dtype != "float32":
        changed = list(expected)
        changed[4] += 1
        with pytest.raises(AssertionError, match="unrounded narrow storage"):
            _compare(case, changed, expected, dtype, target)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_atan2_comparison_enforces_precision_boundary(dtype, target):
    one = {"float16": 0x3C00, "bfloat16": 0x3F80, "float32": 0x3F800000}[dtype]
    reference = _reference(one, one, dtype)
    word = _narrow(reference, dtype)
    limit = {"float16": 1, "bfloat16": 0, "float32": 6}[dtype]
    if dtype == "bfloat16":
        assert {_narrow(reference + offset, dtype) for offset in range(-6, 7)} == {word}
    case = BinaryCase(
        dtype, "ArcTan2", [(one, one)], [word] + [_guard(dtype)] * 8, {}, "test"
    )
    expected = _payload(dtype, target, case.expected)["values"]
    for offset in (-limit, limit):
        actual = _payload(dtype, target, [word + offset] + case.expected[1:])["values"]
        assert _compare(case, actual, expected, dtype, target)[
            "maximumStorageUlpError"
        ] == abs(offset)
    for offset in (-limit - 1, limit + 1):
        actual = _payload(dtype, target, [word + offset] + case.expected[1:])["values"]
        with pytest.raises(AssertionError):
            _compare(case, actual, expected, dtype, target)


def test_atan2_profiles_are_scoped_to_every_binary_shape(tmp_path):
    from crosstl.project import load_project_config
    from demos.integrations.mlx.tests.kernels.test_binary_complete_opengl import (
        BINARY_OPENGL_WORKLOADS,
        _project_config,
    )

    selected = []
    for workload in BINARY_OPENGL_WORKLOADS:
        text = _project_config(workload)
        enabled = workload.operator_type == "ArcTan2"
        assert (f'binary32_atan2_profile = "{PROFILE}"' in text) == enabled
        if enabled:
            selected.append(workload)
    assert len(selected) == 54 and len({workload.shape for workload in selected}) == 18
    path = tmp_path / "crosstl.toml"
    path.write_text(
        _project_config(selected[0], entry_points=[w.entry_point for w in selected])
    )
    assert (
        load_project_config(tmp_path, path).source_options["metal"][
            "binary32_atan2_profile"
        ]
        == PROFILE
    )


def test_ci_requires_pinned_atan2_once_per_native_target():
    root = Path(__file__).resolve().parents[5]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned arctangent"
    )
    selector = "demos/integrations/mlx/tests/kernels/test_current_atan2.py::test_current_atan2_native_parity"
    assert workflow.count(selector) == 1 and selector in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "2"' in step
    assert "--timeout-seconds 180" in step and "pytest -q -n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
