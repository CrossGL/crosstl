"""Pinned addition/subtraction preserve rounding and characterized subnormal policy."""

import itertools
import math
import struct
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BinaryCase,
    run_binary_cases,
)
from demos.integrations.mlx.tests.kernels.test_current_extrema import (
    TYPES,
    _guard,
)
from demos.integrations.mlx.tests.kernels.test_current_extrema import _pairs as _edges
from demos.integrations.mlx.tests.kernels.test_current_extrema import _payload, _request
from tests.test_translator.test_metal_additive_profile import _oracle as _add
from tests.test_translator.test_metal_division import _bfloat
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ADDITIVE"
OPERATIONS = ("Add", "Subtract")


def _pairs(dtype):
    pairs = _edges(dtype)
    first_normal = {"float16": 0x0400, "bfloat16": 0x0080, "float32": 0x00800000}[dtype]
    sign = 1 << (TYPES[dtype][1] - 1)
    for a in range(first_normal, first_normal + 256):
        pairs.extend(
            (
                (a, a + 1),
                (a, (a + 1) ^ sign),
                (a ^ sign, a + 1),
                (a ^ sign, (a + 1) ^ sign),
            )
        )
    one, midpoint = {
        "float16": (0x3C00, 0x1000),
        "bfloat16": (0x3F80, 0x3B80),
        "float32": (0x3F800000, 0x33800000),
    }[dtype]
    pairs.extend(((one, midpoint), (one + 1, midpoint)))
    if dtype == "float32":
        pairs.extend(((0x81E657A8, 0x01F010C6), (0x00800000, 0x807FFFFF)))
    return pairs


def _expected(a, b, dtype, operation):
    if dtype == "float16":
        x, y = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (a, b))
        # Binary64 represents the sum or difference of two finite half values exactly.
        value = x + y if operation == "Add" else x - y
        if math.isnan(value):
            return 0x7E00
        try:
            return struct.unpack("<H", struct.pack("<e", value))[0]
        except OverflowError:
            return 0xFC00 if value < 0 else 0x7C00
    if dtype == "bfloat16":
        a, b = a << 16, b << 16
    result = _add(a, b, True, operation == "Subtract")
    return _bfloat(result) >> 16 if dtype == "bfloat16" else result


def _check_words(actual, expected, dtype, target):
    assert len(actual) == len(expected)
    narrow = dtype != "float32" and target != "opengl"
    bits = TYPES[dtype][1] if narrow else 32
    infinity = TYPES[dtype][2] if narrow else 0x7F800000
    magnitude = (1 << (bits - 1)) - 1
    differences = []
    nan_payload_differences = 0
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert type(got) is int and 0 <= got < 1 << bits
        if got == want:
            continue
        if (got & magnitude) > infinity and (want & magnitude) > infinity:
            nan_payload_differences += 1
        else:
            differences.append({"index": i, "expected": want, "actual": got})
    assert not differences, differences[:20]
    assert actual[-8:] == expected[-8:]
    return {"nanPayloadDifferences": nan_payload_differences, "finiteMismatchCount": 0}


def _cases():
    for dtype, operation in itertools.product(TYPES, OPERATIONS):
        pairs = _pairs(dtype)
        yield BinaryCase(
            dtype=dtype,
            operation=operation,
            pairs=pairs,
            expected=[_expected(a, b, dtype, operation) for a, b in pairs]
            + [_guard(dtype)] * 8,
            provenance=(
                {} if dtype == "float16" else {"binary32AdditiveProfile": "rne-flush"}
            ),
            comparison="exact finite values and zero signs; NaN classification",
        )


@pytest.mark.parametrize(
    "dtype,a,b,operation,expected",
    (
        ("float16", 0x3C00, 0x1000, "Add", 0x3C00),
        ("float16", 0x3C01, 0x1000, "Add", 0x3C02),
        ("float16", 0x0400, 0x8401, "Add", 0x8001),
        ("float16", 0x0400, 0x0401, "Subtract", 0x8001),
        ("bfloat16", 0x3F80, 0x3B80, "Add", 0x3F80),
        ("bfloat16", 0x3F81, 0x3B80, "Add", 0x3F82),
        ("bfloat16", 0x0080, 0x8081, "Add", 0x8000),
        ("bfloat16", 0x0080, 0x0081, "Subtract", 0x8000),
        ("float32", 0x3F800000, 0x33800000, "Add", 0x3F800000),
        ("float32", 0x3F800001, 0x33800000, "Add", 0x3F800002),
        ("float32", 0x00800000, 0x80800001, "Add", 0x80000000),
        ("float32", 0x00800000, 0x00800001, "Subtract", 0x80000000),
    ),
)
def test_additive_reference_rounding_and_cancellation(dtype, a, b, operation, expected):
    assert _expected(a, b, dtype, operation) == expected
    assert (a, b) in _pairs(dtype)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_additive_comparison_rejects_finite_zero_guard_and_type_changes(dtype, target):
    def encoded(words):
        return _payload(dtype, target, words)["values"]

    guard = _guard(dtype)
    expected = encoded([0] + [guard] * 8)
    sign = 1 << (TYPES[dtype][1] - 1)
    for actual in (
        encoded([sign] + [guard] * 8),
        encoded([1] + [guard] * 8),
        encoded([0] + [guard] * 7 + [0]),
        encoded([TYPES[dtype][2] + 1] + [guard] * 8),
        [False] + expected[1:],
    ):
        with pytest.raises(AssertionError):
            _check_words(actual, expected, dtype, target)
    nan = TYPES[dtype][2] + 1
    details = _check_words(
        encoded([nan + 1] + [guard] * 8),
        encoded([nan] + [guard] * 8),
        dtype,
        target,
    )
    assert details == {"nanPayloadDifferences": 1, "finiteMismatchCount": 0}


def test_additive_native_case_inventory():
    cases = list(_cases())
    assert len(cases) == 6 and len({case.entry for case in cases}) == 6
    assert sum(len(case.pairs) for case in cases) == 34192
    for case in cases:
        assert case.pairs == _pairs(case.dtype)
        assert len(case.expected) == len(case.pairs) + 8
        assert case.provenance == (
            {} if case.dtype == "float16" else {"binary32AdditiveProfile": "rne-flush"}
        )


@pytest.mark.parametrize("case", list(_cases()), ids=lambda case: case.entry)
def test_current_additive_native_parity(tmp_path, case, binary_metal_reference):
    run_binary_cases(
        tmp_path,
        REQUIRE_ENV,
        [case],
        request_for=_request,
        guard_for=_guard,
        source_control=binary_metal_reference,
        compare=_check_words,
    )


def test_ci_requires_additive_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_additive.py"
    assert workflow.count(path) == 1
    assert path in step and "pytest -q -n auto" in step
    assert "--timeout-seconds 900" in step
