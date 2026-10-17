"""Pinned floating multiplication through native packages and source controls."""

import math
import struct

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BinaryCase,
)
from demos.integrations.mlx.tests.kernels.test_current_division import _pairs
from demos.integrations.mlx.tests.kernels.test_current_extrema import TYPES, _guard
from tests.test_translator.test_metal_division import _bfloat
from tests.test_translator.test_metal_multiplication_profile import _oracle


def _expected(a, b, dtype="float16"):
    if dtype != "float16":
        if dtype == "bfloat16":
            a, b = a << 16, b << 16
        result = _oracle(a, b, True)
        return _bfloat(result) >> 16 if dtype == "bfloat16" else result
    x, y = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (a, b))
    value = x * y
    if math.isnan(value):
        return 0x7E00
    try:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    except OverflowError:
        return ((a ^ b) & 0x8000) | 0x7C00


def _cases(dtypes=TYPES):
    for dtype in dtypes:
        pairs = _pairs(dtype)
        yield BinaryCase(
            dtype=dtype,
            operation="Multiply",
            pairs=pairs,
            expected=[_expected(a, b, dtype) for a, b in pairs] + [_guard(dtype)] * 8,
            provenance=(
                {}
                if dtype == "float16"
                else {"binary32MultiplicationProfile": "rne-flush"}
            ),
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
    (case,) = _cases(("float16",))
    assert case.entry == "vv_Multiplyfloat16"
    assert case.provenance == {}
    assert len(case.pairs) == 5632 and len(case.expected) == 5640
    assert case.expected[-8:] == [_guard("float16")] * 8
    assert all(0 <= word <= 65535 for pair in case.pairs for word in pair)
    assert {a >> 10 & 31 for a, _ in case.pairs} == set(range(32))
    assert {a >> 15 for a, _ in case.pairs} == {0, 1}


@pytest.mark.parametrize(
    "dtype,a,b,expected",
    (
        ("float32", 0x00800000, 0x3F7FFFFF, 0),
        ("float32", 0x80800000, 0x3F7FFFFF, 0x80000000),
        ("float32", 1, 0x7F000000, 0),
        ("float32", 0x007FFFFF, 0x40000000, 0),
        ("float32", 0x00800000, 0x40000000, 0x01000000),
        ("float32", 0x3F800001, 0x3FC00000, 0x3FC00002),
        ("float32", 0x3F800003, 0x3FC00000, 0x3FC00004),
        ("float32", 0x80000000, 0x3F800000, 0x80000000),
        ("float32", 0x80000000, 0xBF800000, 0),
        ("float32", 0x7F7FFFFF, 0x40000000, 0x7F800000),
        ("float32", 0, 0x7F800000, 0x7FC00000),
        ("bfloat16", 0x0080, 0x3F7F, 0),
        ("bfloat16", 0x8080, 0x3F7F, 0x8000),
        ("bfloat16", 1, 0x7F00, 0),
        ("bfloat16", 0x007F, 0x4000, 0),
        ("bfloat16", 0x0080, 0x4000, 0x0100),
        ("bfloat16", 0x3F81, 0x3FC0, 0x3FC2),
        ("bfloat16", 0x3F83, 0x3FC0, 0x3FC4),
        ("bfloat16", 0x8000, 0x3F80, 0x8000),
        ("bfloat16", 0x8000, 0xBF80, 0),
        ("bfloat16", 0x7F7F, 0x4000, 0x7F80),
        ("bfloat16", 0, 0x7F80, 0x7FC0),
    ),
)
def test_multiplication_reference_rounding_and_exceptional_values(
    dtype, a, b, expected
):
    assert _expected(a, b, dtype) == expected
    assert _expected(b, a, dtype) == expected


def test_multiplication_inventory_and_source_profiles():
    cases = list(_cases())
    assert len(cases) == len({case.entry for case in cases}) == 3
    assert sum(len(case.pairs) for case in cases) == 31232
    for case in cases:
        assert case.pairs == _pairs(case.dtype)
        assert len(case.expected) == len(case.pairs) + 8
        assert case.expected[-8:] == [_guard(case.dtype)] * 8
        assert case.provenance == (
            {}
            if case.dtype == "float16"
            else {"binary32MultiplicationProfile": "rne-flush"}
        )
