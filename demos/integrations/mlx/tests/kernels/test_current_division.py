"""Pinned division through reflected packages and unchanged source controls."""

import math
import struct

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BinaryCase,
)
from demos.integrations.mlx.tests.kernels.test_current_extrema import (
    TYPES,
    _guard,
)
from demos.integrations.mlx.tests.kernels.test_current_extrema import _pairs as _edges
from tests.test_translator.test_division_math import _oracle
from tests.test_translator.test_metal_division import _bfloat


def _pairs(dtype):
    pairs = _edges(dtype)
    fraction_bits, exponent_count, one = {
        "float16": (10, 31, 0x3C00),
        "bfloat16": (7, 255, 0x3F80),
        "float32": (23, 255, 0x3F800000),
    }[dtype]
    sign = 1 << (TYPES[dtype][1] - 1)
    fraction_max = (1 << fraction_bits) - 1
    for exponent in range(1, exponent_count):
        for fraction in (0, 1, fraction_max - 1, fraction_max):
            for sign_bit in (0, sign):
                numerator = (exponent << fraction_bits) | fraction | sign_bit
                for denominator in (one - 1, one, one + 1, one + (1 << fraction_bits)):
                    pairs.append((numerator, denominator))
    return pairs


def _expected(a, b, dtype):
    if dtype == "float16":
        x, y = (struct.unpack("<e", struct.pack("<H", word))[0] for word in (a, b))
        sign = (a ^ b) & 0x8000
        if (
            math.isnan(x)
            or math.isnan(y)
            or (x == y == 0)
            or (math.isinf(x) and math.isinf(y))
        ):
            return 0x7E00
        if math.isinf(x) or y == 0:
            return sign | 0x7C00
        if x == 0 or math.isinf(y):
            return sign
        try:
            return struct.unpack("<H", struct.pack("<e", x / y))[0]
        except OverflowError:
            return sign | 0x7C00
    if dtype == "bfloat16":
        a, b = a << 16, b << 16
    result = _oracle(a, b, flush=True)
    return _bfloat(result) >> 16 if dtype == "bfloat16" else result


def _cases(dtypes):
    for dtype in dtypes:
        pairs = _pairs(dtype)
        yield BinaryCase(
            dtype=dtype,
            operation="Divide",
            pairs=pairs,
            expected=[_expected(a, b, dtype) for a, b in pairs] + [_guard(dtype)] * 8,
            provenance=(
                {} if dtype == "float16" else {"binary32DivisionProfile": "rne-flush"}
            ),
            comparison="exact finite values and zero signs; NaN classification",
        )


@pytest.mark.parametrize(
    "dtype,a,b,expected",
    (
        ("float16", 1, 0x4000, 0),
        ("float16", 3, 0x4000, 2),
        ("float16", 0x8003, 0x4000, 0x8002),
        ("float16", 0x07FF, 0x4000, 0x0400),
        ("float16", 0x7BFF, 0x3800, 0x7C00),
        ("float16", 0x8000, 0x3C00, 0x8000),
        ("float16", 0x3C00, 0, 0x7C00),
        ("float16", 0x7C00, 0x7C00, 0x7E00),
        ("bfloat16", 0x0080, 0x3F81, 0),
        ("bfloat16", 0x8080, 0x3F81, 0x8000),
        ("bfloat16", 0x0080, 0x3F80, 0x0080),
        ("bfloat16", 0, 0, 0x7FC0),
        ("float32", 0x00800000, 0x3F800001, 0),
        ("float32", 0x00FFFFFF, 0x40000000, 0),
        ("float32", 0x80FFFFFF, 0x40000000, 0x80000000),
        ("float32", 0x3F800000, 0x40400000, 0x3EAAAAAB),
    ),
)
def test_division_reference_rounding_and_exceptional_values(dtype, a, b, expected):
    assert _expected(a, b, dtype) == expected


def test_division_inputs_cover_exponents_and_both_signs():
    counts = {"float16": 5632, "bfloat16": 12800, "float32": 12800}
    assert sum(counts.values()) == 31232
    for dtype, count in counts.items():
        pairs = _pairs(dtype)
        assert len(pairs) == count
        width = TYPES[dtype][1]
        assert all(0 <= word < 1 << width for pair in pairs for word in pair)
        assert any(a & (1 << (width - 1)) for a, _ in pairs)
        assert any(a == b == 0 for a, b in pairs)
