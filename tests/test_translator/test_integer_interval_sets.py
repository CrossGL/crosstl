"""Compare fixed-width interval proofs with exact integer evaluation."""

import random

import pytest

from crosstl.translator.codegen.integer_intervals import (
    convert_integer_intervals,
    integer_arithmetic_intervals,
    merge_integer_intervals,
)


def _cast(value, bits, signed):
    value %= 1 << bits
    return value - (1 << bits) if signed and value >= (1 << (bits - 1)) else value


@pytest.mark.parametrize("bits", [2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("signed", [False, True])
def test_integer_conversion_covers_every_value(bits, signed):
    modulus = 1 << bits
    for center in (-modulus, -(modulus // 2), 0, modulus // 2, modulus):
        for width in range(10):
            low, high = center - width, center + width
            intervals = convert_integer_intervals([(low, high)], bits, signed)
            assert all(
                any(a <= _cast(value, bits, signed) <= b for a, b in intervals)
                for value in range(low, high + 1)
            )


@pytest.mark.parametrize("bits", [2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("operator", ["+", "-", "*"])
def test_integer_arithmetic_covers_exact_fixed_width_results(bits, signed, operator):
    rng = random.Random(734)
    modulus = 1 << bits
    for _ in range(150):
        a, c = rng.randrange(-modulus, modulus), rng.randrange(-modulus, modulus)
        b, d = a + rng.randrange(8), c + rng.randrange(8)
        result = integer_arithmetic_intervals(
            operator, [(a, b)], [(c, d)], bits, signed
        )
        if result is None:
            assert signed
            continue
        for left in range(a, b + 1):
            for right in range(c, d + 1):
                x, y = _cast(left, bits, signed), _cast(right, bits, signed)
                value = (
                    x + y if operator == "+" else x - y if operator == "-" else x * y
                )
                if signed:
                    assert -(modulus // 2) <= value < modulus // 2
                expected = _cast(value, bits, signed)
                assert any(low <= expected <= high for low, high in result)


def test_negative_stride_retains_unsigned_wrap_before_signed_conversion():
    result = integer_arithmetic_intervals("*", [(0, 16)], [(-2, -2)], 32, False)
    assert result == [(0, 0), (2**32 - 32, 2**32 - 1)]
    assert convert_integer_intervals(result, 32, True) == [(-32, 0)]


def test_fragmentation_widens_without_losing_reachable_values():
    assert merge_integer_intervals([(i, i) for i in range(0, 40, 2)]) == [(0, 38)]


@pytest.mark.parametrize("bits,signed", [(8, True), (32, True), (64, True)])
def test_signed_overflow_never_becomes_wrapping_proof(bits, signed):
    maximum = (1 << (bits - 1)) - 1
    assert (
        integer_arithmetic_intervals("+", [(maximum, maximum)], [(1, 1)], bits, signed)
        is None
    )
    assert (
        integer_arithmetic_intervals(
            "-", [(0, 0)], [(-maximum - 1, -maximum - 1)], bits, signed
        )
        is None
    )
