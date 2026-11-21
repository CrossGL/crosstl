"""Excerpts of the Mojo standard library packages the feature list is read from, and features keyed by name."""


def by_name(features):
    return {feature.name: feature for feature in features}


MOJO_MATH_INIT = """
from std.utils.numerics import isnan, nan
from .math import (
    Floorable,
    floor,
    frexp,
    gcd,
    isclose,
    mulwide,
    rsqrt,
    sqrt,
)
"""

MOJO_MATH = """
from std.utils.numerics import isnan


@always_inline
def floor[T: Floorable, //](value: T) -> T:
    \"\"\"Floor.\"\"\"
    return value.__floor__()


def _sqrt_nvvm(x: SIMD, out res: type_of(x)):
    pass


def sqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert (
        dtype.is_numeric() or dtype == .bool
    ), "type must be arithmetic or boolean"

    comptime if dtype.is_integral():
        return x
    return x


def sqrt(x: Int) -> Int:
    return x


def rsqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    return x


def frexp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> StaticTuple[
    SIMD[dtype, width], 2
] where dtype.is_floating_point():
    return x


def gcd(m: Int, n: Int, /) -> Int:
    return m


def isclose[
    dtype: DType,
    width: SIMDLength,
    *,
    symmetrical: Bool = True,
](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    *,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) -> SIMD[.bool, width]:
    return a.eq(b)


def mulwide(a: UInt32, b: UInt32) -> UInt64:
    return UInt64(a) * UInt64(b)


def mulwide(a: Int32, b: Int32) -> Int64:
    return Int64(a) * Int64(b)
"""

MOJO_NUMERICS = """
def nan[dtype: DType]() -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), "fp only"
    return 0


def isnan[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[.bool, width]:
    return False
"""

MOJO_BIT_INIT = """
from .bit import (
    pop_count,
    rotate_bits_left,
)
"""

MOJO_BIT = """
def pop_count[dtype: DType, width: SIMDLength, //](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert dtype.is_integral(), "must be integral"
    return val


def rotate_bits_left[
    dtype: DType,
    width: SIMDLength,
    //,
    shift: Int,
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_integral():
    return x
"""
