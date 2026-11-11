"""The Mojo feature list, read from the standard library packages."""

from tools.coverage import probes
from tools.coverage.features import mojo
from tools.coverage.model import (
    Param,
    Signature,
    Type,
)


def by_name(features):
    return {feature.name: feature for feature in features}


MOJO_MATH_INIT = """
from std.utils.numerics import isnan, nan
from .math import (
    Floorable,
    floor,
    frexp,
    gcd,
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


def test_mojo_reads_the_functions_the_packages_export():
    glsl_names = {"floor", "inversesqrt", "bitCount", "isnan"}
    found = mojo.parse(
        MOJO_MATH_INIT, MOJO_MATH, MOJO_NUMERICS, MOJO_BIT_INIT, MOJO_BIT, glsl_names
    )
    features = by_name(found)
    # Traits and private helpers are not functions of the public API.
    assert sorted(features) == [
        "floor",
        "frexp",
        "gcd",
        "isnan",
        "mulwide",
        "nan",
        "pop_count",
        "rotate_bits_left",
        "rsqrt",
        "sqrt",
    ]
    assert {f.group for f in found} == {"math", "bit"}
    assert features["floor"].signature == Signature(
        Type("float"), (Param(Type("float")),)
    )
    # The first definition is probed, and a comptime if on the type does not constrain it.
    assert features["sqrt"].signature.params == (Param(Type("float")),)
    assert features["rsqrt"].signature.result == Type("float")
    assert features["rsqrt"].crossgl == "inversesqrt"
    assert features["isnan"].signature.result == Type("bool")
    assert features["pop_count"].signature.result == Type("int")
    assert features["pop_count"].crossgl == "bitCount"
    assert features["gcd"].signature == Signature(
        Type("int64"), (Param(Type("int64")), Param(Type("int64")))
    )
    # UInt64 is the unsigned 64-bit scalar, and the first of the two definitions is probed.
    assert features["mulwide"].signature == Signature(
        Type("uint64"), (Param(Type("uint")), Param(Type("uint")))
    )
    # frexp returns a pair the probe reads the first value of, and CrossGL writes it the way GLSL does.
    assert features["frexp"].signature.result == Type.other("StaticTuple[Float32, 2]")
    assert features["frexp"].counterpart == Signature(
        Type("float"), (Param(Type("float")), Param(Type("int"), "out"))
    )


def test_mojo_names_the_parameters_no_argument_fixes():
    features = by_name(
        mojo.parse(MOJO_MATH_INIT, MOJO_MATH, MOJO_NUMERICS, MOJO_BIT_INIT, MOJO_BIT)
    )
    assert features["nan"].spelling == "nan[dtype=DType.float32]"
    assert features["nan"].signature == Signature(Type("float"), ())
    assert features["rotate_bits_left"].spelling == "rotate_bits_left[shift=1]"
    assert features["rotate_bits_left"].signature.params == (Param(Type("int")),)
    assert features["sqrt"].spelling == ""


def test_mojo_writes_bit_and_math_functions_crossgl_lacks_as_expressions():
    features = by_name(
        mojo.parse(MOJO_MATH_INIT, MOJO_MATH, MOJO_NUMERICS, MOJO_BIT_INIT, MOJO_BIT)
    )
    # rotate_bits_left[shift=1] shifts the value left by one and brings its top bit back in at the bottom.
    rotate = features["rotate_bits_left"]
    assert rotate.crossgl == "(({0} << 1) | int(uint({0}) >> 31u))"
    assert (
        "int r = ((a0 << 1) | int(uint(a0) >> 31u));" in probes.build(
            rotate, "crossgl"
        ).source
    )
    # mulwide widens both 32-bit factors before it multiplies them.
    assert (
        "uint64_t r = (uint64(a0) * uint64(a1));" in probes.build(
            features["mulwide"], "crossgl"
        ).source
    )
    # gcd needs a loop, so its probe still calls Mojo's name.
    assert features["gcd"].crossgl == "gcd"
