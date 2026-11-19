"""The Mojo feature list, read from the standard library packages: the math, numerics and bit functions."""

from tools.coverage import probes
from tools.coverage.features import mojo
from tools.coverage.model import Param, Signature, Type

from .mojo_cases import (
    MOJO_BIT,
    MOJO_BIT_INIT,
    MOJO_MATH,
    MOJO_MATH_INIT,
    MOJO_NUMERICS,
    by_name,
)


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
        "isclose",
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
    # isclose keeps its keyword defaults: equal values, or finite ones within max(atol, rtol * max |x|).
    isclose = features["isclose"]
    assert isclose.signature == Signature(
        Type("bool"), (Param(Type("float")), Param(Type("float")))
    )
    assert (
        "bool r = ((a0 == a1) || (isfinite(a0) && isfinite(a1)"
        " && abs(a0 - a1) <= max(1e-08, 1e-05 * max(abs(a0), abs(a1)))));"
        in probes.build(isclose, "crossgl").source
    )
    # gcd needs a loop, so its probe still calls Mojo's name.
    assert features["gcd"].crossgl == "gcd"


# Excerpts of MAX 26.7.0's max/gpu/primitives/warp.mojo, max/gpu/sync/__init__.mojo and max/gpu/sync/sync.mojo.


def test_mojo_splits_a_default_at_the_top_level_equals_sign():
    assert mojo._default("ptr: MutPointer[Scalar[dtype], _, address_space=_]") == (
        "ptr: MutPointer[Scalar[dtype], _, address_space=_]",
        "",
    )
    assert mojo._default("mask: Int = -1") == ("mask: Int ", " -1")
