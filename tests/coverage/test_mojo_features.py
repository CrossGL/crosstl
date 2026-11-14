"""The Mojo feature list, read from the standard library packages."""

from tools.coverage import probes
from tools.coverage.features import mojo
from tools.coverage.model import (
    VOID,
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
MAX_WARP = """
from max.gpu.globals import WARP_SIZE


def _shuffle[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    return val


def shuffle_down[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    return val


def shuffle_down[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    return val


def lane_group_reduce[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
    func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing -> SIMD[dtype, width],
    num_lanes: Int,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    return val


def lane_group_sum[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    return val


def sum(val: SIMD) -> Scalar[val.dtype]:
    return val[0]


def max(val: SIMD) -> Scalar[val.dtype]:
    return val[0]


def vote[ret_type: DType](val: Bool) -> Scalar[ret_type]:
    comptime if is_nvidia_gpu():
        comptime assert ret_type == .uint32, "Unsupported return type"
        return 0
    return 0


def match_any[
    dtype: DType,
    //,
    mask_type: DType = (DType.uint32 if WARP_SIZE <= 32 else DType.uint64),
](value: Scalar[dtype]) -> Scalar[mask_type]:
    return 0
"""

MAX_SYNC_INIT = """
from .semaphore import NamedBarrierSemaphore, Semaphore
from .sync import (
    AMDScheduleBarrierMask,
    barrier,
    named_barrier,
    syncwarp,
)
"""

MAX_SYNC = """
def named_barrier[
    num_threads: Int32,
    id: Int32 = 0,  # the barrier's id
]():
    pass


def barrier():
    pass


def syncwarp(mask: Int = -1):
    pass
"""


def _gpu_features():
    return by_name(
        mojo.parse(
            MOJO_MATH_INIT,
            MOJO_MATH,
            MOJO_NUMERICS,
            MOJO_BIT_INIT,
            MOJO_BIT,
            warp=MAX_WARP,
            sync_init=MAX_SYNC_INIT,
            sync=MAX_SYNC,
        )
    )


def test_mojo_reads_max_gpu_warp_and_sync_functions():
    features = _gpu_features()
    warp = {name for name, f in features.items() if f.group == "warp"}
    sync = {name for name, f in features.items() if f.group == "sync"}
    # warp's public API is its definitions without a leading underscore; its sum and max are named through the
    # module, so they stay apart from the std.math functions of the same names.
    assert warp == {
        "warp.shuffle_down",
        "warp.lane_group_reduce",
        "warp.lane_group_sum",
        "warp.sum",
        "warp.max",
        "warp.vote",
        "warp.match_any",
    }
    # max.gpu.sync's functions are those its __init__.mojo imports from the sync module; types are not functions.
    assert sync == {"barrier", "named_barrier", "syncwarp"}
    assert {f.extension for f in features.values() if f.group == "warp"} == {mojo.WARP}
    assert features["barrier"].extension == "max.gpu.sync.barrier"
    # The first overload is probed: the one without a lane mask.
    assert features["warp.shuffle_down"].signature == Signature(
        Type("float"), (Param(Type("float")), Param(Type("uint")))
    )
    assert features["warp.shuffle_down"].crossgl == "subgroupShuffleDown"
    assert features["syncwarp"].signature == Signature(VOID, ())
    assert features["syncwarp"].crossgl == "subgroupBarrier"


def test_mojo_probes_warp_reductions_over_the_whole_warp():
    features = _gpu_features()
    lane_group = features["warp.lane_group_sum"]
    assert lane_group.spelling == "warp.lane_group_sum[num_lanes=warp.WARP_SIZE]"
    assert lane_group.crossgl == "subgroupAdd"
    # Scalar[val.dtype] is the element type of the argument val.
    assert features["warp.sum"].signature == Signature(
        Type("float"), (Param(Type("float")),)
    )
    assert features["warp.max"].crossgl == "subgroupMax"
    # A reduction over a function the kernel passes has no probe and no CrossGL operation.
    reduce = features["warp.lane_group_reduce"]
    assert reduce.signature is None
    assert reduce.crossgl == ""
    assert (
        reduce.note == "no CrossGL warp reduction applies a function the kernel passes"
    )
    # A comment in a parameter list is not part of a parameter's kind.
    assert features["named_barrier"].note == (
        "no probe value for the parameter num_threads: Int32"
    )


def test_mojo_names_the_mask_type_of_votes_and_matches():
    features = _gpu_features()
    # vote's ret_type is required and asserted to be uint32 for NVIDIA.
    vote = features["warp.vote"]
    assert vote.spelling == "warp.vote[ret_type=DType.uint32]"
    assert vote.signature == Signature(Type("uint"), (Param(Type("bool")),))
    assert vote.crossgl == "subgroupBallot({0}).x"
    # match_any's mask type defaults to an expression a host compile cannot fold, so the probe names its first type.
    match = features["warp.match_any"]
    assert match.spelling == "warp.match_any[mask_type=DType.uint32]"
    assert match.signature.result == Type("uint")


def test_mojo_probes_import_the_gpu_module_they_call():
    features = _gpu_features()
    source = probes.build(features["warp.shuffle_down"], "mojo").source
    assert "from max.gpu.primitives import warp\n" in source
    assert "warp.shuffle_down(" in source
    source = probes.build(features["barrier"], "mojo").source
    assert "from max.gpu.sync import barrier\n" in source
    assert "    barrier()\n" in source
    # The CrossGL probe calls CrossGL's operation and imports nothing.
    crossgl = probes.build(features["warp.shuffle_down"], "crossgl").source
    assert "subgroupShuffleDown(" in crossgl
    assert "import" not in crossgl
