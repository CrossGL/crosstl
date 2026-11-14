"""The Mojo feature list, read from the standard library packages."""

from tools.coverage import probes
from tools.coverage.execution import THREADS
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


# Excerpts of MAX 26.7.0's max/gpu/primitives/block.mojo and id.mojo, Mojo 1.2.0's std/_gpu/primitives/id.mojo, and
# std/atomic/__init__.mojo and atomic.mojo.
MAX_BLOCK = """
from max.gpu.primitives import warp


def _block_reduce[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    return val


def sum[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_size: Int,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return val
"""

MAX_ID = """
from max.gpu.globals import WARP_SIZE


@__doc_inline
from std._gpu.primitives.id import (
    block_dim,
    global_idx,
    lane_id,
    thread_idx,
)


@inline(.nodebug)
def warp_id[*, broadcast: Bool = False]() -> Int:
    return 0


@inline(.nodebug)
def sm_id() -> Int:
    return 0
"""

MOJO_LANES = """
comptime thread_idx = _ThreadIdx()
comptime global_idx = _GlobalIdx()


@inline(.nodebug)
def lane_id() -> Int:
    return _lane_id()


@inline(.nodebug)
def _lane_id() -> Int:
    return 0
"""

MOJO_ATOMIC_INIT = """
from .atomic import Atomic, Ordering, fence
"""

MOJO_ATOMIC = """
comptime _DEFAULT_MEMORY_ORDERING = Ordering.SEQUENTIAL


struct Ordering(Equatable):
    def __init__(out self, value: Int):
        pass


@inline(.nodebug)
def fence[
    ordering: Ordering = _DEFAULT_MEMORY_ORDERING,
    *,
    scope: StaticString = "",
]():
    pass


struct Atomic[T: Deinitable & Movable, *, scope: StaticString = ""]:
    var _value: Self.T

    @inline(.always)
    def __init__(out self, var value: Self.T):
        self._value = value^

    @staticmethod
    @inline(.nodebug)
    def load[
        dtype: DType,
        //,
        *,
        ordering: Ordering = _DEFAULT_MEMORY_ORDERING,
    ](ptr: ImmPointer[Scalar[dtype], _, address_space=_]) -> Scalar[
        dtype
    ] where (Self.T == Scalar[dtype]):
        return ptr[]

    @inline(.nodebug)
    def load[*, ordering: Ordering = _DEFAULT_MEMORY_ORDERING](self) -> Self.T:
        return self._value

    @staticmethod
    @inline(.always)
    def fetch_add[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_ARITHMETIC_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        rhs: Scalar[dtype],
    ) -> Scalar[dtype] where (Self.T == Scalar[dtype]):
        return rhs

    @staticmethod
    @inline(.always)
    def _xchg[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_MEMORY_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        value: Scalar[dtype],
    ) -> Scalar[dtype] where (Self.T == Scalar[dtype]):
        return value

    @inline(.always)
    def fetch_sub[
        *, ordering: Ordering = _DEFAULT_ARITHMETIC_ORDERING
    ](mut self, rhs: Self.T) -> Self.T:
        return rhs

    @staticmethod
    @inline(.always)
    def max[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_COMPARISON_ORDERING
    ](
        ptr: MutPointer[Scalar[dtype], _, address_space=_],
        rhs: Scalar[dtype],
    ) where (Self.T == Scalar[dtype]):
        pass


def _helper():
    pass
"""


def _all_features():
    return by_name(
        mojo.parse(
            MOJO_MATH_INIT,
            MOJO_MATH,
            MOJO_NUMERICS,
            MOJO_BIT_INIT,
            MOJO_BIT,
            warp=MAX_WARP,
            block=MAX_BLOCK,
            sync_init=MAX_SYNC_INIT,
            sync=MAX_SYNC,
            ids=MAX_ID,
            lanes=MOJO_LANES,
            atomic_init=MOJO_ATOMIC_INIT,
            atomic=MOJO_ATOMIC,
        )
    )


def test_mojo_reads_block_reductions_and_probes_them_over_one_workgroup():
    features = _all_features()
    assert {name for name, f in features.items() if f.group == "block"} == {"block.sum"}
    block_sum = features["block.sum"]
    assert block_sum.extension == mojo.BLOCK
    # block_size is keyword-only and required; the probe passes the invocations of one subgroup probe's workgroup.
    assert block_sum.spelling == f"block.sum[block_size={THREADS}]"
    assert block_sum.signature == Signature(Type("float"), (Param(Type("float")),))
    assert block_sum.crossgl == ""
    assert block_sum.note == (
        "CrossGL reduces, scans and broadcasts over a subgroup only, not over a workgroup"
    )


def test_mojo_reads_the_gpu_index_functions_but_not_the_index_values():
    features = _all_features()
    # lane_id is re-exported from std._gpu, where thread_idx and global_idx are values rather than functions.
    assert {name for name, f in features.items() if f.group == "id"} == {
        "lane_id",
        "warp_id",
        "sm_id",
    }
    assert features["lane_id"].extension == "max.gpu.lane_id"
    assert features["lane_id"].signature == Signature(Type("int64"), ())
    assert features["lane_id"].crossgl == "int64(gl_SubgroupInvocationID)"
    assert features["warp_id"].crossgl == (
        "int64(gl_LocalInvocationID.x / gl_SubgroupSize)"
    )
    source = probes.build(features["lane_id"], "mojo").source
    assert "from max.gpu import lane_id\n" in source
    assert "    var r: Int = lane_id()\n" in source


def test_mojo_reads_atomic_static_methods_and_lists_value_methods_untested():
    features = _all_features()
    atomic = {name for name, f in features.items() if f.group == "atomic"}
    # Private methods are skipped, and a name with a static and a value overload is probed through the static one.
    assert atomic == {
        "Atomic.load",
        "Atomic.fetch_add",
        "Atomic.fetch_sub",
        "Atomic.max",
        "fence",
    }
    fetch_add = features["Atomic.fetch_add"]
    assert fetch_add.extension == "std.atomic.Atomic std.atomic.Ordering"
    assert fetch_add.spelling == "Atomic.fetch_add[ordering=Ordering.RELAXED]"
    # The pointer is the buffer element the atomic operation updates.
    assert fetch_add.signature == Signature(
        Type("int"), (Param(Type("int"), "inout"), Param(Type("int")))
    )
    assert fetch_add.crossgl == "atomicAdd"
    assert features["Atomic.max"].signature.result == VOID
    assert features["Atomic.load"].counterpart == Signature(
        Type("int"), (Param(Type("int"), "inout"),)
    )
    fetch_sub = features["Atomic.fetch_sub"]
    assert fetch_sub.signature is None
    assert fetch_sub.note == (
        "a method of an Atomic value only, and the probes call Atomic's static methods on a buffer element"
    )
    assert features["fence"].extension == "std.atomic.fence"
    assert features["fence"].crossgl == "memoryBarrier"


def test_mojo_atomic_probes_pass_the_buffer_element_and_discard_the_result():
    features = _all_features()
    source = probes.build(features["Atomic.fetch_add"], "mojo").source
    assert "from std.atomic import Atomic\nfrom std.atomic import Ordering\n" in source
    assert (
        "    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](outp + i, a1)\n" in source
    )
    source = probes.build(features["Atomic.max"], "mojo").source
    assert "    Atomic.max[ordering=Ordering.RELAXED](outp + i, a1)\n" in source
    crossgl = probes.build(features["Atomic.fetch_add"], "crossgl").source
    assert "atomicAdd(outp[i], a1);" in crossgl
    assert "import" not in crossgl


def test_mojo_splits_a_default_at_the_top_level_equals_sign():
    assert mojo._default("ptr: MutPointer[Scalar[dtype], _, address_space=_]") == (
        "ptr: MutPointer[Scalar[dtype], _, address_space=_]",
        "",
    )
    assert mojo._default("mask: Int = -1") == ("mask: Int ", " -1")
