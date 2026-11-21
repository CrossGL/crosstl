"""The Mojo feature list's GPU functions, read from the gpu package: warp, block, sync, index and atomic."""

import pytest

from tools.coverage import probes, validators
from tools.coverage.execution import THREADS
from tools.coverage.features import mojo, mojo_source
from tools.coverage.model import REDUCE_CLOSURE, VOID, Param, Signature, Type

from .mojo_cases import (
    MOJO_BIT,
    MOJO_BIT_INIT,
    MOJO_MATH,
    MOJO_MATH_INIT,
    MOJO_NUMERICS,
    by_name,
)

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
    FuncType: _ReduceClosure,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
    num_lanes: Int,
    *,
    stride: Int = 1,
](val: SIMD[val_type, simd_width], func: FuncType) -> SIMD[
    val_type, simd_width
]:
    return val


def reduce[
    val_type: DType,
    simd_width: SIMDLength,
    FuncType: _ReduceClosure,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
](val: SIMD[val_type, simd_width], func: FuncType) -> SIMD[
    val_type, simd_width
]:
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
        "warp.reduce",
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
    # A reduction over a function the kernel passes is probed with shuffle_xor and a function the probe defines, the
    # sum of its two arguments; CrossGL has no operation that applies a passed function.
    for name in ("warp.reduce", "warp.lane_group_reduce"):
        reduce = features[name]
        assert reduce.signature == Signature(
            Type("float"),
            (Param(Type("float")), Param(Type.other(REDUCE_CLOSURE))),
        )
        assert reduce.crossgl == ""
        assert (
            reduce.note
            == "no CrossGL warp reduction applies a function the kernel passes"
        )
        with pytest.raises(probes.NoSpelling, match="applies a function"):
            probes.build(reduce, "crossgl")
    assert features["warp.reduce"].spelling == "warp.reduce[shuffle=warp.shuffle_xor]"
    assert features["warp.lane_group_reduce"].spelling == (
        "warp.lane_group_reduce[shuffle=warp.shuffle_xor, num_lanes=warp.WARP_SIZE]"
    )
    source = probes.build(features["warp.reduce"], "mojo").source
    assert "\ndef combine[dtype: DType, width: SIMDLength](\n" in source
    assert "warp.reduce[shuffle=warp.shuffle_xor](a0, combine)" in source
    # A function parameter the probes have no value for leaves the definition without a probe.
    defined = mojo_source.definitions(
        "def apply[f: def(Float32) -> Float32](x: Float32) -> Float32:\n    return x\n"
    )
    assert mojo.signature("apply", defined["apply"], "float") == (
        None,
        "",
        "no probe passes a function as the parameter f",
    )
    # A comment in a parameter list is not part of a parameter's kind: num_threads is an Int32, which the call names.
    assert features["named_barrier"].spelling == f"named_barrier[num_threads={THREADS}]"


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

    @__allow_legacy_custom_self_type
    @inline(.always)
    def fetch_sub[
        dtype: DType, //, *, ordering: Ordering = _DEFAULT_ARITHMETIC_ORDERING
    ](mut self: Atomic[Scalar[dtype], scope=_], rhs: Scalar[dtype]) -> Scalar[
        dtype
    ]:
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


def test_mojo_reads_atomic_static_methods_and_value_methods():
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
    # fetch_sub has no static form: its receiver is an Atomic value, and CrossGL adds the negated operand.
    fetch_sub = features["Atomic.fetch_sub"]
    assert fetch_sub.signature == Signature(
        Type("int"), (Param(Type("int"), "atomic"), Param(Type("int")))
    )
    assert fetch_sub.spelling == "Atomic.fetch_sub[ordering=Ordering.RELAXED]"
    assert fetch_sub.note == ""
    assert fetch_sub.crossgl == "atomicAdd({0}, -({1}))"
    assert fetch_sub.counterpart == fetch_add.signature
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


def test_mojo_atomic_value_methods_take_the_buffer_element_as_an_atomic():
    fetch_sub = _all_features()["Atomic.fetch_sub"]
    source = probes.build(fetch_sub, "mojo").source
    assert (
        "    _ = Atomic.fetch_sub[ordering=Ordering.RELAXED]"
        "((outp + i).bitcast[Atomic[Int32]]()[], a1)\n"
    ) in source
    assert "outp: UnsafePointer[Int32, MutAnyOrigin]" in source
    # The CrossGL probe calls the counterpart on the buffer element itself.
    assert "atomicAdd(outp[i], -(a1));" in probes.build(fetch_sub, "crossgl").source
    with pytest.raises(probes.ProbeError, match="glsl has no atomic values"):
        probes.build(fetch_sub, "glsl")


@pytest.mark.skipif(
    not validators.available("mojo"), reason="the Mojo compiler is not installed"
)
def test_mojo_type_checks_the_atomic_value_method_probe():
    fetch_sub = _all_features()["Atomic.fetch_sub"]
    for baseline in (False, True):
        probe = probes.build(fetch_sub, "mojo", baseline=baseline)
        verdict = validators.validate("mojo", probe.source, entry=probe.entry)
        assert verdict.accepted, verdict.log
    # The compiler checks the call: an operand of another type than the Atomic's is an error.
    probe = probes.build(fetch_sub, "mojo")
    wrong = probe.source.replace("()[], a1)", "()[], Float32(1.5))")
    assert wrong != probe.source
    assert not validators.validate("mojo", wrong, entry=probe.entry).accepted


@pytest.mark.skipif(
    not validators.available("mojo"), reason="the Mojo compiler is not installed"
)
def test_mojo_type_checks_the_generic_warp_reduction_probes():
    features = _all_features()
    for name in ("warp.reduce", "warp.lane_group_reduce"):
        for baseline in (False, True):
            probe = probes.build(features[name], "mojo", baseline=baseline)
            verdict = validators.validate("mojo", probe.source, entry=probe.entry)
            assert verdict.accepted, verdict.log
    # The compiler checks the function argument: a value that is not a reduction function is an error.
    probe = probes.build(features["warp.reduce"], "mojo")
    wrong = probe.source.replace("(a0, combine)", "(a0, a0)")
    assert wrong != probe.source
    assert not validators.validate("mojo", wrong, entry=probe.entry).accepted
