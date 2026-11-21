"""Wave (subgroup) operations of the Mojo generator, written as loops over the active lanes of a MAX warp.

CrossGL's wave operations combine the values of the *active* lanes of a subgroup, those that execute the call
(:mod:`crosstl.translator.wave_operations`; HLSL Shader Model 6.0 "Wave Intrinsics"; ``GL_KHR_shader_subgroup``).
MAX's ``warp.sum``, ``warp.max`` and ``warp.prefix_sum`` instead combine every lane of the warp with the full lane
mask (``max/gpu/primitives/warp.mojo``), which NVIDIA leaves undefined when a lane of the mask does not execute the
shuffle (PTX ISA, ``shfl.sync``) and which reads stale values of inactive lanes on AMD. So the Mojo generator names
the active lanes itself and reads each of them with ``shuffle_idx`` under that mask, as the CUDA and HIP generators
do (:mod:`.warp_wave_helpers`):

* the active lanes are NVVM's ``activemask`` on NVIDIA, and MAX's ``vote`` of ``True`` on AMD and Apple GPUs, whose
  ballots (``llvm.amdgcn.ballot``, ``llvm.air.simd_ballot``) count active lanes only;
* a ballot of a predicate is NVVM's ``vote.ballot.sync`` under the active mask on NVIDIA, and ``vote`` elsewhere;
* a lane read is ``shuffle_idx(mask, value, lane)`` per component, through a 32-bit integer for 8- and 16-bit ones,
  which MAX's NVIDIA shuffle does not take.

Every active lane runs the same loop iterations in the same order, so each obtains the same reduction. A Mojo warp is
the subgroup: ``WARP_SIZE`` lanes, 32 on NVIDIA and Apple GPUs and 32 or 64 on AMD, so lane masks are ``UInt64``.
Quad operations need CrossGL's compute quad layout, which this module does not model, and stay refused.

``subgroupBarrier`` waits for the active lanes and orders every memory access around that point
(``GL_KHR_shader_subgroup``; SPIR-V ``OpControlBarrier`` with Subgroup scope and AcquireRelease over uniform,
workgroup and image memory). MAX's ``syncwarp`` is the execution barrier, under the active mask on NVIDIA
(``bar.warp.sync``), and orders no memory on Apple GPUs, so a sequentially consistent system-scope fence
(``std.atomic.fence``, stronger than a subgroup one, and defined on every GPU, unlike MAX's NVIDIA-only
``threadfence``) stands on each side of it. ``subgroupMemoryBarrier*`` are that fence alone.

CrossGL programs may also call CUDA's ``__match_any_sync``. A CUDA warp is 32 lanes of consecutive linear index
(CUDA C++ Programming Guide, "Warp Match Functions"); MAX numbers a warp's lanes the same way (``lane_id``), so the
CUDA warp of a lane is lane / 32 in a 64-lane AMD warp and the whole warp elsewhere. The result is the bit pattern
match of ``match.any.sync.b32``/``.b64`` among the lanes of the caller's CUDA warp that its mask names.
"""

from __future__ import annotations

from ..refusals import MojoRefusal
from ..wave_operations import (
    WAVE_BALLOT_BIT_OPERATIONS,
    WAVE_BALLOT_BOOL_RESULT_OPERATIONS,
    WAVE_BALLOT_UINT_RESULT_OPERATIONS,
    WAVE_CLUSTERED_OPERATIONS,
    WAVE_EXTENDED_PREFIX_OPERATIONS,
)
from .mojo_tables import (
    MOJO_DTYPE_SUFFIX,
    MOJO_SCALAR_DTYPES,
    MOJO_WAVE_BOOL_RESULT_OPS,
    MOJO_WAVE_OP_ARITIES,
    MOJO_WAVE_UINT_RESULT_OPS,
    MOJO_WAVE_UVEC4_RESULT_OPS,
)

__all__ = ["MOJO_WAVE_IMPORTS", "MojoWaveOpMixin", "mojo_wave_arity"]

# The imports the helpers need, under names a program cannot take.
MOJO_WAVE_IMPORTS = (
    "from max.gpu.primitives.warp import shuffle_idx as _crossgl_shuffle_idx, "
    "vote as _crossgl_vote\n"
    "from std.sys import is_nvidia_gpu as _crossgl_is_nvidia_gpu, "
    "llvm_intrinsic as _crossgl_llvm_intrinsic\n"
    "from std.bit import count_trailing_zeros as _crossgl_ctz, "
    "count_leading_zeros as _crossgl_clz, pop_count as _crossgl_pop_count\n"
    "from std.memory import bitcast as _crossgl_bitcast\n"
    "from std.atomic import fence as _crossgl_fence\n"
    "from max.gpu.sync import syncwarp as _crossgl_syncwarp\n"
)

_PRELUDE = """def _crossgl_active_lanes() -> UInt64:
    comptime if _crossgl_is_nvidia_gpu():
        return UInt64(_crossgl_llvm_intrinsic["llvm.nvvm.activemask", UInt32]())
    else:
        return _crossgl_vote[DType.uint64](True)


def _crossgl_lane_ballot(active: UInt64, predicate: Bool) -> UInt64:
    comptime if _crossgl_is_nvidia_gpu():
        return UInt64(
            _crossgl_llvm_intrinsic["llvm.nvvm.vote.ballot.sync", UInt32](
                UInt32(active), predicate
            )
        )
    else:
        return _crossgl_vote[DType.uint64](predicate) & active


def _crossgl_lane_read[
    dtype: DType, width: SIMDLength, //
](active: UInt64, value: SIMD[dtype, width], lane: UInt32) -> SIMD[dtype, width]:
    var result = value
    for component in range(width):
        comptime if dtype in (DType.int8, DType.uint8, DType.int16, DType.uint16):
            result[component] = _crossgl_shuffle_idx(
                UInt(active), value[component].cast[DType.int32](), lane
            ).cast[dtype]()
        else:
            result[component] = _crossgl_shuffle_idx(
                UInt(active), value[component], lane
            )
    return result


def _crossgl_lane_mask(mask: SIMD[DType.uint32, 4]) -> UInt64:
    return ((UInt64(mask[1]) << 32) | UInt64(mask[0])) & (
        UInt64.MAX >> UInt64(64 - WARP_SIZE)
    )


def _crossgl_ballot_vector(bits: UInt64) -> SIMD[DType.uint32, 4]:
    return SIMD[DType.uint32, 4](UInt32(bits & 0xFFFFFFFF), UInt32(bits >> 32), 0, 0)


"""

# Operator of each reduction, scan and partitioned scan: name -> (operator, scope). The scope is "reduce",
# "inclusive", "exclusive" or "cluster"; a partitioned operation adds "partition" (Shader Model 6.5's
# WaveMultiPrefix* scans are exclusive).
_FOLDS = {
    "WaveActiveSum": ("sum", "reduce"),
    "WaveActiveProduct": ("product", "reduce"),
    "WaveActiveMin": ("min", "reduce"),
    "WaveActiveMax": ("max", "reduce"),
    "WaveActiveBitAnd": ("and", "reduce"),
    "WaveActiveBitOr": ("or", "reduce"),
    "WaveActiveBitXor": ("xor", "reduce"),
    "WavePrefixSum": ("sum", "exclusive"),
    "WavePrefixProduct": ("product", "exclusive"),
    "WavePrefixInclusiveSum": ("sum", "inclusive"),
    "WavePrefixInclusiveProduct": ("product", "inclusive"),
    **{
        name: (operator, "inclusive" if inclusive else "exclusive")
        for name, (operator, inclusive) in WAVE_EXTENDED_PREFIX_OPERATIONS.items()
    },
    **{
        name: (operator, "cluster")
        for name, operator in WAVE_CLUSTERED_OPERATIONS.items()
    },
    "WaveMultiPrefixSum": ("sum", "partition"),
    "WaveMultiPrefixProduct": ("product", "partition"),
    "WaveMultiPrefixBitAnd": ("and", "partition"),
    "WaveMultiPrefixBitOr": ("or", "partition"),
    "WaveMultiPrefixBitXor": ("xor", "partition"),
}
_BITWISE = frozenset({"and", "or", "xor"})
_COMBINE = {
    "sum": "({left} + {right})",
    "product": "({left} * {right})",
    "min": "min({left}, {right})",
    "max": "max({left}, {right})",
    "and": "({left} & {right})",
    "or": "({left} | {right})",
    "xor": "({left} ^ {right})",
}
# The identity of each operator, the result of a lane that combines no lane (SPIR-V OpGroupNonUniform*: +inf and
# -inf for floating-point min and max, the type's extremes for integers). Mojo's MAX and MIN are the infinities
# of a floating-point type.
_IDENTITY = {
    "sum": "{type}(0)",
    "product": "{type}(1)",
    "min": "{type}.MAX",
    "max": "{type}.MIN",
    "and": "~{type}(0)",
    "or": "{type}(0)",
    "xor": "{type}(0)",
}
# Lanes a fold combines, given the reading lane ``source`` and the calling lane ``lane``.
_MEMBERS = {
    "reduce": "True",
    "inclusive": "source <= lane",
    "exclusive": "source < lane",
    "cluster": "(source & ~(cluster_size - 1)) == (lane & ~(cluster_size - 1))",
    "partition": "source < lane and ((partition >> source) & 1) != 0",
}
# The lane each relative read names, from the calling lane and the operation's second argument.
_RELATIVE_LANES = {
    "WaveShuffleXor": "lane ^ operand",
    "WaveShuffleUp": "lane - operand",
    "WaveShuffleDown": "lane + operand",
}
_PREDICATE_OPERATIONS = frozenset(
    {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "WaveMultiPrefixCountBits",
    }
)
_ARITIES = {
    **MOJO_WAVE_OP_ARITIES,
    **{name: 2 for name in WAVE_CLUSTERED_OPERATIONS},
    **{name: 1 for name in WAVE_EXTENDED_PREFIX_OPERATIONS},
    **WAVE_BALLOT_BIT_OPERATIONS,
    "WavePrefixInclusiveSum": 1,
    "WavePrefixInclusiveProduct": 1,
    "WaveShuffleXor": 2,
    "WaveShuffleUp": 2,
    "WaveShuffleDown": 2,
}
_QUAD_OPERATIONS = frozenset(
    name for name in MOJO_WAVE_OP_ARITIES if name.startswith("Quad")
)
_UVEC4 = "SIMD[DType.uint32, 4]"
_SUBGROUP_MEMORY_BARRIERS = frozenset(
    {
        "subgroupMemoryBarrier",
        "subgroupMemoryBarrierBuffer",
        "subgroupMemoryBarrierShared",
        "subgroupMemoryBarrierImage",
    }
)
# The unsigned type of each operand type __match_any_sync compares by its bits (match.any.sync.b32 and .b64).
_MATCH_BITS = {
    "DType.int32": "DType.uint32",
    "DType.uint32": "DType.uint32",
    "DType.float32": "DType.uint32",
    "DType.int64": "DType.uint64",
    "DType.uint64": "DType.uint64",
    "DType.float64": "DType.uint64",
}
_FLOAT_DTYPES = frozenset({"DType.float16", "DType.float32", "DType.float64"})


def mojo_wave_arity(operation):
    """The argument count of a wave operation the Mojo generator knows, or ``None``."""
    return _ARITIES.get(operation)


class _Operand:
    """The Mojo spelling of a wave operand: its type, element ``DType``, logical width and helper suffix."""

    def __init__(self, mojo_type, dtype, width):
        self.mojo_type = mojo_type
        self.dtype = dtype
        self.width = width
        suffix = MOJO_DTYPE_SUFFIX[dtype]
        self.suffix = suffix if width == 1 else f"{suffix}x{width}"

    @property
    def is_bool(self):
        return self.dtype == "DType.bool"

    @property
    def is_bool_scalar(self):
        return self.is_bool and self.width == 1


class MojoWaveOpMixin:
    """Wave-operation members of ``MojoCodeGen``."""

    def generate_wave_op(self, expr):
        operation = getattr(expr, "operation", "wave intrinsic")
        args = list(getattr(expr, "arguments", []) or [])
        expected_count = mojo_wave_arity(operation)
        if expected_count is None:
            raise MojoRefusal(f"Unsupported Mojo wave intrinsic {operation}")
        if len(args) != expected_count:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(args)}"
            )
        if operation in _QUAD_OPERATIONS:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation}: the Mojo generator does not lower quad "
                "operations, whose lanes depend on CrossGL's compute quad layout"
            )
        if operation == "WaveGetLaneCount":
            return "UInt32(WARP_SIZE)"
        if operation == "WaveGetLaneIndex":
            return "UInt32(lane_id())"
        return self.mojo_wave_helper_call(operation, args)

    def mojo_wave_result_type(self, operation, args):
        if operation in MOJO_WAVE_UINT_RESULT_OPS or operation in (
            WAVE_BALLOT_UINT_RESULT_OPERATIONS
        ):
            return "uint"
        if operation in MOJO_WAVE_BOOL_RESULT_OPS or operation in (
            WAVE_BALLOT_BOOL_RESULT_OPERATIONS
        ):
            return "bool"
        if operation in MOJO_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if args:
            return self.expression_result_type(args[0]) or "uint"
        return "uint"

    # Helper registry ---------------------------------------------------------------------------------------

    def mojo_wave_helpers(self):
        helpers = getattr(self, "required_wave_helpers", None)
        if helpers is None:
            helpers = self.required_wave_helpers = {}
        return helpers

    def mojo_wave_helper_source(self):
        """The prelude and every helper the program calls, in the order they were first needed."""
        helpers = self.mojo_wave_helpers()
        if not helpers:
            return ""
        return _PRELUDE + "".join(source + "\n\n" for source in helpers.values())

    def require_mojo_wave_helper(self, name, signature, body):
        helpers = self.mojo_wave_helpers()
        if name not in helpers:
            helpers[name] = f"def {name}({signature}:\n" + "".join(
                f"    {line}\n" for line in body
            )
        return name

    # Operands ----------------------------------------------------------------------------------------------

    def mojo_wave_operand(self, operation, arg):
        type_name = self.type_name_string(self.expression_result_type(arg))
        if type_name is None:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation}: the type of its operand is unknown"
            )
        vector = self.vector_type_info(type_name)
        if vector is not None:
            dtype, width = vector[0], vector[1]
        else:
            dtype, width = MOJO_SCALAR_DTYPES.get(type_name), 1
        if dtype not in MOJO_DTYPE_SUFFIX:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation}: a {type_name} operand has no lane read; "
                "Mojo shuffles scalars and vectors of numbers and booleans"
            )
        return _Operand(self.map_type(type_name), dtype, width)

    def mojo_wave_predicate(self, operation, arg):
        type_name = self.type_name_string(self.expression_result_type(arg))
        if type_name not in (None, "bool"):
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation} requires a bool predicate, got {type_name}"
            )
        return f"Bool({self.generate_expression(arg)})"

    def mojo_wave_uint(self, arg):
        return f"UInt32({self.generate_expression(arg)})"

    def mojo_wave_partition(self, operation, arg):
        type_name = self.type_name_string(self.expression_result_type(arg))
        if type_name not in (None, "uvec4"):
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation} requires a uvec4 partition mask, got {type_name}"
            )
        return self.generate_expression(arg)

    # Calls -------------------------------------------------------------------------------------------------

    def mojo_wave_helper_call(self, operation, args):
        if operation == "WaveIsFirstLane":
            name = self.require_mojo_wave_helper(
                "_crossgl_wave_is_first_lane",
                ") -> Bool",
                ["return UInt64(lane_id()) == _crossgl_ctz(_crossgl_active_lanes())"],
            )
            return f"{name}()"
        if operation in _PREDICATE_OPERATIONS:
            return self.mojo_wave_predicate_call(operation, args)
        if operation in WAVE_BALLOT_BIT_OPERATIONS:
            return self.mojo_wave_ballot_bits_call(operation, args)
        operand = self.mojo_wave_operand(operation, args[0])
        value = self.generate_expression(args[0])
        if operation in _FOLDS:
            return self.mojo_wave_fold_call(operation, operand, value, args)
        if operation in {"WaveReadLaneAt", "WaveReadLaneFirst", *_RELATIVE_LANES}:
            return self.mojo_wave_read_call(operation, operand, value, args)
        if operation == "WaveActiveAllEqual":
            return self.mojo_wave_all_equal_call(operand, value)
        if operation == "WaveMatch":
            return self.mojo_wave_match_call(operand, value)
        raise MojoRefusal(f"Unsupported Mojo wave intrinsic {operation}")

    def mojo_wave_predicate_call(self, operation, args):
        predicate = self.mojo_wave_predicate(operation, args[0])
        lines = ["var active = _crossgl_active_lanes()"]
        bits = "_crossgl_lane_ballot(active, predicate)"
        signature = "predicate: Bool"
        call_args = [predicate]
        if operation == "WaveActiveAllTrue":
            lines.append(f"return {bits} == active")
            result = "Bool"
        elif operation == "WaveActiveAnyTrue":
            lines.append(f"return {bits} != 0")
            result = "Bool"
        elif operation == "WaveActiveBallot":
            lines.append(f"return _crossgl_ballot_vector({bits})")
            result = _UVEC4
        elif operation == "WaveActiveCountBits":
            lines.append(f"return UInt32(_crossgl_pop_count({bits}))")
            result = "UInt32"
        else:
            lines.append("var lower = (UInt64(1) << UInt64(lane_id())) - 1")
            if operation == "WaveMultiPrefixCountBits":
                signature += f", mask: {_UVEC4}"
                call_args.append(self.mojo_wave_partition(operation, args[1]))
                lines.append("lower &= _crossgl_lane_mask(mask)")
            lines.append(f"return UInt32(_crossgl_pop_count({bits} & lower))")
            result = "UInt32"
        name = self.require_mojo_wave_helper(
            f"_crossgl_{_snake(operation)}", f"{signature}) -> {result}", lines
        )
        return f"{name}({', '.join(call_args)})"

    def mojo_wave_ballot_bits_call(self, operation, args):
        mask = self.mojo_wave_partition(operation, args[0])
        bits = "_crossgl_lane_mask(mask)"
        signature = f"mask: {_UVEC4}"
        call_args = [mask]
        lane = "UInt64(lane_id())"
        if operation == "WaveBallotBitCount":
            lines, result = [f"return UInt32(_crossgl_pop_count({bits}))"], "UInt32"
        elif operation == "WaveBallotInclusiveBitCount":
            lines = [
                f"return UInt32(_crossgl_pop_count({bits} & ((UInt64(2) << {lane}) - 1)))"
            ]
            result = "UInt32"
        elif operation == "WaveBallotExclusiveBitCount":
            lines = [
                f"return UInt32(_crossgl_pop_count({bits} & ((UInt64(1) << {lane}) - 1)))"
            ]
            result = "UInt32"
        elif operation == "WaveBallotFindLSB":
            lines, result = [f"return UInt32(_crossgl_ctz({bits}))"], "UInt32"
        elif operation == "WaveBallotFindMSB":
            lines, result = [f"return UInt32(63 - _crossgl_clz({bits}))"], "UInt32"
        elif operation == "WaveBallotBitExtract":
            signature += ", index: UInt32"
            call_args.append(self.mojo_wave_uint(args[1]))
            lines = [
                "if index >= 64:",
                "    return False",
                f"return (({bits} >> UInt64(index)) & 1) != 0",
            ]
            result = "Bool"
        else:  # WaveInverseBallot
            lines, result = [f"return (({bits} >> {lane}) & 1) != 0"], "Bool"
        name = self.require_mojo_wave_helper(
            f"_crossgl_{_snake(operation)}", f"{signature}) -> {result}", lines
        )
        return f"{name}({', '.join(call_args)})"

    def mojo_wave_fold_call(self, operation, operand, value, args):
        operator, scope = _FOLDS[operation]
        if operand.is_bool and operator not in _BITWISE:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation} requires a numeric operand, got bool"
            )
        if operator in _BITWISE and operand.dtype in _FLOAT_DTYPES:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation} requires an integer or bool operand"
            )
        signature = f"value: {operand.mojo_type}"
        call_args = [value]
        if scope == "cluster":
            signature += ", cluster_lanes: UInt32"
            call_args.append(self.mojo_wave_uint(args[1]))
        elif scope == "partition":
            signature += f", mask: {_UVEC4}"
            call_args.append(self.mojo_wave_partition(operation, args[1]))
        if operand.is_bool_scalar:
            lines = _bool_fold_lines(operator, scope)
        else:
            lines = _fold_lines(operator, scope, operand)
        name = self.require_mojo_wave_helper(
            f"_crossgl_{_snake(operation)}_{operand.suffix}",
            f"{signature}) -> {operand.mojo_type}",
            lines,
        )
        return f"{name}({', '.join(call_args)})"

    def mojo_wave_read_call(self, operation, operand, value, args):
        signature = f"value: {operand.mojo_type}"
        call_args = [value]
        lines = ["var active = _crossgl_active_lanes()"]
        if operation == "WaveReadLaneAt":
            signature += ", operand: UInt32"
            call_args.append(self.mojo_wave_uint(args[1]))
            source = "operand"
        elif operation == "WaveReadLaneFirst":
            source = "UInt32(_crossgl_ctz(active))"
        else:
            signature += ", operand: UInt32"
            call_args.append(self.mojo_wave_uint(args[1]))
            lines.append("var lane = UInt32(lane_id())")
            source = _RELATIVE_LANES[operation]
        if operand.is_bool_scalar:
            lines.append(
                "return Bool(_crossgl_lane_read(active, Scalar[DType.bool](value), "
                f"{source}))"
            )
        else:
            lines.append(f"return _crossgl_lane_read(active, value, {source})")
        name = self.require_mojo_wave_helper(
            f"_crossgl_{_snake(operation)}_{operand.suffix}",
            f"{signature}) -> {operand.mojo_type}",
            lines,
        )
        return f"{name}({', '.join(call_args)})"

    def mojo_wave_all_equal_call(self, operand, value):
        if operand.width != 1:
            raise MojoRefusal(
                "Mojo wave intrinsic WaveActiveAllEqual requires a scalar operand"
            )
        lines = ["var active = _crossgl_active_lanes()"]
        if operand.is_bool_scalar:
            lines += [
                "var bits = _crossgl_lane_ballot(active, value)",
                "return bits == active or bits == 0",
            ]
        else:
            lines += [
                "var first = _crossgl_lane_read(active, value, UInt32(_crossgl_ctz(active)))",
                "return _crossgl_lane_ballot(active, Bool(value == first)) == active",
            ]
        name = self.require_mojo_wave_helper(
            f"_crossgl_wave_active_all_equal_{operand.suffix}",
            f"value: {operand.mojo_type}) -> Bool",
            lines,
        )
        return f"{name}({value})"

    def mojo_wave_match_call(self, operand, value):
        if operand.is_bool_scalar:
            lines = [
                "var active = _crossgl_active_lanes()",
                "var bits = _crossgl_lane_ballot(active, value)",
                "return _crossgl_ballot_vector(bits if value else active & ~bits)",
            ]
        else:
            if operand.width == 1:
                same = "Bool(other == value)"
            else:
                same = " and ".join(
                    f"Bool(other[{index}] == value[{index}])"
                    for index in range(operand.width)
                )
            lines = [
                "var active = _crossgl_active_lanes()",
                "var matching = UInt64(0)",
                "var remaining = active",
                "while remaining != 0:",
                "    var source = _crossgl_ctz(remaining)",
                "    var other = _crossgl_lane_read(active, value, UInt32(source))",
                f"    if {same}:",
                "        matching |= UInt64(1) << source",
                "    remaining &= remaining - 1",
                "return _crossgl_ballot_vector(matching)",
            ]
        name = self.require_mojo_wave_helper(
            f"_crossgl_wave_match_{operand.suffix}",
            f"value: {operand.mojo_type}) -> {_UVEC4}",
            lines,
        )
        return f"{name}({value})"

    def mojo_subgroup_barrier_call(self, func_name):
        """``subgroupBarrier()``, a fence on each side of a warp barrier over the active lanes, or one of the
        ``subgroupMemoryBarrier*`` fences; ``None`` for any other name."""
        if func_name == "subgroupBarrier":
            body = [
                "_crossgl_fence()",
                "_crossgl_syncwarp(Int(_crossgl_active_lanes()))",
                "_crossgl_fence()",
            ]
        elif func_name in _SUBGROUP_MEMORY_BARRIERS:
            func_name, body = "subgroupMemoryBarrier", ["_crossgl_fence()"]
        else:
            return None
        name = self.require_mojo_wave_helper(f"_crossgl_{_snake(func_name)}", ")", body)
        return f"{name}()"

    def generate_cuda_match_any_call(self, args):
        """CUDA's ``__match_any_sync(mask, value)``: the lanes of the caller's CUDA warp in ``mask`` whose
        ``value`` has the caller's bit pattern, as a 32-bit lane mask."""
        if len(args) != 2:
            raise MojoRefusal(f"__match_any_sync requires 2 arguments, got {len(args)}")
        operand = self.mojo_wave_operand("__match_any_sync", args[1])
        bits = _MATCH_BITS.get(operand.dtype)
        if operand.width != 1 or bits is None:
            raise MojoRefusal(
                "__match_any_sync: CUDA matches a 32- or 64-bit integer or floating-point "
                f"scalar, got {operand.mojo_type}"
            )
        same = f"_crossgl_bitcast[{bits}](other) == _crossgl_bitcast[{bits}](value)"
        lines = [
            "var active = _crossgl_active_lanes()",
            "var warp = UInt64(lane_id()) >> 5",
            "var matching = UInt32(0)",
            "var remaining = active",
            "while remaining != 0:",
            "    var source = _crossgl_ctz(remaining)",
            "    var other = _crossgl_lane_read(active, value, UInt32(source))",
            "    var bit = UInt32(1) << UInt32(source & 31)",
            f"    if (source >> 5) == warp and (mask & bit) != 0 and {same}:",
            "        matching |= bit",
            "    remaining &= remaining - 1",
            "return matching",
        ]
        name = self.require_mojo_wave_helper(
            f"_crossgl_cuda_match_any_{operand.suffix}",
            f"mask: UInt32, value: {operand.mojo_type}) -> UInt32",
            lines,
        )
        mask = self.mojo_wave_uint(args[0])
        return f"{name}({mask}, {self.generate_expression(args[1])})"


def _snake(operation):
    words = []
    for character in operation:
        if character.isupper() and words:
            words.append("_")
        words.append(character.lower())
    return "".join(words)


def _identity(operator, operand):
    if operand.is_bool:
        return f"{operand.mojo_type}({'True' if operator == 'and' else 'False'})"
    return _IDENTITY[operator].format(type=operand.mojo_type)


def _member_setup(scope):
    lines = ["var active = _crossgl_active_lanes()"]
    if scope != "reduce":
        lines.append("var lane = UInt64(lane_id())")
    if scope == "cluster":
        lines.append("var cluster_size = UInt64(cluster_lanes)")
    if scope == "partition":
        lines.append("var partition = _crossgl_lane_mask(mask) & active")
    return lines


def _fold_lines(operator, scope, operand):
    """A fold over the member lanes, lowest first; it starts from the first member, not the identity.

    Starting from the first member keeps the result exact where the identity would change it, as ``0.0 + -0.0``
    does. Every active lane reads every active lane, so all of them execute the same shuffles.
    """
    combine = _COMBINE[operator].format(left="result", right="other")
    lines = _member_setup(scope) + [
        f"var result = {_identity(operator, operand)}",
        "var found = False",
        "var remaining = active",
        "while remaining != 0:",
        "    var source = _crossgl_ctz(remaining)",
        "    var other = _crossgl_lane_read(active, value, UInt32(source))",
    ]
    member = _MEMBERS[scope]
    if member == "True":
        lines += ["    if found:", f"        result = {combine}"]
    else:
        lines += [
            f"    if {member}:",
            "        if found:",
            f"            result = {combine}",
        ]
    indent = "    " if member == "True" else "        "
    lines += [
        f"{indent}else:",
        f"{indent}    result = other",
        f"{indent}    found = True",
        "    remaining &= remaining - 1",
        "return result",
    ]
    return lines


def _bool_fold_lines(operator, scope):
    """A logical and, or or xor of a bool over the member lanes, from a ballot of the active lanes."""
    lines = _member_setup(scope)
    if scope == "reduce":
        lines.append("var members = active")
    elif scope in {"inclusive", "exclusive"}:
        shift = "UInt64(2)" if scope == "inclusive" else "UInt64(1)"
        lines.append(f"var members = active & (({shift} << lane) - 1)")
    elif scope == "cluster":
        lines += [
            "var first = lane & ~(cluster_size - 1)",
            "var cluster = (UInt64.MAX >> (UInt64(64) - cluster_size)) << first",
            "var members = active & cluster",
        ]
    else:
        lines.append("var members = partition & ((UInt64(1) << lane) - 1)")
    lines.append("var bits = _crossgl_lane_ballot(active, value) & members")
    if operator == "and":
        lines.append("return bits == members")
    elif operator == "or":
        lines.append("return bits != 0")
    else:
        lines.append("return (_crossgl_pop_count(bits) & 1) == 1")
    return lines
