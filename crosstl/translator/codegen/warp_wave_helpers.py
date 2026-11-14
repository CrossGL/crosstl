"""Wave (subgroup) operations of HLSL, GLSL and MSL as CUDA and HIP warp-level helpers.

HLSL's ``WaveActive*`` and ``WavePrefix*`` functions, GLSL's ``subgroup*`` functions and MSL's ``simd_*`` functions
combine the values of the *active* lanes of a wave: those that execute the call (HLSL Shader Model 6.0 "Wave
Intrinsics"; ``GL_KHR_shader_subgroup``; MSL section 6.9.2). Every active lane receives the reduction, and a prefix
operation combines the active lanes with a lower index. Inactive lanes take no part, and lane 0 need not be active.

CUDA and HIP name the active lanes with ``__activemask()`` and ``__ballot(1)``; a lane reads another lane's value with
``__shfl_sync`` and ``__shfl``, which are defined only when the source lane is active (CUDA C++ Programming Guide,
"Warp Shuffle Functions"; HIP programming guide, "Warp cross-lane functions"). The helpers therefore visit the
active lanes in ascending order, the set bits of the mask, and read each lane by its index. Every active lane runs
the same iterations, so each obtains the same reduction, combined in the same order.

The CUDA warp has 32 lanes; a HIP wavefront has ``warpSize`` lanes, 32 or 64, so HIP masks are 64 bits wide and an
HLSL ballot (``uint4``) receives both halves.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CUDA_WARP",
    "HIP_WAVEFRONT",
    "WarpDialect",
    "wave_combine_expression",
    "wave_first_lane_test",
    "wave_helper_body",
    "wave_identity_expression",
]


@dataclass(frozen=True)
class WarpDialect:
    """How one language spells the warp-level primitives the helpers need."""

    mask_type: str
    active_mask: str
    one: str
    zero: str
    first_lane_function: str
    popcount_function: str

    def shuffle(self, mask: str, value: str, lane: str) -> str:
        raise NotImplementedError

    def shuffle_xor(self, mask: str, value: str, lane_mask: str) -> str:
        raise NotImplementedError

    def ballot(self, mask: str, predicate: str) -> str:
        raise NotImplementedError

    def vote(self, kind: str, mask: str, predicate: str) -> str:
        raise NotImplementedError

    def ballot_uint4(self, bits: str) -> str:
        raise NotImplementedError

    def partition_mask(self, mask: str) -> str:
        raise NotImplementedError

    def first_lane(self, mask: str) -> str:
        return f"({self.first_lane_function}({mask}) - 1)"

    def popcount(self, bits: str) -> str:
        return f"static_cast<uint>({self.popcount_function}({bits}))"


class _CudaWarp(WarpDialect):
    def shuffle(self, mask, value, lane):
        return f"__shfl_sync({mask}, {value}, static_cast<int>({lane}))"

    def shuffle_xor(self, mask, value, lane_mask):
        return f"__shfl_xor_sync({mask}, {value}, {lane_mask})"

    def ballot(self, mask, predicate):
        return f"__ballot_sync({mask}, {predicate})"

    def vote(self, kind, mask, predicate):
        return f"(__{kind}_sync({mask}, {predicate}) != 0)"

    def ballot_uint4(self, bits):
        return f"make_uint4({bits}, 0u, 0u, 0u)"

    def partition_mask(self, mask):
        return f"{mask}.x"


class _HipWavefront(WarpDialect):
    # HIP's cross-lane functions take no mask: a lane may only read lanes that are active, which the helpers ensure.
    def shuffle(self, mask, value, lane):
        return f"__shfl({value}, static_cast<int>({lane}))"

    def shuffle_xor(self, mask, value, lane_mask):
        return f"__shfl_xor({value}, {lane_mask})"

    def ballot(self, mask, predicate):
        return f"(__ballot({predicate}) & {mask})"

    def vote(self, kind, mask, predicate):
        return f"(__{kind}({predicate}) != 0)"

    def ballot_uint4(self, bits):
        return (
            f"make_uint4(static_cast<uint>({bits}), static_cast<uint>({bits} >> 32), "
            "0u, 0u)"
        )

    def partition_mask(self, mask):
        return f"((static_cast<unsigned long long>({mask}.y) << 32) | {mask}.x)"


CUDA_WARP = _CudaWarp(
    mask_type="unsigned int",
    active_mask="__activemask()",
    one="1u",
    zero="0u",
    first_lane_function="__ffs",
    popcount_function="__popc",
)
HIP_WAVEFRONT = _HipWavefront(
    mask_type="unsigned long long",
    active_mask="__ballot(1)",
    one="1ull",
    zero="0ull",
    first_lane_function="__ffsll",
    popcount_function="__popcll",
)

_REDUCTIONS = frozenset(
    {
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
        "WaveActiveMin",
        "WaveActiveMax",
    }
)
_PREFIXES = frozenset({"WavePrefixSum", "WavePrefixProduct"})
_MULTI_PREFIXES = frozenset(
    {
        "WaveMultiPrefixSum",
        "WaveMultiPrefixProduct",
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
)
_QUAD_XOR = {
    "QuadReadAcrossX": "1",
    "QuadReadAcrossY": "2",
    "QuadReadAcrossDiagonal": "3",
}


def wave_identity_expression(operation: str, result_type: str) -> str:
    """The identity element of a prefix operation, the result of a lane with no lower active lane."""
    if operation in {"WaveActiveBitAnd", "WaveMultiPrefixBitAnd"}:
        return f"static_cast<{result_type}>(~0ull)"
    if operation in {
        "WaveActiveProduct",
        "WavePrefixProduct",
        "WaveMultiPrefixProduct",
    }:
        return f"static_cast<{result_type}>(1)"
    return f"static_cast<{result_type}>(0)"


def wave_combine_expression(operation: str, left: str, right: str) -> str:
    """``left`` combined with ``right`` by the operation's binary operator."""
    if operation in {"WaveActiveSum", "WavePrefixSum", "WaveMultiPrefixSum"}:
        return f"({left} + {right})"
    if operation in {
        "WaveActiveProduct",
        "WavePrefixProduct",
        "WaveMultiPrefixProduct",
    }:
        return f"({left} * {right})"
    if operation in {"WaveActiveBitAnd", "WaveMultiPrefixBitAnd"}:
        return f"({left} & {right})"
    if operation in {"WaveActiveBitOr", "WaveMultiPrefixBitOr"}:
        return f"({left} | {right})"
    if operation in {"WaveActiveBitXor", "WaveMultiPrefixBitXor"}:
        return f"({left} ^ {right})"
    if operation == "WaveActiveMin":
        return f"(({right}) < ({left}) ? ({right}) : ({left}))"
    if operation == "WaveActiveMax":
        return f"(({right}) > ({left}) ? ({right}) : ({left}))"
    raise ValueError(f"{operation} has no combining operator")


def wave_first_lane_test(dialect: WarpDialect, lane_index: str) -> str:
    """Whether the current lane is the lowest active lane (``WaveIsFirstLane``, ``subgroupElect``)."""
    return (
        f"(static_cast<int>({lane_index}) == {dialect.first_lane(dialect.active_mask)})"
    )


def _active_lanes_loop(dialect: WarpDialect, mask: str, lane: str, body: str) -> str:
    """A loop that binds ``lane`` to each set bit of ``mask``, lowest first, and runs ``body`` for it."""
    return (
        f"    for ({dialect.mask_type} remaining = {mask}; remaining != {dialect.zero}; "
        "remaining &= remaining - 1) {\n"
        f"        int {lane} = {dialect.first_lane('remaining')};\n"
        f"{body}"
        "    }\n"
    )


def _lower_lanes_mask(dialect: WarpDialect) -> str:
    return f"(({dialect.one} << lane_index) - {dialect.one})"


def _reduction(dialect, operation, result_type):
    combine = wave_combine_expression(operation, "result", "other")
    first = dialect.shuffle("active", "value", dialect.first_lane("active"))
    loop = _active_lanes_loop(
        dialect,
        "active & (active - 1)",
        "source",
        f"        {result_type} other = {dialect.shuffle('active', 'value', 'source')};\n"
        f"        result = {combine};\n",
    )
    return (
        f"    {dialect.mask_type} active = {dialect.active_mask};\n"
        f"    {result_type} result = {first};\n"
        f"{loop}"
        "    return result;\n"
    )


def _prefix(dialect, operation, result_type, lane_index, partition=None):
    combine = wave_combine_expression(operation, "result", "other")
    member = "source < static_cast<int>(lane_index)"
    if partition is not None:
        member += f" && ((partition >> source) & {dialect.one}) != {dialect.zero}"
    loop = _active_lanes_loop(
        dialect,
        "active",
        "source",
        f"        {result_type} other = {dialect.shuffle('active', 'value', 'source')};\n"
        f"        if ({member}) {{\n"
        f"            result = {combine};\n"
        "        }\n",
    )
    partition_line = (
        f"    {dialect.mask_type} partition = {dialect.partition_mask(partition)} & active;\n"
        if partition is not None
        else ""
    )
    return (
        f"    {dialect.mask_type} active = {dialect.active_mask};\n"
        f"{partition_line}"
        f"    uint lane_index = {lane_index};\n"
        f"    {result_type} result = {wave_identity_expression(operation, result_type)};\n"
        f"{loop}"
        "    return result;\n"
    )


def _count_bits(dialect, lane_index, prefix, partition=None):
    bits = f"{dialect.ballot('active', 'predicate')}"
    if partition is not None:
        bits += " & partition"
    if prefix:
        bits += " & lower_lanes"
    lines = [f"    {dialect.mask_type} active = {dialect.active_mask};\n"]
    if partition is not None:
        lines.append(
            f"    {dialect.mask_type} partition = {dialect.partition_mask(partition)} & active;\n"
        )
    if prefix:
        lines.append(f"    uint lane_index = {lane_index};\n")
        lines.append(
            f"    {dialect.mask_type} lower_lanes = {_lower_lanes_mask(dialect)};\n"
        )
    lines.append(f"    return {dialect.popcount(bits)};\n")
    return "".join(lines)


def _match(dialect, result_type):
    loop = _active_lanes_loop(
        dialect,
        "active",
        "candidate",
        f"        {result_type} other = {dialect.shuffle('active', 'value', 'candidate')};\n"
        "        if (value == other) {\n"
        f"            matching |= {dialect.one} << candidate;\n"
        "        }\n",
    )
    return (
        f"    {dialect.mask_type} active = {dialect.active_mask};\n"
        f"    {dialect.mask_type} matching = {dialect.zero};\n"
        f"{loop}"
        f"    return {dialect.ballot_uint4('matching')};\n"
    )


def wave_helper_body(
    dialect: WarpDialect,
    operation: str,
    result_type: str,
    value_type: str,
    lane_index: str,
) -> str | None:
    """The statements of the helper that evaluates ``operation``, or ``None`` when it has none.

    ``result_type`` is the helper's return type, ``value_type`` the type of its value parameter and ``lane_index``
    the expression of the current lane's index in its warp. Parameters are named ``value``, ``predicate``,
    ``lane`` and ``mask``.
    """
    active = f"    {dialect.mask_type} active = {dialect.active_mask};\n"
    if operation in _REDUCTIONS:
        return _reduction(dialect, operation, result_type)
    if operation in _PREFIXES:
        return _prefix(dialect, operation, result_type, lane_index)
    if operation in _MULTI_PREFIXES:
        return _prefix(dialect, operation, result_type, lane_index, partition="mask")
    if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
        kind = "all" if operation == "WaveActiveAllTrue" else "any"
        return active + f"    return {dialect.vote(kind, 'active', 'predicate')};\n"
    if operation == "WaveActiveAllEqual":
        first = dialect.shuffle("active", "value", dialect.first_lane("active"))
        return (
            active
            + f"    {value_type} first = {first};\n"
            + f"    return {dialect.vote('all', 'active', 'value == first')};\n"
        )
    if operation == "WaveActiveBallot":
        return (
            active
            + f"    {dialect.mask_type} bits = {dialect.ballot('active', 'predicate')};\n"
            + f"    return {dialect.ballot_uint4('bits')};\n"
        )
    if operation == "WaveActiveCountBits":
        return _count_bits(dialect, lane_index, prefix=False)
    if operation == "WavePrefixCountBits":
        return _count_bits(dialect, lane_index, prefix=True)
    if operation == "WaveMultiPrefixCountBits":
        return _count_bits(dialect, lane_index, prefix=True, partition="mask")
    if operation == "WaveReadLaneAt":
        return active + f"    return {dialect.shuffle('active', 'value', 'lane')};\n"
    if operation == "WaveReadLaneFirst":
        first = dialect.first_lane("active")
        return active + f"    return {dialect.shuffle('active', 'value', first)};\n"
    if operation in _QUAD_XOR:
        return (
            active
            + f"    return {dialect.shuffle_xor('active', 'value', _QUAD_XOR[operation])};\n"
        )
    if operation == "QuadReadLaneAt":
        return (
            active
            + f"    uint lane_index = {lane_index};\n"
            + "    uint source_lane = (lane_index & ~3u) | (lane & 3u);\n"
            + f"    return {dialect.shuffle('active', 'value', 'source_lane')};\n"
        )
    if operation == "WaveMatch":
        return _match(dialect, value_type)
    return None
