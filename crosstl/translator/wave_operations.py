"""Canonical CrossGL subgroup operations beyond the Shader Model 6.0 wave set.

CrossGL spells subgroup operations with the HLSL wave names (``WaveActiveSum``, ``WavePrefixSum``, ...). Vulkan
GLSL and SPIR-V express more than Shader Model 6.0 does: scans with min, max and bitwise operators, reductions over
clusters of lanes, and arithmetic on ballot masks. These are the CrossGL operations for them. Their semantics follow
GL_KHR_shader_subgroup and the SPIR-V ``OpGroupNonUniform*`` instructions over the active lanes of the subgroup:

* ``WavePrefix{Inclusive,Exclusive}{Min,Max,BitAnd,BitOr,BitXor}(value)``: scans over the active lanes with a
  lower (inclusive: lower or equal) lane index. An exclusive scan of the first active lane yields the identity of
  the operator.
* ``WaveClustered{Sum,Product,Min,Max,BitAnd,BitOr,BitXor}(value, clusterSize)``: the reduction over the active
  lanes of the same aligned cluster. ``clusterSize`` is a constant power of two.
* ``WaveBallot{BitCount,InclusiveBitCount,ExclusiveBitCount,FindLSB,FindMSB}(mask)``,
  ``WaveBallotBitExtract(mask, lane)`` and ``WaveInverseBallot(mask)``: arithmetic on a ``uint4`` ballot whose bit
  ``i`` stands for lane ``i``. Only the bits below the subgroup size take part.

``WaveActiveBitAnd``, ``WaveActiveBitOr`` and ``WaveActiveBitXor`` (and their scan and cluster forms) also accept
``bool`` operands, where they are the logical operators, as ``subgroupAnd(bool)`` is in GLSL.

Partitioned operations take a ``uint4`` partition mask, as ``WaveMatch`` returns and Shader Model 6.5's
``WaveMultiPrefix*`` functions take, and combine the active lanes whose bit is set in it. Every lane of a partition
passes the same mask, and the mask contains the calling lane (GL_NV_shader_subgroup_partitioned; SPIR-V
``PartitionedReduceNV`` and the partitioned scan group operations). The names are Slang's:

* ``WaveMulti{Sum,Product,Min,Max,BitAnd,BitOr,BitXor}(value, mask)``: the reduction over the partition.
* ``WaveMultiPrefixInclusive{Sum,Product,Min,Max,BitAnd,BitOr,BitXor}(value, mask)`` and
  ``WaveMultiPrefixExclusive{Min,Max}(value, mask)``: scans over the lanes of the partition with a lower or equal
  (exclusive: lower) lane index. Shader Model 6.5's ``WaveMultiPrefix{Sum,Product,BitAnd,BitOr,BitXor}`` are the
  exclusive scans of the other operators.

:mod:`crosstl.translator.semantics.partitioned_waves` lowers the partitioned operations to loops over the lanes of
the subgroup for every target that does not register them as native.

Rotations read the value of another lane at a fixed distance, wrapping around (GL_KHR_shader_subgroup_rotate; SPIR-V
``OpGroupNonUniformRotateKHR``; MSL ``simd_shuffle_rotate_down``). ``delta`` is a dynamically uniform ``uint``:

* ``WaveRotate(value, delta)``: the value of lane ``(WaveGetLaneIndex() + delta) & (WaveGetLaneCount() - 1)``.
* ``WaveClusteredRotate(value, delta, clusterSize)``: the rotation within the aligned cluster of the calling lane,
  the value of lane ``((WaveGetLaneIndex() + delta) & (clusterSize - 1)) + (WaveGetLaneIndex() & ~(clusterSize -
  1))``. ``clusterSize`` is a constant power of two no larger than the subgroup.

The result of reading an inactive lane is undefined. :mod:`crosstl.translator.semantics.subgroup_rotations` lowers the
rotations to ``WaveReadLaneAt`` for every target that does not register them as native.
"""

from __future__ import annotations

# Scan operations: name -> (operator, inclusive).
WAVE_EXTENDED_PREFIX_OPERATIONS = {
    "WavePrefixExclusiveMin": ("min", False),
    "WavePrefixInclusiveMin": ("min", True),
    "WavePrefixExclusiveMax": ("max", False),
    "WavePrefixInclusiveMax": ("max", True),
    "WavePrefixExclusiveBitAnd": ("and", False),
    "WavePrefixInclusiveBitAnd": ("and", True),
    "WavePrefixExclusiveBitOr": ("or", False),
    "WavePrefixInclusiveBitOr": ("or", True),
    "WavePrefixExclusiveBitXor": ("xor", False),
    "WavePrefixInclusiveBitXor": ("xor", True),
}

# Clustered reductions: name -> operator.
WAVE_CLUSTERED_OPERATIONS = {
    "WaveClusteredSum": "sum",
    "WaveClusteredProduct": "product",
    "WaveClusteredMin": "min",
    "WaveClusteredMax": "max",
    "WaveClusteredBitAnd": "and",
    "WaveClusteredBitOr": "or",
    "WaveClusteredBitXor": "xor",
}

# Ballot arithmetic: name -> argument count.
WAVE_BALLOT_BIT_OPERATIONS = {
    "WaveBallotBitCount": 1,
    "WaveBallotInclusiveBitCount": 1,
    "WaveBallotExclusiveBitCount": 1,
    "WaveBallotFindLSB": 1,
    "WaveBallotFindMSB": 1,
    "WaveBallotBitExtract": 2,
    "WaveInverseBallot": 1,
}

WAVE_BALLOT_BOOL_RESULT_OPERATIONS = frozenset(
    {"WaveBallotBitExtract", "WaveInverseBallot"}
)
WAVE_BALLOT_UINT_RESULT_OPERATIONS = frozenset(
    set(WAVE_BALLOT_BIT_OPERATIONS) - WAVE_BALLOT_BOOL_RESULT_OPERATIONS
)

# Every extended operation with its argument count.
WAVE_EXTENDED_ARITIES = {
    **{name: 1 for name in WAVE_EXTENDED_PREFIX_OPERATIONS},
    **{name: 2 for name in WAVE_CLUSTERED_OPERATIONS},
    **WAVE_BALLOT_BIT_OPERATIONS,
}

WAVE_EXTENDED_OPERATIONS = frozenset(WAVE_EXTENDED_ARITIES)

# Operations whose result has the type of their first argument.
WAVE_EXTENDED_VALUE_OPERATIONS = frozenset(
    set(WAVE_EXTENDED_PREFIX_OPERATIONS) | set(WAVE_CLUSTERED_OPERATIONS)
)

WAVE_BITWISE_OPERATORS = frozenset({"and", "or", "xor"})

# Partitioned reductions and scans that take a ``uint4`` partition mask: name -> (operator, scope). The scope is
# ``"reduce"``, ``"inclusive"`` or ``"exclusive"``. Shader Model 6.5's exclusive ``WaveMultiPrefix*`` scans of sum,
# product and the bitwise operators are generator operations and are not listed here.
WAVE_PARTITIONED_OPERATIONS = {
    "WaveMultiSum": ("sum", "reduce"),
    "WaveMultiProduct": ("product", "reduce"),
    "WaveMultiMin": ("min", "reduce"),
    "WaveMultiMax": ("max", "reduce"),
    "WaveMultiBitAnd": ("and", "reduce"),
    "WaveMultiBitOr": ("or", "reduce"),
    "WaveMultiBitXor": ("xor", "reduce"),
    "WaveMultiPrefixInclusiveSum": ("sum", "inclusive"),
    "WaveMultiPrefixInclusiveProduct": ("product", "inclusive"),
    "WaveMultiPrefixInclusiveMin": ("min", "inclusive"),
    "WaveMultiPrefixInclusiveMax": ("max", "inclusive"),
    "WaveMultiPrefixInclusiveBitAnd": ("and", "inclusive"),
    "WaveMultiPrefixInclusiveBitOr": ("or", "inclusive"),
    "WaveMultiPrefixInclusiveBitXor": ("xor", "inclusive"),
    "WaveMultiPrefixExclusiveMin": ("min", "exclusive"),
    "WaveMultiPrefixExclusiveMax": ("max", "exclusive"),
}

# Rotations: name -> argument count.
WAVE_ROTATE_OPERATIONS = {
    "WaveRotate": 2,
    "WaveClusteredRotate": 3,
}

# GL_KHR_shader_subgroup_{basic,vote,arithmetic,ballot,shuffle,shuffle_relative,clustered,quad,rotate},
# GL_EXT_shader_quad_control, GL_NV_shader_subgroup_partitioned and GL_ARB_shader_group_vote functions that have a
# CrossGL wave operation.
GLSL_SUBGROUP_FUNCTIONS = {
    "subgroupElect": "WaveIsFirstLane",
    "subgroupAll": "WaveActiveAllTrue",
    "subgroupAny": "WaveActiveAnyTrue",
    "subgroupAllEqual": "WaveActiveAllEqual",
    "subgroupAdd": "WaveActiveSum",
    "subgroupMul": "WaveActiveProduct",
    "subgroupMin": "WaveActiveMin",
    "subgroupMax": "WaveActiveMax",
    "subgroupAnd": "WaveActiveBitAnd",
    "subgroupOr": "WaveActiveBitOr",
    "subgroupXor": "WaveActiveBitXor",
    "subgroupInclusiveAdd": "WavePrefixInclusiveSum",
    "subgroupExclusiveAdd": "WavePrefixSum",
    "subgroupInclusiveMul": "WavePrefixInclusiveProduct",
    "subgroupExclusiveMul": "WavePrefixProduct",
    "subgroupInclusiveMin": "WavePrefixInclusiveMin",
    "subgroupExclusiveMin": "WavePrefixExclusiveMin",
    "subgroupInclusiveMax": "WavePrefixInclusiveMax",
    "subgroupExclusiveMax": "WavePrefixExclusiveMax",
    "subgroupInclusiveAnd": "WavePrefixInclusiveBitAnd",
    "subgroupExclusiveAnd": "WavePrefixExclusiveBitAnd",
    "subgroupInclusiveOr": "WavePrefixInclusiveBitOr",
    "subgroupExclusiveOr": "WavePrefixExclusiveBitOr",
    "subgroupInclusiveXor": "WavePrefixInclusiveBitXor",
    "subgroupExclusiveXor": "WavePrefixExclusiveBitXor",
    "subgroupClusteredAdd": "WaveClusteredSum",
    "subgroupClusteredMul": "WaveClusteredProduct",
    "subgroupClusteredMin": "WaveClusteredMin",
    "subgroupClusteredMax": "WaveClusteredMax",
    "subgroupClusteredAnd": "WaveClusteredBitAnd",
    "subgroupClusteredOr": "WaveClusteredBitOr",
    "subgroupClusteredXor": "WaveClusteredBitXor",
    "subgroupBallot": "WaveActiveBallot",
    "subgroupInverseBallot": "WaveInverseBallot",
    "subgroupBallotBitExtract": "WaveBallotBitExtract",
    "subgroupBallotBitCount": "WaveBallotBitCount",
    "subgroupBallotInclusiveBitCount": "WaveBallotInclusiveBitCount",
    "subgroupBallotExclusiveBitCount": "WaveBallotExclusiveBitCount",
    "subgroupBallotFindLSB": "WaveBallotFindLSB",
    "subgroupBallotFindMSB": "WaveBallotFindMSB",
    "subgroupBroadcastFirst": "WaveReadLaneFirst",
    # subgroupBroadcast takes a constant lane, which is a special case of an arbitrary lane read.
    "subgroupBroadcast": "WaveReadLaneAt",
    "subgroupShuffle": "WaveReadLaneAt",
    "subgroupShuffleXor": "WaveShuffleXor",
    "subgroupShuffleUp": "WaveShuffleUp",
    "subgroupShuffleDown": "WaveShuffleDown",
    "subgroupQuadBroadcast": "QuadReadLaneAt",
    "subgroupQuadSwapHorizontal": "QuadReadAcrossX",
    "subgroupQuadSwapVertical": "QuadReadAcrossY",
    "subgroupQuadSwapDiagonal": "QuadReadAcrossDiagonal",
    # GL_EXT_shader_quad_control; SPIR-V OpGroupNonUniformQuadAllKHR and OpGroupNonUniformQuadAnyKHR.
    "subgroupQuadAll": "QuadAll",
    "subgroupQuadAny": "QuadAny",
    # GL_KHR_shader_subgroup_rotate.
    "subgroupRotate": "WaveRotate",
    "subgroupClusteredRotate": "WaveClusteredRotate",
    "subgroupPartitionNV": "WaveMatch",
    "subgroupPartitionedExclusiveAddNV": "WaveMultiPrefixSum",
    "subgroupPartitionedExclusiveMulNV": "WaveMultiPrefixProduct",
    "subgroupPartitionedExclusiveAndNV": "WaveMultiPrefixBitAnd",
    "subgroupPartitionedExclusiveOrNV": "WaveMultiPrefixBitOr",
    "subgroupPartitionedExclusiveXorNV": "WaveMultiPrefixBitXor",
    "subgroupPartitionedExclusiveMinNV": "WaveMultiPrefixExclusiveMin",
    "subgroupPartitionedExclusiveMaxNV": "WaveMultiPrefixExclusiveMax",
    "subgroupPartitionedAddNV": "WaveMultiSum",
    "subgroupPartitionedMulNV": "WaveMultiProduct",
    "subgroupPartitionedMinNV": "WaveMultiMin",
    "subgroupPartitionedMaxNV": "WaveMultiMax",
    "subgroupPartitionedAndNV": "WaveMultiBitAnd",
    "subgroupPartitionedOrNV": "WaveMultiBitOr",
    "subgroupPartitionedXorNV": "WaveMultiBitXor",
    "subgroupPartitionedInclusiveAddNV": "WaveMultiPrefixInclusiveSum",
    "subgroupPartitionedInclusiveMulNV": "WaveMultiPrefixInclusiveProduct",
    "subgroupPartitionedInclusiveMinNV": "WaveMultiPrefixInclusiveMin",
    "subgroupPartitionedInclusiveMaxNV": "WaveMultiPrefixInclusiveMax",
    "subgroupPartitionedInclusiveAndNV": "WaveMultiPrefixInclusiveBitAnd",
    "subgroupPartitionedInclusiveOrNV": "WaveMultiPrefixInclusiveBitOr",
    "subgroupPartitionedInclusiveXorNV": "WaveMultiPrefixInclusiveBitXor",
    # GLSL 4.60 (8.18, "Shader Invocation Group Functions") and GL_ARB_shader_group_vote. The invocation group is
    # implementation-defined; SPIR-V gives it subgroup scope (SPV_KHR_subgroup_vote OpSubgroupAllKHR and friends).
    "allInvocations": "WaveActiveAllTrue",
    "anyInvocation": "WaveActiveAnyTrue",
    "allInvocationsEqual": "WaveActiveAllEqual",
    "allInvocationsARB": "WaveActiveAllTrue",
    "anyInvocationARB": "WaveActiveAnyTrue",
    "allInvocationsEqualARB": "WaveActiveAllEqual",
}

# GLSL lane-mask builtins and the lanes each one selects, relative to the calling lane.
SUBGROUP_LANE_MASK_BUILTINS = {
    "gl_SubgroupEqMask": "==",
    "gl_SubgroupGeMask": ">=",
    "gl_SubgroupGtMask": ">",
    "gl_SubgroupLeMask": "<=",
    "gl_SubgroupLtMask": "<",
}


def wave_extended_result_type(operation, argument_type):
    """The CrossGL result type of an extended operation, or ``None`` when unknown."""
    if operation in WAVE_BALLOT_BOOL_RESULT_OPERATIONS:
        return "bool"
    if operation in WAVE_BALLOT_UINT_RESULT_OPERATIONS:
        return "uint"
    if operation in WAVE_EXTENDED_VALUE_OPERATIONS:
        return argument_type
    return None


def wave_operation_operator(operation):
    """The scan or cluster operator of ``operation`` and its scope, or ``None``.

    The scope is ``"inclusive"``, ``"exclusive"`` or ``"cluster"``.
    """
    prefix = WAVE_EXTENDED_PREFIX_OPERATIONS.get(operation)
    if prefix is not None:
        operator, inclusive = prefix
        return operator, "inclusive" if inclusive else "exclusive"
    operator = WAVE_CLUSTERED_OPERATIONS.get(operation)
    if operator is not None:
        return operator, "cluster"
    return None


def cluster_size_value(literal):
    """The cluster size a constant argument names, or ``None`` when it is not a power of two."""
    if isinstance(literal, bool) or not isinstance(literal, int):
        return None
    if literal < 1 or literal & (literal - 1):
        return None
    return literal
