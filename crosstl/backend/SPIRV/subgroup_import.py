"""Import the SPIR-V subgroup vote, ballot, shuffle, quad, rotate and partition instructions.

CrossGL spells subgroup operations with the GLSL names of GL_KHR_shader_subgroup (vote, ballot, shuffle,
shuffle_relative, quad), GL_KHR_shader_subgroup_rotate, GL_EXT_shader_quad_control and GL_NV_shader_subgroup_partitioned,
so each instruction maps to the GLSL function that glslang compiles to it (GLSL to SPIR-V mapping of those
extensions). The quad votes are the exception: CrossGL spells them ``QuadAll`` and ``QuadAny``, as HLSL SM 6.7 does.

- ``OpGroupNonUniformElect``, ``All``, ``Any``, ``AllEqual``, ``Ballot``, ``InverseBallot``, ``BallotBitExtract``,
  ``BallotFindLSB``, ``BallotFindMSB``, ``Shuffle``, ``ShuffleXor``, ``ShuffleUp``, ``ShuffleDown`` and
  ``QuadBroadcast`` take the operands after their execution scope as the GLSL arguments.
- ``OpGroupNonUniformBallotBitCount`` names its group operation: ``Reduce`` is ``subgroupBallotBitCount``,
  ``InclusiveScan`` and ``ExclusiveScan`` the inclusive and exclusive bit counts.
- ``OpGroupNonUniformQuadSwap`` names its direction with a constant: 0 is horizontal, 1 vertical and 2 diagonal
  (SPIR-V 1.6 section 3.52.24).
- ``OpGroupNonUniformRotateKHR`` with a cluster size is ``subgroupClusteredRotate``.
- ``OpSubgroupAllKHR``, ``OpSubgroupAnyKHR`` and ``OpSubgroupAllEqualKHR`` (SPV_KHR_subgroup_vote),
  ``OpGroupNonUniformQuadAllKHR`` and ``OpGroupNonUniformQuadAnyKHR`` (SPV_KHR_quad_control) and
  ``OpGroupNonUniformPartitionNV`` (SPV_NV_shader_subgroup_partitioned) have no execution scope operand.

Vulkan limits the execution scope of every non-uniform group instruction to ``Subgroup`` (Vulkan 1.3 appendix A,
VUID-StandaloneSpirv-None-04642), so the scope adds nothing to the call. An instruction whose group operation or
quad direction is not one of the listed values is refused with
:class:`~crosstl.backend.SPIRV.ray_tracing_import.SpirvInstructionUnsupportedError`.
"""

from __future__ import annotations

from .ray_tracing_import import SpirvInstructionUnsupportedError
from .VulkanAst import FunctionCallNode

__all__ = ["OPCODES", "spirv_subgroup_call"]

# opcode -> (CrossGL function, whether the instruction has an execution scope operand, argument count)
_DIRECT = {
    "OpGroupNonUniformElect": ("subgroupElect", True, 0),
    "OpGroupNonUniformAll": ("subgroupAll", True, 1),
    "OpGroupNonUniformAny": ("subgroupAny", True, 1),
    "OpGroupNonUniformAllEqual": ("subgroupAllEqual", True, 1),
    "OpGroupNonUniformBallot": ("subgroupBallot", True, 1),
    "OpGroupNonUniformInverseBallot": ("subgroupInverseBallot", True, 1),
    "OpGroupNonUniformBallotBitExtract": ("subgroupBallotBitExtract", True, 2),
    "OpGroupNonUniformBallotFindLSB": ("subgroupBallotFindLSB", True, 1),
    "OpGroupNonUniformBallotFindMSB": ("subgroupBallotFindMSB", True, 1),
    "OpGroupNonUniformShuffle": ("subgroupShuffle", True, 2),
    "OpGroupNonUniformShuffleXor": ("subgroupShuffleXor", True, 2),
    "OpGroupNonUniformShuffleUp": ("subgroupShuffleUp", True, 2),
    "OpGroupNonUniformShuffleDown": ("subgroupShuffleDown", True, 2),
    "OpGroupNonUniformQuadBroadcast": ("subgroupQuadBroadcast", True, 2),
    "OpSubgroupAllKHR": ("subgroupAll", False, 1),
    "OpSubgroupAnyKHR": ("subgroupAny", False, 1),
    "OpSubgroupAllEqualKHR": ("subgroupAllEqual", False, 1),
    "OpGroupNonUniformQuadAllKHR": ("QuadAll", False, 1),
    "OpGroupNonUniformQuadAnyKHR": ("QuadAny", False, 1),
    "OpGroupNonUniformPartitionNV": ("subgroupPartitionNV", False, 1),
    "OpGroupNonUniformPartitionEXT": ("subgroupPartitionNV", False, 1),
}
_BALLOT_BIT_COUNTS = {
    "Reduce": "subgroupBallotBitCount",
    "InclusiveScan": "subgroupBallotInclusiveBitCount",
    "ExclusiveScan": "subgroupBallotExclusiveBitCount",
}
_QUAD_SWAPS = {
    0: "subgroupQuadSwapHorizontal",
    1: "subgroupQuadSwapVertical",
    2: "subgroupQuadSwapDiagonal",
}
OPCODES = frozenset(
    {
        *_DIRECT,
        "OpGroupNonUniformBallotBitCount",
        "OpGroupNonUniformQuadSwap",
        "OpGroupNonUniformRotateKHR",
    }
)


def spirv_subgroup_call(opcode, operands, operand_expression, integer_constant):
    """The CrossGL call for a subgroup instruction, or ``None`` when ``opcode`` is not one of them.

    ``operands`` starts with the result type. ``operand_expression`` maps an operand to its expression and
    ``integer_constant`` an operand to its integer value or ``None``.
    """
    if opcode in _DIRECT:
        name, scoped, count = _DIRECT[opcode]
        first = 2 if scoped else 1
        arguments = operands[first : first + count]
        if len(arguments) != count:
            return None
        return FunctionCallNode(name, [operand_expression(a) for a in arguments])
    if opcode == "OpGroupNonUniformBallotBitCount" and len(operands) >= 4:
        name = _BALLOT_BIT_COUNTS.get(operands[2])
        if name is None:
            raise SpirvInstructionUnsupportedError(
                opcode,
                f"uses group operation {operands[2]}, which has no CrossGL subgroup function",
            )
        return FunctionCallNode(name, [operand_expression(operands[3])])
    if opcode == "OpGroupNonUniformQuadSwap" and len(operands) >= 4:
        name = _QUAD_SWAPS.get(integer_constant(operands[3]))
        if name is None:
            raise SpirvInstructionUnsupportedError(
                opcode,
                f"swaps in direction {operands[3]}, which has no CrossGL subgroup function",
            )
        return FunctionCallNode(name, [operand_expression(operands[2])])
    if opcode == "OpGroupNonUniformRotateKHR" and len(operands) >= 4:
        arguments = [operand_expression(operand) for operand in operands[2:5]]
        name = "subgroupClusteredRotate" if len(arguments) == 3 else "subgroupRotate"
        return FunctionCallNode(name, arguments)
    return None
