"""SPIR-V input imports the subgroup vote, ballot, shuffle, quad, rotate and partition instructions.

Each instruction becomes the GLSL subgroup function glslang compiles to it, and the quad votes become CrossGL's
``QuadAll`` and ``QuadAny`` (``subgroup_import.py``). Before, the importer dropped them, and the translation failed
later on an unknown ``value_N``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.ray_tracing_import import SpirvInstructionUnsupportedError

STORE = {
    "bool": "%s = OpSelect %uint %r %uint_1 %uint_0",
    "uint": "%s = OpCopyObject %uint %r",
    "uvec4": "%s = OpCompositeExtract %uint %r 0",
    "float": "%s = OpConvertFToU %uint %r",
}


def module(instruction, result):
    """A compute module with a predicate ``%p``, a float ``%x`` and a ballot ``%b`` that stores ``%r``."""
    return f"""OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformVote
OpCapability GroupNonUniformBallot
OpCapability GroupNonUniformShuffle
OpCapability GroupNonUniformShuffleRelative
OpCapability GroupNonUniformQuad
OpCapability GroupNonUniformRotateKHR
OpCapability GroupNonUniformPartitionedNV
OpCapability QuadControlKHR
OpCapability SubgroupVoteKHR
OpExtension "SPV_KHR_subgroup_rotate"
OpExtension "SPV_NV_shader_subgroup_partitioned"
OpExtension "SPV_KHR_quad_control"
OpExtension "SPV_KHR_subgroup_vote"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %floats ArrayStride 4
OpDecorate %In Block
OpMemberDecorate %In 0 Offset 0
OpDecorate %inp Binding 0
OpDecorate %inp DescriptorSet 0
OpDecorate %uints ArrayStride 4
OpDecorate %Out Block
OpMemberDecorate %Out 0 Offset 0
OpDecorate %outp Binding 1
OpDecorate %outp DescriptorSet 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%uvec4 = OpTypeVector %uint 4
%v3uint = OpTypeVector %uint 3
%ptr_in_v3uint = OpTypePointer Input %v3uint
%ptr_in_uint = OpTypePointer Input %uint
%gid = OpVariable %ptr_in_v3uint Input
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%int_0 = OpConstant %int 0
%float_0 = OpConstant %float 0
%floats = OpTypeRuntimeArray %float
%In = OpTypeStruct %floats
%ptr_sb_In = OpTypePointer StorageBuffer %In
%inp = OpVariable %ptr_sb_In StorageBuffer
%ptr_sb_float = OpTypePointer StorageBuffer %float
%uints = OpTypeRuntimeArray %uint
%Out = OpTypeStruct %uints
%ptr_sb_Out = OpTypePointer StorageBuffer %Out
%outp = OpVariable %ptr_sb_Out StorageBuffer
%ptr_sb_uint = OpTypePointer StorageBuffer %uint
%main = OpFunction %void None %fn
%entry = OpLabel
%gx = OpAccessChain %ptr_in_uint %gid %uint_0
%i = OpLoad %uint %gx
%xp = OpAccessChain %ptr_sb_float %inp %int_0 %i
%x = OpLoad %float %xp
%p = OpFOrdGreaterThan %bool %x %float_0
%b = OpGroupNonUniformBallot %uvec4 %uint_3 %p
%r = {instruction}
{STORE[result]}
%op = OpAccessChain %ptr_sb_uint %outp %int_0 %i
OpStore %op %s
OpReturn
OpFunctionEnd
"""


CASES = [
    ("OpGroupNonUniformElect %bool %uint_3", "bool", "subgroupElect()"),
    ("OpGroupNonUniformAll %bool %uint_3 %p", "bool", "subgroupAll("),
    ("OpGroupNonUniformAny %bool %uint_3 %p", "bool", "subgroupAny("),
    ("OpGroupNonUniformAllEqual %bool %uint_3 %x", "bool", "subgroupAllEqual("),
    ("OpSubgroupAllKHR %bool %p", "bool", "subgroupAll("),
    ("OpSubgroupAnyKHR %bool %p", "bool", "subgroupAny("),
    ("OpSubgroupAllEqualKHR %bool %x", "bool", "subgroupAllEqual("),
    ("OpGroupNonUniformBallot %uvec4 %uint_3 %p", "uvec4", "subgroupBallot("),
    (
        "OpGroupNonUniformInverseBallot %bool %uint_3 %b",
        "bool",
        "subgroupInverseBallot(",
    ),
    (
        "OpGroupNonUniformBallotBitExtract %bool %uint_3 %b %uint_1",
        "bool",
        "subgroupBallotBitExtract(",
    ),
    (
        "OpGroupNonUniformBallotBitCount %uint %uint_3 Reduce %b",
        "uint",
        "subgroupBallotBitCount(",
    ),
    (
        "OpGroupNonUniformBallotBitCount %uint %uint_3 InclusiveScan %b",
        "uint",
        "subgroupBallotInclusiveBitCount(",
    ),
    (
        "OpGroupNonUniformBallotBitCount %uint %uint_3 ExclusiveScan %b",
        "uint",
        "subgroupBallotExclusiveBitCount(",
    ),
    (
        "OpGroupNonUniformBallotFindLSB %uint %uint_3 %b",
        "uint",
        "subgroupBallotFindLSB(",
    ),
    (
        "OpGroupNonUniformBallotFindMSB %uint %uint_3 %b",
        "uint",
        "subgroupBallotFindMSB(",
    ),
    ("OpGroupNonUniformShuffle %float %uint_3 %x %uint_1", "float", "subgroupShuffle("),
    (
        "OpGroupNonUniformShuffleXor %float %uint_3 %x %uint_1",
        "float",
        "subgroupShuffleXor(",
    ),
    (
        "OpGroupNonUniformShuffleUp %float %uint_3 %x %uint_1",
        "float",
        "subgroupShuffleUp(",
    ),
    (
        "OpGroupNonUniformShuffleDown %float %uint_3 %x %uint_1",
        "float",
        "subgroupShuffleDown(",
    ),
    (
        "OpGroupNonUniformQuadBroadcast %float %uint_3 %x %uint_1",
        "float",
        "subgroupQuadBroadcast(",
    ),
    (
        "OpGroupNonUniformQuadSwap %float %uint_3 %x %uint_0",
        "float",
        "subgroupQuadSwapHorizontal(",
    ),
    (
        "OpGroupNonUniformQuadSwap %float %uint_3 %x %uint_1",
        "float",
        "subgroupQuadSwapVertical(",
    ),
    (
        "OpGroupNonUniformQuadSwap %float %uint_3 %x %uint_2",
        "float",
        "subgroupQuadSwapDiagonal(",
    ),
    ("OpGroupNonUniformQuadAllKHR %bool %p", "bool", "QuadAll("),
    ("OpGroupNonUniformQuadAnyKHR %bool %p", "bool", "QuadAny("),
    ("OpGroupNonUniformPartitionNV %uvec4 %x", "uvec4", "subgroupPartitionNV("),
]
# GL_KHR_shader_subgroup_rotate has no CrossGL operation yet, so these are checked on import only.
ROTATES = [
    ("OpGroupNonUniformRotateKHR %float %uint_3 %x %uint_1", "subgroupRotate("),
    (
        "OpGroupNonUniformRotateKHR %float %uint_3 %x %uint_1 %uint_4",
        "subgroupClusteredRotate(",
    ),
]


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def validate(tmp_path, code, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    source, binary = tmp_path / f"{name}.spvasm", tmp_path / f"{name}.spv"
    source.write_text(code, encoding="utf-8")
    run_tool(["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)])
    run_tool(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def import_crossgl(tmp_path, code):
    path = tmp_path / "subgroup.spvasm"
    path.write_text(code, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl")


@pytest.mark.parametrize("instruction, result, call", CASES)
def test_subgroup_instruction_imports_and_round_trips(
    tmp_path, instruction, result, call
):
    source = module(instruction, result)
    validate(tmp_path, source, "probe")
    crossgl = import_crossgl(tmp_path, source)
    assert call in crossgl, crossgl
    assert not re.search(r"\bvalue_\d+\b", crossgl), crossgl
    path = tmp_path / "subgroup.cgl"
    path.write_text(crossgl, encoding="utf-8")
    translated = crosstl.translate(str(path), backend="vulkan")
    # SPV_KHR_subgroup_vote's votes come back in their GroupNonUniform form, and SPIRV-Headers now spells
    # OpGroupNonUniformPartitionNV as its alias OpGroupNonUniformPartitionEXT.
    opcode = instruction.split()[0].replace("OpSubgroup", "OpGroupNonUniform")
    opcode = (
        opcode.replace("KHR", "") if "Subgroup" in instruction.split()[0] else opcode
    )
    opcode = opcode.replace("PartitionNV", "Partition")
    assert opcode in translated, translated
    validate(tmp_path, translated, "round")


@pytest.mark.parametrize("instruction, call", ROTATES)
def test_rotate_imports_as_the_glsl_function(tmp_path, instruction, call):
    crossgl = import_crossgl(tmp_path, module(instruction, "float"))
    assert call in crossgl, crossgl


@pytest.mark.parametrize(
    "instruction, result, reason",
    [
        (
            "OpGroupNonUniformQuadSwap %float %uint_3 %x %uint_3",
            "float",
            "swaps in direction",
        ),
        (
            "OpGroupNonUniformBallotBitCount %uint %uint_3 ClusteredReduce %b %uint_4",
            "uint",
            "uses group operation ClusteredReduce",
        ),
    ],
)
def test_unknown_operation_is_refused(tmp_path, instruction, result, reason):
    with pytest.raises(SpirvInstructionUnsupportedError, match=reason):
        import_crossgl(tmp_path, module(instruction, result))
