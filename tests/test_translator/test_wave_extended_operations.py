"""GL_KHR_shader_subgroup operations through CrossGL: import, native lowering and fail-closed refusals.

GLSL subgroup built-ins import as the canonical CrossGL wave operations (``crosstl.translator.wave_operations``).
GLSL and SPIR-V lower them to the matching built-in or ``OpGroupNonUniform*`` instruction; HLSL and Metal build the
scans, clustered reductions and ballot arithmetic Shader Model 6 and MSL lack from wave reads over the active lanes.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    WaveOpNode,
    iter_ast_child_nodes,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.wave_operations import WAVE_EXTENDED_OPERATIONS

GLSL_SUBGROUPS = """#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_KHR_shader_subgroup_clustered : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Ints { uint u[]; };
layout(std430, binding = 1) buffer Floats { float f[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    float x = f[i];
    uint k = u[i];
    float a0 = subgroupInclusiveMin(x) + subgroupExclusiveMax(x) + subgroupAdd(x);
    uint a1 = subgroupInclusiveAnd(k) ^ subgroupExclusiveOr(k) ^ subgroupInclusiveXor(k);
    float a2 = subgroupClusteredAdd(x, 4u) + subgroupClusteredMax(x, 8u);
    uint a3 = subgroupClusteredOr(k, 2u);
    uvec4 bal = subgroupBallot(x > 0.0);
    uint a4 = subgroupBallotBitCount(bal) + subgroupBallotInclusiveBitCount(bal)
        + subgroupBallotExclusiveBitCount(bal);
    uint a5 = subgroupBallotFindLSB(bal) + subgroupBallotFindMSB(bal);
    bool a6 = subgroupBallotBitExtract(bal, 3u) && subgroupInverseBallot(bal);
    bool a7 = subgroupAnd(x > 1.0) || subgroupOr(x > 2.0) || subgroupXor(x > 3.0);
    uint a8 = subgroupBallotBitCount(subgroupBallot(k > 2u))
        + subgroupBallotExclusiveBitCount(subgroupBallot(k > 3u));
    uvec4 m = gl_SubgroupEqMask | gl_SubgroupGeMask | gl_SubgroupGtMask
        | gl_SubgroupLeMask | gl_SubgroupLtMask;
    float a9 = subgroupShuffleUp(x, 1u) + subgroupShuffleDown(x, 2u)
        + subgroupShuffle(x, k) + subgroupBroadcast(x, 1u) + subgroupQuadSwapDiagonal(x);
    bool a10 = subgroupElect() && subgroupAllEqual(k);
    u[i] = a1 + a3 + a4 + a5 + a8 + m.x + (a6 ? 1u : 0u) + (a7 ? 2u : 0u) + (a10 ? 4u : 0u);
    f[i] = a0 + a2 + a9;
}
"""

# The instruction glslang emits for each call above, with its group operation.
EXPECTED_SPIRV = sorted(
    [
        "OpGroupNonUniformFMin InclusiveScan",
        "OpGroupNonUniformFMax ExclusiveScan",
        "OpGroupNonUniformFAdd Reduce",
        "OpGroupNonUniformBitwiseAnd InclusiveScan",
        "OpGroupNonUniformBitwiseOr ExclusiveScan",
        "OpGroupNonUniformBitwiseXor InclusiveScan",
        "OpGroupNonUniformFAdd ClusteredReduce",
        "OpGroupNonUniformFMax ClusteredReduce",
        "OpGroupNonUniformBitwiseOr ClusteredReduce",
        "OpGroupNonUniformBallot",
        "OpGroupNonUniformBallotBitCount Reduce",
        "OpGroupNonUniformBallotBitCount InclusiveScan",
        "OpGroupNonUniformBallotBitCount ExclusiveScan",
        "OpGroupNonUniformBallotFindLSB",
        "OpGroupNonUniformBallotFindMSB",
        "OpGroupNonUniformBallotBitExtract",
        "OpGroupNonUniformInverseBallot",
        "OpGroupNonUniformLogicalAnd Reduce",
        "OpGroupNonUniformLogicalOr Reduce",
        "OpGroupNonUniformLogicalXor Reduce",
        "OpGroupNonUniformBallot",
        "OpGroupNonUniformBallotBitCount Reduce",
        "OpGroupNonUniformBallot",
        "OpGroupNonUniformBallotBitCount ExclusiveScan",
        "OpGroupNonUniformShuffleUp",
        "OpGroupNonUniformShuffleDown",
        "OpGroupNonUniformShuffle",
        "OpGroupNonUniformShuffle",
        "OpGroupNonUniformQuadSwap",
        "OpGroupNonUniformElect",
        "OpGroupNonUniformAllEqual",
    ]
)

GROUP_OPERATIONS = ("Reduce", "InclusiveScan", "ExclusiveScan", "ClusteredReduce")


def _translate(tmp_path, source, backend, name="shader.comp"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _crossgl(body, declarations=""):
    return f"""shader Waves {{
        RWStructuredBuffer<uint> results @register(u0);
        RWStructuredBuffer<float> values @register(u1);
        {declarations}
        compute {{
            @numthreads(64, 1, 1)
            void main(uint i @gl_GlobalInvocationID) {{
                {body}
            }}
        }}
    }}"""


def _spirv_operations(assembly):
    operations = []
    for line in assembly.splitlines():
        match = re.search(r"(OpGroupNonUniform\w+)", line)
        if match is None:
            continue
        words = line.split()
        group = next((word for word in words if word in GROUP_OPERATIONS), None)
        operations.append(f"{match.group(1)} {group}" if group else match.group(1))
    return sorted(operations)


def test_glsl_subgroup_builtins_import_as_canonical_wave_operations(tmp_path):
    crossgl = _translate(tmp_path, GLSL_SUBGROUPS, "cgl")

    assert re.findall(r"\bsubgroup[A-Z]\w*\s*\(", crossgl) == []
    for operation in (
        "WavePrefixInclusiveMin",
        "WavePrefixExclusiveMax",
        "WaveActiveSum",
        "WavePrefixInclusiveBitAnd",
        "WavePrefixExclusiveBitOr",
        "WavePrefixInclusiveBitXor",
        "WaveClusteredSum",
        "WaveClusteredMax",
        "WaveClusteredBitOr",
        "WaveActiveBallot",
        "WaveBallotBitCount",
        "WaveBallotInclusiveBitCount",
        "WaveBallotExclusiveBitCount",
        "WaveBallotFindLSB",
        "WaveBallotFindMSB",
        "WaveBallotBitExtract",
        "WaveInverseBallot",
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
        "WaveShuffleUp",
        "WaveShuffleDown",
        "WaveReadLaneAt",
        "QuadReadAcrossDiagonal",
        "WaveIsFirstLane",
        "WaveActiveAllEqual",
    ):
        assert f"{operation}(" in crossgl, operation
    # Counting a fresh ballot is the predicate count.
    assert "WaveActiveCountBits((k > 2u))" in crossgl
    assert "WavePrefixCountBits((k > 3u))" in crossgl


def test_extended_operations_parse_as_wave_operations():
    ast = parse(_crossgl("results[i] = WaveClusteredBitXor(results[i], 2u);"))
    operations = {
        node.operation
        for node in _walk(ast)
        if isinstance(node, WaveOpNode) and node.operation in WAVE_EXTENDED_OPERATIONS
    }
    assert operations == {"WaveClusteredBitXor"}


def _walk(node):
    yield node
    for name, value in vars(node).items():
        if name not in AST_CHILD_FIELD_EXCLUSIONS:
            for child in iter_ast_child_nodes(value):
                yield from _walk(child)


def test_user_function_shadows_subgroup_builtin(tmp_path):
    source = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Floats { float f[]; };
float subgroupAdd(float value) { return value * 2.0; }
void main() { f[0] = subgroupAdd(f[0]); }
"""
    crossgl = _translate(tmp_path, source, "cgl")

    assert "subgroupAdd(" in crossgl
    assert "WaveActiveSum" not in crossgl


def test_glsl_round_trip_keeps_subgroup_builtins(tmp_path):
    glsl = _translate(tmp_path, GLSL_SUBGROUPS, "opengl")

    for spelling in (
        "subgroupInclusiveMin(",
        "subgroupClusteredAdd(x, 4u)",
        "subgroupBallotInclusiveBitCount(",
        "subgroupBallotFindMSB(",
        "subgroupInverseBallot(",
        "subgroupXor(",
        "gl_SubgroupLtMask",
    ):
        assert spelling in glsl, spelling
    assert "#extension GL_KHR_shader_subgroup_clustered : require" in glsl
    glslang = shutil.which("glslangValidator")
    if glslang:
        path = tmp_path / "round_trip.comp"
        path.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [glslang, "--target-env", "vulkan1.1", "-S", "comp", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_lowers_each_subgroup_builtin_as_glslang_does(tmp_path):
    assembly = _translate(tmp_path, GLSL_SUBGROUPS, "vulkan")

    assert _spirv_operations(assembly) == EXPECTED_SPIRV
    for capability in (
        "GroupNonUniformArithmetic",
        "GroupNonUniformClustered",
        "GroupNonUniformBallot",
        "GroupNonUniformVote",
    ):
        assert f"OpCapability {capability}" in assembly
    for mask in ("Eq", "Ge", "Gt", "Le", "Lt"):
        assert re.search(rf"OpDecorate %\w+ BuiltIn Subgroup{mask}Mask", assembly)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        source = tmp_path / "out.spvasm"
        binary = tmp_path / "out.spv"
        source.write_text(assembly, encoding="utf-8")
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
            check=True,
        )
        result = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_builds_extended_operations_from_shader_model_6_intrinsics(tmp_path):
    hlsl = _translate(tmp_path, GLSL_SUBGROUPS, "directx")

    assert re.findall(r"\bsubgroup[A-Z]\w*\s*\(", hlsl) == []
    assert "gl_Subgroup" not in hlsl
    # Lane-loop scans read every lane with a uniform index and keep active lanes only.
    assert "WaveReadLaneAt(value, source)" in hlsl
    assert "uint4 active = WaveActiveBallot(true);" in hlsl
    assert "float result = (float)(asfloat(0xff800000u));" in hlsl
    assert "__crossgl_WaveClusteredSum_float(x, 4u)" in hlsl
    # Logical reductions of bool values.
    assert "WaveActiveAllTrue((x > 1.0))" in hlsl
    assert "WaveActiveAnyTrue((x > 2.0))" in hlsl
    assert "((WaveActiveCountBits((x > 3.0)) & 1u) != 0u)" in hlsl
    assert "WaveActiveCountBits((k > 2u))" in hlsl
    assert "WavePrefixCountBits((k > 3u))" in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        path = tmp_path / "out.hlsl"
        path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(path), "-Fo", str(path) + ".o"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_metal_builds_extended_operations_from_simd_group_functions(tmp_path):
    metal = _translate(tmp_path, GLSL_SUBGROUPS, "metal")

    assert re.findall(r"\bsubgroup[A-Z]\w*\s*\(", metal) == []
    assert "gl_Subgroup" not in metal
    assert "[[thread_index_in_simdgroup]]" in metal
    assert "[[threads_per_simdgroup]]" in metal
    assert "simd_shuffle(value, ushort(source))" in metal
    assert "uint4 active = __crossgl_metal_wave_ballot(true);" in metal
    # Clusters of four lanes are quad-groups.
    assert "quad_sum(x)" in metal
    assert "simd_all(x > 1.0)" in metal
    assert "simd_any(x > 2.0)" in metal
    if shutil.which("xcrun"):
        path = tmp_path / "out.metal"
        path.write_text(metal, encoding="utf-8")
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(path)]
            + ["-o", str(tmp_path / "out.air")],
            capture_output=True,
            text=True,
        )
        if "unable to find utility" in result.stderr:
            pytest.skip("xcrun has no Metal compiler")
        assert result.returncode == 0, result.stderr


def test_boolean_vector_reductions_are_componentwise(tmp_path):
    source = _crossgl(
        "bool2 p = bool2(values[i] > 0.0, values[i] > 1.0);"
        "bool2 q = WaveActiveBitXor(p);"
        "results[i] = q.x && q.y ? 1u : 0u;"
    )
    hlsl = HLSLCodeGen().generate(parse(source))
    metal = MetalCodeGen().generate(parse(source))

    assert "bool2 __crossgl_WaveActiveBitXor_bool2(bool2 value)" in hlsl
    assert "WaveActiveCountBits(value.y)" in hlsl
    assert "static bool2 __crossgl_WaveActiveBitXor_bool2(bool2 value)" in metal


@pytest.mark.parametrize("size", ("3u", "0u"))
def test_cluster_size_must_be_a_power_of_two(size):
    source = _crossgl(f"results[i] = WaveClusteredSum(results[i], {size});")

    with pytest.raises(ValueError, match="power of two"):
        HLSLCodeGen().generate(parse(source))
    with pytest.raises(MetalLoweringRefused, match="power of two"):
        MetalCodeGen().generate(parse(source))
    with pytest.raises(SpirvLoweringRefused, match="power-of-two cluster size"):
        VulkanSPIRVCodeGen().generate(parse(source))


def test_float_bitwise_scans_are_refused():
    source = _crossgl("values[i] = WavePrefixInclusiveBitAnd(values[i]);")

    with pytest.raises(ValueError, match="integer or bool"):
        HLSLCodeGen().generate(parse(source))
    with pytest.raises(MetalLoweringRefused, match="integer or bool"):
        MetalCodeGen().generate(parse(source))
    with pytest.raises(SpirvLoweringRefused, match="no and operator"):
        VulkanSPIRVCodeGen().generate(parse(source))


def test_metal_refuses_an_unrecognized_wave_operation():
    ast = parse(_crossgl("results[i] = WaveActiveSum(results[i]);"))
    for node in _walk(ast):
        if isinstance(node, WaveOpNode):
            node.operation = "WaveActiveUnknown"

    with pytest.raises(MetalLoweringRefused, match="WaveActiveUnknown"):
        MetalCodeGen().generate(ast)


def test_hlsl_lane_masks_follow_the_lane_index():
    source = _crossgl("uint4 m = gl_SubgroupGtMask; results[i] = m.x;")
    hlsl = HLSLCodeGen().generate(parse(source))

    assert (
        "(~__crossgl_ballot_lanes_below(WaveGetLaneIndex() + 1u) & "
        "__crossgl_ballot_lanes_below(WaveGetLaneCount()))"
    ) in hlsl
    assert "uint4 __crossgl_ballot_lanes_below(uint lane)" in hlsl


def test_crossgl_accepts_the_glsl_spelling_of_wave_operations():
    source = _crossgl(
        "values[i] = subgroupAdd(values[i]) + subgroupClusteredMax(values[i], 4u);"
    )
    operations = {
        node.operation for node in _walk(parse(source)) if isinstance(node, WaveOpNode)
    }
    hlsl = HLSLCodeGen().generate(parse(source))
    metal = MetalCodeGen().generate(parse(source))

    assert operations == {"WaveActiveSum", "WaveClusteredMax"}
    assert "subgroup" not in hlsl and "subgroup" not in metal
    assert "WaveActiveSum(values[i])" in hlsl
    assert "simd_sum(values[i]) + quad_max(values[i])" in metal


GROUP_VOTE = """#version 460
#extension GL_ARB_shader_group_vote : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Results { uint results[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    bool a = allInvocations(i < 32u);
    bool b = anyInvocationARB(i == 3u);
    bool c = allInvocationsEqual(i / 64u);
    results[i] = uint(a) + uint(b) * 2u + uint(c) * 4u;
}
"""


def test_shader_group_vote_functions_are_canonical_wave_votes(tmp_path):
    # GLSL 4.60 section 8.18 and GL_ARB_shader_group_vote; glslang gives the group subgroup scope.
    crossgl = _crossgl(
        "results[i] = uint(allInvocations(i < 32u)) + uint(anyInvocation(i == 3u))"
        " + uint(allInvocationsEqual(i / 64u)) + uint(allInvocationsEqualARB(i));"
    )
    operations = {
        node.operation for node in _walk(parse(crossgl)) if isinstance(node, WaveOpNode)
    }
    assert operations == {
        "WaveActiveAllEqual",
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
    }

    glsl = _translate(tmp_path, GROUP_VOTE, "opengl")
    hlsl = _translate(tmp_path, GROUP_VOTE, "directx")
    metal = _translate(tmp_path, GROUP_VOTE, "metal")
    spirv = _translate(tmp_path, GROUP_VOTE, "vulkan")

    assert "#extension GL_KHR_shader_subgroup_vote : require" in glsl
    assert "subgroupAll((i < 32u))" in glsl
    assert "subgroupAny((i == 3u))" in glsl
    assert "subgroupAllEqual((i / 64u))" in glsl
    assert "Invocation(" not in glsl and "Invocations(" not in glsl
    assert "WaveActiveAllTrue((i < 32u))" in hlsl
    assert "WaveActiveAnyTrue((i == 3u))" in hlsl
    assert "WaveActiveAllEqual((i / 64u))" in hlsl
    assert "simd_all(i < 32u)" in metal and "simd_any(i == 3u)" in metal
    assert "simd_all(i / 64u == simd_broadcast_first(i / 64u))" in metal
    assert _spirv_operations(spirv) == [
        "OpGroupNonUniformAll",
        "OpGroupNonUniformAllEqual",
        "OpGroupNonUniformAny",
    ]
