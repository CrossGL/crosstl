"""SPIR-V import keeps row-major matrix members, nested prefix operators and patch outputs exact."""

import re
import shutil
import subprocess

import pytest

import crosstl

# glslang's module for: struct Rows { mat2 m; vec2 w; }; layout(std430, binding = 1) buffer B1
# { layout(row_major) Rows outp[]; }; layout(std140, binding = 3) uniform Params { layout(row_major) mat3x2 p;
# layout(row_major) mat3 q; }; with outp[i].m[0][1] = p[1][0] + q[2][1].
ROW_MAJOR = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 1 1 1
OpName %Rows "Rows"
OpMemberName %Rows 0 "m"
OpMemberName %Rows 1 "w"
OpName %B1 "B1"
OpMemberName %B1 0 "outp"
OpName %Params "Params"
OpMemberName %Params 0 "p"
OpMemberName %Params 1 "q"
OpDecorate %gid BuiltIn GlobalInvocationId
OpMemberDecorate %Rows 0 RowMajor
OpMemberDecorate %Rows 0 MatrixStride 8
OpMemberDecorate %Rows 0 Offset 0
OpMemberDecorate %Rows 1 Offset 16
OpDecorate %_runtimearr_Rows ArrayStride 24
OpDecorate %B1 Block
OpMemberDecorate %B1 0 Offset 0
OpDecorate %buf Binding 1
OpDecorate %buf DescriptorSet 0
OpDecorate %Params Block
OpMemberDecorate %Params 0 RowMajor
OpMemberDecorate %Params 0 MatrixStride 16
OpMemberDecorate %Params 0 Offset 0
OpMemberDecorate %Params 1 RowMajor
OpMemberDecorate %Params 1 MatrixStride 16
OpMemberDecorate %Params 1 Offset 32
OpDecorate %params Binding 3
OpDecorate %params DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gid = OpVariable %_ptr_Input_v3uint Input
%_ptr_Input_uint = OpTypePointer Input %uint
%uint_0 = OpConstant %uint 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v3float = OpTypeVector %float 3
%mat2v2float = OpTypeMatrix %v2float 2
%mat3v2float = OpTypeMatrix %v2float 3
%mat3v3float = OpTypeMatrix %v3float 3
%Rows = OpTypeStruct %mat2v2float %v2float
%_runtimearr_Rows = OpTypeRuntimeArray %Rows
%B1 = OpTypeStruct %_runtimearr_Rows
%_ptr_StorageBuffer_B1 = OpTypePointer StorageBuffer %B1
%buf = OpVariable %_ptr_StorageBuffer_B1 StorageBuffer
%Params = OpTypeStruct %mat3v2float %mat3v3float
%_ptr_Uniform_Params = OpTypePointer Uniform %Params
%params = OpVariable %_ptr_Uniform_Params Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%main = OpFunction %void None %3
%5 = OpLabel
%10 = OpAccessChain %_ptr_Input_uint %gid %uint_0
%11 = OpLoad %uint %10
%12 = OpAccessChain %_ptr_Uniform_float %params %int_0 %int_1 %int_0
%13 = OpLoad %float %12
%14 = OpAccessChain %_ptr_Uniform_float %params %int_1 %int_2 %int_1
%15 = OpLoad %float %14
%16 = OpFAdd %float %13 %15
%17 = OpAccessChain %_ptr_StorageBuffer_float %buf %int_0 %11 %int_0 %int_0 %int_1
OpStore %17 %16
OpReturn
OpFunctionEnd
"""

# glslang's module for: float a = -(-x); int e = ~(~k); bool g = !(!b); stored to a buffer.
NESTED_UNARY = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpDecorate %Out Block
OpMemberDecorate %Out 0 Offset 0
OpDecorate %out Binding 0
OpDecorate %out DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%bool = OpTypeBool
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_5 = OpConstant %int 5
%float_2 = OpConstant %float 2
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%true = OpConstantTrue %bool
%_runtimearr_float = OpTypeRuntimeArray %float
%Out = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Out = OpTypePointer StorageBuffer %Out
%out = OpVariable %_ptr_StorageBuffer_Out StorageBuffer
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%_ptr_Function_float = OpTypePointer Function %float
%main = OpFunction %void None %3
%5 = OpLabel
%x = OpVariable %_ptr_Function_float Function
OpStore %x %float_2
%20 = OpLoad %float %x
%21 = OpFNegate %float %20
%22 = OpFNegate %float %21
%23 = OpNot %int %int_5
%24 = OpNot %int %23
%25 = OpConvertSToF %float %24
%26 = OpLogicalNot %bool %true
%27 = OpLogicalNot %bool %26
%28 = OpSelect %float %27 %float_1 %float_0
%29 = OpFAdd %float %22 %25
%30 = OpFAdd %float %29 %28
%31 = OpAccessChain %_ptr_StorageBuffer_float %out %int_0 %int_0
OpStore %31 %30
OpReturn
OpFunctionEnd
"""

PATCH_OUTPUT = """shader main {
    tessellation_control {
        layout(vertices = 3) out;
        float4 outPosition[3] @output @location(0);
        int gl_InvocationID @input @gl_InvocationID;
        float4 position[32] @input @location(0);
        float4 extra @output @location(1) @patch;
        float gl_TessLevelOuter[4] @output @gl_TessLevelOuter @patch;
        float gl_TessLevelInner[2] @output @gl_TessLevelInner @patch;
        void main() {
            outPosition[gl_InvocationID] = position[gl_InvocationID] * 2.0;
            extra = position[0];
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = 1.0;
            gl_TessLevelOuter[2] = 1.0;
            gl_TessLevelInner[0] = 1.0;
        }
    }
}
"""


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _members(crossgl):
    return {
        match[2]: (match[1], match[3].split())
        for match in re.finditer(
            r"^\s*(\w+) (\w+)(?:\[\])?((?: @[\w()]+)*);", crossgl, re.M
        )
    }


def test_row_major_members_import_as_row_major(tmp_path):
    crossgl = _translate(tmp_path, ROW_MAJOR, "shader.spvasm", "cgl")
    members = _members(crossgl)
    # A struct member, and both members of a uniform block, keep their RowMajor order.
    assert members["m"] == ("mat2", ["@row_major"])
    assert members["w"] == ("float2", [])
    assert members["p"] == ("mat3x2", ["@row_major"])
    assert members["q"] == ("mat3", ["@row_major"])


def test_square_matrices_import_as_matn(tmp_path):
    crossgl = _translate(tmp_path, ROW_MAJOR, "shader.spvasm", "cgl")
    assert not re.search(r"\bfloat[234]x[234]\b", crossgl)
    assert re.search(r"\bmat2 m\b", crossgl) and re.search(r"\bmat3 q\b", crossgl)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not on PATH")
    path = tmp_path / "round_trip.spvasm"
    binary = tmp_path / "round_trip.spv"
    path.write_text(assembly)
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _chain_index(assembly, pointer):
    """``(base, index)`` of the one-index ``OpAccessChain`` that defines ``pointer``."""
    match = re.search(
        rf"^\s*{re.escape(pointer)} = OpAccessChain %\w+ (%\w+) (%\w+)\s*$",
        assembly,
        re.M,
    )
    assert match, pointer
    return match[1], match[2]


def _constant(assembly, value_id):
    match = re.search(
        rf"^\s*{re.escape(value_id)} = OpConstant %\w+ (\d+)\s*$", assembly, re.M
    )
    assert match, value_id
    return int(match[1])


def test_row_major_round_trip_keeps_the_stored_order(tmp_path):
    assembly = _translate(tmp_path, ROW_MAJOR, "shader.spvasm", "vulkan")
    # The uniform block members are decorated RowMajor again.
    assert len(re.findall(r"OpMemberDecorate %\w+ [01] RowMajor\b", assembly)) == 2
    # The storage struct holds its matrix transposed, so m[0][1] is stored as element [1][0].
    pointer = re.search(r"^\s*OpStore (%\w+) %\w+\s*$", assembly, re.M)[1]
    column, component = _chain_index(assembly, pointer)
    _matrix, column_index = _chain_index(assembly, column)
    assert (_constant(assembly, column_index), _constant(assembly, component)) == (1, 0)
    _validate(tmp_path, assembly)


def test_nested_prefix_operators_stay_unary(tmp_path):
    crossgl = _translate(tmp_path, NESTED_UNARY, "shader.spvasm", "cgl")
    assert "--" not in crossgl and "++" not in crossgl
    assert re.search(r"-\(-\w+\)", crossgl)
    assert re.search(r"!\(!\w+\)", crossgl)
    assembly = _translate(tmp_path, NESTED_UNARY, "shader.spvasm", "vulkan")
    assert assembly.count("OpFNegate") == 2
    assert "OpFSub" not in assembly
    _validate(tmp_path, assembly)


def test_patch_attribute_decorates_a_located_output(tmp_path):
    assembly = _translate(tmp_path, PATCH_OUTPUT, "shader.cgl", "vulkan")
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', assembly))
    patched = {
        names.get(i, i) for i in re.findall(r"OpDecorate %(\w+) Patch\b", assembly)
    }
    assert patched == {"extra", "gl_TessLevelOuter", "gl_TessLevelInner"}
    _validate(tmp_path, assembly)
