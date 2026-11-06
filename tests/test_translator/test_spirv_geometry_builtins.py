"""Geometry stages imported from SPIR-V keep their primitive ID input apart and lower to HLSL streams.

SPIR-V names a geometry stage's incoming primitive ID and the one it writes with the same ``BuiltIn PrimitiveId``,
on an Input and an Output variable. GLSL calls them ``gl_PrimitiveIDIn`` and ``gl_PrimitiveID`` (GLSL 4.60 section
7.1.4). The importer named both ``gl_PrimitiveID``, so ``gl_PrimitiveID = gl_PrimitiveIDIn;`` came back as
``gl_PrimitiveID = gl_PrimitiveID;``, which every target compiled as a read of the stage's own output.

The importer also declares the built-ins a stage uses as stage variables. The HLSL geometry-stream rewrite took a
declared ``gl_InvocationID`` for a per-vertex input and refused it; it now lowers declared built-ins as it lowers
undeclared ones, to ``SV_GSInstanceID``, ``SV_PrimitiveID`` and output-structure members. A tessellation stage
whose HLSL entry attributes are not known is refused with a typed diagnostic instead of a bare ``ValueError``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXStageAttributesMissingError

# glslang 15 output, comments and the debug names of locals trimmed, for:
#   layout(triangles, invocations = 2) in;  layout(triangle_strip, max_vertices = 3) out;
#   layout(location = 0) in vec2 inUV[];  layout(location = 0) out vec2 outUV;
#   layout(location = 1) flat out int outInfo;
#   for (int i = 0; i < 3; ++i) {
#       gl_Layer = gl_InvocationID; gl_PrimitiveID = gl_PrimitiveIDIn; outUV = inUV[i];
#       outInfo = gl_InvocationID * 16 + gl_PrimitiveIDIn; gl_Position = gl_in[i].gl_Position; EmitVertex();
#   }
#   EndPrimitive();
GEOMETRY = """OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %gl_Layer %gl_InvocationID %gl_PrimitiveID %gl_PrimitiveIDIn %outUV %inUV %outInfo %_ %gl_in
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 2
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpName %gl_Layer "gl_Layer"
OpName %gl_InvocationID "gl_InvocationID"
OpName %gl_PrimitiveID "gl_PrimitiveID"
OpName %gl_PrimitiveIDIn "gl_PrimitiveIDIn"
OpName %outUV "outUV"
OpName %inUV "inUV"
OpName %outInfo "outInfo"
OpName %gl_PerVertex "gl_PerVertex"
OpMemberName %gl_PerVertex 0 "gl_Position"
OpMemberName %gl_PerVertex 1 "gl_PointSize"
OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
OpMemberName %gl_PerVertex 3 "gl_CullDistance"
OpName %_ ""
OpName %gl_PerVertex_0 "gl_PerVertex"
OpMemberName %gl_PerVertex_0 0 "gl_Position"
OpMemberName %gl_PerVertex_0 1 "gl_PointSize"
OpMemberName %gl_PerVertex_0 2 "gl_ClipDistance"
OpMemberName %gl_PerVertex_0 3 "gl_CullDistance"
OpName %gl_in "gl_in"
OpDecorate %gl_Layer BuiltIn Layer
OpDecorate %gl_InvocationID BuiltIn InvocationId
OpDecorate %gl_PrimitiveID BuiltIn PrimitiveId
OpDecorate %gl_PrimitiveIDIn BuiltIn PrimitiveId
OpDecorate %outUV Location 0
OpDecorate %inUV Location 0
OpDecorate %outInfo Flat
OpDecorate %outInfo Location 1
OpDecorate %gl_PerVertex Block
OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
OpDecorate %gl_PerVertex_0 Block
OpMemberDecorate %gl_PerVertex_0 0 BuiltIn Position
OpMemberDecorate %gl_PerVertex_0 1 BuiltIn PointSize
OpMemberDecorate %gl_PerVertex_0 2 BuiltIn ClipDistance
OpMemberDecorate %gl_PerVertex_0 3 BuiltIn CullDistance
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_3 = OpConstant %int 3
%bool = OpTypeBool
%_ptr_Output_int = OpTypePointer Output %int
%gl_Layer = OpVariable %_ptr_Output_int Output
%_ptr_Input_int = OpTypePointer Input %int
%gl_InvocationID = OpVariable %_ptr_Input_int Input
%gl_PrimitiveID = OpVariable %_ptr_Output_int Output
%gl_PrimitiveIDIn = OpVariable %_ptr_Input_int Input
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Output_v2float = OpTypePointer Output %v2float
%outUV = OpVariable %_ptr_Output_v2float Output
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%_arr_v2float_uint_3 = OpTypeArray %v2float %uint_3
%_ptr_Input__arr_v2float_uint_3 = OpTypePointer Input %_arr_v2float_uint_3
%inUV = OpVariable %_ptr_Input__arr_v2float_uint_3 Input
%_ptr_Input_v2float = OpTypePointer Input %v2float
%outInfo = OpVariable %_ptr_Output_int Output
%int_16 = OpConstant %int 16
%v4float = OpTypeVector %float 4
%uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_PerVertex = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
%_ = OpVariable %_ptr_Output_gl_PerVertex Output
%gl_PerVertex_0 = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_arr_gl_PerVertex_0_uint_3 = OpTypeArray %gl_PerVertex_0 %uint_3
%_ptr_Input__arr_gl_PerVertex_0_uint_3 = OpTypePointer Input %_arr_gl_PerVertex_0_uint_3
%gl_in = OpVariable %_ptr_Input__arr_gl_PerVertex_0_uint_3 Input
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%int_1 = OpConstant %int 1
%main = OpFunction %void None %3
%5 = OpLabel
%i = OpVariable %_ptr_Function_int Function
OpStore %i %int_0
OpBranch %10
%10 = OpLabel
OpLoopMerge %12 %13 None
OpBranch %14
%14 = OpLabel
%15 = OpLoad %int %i
%18 = OpSLessThan %bool %15 %int_3
OpBranchConditional %18 %11 %12
%11 = OpLabel
%23 = OpLoad %int %gl_InvocationID
OpStore %gl_Layer %23
%26 = OpLoad %int %gl_PrimitiveIDIn
OpStore %gl_PrimitiveID %26
%36 = OpLoad %int %i
%38 = OpAccessChain %_ptr_Input_v2float %inUV %36
%39 = OpLoad %v2float %38
OpStore %outUV %39
%41 = OpLoad %int %gl_InvocationID
%43 = OpIMul %int %41 %int_16
%44 = OpLoad %int %gl_PrimitiveIDIn
%45 = OpIAdd %int %43 %44
OpStore %outInfo %45
%56 = OpLoad %int %i
%58 = OpAccessChain %_ptr_Input_v4float %gl_in %56 %int_0
%59 = OpLoad %v4float %58
%61 = OpAccessChain %_ptr_Output_v4float %_ %int_0
OpStore %61 %59
OpEmitVertex
OpBranch %13
%13 = OpLabel
%62 = OpLoad %int %i
%64 = OpIAdd %int %62 %int_1
OpStore %i %64
OpBranch %10
%12 = OpLabel
OpEndPrimitive
OpReturn
OpFunctionEnd
"""

FRAGMENT = """OpCapability Shader
OpCapability Geometry
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %c %gl_PrimitiveID
OpExecutionMode %main OriginUpperLeft
OpName %c "c"
OpName %gl_PrimitiveID "gl_PrimitiveID"
OpDecorate %c Location 0
OpDecorate %gl_PrimitiveID BuiltIn PrimitiveId
OpDecorate %gl_PrimitiveID Flat
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%c = OpVariable %_ptr_Output_v4float Output
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_PrimitiveID = OpVariable %_ptr_Input_int Input
%main = OpFunction %void None %3
%5 = OpLabel
%13 = OpLoad %int %gl_PrimitiveID
%14 = OpConvertSToF %float %13
%15 = OpCompositeConstruct %v4float %14 %14 %14 %14
OpStore %c %15
OpReturn
OpFunctionEnd
"""

# A geometry stage written the way the SPIR-V importer writes one: its built-ins are declared stage variables.
DECLARED_BUILTINS_CGL = """
shader main {
    geometry {
        layout(points, invocations = 4) in;
        layout(points, max_vertices = 1) out;
        int gl_Layer @output @gl_Layer;
        int gl_InvocationID @input @gl_InvocationID;
        int gl_PrimitiveIDIn @input @gl_PrimitiveIDIn;
        float4 gl_Position @output @gl_Position;
        float gl_PointSize @output @gl_PointSize;
        float value[1] @input @location(0);
        float outValue @output @location(0);
        void main() {
            gl_Layer = gl_InvocationID;
            outValue = value[0] + float(gl_PrimitiveIDIn);
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
        }
    }
}
"""

TOOLS = {"directx": "dxc", "opengl": "glslangValidator", "vulkan": "spirv-val"}


def _translate(tmp_path, source, backend, suffix=".spvasm"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile(tmp_path, code, backend, stage="geom"):
    """Compile the output with the native tool when it is installed."""
    if shutil.which(TOOLS[backend]) is None:
        return
    if backend == "directx":
        (tmp_path / "out.hlsl").write_text(code)
        profile = {"geom": "gs_6_0", "frag": "ps_6_0"}[stage]
        entry = {"geom": "GSMain", "frag": "PSMain"}[stage]
        _run(["dxc", "-T", profile, "-E", entry, "out.hlsl"], tmp_path)
    elif backend == "opengl":
        (tmp_path / f"out.{stage}").write_text(code)
        _run(["glslangValidator", f"out.{stage}"], tmp_path)
    elif shutil.which("spirv-as"):
        (tmp_path / "out.spvasm").write_text(code)
        _run(["spirv-as", "out.spvasm", "-o", "out.spv"], tmp_path)
        _run(["spirv-val", "out.spv"], tmp_path)


def _lines(code):
    return [line.strip() for line in code.splitlines()]


def test_geometry_primitive_id_input_imports_as_gl_primitive_id_in(tmp_path):
    lines = _lines(_translate(tmp_path, GEOMETRY, "cgl"))
    assert "int gl_PrimitiveIDIn @input @gl_PrimitiveIDIn;" in lines
    assert "int gl_PrimitiveID @output @gl_PrimitiveID;" in lines
    assert "gl_PrimitiveID = gl_PrimitiveIDIn;" in lines
    assert "outInfo = ((gl_InvocationID * 16) + gl_PrimitiveIDIn);" in lines


def test_fragment_primitive_id_input_keeps_its_name(tmp_path):
    lines = _lines(_translate(tmp_path, FRAGMENT, "cgl"))
    assert "int gl_PrimitiveID @input @gl_PrimitiveID @flat;" in lines
    assert not any("gl_PrimitiveIDIn" in line for line in lines)


@pytest.mark.parametrize(
    ("backend", "spellings"),
    [
        ("opengl", ("gl_PrimitiveID = gl_PrimitiveIDIn;",)),
        ("vulkan", ('"gl_PrimitiveIDIn"', '"gl_PrimitiveID"')),
        (
            "directx",
            (
                "[instance(2)]",
                "uint invocationID : SV_GSInstanceID",
                "uint primitiveID : SV_PrimitiveID",
                "output.primitiveID = uint(int(primitiveID));",
                "output.layer = uint(int(invocationID));",
            ),
        ),
    ],
)
def test_geometry_from_spirv_compiles_for_each_target(tmp_path, backend, spellings):
    code = _translate(tmp_path, GEOMETRY, backend)
    for spelling in spellings:
        assert spelling in code
    _compile(tmp_path, code, backend)


def test_spirv_round_trip_keeps_two_primitive_id_variables(tmp_path):
    code = _translate(tmp_path, GEOMETRY, "vulkan")
    assert code.count("BuiltIn PrimitiveId") == 2


def test_geometry_with_declared_builtins_lowers_to_hlsl_streams(tmp_path):
    code = _translate(tmp_path, DECLARED_BUILTINS_CGL, "directx", suffix=".cgl")
    for spelling in (
        "[instance(4)]",
        "uint invocationID : SV_GSInstanceID",
        "uint primitiveID : SV_PrimitiveID",
        "inout PointStream<GeometryOutput> stream",
        "stream.Append(output);",
    ):
        assert spelling in code
    # The declared built-ins become entry parameters and output members, not globals.
    assert "int gl_InvocationID" not in code
    assert "gl_PointSize" not in code
    _compile(tmp_path, code, "directx")


def test_fragment_primitive_id_from_spirv_compiles(tmp_path):
    for backend in ("opengl", "directx"):
        code = _translate(tmp_path, FRAGMENT, backend)
        assert "gl_PrimitiveIDIn" not in code
        _compile(tmp_path, code, backend, stage="frag")


TESSELLATION_EVALUATION_CGL = """
shader main {
    tessellation_evaluation {
        layout(triangles, equal_spacing, cw) in;
        float3 gl_TessCoord @input @gl_TessCoord;
        float4 gl_Position @output @gl_Position;
        void main() {
            gl_Position = float4(gl_TessCoord, 1.0);
        }
    }
}
"""


def test_tessellation_stage_without_hlsl_attributes_is_a_typed_refusal(tmp_path):
    with pytest.raises(DirectXStageAttributesMissingError) as error:
        _translate(tmp_path, TESSELLATION_EVALUATION_CGL, "directx", suffix=".cgl")
    assert error.value.stage == "tessellation_evaluation"
    assert error.value.missing == ("domain",)
    assert error.value.code.endswith("directx-stage-attributes-missing")
