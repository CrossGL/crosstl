SPIRV_VERTEX_POSITION_OUTPUT_SOURCE = """
; SPIR-V
; Version: 1.0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %pos
OpName %main "main"
OpName %pos "gl_Position"
OpDecorate %pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%const_pos = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_1
%pos = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpStore %pos %const_pos
OpReturn
OpFunctionEnd
"""


def assert_spirv_position_output_wgsl_contract(wgsl):
    assert "struct VertexOutput" in wgsl
    assert "@builtin(position) position: vec4<f32>," in wgsl
    assert "@vertex\nfn vertex_main() -> VertexOutput" in wgsl
    assert "output.position = vec4<f32>(0, 0, 0, 1);" in wgsl
    assert "vec4<f32>(f32(0), f32(0), f32(0), f32(1))" not in wgsl
    assert "return output;" in wgsl
    assert "gl_Position" not in wgsl
