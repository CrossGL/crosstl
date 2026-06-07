import re
from typing import List

import pytest

from crosstl.backend.SPIRV import VulkanCrossGLCodeGen
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser
from crosstl.translator import parse as parse_crossgl
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.source_registry import BINARY_SPIRV_UNSUPPORTED_MESSAGE


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = VulkanCrossGLCodeGen.VulkanToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    lexer = VulkanLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    parser = VulkanParser(tokens)
    return parser.parse()


FRAGMENT_SHADER = """
void main() {
    vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
    gl_FragColor = color;
}
"""

LAYOUT_SHADER = """
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 tint;
} camera;
layout(set = 0, binding = 1) buffer Particles {
    vec4 position;
    float mass;
} particles;
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

ARRAY_LAYOUT_SHADER = """
layout(set = 0, binding = 0) uniform Bones {
    mat4 transforms[64];
    vec4 weights[4];
} bones;
layout(set = 0, binding = 1) buffer Particles {
    vec4 positions[128];
    float masses[128];
} particles;
void main() {
    gl_Position = transforms[0] * vec4(1.0, 0.0, 0.0, 1.0);
}
"""

RESOURCE_UNIFORM_SHADER = """
layout(set = 0, binding = 1) uniform sampler2D albedoTex;
uniform samplerCube skybox;
void main() {
    vec4 color = texture(albedoTex, vec2(0.5, 0.5));
    gl_FragColor = color;
}
"""

SPIRV_TOOLS_BASIC_INTERFACE_ASSEMBLY = """
; Reduced from Khronos SPIRV-Tools test/diff/diff_files/basic_src.spvasm.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %22 "main" %4 %14 %19
OpName %4 "_ua_position"
OpName %14 "ANGLEXfbPosition"
OpName %19 ""
OpDecorate %4 Location 0
OpDecorate %14 Location 0
OpMemberDecorate %17 0 BuiltIn Position
OpDecorate %17 Block
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%17 = OpTypeStruct %2
%20 = OpTypeVoid
%3 = OpTypePointer Input %2
%13 = OpTypePointer Output %2
%18 = OpTypePointer Output %17
%21 = OpTypeFunction %20
%4 = OpVariable %3 Input
%14 = OpVariable %13 Output
%19 = OpVariable %18 Output
%22 = OpFunction %20 None %21
%23 = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_NON_MAIN_VERTEX_ENTRY_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %vs "vs_main" %pos
OpName %pos "outPosition"
OpDecorate %pos BuiltIn Position
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%void = OpTypeVoid
%ptr_output_v4float = OpTypePointer Output %v4float
%fn = OpTypeFunction %void
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%const_pos = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_1
%pos = OpVariable %ptr_output_v4float Output
%vs = OpFunction %void None %fn
%label = OpLabel
OpStore %pos %const_pos
OpReturn
OpFunctionEnd
"""

SPIRV_MULTI_ENTRYPOINT_REUSED_LOCATION_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %vs "vs_main" %vertex_in %varying_out %position
OpEntryPoint Fragment %fs "fs_main" %varying_in %frag_out
OpExecutionMode %fs OriginUpperLeft
OpName %vertex_in "vertexIn"
OpName %varying_out "vColor"
OpName %varying_in "fColor"
OpName %frag_out "fragOut"
OpDecorate %vertex_in Location 0
OpDecorate %varying_out Location 0
OpDecorate %varying_in Location 0
OpDecorate %frag_out Location 0
OpDecorate %position BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%vertex_in = OpVariable %ptr_input_v4float Input
%varying_out = OpVariable %ptr_output_v4float Output
%position = OpVariable %ptr_output_v4float Output
%varying_in = OpVariable %ptr_input_v4float Input
%frag_out = OpVariable %ptr_output_v4float Output
%vs = OpFunction %void None %fn
%vs_label = OpLabel
%loaded_vertex = OpLoad %v4float %vertex_in
OpStore %position %loaded_vertex
OpStore %varying_out %loaded_vertex
OpReturn
OpFunctionEnd
%fs = OpFunction %void None %fn
%fs_label = OpLabel
%loaded_frag = OpLoad %v4float %varying_in
OpStore %frag_out %loaded_frag
OpReturn
OpFunctionEnd
"""

SPIRV_MATRIX_INTERFACE_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %model
OpName %model "model"
OpDecorate %model Location 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%mat4 = OpTypeMatrix %v4float 4
%void = OpTypeVoid
%ptr_input_mat4 = OpTypePointer Input %mat4
%fn = OpTypeFunction %void
%model = OpVariable %ptr_input_mat4 Input
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_PRIVATE_GLOBAL_ASSEMBLY = """
; Reduced from glslangValidator -V -H output for GLSL module-scope globals:
; vec4 privateColor; float privateWeight = 1.0;
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color
OpExecutionMode %main OriginUpperLeft
OpName %private_color "privateColor"
OpName %private_weight "privateWeight"
OpName %out_color "outColor"
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_private_v4float = OpTypePointer Private %v4float
%ptr_private_float = OpTypePointer Private %float
%ptr_output_v4float = OpTypePointer Output %v4float
%one = OpConstant %float 1.0
%zero = OpConstant %float 0.0
%red = OpConstantComposite %v4float %one %zero %zero %one
%private_color = OpVariable %ptr_private_v4float Private
%private_weight = OpVariable %ptr_private_float Private %one
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpStore %private_color %red
%loaded = OpLoad %v4float %private_color
OpStore %out_color %loaded
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_SIMPLE_MAT_MATRIX_TIMES_VECTOR_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.simpleMat.vert.out
; Reduced from MatrixTimesVector in the vertex main body.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %glPos %mvp %v
OpName %glPos "glPos"
OpName %mvp "mvp"
OpName %v "v"
OpDecorate %glPos Location 5
OpDecorate %mvp Location 0
OpDecorate %v Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%mat4 = OpTypeMatrix %v4float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_output_mat4 = OpTypePointer Output %mat4
%ptr_input_v4float = OpTypePointer Input %v4float
%glPos = OpVariable %ptr_output_v4float Output
%mvp = OpVariable %ptr_output_mat4 Output
%v = OpVariable %ptr_input_v4float Input
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_mvp = OpLoad %mat4 %mvp
%loaded_v = OpLoad %v4float %v
%transformed = OpMatrixTimesVector %v4float %loaded_mvp %loaded_v
OpStore %glPos %transformed
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_MATRIX_TRANSPOSE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.matrix.frag.out
; Reduced from Test/spv.matrix.frag transpose(sum34).
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %sum34 %m43
OpExecutionMode %main OriginUpperLeft
OpName %sum34 "sum34"
OpName %m43 "m43"
OpDecorate %sum34 Location 0
OpDecorate %m43 Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%v3float = OpTypeVector %float 3
%mat3x4 = OpTypeMatrix %v4float 3
%mat4x3 = OpTypeMatrix %v3float 4
%ptr_input_mat3x4 = OpTypePointer Input %mat3x4
%ptr_output_mat4x3 = OpTypePointer Output %mat4x3
%sum34 = OpVariable %ptr_input_mat3x4 Input
%m43 = OpVariable %ptr_output_mat4x3 Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %mat3x4 %sum34
%transposed = OpTranspose %mat4x3 %loaded
OpStore %m43 %transposed
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_OUTER_PRODUCT_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Reduced from the core OpOuterProduct instruction definition.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %column %row %basis
OpExecutionMode %main OriginUpperLeft
OpName %column "column"
OpName %row "row"
OpName %basis "basis"
OpDecorate %column Location 0
OpDecorate %row Location 1
OpDecorate %basis Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v3float = OpTypeVector %float 3
%mat3x2 = OpTypeMatrix %v2float 3
%ptr_input_v3float = OpTypePointer Input %v3float
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_mat3x2 = OpTypePointer Output %mat3x2
%column = OpVariable %ptr_input_v3float Input
%row = OpVariable %ptr_input_v2float Input
%basis = OpVariable %ptr_output_mat3x2 Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_column = OpLoad %v3float %column
%loaded_row = OpLoad %v2float %row
%outer = OpOuterProduct %mat3x2 %loaded_column %loaded_row
OpStore %basis %outer
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_FCONVERT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.matrix.frag.out
; Reduced from Test/spv.matrix.frag float-to-double FConvert in sum34 -> dm.
OpCapability Shader
OpCapability Float64
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %double_out
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %double_out "doubleOut"
OpDecorate %input_value Location 0
OpDecorate %double_out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%double = OpTypeFloat 64
%ptr_input_float = OpTypePointer Input %float
%ptr_output_double = OpTypePointer Output %double
%input_value = OpVariable %ptr_input_float Input
%double_out = OpVariable %ptr_output_double Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%converted = OpFConvert %double %loaded
OpStore %double_out %converted
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_FLOAT16_INTERFACE_ASSEMBLY = """
; Source tool: glslangValidator -V -H, reduced from fragment shader code using
; GL_EXT_shader_explicit_arithmetic_types_float16 / Float16 storage.
OpCapability Shader
OpCapability Float16
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_half %input_matrix %half_out %matrix_out %float_out
OpExecutionMode %main OriginUpperLeft
OpName %input_half "inputHalf"
OpName %input_matrix "inputMatrix"
OpName %half_out "halfOut"
OpName %matrix_out "matrixOut"
OpName %float_out "floatOut"
OpDecorate %input_half Location 0
OpDecorate %input_matrix Location 1
OpDecorate %half_out Location 0
OpDecorate %float_out Location 1
OpDecorate %matrix_out Location 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%v2half = OpTypeVector %half 2
%v4half = OpTypeVector %half 4
%mat2half = OpTypeMatrix %v2half 2
%ptr_input_v4half = OpTypePointer Input %v4half
%ptr_input_mat2half = OpTypePointer Input %mat2half
%ptr_output_v4half = OpTypePointer Output %v4half
%ptr_output_mat2half = OpTypePointer Output %mat2half
%ptr_output_float = OpTypePointer Output %float
%input_half = OpVariable %ptr_input_v4half Input
%input_matrix = OpVariable %ptr_input_mat2half Input
%half_out = OpVariable %ptr_output_v4half Output
%matrix_out = OpVariable %ptr_output_mat2half Output
%float_out = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4half %input_half
%loaded_matrix = OpLoad %mat2half %input_matrix
%component = OpCompositeExtract %half %loaded 0
%wide = OpFConvert %float %component
OpStore %half_out %loaded
OpStore %matrix_out %loaded_matrix
OpStore %float_out %wide
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_INT16_INT64_INTERFACE_ASSEMBLY = """
; Source tool: glslangValidator -V -H, reduced from fragment shader code using
; GL_EXT_shader_explicit_arithmetic_types_int16/int64 and integer conversions.
OpCapability Shader
OpCapability Int16
OpCapability Int64
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %signed16_in %unsigned16_in %signed16_out %unsigned16_out %signed64_out %unsigned64_out
OpExecutionMode %main OriginUpperLeft
OpName %signed16_in "signed16In"
OpName %unsigned16_in "unsigned16In"
OpName %signed16_out "signed16Out"
OpName %unsigned16_out "unsigned16Out"
OpName %signed64_out "signed64Out"
OpName %unsigned64_out "unsigned64Out"
OpDecorate %signed16_in Location 0
OpDecorate %unsigned16_in Location 1
OpDecorate %signed16_out Location 0
OpDecorate %unsigned16_out Location 1
OpDecorate %signed64_out Location 2
OpDecorate %unsigned64_out Location 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%short = OpTypeInt 16 1
%ushort = OpTypeInt 16 0
%long = OpTypeInt 64 1
%ulong = OpTypeInt 64 0
%v4short = OpTypeVector %short 4
%v3ushort = OpTypeVector %ushort 3
%ptr_input_v4short = OpTypePointer Input %v4short
%ptr_input_v3ushort = OpTypePointer Input %v3ushort
%ptr_output_v4short = OpTypePointer Output %v4short
%ptr_output_v3ushort = OpTypePointer Output %v3ushort
%ptr_output_long = OpTypePointer Output %long
%ptr_output_ulong = OpTypePointer Output %ulong
%signed16_in = OpVariable %ptr_input_v4short Input
%unsigned16_in = OpVariable %ptr_input_v3ushort Input
%signed16_out = OpVariable %ptr_output_v4short Output
%unsigned16_out = OpVariable %ptr_output_v3ushort Output
%signed64_out = OpVariable %ptr_output_long Output
%unsigned64_out = OpVariable %ptr_output_ulong Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_signed16 = OpLoad %v4short %signed16_in
%loaded_unsigned16 = OpLoad %v3ushort %unsigned16_in
%signed_component = OpCompositeExtract %short %loaded_signed16 0
%unsigned_component = OpCompositeExtract %ushort %loaded_unsigned16 1
%wide_signed = OpSConvert %long %signed_component
%wide_unsigned = OpUConvert %ulong %unsigned_component
OpStore %signed16_out %loaded_signed16
OpStore %unsigned16_out %loaded_unsigned16
OpStore %signed64_out %wide_signed
OpStore %unsigned64_out %wide_unsigned
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_INT8_INTERFACE_ASSEMBLY = """
; Source tool: glslangValidator -V -H, reduced from fragment shader code using
; GL_EXT_shader_explicit_arithmetic_types_int8 vector interfaces.
OpCapability Shader
OpCapability Int8
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %signed8_in %unsigned8_in %signed8_out %unsigned8_out %signed8_scalar_out %unsigned8_scalar_out
OpExecutionMode %main OriginUpperLeft
OpName %signed8_in "signed8In"
OpName %unsigned8_in "unsigned8In"
OpName %signed8_out "signed8Out"
OpName %unsigned8_out "unsigned8Out"
OpName %signed8_scalar_out "signed8ScalarOut"
OpName %unsigned8_scalar_out "unsigned8ScalarOut"
OpDecorate %signed8_in Location 0
OpDecorate %unsigned8_in Location 1
OpDecorate %signed8_out Location 0
OpDecorate %unsigned8_out Location 1
OpDecorate %signed8_scalar_out Location 2
OpDecorate %unsigned8_scalar_out Location 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%char = OpTypeInt 8 1
%uchar = OpTypeInt 8 0
%v4char = OpTypeVector %char 4
%v2uchar = OpTypeVector %uchar 2
%ptr_input_v4char = OpTypePointer Input %v4char
%ptr_input_v2uchar = OpTypePointer Input %v2uchar
%ptr_output_v4char = OpTypePointer Output %v4char
%ptr_output_v2uchar = OpTypePointer Output %v2uchar
%ptr_output_char = OpTypePointer Output %char
%ptr_output_uchar = OpTypePointer Output %uchar
%signed8_in = OpVariable %ptr_input_v4char Input
%unsigned8_in = OpVariable %ptr_input_v2uchar Input
%signed8_out = OpVariable %ptr_output_v4char Output
%unsigned8_out = OpVariable %ptr_output_v2uchar Output
%signed8_scalar_out = OpVariable %ptr_output_char Output
%unsigned8_scalar_out = OpVariable %ptr_output_uchar Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_signed8 = OpLoad %v4char %signed8_in
%loaded_unsigned8 = OpLoad %v2uchar %unsigned8_in
%signed_component = OpCompositeExtract %char %loaded_signed8 0
%unsigned_component = OpCompositeExtract %uchar %loaded_unsigned8 1
OpStore %signed8_out %loaded_signed8
OpStore %unsigned8_out %loaded_unsigned8
OpStore %signed8_scalar_out %signed_component
OpStore %unsigned8_scalar_out %unsigned_component
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_PRECISE_DOT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.precise.dot.vert.out
; Reduced from Test/spv.precise.dot.vert precise dot(v, v).
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %out_value %input_vec
OpName %out_value "outValue"
OpName %input_vec "inputVec"
OpDecorate %out_value Location 0
OpDecorate %input_vec Location 0
OpDecorate %dot NoContraction
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_float = OpTypePointer Output %float
%ptr_input_v4float = OpTypePointer Input %v4float
%out_value = OpVariable %ptr_output_float Output
%input_vec = OpVariable %ptr_input_v4float Input
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %input_vec
%dot = OpDot %float %loaded %loaded
OpStore %out_value %dot
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_INTEGER_DOT_PRODUCT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.int_dot.frag.out
; Reduced from SPV_KHR_integer_dot_product SDotKHR vector operations.
OpCapability Shader
OpCapability DotProductKHR
OpExtension "SPV_KHR_integer_dot_product"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_a %input_b %out_value
OpExecutionMode %main OriginUpperLeft
OpName %input_a "inputA"
OpName %input_b "inputB"
OpName %out_value "outValue"
OpDecorate %input_a Location 0
OpDecorate %input_b Location 1
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%v4int = OpTypeVector %int 4
%ptr_input_v4int = OpTypePointer Input %v4int
%ptr_output_int = OpTypePointer Output %int
%input_a = OpVariable %ptr_input_v4int Input
%input_b = OpVariable %ptr_input_v4int Input
%out_value = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_a = OpLoad %v4int %input_a
%loaded_b = OpLoad %v4int %input_b
%dot = OpSDotKHR %int %loaded_a %loaded_b
OpStore %out_value %dot
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_VECTOR_EXTRACT_DYNAMIC_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Reduced from the core OpVectorExtractDynamic instruction definition.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_vec %index %out_value
OpExecutionMode %main OriginUpperLeft
OpName %input_vec "inputVec"
OpName %index "index"
OpName %out_value "outValue"
OpDecorate %input_vec Location 0
OpDecorate %index Flat
OpDecorate %index Location 1
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_input_uint = OpTypePointer Input %uint
%ptr_output_float = OpTypePointer Output %float
%input_vec = OpVariable %ptr_input_v4float Input
%index = OpVariable %ptr_input_uint Input
%out_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_vec = OpLoad %v4float %input_vec
%loaded_index = OpLoad %uint %index
%component = OpVectorExtractDynamic %float %loaded_vec %loaded_index
OpStore %out_value %component
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_VECTOR_NEGATE_EXTRACT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/opt/fold_test.cpp
; Reduced from vector OpFNegate and OpCompositeExtract folding patterns.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_vec %out_value
OpExecutionMode %main OriginUpperLeft
OpName %input_vec "inputVec"
OpName %out_value "outValue"
OpDecorate %input_vec Location 0
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_float = OpTypePointer Output %float
%input_vec = OpVariable %ptr_input_v4float Input
%out_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %input_vec
%negated = OpFNegate %v4float %loaded
%component = OpCompositeExtract %float %negated 0
OpStore %out_value %component
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_STRUCT_COMPOSITE_EXTRACT_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.core.grammar.json
; Reduced from the core OpCompositeExtract instruction definition where Composite
; is an OpTypeStruct value and Indexes select a structure member.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_value
OpExecutionMode %main OriginUpperLeft
OpName %Pair "Pair"
OpMemberName %Pair 0 "weight"
OpMemberName %Pair 1 "index"
OpName %pair "pair"
OpName %out_value "outValue"
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%Pair = OpTypeStruct %float %uint
%ptr_function_pair = OpTypePointer Function %Pair
%ptr_output_float = OpTypePointer Output %float
%out_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%pair = OpVariable %ptr_function_pair Function
%loaded_pair = OpLoad %Pair %pair
%weight = OpCompositeExtract %float %loaded_pair 0
OpStore %out_value %weight
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_ARRAY_RETURN_COMPOSITE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/binary_to_text_test.cpp IndentTest.ReorderedNested
; Reduced from helper function returning OpTypeArray via OpCompositeConstruct.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_value
OpExecutionMode %main OriginUpperLeft
OpName %make_array "ff(vf2;f1;"
OpName %out_value "outValue"
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn_void = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%two = OpConstant %uint 2
%zero = OpConstant %float 0.0
%one = OpConstant %float 1.0
%array_v4 = OpTypeArray %v4float %two
%fn_array = OpTypeFunction %array_v4
%ptr_output_float = OpTypePointer Output %float
%out_value = OpVariable %ptr_output_float Output
%make_array = OpFunction %array_v4 None %fn_array
%helper_label = OpLabel
%first = OpCompositeConstruct %v4float %zero %zero %zero %zero
%second = OpCompositeConstruct %v4float %one %one %one %one
%array_value = OpCompositeConstruct %array_v4 %first %second
OpReturnValue %array_value
OpFunctionEnd
%main = OpFunction %void None %fn_void
%main_label = OpLabel
%call = OpFunctionCall %array_v4 %make_array
%component = OpCompositeExtract %float %call 1 2
OpStore %out_value %component
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_VECTOR_INSERT_DYNAMIC_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Reduced from the core OpVectorInsertDynamic instruction definition.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_vec %insert_value %index %out_vec
OpExecutionMode %main OriginUpperLeft
OpName %input_vec "inputVec"
OpName %insert_value "insertValue"
OpName %index "index"
OpName %out_vec "outVec"
OpDecorate %input_vec Location 0
OpDecorate %insert_value Location 1
OpDecorate %index Flat
OpDecorate %index Location 2
OpDecorate %out_vec Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_input_float = OpTypePointer Input %float
%ptr_input_uint = OpTypePointer Input %uint
%ptr_output_v4float = OpTypePointer Output %v4float
%input_vec = OpVariable %ptr_input_v4float Input
%insert_value = OpVariable %ptr_input_float Input
%index = OpVariable %ptr_input_uint Input
%out_vec = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_vec = OpLoad %v4float %input_vec
%loaded_value = OpLoad %float %insert_value
%loaded_index = OpLoad %uint %index
%inserted = OpVectorInsertDynamic %v4float %loaded_vec %loaded_value %loaded_index
OpStore %out_vec %inserted
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_UNDEF_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Reduced from the core OpUndef instruction definition.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_vec
OpExecutionMode %main OriginUpperLeft
OpName %undef "undefValue"
OpName %body_undef "bodyUndef"
OpName %out_vec "outVec"
OpDecorate %out_vec Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%undef = OpUndef %v4float
%out_vec = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%body_undef = OpUndef %v4float
OpStore %out_vec %undef
OpStore %out_vec %body_undef
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_CONSTANT_NULL_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Reduced from the core OpConstantNull instruction definition.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_vec
OpExecutionMode %main OriginUpperLeft
OpName %null_vec "nullVec"
OpName %body_null "bodyNull"
OpName %out_vec "outVec"
OpDecorate %out_vec Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%null_vec = OpConstantNull %v4float
%out_vec = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%body_null = OpConstantNull %v4float
OpStore %out_vec %null_vec
OpStore %out_vec %body_null
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_FRAGMENT_TERMINATION_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Reduced from OpKill, OpTerminateInvocation, and OpDemoteToHelperInvocation.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%label = OpLabel
OpKill
OpTerminateInvocation
OpDemoteToHelperInvocation
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_VOID_FUNCTION_CALL_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Source tool: glslangValidator -V -H, reduced from a fragment shader where
; main calls void helper() before storing an output color.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %helper "helper("
OpName %out_color "outColor"
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%one = OpConstant %float 1.0
%white = OpConstantComposite %v4float %one %one %one %one
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%main_label = OpLabel
%call = OpFunctionCall %void %helper
OpStore %out_color %white
OpReturn
OpFunctionEnd
%helper = OpFunction %void None %fn
%helper_label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_POINTER_PARAMETER_FUNCTION_CALL_ASSEMBLY = """
; Source tool: glslangValidator -V -H, reduced from a helper taking an inout
; vec4 parameter backed by a Function storage-class pointer.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %helper "helper("
OpName %value "value"
OpName %out_color "outColor"
OpDecorate %out_color Location 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_function_v4float = OpTypePointer Function %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%helper_fn = OpTypeFunction %void %ptr_function_v4float
%main_fn = OpTypeFunction %void
%zero = OpConstant %float 0.0
%one = OpConstant %float 1.0
%red = OpConstantComposite %v4float %one %zero %zero %one
%out_color = OpVariable %ptr_output_v4float Output
%helper = OpFunction %void None %helper_fn
%value = OpFunctionParameter %ptr_function_v4float
%helper_label = OpLabel
%loaded = OpLoad %v4float %value
%negated = OpFNegate %v4float %loaded
OpStore %value %negated
OpReturn
OpFunctionEnd
%main = OpFunction %void None %main_fn
%main_label = OpLabel
%local = OpVariable %ptr_function_v4float Function %red
%call = OpFunctionCall %void %helper %local
%local_value = OpLoad %v4float %local
OpStore %out_color %local_value
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_SELECTION_MERGE_ASSEMBLY = """
; Source tool: glslangValidator -V -H, reduced from a fragment shader
; containing if (value > 0.0) { outColor = red; } else { outColor = blue; }.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %value %out_color
OpExecutionMode %main OriginUpperLeft
OpName %value "value"
OpName %out_color "outColor"
OpDecorate %value Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%ptr_input_float = OpTypePointer Input %float
%ptr_output_v4float = OpTypePointer Output %v4float
%zero = OpConstant %float 0.0
%one = OpConstant %float 1.0
%red = OpConstantComposite %v4float %one %zero %zero %one
%blue = OpConstantComposite %v4float %zero %zero %one %one
%value = OpVariable %ptr_input_float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %value
%gt = OpFOrdGreaterThan %bool %loaded %zero
OpSelectionMerge %merge None
OpBranchConditional %gt %then %else
%then = OpLabel
OpStore %out_color %red
OpBranch %merge
%else = OpLabel
OpStore %out_color %blue
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_SWITCH_ASSEMBLY = """
; Source tool: glslangValidator -V -H, reduced from a fragment shader
; containing switch(mode) with two cases and a default color store.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %mode %out_color
OpExecutionMode %main OriginUpperLeft
OpName %mode "mode"
OpName %out_color "outColor"
OpDecorate %mode Flat
OpDecorate %mode Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v4float = OpTypeVector %float 4
%ptr_input_int = OpTypePointer Input %int
%ptr_output_v4float = OpTypePointer Output %v4float
%zero_f = OpConstant %float 0.0
%one_f = OpConstant %float 1.0
%red = OpConstantComposite %v4float %one_f %zero_f %zero_f %one_f
%green = OpConstantComposite %v4float %zero_f %one_f %zero_f %one_f
%blue = OpConstantComposite %v4float %zero_f %zero_f %one_f %one_f
%mode = OpVariable %ptr_input_int Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_mode = OpLoad %int %mode
OpSelectionMerge %merge None
OpSwitch %loaded_mode %default 0 %case_zero 1 %case_one
%case_zero = OpLabel
OpStore %out_color %red
OpBranch %merge
%case_one = OpLabel
OpStore %out_color %green
OpBranch %merge
%default = OpLabel
OpStore %out_color %blue
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_GROUPED_SWITCH_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://registry.khronos.org/SPIR-V/specs/unified1/MachineReadableGrammar.html
; Source tool: glslangValidator -V then spirv-dis, reduced from a fragment
; shader containing case 0: case 1: in the same switch body.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %mode %out_color
OpExecutionMode %main OriginUpperLeft
OpName %mode "mode"
OpName %out_color "outColor"
OpDecorate %mode Flat
OpDecorate %mode Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v4float = OpTypeVector %float 4
%ptr_input_int = OpTypePointer Input %int
%ptr_output_v4float = OpTypePointer Output %v4float
%zero_f = OpConstant %float 0.0
%one_f = OpConstant %float 1.0
%red = OpConstantComposite %v4float %one_f %zero_f %zero_f %one_f
%blue = OpConstantComposite %v4float %zero_f %zero_f %one_f %one_f
%mode = OpVariable %ptr_input_int Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_mode = OpLoad %int %mode
OpSelectionMerge %merge None
OpSwitch %loaded_mode %default 0 %case_grouped 1 %case_grouped
%default = OpLabel
OpStore %out_color %blue
OpBranch %merge
%case_grouped = OpLabel
OpStore %out_color %red
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_SIMPLE_LOOP_MERGE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/opt/loop_optimizations/unroll_simple.cpp
; Source test: DoNotDuplicateDecorationsOnLoopCarriedValue.
; Reduced from the OpLoopMerge header -> condition -> body -> latch shape,
; adapted to a function-local counter so no phi reconstruction is required.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_value
OpExecutionMode %main OriginUpperLeft
OpName %out_value "outValue"
OpName %counter "counter"
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%bool = OpTypeBool
%ptr_function_int = OpTypePointer Function %int
%ptr_output_int = OpTypePointer Output %int
%zero = OpConstant %int 0
%one = OpConstant %int 1
%four = OpConstant %int 4
%out_value = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%entry = OpLabel
%counter = OpVariable %ptr_function_int Function %zero
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %latch None
OpBranch %condition
%condition = OpLabel
%i = OpLoad %int %counter
%lt = OpSLessThan %bool %i %four
OpBranchConditional %lt %body %merge
%body = OpLabel
OpStore %out_value %i
OpBranch %latch
%latch = OpLabel
%current = OpLoad %int %counter
%next = OpIAdd %int %current %one
OpStore %counter %next
OpBranch %header
%merge = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_OPSELECT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.1.4.OpSelect.frag.out
; Reduced from Test/spv.1.4.OpSelect.frag OpSelect scalar/vector/matrix/struct cases.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outv %cond %in1 %in2
OpExecutionMode %main OriginUpperLeft
OpName %outv "outv"
OpName %cond "cond"
OpName %in1 "in1"
OpName %in2 "in2"
OpName %S1 "S1"
OpMemberName %S1 0 "a"
OpMemberName %S1 1 "b"
OpName %fv "fv"
OpDecorate %outv Location 0
OpDecorate %cond Flat
OpDecorate %cond Location 4
OpDecorate %in1 Flat
OpDecorate %in1 Location 0
OpDecorate %in2 Flat
OpDecorate %in2 Location 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%bool = OpTypeBool
%ptr_output_float = OpTypePointer Output %float
%ptr_input_int = OpTypePointer Input %int
%ptr_function_float = OpTypePointer Function %float
%v4int = OpTypeVector %int 4
%ptr_function_v4int = OpTypePointer Function %v4int
%v3float = OpTypeVector %float 3
%mat3 = OpTypeMatrix %v3float 3
%ptr_function_mat3 = OpTypePointer Function %mat3
%S1 = OpTypeStruct %float %int
%ptr_function_S1 = OpTypePointer Function %S1
%ptr_input_S1 = OpTypePointer Input %S1
%one_f = OpConstant %float 1.0
%two_f = OpConstant %float 2.0
%zero_i = OpConstant %int 0
%one_i = OpConstant %int 1
%two_i = OpConstant %int 2
%eight_i = OpConstant %int 8
%twenty_i = OpConstant %int 20
%five_i = OpConstant %int 5
%outv = OpVariable %ptr_output_float Output
%cond = OpVariable %ptr_input_int Input
%in1 = OpVariable %ptr_input_S1 Input
%in2 = OpVariable %ptr_input_S1 Input
%main = OpFunction %void None %fn
%label = OpLabel
%iv1 = OpVariable %ptr_function_v4int Function
%iv2 = OpVariable %ptr_function_v4int Function
%m1 = OpVariable %ptr_function_mat3 Function
%m2 = OpVariable %ptr_function_mat3 Function
%fv = OpVariable %ptr_function_S1 Function
%loaded_cond = OpLoad %int %cond
%lt = OpSLessThan %bool %loaded_cond %eight_i
%selected_float = OpSelect %float %lt %one_f %two_f
OpStore %outv %selected_float
%iv1_value = OpCompositeConstruct %v4int %one_i %one_i %one_i %one_i
OpStore %iv1 %iv1_value
%iv2_value = OpCompositeConstruct %v4int %two_i %two_i %two_i %two_i
OpStore %iv2 %iv2_value
%gt_zero = OpSGreaterThan %bool %loaded_cond %zero_i
%iv1_loaded = OpLoad %v4int %iv1
%iv2_loaded = OpLoad %v4int %iv2
%selected_vec = OpSelect %v4int %gt_zero %iv1_loaded %iv2_loaded
%selected_component = OpCompositeExtract %int %selected_vec 2
%component_f = OpConvertSToF %float %selected_component
%current_outv = OpLoad %float %outv
%vec_product = OpFMul %float %current_outv %component_f
OpStore %outv %vec_product
%col0 = OpConstantComposite %v3float %one_f %one_f %one_f
%col1 = OpConstantComposite %v3float %two_f %two_f %two_f
%m1_value = OpConstantComposite %mat3 %col0 %col0 %col0
%m2_value = OpConstantComposite %mat3 %col1 %col1 %col1
OpStore %m1 %m1_value
OpStore %m2 %m2_value
%lt_twenty = OpSLessThan %bool %loaded_cond %twenty_i
%m1_loaded = OpLoad %mat3 %m1
%m2_loaded = OpLoad %mat3 %m2
%selected_mat = OpSelect %mat3 %lt_twenty %m1_loaded %m2_loaded
%matrix_component = OpCompositeExtract %float %selected_mat 2 1
%after_vec = OpLoad %float %outv
%matrix_product = OpFMul %float %after_vec %matrix_component
OpStore %outv %matrix_product
%gt_five = OpSGreaterThan %bool %loaded_cond %five_i
%in1_loaded = OpLoad %S1 %in1
%in2_loaded = OpLoad %S1 %in2
%selected_struct = OpSelect %S1 %gt_five %in1_loaded %in2_loaded
OpStore %fv %selected_struct
%field_a = OpAccessChain %ptr_function_float %fv %zero_i
%a_value = OpLoad %float %field_a
%after_mat = OpLoad %float %outv
%struct_product = OpFMul %float %after_mat %a_value
OpStore %outv %struct_product
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_ANY_ALL_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/web.operations.frag.out
; Reduced from OpAny/OpAll boolean-vector reductions in the fragment main body.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %value %any_out %all_out
OpExecutionMode %main OriginUpperLeft
OpName %value "value"
OpName %any_out "anyOut"
OpName %all_out "allOut"
OpName %has_any "hasAny"
OpName %has_all "hasAll"
OpDecorate %value Location 0
OpDecorate %any_out Location 0
OpDecorate %all_out Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%v4bool = OpTypeVector %bool 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_bool = OpTypePointer Output %bool
%value = OpVariable %ptr_input_v4float Input
%any_out = OpVariable %ptr_output_bool Output
%all_out = OpVariable %ptr_output_bool Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %value
%neq = OpFUnordNotEqual %v4bool %loaded %loaded
%has_any = OpAny %bool %neq
OpStore %any_out %has_any
%eq = OpFOrdEqual %v4bool %loaded %loaded
%has_all = OpAll %bool %eq
OpStore %all_out %has_all
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_LOCAL_ARRAY_VARIABLE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/web.operations.frag.out
; Reduced from function-local OpTypeArray variables in the fragment main body.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_value
OpExecutionMode %main OriginUpperLeft
OpName %main "main"
OpName %out_value "outValue"
OpName %arr "arr"
OpDecorate %out_value Location 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%arr_int_2 = OpTypeArray %int %uint_2
%ptr_function_arr_int_2 = OpTypePointer Function %arr_int_2
%ptr_output_float = OpTypePointer Output %float
%fn = OpTypeFunction %void
%out_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%arr = OpVariable %ptr_function_arr_int_2 Function
OpReturn
OpFunctionEnd
"""

SPIRV_PUSH_CONSTANT_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpName %PushConstants "PushConstants"
OpName %pc "pc"
OpMemberName %PushConstants 0 "model"
OpMemberName %PushConstants 1 "tint"
OpDecorate %PushConstants Block
OpMemberDecorate %PushConstants 0 Offset 0
OpMemberDecorate %PushConstants 1 Offset 64
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%mat4 = OpTypeMatrix %v4float 4
%PushConstants = OpTypeStruct %mat4 %v4float
%ptr_pc = OpTypePointer PushConstant %PushConstants
%void = OpTypeVoid
%fn = OpTypeFunction %void
%pc = OpVariable %ptr_pc PushConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_UNIFORM_BLOCK_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpName %Camera "Camera"
OpName %camera "camera"
OpMemberName %Camera 0 "viewProj"
OpMemberName %Camera 1 "tint"
OpDecorate %Camera Block
OpDecorate %camera DescriptorSet 0
OpDecorate %camera Binding 2
OpMemberDecorate %Camera 0 Offset 0
OpMemberDecorate %Camera 1 Offset 64
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%mat4 = OpTypeMatrix %v4float 4
%Camera = OpTypeStruct %mat4 %v4float
%ptr_camera = OpTypePointer Uniform %Camera
%void = OpTypeVoid
%fn = OpTypeFunction %void
%camera = OpVariable %ptr_camera Uniform
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_BUFFER_BLOCK_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %Data "Data"
OpName %data "data"
OpMemberName %Data 0 "value"
OpDecorate %Data BufferBlock
OpDecorate %data DescriptorSet 0
OpDecorate %data Binding 1
OpMemberDecorate %Data 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%Data = OpTypeStruct %v4float
%ptr_data = OpTypePointer Uniform %Data
%data = OpVariable %ptr_data Uniform
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_LEGACY_BUFFER_BLOCK_MEMBER_ACCESS_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %bName "bName"
OpMemberName %bName 0 "size"
OpName %bInst "bInst"
OpDecorate %bName BufferBlock
OpMemberDecorate %bName 0 Offset 0
OpDecorate %bInst DescriptorSet 0
OpDecorate %bInst Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%bName = OpTypeStruct %int
%_ptr_Uniform_bName = OpTypePointer Uniform %bName
%bInst = OpVariable %_ptr_Uniform_bName Uniform
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%_ptr_Uniform_int = OpTypePointer Uniform %int
%main = OpFunction %void None %fn
%label = OpLabel
%size_ptr = OpAccessChain %_ptr_Uniform_int %bInst %int_0
%size = OpLoad %int %size_ptr
%next = OpIAdd %int %size %int_1
OpStore %size_ptr %next
OpReturn
OpFunctionEnd
"""

SPIRV_READONLY_BUFFER_BLOCK_ASSEMBLY = """
; Reduced from readonly storage-buffer SPIR-V decoration patterns.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %Data "Data"
OpName %data "data"
OpMemberName %Data 0 "value"
OpDecorate %Data BufferBlock
OpDecorate %data DescriptorSet 0
OpDecorate %data Binding 1
OpDecorate %data NonWritable
OpMemberDecorate %Data 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%Data = OpTypeStruct %v4float
%ptr_data = OpTypePointer Uniform %Data
%data = OpVariable %ptr_data Uniform
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_RUNTIME_ARRAY_BUFFER_BLOCK_ASSEMBLY = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %StorageBuffer "StorageBuffer"
OpName %storage "storage"
OpMemberName %StorageBuffer 0 "header"
OpMemberName %StorageBuffer 1 "payload"
OpDecorate %StorageBuffer BufferBlock
OpDecorate %storage DescriptorSet 0
OpDecorate %storage Binding 4
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%runtimearr_uint = OpTypeRuntimeArray %uint
%StorageBuffer = OpTypeStruct %uint %runtimearr_uint
%ptr_storage = OpTypePointer Uniform %StorageBuffer
%storage = OpVariable %ptr_storage Uniform
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_ANONYMOUS_RESOURCE_BLOCK_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d
; Source path: test/diff/diff_files/small_functions_small_diffs_src.spvasm
; Reduced from descriptor block variables named with empty OpName strings:
; OpName %19 "" and OpName %24 "".
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %4 "main" %30
OpExecutionMode %4 LocalSize 1 1 1
OpName %17 "BufferOut"
OpMemberName %17 0 "o"
OpName %19 ""
OpName %22 "BufferIn"
OpMemberName %22 0 "i"
OpName %24 ""
OpName %30 "result"
OpMemberDecorate %17 0 Offset 0
OpDecorate %17 BufferBlock
OpDecorate %19 DescriptorSet 0
OpDecorate %19 Binding 1
OpMemberDecorate %22 0 Offset 0
OpDecorate %22 Block
OpDecorate %24 DescriptorSet 0
OpDecorate %24 Binding 0
OpDecorate %30 Location 0
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%16 = OpTypeInt 32 0
%17 = OpTypeStruct %16
%18 = OpTypePointer Uniform %17
%19 = OpVariable %18 Uniform
%20 = OpTypeInt 32 1
%21 = OpConstant %20 0
%22 = OpTypeStruct %16
%23 = OpTypePointer Uniform %22
%24 = OpVariable %23 Uniform
%25 = OpTypePointer Uniform %16
%29 = OpTypePointer Output %16
%30 = OpVariable %29 Output
%4 = OpFunction %2 None %3
%5 = OpLabel
%26 = OpAccessChain %25 %24 %21
%27 = OpLoad %16 %26
%28 = OpAccessChain %25 %19 %21
OpStore %28 %27
OpStore %30 %27
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_ARRAY_LENGTH_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source version: SPIR-V 1.6 Revision 7, unified spec.
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: df032578c737d361b754fc569b70aa29b5f8c7d4
; Source path: test/val/val_memory_test.cpp
; Reduced from ValidateMemory::ArrayLenIndexCorrectWith2Members and adapted to
; a storage-buffer fragment fixture so generated CrossGL can store the result.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %count_out
OpExecutionMode %main OriginUpperLeft
OpName %StorageBuffer "StorageBuffer"
OpName %storage "storage"
OpName %count_out "countOut"
OpMemberName %StorageBuffer 0 "header"
OpMemberName %StorageBuffer 1 "payload"
OpDecorate %runtimearr_uint ArrayStride 4
OpDecorate %StorageBuffer Block
OpDecorate %storage DescriptorSet 0
OpDecorate %storage Binding 0
OpDecorate %count_out Location 0
OpMemberDecorate %StorageBuffer 0 Offset 0
OpMemberDecorate %StorageBuffer 1 Offset 4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%runtimearr_uint = OpTypeRuntimeArray %uint
%StorageBuffer = OpTypeStruct %uint %runtimearr_uint
%ptr_storage = OpTypePointer StorageBuffer %StorageBuffer
%ptr_output_uint = OpTypePointer Output %uint
%storage = OpVariable %ptr_storage StorageBuffer
%count_out = OpVariable %ptr_output_uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%length = OpArrayLength %uint %storage 1
OpStore %count_out %length
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_CONSTANT_ASSEMBLY = """
; Reduced from common Vulkan specialization constant assembly output.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpName %max_lights "MAX_LIGHTS"
OpName %enable_shadows "ENABLE_SHADOWS"
OpDecorate %max_lights SpecId 0
OpDecorate %enable_shadows SpecId 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%max_lights = OpSpecConstant %uint 4
%enable_shadows = OpSpecConstantTrue %bool
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_NUMERIC_ID_SPEC_CONSTANT_ASSEMBLY = """
; Reduced from Vulkan-Samples compute_nbody/glsl/particle_calculate.comp.spv.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpDecorate %104 SpecId 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%104 = OpSpecConstant %uint 1
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_CONSTANT_COMPOSITE_OP_ASSEMBLY = """
; Reduced from Vulkan compute shaders with specialization-driven local size.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %Block "SizedBlock"
OpMemberName %Block 0 "values"
OpName %width "WORKGROUP_WIDTH"
OpName %height "WORKGROUP_HEIGHT"
OpName %total "WORKGROUP_TOTAL"
OpDecorate %width SpecId 0
OpDecorate %height SpecId 1
OpDecorate %size BuiltIn WorkgroupSize
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%arr = OpTypeArray %uint %total
%Block = OpTypeStruct %arr
%width = OpSpecConstant %uint 8
%height = OpSpecConstant %uint 4
%one = OpConstant %uint 1
%total = OpSpecConstantOp %uint IAdd %width %height
%size = OpSpecConstantComposite %v3uint %width %height %one
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_WEB_COMP_BARRIER_ASSEMBLY = """
; Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Test/baseResults/web.comp.out OpControlBarrier/OpMemoryBarrier.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %outColor
OpExecutionMode %main LocalSize 2 5 7
OpName %outColor "outColor"
OpDecorate %outColor Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_264 = OpConstant %uint 264
%uint_3400 = OpConstant %uint 3400
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_out_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %ptr_out_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpControlBarrier %uint_2 %uint_2 %uint_264
OpMemoryBarrier %uint_1 %uint_3400
OpMemoryBarrier %uint_2 %uint_3400
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_NAMED_BARRIER_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/val/val_barriers_test.cpp
; Reduced from OpNamedBarrierInitializeSuccess and OpMemoryNamedBarrierSuccess.
OpCapability Shader
OpCapability NamedBarrier
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out_value
OpExecutionMode %main LocalSize 1 1 1
OpName %out_value "outValue"
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%named_barrier = OpTypeNamedBarrier
%ptr_output_uint = OpTypePointer Output %uint
%u32_4 = OpConstant %uint 4
%workgroup = OpConstant %uint 2
%acquire_release_workgroup = OpConstant %uint 264
%out_value = OpVariable %ptr_output_uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%barrier = OpNamedBarrierInitialize %named_barrier %u32_4
OpMemoryNamedBarrier %barrier %workgroup %acquire_release_workgroup
OpStore %out_value %u32_4
OpReturn
OpFunctionEnd
"""

SPIRV_SUBGROUP_BROADCAST_REDUCE_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source pattern: reduced from Vulkan subgroupBroadcastFirst/subgroupAdd style
; compute shader assembly emitted by glslang for subgroup arithmetic builtins.
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformArithmetic
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %value %broadcast_out %sum_out
OpExecutionMode %main LocalSize 32 1 1
OpName %value "value"
OpName %broadcast_out "broadcastOut"
OpName %sum_out "sumOut"
OpDecorate %value Location 0
OpDecorate %broadcast_out Location 0
OpDecorate %sum_out Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%ptr_input_uint = OpTypePointer Input %uint
%ptr_output_uint = OpTypePointer Output %uint
%value = OpVariable %ptr_input_uint Input
%broadcast_out = OpVariable %ptr_output_uint Output
%sum_out = OpVariable %ptr_output_uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %uint %value
%broadcast = OpGroupNonUniformBroadcastFirst %uint %uint_3 %loaded
%sum = OpGroupNonUniformIAdd %uint %uint_3 Reduce %loaded
OpStore %broadcast_out %broadcast
OpStore %sum_out %sum
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_SUBGROUP_BROADCAST_ASSEMBLY = """
; Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Test/glsl.450.subgroup.frag ballot_works subgroupBroadcast(f4, 0).
; Lowering source: SPIRV/GlslangToSpv.cpp maps EOpSubgroupBroadcast to
; OpGroupNonUniformBroadcast.
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformBallot
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_color %out_color
OpExecutionMode %main OriginUpperLeft
OpName %in_color "inColor"
OpName %out_color "outColor"
OpDecorate %in_color Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%scope_subgroup = OpConstant %uint 3
%lane_zero = OpConstant %uint 0
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%in_color = OpVariable %ptr_input_v4float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %in_color
%broadcast = OpGroupNonUniformBroadcast %v4float %scope_subgroup %loaded %lane_zero
OpStore %out_color %broadcast
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_ATOMIC_LOAD_STORE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source paths: Test/baseResults/spv.atomic.comp.out and
; Test/baseResults/spv.atomicStoreInt64.comp.out.
; Reduced from OpAtomicLoad/OpAtomicStore patterns and adapted to a storage
; buffer member so generated CrossGL exposes both lowered operations.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %old_value
OpExecutionMode %main OriginUpperLeft
OpName %CounterBlock "CounterBlock"
OpMemberName %CounterBlock 0 "counter"
OpName %counter_block "counterBlock"
OpName %old_value "oldValue"
OpDecorate %CounterBlock Block
OpDecorate %counter_block DescriptorSet 0
OpDecorate %counter_block Binding 0
OpDecorate %old_value Location 0
OpMemberDecorate %CounterBlock 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%CounterBlock = OpTypeStruct %int
%ptr_storage_counter_block = OpTypePointer StorageBuffer %CounterBlock
%ptr_storage_int = OpTypePointer StorageBuffer %int
%ptr_output_int = OpTypePointer Output %int
%counter_block = OpVariable %ptr_storage_counter_block StorageBuffer
%old_value = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%label = OpLabel
%counter_ptr = OpAccessChain %ptr_storage_int %counter_block %int_0
%old = OpAtomicLoad %int %counter_ptr %uint_1 %uint_0
OpStore %old_value %old
OpAtomicStore %counter_ptr %uint_1 %uint_0 %int_1
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_WEB_COMP_ATOMIC_IADD_ASSEMBLY = """
; Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Test/baseResults/web.comp.out OpAtomicIAdd on a buffer-block member.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %old_value
OpExecutionMode %main OriginUpperLeft
OpName %CounterBlock "CounterBlock"
OpMemberName %CounterBlock 0 "counter"
OpName %counter_block "counterBlock"
OpName %old_value "oldValue"
OpDecorate %CounterBlock Block
OpDecorate %counter_block DescriptorSet 0
OpDecorate %counter_block Binding 0
OpDecorate %old_value Location 0
OpMemberDecorate %CounterBlock 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%CounterBlock = OpTypeStruct %int
%ptr_storage_counter_block = OpTypePointer StorageBuffer %CounterBlock
%ptr_storage_int = OpTypePointer StorageBuffer %int
%ptr_output_int = OpTypePointer Output %int
%counter_block = OpVariable %ptr_storage_counter_block StorageBuffer
%old_value = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%label = OpLabel
%counter_ptr = OpAccessChain %ptr_storage_int %counter_block %int_0
%old = OpAtomicIAdd %int %counter_ptr %uint_1 %uint_0 %int_2
OpStore %old_value %old
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_IMAGE_TEXEL_POINTER_ATOMIC_IADD_ASSEMBLY = """
; Reduced from glslangValidator 16.3.0 -V -H output for:
; layout(r32ui, set = 0, binding = 0) uniform uimage2D img;
; imageAtomicAdd(img, ivec2(0, 1), 3u);
; SPIRV-Headers unified1 grammar declares OpImageTexelPointer as
; Result Type, Result, Image, Coordinate, Sample.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %img "img"
OpName %old "old"
OpDecorate %img Binding 0
OpDecorate %img DescriptorSet 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%image = OpTypeImage %uint 2D 0 0 0 2 R32ui
%ptr_image_uniform = OpTypePointer UniformConstant %image
%ptr_image_uint = OpTypePointer Image %uint
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_3 = OpConstant %uint 3
%coord = OpConstantComposite %v2int %int_0 %int_1
%img = OpVariable %ptr_image_uniform UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
%texel_ptr = OpImageTexelPointer %ptr_image_uint %img %coord %uint_0
%old = OpAtomicIAdd %uint %texel_ptr %uint_1 %uint_0 %uint_3
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_WEB_COMP_ATOMIC_COMPARE_EXCHANGE_ASSEMBLY = """
; Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Test/baseResults/web.comp.out OpAtomicCompareExchange on a buffer-block member.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %old_value
OpExecutionMode %main OriginUpperLeft
OpName %CounterBlock "CounterBlock"
OpMemberName %CounterBlock 0 "counter"
OpName %counter_block "counterBlock"
OpName %old_value "oldValue"
OpDecorate %CounterBlock Block
OpDecorate %counter_block DescriptorSet 0
OpDecorate %counter_block Binding 0
OpDecorate %old_value Location 0
OpMemberDecorate %CounterBlock 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%int_5 = OpConstant %int 5
%CounterBlock = OpTypeStruct %int
%ptr_storage_counter_block = OpTypePointer StorageBuffer %CounterBlock
%ptr_storage_int = OpTypePointer StorageBuffer %int
%ptr_output_int = OpTypePointer Output %int
%counter_block = OpVariable %ptr_storage_counter_block StorageBuffer
%old_value = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%label = OpLabel
%counter_ptr = OpAccessChain %ptr_storage_int %counter_block %int_0
%old = OpAtomicCompareExchange %int %counter_ptr %uint_1 %uint_0 %uint_0 %int_2 %int_5
OpStore %old_value %old
OpReturn
OpFunctionEnd
"""

SPIRV_ATOMIC_INC_DEC_SUB_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source paths: Test/baseResults/spv.atomic.comp.out and
; Test/baseResults/spv.460.frag.out.
; Reduced from OpAtomicIIncrement, OpAtomicIDecrement, and OpAtomicISub.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %old_value
OpExecutionMode %main OriginUpperLeft
OpName %CounterBlock "CounterBlock"
OpMemberName %CounterBlock 0 "counter"
OpName %counter_block "counterBlock"
OpName %old_value "oldValue"
OpDecorate %CounterBlock Block
OpDecorate %counter_block DescriptorSet 0
OpDecorate %counter_block Binding 0
OpDecorate %old_value Location 0
OpMemberDecorate %CounterBlock 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%CounterBlock = OpTypeStruct %int
%ptr_storage_counter_block = OpTypePointer StorageBuffer %CounterBlock
%ptr_storage_int = OpTypePointer StorageBuffer %int
%ptr_output_int = OpTypePointer Output %int
%counter_block = OpVariable %ptr_storage_counter_block StorageBuffer
%old_value = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%label = OpLabel
%counter_ptr = OpAccessChain %ptr_storage_int %counter_block %int_0
%old_inc = OpAtomicIIncrement %int %counter_ptr %uint_1 %uint_0
OpStore %old_value %old_inc
%old_dec = OpAtomicIDecrement %int %counter_ptr %uint_1 %uint_0
OpStore %old_value %old_dec
%old_sub = OpAtomicISub %int %counter_ptr %uint_1 %uint_0 %int_2
OpStore %old_value %old_sub
OpReturn
OpFunctionEnd
"""

SPIRV_UNIFORM_CONSTANT_RESOURCE_ASSEMBLY = """
; Reduced from combined image/sampler SPIR-V assembly emitted by Vulkan toolchains.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpName %combined "combinedTex"
OpName %linear_sampler "linearSampler"
OpDecorate %combined DescriptorSet 0
OpDecorate %combined Binding 0
OpDecorate %linear_sampler DescriptorSet 0
OpDecorate %linear_sampler Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %img
%sampler = OpTypeSampler
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_sampler = OpTypePointer UniformConstant %sampler
%combined = OpVariable %ptr_sampled UniformConstant
%linear_sampler = OpVariable %ptr_sampler UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_VULKAN_SUBPASS_INPUT_ATTACHMENT_ASSEMBLY = """
; Source: Vulkan GLSL/SPIR-V mappings for subpass inputs.
; layout(input_attachment_index=i, set=m, binding=n) uniform subpassInput
; maps to DescriptorSet, Binding, and InputAttachmentIndex decorations.
OpCapability Shader
OpCapability InputAttachment
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpName %scene_input "sceneInput"
OpDecorate %scene_input DescriptorSet 1
OpDecorate %scene_input Binding 2
OpDecorate %scene_input InputAttachmentIndex 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%image = OpTypeImage %float SubpassData 0 0 0 2 Unknown
%ptr_input_attachment = OpTypePointer UniformConstant %image
%scene_input = OpVariable %ptr_input_attachment UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_RAY_TRACING_ACCELERATION_STRUCTURE_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source extension: https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_tracing.html
; Reduced from the OpTypeAccelerationStructureKHR opaque type declaration and
; a descriptor-backed UniformConstant OpVariable.
OpCapability RayTracingKHR
OpExtension "SPV_KHR_ray_tracing"
OpMemoryModel Logical GLSL450
OpEntryPoint RayGenerationKHR %main "main"
OpName %top_level_as "topLevelAS"
OpDecorate %top_level_as DescriptorSet 0
OpDecorate %top_level_as Binding 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%accel = OpTypeAccelerationStructureKHR
%ptr_accel = OpTypePointer UniformConstant %accel
%top_level_as = OpVariable %ptr_accel UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_RAY_QUERY_CONVERT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source paths: Test/rayQuery-OpConvertUToAccelerationStructureKHR.comp and
; Test/baseResults/rayQuery-OpConvertUToAccelerationStructureKHR.comp.out
; Reduced from rayQueryInitializeEXT(rayQuery, accelerationStructureEXT(tlas), ...).
OpCapability Shader
OpCapability RayQueryKHR
OpExtension "SPV_KHR_ray_query"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %main "main"
OpName %ray_query "rayQuery"
OpName %Params "Params"
OpMemberName %Params 0 "tlas"
OpName %params "params"
OpDecorate %Params Block
OpMemberDecorate %Params 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%rayquery = OpTypeRayQueryKHR
%ptr_private_rayquery = OpTypePointer Private %rayquery
%uint = OpTypeInt 32 0
%uvec2 = OpTypeVector %uint 2
%Params = OpTypeStruct %uvec2
%ptr_pc = OpTypePointer PushConstant %Params
%int = OpTypeInt 32 1
%zero_i = OpConstant %int 0
%ptr_pc_uvec2 = OpTypePointer PushConstant %uvec2
%accel = OpTypeAccelerationStructureKHR
%zero = OpConstant %uint 0
%float = OpTypeFloat 32
%vec3 = OpTypeVector %float 3
%zero_f = OpConstant %float 0.0
%origin = OpConstantComposite %vec3 %zero_f %zero_f %zero_f
%one_f = OpConstant %float 1.0
%direction = OpConstantComposite %vec3 %one_f %one_f %one_f
%ray_query = OpVariable %ptr_private_rayquery Private
%params = OpVariable %ptr_pc PushConstant
%main = OpFunction %void None %fn
%label = OpLabel
%ptr = OpAccessChain %ptr_pc_uvec2 %params %zero_i
%encoded = OpLoad %uvec2 %ptr
%tlas = OpConvertUToAccelerationStructureKHR %accel %encoded
OpRayQueryInitializeKHR %ray_query %tlas %zero %zero %origin %zero_f %direction %one_f
OpRayQueryTerminateKHR %ray_query
OpReturn
OpFunctionEnd
"""

SPIRV_STORAGE_IMAGE_FORMAT_ASSEMBLY = """
; Reduced from Vulkan storage-image SPIR-V mapping examples.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpName %storage_image "storageImage"
OpDecorate %storage_image DescriptorSet 0
OpDecorate %storage_image Binding 0
OpDecorate %storage_image NonWritable
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%image = OpTypeImage %uint 2D 0 0 0 2 R32ui
%ptr_storage_image = OpTypePointer UniformConstant %image
%storage_image = OpVariable %ptr_storage_image UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_STORAGE_IMAGE_ARRAY_FORMAT_ASSEMBLY = """
; Reduced from Vulkan descriptor arrays of formatted storage images.
; The Image Format lives on the OpTypeImage element, not the OpTypeArray.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %storage_images "storageImages"
OpDecorate %storage_images DescriptorSet 0
OpDecorate %storage_images Binding 0
OpDecorate %storage_images NonWritable
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%count = OpConstant %uint 4
%image = OpTypeImage %uint 2D 0 0 0 2 R32ui
%image_array = OpTypeArray %image %count
%ptr_storage_images = OpTypePointer UniformConstant %image_array
%storage_images = OpVariable %ptr_storage_images UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_NON_32BIT_INTEGER_IMAGE_FAMILY_ASSEMBLY = """
; Reduced from SPIR-V explicit integer-width image declarations.
; Preserved i16/u16 sampled types should still select signed integer resources.
OpCapability Shader
OpCapability Int16
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpName %short_tex "shortTex"
OpName %signed_storage "signedStorage"
OpDecorate %short_tex DescriptorSet 0
OpDecorate %short_tex Binding 0
OpDecorate %signed_storage DescriptorSet 0
OpDecorate %signed_storage Binding 1
OpDecorate %signed_storage NonReadable
%void = OpTypeVoid
%fn = OpTypeFunction %void
%short = OpTypeInt 16 1
%ushort = OpTypeInt 16 0
%sampled_image = OpTypeImage %ushort 2D 0 0 0 1 Unknown
%combined = OpTypeSampledImage %sampled_image
%storage_image = OpTypeImage %short 2D 0 0 0 2 R16i
%ptr_sampled = OpTypePointer UniformConstant %combined
%ptr_storage = OpTypePointer UniformConstant %storage_image
%short_tex = OpVariable %ptr_sampled UniformConstant
%signed_storage = OpVariable %ptr_storage UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_SAMPLED_IMAGE_FETCH_ASSEMBLY = """
; Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Test/baseResults/web.separate.frag.out OpSampledImage and
; Test/baseResults/web.texture.frag.out OpImageSampleExplicitLod/OpImageFetch.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %sample_coord %fetch_coord %sample_color %fetch_color
OpExecutionMode %main OriginUpperLeft
OpName %texture_only "textureOnly"
OpName %linear_sampler "linearSampler"
OpName %sample_coord "sampleCoord"
OpName %fetch_coord "fetchCoord"
OpName %sample_color "sampleColor"
OpName %fetch_color "fetchColor"
OpDecorate %texture_only DescriptorSet 0
OpDecorate %texture_only Binding 0
OpDecorate %linear_sampler DescriptorSet 0
OpDecorate %linear_sampler Binding 1
OpDecorate %sample_coord Location 0
OpDecorate %fetch_coord Location 1
OpDecorate %sample_color Location 0
OpDecorate %fetch_color Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%sampler = OpTypeSampler
%ptr_texture = OpTypePointer UniformConstant %image
%ptr_sampler = OpTypePointer UniformConstant %sampler
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_input_v2int = OpTypePointer Input %v2int
%ptr_output_v4float = OpTypePointer Output %v4float
%lod = OpConstant %float 1.0
%zero = OpConstant %int 0
%texture_only = OpVariable %ptr_texture UniformConstant
%linear_sampler = OpVariable %ptr_sampler UniformConstant
%sample_coord = OpVariable %ptr_input_v2float Input
%fetch_coord = OpVariable %ptr_input_v2int Input
%sample_color = OpVariable %ptr_output_v4float Output
%fetch_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_texture = OpLoad %image %texture_only
%loaded_sampler = OpLoad %sampler %linear_sampler
%combined = OpSampledImage %sampled %loaded_texture %loaded_sampler
%uv = OpLoad %v2float %sample_coord
%sample = OpImageSampleExplicitLod %v4float %combined %uv Lod %lod
OpStore %sample_color %sample
%image_only = OpImage %image %combined
%pixel = OpLoad %v2int %fetch_coord
%fetch = OpImageFetch %v4float %image_only %pixel Lod %zero
OpStore %fetch_color %fetch
OpReturn
OpFunctionEnd
"""

SPIRV_DESCRIPTOR_INDEXING_NONUNIFORM_ASSEMBLY = """
; Reduced from descriptor-indexed sampled texture SPIR-V emitted by Vulkan
; toolchains for GL_EXT_nonuniform_qualifier/nonuniformEXT material indexing.
OpCapability Shader
OpCapability ShaderNonUniform
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %material_index %uv %out_color
OpExecutionMode %main OriginUpperLeft
OpName %textures "textures"
OpName %linear_sampler "linearSampler"
OpName %material_index "materialIndex"
OpName %uv "uv"
OpName %out_color "outColor"
OpDecorate %textures DescriptorSet 0
OpDecorate %textures Binding 0
OpDecorate %linear_sampler DescriptorSet 0
OpDecorate %linear_sampler Binding 1
OpDecorate %material_index Location 0
OpDecorate %uv Location 1
OpDecorate %out_color Location 0
OpDecorate %idx NonUniform
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%sampler = OpTypeSampler
%array_count = OpConstant %int 4
%zero = OpConstant %float 0.0
%texture_array = OpTypeArray %image %array_count
%ptr_textures = OpTypePointer UniformConstant %texture_array
%ptr_texture = OpTypePointer UniformConstant %image
%ptr_sampler = OpTypePointer UniformConstant %sampler
%ptr_input_int = OpTypePointer Input %int
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%textures = OpVariable %ptr_textures UniformConstant
%linear_sampler = OpVariable %ptr_sampler UniformConstant
%material_index = OpVariable %ptr_input_int Input
%uv = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_index = OpLoad %int %material_index
%idx = OpCopyObject %int %loaded_index
%texture_ptr = OpAccessChain %ptr_texture %textures %idx
%loaded_texture = OpLoad %image %texture_ptr
%loaded_sampler = OpLoad %sampler %linear_sampler
%combined = OpSampledImage %sampled %loaded_texture %loaded_sampler
%loaded_uv = OpLoad %v2float %uv
%sample = OpImageSampleExplicitLod %v4float %combined %loaded_uv Lod %zero
OpStore %out_color %sample
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMPLICIT_LOD_BIAS_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 62138e5bb72e73a202d2a10360367754f94a621d
; Source path: test/val/val_image_test.cpp
; Reduced from ValidateImage::SampleImplicitLodSuccess OpImageSampleImplicitLod
; with the Bias image operand.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %uv "uv"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %uv Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%bias = OpConstant %float 0.25
%color_tex = OpVariable %ptr_sampled UniformConstant
%uv = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%loaded_uv = OpLoad %v2float %uv
%sample = OpImageSampleImplicitLod %v4float %loaded_tex %loaded_uv Bias %bias
OpStore %out_color %sample
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_IMPLICIT_LOD_MIN_LOD_OFFSET_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_a_id_image_operands_a_image_operands
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.core.grammar.json
; Source example: KhronosGroup/SPIRV-Tools source/validate_image.cpp ValidateImageOperands
; Reduced from OpImageSampleImplicitLod with the ConstOffset and MinLod image operands.
OpCapability Shader
OpCapability MinLod
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %uv "uv"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %uv Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%min_lod = OpConstant %float 0.5
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%color_tex = OpVariable %ptr_sampled UniformConstant
%uv = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%loaded_uv = OpLoad %v2float %uv
%sample = OpImageSampleImplicitLod %v4float %loaded_tex %loaded_uv ConstOffset|MinLod %offset %min_lod
OpStore %out_color %sample
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_DREF_SAMPLE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/web.texture.frag.out
; Reduced from OpImageSampleDrefExplicitLod %float %57 %58 %66 Lod|ConstOffset %61 %65.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %shadow_tex %coord %depth_out
OpExecutionMode %main OriginUpperLeft
OpName %shadow_tex "shadowTex"
OpName %coord "coord"
OpName %depth_out "depthOut"
OpDecorate %shadow_tex DescriptorSet 0
OpDecorate %shadow_tex Binding 0
OpDecorate %coord Location 0
OpDecorate %depth_out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%image = OpTypeImage %float 2D 1 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_shadow = OpTypePointer UniformConstant %sampled
%ptr_input_v3float = OpTypePointer Input %v3float
%ptr_output_float = OpTypePointer Output %float
%lod = OpConstant %float 0.0
%shadow_tex = OpVariable %ptr_shadow UniformConstant
%coord = OpVariable %ptr_input_v3float Input
%depth_out = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_shadow = OpLoad %sampled %shadow_tex
%loaded_coord = OpLoad %v3float %coord
%depth = OpCompositeExtract %float %loaded_coord 2
%sample = OpImageSampleDrefExplicitLod %float %loaded_shadow %loaded_coord %depth Lod %lod
OpStore %depth_out %sample
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_DREF_SAMPLE_LOD_OFFSET_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/web.texture.frag.out
; Reduced from OpImageSampleDrefExplicitLod %float %57 %58 %66 Lod|ConstOffset %61 %65.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %shadow_tex %coord %depth_out
OpExecutionMode %main OriginUpperLeft
OpName %shadow_tex "shadowTex"
OpName %coord "coord"
OpName %depth_out "depthOut"
OpDecorate %shadow_tex DescriptorSet 0
OpDecorate %shadow_tex Binding 0
OpDecorate %coord Location 0
OpDecorate %depth_out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%v3float = OpTypeVector %float 3
%image = OpTypeImage %float 2D 1 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_shadow = OpTypePointer UniformConstant %sampled
%ptr_input_v3float = OpTypePointer Input %v3float
%ptr_output_float = OpTypePointer Output %float
%lod = OpConstant %float 0.0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%shadow_tex = OpVariable %ptr_shadow UniformConstant
%coord = OpVariable %ptr_input_v3float Input
%depth_out = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_shadow = OpLoad %sampled %shadow_tex
%loaded_coord = OpLoad %v3float %coord
%depth = OpCompositeExtract %float %loaded_coord 2
%sample = OpImageSampleDrefExplicitLod %float %loaded_shadow %loaded_coord %depth Lod|ConstOffset %lod %offset
OpStore %depth_out %sample
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_TEXTURE_LOD_OFFSET_ASSEMBLY = """
; Reduced from glslangValidator output for textureLodOffset(colorTex, uv, 1.0, ivec2(1, 2)).
; glslang emits OpImageSampleExplicitLod with the combined Lod|ConstOffset image operand mask.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %uv "uv"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %uv Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%lod = OpConstant %float 1.0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%color_tex = OpVariable %ptr_sampled UniformConstant
%uv = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%loaded_uv = OpLoad %v2float %uv
%sample = OpImageSampleExplicitLod %v4float %loaded_tex %loaded_uv Lod|ConstOffset %lod %offset
OpStore %out_color %sample
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_TEXTURE_PROJ_OFFSET_ASSEMBLY = """
; Reduced from glslangValidator output for textureProjOffset,
; textureProjLodOffset, and textureProjGradOffset on a 2D sampled image.
; glslang emits OpImageSampleProj* with ConstOffset image operands.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uvq %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %uvq "uvq"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %uvq Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v3float = OpTypeVector %float 3
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v3float = OpTypePointer Input %v3float
%ptr_output_v4float = OpTypePointer Output %v4float
%zero = OpConstant %float 0.0
%half = OpConstant %float 0.5
%lod = OpConstant %float 1.0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%dx = OpConstantComposite %v2float %half %zero
%dy = OpConstantComposite %v2float %zero %half
%color_tex = OpVariable %ptr_sampled UniformConstant
%uvq = OpVariable %ptr_input_v3float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%loaded_uvq = OpLoad %v3float %uvq
%implicit_offset = OpImageSampleProjImplicitLod %v4float %loaded_tex %loaded_uvq ConstOffset %offset
%lod_offset = OpImageSampleProjExplicitLod %v4float %loaded_tex %loaded_uvq Lod|ConstOffset %lod %offset
%grad_offset = OpImageSampleProjExplicitLod %v4float %loaded_tex %loaded_uvq Grad|ConstOffset %dx %dy %offset
OpStore %out_color %implicit_offset
OpStore %out_color %lod_offset
OpStore %out_color %grad_offset
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_TEXEL_FETCH_OFFSET_ASSEMBLY = """
; Reduced from glslangValidator output for texelFetchOffset(colorTex, pixel, 2, ivec2(1, 2)).
; glslang emits OpImageFetch with the combined Lod|ConstOffset image operand mask.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %pixel %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %pixel "pixel"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %pixel Flat
OpDecorate %pixel Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2int = OpTypePointer Input %v2int
%ptr_output_v4float = OpTypePointer Output %v4float
%lod = OpConstant %int 2
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%color_tex = OpVariable %ptr_sampled UniformConstant
%pixel = OpVariable %ptr_input_v2int Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%image_only = OpImage %image %loaded_tex
%loaded_pixel = OpLoad %v2int %pixel
%fetch = OpImageFetch %v4float %image_only %loaded_pixel Lod|ConstOffset %lod %offset
OpStore %out_color %fetch
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_TEXEL_FETCH_MS_SAMPLE_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_a_id_image_operands_a_image_operands
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.core.grammar.json
; Reduced from GLSL texelFetch(sampler2DMS, pixel, sample), where SPIR-V
; carries the multisample index as an OpImageFetch Sample image operand.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %pixel %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %pixel "pixel"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %pixel Flat
OpDecorate %pixel Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 1 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2int = OpTypePointer Input %v2int
%ptr_output_v4float = OpTypePointer Output %v4float
%sample_id = OpConstant %int 2
%color_tex = OpVariable %ptr_sampled UniformConstant
%pixel = OpVariable %ptr_input_v2int Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%image_only = OpImage %image %loaded_tex
%loaded_pixel = OpLoad %v2int %pixel
%fetch = OpImageFetch %v4float %image_only %loaded_pixel Sample %sample_id
OpStore %out_color %fetch
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_TEXTURE_GATHER_OFFSET_ASSEMBLY = """
; Reduced from glslangValidator-style output for textureGatherOffset(colorTex, uv, ivec2(1, 2), 2).
; SPIR-V lowers this to OpImageGather with Component and ConstOffset operands.
OpCapability Shader
OpCapability ImageGatherExtended
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %out_color
OpExecutionMode %main OriginUpperLeft
OpName %color_tex "colorTex"
OpName %uv "uv"
OpName %out_color "outColor"
OpDecorate %color_tex DescriptorSet 0
OpDecorate %color_tex Binding 0
OpDecorate %uv Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v4float = OpTypePointer Output %v4float
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%color_tex = OpVariable %ptr_sampled UniformConstant
%uv = OpVariable %ptr_input_v2float Input
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_tex = OpLoad %sampled %color_tex
%loaded_uv = OpLoad %v2float %uv
%gather = OpImageGather %v4float %loaded_tex %loaded_uv %int_2 ConstOffset %offset
OpStore %out_color %gather
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_DREF_GATHER_OFFSET_ASSEMBLY = """
; Reduced from glslangValidator-style output for textureGatherCompareOffset(shadowTex, uv, depth, ivec2(1, 2)).
; SPIR-V lowers this to OpImageDrefGather with Dref and ConstOffset operands.
OpCapability Shader
OpCapability ImageGatherExtended
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %depth %depth_samples
OpExecutionMode %main OriginUpperLeft
OpName %shadow_tex "shadowTex"
OpName %uv "uv"
OpName %depth "depth"
OpName %depth_samples "depthSamples"
OpDecorate %shadow_tex DescriptorSet 0
OpDecorate %shadow_tex Binding 0
OpDecorate %uv Location 0
OpDecorate %depth Location 1
OpDecorate %depth_samples Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image = OpTypeImage %float 2D 1 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_input_float = OpTypePointer Input %float
%ptr_output_v4float = OpTypePointer Output %v4float
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%offset = OpConstantComposite %v2int %int_1 %int_2
%shadow_tex = OpVariable %ptr_sampled UniformConstant
%uv = OpVariable %ptr_input_v2float Input
%depth = OpVariable %ptr_input_float Input
%depth_samples = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_shadow = OpLoad %sampled %shadow_tex
%loaded_uv = OpLoad %v2float %uv
%loaded_depth = OpLoad %float %depth
%gather = OpImageDrefGather %v4float %loaded_shadow %loaded_uv %loaded_depth ConstOffset %offset
OpStore %depth_samples %gather
OpReturn
OpFunctionEnd
"""

SPIRV_VULKAN_SAMPLES_IMAGE_WRITE_ASSEMBLY = """
; Reduced from KhronosGroup/Vulkan-Samples@ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a
; shaders/timeline_semaphore/glsl/game_of_life_init.comp imageStore(Image, index, ...).
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 8 8 1
OpName %image "Image"
OpDecorate %image DescriptorSet 0
OpDecorate %image Binding 0
OpDecorate %image NonReadable
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image_type = OpTypeImage %float 2D 0 0 0 2 Rgba8
%ptr_image = OpTypePointer UniformConstant %image_type
%zero_i = OpConstant %int 0
%one_f = OpConstant %float 1.0
%zero_f = OpConstant %float 0.0
%coord = OpConstantComposite %v2int %zero_i %zero_i
%texel = OpConstantComposite %v4float %one_f %one_f %one_f %zero_f
%image = OpVariable %ptr_image UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_image = OpLoad %image_type %image
OpImageWrite %loaded_image %coord %texel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_IMAGE_WRITE_SAMPLE_ASSEMBLY = """
; Reduced from glslangValidator 16.3.0 stdin output for
; imageStore(image2DMS, coord, sample, texel), where glslang emits
; OpImageWrite Image Coordinate Texel Sample SampleId.
; SPIRV-Headers unified1 grammar declares OpImageWrite's optional
; ImageOperands and ImageOperands.Sample's single IdRef parameter.
OpCapability Shader
OpCapability StorageImageMultisample
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %image
OpExecutionMode %main LocalSize 1 1 1
OpName %image "image"
OpDecorate %image DescriptorSet 0
OpDecorate %image Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%v4float = OpTypeVector %float 4
%image_type = OpTypeImage %float 2D 0 0 1 2 Rgba16f
%ptr_image = OpTypePointer UniformConstant %image_type
%zero_i = OpConstant %int 0
%one_i = OpConstant %int 1
%sample = OpConstant %int 2
%one_f = OpConstant %float 1.0
%zero_f = OpConstant %float 0.0
%coord = OpConstantComposite %v2int %zero_i %one_i
%texel = OpConstantComposite %v4float %one_f %zero_f %zero_f %one_f
%image = OpVariable %ptr_image UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_image = OpLoad %image_type %image
OpImageWrite %loaded_image %coord %texel Sample %sample
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMAGE_READ_WRITE_LOD_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/val/val_image_test.cpp
; Reduced from ValidateImage::ReadLodAMDSuccess1 and WriteLodAMDSuccess1,
; where SPV_AMD_shader_image_load_store_lod allows OpImageRead/OpImageWrite
; with ImageOperands.Lod.
OpCapability Shader
OpCapability StorageImageReadWithoutFormat
OpCapability StorageImageWriteWithoutFormat
OpCapability ImageReadWriteLodAMD
OpExtension "SPV_AMD_shader_image_load_store_lod"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color %storage_image
OpExecutionMode %main OriginUpperLeft
OpName %out_color "outColor"
OpName %storage_image "storageImage"
OpDecorate %out_color Location 0
OpDecorate %storage_image DescriptorSet 0
OpDecorate %storage_image Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v2uint = OpTypeVector %uint 2
%v4uint = OpTypeVector %uint 4
%image_type = OpTypeImage %uint 2D 0 0 0 2 Unknown
%ptr_image = OpTypePointer UniformConstant %image_type
%ptr_output_v4uint = OpTypePointer Output %v4uint
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%coord = OpConstantComposite %v2uint %zero %one
%texel = OpConstantComposite %v4uint %one %zero %zero %one
%storage_image = OpVariable %ptr_image UniformConstant
%out_color = OpVariable %ptr_output_v4uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%img = OpLoad %image_type %storage_image
%read = OpImageRead %v4uint %img %coord Lod %zero
OpStore %out_color %read
OpImageWrite %img %coord %texel Lod %zero
OpReturn
OpFunctionEnd
"""

SPIRV_STORE_BODY_ASSEMBLY = """
; Reduced from a fragment output store.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %frag_color
OpExecutionMode %main OriginUpperLeft
OpName %frag_color "fragColor"
OpDecorate %frag_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%one = OpConstant %float 1.0
%zero = OpConstant %float 0.0
%red = OpConstantComposite %v4float %one %zero %zero %one
%frag_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpStore %frag_color %red
OpReturn
OpFunctionEnd
"""

SPIRV_VECTOR_SHUFFLE_BODY_ASSEMBLY = """
; Reduced from Khronos Vulkan-Samples timeline_semaphore/glsl/render.frag swizzles.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %value %color
OpExecutionMode %main OriginUpperLeft
OpName %value "value"
OpName %color "color"
OpDecorate %value Location 0
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v3float = OpTypePointer Output %v3float
%value = OpVariable %ptr_input_v4float Input
%color = OpVariable %ptr_output_v3float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %value
%rgb = OpVectorShuffle %v3float %loaded %loaded 0 1 2
OpStore %color %rgb
OpReturn
OpFunctionEnd
"""

SPIRV_CROSS_COPY_MEMORY_INTERFACE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Cross
; Source commit: 146679ff8255a6068518685599d7fb8761d1b570
; Source path: shaders-msl/asm/vert/copy-memory-interface.asm.vert
; Reduced from two interface OpCopyMemory instructions copying inputs to outputs.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %v0 %v1 %position %o1
OpName %main "main"
OpName %v0 "v0"
OpName %v1 "v1"
OpName %position "o0"
OpName %o1 "o1"
OpDecorate %v0 Location 0
OpDecorate %v1 Location 1
OpDecorate %position BuiltIn Position
OpDecorate %o1 Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%v0 = OpVariable %ptr_input_v4float Input
%v1 = OpVariable %ptr_input_v4float Input
%position = OpVariable %ptr_output_v4float Output
%o1 = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpCopyMemory %position %v0
OpCopyMemory %o1 %v1
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_COPY_MEMORY_SIZED_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/val/val_memory_test.cpp
; Source test: ValidateMemory::CopyMemorySizedNoAccessGood.
; Reduced from OpCopyMemorySized %var1 %var2 %int_16.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %out_value
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %out_value "outValue"
OpName %source "source"
OpName %target "target"
OpDecorate %input_value Location 0
OpDecorate %out_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_function_v4float = OpTypePointer Function %v4float
%uint_16 = OpConstant %uint 16
%input_value = OpVariable %ptr_input_v4float Input
%out_value = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%source = OpVariable %ptr_function_v4float Function
%target = OpVariable %ptr_function_v4float Function
%loaded = OpLoad %v4float %input_value
OpStore %source %loaded
OpCopyMemorySized %target %source %uint_16
%copied = OpLoad %v4float %target
OpStore %out_value %copied
OpReturn
OpFunctionEnd
"""

SPIRV_CROSS_IMAGE_QUERY_SIZE_LOD_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Cross
; Source commit: 146679ff8255a6068518685599d7fb8761d1b570
; Source path: shaders/asm/frag/image-extract-reuse.asm.frag
; Reduced from OpImageQuerySizeLod on a sampled 2D image.
OpCapability Shader
OpCapability ImageQuery
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %Size
OpExecutionMode %main OriginUpperLeft
OpName %Size "Size"
OpName %uTexture "uTexture"
OpDecorate %Size Location 0
OpDecorate %uTexture DescriptorSet 0
OpDecorate %uTexture Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%ptr_output_v2int = OpTypePointer Output %v2int
%Size = OpVariable %ptr_output_v2int Output
%float = OpTypeFloat 32
%image_type = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled_type = OpTypeSampledImage %image_type
%ptr_sampled = OpTypePointer UniformConstant %sampled_type
%uTexture = OpVariable %ptr_sampled UniformConstant
%zero = OpConstant %int 0
%one = OpConstant %int 1
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %sampled_type %uTexture
%image = OpImage %image_type %loaded
%size0 = OpImageQuerySizeLod %v2int %image %zero
%size1 = OpImageQuerySizeLod %v2int %image %one
%sum = OpIAdd %v2int %size0 %size1
OpStore %Size %sum
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMAGE_QUERY_SIZE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: df032578c737d361b754fc569b70aa29b5f8c7d4
; Source path: test/val/val_image_test.cpp
; Reduced from ValidateImage::QuerySizeSuccess OpImageQuerySize on a multisampled image.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %Size %uImage
OpExecutionMode %main OriginUpperLeft
OpName %Size "Size"
OpName %uImage "uImage"
OpDecorate %Size Location 0
OpDecorate %uImage DescriptorSet 0
OpDecorate %uImage Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%float = OpTypeFloat 32
%image_type = OpTypeImage %float 2D 0 0 1 1 Unknown
%ptr_output_v2int = OpTypePointer Output %v2int
%ptr_uniformconstant_image = OpTypePointer UniformConstant %image_type
%Size = OpVariable %ptr_output_v2int Output
%uImage = OpVariable %ptr_uniformconstant_image UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %image_type %uImage
%size = OpImageQuerySize %v2int %loaded
OpStore %Size %size
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMAGE_QUERY_FORMAT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/val/val_image_test.cpp
; Reduced from ValidateImage::QueryFormatSuccess and QueryOrderSuccess.
OpCapability Shader
OpCapability ImageQuery
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %format_out %order_out %query_image
OpExecutionMode %main OriginUpperLeft
OpName %format_out "formatOut"
OpName %order_out "orderOut"
OpName %query_image "queryImage"
OpDecorate %format_out Location 0
OpDecorate %order_out Location 1
OpDecorate %query_image DescriptorSet 0
OpDecorate %query_image Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%ptr_output_uint = OpTypePointer Output %uint
%ptr_image = OpTypePointer UniformConstant %image
%format_out = OpVariable %ptr_output_uint Output
%order_out = OpVariable %ptr_output_uint Output
%query_image = OpVariable %ptr_image UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %image %query_image
%format = OpImageQueryFormat %uint %loaded
OpStore %format_out %format
%order = OpImageQueryOrder %uint %loaded
OpStore %order_out %order
OpReturn
OpFunctionEnd
"""

SPIRV_IMAGE_QUERY_LOD_LEVELS_SAMPLES_ASSEMBLY = """
; Source: SPIR-V spec image query instructions plus Vulkan image query semantics.
; Reduced to OpImageQueryLod, OpImageQueryLevels, and OpImageQuerySamples.
OpCapability Shader
OpCapability ImageQuery
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %lod_out %levels_out %samples_out %query_tex %ms_tex
OpExecutionMode %main OriginUpperLeft
OpName %uv "uv"
OpName %lod_out "lodOut"
OpName %levels_out "levelsOut"
OpName %samples_out "samplesOut"
OpName %query_tex "queryTex"
OpName %ms_tex "msTex"
OpDecorate %uv Location 0
OpDecorate %lod_out Location 0
OpDecorate %levels_out Location 1
OpDecorate %samples_out Location 2
OpDecorate %query_tex DescriptorSet 0
OpDecorate %query_tex Binding 0
OpDecorate %ms_tex DescriptorSet 0
OpDecorate %ms_tex Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%ptr_input_v2float = OpTypePointer Input %v2float
%ptr_output_v2float = OpTypePointer Output %v2float
%ptr_output_int = OpTypePointer Output %int
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ms_image = OpTypeImage %float 2D 0 0 1 1 Unknown
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_ms_image = OpTypePointer UniformConstant %ms_image
%uv = OpVariable %ptr_input_v2float Input
%lod_out = OpVariable %ptr_output_v2float Output
%levels_out = OpVariable %ptr_output_int Output
%samples_out = OpVariable %ptr_output_int Output
%query_tex = OpVariable %ptr_sampled UniformConstant
%ms_tex = OpVariable %ptr_ms_image UniformConstant
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_query = OpLoad %sampled %query_tex
%loaded_uv = OpLoad %v2float %uv
%lod = OpImageQueryLod %v2float %loaded_query %loaded_uv
OpStore %lod_out %lod
%image_only = OpImage %image %loaded_query
%levels = OpImageQueryLevels %int %image_only
OpStore %levels_out %levels
%loaded_ms = OpLoad %ms_image %ms_tex
%samples = OpImageQuerySamples %int %loaded_ms
OpStore %samples_out %samples
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMAGE_READ_SAMPLE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: df032578c737d361b754fc569b70aa29b5f8c7d4
; Source path: test/val/val_image_test.cpp
; Reduced from ValidateImage::ImageMSArray_SampledTypeDoesNotRequireCapability
; OpImageRead %v4float %18 %10 Sample %uint_2.
OpCapability Shader
OpCapability StorageImageMultisample
OpCapability StorageImageReadWithoutFormat
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color %var_image
OpExecutionMode %main OriginUpperLeft
OpName %color "color"
OpDecorate %color Location 0
OpDecorate %var_image DescriptorSet 0
OpDecorate %var_image Binding 1
%void = OpTypeVoid
%func = OpTypeFunction %void
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%uint_2 = OpConstant %u32 2
%uint_1 = OpConstant %u32 1
%v2uint = OpTypeVector %u32 2
%v4float = OpTypeVector %f32 4
%image = OpTypeImage %f32 2D 2 0 1 2 Unknown
%ptr_image = OpTypePointer UniformConstant %image
%ptr_output_v4float = OpTypePointer Output %v4float
%10 = OpConstantComposite %v2uint %uint_1 %uint_2
%var_image = OpVariable %ptr_image UniformConstant
%color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %func
%main_lab = OpLabel
%18 = OpLoad %image %var_image
%19 = OpImageRead %v4float %18 %10 Sample %uint_2
OpStore %color %19
OpReturn
OpFunctionEnd
"""

SPIRV_CROSS_SPARSE_TEXTURE_RESIDENT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Cross
; Source commit: 146679ff8255a6068518685599d7fb8761d1b570
; Source path: shaders-no-opt/asm/frag/sparse-texture-feedback-uint-code.asm.desktop.frag
; Reduced from OpImageSparseSampleImplicitLod, result extracts, and
; OpImageSparseTexelsResident.
OpCapability Shader
OpCapability SparseResidency
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %vUV %FragColor
OpExecutionMode %main OriginUpperLeft
OpName %ret "ret"
OpName %uSamp "uSamp"
OpName %vUV "vUV"
OpName %texel "texel"
OpName %ResType "ResType"
OpName %FragColor "FragColor"
OpDecorate %uSamp DescriptorSet 0
OpDecorate %uSamp Binding 0
OpDecorate %vUV Location 0
OpDecorate %FragColor Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%ptr_function_bool = OpTypePointer Function %bool
%float = OpTypeFloat 32
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%uSamp = OpVariable %ptr_sampled UniformConstant
%v2float = OpTypeVector %float 2
%ptr_input_v2float = OpTypePointer Input %v2float
%vUV = OpVariable %ptr_input_v2float Input
%v4float = OpTypeVector %float 4
%ptr_function_v4float = OpTypePointer Function %v4float
%uint = OpTypeInt 32 0
%ResType = OpTypeStruct %uint %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%FragColor = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%ret = OpVariable %ptr_function_bool Function
%texel = OpVariable %ptr_function_v4float Function
%loaded_tex = OpLoad %sampled %uSamp
%loaded_uv = OpLoad %v2float %vUV
%sparse = OpImageSparseSampleImplicitLod %ResType %loaded_tex %loaded_uv
%texel_value = OpCompositeExtract %v4float %sparse 1
OpStore %texel %texel_value
%resident_code = OpCompositeExtract %uint %sparse 0
%resident = OpImageSparseTexelsResident %bool %resident_code
OpStore %ret %resident
OpStore %FragColor %texel_value
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_SPARSE_DREF_SAMPLE_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Reduced from OpImageSparseSampleDrefExplicitLod image operands and result
; extraction rules.
OpCapability Shader
OpCapability SparseResidency
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %shadow_tex %coord %depth_out
OpExecutionMode %main OriginUpperLeft
OpName %shadow_tex "shadowTex"
OpName %coord "coord"
OpName %depth_out "depthOut"
OpDecorate %shadow_tex DescriptorSet 0
OpDecorate %shadow_tex Binding 0
OpDecorate %coord Location 0
OpDecorate %depth_out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v3float = OpTypeVector %float 3
%image = OpTypeImage %float 2D 1 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%res_type = OpTypeStruct %uint %float
%ptr_shadow = OpTypePointer UniformConstant %sampled
%ptr_input_v3float = OpTypePointer Input %v3float
%ptr_output_float = OpTypePointer Output %float
%zero = OpConstant %float 0.0
%shadow_tex = OpVariable %ptr_shadow UniformConstant
%coord = OpVariable %ptr_input_v3float Input
%depth_out = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_shadow = OpLoad %sampled %shadow_tex
%loaded_coord = OpLoad %v3float %coord
%depth = OpCompositeExtract %float %loaded_coord 2
%sparse = OpImageSparseSampleDrefExplicitLod %res_type %loaded_shadow %loaded_coord %depth Lod %zero
%texel = OpCompositeExtract %float %sparse 1
OpStore %depth_out %texel
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMAGE_SPARSE_READ_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/text_to_binary.image_test.cpp
; Reduced from OpImageSparseRead text-to-binary coverage plus sparse result
; extraction rules.
OpCapability Shader
OpCapability SparseResidency
OpCapability StorageImageReadWithoutFormat
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color %storage_image
OpExecutionMode %main OriginUpperLeft
OpName %out_color "outColor"
OpName %storage_image "storageImage"
OpName %resident "resident"
OpDecorate %out_color Location 0
OpDecorate %storage_image DescriptorSet 0
OpDecorate %storage_image Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%ptr_function_bool = OpTypePointer Function %bool
%uint = OpTypeInt 32 0
%v2uint = OpTypeVector %uint 2
%v4uint = OpTypeVector %uint 4
%res_type = OpTypeStruct %uint %v4uint
%image_type = OpTypeImage %uint 2D 0 0 0 2 Unknown
%ptr_image = OpTypePointer UniformConstant %image_type
%ptr_output_v4uint = OpTypePointer Output %v4uint
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%coord = OpConstantComposite %v2uint %zero %one
%storage_image = OpVariable %ptr_image UniformConstant
%out_color = OpVariable %ptr_output_v4uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%resident = OpVariable %ptr_function_bool Function
%img = OpLoad %image_type %storage_image
%sparse = OpImageSparseRead %res_type %img %coord
%texel = OpCompositeExtract %v4uint %sparse 1
OpStore %out_color %texel
%code = OpCompositeExtract %uint %sparse 0
%is_resident = OpImageSparseTexelsResident %bool %code
OpStore %resident %is_resident
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_IMAGE_SPARSE_FETCH_GATHER_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: f3f1169512c713d979a7aa1bc0c6c0fd89f0a85f
; Source path: test/val/val_image_test.cpp
; Reduced from sparse image fetch/gather validation coverage plus sparse
; result extraction rules.
OpCapability Shader
OpCapability SparseResidency
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %fetch_out %gather_out %resident_out %image_var %sampled_var
OpExecutionMode %main OriginUpperLeft
OpName %fetch_out "fetchOut"
OpName %gather_out "gatherOut"
OpName %resident_out "residentOut"
OpName %image_var "imageVar"
OpName %sampled_var "sampledVar"
OpDecorate %fetch_out Location 0
OpDecorate %gather_out Location 1
OpDecorate %resident_out Location 2
OpDecorate %image_var DescriptorSet 0
OpDecorate %image_var Binding 0
OpDecorate %sampled_var DescriptorSet 0
OpDecorate %sampled_var Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%ptr_output_bool = OpTypePointer Output %bool
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2int = OpTypeVector %uint 2
%v4float = OpTypeVector %float 4
%res_type = OpTypeStruct %uint %v4float
%image = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampled = OpTypeSampledImage %image
%ptr_image = OpTypePointer UniformConstant %image
%ptr_sampled = OpTypePointer UniformConstant %sampled
%ptr_output_v4float = OpTypePointer Output %v4float
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%coord = OpConstantComposite %v2int %zero %one
%image_var = OpVariable %ptr_image UniformConstant
%sampled_var = OpVariable %ptr_sampled UniformConstant
%fetch_out = OpVariable %ptr_output_v4float Output
%gather_out = OpVariable %ptr_output_v4float Output
%resident_out = OpVariable %ptr_output_bool Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_image = OpLoad %image %image_var
%fetch_sparse = OpImageSparseFetch %res_type %loaded_image %coord Lod %zero
%fetch_texel = OpCompositeExtract %v4float %fetch_sparse 1
OpStore %fetch_out %fetch_texel
%fetch_code = OpCompositeExtract %uint %fetch_sparse 0
%resident = OpImageSparseTexelsResident %bool %fetch_code
OpStore %resident_out %resident
%loaded_sampled = OpLoad %sampled %sampled_var
%gather_sparse = OpImageSparseGather %res_type %loaded_sampled %coord %zero
%gather_texel = OpCompositeExtract %v4float %gather_sparse 1
OpStore %gather_out %gather_texel
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_BITCAST_SUCCESS_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: df032578c737d361b754fc569b70aa29b5f8c7d4
; Source path: test/val/val_conversion_test.cpp
; Reduced from ValidateConversion::BitcastSuccess OpBitcast %f32 %u32_1
; and OpBitcast %f32vec2 %u32vec2_12.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %float_out %vec_out
OpExecutionMode %main OriginUpperLeft
OpName %float_out "floatOut"
OpName %vec_out "vecOut"
OpDecorate %float_out Location 0
OpDecorate %vec_out Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v2float = OpTypeVector %float 2
%ptr_output_float = OpTypePointer Output %float
%ptr_output_v2float = OpTypePointer Output %v2float
%u32_1 = OpConstant %uint 1
%u32_2 = OpConstant %uint 2
%u32vec2_12 = OpConstantComposite %v2uint %u32_1 %u32_2
%float_out = OpVariable %ptr_output_float Output
%vec_out = OpVariable %ptr_output_v2float Output
%main = OpFunction %void None %fn
%label = OpLabel
%float_value = OpBitcast %float %u32_1
OpStore %float_out %float_value
%vec_value = OpBitcast %v2float %u32vec2_12
OpStore %vec_out %vec_value
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_SAT_CONVERT_STOU_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 1c74ae51a0478e15e3460fcc536b46c792684d2e
; Source path: test/val/val_conversion_test.cpp
; Reduced from ValidateConversion::SatConvertSToUSuccess
; OpSatConvertSToU %u32 %u64_2.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %signed_in %unsigned_out
OpExecutionMode %main OriginUpperLeft
OpName %signed_in "signedIn"
OpName %unsigned_out "unsignedOut"
OpDecorate %signed_in Location 0
OpDecorate %unsigned_out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%ptr_input_int = OpTypePointer Input %int
%ptr_output_uint = OpTypePointer Output %uint
%signed_in = OpVariable %ptr_input_int Input
%unsigned_out = OpVariable %ptr_output_uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %int %signed_in
%saturated = OpSatConvertSToU %uint %loaded
OpStore %unsigned_out %saturated
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_BIT_INSTRUCTIONS_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Reduced from core bit instruction definitions for OpBitCount, OpBitReverse,
; OpBitFieldUExtract, OpBitFieldSExtract, and OpBitFieldInsert.
OpCapability Shader
OpCapability BitInstructions
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %mask %signed_mask %offset %bits %count_out %reverse_out %uextract_out %sextract_out %insert_out
OpExecutionMode %main OriginUpperLeft
OpName %mask "mask"
OpName %signed_mask "signedMask"
OpName %offset "offset"
OpName %bits "bits"
OpName %count_out "countOut"
OpName %reverse_out "reverseOut"
OpName %uextract_out "uExtractOut"
OpName %sextract_out "sExtractOut"
OpName %insert_out "insertOut"
OpDecorate %mask Location 0
OpDecorate %signed_mask Location 1
OpDecorate %offset Location 2
OpDecorate %bits Location 3
OpDecorate %count_out Location 0
OpDecorate %reverse_out Location 1
OpDecorate %uextract_out Location 2
OpDecorate %sextract_out Location 3
OpDecorate %insert_out Location 4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%ptr_input_uint = OpTypePointer Input %uint
%ptr_input_int = OpTypePointer Input %int
%ptr_output_uint = OpTypePointer Output %uint
%ptr_output_int = OpTypePointer Output %int
%mask = OpVariable %ptr_input_uint Input
%signed_mask = OpVariable %ptr_input_int Input
%offset = OpVariable %ptr_input_int Input
%bits = OpVariable %ptr_input_int Input
%count_out = OpVariable %ptr_output_uint Output
%reverse_out = OpVariable %ptr_output_uint Output
%uextract_out = OpVariable %ptr_output_uint Output
%sextract_out = OpVariable %ptr_output_int Output
%insert_out = OpVariable %ptr_output_uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_mask = OpLoad %uint %mask
%loaded_signed = OpLoad %int %signed_mask
%loaded_offset = OpLoad %int %offset
%loaded_bits = OpLoad %int %bits
%counted = OpBitCount %uint %loaded_mask
OpStore %count_out %counted
%reversed = OpBitReverse %uint %loaded_mask
OpStore %reverse_out %reversed
%uextracted = OpBitFieldUExtract %uint %loaded_mask %loaded_offset %loaded_bits
OpStore %uextract_out %uextracted
%sextracted = OpBitFieldSExtract %int %loaded_signed %loaded_offset %loaded_bits
OpStore %sextract_out %sextracted
%inserted = OpBitFieldInsert %uint %loaded_mask %uextracted %loaded_offset %loaded_bits
OpStore %insert_out %inserted
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_COPY_OBJECT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: df032578c737d361b754fc569b70aa29b5f8c7d4
; Source path: test/opt/fold_test.cpp
; Reduced from RedundantBitcastTest/MergeNegateTest CHECK patterns expecting
; OpCopyObject to preserve the original value after folding.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %color
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %color "color"
OpDecorate %input_value Location 0
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%input_value = OpVariable %ptr_input_float Input
%color = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%copy = OpCopyObject %float %loaded
OpStore %color %copy
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_COPY_LOGICAL_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
; Reduced from the core OpCopyLogical instruction definition for logical objects.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %flag_in %flag_out
OpExecutionMode %main OriginUpperLeft
OpName %flag_in "flagIn"
OpName %flag_out "flagOut"
OpDecorate %flag_in Location 0
OpDecorate %flag_out Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%ptr_input_bool = OpTypePointer Input %bool
%ptr_output_bool = OpTypePointer Output %bool
%flag_in = OpVariable %ptr_input_bool Input
%flag_out = OpVariable %ptr_output_bool Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %bool %flag_in
%copy = OpCopyLogical %bool %loaded
OpStore %flag_out %copy
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_QUANTIZE_TO_F16_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/opt/fold_test.cpp
; Reduced from FloatScalarInstructionFoldingTest OpQuantizeToF16 cases.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %color
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %color "color"
OpDecorate %input_value Location 0
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%input_value = OpVariable %ptr_input_float Input
%color = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%quantized = OpQuantizeToF16 %float %loaded
OpStore %color %quantized
OpReturn
OpFunctionEnd
"""

SPIRV_CROSS_COMPOSITE_INSERT_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Cross
; Source commit: 146679ff8255a6068518685599d7fb8761d1b570
; Source path: shaders-no-opt/asm/frag/composite-insert-hoisted-temporaries-2.asm.frag
; Reduced from chained OpCompositeInsert rebuilding a vec2.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %value0 %value1 %FragColor
OpExecutionMode %main OriginUpperLeft
OpName %value0 "value0"
OpName %value1 "value1"
OpName %FragColor "FragColor"
OpDecorate %value0 Location 0
OpDecorate %value1 Location 1
OpDecorate %FragColor Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%ptr_input_float = OpTypePointer Input %float
%ptr_output_v2float = OpTypePointer Output %v2float
%zero = OpConstant %float 0.0
%base = OpConstantComposite %v2float %zero %zero
%value0 = OpVariable %ptr_input_float Input
%value1 = OpVariable %ptr_input_float Input
%FragColor = OpVariable %ptr_output_v2float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded0 = OpLoad %float %value0
%loaded1 = OpLoad %float %value1
%a = OpCompositeInsert %v2float %loaded0 %base 0
%b = OpCompositeInsert %v2float %loaded1 %a 1
OpStore %FragColor %b
OpReturn
OpFunctionEnd
"""

SPIRV_GLSL_STD450_EXTINST_BODY_ASSEMBLY = """
; Reduced from Khronos glslang Test/baseResults/web.basic.vert.out Normalize.
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %input_vec %output_vec
OpName %input_vec "inputVec"
OpName %output_vec "outputVec"
OpDecorate %input_vec Location 0
OpDecorate %output_vec Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%input_vec = OpVariable %ptr_input_v4float Input
%output_vec = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %input_vec
%normalized = OpExtInst %v4float %std450 Normalize %loaded
OpStore %output_vec %normalized
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_STD450_SQRT_ASSEMBLY = """
; Reduced from Khronos SPIRV-Tools test/diff/diff_files/extra_if_block_src.spvasm.
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %output_value
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %output_value "outputValue"
OpDecorate %input_value Location 0
OpDecorate %output_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%input_value = OpVariable %ptr_input_float Input
%output_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%root = OpExtInst %float %std450 Sqrt %loaded
OpStore %output_value %root
OpReturn
OpFunctionEnd
"""

SPIRV_NUMERIC_GLSL_STD450_SQRT_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
; Source enum: KhronosGroup/SPIRV-Headers include/spirv/unified1/GLSL.std.450.h.
; Reduced from OpExtInst binary form where GLSLstd450Sqrt has instruction number 31.
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %output_value
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %output_value "outputValue"
OpDecorate %input_value Location 0
OpDecorate %output_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%input_value = OpVariable %ptr_input_float Input
%output_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%root = OpExtInst %float %std450 31 %loaded
OpStore %output_value %root
OpReturn
OpFunctionEnd
"""

SPIRV_GLSL_STD450_STRUCT_EXTINST_ASSEMBLY = """
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
; Reduced from GLSL.std.450 ModfStruct and FrexpStruct extended instructions.
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %output_value
OpExecutionMode %main OriginUpperLeft
OpName %ModfResult "ModfResult"
OpMemberName %ModfResult 0 "fractional"
OpMemberName %ModfResult 1 "whole"
OpName %FrexpResult "FrexpResult"
OpMemberName %FrexpResult 0 "significand"
OpMemberName %FrexpResult 1 "exponent"
OpName %input_value "inputValue"
OpName %output_value "outputValue"
OpDecorate %input_value Location 0
OpDecorate %output_value Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%ModfResult = OpTypeStruct %float %float
%FrexpResult = OpTypeStruct %float %int
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%input_value = OpVariable %ptr_input_float Input
%output_value = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%modf_parts = OpExtInst %ModfResult %std450 ModfStruct %loaded
%fractional = OpCompositeExtract %float %modf_parts 0
%frexp_parts = OpExtInst %FrexpResult %std450 52 %loaded
%significand = OpCompositeExtract %float %frexp_parts 0
%combined = OpFAdd %float %fractional %significand
OpStore %output_value %combined
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_NONSEMANTIC_DEBUG_PRINTF_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/diff/diff_files/string_in_ext_inst_src.spvasm
; Reduced from OpString used as a NonSemantic.DebugPrintf OpExtInst argument.
OpCapability Shader
OpExtension "SPV_KHR_non_semantic_info"
%std450 = OpExtInstImport "GLSL.std.450"
%debug_printf = OpExtInstImport "NonSemantic.DebugPrintf"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %main "main"
OpName %foo "foo"
%fmt = OpString "unsigned == %u"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%ptr_function_uint = OpTypePointer Function %uint
%uint_127 = OpConstant %uint 127
%main = OpFunction %void None %fn
%label = OpLabel
%foo = OpVariable %ptr_function_uint Function
OpStore %foo %uint_127
%loaded = OpLoad %uint %foo
%call = OpExtInst %void %debug_printf 1 %fmt %loaded
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_DEBUG_FUNCTION_NAME_ASSEMBLY = """
; Reduced from Khronos SPIRV-Tools test/diff/diff_files/extra_if_block_src.spvasm.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color
OpExecutionMode %main OriginUpperLeft
OpName %helper "f1("
OpName %color "color"
OpDecorate %color Location 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%fn_void = OpTypeFunction %void
%fn_float = OpTypeFunction %float
%zero = OpConstant %float 0
%color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn_void
%main_label = OpLabel
%value = OpFunctionCall %float %helper
%out = OpCompositeConstruct %v4float %value %value %zero %zero
OpStore %color %out
OpReturn
OpFunctionEnd
%helper = OpFunction %float None %fn_float
%helper_label = OpLabel
OpReturnValue %zero
OpFunctionEnd
"""

SPIRV_TOOLS_POINTER_PARAMETER_BLOCK_ASSEMBLY = """
; Reduced from Khronos SPIRV-Tools test/diff/diff_files/different_decorations_fragment_src.spvasm.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %color
OpExecutionMode %main OriginUpperLeft
OpName %helper "helper("
OpName %param "param"
OpName %Block "Block"
OpMemberName %Block 0 "value"
OpName %block "block"
OpName %input_value "inputValue"
OpName %color "color"
OpDecorate %input_value Location 0
OpDecorate %color Location 0
OpDecorate %Block Block
OpDecorate %block DescriptorSet 0
OpDecorate %block Binding 0
OpMemberDecorate %Block 0 Offset 0
%void = OpTypeVoid
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%Block = OpTypeStruct %v4float
%ptr_input_v4float = OpTypePointer Input %v4float
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_uniform_block = OpTypePointer Uniform %Block
%ptr_function_v4float = OpTypePointer Function %v4float
%fn_void = OpTypeFunction %void
%fn_helper = OpTypeFunction %v4float %ptr_function_v4float
%input_value = OpVariable %ptr_input_v4float Input
%color = OpVariable %ptr_output_v4float Output
%block = OpVariable %ptr_uniform_block Uniform
%helper = OpFunction %v4float None %fn_helper
%param = OpFunctionParameter %ptr_function_v4float
%helper_label = OpLabel
%loaded_param = OpLoad %v4float %param
OpReturnValue %loaded_param
OpFunctionEnd
%main = OpFunction %void None %fn_void
%main_label = OpLabel
%local = OpVariable %ptr_function_v4float Function
%loaded_input = OpLoad %v4float %input_value
OpStore %local %loaded_input
%result = OpFunctionCall %v4float %helper %local
OpStore %color %result
OpReturn
OpFunctionEnd
"""

GLSLANG_STD450_FACEFORWARD_ASSEMBLY = """
; Reduced from glslang 16.3.0 fragment output for GLSL faceforward().
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color %normal %incident %reference
OpExecutionMode %main OriginUpperLeft
OpName %color "color"
OpName %normal "normal"
OpName %incident "incident"
OpName %reference "reference"
OpDecorate %color Location 0
OpDecorate %normal Location 0
OpDecorate %incident Location 1
OpDecorate %reference Location 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%ptr_output_v3float = OpTypePointer Output %v3float
%ptr_input_v3float = OpTypePointer Input %v3float
%color = OpVariable %ptr_output_v3float Output
%normal = OpVariable %ptr_input_v3float Input
%incident = OpVariable %ptr_input_v3float Input
%reference = OpVariable %ptr_input_v3float Input
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_normal = OpLoad %v3float %normal
%loaded_incident = OpLoad %v3float %incident
%loaded_reference = OpLoad %v3float %reference
%facing = OpExtInst %v3float %std450 FaceForward %loaded_normal %loaded_incident %loaded_reference
OpStore %color %facing
OpReturn
OpFunctionEnd
"""

GLSLANG_STD450_MATRIX_OPS_ASSEMBLY = """
; Reduced from Khronos glslang Test/baseResults/spv.matrix2.frag.out.
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_matrix %determinant_out %inverse_out
OpExecutionMode %main OriginUpperLeft
OpName %input_matrix "inputMatrix"
OpName %determinant_out "determinantOut"
OpName %inverse_out "inverseOut"
OpDecorate %input_matrix Location 0
OpDecorate %determinant_out Location 0
OpDecorate %inverse_out Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2 = OpTypeMatrix %v2float 2
%ptr_input_mat2 = OpTypePointer Input %mat2
%ptr_output_float = OpTypePointer Output %float
%ptr_output_mat2 = OpTypePointer Output %mat2
%input_matrix = OpVariable %ptr_input_mat2 Input
%determinant_out = OpVariable %ptr_output_float Output
%inverse_out = OpVariable %ptr_output_mat2 Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %mat2 %input_matrix
%det = OpExtInst %float %std450 Determinant %loaded
OpStore %determinant_out %det
%inverse = OpExtInst %mat2 %std450 MatrixInverse %loaded
OpStore %inverse_out %inverse
OpReturn
OpFunctionEnd
"""

GLSLANG_STD450_INTERPOLATION_ASSEMBLY = """
; Reduced from Khronos glslang Test/baseResults/spv.interpOps.frag.out.
OpCapability Shader
%std450 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %centroid_out %sample_out %offset_out
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %centroid_out "centroidOut"
OpName %sample_out "sampleOut"
OpName %offset_out "offsetOut"
OpDecorate %input_value Location 0
OpDecorate %centroid_out Location 0
OpDecorate %sample_out Location 1
OpDecorate %offset_out Location 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%one = OpConstant %int 1
%zero = OpConstant %float 0.0
%offset = OpConstantComposite %v2float %zero %zero
%input_value = OpVariable %ptr_input_float Input
%centroid_out = OpVariable %ptr_output_float Output
%sample_out = OpVariable %ptr_output_float Output
%offset_out = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%centroid = OpExtInst %float %std450 InterpolateAtCentroid %loaded
OpStore %centroid_out %centroid
%sample = OpExtInst %float %std450 InterpolateAtSample %loaded %one
OpStore %sample_out %sample
%offset_value = OpExtInst %float %std450 InterpolateAtOffset %loaded %offset
OpStore %offset_out %offset_value
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_DERIVATIVE_OPS_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/glslang
; Source commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
; Source path: Test/baseResults/spv.computeShaderDerivatives.comp.out
; Reduced from scalar DPdx/DPdyFine/FwidthCoarse derivative stores.
OpCapability Shader
OpCapability DerivativeControl
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_value %dx_out %dy_out %width_out
OpExecutionMode %main OriginUpperLeft
OpName %input_value "inputValue"
OpName %dx_out "dxOut"
OpName %dy_out "dyOut"
OpName %width_out "widthOut"
OpDecorate %input_value Location 0
OpDecorate %dx_out Location 0
OpDecorate %dy_out Location 1
OpDecorate %width_out Location 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%ptr_input_float = OpTypePointer Input %float
%ptr_output_float = OpTypePointer Output %float
%input_value = OpVariable %ptr_input_float Input
%dx_out = OpVariable %ptr_output_float Output
%dy_out = OpVariable %ptr_output_float Output
%width_out = OpVariable %ptr_output_float Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %float %input_value
%dx = OpDPdx %float %loaded
OpStore %dx_out %dx
%dy_fine = OpDPdyFine %float %loaded
OpStore %dy_out %dy_fine
%width_coarse = OpFwidthCoarse %float %loaded
OpStore %width_out %width_coarse
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_GEOMETRY_EXECUTION_MODES_ASSEMBLY = """
; Reduced from glslangValidator-style SPIR-V assembly for:
; layout(triangles, invocations = 2) in;
; layout(triangle_strip, max_vertices = 3) out;
OpCapability Shader
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main" %in_color %out_color
OpExecutionMode %main Triangles
OpExecutionMode %main Invocations 2
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 3
OpName %in_color "inColor"
OpName %out_color "outColor"
OpDecorate %in_color Location 0
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v3float = OpTypeVector %float 3
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%arr_v3float_3 = OpTypeArray %v3float %uint_3
%ptr_input_arr_v3float_3 = OpTypePointer Input %arr_v3float_3
%ptr_output_v3float = OpTypePointer Output %v3float
%in_color = OpVariable %ptr_input_arr_v3float_3 Input
%out_color = OpVariable %ptr_output_v3float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_TESSELLATION_EVALUATION_EXECUTION_MODES_ASSEMBLY = """
OpCapability Tessellation
OpMemoryModel Logical GLSL450
OpEntryPoint TessellationEvaluation %main "main"
OpExecutionMode %main Triangles
OpExecutionMode %main SpacingEqual
OpExecutionMode %main VertexOrderCcw
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_CROSS_GEOMETRY_EMIT_PRIMITIVE_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Cross
; Source commit: 146679ff8255a6068518685599d7fb8761d1b570
; Source path: shaders/asm/geom/unroll-glposition-load.asm.geom
; Reduced from the geometry helper body containing OpEmitVertex and OpEndPrimitive.
OpCapability Geometry
OpMemoryModel Logical GLSL450
OpEntryPoint Geometry %main "main"
OpExecutionMode %main Triangles
OpExecutionMode %main OutputTriangleStrip
OpExecutionMode %main OutputVertices 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%label = OpLabel
OpEmitVertex
OpEndPrimitive
OpReturn
OpFunctionEnd
"""

SPIRV_CROSS_ISNAN_ISINF_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Cross
; Source commit: 146679ff8255a6068518685599d7fb8761d1b570
; Source path: shaders/asm/comp/logical.asm.comp
; Reduced from OpIsInf/OpIsNan predicate stores in main.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %nan_out "nanOut"
OpName %inf_out "infOut"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%ptr_function_bool = OpTypePointer Function %bool
%input_value = OpConstant %float 1.0
%main = OpFunction %void None %fn
%label = OpLabel
%nan_out = OpVariable %ptr_function_bool Function
%inf_out = OpVariable %ptr_function_bool Function
%is_nan = OpIsNan %bool %input_value
%is_inf = OpIsInf %bool %input_value
OpStore %nan_out %is_nan
OpStore %inf_out %is_inf
OpReturn
OpFunctionEnd
"""

SPIRV_LOCAL_SIZE_ID_ASSEMBLY = """
; Reduced from specialization-driven compute local sizes.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionModeId %main LocalSizeId %width %height %depth
OpName %width "WORKGROUP_WIDTH"
OpName %height "WORKGROUP_HEIGHT"
OpDecorate %width SpecId 0
OpDecorate %height SpecId 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%width = OpSpecConstant %uint 8
%height = OpSpecConstant %uint 4
%depth = OpConstant %uint 1
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_FLAT_LOCATION_ASSEMBLY = """
; Reduced from Khronos SPIRV-Tools test/val/val_image_test.cpp CommonTypes.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %input_flat_u32
OpExecutionMode %main OriginUpperLeft
OpName %input_flat_u32 "input_flat_u32"
OpDecorate %input_flat_u32 Flat
OpDecorate %input_flat_u32 Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%ptr_input_u32 = OpTypePointer Input %u32
%input_flat_u32 = OpVariable %ptr_input_u32 Input
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_SPEC_NOPERSPECTIVE_INTERFACE_ASSEMBLY = """
; Reduced from the Khronos SPIR-V spec section 1.10 example.
; The source declares: noperspective in vec4 color2;
; The corresponding assembly decorates the input with NoPerspective and no Location.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color2
OpExecutionMode %main OriginLowerLeft
OpName %color2 "color2"
OpDecorate %color2 NoPerspective
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_v4float = OpTypePointer Input %v4float
%color2 = OpVariable %ptr_input_v4float Input
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_RELAXED_PRECISION_INTERFACE_ASSEMBLY = """
; Reduced from glslangValidator -V -H output for ESSL 310 mediump fragment IO.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color %in_color
OpExecutionMode %main OriginUpperLeft
OpName %out_color "outColor"
OpName %in_color "color"
OpDecorate %out_color RelaxedPrecision
OpDecorate %out_color Location 0
OpDecorate %in_color RelaxedPrecision
OpDecorate %in_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%ptr_input_v4float = OpTypePointer Input %v4float
%out_color = OpVariable %ptr_output_v4float Output
%in_color = OpVariable %ptr_input_v4float Input
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %v4float %in_color
OpStore %out_color %loaded
OpReturn
OpFunctionEnd
"""

SPIRV_VERTEX_ID_INSTANCE_ID_BUILTINS_ASSEMBLY = """
; Reduced from legacy GLSL vertex built-ins accepted by SPIR-V assembly tools.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %vertex_id %instance_id %position
OpDecorate %vertex_id BuiltIn VertexId
OpDecorate %instance_id BuiltIn InstanceId
OpDecorate %position BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_input_int = OpTypePointer Input %int
%ptr_output_v4float = OpTypePointer Output %v4float
%vertex_id = OpVariable %ptr_input_int Input
%instance_id = OpVariable %ptr_input_int Input
%position = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_FRAGMENT_SAMPLE_AND_VIEW_BUILTINS_ASSEMBLY = """
; Reduced from Vulkan fragment interface built-ins found in SwiftShader/glslang output.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %sample_id %sample_position %sample_mask_in %layer %viewport
OpExecutionMode %main OriginUpperLeft
OpDecorate %sample_id BuiltIn SampleId
OpDecorate %sample_position BuiltIn SamplePosition
OpDecorate %sample_mask_in BuiltIn SampleMask
OpDecorate %layer BuiltIn Layer
OpDecorate %viewport BuiltIn ViewportIndex
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%one = OpConstant %int 1
%sample_mask_array = OpTypeArray %int %one
%ptr_input_int = OpTypePointer Input %int
%ptr_input_vec2 = OpTypePointer Input %v2float
%ptr_input_mask = OpTypePointer Input %sample_mask_array
%sample_id = OpVariable %ptr_input_int Input
%sample_position = OpVariable %ptr_input_vec2 Input
%sample_mask_in = OpVariable %ptr_input_mask Input
%layer = OpVariable %ptr_input_int Input
%viewport = OpVariable %ptr_input_int Input
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_FRAGMENT_SAMPLE_MASK_BODY_ASSEMBLY = """
; Reduced from a fragment shader that copies gl_SampleMaskIn[0] to gl_SampleMask[0].
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %sample_mask_in %sample_mask_out
OpExecutionMode %main OriginUpperLeft
OpDecorate %sample_mask_in BuiltIn SampleMask
OpDecorate %sample_mask_out BuiltIn SampleMask
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%zero = OpConstant %int 0
%one = OpConstant %int 1
%sample_mask_array = OpTypeArray %int %one
%ptr_input_mask = OpTypePointer Input %sample_mask_array
%ptr_output_mask = OpTypePointer Output %sample_mask_array
%ptr_input_int = OpTypePointer Input %int
%ptr_output_int = OpTypePointer Output %int
%sample_mask_in = OpVariable %ptr_input_mask Input
%sample_mask_out = OpVariable %ptr_output_mask Output
%main = OpFunction %void None %fn
%label = OpLabel
%in_mask0 = OpAccessChain %ptr_input_int %sample_mask_in %zero
%mask0 = OpLoad %int %in_mask0
%out_mask0 = OpAccessChain %ptr_output_int %sample_mask_out %zero
OpStore %out_mask0 %mask0
OpReturn
OpFunctionEnd
"""

SPIRV_FRAGMENT_HELPER_INVOCATION_BUILTIN_ASSEMBLY = """
; Reduced from a fragment shader interface that reads gl_HelperInvocation.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %helper %out_flag
OpExecutionMode %main OriginUpperLeft
OpName %out_flag "outFlag"
OpDecorate %helper BuiltIn HelperInvocation
OpDecorate %out_flag Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%ptr_input_bool = OpTypePointer Input %bool
%ptr_output_bool = OpTypePointer Output %bool
%helper = OpVariable %ptr_input_bool Input
%out_flag = OpVariable %ptr_output_bool Output
%main = OpFunction %void None %fn
%label = OpLabel
%loaded = OpLoad %bool %helper
OpStore %out_flag %loaded
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_FRAGMENT_BARYCENTRIC_ASSEMBLY = """
; Source repo: https://chromium.googlesource.com/external/github.com/KhronosGroup/glslang.git
; Source ref: refs/heads/vulkan-sdk-1.3.275
; Source path: Test/baseResults/spv.fragmentShaderBarycentric4.frag.out
; Reduced from gl_BaryCoordNoPerspEXT and PerVertexKHR fragment inputs.
OpCapability Shader
OpCapability FragmentBarycentricKHR
OpExtension "SPV_KHR_fragment_shader_barycentric"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_value %bary_coord %vertex_ids
OpExecutionMode %main OriginUpperLeft
OpName %out_value "outValue"
OpName %bary_coord "gl_BaryCoordNoPerspEXT"
OpName %vertex_ids "vertexIDs"
OpDecorate %out_value Location 0
OpDecorate %bary_coord BuiltIn BaryCoordNoPerspKHR
OpDecorate %vertex_ids Location 1
OpDecorate %vertex_ids PerVertexKHR
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%v3float = OpTypeVector %float 3
%arr_v3float_3 = OpTypeArray %v3float %uint_3
%ptr_output_float = OpTypePointer Output %float
%ptr_input_v3float = OpTypePointer Input %v3float
%ptr_input_arr_v3float_3 = OpTypePointer Input %arr_v3float_3
%out_value = OpVariable %ptr_output_float Output
%bary_coord = OpVariable %ptr_input_v3float Input
%vertex_ids = OpVariable %ptr_input_arr_v3float_3 Input
%main = OpFunction %void None %fn
%label = OpLabel
%loaded_bary = OpLoad %v3float %bary_coord
%x = OpCompositeExtract %float %loaded_bary 0
OpStore %out_value %x
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_GLPERVERTEX_ACCESS_CHAIN_ASSEMBLY = """
; Reduced from KhronosGroup/SPIRV-Tools@96545708d0fb060ec6d1e67e85de593bcf24dd21
; test/diff/diff_files/spec_constant_array_size_src.spvasm.
; Generator: Google ANGLE Shader Compiler; 0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %22 "main" %4 %19
OpSource GLSL 450
OpName %4 "_ua_position"
OpName %17 "gl_PerVertex"
OpMemberName %17 0 "gl_Position"
OpMemberName %17 1 "gl_PointSize"
OpMemberName %17 2 "gl_ClipDistance"
OpMemberName %17 3 "gl_CullDistance"
OpName %19 ""
OpName %22 "main"
OpDecorate %4 Location 0
OpMemberDecorate %17 1 RelaxedPrecision
OpMemberDecorate %17 0 BuiltIn Position
OpMemberDecorate %17 1 BuiltIn PointSize
OpMemberDecorate %17 2 BuiltIn ClipDistance
OpMemberDecorate %17 3 BuiltIn CullDistance
OpDecorate %17 Block
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%5 = OpTypeInt 32 0
%8 = OpTypeVector %5 4
%15 = OpConstant %5 8
%16 = OpTypeArray %1 %15
%17 = OpTypeStruct %2 %1 %16 %16
%20 = OpTypeVoid
%25 = OpConstant %5 0
%3 = OpTypePointer Input %2
%13 = OpTypePointer Output %2
%18 = OpTypePointer Output %17
%21 = OpTypeFunction %20
%4 = OpVariable %3 Input
%19 = OpVariable %18 Output
%22 = OpFunction %20 None %21
%23 = OpLabel
%24 = OpLoad %2 %4
%26 = OpAccessChain %13 %19 %25
OpStore %26 %24
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_MULTILINE_OPSOURCE_ASSEMBLY = """
; Source syntax: https://chromium.googlesource.com/external/github.com/KhronosGroup/SPIRV-Tools/+/refs/tags/vulkan-sdk-1.4.304.1/docs/syntax.md
; Source spec: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSource
; Reduced from the SPIRV-Tools literal string syntax where strings can contain
; newlines and an OpSource instruction can carry source-level text.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %pos
OpSource GLSL 450 "#version 450
void main() {
    gl_Position = vec4(0.0);
}
"
OpName %pos "pos"
OpDecorate %pos Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%pos = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%label = OpLabel
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_PTR_ACCESS_CHAIN_ASSEMBLY = """
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/opt/combine_access_chains_test.cpp
; Reduced from CombineAccessChainsTest OpPtrAccessChain fixtures and adapted
; to store through the derived pointers so CrossGL output exposes the lowering.
OpCapability Shader
OpCapability VariablePointers
OpCapability Addresses
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_a %out_b
OpExecutionMode %main OriginUpperLeft
OpName %workgroup_values "workgroupValues"
OpName %out_a "outA"
OpName %out_b "outB"
OpName %ptr_access "ptrAccess"
OpName %in_bounds_ptr "inBoundsPtr"
OpDecorate %out_a Location 0
OpDecorate %out_b Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%arr_uint = OpTypeArray %uint %uint_4
%arr_arr_uint = OpTypeArray %arr_uint %uint_4
%ptr_workgroup_uint = OpTypePointer Workgroup %uint
%ptr_workgroup_arr_uint = OpTypePointer Workgroup %arr_uint
%ptr_workgroup_arr_arr_uint = OpTypePointer Workgroup %arr_arr_uint
%ptr_output_uint = OpTypePointer Output %uint
%workgroup_values = OpVariable %ptr_workgroup_arr_arr_uint Workgroup
%out_a = OpVariable %ptr_output_uint Output
%out_b = OpVariable %ptr_output_uint Output
%main = OpFunction %void None %fn
%label = OpLabel
%access = OpAccessChain %ptr_workgroup_arr_uint %workgroup_values %uint_1
%ptr_access = OpPtrAccessChain %ptr_workgroup_uint %access %uint_2
%value = OpLoad %uint %ptr_access
OpStore %out_a %value
%in_bounds_ptr = OpInBoundsPtrAccessChain %ptr_workgroup_uint %workgroup_values %uint_0 %uint_1
%value_in_bounds = OpLoad %uint %in_bounds_ptr
OpStore %out_b %value_in_bounds
OpReturn
OpFunctionEnd
"""

SPIRV_GLSLANG_WORKGROUP_SHARED_ASSEMBLY = """
; Reduced from glslangValidator -V -H output for GLSL compute:
; shared uint sharedData[4]; sharedData[0] = 7u.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %shared_data "sharedData"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_7 = OpConstant %uint 7
%arr_uint = OpTypeArray %uint %uint_4
%ptr_workgroup_uint = OpTypePointer Workgroup %uint
%ptr_workgroup_arr_uint = OpTypePointer Workgroup %arr_uint
%shared_data = OpVariable %ptr_workgroup_arr_uint Workgroup
%main = OpFunction %void None %fn
%label = OpLabel
%element = OpAccessChain %ptr_workgroup_uint %shared_data %uint_0
OpStore %element %uint_7
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_EXTENDED_ARITHMETIC_ASSEMBLY = """
; Source grammar: https://github.com/KhronosGroup/SPIRV-Headers/blob/1e770e7de8373a8dd49f23416cf7ca4001d01040/include/spirv/unified1/spirv.core.grammar.json
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/val/val_arithmetics_test.cpp
; Reduced from ValidateArithmetics IAddCarry/SMulExtended success fixtures.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_sum %out_carry %out_low %out_high
OpExecutionMode %main OriginUpperLeft
OpName %out_sum "outSum"
OpName %out_carry "outCarry"
OpName %out_low "outLow"
OpName %out_high "outHigh"
OpDecorate %out_sum Location 0
OpDecorate %out_carry Location 1
OpDecorate %out_low Location 2
OpDecorate %out_high Location 3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%int_2 = OpConstant %int 2
%int_3 = OpConstant %int 3
%struct_u32_u32 = OpTypeStruct %uint %uint
%struct_i32_i32 = OpTypeStruct %int %int
%ptr_output_uint = OpTypePointer Output %uint
%ptr_output_int = OpTypePointer Output %int
%out_sum = OpVariable %ptr_output_uint Output
%out_carry = OpVariable %ptr_output_uint Output
%out_low = OpVariable %ptr_output_int Output
%out_high = OpVariable %ptr_output_int Output
%main = OpFunction %void None %fn
%label = OpLabel
%add = OpIAddCarry %struct_u32_u32 %uint_0 %uint_1
%sum = OpCompositeExtract %uint %add 0
%carry = OpCompositeExtract %uint %add 1
OpStore %out_sum %sum
OpStore %out_carry %carry
%mul = OpSMulExtended %struct_i32_i32 %int_2 %int_3
%low = OpCompositeExtract %int %mul 0
%high = OpCompositeExtract %int %mul 1
OpStore %out_low %low
OpStore %out_high %high
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_SAME_VALUE_PHI_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/opt/dead_insert_elim_test.cpp
; Reduced from the same-value OpPhi pattern: %41 = OpPhi %S %4 %if_true %4 %if_merge.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color
OpExecutionMode %main OriginUpperLeft
OpName %out_color "outColor"
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%zero = OpConstant %float 0.0
%one = OpConstant %float 1.0
%red = OpConstantComposite %v4float %one %zero %zero %one
%true = OpConstantTrue %bool
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpSelectionMerge %merge None
OpBranchConditional %true %if_true %if_merge
%if_true = OpLabel
OpBranch %merge
%if_merge = OpLabel
OpBranch %merge
%merge = OpLabel
%phi = OpPhi %v4float %red %if_true %red %if_merge
OpStore %out_color %phi
OpReturn
OpFunctionEnd
"""

SPIRV_TOOLS_DIRECT_SELECTION_PHI_ASSEMBLY = """
; Source repo: https://github.com/KhronosGroup/SPIRV-Tools
; Source commit: 9b51d3d78717e29efd75adf1856cdbcc644eda7a
; Source path: test/opt/block_merge_test.cpp
; Reduced from PhiInSuccessorOfMergedBlock's direct OpPhi merge pattern.
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_color
OpExecutionMode %main OriginUpperLeft
OpName %out_color "outColor"
OpDecorate %out_color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%ptr_output_v4float = OpTypePointer Output %v4float
%zero = OpConstant %float 0.0
%one = OpConstant %float 1.0
%red = OpConstantComposite %v4float %one %zero %zero %one
%blue = OpConstantComposite %v4float %zero %zero %one %one
%true = OpConstantTrue %bool
%out_color = OpVariable %ptr_output_v4float Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpSelectionMerge %merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %merge
%else = OpLabel
OpBranch %merge
%merge = OpLabel
%phi = OpPhi %v4float %red %then %blue %else
OpStore %out_color %phi
OpReturn
OpFunctionEnd
"""


def test_vulkan_to_crossgl_emits_fragment_main():
    tokens = tokenize_code(FRAGMENT_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(ast.functions) == 1
    assert ast.functions[0].return_type == "void"
    assert ast.functions[0].name == "main"
    assert "fragment {" in generated_code
    assert "void main()" in generated_code
    assert "float4 color = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "gl_FragColor = color;" in generated_code


def test_vulkan_void_parameter_main_codegen():
    code = """
    void main(void) {
        gl_FragColor = vec4(1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.functions[0].params == []
    assert "void main()" in generated_code
    assert "void main(void)" not in generated_code
    assert "gl_FragColor = float4(1.0);" in generated_code


def test_float_suffix_literals_codegen_from_vulkan_sample_style():
    code = """
    void main() {
        vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
        gl_Position = vec4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
        float tiny = 2.3283064365386963e-10;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "gl_Position = float4(((uv * 2.0) - 1.0), 0.0, 1.0);" in generated_code
    assert "float tiny = 2.3283064365386963e-10;" in generated_code
    assert "2.0f" not in generated_code
    assert "0.0f" not in generated_code
    assert "1.0f" not in generated_code


def test_standalone_function_call_codegen():
    code = """
    void helper(int amount, int current) {
    }

    void main() {
        int value = 1;
        helper(1, value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "helper(1, value);" in generated_code
    assert "int value = 1;" in generated_code
    assert " helper;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_member_call_statement_codegen():
    code = """
    void main() {
        image.store(1, value);
        int result = object.method();
        objects[0].store(value);
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "image.store(1, value);" in generated_code
    assert "int result = object.method();" in generated_code
    assert "objects[0].store(value);" in generated_code
    assert " image.store;" not in generated_code
    assert "int value = 1;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_function_parameter_qualifiers_codegen_smoke():
    code = """
    void accumulate(in vec3 normal, inout float weight, out vec4 color) {
        color = vec4(normal, weight);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void accumulate(in float3 normal, inout float weight, out float4 color)"
        in generated_code
    )
    assert "color = float4(normal, weight);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_resource_array_parameter_qualifiers_codegen():
    code = """
    void sampleResources(in sampler2D textures[4], inout uimage2D outputs[2]) {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void sampleResources(in Texture2D textures[4], "
        "inout RWTexture2D<uint> outputs[2])" in generated_code
    )


def test_function_parameter_array_suffixes_codegen():
    code = """
    void sampleResources(sampler2D textures[4], uimage2D outputs[2]) {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void sampleResources(Texture2D textures[4], RWTexture2D<uint> outputs[2])"
        in generated_code
    )


def test_member_access_assignment_codegen():
    code = """
    void main() {
        color.r = 1.0;
        color.g += 0.5;
        objects[0].field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "color.r = 1.0;" in generated_code
    assert "color.g += 0.5;" in generated_code
    assert "objects[0].field = value;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_assignment_expression_associativity_codegen():
    code = """
    void main() {
        a = b = c;
        a += b = c;
        int value = a = b;
        int grouped = (a = b) + c;
        int rightGrouped = a + (b = c);
        bool condition = (a = b) ? yes : no;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "a = b = c;" in generated_code
    assert "a += b = c;" in generated_code
    assert "int value = a = b;" in generated_code
    assert "int grouped = ((a = b) + c);" in generated_code
    assert "int rightGrouped = (a + (b = c));" in generated_code
    assert "bool condition = ((a = b) ? yes : no);" in generated_code


def test_typed_local_array_declaration_codegen():
    code = """
    void main() {
        float weights[4];
        weights[0] = 1.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float weights[4];" in generated_code
    assert "weights[0] = 1.0;" in generated_code
    assert "float weights;" not in generated_code


def test_translate_api_accepts_spirv_source(tmp_path):
    import crosstl

    shader_path = tmp_path / "fragment.spvasm"
    shader_path.write_text(FRAGMENT_SHADER, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="rust", format_output=False
    )

    assert '#[cfg_attr(feature = "crossgl_gpu", fragment_shader)]' in generated_code
    assert "pub fn main()" in generated_code
    assert (
        "let color: Vec4<f32> = Vec4::<f32>::new(1.0, 0.0, 0.0, 1.0);" in generated_code
    )


def test_translate_api_rejects_binary_spv_source_with_clear_error(tmp_path):
    import crosstl

    shader_path = tmp_path / "fragment.spv"
    shader_path.write_bytes(b"\x03\x02\x23\x07")

    with pytest.raises(ValueError, match=re.escape(BINARY_SPIRV_UNSUPPORTED_MESSAGE)):
        crosstl.translate(str(shader_path), backend="rust", format_output=False)


def test_translate_api_rejects_binary_spv_mislabeled_as_spvasm(tmp_path):
    import crosstl

    shader_path = tmp_path / "fragment.spvasm"
    shader_path.write_bytes(b"\x03\x02\x23\x07")

    with pytest.raises(ValueError, match=re.escape(BINARY_SPIRV_UNSUPPORTED_MESSAGE)):
        crosstl.translate(str(shader_path), backend="rust", format_output=False)


def test_spirv_assembly_location_decorated_interfaces_codegen():
    tokens = tokenize_code(SPIRV_TOOLS_BASIC_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 _ua_position @input @location(0);" in generated_code
    assert "float4 ANGLEXfbPosition @output @location(0);" in generated_code
    assert "float4 gl_Position @output @gl_Position;" in generated_code
    assert "%4" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_non_main_entrypoint_reparse_preserves_stage_body():
    tokens = tokenize_code(SPIRV_NON_MAIN_VERTEX_ENTRY_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void vs_main() @stage_entry" in generated_code
    assert "gl_Position = float4(0, 0, 0, 1);" in generated_code

    reparsed = parse_crossgl(generated_code)
    vertex_stage = reparsed.stages[ShaderStage.VERTEX]

    assert vertex_stage.entry_point.name == "vs_main"
    assert len(vertex_stage.entry_point.body.statements) == 2
    assert vertex_stage.local_functions == []

    downstream_code = VulkanSPIRVCodeGen().generate(reparsed)

    assert re.search(r'OpEntryPoint Vertex %\d+ "vs_main"', downstream_code)
    assert downstream_code.count("OpFunction ") == 1
    assert "OpStore" in downstream_code


def test_spirv_assembly_multi_entrypoint_interfaces_scope_locations():
    tokens = tokenize_code(SPIRV_MULTI_ENTRYPOINT_REUSED_LOCATION_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 vertexIn @input @location(0);" in generated_code
    assert "float4 vColor @output @location(0);" in generated_code
    assert "float4 fColor @input @location(0);" in generated_code
    assert "float4 fragOut @output @location(0);" in generated_code
    assert generated_code.index("float4 vertexIn") < generated_code.index(
        "void vs_main() @stage_entry"
    )
    assert generated_code.index("float4 fColor") < generated_code.index(
        "void fs_main() @stage_entry"
    )

    reparsed = parse_crossgl(generated_code)
    downstream_code = VulkanSPIRVCodeGen().generate(reparsed)

    assert re.search(r'OpEntryPoint Vertex %\d+ "vs_main"', downstream_code)
    assert re.search(r'OpEntryPoint Fragment %\d+ "fs_main"', downstream_code)
    assert downstream_code.count("OpFunction ") == 2


def test_spirv_assembly_matrix_interface_codegen():
    tokens = tokenize_code(SPIRV_MATRIX_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4x4 model @input @location(0);" in generated_code
    assert "%model" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_private_global_variables_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_PRIVATE_GLOBAL_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outColor @output @location(0);" in generated_code
    assert "float4 privateColor;" in generated_code
    assert "float privateWeight = 1.0;" in generated_code
    assert "privateColor = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "outColor = privateColor;" in generated_code
    assert "outColor = loaded;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_simple_mat_matrix_times_vector_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_SIMPLE_MAT_MATRIX_TIMES_VECTOR_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 glPos @output @location(5);" in generated_code
    assert "float4x4 mvp @output @location(0);" in generated_code
    assert "float4 v @input @location(0);" in generated_code
    assert "glPos = (mvp * v);" in generated_code
    assert "glPos = transformed;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_matrix_transpose_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_MATRIX_TRANSPOSE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float3x4 sum34 @input @location(0);" in generated_code
    assert "float4x3 m43 @output @location(0);" in generated_code
    assert "m43 = transpose(sum34);" in generated_code
    assert "m43 = transposed;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_outer_product_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_OUTER_PRODUCT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float3 column @input @location(0);" in generated_code
    assert "float2 row @input @location(1);" in generated_code
    assert "float3x2 basis @output @location(0);" in generated_code
    assert "basis = outerProduct(column, row);" in generated_code
    assert "basis = outer;" not in generated_code
    assert "spirv_OuterProduct" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_fconvert_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_FCONVERT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float inputValue @input @location(0);" in generated_code
    assert "double doubleOut @output @location(0);" in generated_code
    assert "doubleOut = double(inputValue);" in generated_code
    assert "doubleOut = converted;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_float16_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_FLOAT16_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "half4 inputHalf @input @location(0);" in generated_code
    assert "half2x2 inputMatrix @input @location(1);" in generated_code
    assert "half4 halfOut @output @location(0);" in generated_code
    assert "float floatOut @output @location(1);" in generated_code
    assert "half2x2 matrixOut @output @location(2);" in generated_code
    assert "halfOut = inputHalf;" in generated_code
    assert "matrixOut = inputMatrix;" in generated_code
    assert "floatOut = float(inputHalf[0]);" in generated_code
    assert "float4 inputHalf" not in generated_code
    assert "float4 halfOut" not in generated_code
    assert "half2 inputMatrix" not in generated_code
    assert "floatOut = wide;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_int16_int64_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_INT16_INT64_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "vec4<i16> signed16In @input @location(0);" in generated_code
    assert "vec3<u16> unsigned16In @input @location(1);" in generated_code
    assert "vec4<i16> signed16Out @output @location(0);" in generated_code
    assert "vec3<u16> unsigned16Out @output @location(1);" in generated_code
    assert "i64 signed64Out @output @location(2);" in generated_code
    assert "u64 unsigned64Out @output @location(3);" in generated_code
    assert "signed16Out = signed16In;" in generated_code
    assert "unsigned16Out = unsigned16In;" in generated_code
    assert "signed64Out = i64(signed16In[0]);" in generated_code
    assert "unsigned64Out = u64(unsigned16In[1]);" in generated_code
    assert "ivec4 signed16In" not in generated_code
    assert "uvec3 unsigned16In" not in generated_code
    assert "int signed64Out" not in generated_code
    assert "uint unsigned64Out" not in generated_code
    assert "signed64Out = wide_signed;" not in generated_code
    assert "unsigned64Out = wide_unsigned;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_int8_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_INT8_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "vec4<i8> signed8In @input @location(0);" in generated_code
    assert "vec2<u8> unsigned8In @input @location(1);" in generated_code
    assert "vec4<i8> signed8Out @output @location(0);" in generated_code
    assert "vec2<u8> unsigned8Out @output @location(1);" in generated_code
    assert "i8 signed8ScalarOut @output @location(2);" in generated_code
    assert "u8 unsigned8ScalarOut @output @location(3);" in generated_code
    assert "signed8Out = signed8In;" in generated_code
    assert "unsigned8Out = unsigned8In;" in generated_code
    assert "signed8ScalarOut = signed8In[0];" in generated_code
    assert "unsigned8ScalarOut = unsigned8In[1];" in generated_code
    assert "ivec4 signed8In" not in generated_code
    assert "uvec2 unsigned8In" not in generated_code
    assert "int signed8ScalarOut" not in generated_code
    assert "uint unsigned8ScalarOut" not in generated_code
    assert "signed8ScalarOut = signed_component;" not in generated_code
    assert "unsigned8ScalarOut = unsigned_component;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_precise_dot_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_PRECISE_DOT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float outValue @output @location(0);" in generated_code
    assert "float4 inputVec @input @location(0);" in generated_code
    assert "outValue = dot(inputVec, inputVec);" in generated_code
    assert "outValue = dot;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_integer_dot_product_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_INTEGER_DOT_PRODUCT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int4 inputA @input @location(0);" in generated_code
    assert "int4 inputB @input @location(1);" in generated_code
    assert "int outValue @output @location(0);" in generated_code
    assert "outValue = spirvSDot(inputA, inputB);" in generated_code
    assert "outValue = dot;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_vector_extract_dynamic_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_VECTOR_EXTRACT_DYNAMIC_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 inputVec @input @location(0);" in generated_code
    assert "uint index @input @location(1) @flat;" in generated_code
    assert "float outValue @output @location(0);" in generated_code
    assert "outValue = inputVec[index];" in generated_code
    assert "outValue = component;" not in generated_code
    assert "OpVectorExtractDynamic" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_vector_negate_then_extract_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_VECTOR_NEGATE_EXTRACT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 inputVec @input @location(0);" in generated_code
    assert "float outValue @output @location(0);" in generated_code
    assert "outValue = (-inputVec)[0];" in generated_code
    assert "outValue = -inputVec[0];" not in generated_code
    assert "outValue = component;" not in generated_code
    assert "OpFNegate" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_struct_composite_extract_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_STRUCT_COMPOSITE_EXTRACT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "struct Pair" in generated_code
    assert "float weight;" in generated_code
    assert "uint index;" in generated_code
    assert "Pair pair;" in generated_code
    assert "outValue = pair.weight;" in generated_code
    assert "outValue = pair[0];" not in generated_code
    assert "OpCompositeExtract" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_array_return_composite_construct_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_ARRAY_RETURN_COMPOSITE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4[2] ff()" in generated_code
    assert (
        "return {float4(0.0, 0.0, 0.0, 0.0), "
        "float4(1.0, 1.0, 1.0, 1.0)};" in generated_code
    )
    assert "outValue = ff()[1][2];" in generated_code
    assert "%array_v4" not in generated_code
    assert "OpCompositeConstruct" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_vector_insert_dynamic_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_VECTOR_INSERT_DYNAMIC_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 inputVec @input @location(0);" in generated_code
    assert "float insertValue @input @location(1);" in generated_code
    assert "uint index @input @location(2) @flat;" in generated_code
    assert "float4 outVec @output @location(0);" in generated_code
    assert (
        "outVec = spirvVectorInsertDynamic(inputVec, insertValue, index);"
        in generated_code
    )
    assert "outVec = inserted;" not in generated_code
    assert "OpVectorInsertDynamic" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_undef_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_UNDEF_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outVec @output @location(0);" in generated_code
    assert generated_code.count("outVec = spirvUndef_vec4();") == 2
    assert "outVec = undefValue;" not in generated_code
    assert "outVec = bodyUndef;" not in generated_code
    assert "OpUndef" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_constant_null_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_CONSTANT_NULL_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outVec @output @location(0);" in generated_code
    assert generated_code.count("outVec = spirvNull_vec4();") == 2
    assert "outVec = nullVec;" not in generated_code
    assert "outVec = bodyNull;" not in generated_code
    assert "OpConstantNull" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_fragment_termination_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_FRAGMENT_TERMINATION_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert generated_code.count("discard;") == 3
    assert "OpKill" not in generated_code
    assert "OpTerminateInvocation" not in generated_code
    assert "OpDemoteToHelperInvocation" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_glslang_void_function_call_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_VOID_FUNCTION_CALL_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outColor @output @location(0);" in generated_code
    assert "helper();" in generated_code
    assert "outColor = float4(1.0, 1.0, 1.0, 1.0);" in generated_code
    assert "void helper()" in generated_code
    assert "OpFunctionCall" not in generated_code
    assert "call;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_glslang_pointer_parameter_function_call_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_POINTER_PARAMETER_FUNCTION_CALL_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outColor @output @location(0);" in generated_code
    assert "void helper(inout float4 value)" in generated_code
    assert "value = -value;" in generated_code
    assert "float4 local = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "helper(local);" in generated_code
    assert "outColor = local;" in generated_code
    assert "float4* value" not in generated_code
    assert "outColor = local_value;" not in generated_code
    assert "OpFunctionParameter" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_glslang_selection_merge_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_SELECTION_MERGE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float value @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "if ((value > 0.0)) {" in generated_code
    assert "outColor = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "} else {" in generated_code
    assert "outColor = float4(0.0, 0.0, 1.0, 1.0);" in generated_code
    assert "OpSelectionMerge" not in generated_code
    assert "OpBranchConditional" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_glslang_switch_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_SWITCH_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int mode @input @location(0) @flat;" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "switch (mode) {" in generated_code
    assert "case 0:" in generated_code
    assert "outColor = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "case 1:" in generated_code
    assert "outColor = float4(0.0, 1.0, 0.0, 1.0);" in generated_code
    assert "default:" in generated_code
    assert "outColor = float4(0.0, 0.0, 1.0, 1.0);" in generated_code
    assert generated_code.count("break;") == 3
    assert "OpSwitch" not in generated_code
    assert "OpSelectionMerge" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_glslang_grouped_switch_targets_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_GROUPED_SWITCH_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int mode @input @location(0) @flat;" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "switch (mode) {" in generated_code
    assert generated_code.index("case 0:") < generated_code.index("case 1:")
    assert generated_code.index("case 1:") < generated_code.index(
        "outColor = float4(1.0, 0.0, 0.0, 1.0);"
    )
    assert generated_code.count("outColor = float4(1.0, 0.0, 0.0, 1.0);") == 1
    assert generated_code.count("outColor = float4(0.0, 0.0, 1.0, 1.0);") == 1
    assert generated_code.count("break;") == 2
    assert "OpSwitch" not in generated_code
    assert "OpSelectionMerge" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_simple_loop_merge_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_SIMPLE_LOOP_MERGE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int outValue @output @location(0);" in generated_code
    assert "int counter = 0;" in generated_code
    assert "while ((counter < 4)) {" in generated_code
    assert "outValue = counter;" in generated_code
    assert "counter = (counter + 1);" in generated_code
    assert generated_code.index("outValue = counter;") < generated_code.index(
        "counter = (counter + 1);"
    )
    assert "OpLoopMerge" not in generated_code
    assert "OpBranchConditional" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_opselect_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_OPSELECT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "outv = ((cond < 8) ? 1.0 : 2.0);" in generated_code
    assert "outv = (outv * float(((cond > 0) ? iv1 : iv2)[2]));" in generated_code
    assert "outv = (outv * ((cond < 20) ? m1 : m2)[2][1]);" in generated_code
    assert "fv = ((cond > 5) ? in1 : in2);" in generated_code
    assert "outv = (outv * fv.a);" in generated_code
    assert "selected_float" not in generated_code
    assert "selected_vec" not in generated_code
    assert "selected_mat" not in generated_code
    assert "selected_struct" not in generated_code
    assert "fv[0]" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_any_all_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_ANY_ALL_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "bool anyOut @output @location(0);" in generated_code
    assert "bool allOut @output @location(1);" in generated_code
    assert "anyOut = any((value != value));" in generated_code
    assert "allOut = all((value == value));" in generated_code
    assert "anyOut = hasAny;" not in generated_code
    assert "allOut = hasAll;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_function_local_array_variable_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_LOCAL_ARRAY_VARIABLE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int arr[2];" in generated_code
    assert "%arr_int_2 arr;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_push_constant_block_codegen():
    tokens = tokenize_code(SPIRV_PUSH_CONSTANT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "cbuffer PushConstants @push_constant {" in generated_code
    assert "float4x4 model;" in generated_code
    assert "float4 tint;" in generated_code
    assert "%pc" not in generated_code


def test_spirv_assembly_uniform_block_codegen():
    tokens = tokenize_code(SPIRV_UNIFORM_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "cbuffer Camera @set(0) @binding(2) {" in generated_code
    assert "float4x4 viewProj;" in generated_code
    assert "float4 tint;" in generated_code
    assert "%camera" not in generated_code


def test_spirv_assembly_buffer_block_codegen():
    tokens = tokenize_code(SPIRV_BUFFER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Data" in generated_code
    assert "float4 value;" in generated_code
    assert "RWStructuredBuffer<Data> data @set(0) @binding(1);" in generated_code
    assert "%data" not in generated_code


def test_spirv_assembly_legacy_buffer_block_member_access_codegen_reparse():
    tokens = tokenize_code(SPIRV_LEGACY_BUFFER_BLOCK_MEMBER_ACCESS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "RWStructuredBuffer<bName> bInst @set(0) @binding(0);" in generated_code
    assert "bInst[0].size = (bInst[0].size + 1);" in generated_code
    assert "bInst[0] = (bInst[0] + 1);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_readonly_buffer_block_codegen():
    tokens = tokenize_code(SPIRV_READONLY_BUFFER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Data" in generated_code
    assert "float4 value;" in generated_code
    assert "StructuredBuffer<Data> data @set(0) @binding(1);" in generated_code
    assert "RWStructuredBuffer<Data> data" not in generated_code


def test_spirv_assembly_runtime_array_buffer_block_codegen():
    tokens = tokenize_code(SPIRV_RUNTIME_ARRAY_BUFFER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct StorageBuffer" in generated_code
    assert "uint header;" in generated_code
    assert "uint payload[];" in generated_code
    assert (
        "RWStructuredBuffer<StorageBuffer> storage @set(0) @binding(4);"
        in generated_code
    )
    assert "%storage" not in generated_code


def test_spirv_tools_anonymous_resource_block_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_ANONYMOUS_RESOURCE_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert (
        "RWStructuredBuffer<BufferOut> value_19 @set(0) @binding(1);" in generated_code
    )
    assert "cbuffer BufferIn @set(0) @binding(0) {" in generated_code
    assert "uint i;" in generated_code
    assert "uint result @output @location(0);" in generated_code
    assert "value_19[0].o = i;" in generated_code
    assert "result = i;" in generated_code
    assert "RWStructuredBuffer<BufferOut> 19" not in generated_code
    assert "value_24" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_array_length_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_ARRAY_LENGTH_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "struct StorageBuffer" in generated_code
    assert "uint payload[];" in generated_code
    assert (
        "RWStructuredBuffer<StorageBuffer> storage @set(0) @binding(0);"
        in generated_code
    )
    assert "uint countOut @output @location(0);" in generated_code
    assert "countOut = spirvArrayLength(storage, 1);" in generated_code
    assert "countOut = length;" not in generated_code
    assert "OpArrayLength" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_specialization_constants_codegen():
    tokens = tokenize_code(SPIRV_SPEC_CONSTANT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const uint MAX_LIGHTS @constant_id(0) = 4;" in generated_code
    assert "const bool ENABLE_SHADOWS @constant_id(1) = true;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_numeric_id_specialization_constant_codegen():
    tokens = tokenize_code(SPIRV_NUMERIC_ID_SPEC_CONSTANT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const uint spec_constant_0 @constant_id(0) = 1;" in generated_code
    assert "const uint 104" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_specialization_constant_composite_and_op_codegen():
    tokens = tokenize_code(SPIRV_SPEC_CONSTANT_COMPOSITE_OP_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SizedBlock" in generated_code
    assert "uint values[(WORKGROUP_WIDTH + WORKGROUP_HEIGHT)];" in generated_code
    assert "const uint WORKGROUP_WIDTH @constant_id(0) = 8;" in generated_code
    assert "const uint WORKGROUP_HEIGHT @constant_id(1) = 4;" in generated_code
    assert (
        "const uint WORKGROUP_TOTAL = (WORKGROUP_WIDTH + WORKGROUP_HEIGHT);"
        in generated_code
    )
    assert (
        "const uint3 gl_WorkGroupSize @gl_WorkGroupSize = "
        "uint3(WORKGROUP_WIDTH, WORKGROUP_HEIGHT, 1);" in generated_code
    )
    assert "OpSpecConstant" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_web_comp_barrier_instructions_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_WEB_COMP_BARRIER_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = 2, local_size_y = 5, local_size_z = 7) in;"
        in generated_code
    )
    assert "spirvControlBarrier(2, 2, 264);" in generated_code
    assert "spirvMemoryBarrier(1, 3400);" in generated_code
    assert "spirvMemoryBarrier(2, 3400);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_named_barrier_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_NAMED_BARRIER_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "compute {" in generated_code
    assert "uint outValue @output @location(0);" in generated_code
    assert (
        "spirvMemoryNamedBarrier(spirvNamedBarrierInitialize(4), 2, 264);"
        in generated_code
    )
    assert "outValue = 4;" in generated_code
    assert "OpMemoryNamedBarrier" not in generated_code
    assert "OpNamedBarrierInitialize" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_subgroup_broadcast_and_reduce_codegen_reparse():
    tokens = tokenize_code(SPIRV_SUBGROUP_BROADCAST_REDUCE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;"
        in generated_code
    )
    assert "uint value @input @location(0);" in generated_code
    assert "uint broadcastOut @output @location(0);" in generated_code
    assert "uint sumOut @output @location(1);" in generated_code
    assert "broadcastOut = subgroupBroadcastFirst(value);" in generated_code
    assert "sumOut = subgroupAdd(value);" in generated_code
    assert "broadcastOut = broadcast;" not in generated_code
    assert "sumOut = sum;" not in generated_code
    assert "OpGroupNonUniform" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_subgroup_broadcast_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_SUBGROUP_BROADCAST_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 inColor @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "outColor = subgroupBroadcast(inColor, 0);" in generated_code
    assert "outColor = broadcast;" not in generated_code
    assert "OpGroupNonUniformBroadcast" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_atomic_load_store_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_ATOMIC_LOAD_STORE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert (
        "RWStructuredBuffer<CounterBlock> counterBlock @set(0) @binding(0);"
        in generated_code
    )
    assert "int oldValue @output @location(0);" in generated_code
    assert "oldValue = atomicLoad(counterBlock[0].counter);" in generated_code
    assert "atomicStore(counterBlock[0].counter, 1);" in generated_code
    assert "oldValue = old;" not in generated_code
    assert "OpAtomicLoad" not in generated_code
    assert "OpAtomicStore" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_web_comp_atomic_iadd_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_WEB_COMP_ATOMIC_IADD_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert (
        "RWStructuredBuffer<CounterBlock> counterBlock @set(0) @binding(0);"
        in generated_code
    )
    assert "int oldValue @output @location(0);" in generated_code
    assert "oldValue = atomicAdd(counterBlock[0].counter, 2);" in generated_code
    assert "oldValue = old;" not in generated_code
    assert "OpAtomicIAdd" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_image_texel_pointer_atomic_iadd_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_IMAGE_TEXEL_POINTER_ATOMIC_IADD_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "RWTexture2D<uint> img @set(0) @binding(0) @r32ui;" in generated_code
    assert "atomicAdd(spirvImageTexelPointer(img, int2(0, 1), 0), 3);" in generated_code
    assert "texelPtr" not in generated_code
    assert "OpImageTexelPointer" not in generated_code
    assert "OpAtomicIAdd" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_web_comp_atomic_compare_exchange_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_WEB_COMP_ATOMIC_COMPARE_EXCHANGE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert (
        "RWStructuredBuffer<CounterBlock> counterBlock @set(0) @binding(0);"
        in generated_code
    )
    assert "int oldValue @output @location(0);" in generated_code
    assert "oldValue = atomicCompSwap(counterBlock[0].counter, 5, 2);" in generated_code
    assert "oldValue = old;" not in generated_code
    assert "OpAtomicCompareExchange" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_atomic_increment_decrement_sub_codegen_reparse():
    tokens = tokenize_code(SPIRV_ATOMIC_INC_DEC_SUB_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert (
        "RWStructuredBuffer<CounterBlock> counterBlock @set(0) @binding(0);"
        in generated_code
    )
    assert "oldValue = atomicAdd(counterBlock[0].counter, 1);" in generated_code
    assert "oldValue = atomicAdd(counterBlock[0].counter, -1);" in generated_code
    assert "oldValue = atomicAdd(counterBlock[0].counter, -2);" in generated_code
    assert "oldValue = old_" not in generated_code
    assert "OpAtomicIIncrement" not in generated_code
    assert "OpAtomicIDecrement" not in generated_code
    assert "OpAtomicISub" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_uniform_constant_resources_codegen():
    tokens = tokenize_code(SPIRV_UNIFORM_CONSTANT_RESOURCE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Texture2D combinedTex @set(0) @binding(0);" in generated_code
    assert "sampler linearSampler @set(0) @binding(1);" in generated_code
    assert "%combined" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_ray_tracing_acceleration_structure_resource_codegen_reparse():
    tokens = tokenize_code(SPIRV_RAY_TRACING_ACCELERATION_STRUCTURE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "accelerationStructureEXT topLevelAS @set(0) @binding(3);" in generated_code
    assert "%top_level_as" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_glslang_ray_query_convert_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_RAY_QUERY_CONVERT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "cbuffer Params @push_constant" in generated_code
    assert "uint2 tlas;" in generated_code
    assert "rayQueryEXT rayQuery;" in generated_code
    assert (
        "rayQueryInitializeEXT(rayQuery, accelerationStructureEXT(tlas), 0, 0, "
        "float3(0.0, 0.0, 0.0), 0.0, float3(1.0, 1.0, 1.0), 1.0);" in generated_code
    )
    assert "rayQueryTerminateEXT(rayQuery);" in generated_code
    assert "OpRayQuery" not in generated_code
    assert "%ray_query" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_storage_image_format_codegen():
    tokens = tokenize_code(SPIRV_STORAGE_IMAGE_FORMAT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "RWTexture2D<uint> storageImage @set(0) @binding(0) @r32ui @readonly;"
        in generated_code
    )
    assert "%storage_image" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_storage_image_array_format_codegen():
    tokens = tokenize_code(SPIRV_STORAGE_IMAGE_ARRAY_FORMAT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "RWTexture2D<uint> storageImages[4] @set(0) @binding(0) @r32ui @readonly;"
        in generated_code
    )
    assert "storageImages[4] @set(0) @binding(0) @readonly;" not in generated_code
    assert "%storage_images" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_non_32bit_integer_images_preserve_resource_family_codegen_reparse():
    tokens = tokenize_code(SPIRV_NON_32BIT_INTEGER_IMAGE_FAMILY_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "usampler2D shortTex @set(0) @binding(0);" in generated_code
    assert (
        "RWTexture2D<int> signedStorage @set(0) @binding(1) @r16i @writeonly;"
        in generated_code
    )
    assert "Texture2D shortTex" not in generated_code
    assert "RWTexture2D signedStorage" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_descriptor_indexing_nonuniform_codegen_reparse():
    tokens = tokenize_code(SPIRV_DESCRIPTOR_INDEXING_NONUNIFORM_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D textures[4] @set(0) @binding(0);" in generated_code
    assert "sampler linearSampler @set(0) @binding(1);" in generated_code
    assert "int materialIndex @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = textureLod("
        "Texture2D(textures[nonuniformEXT(materialIndex)], linearSampler), "
        "uv, 0.0);"
    ) in generated_code
    assert "textures[materialIndex]" not in generated_code
    assert "OpDecorate" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_image_write_sample_operand_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_IMAGE_WRITE_SAMPLE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "RWTexture2DMS image @set(0) @binding(0) @rgba16f;" in generated_code
    assert (
        "imageStore(image, int2(0, 1), 2, float4(1.0, 0.0, 0.0, 1.0));"
        in generated_code
    )
    assert (
        "imageStore(image, int2(0, 1), float4(1.0, 0.0, 0.0, 1.0));"
        not in generated_code
    )
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_image_read_write_lod_operand_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_IMAGE_READ_WRITE_LOD_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "RWTexture2D<uint> storageImage @set(0) @binding(0);" in generated_code
    assert "uint4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = spirvImageLoadLod(storageImage, uint2(0, 1), 0);" in generated_code
    )
    assert (
        "spirvImageStoreLod(storageImage, uint2(0, 1), 0, uint4(1, 0, 0, 1));"
        in generated_code
    )
    assert "outColor = imageLoad(storageImage, uint2(0, 1));" not in generated_code
    assert (
        "imageStore(storageImage, uint2(0, 1), uint4(1, 0, 0, 1));"
        not in generated_code
    )
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_opstore_body_codegen():
    tokens = tokenize_code(SPIRV_STORE_BODY_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.functions[0].body
    assert "fragColor = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "fragment {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_vector_shuffle_swizzle_body_codegen():
    tokens = tokenize_code(SPIRV_VECTOR_SHUFFLE_BODY_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 value @input @location(0);" in generated_code
    assert "float3 color @output @location(0);" in generated_code
    assert "color = value.xyz;" in generated_code
    assert "color = rgb;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_cross_copy_memory_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_CROSS_COPY_MEMORY_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 v0 @input @location(0);" in generated_code
    assert "float4 v1 @input @location(1);" in generated_code
    assert "float4 gl_Position @output @gl_Position;" in generated_code
    assert "float4 o1 @output @location(1);" in generated_code
    assert "gl_Position = v0;" in generated_code
    assert "o1 = v1;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_copy_memory_sized_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_COPY_MEMORY_SIZED_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 inputValue @input @location(0);" in generated_code
    assert "float4 outValue @output @location(0);" in generated_code
    assert "float4 source;" in generated_code
    assert "float4 target;" in generated_code
    assert "source = inputValue;" in generated_code
    assert "spirvCopyMemorySized(target, source, 16);" in generated_code
    assert "outValue = target;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_cross_image_query_size_lod_codegen():
    tokens = tokenize_code(SPIRV_CROSS_IMAGE_QUERY_SIZE_LOD_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int2 Size @output @location(0);" in generated_code
    assert "Texture2D uTexture @set(0) @binding(0);" in generated_code
    assert (
        "Size = (textureSize(uTexture, 0) + textureSize(uTexture, 1));"
        in generated_code
    )
    assert "Size = (size0 + size1);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_image_query_size_codegen():
    tokens = tokenize_code(SPIRV_TOOLS_IMAGE_QUERY_SIZE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int2 Size @output @location(0);" in generated_code
    assert "Texture2DMS uImage @set(0) @binding(0);" in generated_code
    assert "Size = textureSize(uImage);" in generated_code
    assert "Size = size;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_image_query_format_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_IMAGE_QUERY_FORMAT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "uint formatOut @output @location(0);" in generated_code
    assert "uint orderOut @output @location(1);" in generated_code
    assert "Texture2D queryImage @set(0) @binding(0);" in generated_code
    assert "formatOut = spirvImageQueryFormat(queryImage);" in generated_code
    assert "orderOut = spirvImageQueryOrder(queryImage);" in generated_code
    assert "formatOut = format;" not in generated_code
    assert "orderOut = order;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_image_query_lod_levels_samples_codegen_reparse():
    tokens = tokenize_code(SPIRV_IMAGE_QUERY_LOD_LEVELS_SAMPLES_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float2 uv @input @location(0);" in generated_code
    assert "float2 lodOut @output @location(0);" in generated_code
    assert "int levelsOut @output @location(1);" in generated_code
    assert "int samplesOut @output @location(2);" in generated_code
    assert "Texture2D queryTex @set(0) @binding(0);" in generated_code
    assert "Texture2DMS msTex @set(0) @binding(1);" in generated_code
    assert "lodOut = textureQueryLod(queryTex, uv);" in generated_code
    assert "levelsOut = textureQueryLevels(queryTex);" in generated_code
    assert "samplesOut = textureSamples(msTex);" in generated_code
    assert "lodOut = lod;" not in generated_code
    assert "levelsOut = levels;" not in generated_code
    assert "samplesOut = samples;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_image_read_sample_operand_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_IMAGE_READ_SAMPLE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "RWTexture2DMS var_image @set(0) @binding(1);" in generated_code
    assert "float4 color @output @location(0);" in generated_code
    assert "color = imageLoad(var_image, uint2(1, 2), 2);" in generated_code
    assert "imageLoad(var_image, uint2(1, 2));" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_cross_sparse_texture_resident_codegen_reparse():
    tokens = tokenize_code(SPIRV_CROSS_SPARSE_TEXTURE_RESIDENT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D uSamp @set(0) @binding(0);" in generated_code
    assert "float2 vUV @input @location(0);" in generated_code
    assert "float4 FragColor @output @location(0);" in generated_code
    assert (
        "texel = spirvSparseTexel(spirvImageSparseSample(uSamp, vUV));"
        in generated_code
    )
    assert (
        "ret = spirvSparseTexelsResident("
        "spirvSparseResidencyCode(spirvImageSparseSample(uSamp, vUV)));"
        in generated_code
    )
    assert (
        "FragColor = spirvSparseTexel(spirvImageSparseSample(uSamp, vUV));"
        in generated_code
    )
    assert "texel = sparse[1];" not in generated_code
    assert "ret = resident;" not in generated_code
    assert "OpImageSparse" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_sparse_dref_sample_keeps_dref_helper_name_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_SPARSE_DREF_SAMPLE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "sampler2DShadow shadowTex @set(0) @binding(0);" in generated_code
    assert (
        "depthOut = spirvSparseTexel("
        "spirvImageSparseSampleDrefLod(shadowTex, coord, coord[2], 0.0));"
        in generated_code
    )
    assert "spirvImageSparseSampleDrefCompare" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_sparse_image_read_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_IMAGE_SPARSE_READ_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "RWTexture2D<uint> storageImage @set(0) @binding(0);" in generated_code
    assert "uint4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = spirvSparseTexel("
        "spirvImageSparseRead(storageImage, uint2(0, 1)));" in generated_code
    )
    assert (
        "resident = spirvSparseTexelsResident("
        "spirvSparseResidencyCode("
        "spirvImageSparseRead(storageImage, uint2(0, 1))));" in generated_code
    )
    assert "outColor = sparse[1];" not in generated_code
    assert "resident = is_resident;" not in generated_code
    assert "OpImageSparseRead" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_sparse_image_fetch_gather_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_IMAGE_SPARSE_FETCH_GATHER_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D imageVar @set(0) @binding(0);" in generated_code
    assert "Texture2D sampledVar @set(0) @binding(1);" in generated_code
    assert "float4 fetchOut @output @location(0);" in generated_code
    assert "float4 gatherOut @output @location(1);" in generated_code
    assert "bool residentOut @output @location(2);" in generated_code
    assert (
        "fetchOut = spirvSparseTexel("
        "spirvImageSparseFetch(imageVar, uint2(0, 1), 0));" in generated_code
    )
    assert (
        "residentOut = spirvSparseTexelsResident("
        "spirvSparseResidencyCode("
        "spirvImageSparseFetch(imageVar, uint2(0, 1), 0)));" in generated_code
    )
    assert (
        "gatherOut = spirvSparseTexel("
        "spirvImageSparseGather(sampledVar, uint2(0, 1), 0));" in generated_code
    )
    assert "fetchOut = fetch_sparse[1];" not in generated_code
    assert "gatherOut = gather_sparse[1];" not in generated_code
    assert "residentOut = resident;" not in generated_code
    assert "OpImageSparseFetch" not in generated_code
    assert "OpImageSparseGather" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_dref_sample_preserves_compare_operand_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_DREF_SAMPLE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "sampler2DShadow shadowTex @set(0) @binding(0);" in generated_code
    assert "float depthOut @output @location(0);" in generated_code
    assert (
        "depthOut = textureCompareLod(shadowTex, coord, coord[2], 0.0);"
        in generated_code
    )
    assert (
        "depthOut = textureLod(shadowTex, coord, coord[2], 0.0);" not in generated_code
    )
    assert "Unhandled statement type" not in generated_code


def test_glslang_dref_lod_offset_sample_preserves_const_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_DREF_SAMPLE_LOD_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "sampler2DShadow shadowTex @set(0) @binding(0);" in generated_code
    assert "float depthOut @output @location(0);" in generated_code
    assert (
        "depthOut = textureCompareLodOffset("
        "shadowTex, coord, coord[2], 0.0, int2(1, 2));" in generated_code
    )
    assert "depthOut = textureLodOffset(" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_implicit_lod_bias_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_IMPLICIT_LOD_BIAS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D colorTex @set(0) @binding(0);" in generated_code
    assert "float2 uv @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "outColor = texture(colorTex, uv, 0.25);" in generated_code
    assert "outColor = texture(colorTex, uv);" not in generated_code
    assert "outColor = sample;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_implicit_lod_min_lod_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_IMPLICIT_LOD_MIN_LOD_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D colorTex @set(0) @binding(0);" in generated_code
    assert "float2 uv @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = spirvTextureOffsetMinLod(colorTex, uv, int2(1, 2), 0.5);"
        in generated_code
    )
    assert "outColor = textureOffset(colorTex, uv, 0.5);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_texture_lod_offset_preserves_const_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_TEXTURE_LOD_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D colorTex @set(0) @binding(0);" in generated_code
    assert "float2 uv @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = textureLodOffset(colorTex, uv, 1.0, int2(1, 2));" in generated_code
    )
    assert "outColor = textureLod(colorTex, uv, 1.0);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_texture_proj_offset_preserves_const_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_TEXTURE_PROJ_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D colorTex @set(0) @binding(0);" in generated_code
    assert "float3 uvq @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "outColor = textureProjOffset(colorTex, uvq, int2(1, 2));" in generated_code
    assert (
        "outColor = textureProjLodOffset(colorTex, uvq, 1.0, int2(1, 2));"
        in generated_code
    )
    assert (
        "outColor = textureProjGradOffset(colorTex, uvq, float2(0.5, 0.0), "
        "float2(0.0, 0.5), int2(1, 2));" in generated_code
    )
    assert "outColor = textureProj(colorTex, uvq);" not in generated_code
    assert "outColor = textureProjLod(colorTex, uvq, 1.0);" not in generated_code
    assert (
        "outColor = textureProjGrad(colorTex, uvq, float2(0.5, 0.0), "
        "float2(0.0, 0.5));" not in generated_code
    )
    assert "Unhandled statement type" not in generated_code


def test_glslang_texel_fetch_offset_preserves_const_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_TEXEL_FETCH_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D colorTex @set(0) @binding(0);" in generated_code
    assert "int2 pixel @input @location(0) @flat;" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = texelFetchOffset(colorTex, pixel, 2, int2(1, 2));" in generated_code
    )
    assert "outColor = texelFetch(colorTex, pixel, 2);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_texel_fetch_ms_preserves_sample_operand_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_TEXEL_FETCH_MS_SAMPLE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2DMS colorTex @set(0) @binding(0);" in generated_code
    assert "int2 pixel @input @location(0) @flat;" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert "outColor = texelFetch(colorTex, pixel, 2);" in generated_code
    assert "outColor = texelFetch(colorTex, pixel);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_texture_gather_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_TEXTURE_GATHER_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "Texture2D colorTex @set(0) @binding(0);" in generated_code
    assert "float2 uv @input @location(0);" in generated_code
    assert "float4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = textureGatherOffset(colorTex, uv, int2(1, 2), 2);" in generated_code
    )
    assert "outColor = textureGather(colorTex, uv, 2);" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_dref_gather_offset_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_DREF_GATHER_OFFSET_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "sampler2DShadow shadowTex @set(0) @binding(0);" in generated_code
    assert "float2 uv @input @location(0);" in generated_code
    assert "float depth @input @location(1);" in generated_code
    assert "float4 depthSamples @output @location(0);" in generated_code
    assert (
        "depthSamples = textureGatherCompareOffset(shadowTex, uv, depth, int2(1, 2));"
        in generated_code
    )
    assert (
        "depthSamples = textureGatherCompare(shadowTex, uv, depth);"
        not in generated_code
    )
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_bitcast_success_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_BITCAST_SUCCESS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float floatOut @output @location(0);" in generated_code
    assert "float2 vecOut @output @location(1);" in generated_code
    assert "floatOut = uintBitsToFloat(1);" in generated_code
    assert "vecOut = uintBitsToFloat(uint2(1, 2));" in generated_code
    assert "float_value" not in generated_code
    assert "vec_value" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_sat_convert_stou_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_SAT_CONVERT_STOU_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int signedIn @input @location(0);" in generated_code
    assert "uint unsignedOut @output @location(0);" in generated_code
    assert "unsignedOut = spirvSatConvertSToU(signedIn);" in generated_code
    assert "unsignedOut = saturated;" not in generated_code
    assert "OpSatConvertSToU" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_core_bit_instructions_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_BIT_INSTRUCTIONS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "uint mask @input @location(0);" in generated_code
    assert "int signedMask @input @location(1);" in generated_code
    assert "countOut = bitCount(mask);" in generated_code
    assert "reverseOut = bitfieldReverse(mask);" in generated_code
    assert "uExtractOut = bitfieldExtract(mask, offset, bits);" in generated_code
    assert "sExtractOut = bitfieldExtract(signedMask, offset, bits);" in generated_code
    assert (
        "insertOut = bitfieldInsert(mask, bitfieldExtract(mask, offset, bits), offset, bits);"
        in generated_code
    )
    assert "counted" not in generated_code
    assert "uextracted" not in generated_code
    assert "spirv_Bit" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_copy_object_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_COPY_OBJECT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float inputValue @input @location(0);" in generated_code
    assert "float color @output @location(0);" in generated_code
    assert "color = inputValue;" in generated_code
    assert "color = copy;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_copy_logical_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_COPY_LOGICAL_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "bool flagIn @input @location(0);" in generated_code
    assert "bool flagOut @output @location(0);" in generated_code
    assert "flagOut = flagIn;" in generated_code
    assert "flagOut = copy;" not in generated_code
    assert "OpCopyLogical" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_quantize_to_f16_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_QUANTIZE_TO_F16_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float inputValue @input @location(0);" in generated_code
    assert "float color @output @location(0);" in generated_code
    assert "color = spirvQuantizeToF16(inputValue);" in generated_code
    assert "color = quantized;" not in generated_code
    assert "OpQuantizeToF16" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_cross_composite_insert_vector_codegen():
    tokens = tokenize_code(SPIRV_CROSS_COMPOSITE_INSERT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float value0 @input @location(0);" in generated_code
    assert "float value1 @input @location(1);" in generated_code
    assert "float2 FragColor @output @location(0);" in generated_code
    assert "FragColor = float2(value0, value1);" in generated_code
    assert "FragColor = b;" not in generated_code
    assert "spirvCompositeInsert" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_glsl_std450_extinst_body_codegen():
    tokens = tokenize_code(SPIRV_GLSL_STD450_EXTINST_BODY_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 inputVec @input @location(0);" in generated_code
    assert "float4 outputVec @output @location(0);" in generated_code
    assert "outputVec = normalize(inputVec);" in generated_code
    assert "outputVec = normalized;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_std450_sqrt_extinst_codegen():
    tokens = tokenize_code(SPIRV_TOOLS_STD450_SQRT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float inputValue @input @location(0);" in generated_code
    assert "float outputValue @output @location(0);" in generated_code
    assert "outputValue = sqrt(inputValue);" in generated_code
    assert "spirv_GLSL_std_450_Sqrt" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_numeric_glsl_std450_sqrt_extinst_codegen():
    tokens = tokenize_code(SPIRV_NUMERIC_GLSL_STD450_SQRT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float inputValue @input @location(0);" in generated_code
    assert "float outputValue @output @location(0);" in generated_code
    assert "outputValue = sqrt(inputValue);" in generated_code
    assert "spirv_GLSL_std_450_instruction_31" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glsl_std450_struct_extinst_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSL_STD450_STRUCT_EXTINST_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "struct ModfResult" in generated_code
    assert "float fractional;" in generated_code
    assert "float whole;" in generated_code
    assert "struct FrexpResult" in generated_code
    assert "float significand;" in generated_code
    assert "int exponent;" in generated_code
    assert (
        "outputValue = (modf(inputValue).fractional + frexp(inputValue).significand);"
        in generated_code
    )
    assert "spirv_GLSL_std_450_ModfStruct" not in generated_code
    assert "spirv_GLSL_std_450_instruction_52" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_nonsemantic_debug_printf_opstring_extinst_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_NONSEMANTIC_DEBUG_PRINTF_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert 'debugPrintfEXT("unsigned == %u", foo);' in generated_code
    assert "debugPrintfEXT(value_fmt" not in generated_code
    assert "spirv_NonSemantic_DebugPrintf_1" not in generated_code
    assert "OpString" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_debug_function_names_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_DEBUG_FUNCTION_NAME_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float f1()" in generated_code
    assert "f1(()" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_pointer_parameter_block_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_POINTER_PARAMETER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "cbuffer Block @set(0) @binding(0)" in generated_code
    assert "struct Block" not in generated_code
    assert "float4 helper(float4 param)" in generated_code
    assert "%ptr_function_v4float" not in generated_code
    assert "helper(()" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_std450_faceforward_extinst_codegen():
    tokens = tokenize_code(GLSLANG_STD450_FACEFORWARD_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float3 normal @input @location(0);" in generated_code
    assert "float3 incident @input @location(1);" in generated_code
    assert "float3 reference @input @location(2);" in generated_code
    assert "float3 color @output @location(0);" in generated_code
    assert "color = faceforward(normal, incident, reference);" in generated_code
    assert "spirv_GLSL_std_450_FaceForward" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_std450_matrix_extinst_codegen():
    tokens = tokenize_code(GLSLANG_STD450_MATRIX_OPS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float2x2 inputMatrix @input @location(0);" in generated_code
    assert "float determinantOut @output @location(0);" in generated_code
    assert "float2x2 inverseOut @output @location(1);" in generated_code
    assert "determinantOut = determinant(inputMatrix);" in generated_code
    assert "inverseOut = inverse(inputMatrix);" in generated_code
    assert "spirv_GLSL_std_450_Determinant" not in generated_code
    assert "spirv_GLSL_std_450_MatrixInverse" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_std450_interpolation_extinst_codegen():
    tokens = tokenize_code(GLSLANG_STD450_INTERPOLATION_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float inputValue @input @location(0);" in generated_code
    assert "centroidOut = interpolateAtCentroid(inputValue);" in generated_code
    assert "sampleOut = interpolateAtSample(inputValue, 1);" in generated_code
    assert (
        "offsetOut = interpolateAtOffset(inputValue, float2(0.0, 0.0));"
        in generated_code
    )
    assert "spirv_GLSL_std_450_InterpolateAtCentroid" not in generated_code
    assert "spirv_GLSL_std_450_InterpolateAtSample" not in generated_code
    assert "spirv_GLSL_std_450_InterpolateAtOffset" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_derivative_ops_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_DERIVATIVE_OPS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float inputValue @input @location(0);" in generated_code
    assert "dxOut = dFdx(inputValue);" in generated_code
    assert "dyOut = dFdyFine(inputValue);" in generated_code
    assert "widthOut = fwidthCoarse(inputValue);" in generated_code
    assert "dxOut = dx;" not in generated_code
    assert "dyOut = dy_fine;" not in generated_code
    assert "widthOut = width_coarse;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_geometry_execution_modes_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_GEOMETRY_EXECUTION_MODES_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "geometry {" in generated_code
    assert "layout(triangles, invocations = 2) in;" in generated_code
    assert "layout(triangle_strip, max_vertices = 3) out;" in generated_code
    assert "float3 inColor[3] @input @location(0);" in generated_code
    assert "float3 outColor @output @location(0);" in generated_code
    assert "fragment {" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_cross_geometry_emit_primitive_codegen_reparse():
    tokens = tokenize_code(SPIRV_CROSS_GEOMETRY_EMIT_PRIMITIVE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "geometry {" in generated_code
    assert "layout(triangles) in;" in generated_code
    assert "layout(triangle_strip, max_vertices = 1) out;" in generated_code
    assert "EmitVertex();" in generated_code
    assert "EndPrimitive();" in generated_code
    assert "OpEmitVertex" not in generated_code
    assert "OpEndPrimitive" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tessellation_evaluation_execution_modes_codegen_reparse():
    tokens = tokenize_code(SPIRV_TESSELLATION_EVALUATION_EXECUTION_MODES_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "tessellation_evaluation {" in generated_code
    assert "layout(triangles, equal_spacing, ccw) in;" in generated_code
    assert "fragment {" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_cross_isnan_isinf_codegen_reparse():
    tokens = tokenize_code(SPIRV_CROSS_ISNAN_ISINF_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "nanOut = isnan(1.0);" in generated_code
    assert "infOut = isinf(1.0);" in generated_code
    assert "nanOut = is_nan;" not in generated_code
    assert "infOut = is_inf;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_local_size_id_codegen():
    tokens = tokenize_code(SPIRV_LOCAL_SIZE_ID_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = WORKGROUP_WIDTH, local_size_y = WORKGROUP_HEIGHT, "
        "local_size_z = 1) in;" in generated_code
    )
    assert "const uint WORKGROUP_WIDTH @constant_id(0) = 8;" in generated_code
    assert "const uint WORKGROUP_HEIGHT @constant_id(1) = 4;" in generated_code
    assert "fragment {" not in generated_code


def test_spirv_assembly_flat_location_interface_codegen():
    tokens = tokenize_code(SPIRV_TOOLS_FLAT_LOCATION_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "uint input_flat_u32 @input @location(0) @flat;" in generated_code
    assert "%input_flat_u32" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_spec_noperspective_only_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_SPEC_NOPERSPECTIVE_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 color2 @input @noperspective;" in generated_code
    assert "NoPerspective" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_relaxed_precision_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_RELAXED_PRECISION_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 color @input @location(0) @mediump;" in generated_code
    assert "float4 outColor @output @location(0) @mediump;" in generated_code
    assert "outColor = color;" in generated_code
    assert "RelaxedPrecision" not in generated_code
    assert "outColor = loaded;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_vertex_id_instance_id_builtin_aliases_codegen_reparse():
    tokens = tokenize_code(SPIRV_VERTEX_ID_INSTANCE_ID_BUILTINS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "int gl_VertexID @input @gl_VertexID;" in generated_code
    assert "int gl_InstanceID @input @gl_InstanceID;" in generated_code
    assert "float4 gl_Position @output @gl_Position;" in generated_code
    assert "@builtin(vertexid)" not in generated_code
    assert "@builtin(instanceid)" not in generated_code
    assert "int VertexId @input" not in generated_code
    assert "int InstanceId @input" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_fragment_sample_and_view_builtins_codegen():
    tokens = tokenize_code(SPIRV_FRAGMENT_SAMPLE_AND_VIEW_BUILTINS_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int gl_SampleID @input @gl_SampleID;" in generated_code
    assert "float2 gl_SamplePosition @input @gl_SamplePosition;" in generated_code
    assert "int gl_SampleMaskIn[1] @input @gl_SampleMaskIn;" in generated_code
    assert "int gl_Layer @input @gl_Layer;" in generated_code
    assert "int gl_ViewportIndex @input @gl_ViewportIndex;" in generated_code
    assert "@builtin(sampleid)" not in generated_code
    assert "@builtin(samplemask)" not in generated_code
    assert "@builtin(layer)" not in generated_code


def test_spirv_assembly_sample_mask_body_uses_storage_aware_names():
    tokens = tokenize_code(SPIRV_FRAGMENT_SAMPLE_MASK_BODY_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int gl_SampleMaskIn[1] @input @gl_SampleMaskIn;" in generated_code
    assert "int gl_SampleMask[1] @output @gl_SampleMask;" in generated_code
    assert "gl_SampleMask[0] = gl_SampleMaskIn[0];" in generated_code
    assert "SampleMask[0] = SampleMask[0];" not in generated_code


def test_spirv_assembly_fragment_helper_invocation_builtin_codegen_reparse():
    tokens = tokenize_code(SPIRV_FRAGMENT_HELPER_INVOCATION_BUILTIN_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "bool gl_HelperInvocation @input @gl_HelperInvocation;" in generated_code
    assert "bool outFlag @output @location(0);" in generated_code
    assert "outFlag = gl_HelperInvocation;" in generated_code
    assert "@builtin(helperinvocation)" not in generated_code
    assert "bool HelperInvocation @input" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_fragment_barycentric_interface_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_FRAGMENT_BARYCENTRIC_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float outValue @output @location(0);" in generated_code
    assert (
        "float3 gl_BaryCoordNoPerspEXT @input @gl_BaryCoordNoPerspEXT;"
        in generated_code
    )
    assert "float3 vertexIDs[3] @input @location(1) @pervertexEXT;" in generated_code
    assert "outValue = gl_BaryCoordNoPerspEXT[0];" in generated_code
    assert "@builtin(barycoordnoperspkhr)" not in generated_code
    assert "PerVertexKHR" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_gl_pervertex_access_chain_codegen():
    tokens = tokenize_code(SPIRV_TOOLS_GLPERVERTEX_ACCESS_CHAIN_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 _ua_position @input @location(0);" in generated_code
    assert "float4 gl_Position @output @gl_Position;" in generated_code
    assert "float gl_ClipDistance[8] @output @gl_ClipDistance;" in generated_code
    assert "gl_Position = _ua_position;" in generated_code
    assert "value_19[0]" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_assembly_multiline_opsource_literal_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_MULTILINE_OPSOURCE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 pos @output @location(0);" in generated_code
    assert "vertex {" in generated_code
    assert "void main()" in generated_code
    assert "OpSource" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_ptr_access_chain_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_PTR_ACCESS_CHAIN_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "groupshared uint workgroupValues[4][4];" in generated_code
    assert "uint outA @output @location(0);" in generated_code
    assert "uint outB @output @location(1);" in generated_code
    assert "outA = workgroupValues[1][2];" in generated_code
    assert "outB = workgroupValues[0][1];" in generated_code
    assert "outA = ptrAccess;" not in generated_code
    assert "outB = inBoundsPtr;" not in generated_code
    assert "OpPtrAccessChain" not in generated_code
    assert "OpInBoundsPtrAccessChain" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_glslang_workgroup_shared_variable_codegen_reparse():
    tokens = tokenize_code(SPIRV_GLSLANG_WORKGROUP_SHARED_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "groupshared uint sharedData[4];" in generated_code
    assert "sharedData[0] = 7;" in generated_code
    assert "compute {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_extended_arithmetic_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_EXTENDED_ARITHMETIC_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "uint outSum @output @location(0);" in generated_code
    assert "uint outCarry @output @location(1);" in generated_code
    assert "int outLow @output @location(2);" in generated_code
    assert "int outHigh @output @location(3);" in generated_code
    assert "outSum = spirvIAddCarry(0, 1)[0];" in generated_code
    assert "outCarry = spirvIAddCarry(0, 1)[1];" in generated_code
    assert "outLow = spirvSMulExtended(2, 3)[0];" in generated_code
    assert "outHigh = spirvSMulExtended(2, 3)[1];" in generated_code
    assert "OpIAddCarry" not in generated_code
    assert "OpSMulExtended" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_same_value_phi_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_SAME_VALUE_PHI_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outColor @output @location(0);" in generated_code
    assert "outColor = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "outColor = phi;" not in generated_code
    assert "spirvPhi" not in generated_code
    assert "OpPhi" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_spirv_tools_direct_selection_phi_codegen_reparse():
    tokens = tokenize_code(SPIRV_TOOLS_DIRECT_SELECTION_PHI_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert "float4 outColor @output @location(0);" in generated_code
    assert (
        "outColor = (true ? float4(1.0, 0.0, 0.0, 1.0) : "
        "float4(0.0, 0.0, 1.0, 1.0));" in generated_code
    )
    assert "outColor = phi;" not in generated_code
    assert "spirvPhi" not in generated_code
    assert "OpPhi" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_translate_api_accepts_spirv_assembly_uniform_constant_resources(tmp_path):
    import crosstl

    shader_path = tmp_path / "resources.spvasm"
    shader_path.write_text(SPIRV_UNIFORM_CONSTANT_RESOURCE_ASSEMBLY, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="cgl", format_output=False
    )

    assert "Texture2D combinedTex @set(0) @binding(0);" in generated_code
    assert "sampler linearSampler @set(0) @binding(1);" in generated_code


def test_spirv_assembly_subpass_input_attachment_index_codegen_reparse():
    tokens = tokenize_code(SPIRV_VULKAN_SUBPASS_INPUT_ATTACHMENT_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    parse_crossgl(generated_code)
    assert (
        "subpassInput sceneInput @set(1) @binding(2) @input_attachment_index(3);"
        in generated_code
    )
    assert "Texture2D sceneInput" not in generated_code
    assert "InputAttachmentIndex" not in generated_code


def test_translate_api_accepts_spirv_assembly_specialization_constants(tmp_path):
    import crosstl

    shader_path = tmp_path / "constants.spvasm"
    shader_path.write_text(SPIRV_SPEC_CONSTANT_ASSEMBLY, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="cgl", format_output=False
    )

    assert "const uint MAX_LIGHTS @constant_id(0) = 4;" in generated_code
    assert "const bool ENABLE_SHADOWS @constant_id(1) = true;" in generated_code


def test_translate_api_accepts_location_decorated_spirv_assembly(tmp_path):
    import crosstl

    shader_path = tmp_path / "fragment.spvasm"
    shader_path.write_text(SPIRV_TOOLS_BASIC_INTERFACE_ASSEMBLY, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="cgl", format_output=False
    )

    assert "float4 _ua_position @input @location(0);" in generated_code
    assert "float4 ANGLEXfbPosition @output @location(0);" in generated_code
    assert "float4 gl_Position @output @gl_Position;" in generated_code


def test_translate_api_rejects_unsupported_spirv_assembly_with_clear_error(tmp_path):
    import crosstl

    shader_path = tmp_path / "fragment.spvasm"
    shader_path.write_text(
        """
        OpCapability Shader
        OpMemoryModel Logical GLSL450
        %void = OpTypeVoid
        """,
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError, match="only partially supported"):
        crosstl.translate(str(shader_path), backend="cgl", format_output=False)


def test_vulkan_layout_blocks_emit_crossgl_resources():
    tokens = tokenize_code(LAYOUT_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(ast.global_variables) == 3
    assert ast.global_variables[0].block_name == "Camera"
    assert ast.global_variables[1].layout_type == "BUFFER"
    assert ast.global_variables[2].data_type == "vec3"
    assert "cbuffer Camera" in generated_code
    assert "float4x4 viewProj;" in generated_code
    assert "struct Particles" in generated_code
    assert (
        "RWStructuredBuffer<Particles> particles @set(0) @binding(1);" in generated_code
    )
    assert "float3 position @input @location(0);" in generated_code
    assert "vertex {" in generated_code
    assert "gl_Position = float4(position, 1.0);" in generated_code


def test_vulkan_layout_interpolation_qualifier_codegen():
    code = """
    layout(location = 0) flat in int faceID;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].declaration_qualifiers == ["flat"]
    assert "int faceID @input @location(0) @flat;" in generated_code


def test_vulkan_layout_component_and_index_codegen():
    code = """
    layout(location = 1, component = 2) noperspective in vec4 color;
    layout(location = 0, index = 1) out vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 color @input @location(1) @component(2) @noperspective;"
        in generated_code
    )
    assert "float4 fragColor @output @location(0) @index(1);" in generated_code


def test_vulkan_readonly_buffer_layout_codegen():
    code = """
    layout(set = 0, binding = 0) readonly buffer Particles {
        vec4 pos[];
    } particles;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].declaration_qualifiers == ["readonly"]
    assert "struct Particles" in generated_code
    assert "float4 pos[];" in generated_code
    assert (
        "StructuredBuffer<Particles> particles @set(0) @binding(0);" in generated_code
    )
    assert "RWStructuredBuffer<Particles> particles;" not in generated_code


def test_vulkan_compute_local_size_layout_codegen():
    code = """
    layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;"
        in generated_code
    )
    assert "void main()" in generated_code
    assert "fragment {" not in generated_code


def test_spirv_assembly_compute_entry_point_uses_execution_metadata():
    code = """
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint GLCompute %main "main"
    OpExecutionMode %main LocalSize 8 4 1
    %void = OpTypeVoid
    %fn = OpTypeFunction %void
    %main = OpFunction %void None %fn
    %label = OpLabel
    OpReturn
    OpFunctionEnd
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;"
        in generated_code
    )
    assert "void main()" in generated_code
    assert "fragment {" not in generated_code


def test_vulkan_fragment_early_tests_layout_codegen():
    code = """
    layout(early_fragment_tests) in;
    void main() {
        gl_FragColor = vec4(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fragment {" in generated_code
    assert "layout(early_fragment_tests) in;" in generated_code
    assert generated_code.index("layout(early_fragment_tests) in;") < (
        generated_code.index("void main()")
    )


def test_vulkan_specialization_constant_layout_codegen():
    code = """
    layout(constant_id = 0) const uint a = 1;
    layout(constant_id = 1) const float scale = 3.0;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const uint a @constant_id(0) = 1;" in generated_code
    assert "const float scale @constant_id(1) = 3.0;" in generated_code
    assert "layout(constant_id" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_vulkan_uniform_block_instance_member_access_flattens_to_cbuffer_member():
    code = """
    layout(set = 0, binding = 0) uniform Camera {
        mat4 viewProj;
    } camera;
    layout(location = 0) in vec3 position;
    void main() {
        gl_Position = camera.viewProj * vec4(position, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "cbuffer Camera" in generated_code
    assert "float4x4 viewProj;" in generated_code
    assert "gl_Position = (viewProj * float4(position, 1.0));" in generated_code
    assert "camera.viewProj" not in generated_code


def test_vulkan_push_constant_block_emits_marked_cbuffer():
    code = """
    layout(push_constant) uniform PushConstants {
        mat4 model;
        vec4 tint;
    } pc;
    void main() {
        gl_Position = pc.model * vec4(1.0, 0.0, 0.0, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].push_constant is True
    assert "cbuffer PushConstants @push_constant {" in generated_code
    assert "cbuffer PushConstants {\n" not in generated_code
    assert "float4x4 model;" in generated_code
    assert "float4 tint;" in generated_code
    assert "gl_Position = (model * float4(1.0, 0.0, 0.0, 1.0));" in generated_code
    assert "pc.model" not in generated_code


def test_vulkan_layout_blocks_preserve_array_members():
    tokens = tokenize_code(ARRAY_LAYOUT_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].struct_fields == [
        ("mat4", "transforms[64]"),
        ("vec4", "weights[4]"),
    ]
    assert ast.global_variables[1].struct_fields == [
        ("vec4", "positions[128]"),
        ("float", "masses[128]"),
    ]
    assert "float4x4 transforms[64];" in generated_code
    assert "float4 weights[4];" in generated_code
    assert "float4 positions[128];" in generated_code
    assert "float masses[128];" in generated_code
    assert "transforms[0] * float4(1.0, 0.0, 0.0, 1.0)" in generated_code


def test_vulkan_layout_blocks_preserve_custom_struct_members():
    code = """
    struct Light {
        vec3 position;
    };
    layout(set = 0, binding = 0) uniform Scene {
        Light lights[4];
        mat4 view;
    } scene;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].struct_fields == [
        ("Light", "lights[4]"),
        ("mat4", "view"),
    ]
    assert "struct Light" in generated_code
    assert "float3 position;" in generated_code
    assert "cbuffer Scene" in generated_code
    assert "Light lights[4];" in generated_code
    assert "float4x4 view;" in generated_code


def test_vulkan_custom_struct_uniform_type_codegen():
    code = """
    struct Light {
        vec3 position;
    };
    layout(set = 0, binding = 1) uniform Light activeLight;
    uniform Light fallbackLight;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "Light"
    assert ast.global_variables[0].variable_name == "activeLight"
    assert ast.global_variables[1].vtype == "Light"
    assert ast.global_variables[1].name == "fallbackLight"
    assert "Light activeLight;" in generated_code
    assert "Light fallbackLight;" in generated_code


def test_standalone_struct_array_members_codegen():
    code = """
    struct LightBlock {
        vec3 positions[4];
        float weights[4];
    };
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct LightBlock" in generated_code
    assert "float3 positions[4];" in generated_code
    assert "float weights[4];" in generated_code


def test_vulkan_resource_uniforms_emit_crossgl_resources():
    tokens = tokenize_code(RESOURCE_UNIFORM_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "sampler2D"
    assert ast.global_variables[0].variable_name == "albedoTex"
    assert ast.global_variables[1].vtype == "samplerCube"
    assert ast.global_variables[1].name == "skybox"
    assert "Texture2D albedoTex;" in generated_code
    assert "TextureCube skybox;" in generated_code


def test_vulkan_one_dimensional_sampler_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform sampler1D ramp;
    uniform sampler1DArray ramps;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "sampler1D"
    assert ast.global_variables[0].variable_name == "ramp"
    assert ast.global_variables[1].vtype == "sampler1DArray"
    assert ast.global_variables[1].name == "ramps"
    assert "Texture1D ramp;" in generated_code
    assert "Texture1DArray ramps;" in generated_code


def test_vulkan_integer_one_dimensional_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isampler1D signedRamp;
    layout(set = 0, binding = 1) uniform usampler1D unsignedRamp;
    layout(set = 0, binding = 2) uniform isampler1DArray signedRamps;
    layout(set = 0, binding = 3) uniform usampler1DArray unsignedRamps;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert [variable.data_type for variable in ast.global_variables] == [
        "isampler1D",
        "usampler1D",
        "isampler1DArray",
        "usampler1DArray",
    ]
    assert "isampler1D signedRamp;" in generated_code
    assert "usampler1D unsignedRamp;" in generated_code
    assert "isampler1DArray signedRamps;" in generated_code
    assert "usampler1DArray unsignedRamps;" in generated_code
    assert "Texture1D<int> signedRamp;" not in generated_code
    assert "Texture1D<uint> unsignedRamp;" not in generated_code
    assert "Texture1DArray<int> signedRamps;" not in generated_code
    assert "Texture1DArray<uint> unsignedRamps;" not in generated_code


def test_vulkan_integer_2d_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isampler2D signedTex;
    layout(set = 0, binding = 1) uniform usampler2D unsignedTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "isampler2D"
    assert ast.global_variables[0].variable_name == "signedTex"
    assert ast.global_variables[1].data_type == "usampler2D"
    assert ast.global_variables[1].variable_name == "unsignedTex"
    assert "isampler2D signedTex;" in generated_code
    assert "usampler2D unsignedTex;" in generated_code
    assert "Texture2D<int> signedTex;" not in generated_code
    assert "Texture2D<uint> unsignedTex;" not in generated_code


def test_vulkan_integer_2d_array_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isampler2DArray signedLayers;
    layout(set = 0, binding = 1) uniform usampler2DArray unsignedLayers;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "isampler2DArray"
    assert ast.global_variables[0].variable_name == "signedLayers"
    assert ast.global_variables[1].data_type == "usampler2DArray"
    assert ast.global_variables[1].variable_name == "unsignedLayers"
    assert "isampler2DArray signedLayers;" in generated_code
    assert "usampler2DArray unsignedLayers;" in generated_code
    assert "Texture2DArray<int> signedLayers;" not in generated_code
    assert "Texture2DArray<uint> unsignedLayers;" not in generated_code


def test_vulkan_integer_multisample_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isampler2DMS signedSamples;
    layout(set = 0, binding = 1) uniform usampler2DMS unsignedSamples;
    layout(set = 0, binding = 2) uniform isampler2DMSArray signedSampleLayers;
    layout(set = 0, binding = 3) uniform usampler2DMSArray unsignedSampleLayers;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert [variable.data_type for variable in ast.global_variables] == [
        "isampler2DMS",
        "usampler2DMS",
        "isampler2DMSArray",
        "usampler2DMSArray",
    ]
    assert "isampler2DMS signedSamples;" in generated_code
    assert "usampler2DMS unsignedSamples;" in generated_code
    assert "isampler2DMSArray signedSampleLayers;" in generated_code
    assert "usampler2DMSArray unsignedSampleLayers;" in generated_code
    assert "Texture2DMS<int> signedSamples;" not in generated_code
    assert "Texture2DMS<uint> unsignedSamples;" not in generated_code
    assert "Texture2DMSArray<int> signedSampleLayers;" not in generated_code
    assert "Texture2DMSArray<uint> unsignedSampleLayers;" not in generated_code


def test_vulkan_integer_3d_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isampler3D signedVolume;
    layout(set = 0, binding = 1) uniform usampler3D unsignedVolume;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "isampler3D"
    assert ast.global_variables[0].variable_name == "signedVolume"
    assert ast.global_variables[1].data_type == "usampler3D"
    assert ast.global_variables[1].variable_name == "unsignedVolume"
    assert "isampler3D signedVolume;" in generated_code
    assert "usampler3D unsignedVolume;" in generated_code
    assert "Texture3D<int> signedVolume;" not in generated_code
    assert "Texture3D<uint> unsignedVolume;" not in generated_code


def test_vulkan_integer_cube_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isamplerCube signedCube;
    layout(set = 0, binding = 1) uniform usamplerCube unsignedCube;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "isamplerCube"
    assert ast.global_variables[0].variable_name == "signedCube"
    assert ast.global_variables[1].data_type == "usamplerCube"
    assert ast.global_variables[1].variable_name == "unsignedCube"
    assert "isamplerCube signedCube;" in generated_code
    assert "usamplerCube unsignedCube;" in generated_code
    assert "TextureCube<int> signedCube;" not in generated_code
    assert "TextureCube<uint> unsignedCube;" not in generated_code


def test_vulkan_integer_cube_array_sampler_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isamplerCubeArray signedCubeLayers;
    layout(set = 0, binding = 1) uniform usamplerCubeArray unsignedCubeLayers;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "isamplerCubeArray"
    assert ast.global_variables[0].variable_name == "signedCubeLayers"
    assert ast.global_variables[1].data_type == "usamplerCubeArray"
    assert ast.global_variables[1].variable_name == "unsignedCubeLayers"
    assert "isamplerCubeArray signedCubeLayers;" in generated_code
    assert "usamplerCubeArray unsignedCubeLayers;" in generated_code
    assert "TextureCubeArray<int> signedCubeLayers;" not in generated_code
    assert "TextureCubeArray<uint> unsignedCubeLayers;" not in generated_code


def test_vulkan_one_dimensional_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform image1D line;
    layout(set = 0, binding = 1) uniform image1DArray lines;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "image1D"
    assert ast.global_variables[0].variable_name == "line"
    assert ast.global_variables[1].data_type == "image1DArray"
    assert ast.global_variables[1].variable_name == "lines"
    assert "RWTexture1D line;" in generated_code
    assert "RWTexture1DArray lines;" in generated_code
    assert "image1D line;" not in generated_code
    assert "image1DArray lines;" not in generated_code


def test_vulkan_image2d_array_uniform_emits_crossgl_resource():
    code = """
    layout(set = 0, binding = 0) uniform image2DArray layers;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "image2DArray"
    assert ast.global_variables[0].variable_name == "layers"
    assert "RWTexture2DArray layers;" in generated_code
    assert "image2DArray layers;" not in generated_code


def test_vulkan_typed_2d_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimage2D signedImage;
    layout(set = 0, binding = 1) uniform uimage2D counters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "iimage2D"
    assert ast.global_variables[0].variable_name == "signedImage"
    assert ast.global_variables[1].data_type == "uimage2D"
    assert ast.global_variables[1].variable_name == "counters"
    assert "RWTexture2D<int> signedImage;" in generated_code
    assert "RWTexture2D<uint> counters;" in generated_code
    assert "iimage2D signedImage;" not in generated_code
    assert "uimage2D counters;" not in generated_code


def test_vulkan_storage_image_layout_and_access_qualifiers_codegen():
    code = """
    layout(set = 3, binding = 7, r32ui) coherent readonly uniform uimage2D counters;
    layout(set = 4, binding = 8, rgba32f) writeonly uniform image2D outImage;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "RWTexture2D<uint> counters @set(3) @binding(7) @r32ui @coherent @readonly;"
        in generated_code
    )
    assert (
        "RWTexture2D outImage @set(4) @binding(8) @rgba32f @writeonly;"
        in generated_code
    )
    assert "uimage2D counters @binding" not in generated_code
    assert "image2D outImage @binding" not in generated_code


def test_vulkan_storage_image_symbolic_binding_codegen():
    code = """
    layout(set = RESOURCE_SET, binding = OUT_IMAGE_BINDING, rgba32f) writeonly uniform image2D outImage;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "RWTexture2D outImage @set(RESOURCE_SET) @binding(OUT_IMAGE_BINDING) "
        "@rgba32f @writeonly;" in generated_code
    )


def test_vulkan_typed_1d_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimage1D signedLine;
    layout(set = 0, binding = 1) uniform uimage1D counters;
    layout(set = 0, binding = 2) uniform iimage1DArray signedLayers;
    layout(set = 0, binding = 3) uniform uimage1DArray layerCounters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert [variable.data_type for variable in ast.global_variables] == [
        "iimage1D",
        "uimage1D",
        "iimage1DArray",
        "uimage1DArray",
    ]
    assert "RWTexture1D<int> signedLine;" in generated_code
    assert "RWTexture1D<uint> counters;" in generated_code
    assert "RWTexture1DArray<int> signedLayers;" in generated_code
    assert "RWTexture1DArray<uint> layerCounters;" in generated_code
    assert "iimage1D signedLine;" not in generated_code
    assert "uimage1D counters;" not in generated_code
    assert "iimage1DArray signedLayers;" not in generated_code
    assert "uimage1DArray layerCounters;" not in generated_code


def test_vulkan_typed_3d_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimage3D signedVolume;
    layout(set = 0, binding = 1) uniform uimage3D counters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "iimage3D"
    assert ast.global_variables[0].variable_name == "signedVolume"
    assert ast.global_variables[1].data_type == "uimage3D"
    assert ast.global_variables[1].variable_name == "counters"
    assert "RWTexture3D<int> signedVolume;" in generated_code
    assert "RWTexture3D<uint> counters;" in generated_code
    assert "iimage3D signedVolume;" not in generated_code
    assert "uimage3D counters;" not in generated_code


def test_vulkan_typed_cube_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimageCube signedCube;
    layout(set = 0, binding = 1) uniform uimageCube cubeCounters;
    layout(set = 0, binding = 2) uniform iimageCubeArray signedCubeLayers;
    layout(set = 0, binding = 3) uniform uimageCubeArray cubeLayerCounters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert [variable.data_type for variable in ast.global_variables] == [
        "iimageCube",
        "uimageCube",
        "iimageCubeArray",
        "uimageCubeArray",
    ]
    assert "RWTextureCube<int> signedCube;" in generated_code
    assert "RWTextureCube<uint> cubeCounters;" in generated_code
    assert "RWTextureCubeArray<int> signedCubeLayers;" in generated_code
    assert "RWTextureCubeArray<uint> cubeLayerCounters;" in generated_code
    assert "iimageCube signedCube;" not in generated_code
    assert "uimageCube cubeCounters;" not in generated_code
    assert "iimageCubeArray signedCubeLayers;" not in generated_code
    assert "uimageCubeArray cubeLayerCounters;" not in generated_code


def test_vulkan_typed_2d_array_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimage2DArray signedLayers;
    layout(set = 0, binding = 1) uniform uimage2DArray counters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "iimage2DArray"
    assert ast.global_variables[0].variable_name == "signedLayers"
    assert ast.global_variables[1].data_type == "uimage2DArray"
    assert ast.global_variables[1].variable_name == "counters"
    assert "RWTexture2DArray<int> signedLayers;" in generated_code
    assert "RWTexture2DArray<uint> counters;" in generated_code
    assert "iimage2DArray signedLayers;" not in generated_code
    assert "uimage2DArray counters;" not in generated_code


def test_vulkan_typed_2d_ms_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimage2DMS signedSamples;
    layout(set = 0, binding = 1) uniform uimage2DMS sampleCounters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "iimage2DMS"
    assert ast.global_variables[0].variable_name == "signedSamples"
    assert ast.global_variables[1].data_type == "uimage2DMS"
    assert ast.global_variables[1].variable_name == "sampleCounters"
    assert "RWTexture2DMS<int> signedSamples;" in generated_code
    assert "RWTexture2DMS<uint> sampleCounters;" in generated_code
    assert "iimage2DMS signedSamples;" not in generated_code
    assert "uimage2DMS sampleCounters;" not in generated_code


def test_vulkan_typed_2d_ms_array_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimage2DMSArray signedLayers;
    layout(set = 0, binding = 1) uniform uimage2DMSArray sampleCounters;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "iimage2DMSArray"
    assert ast.global_variables[0].variable_name == "signedLayers"
    assert ast.global_variables[1].data_type == "uimage2DMSArray"
    assert ast.global_variables[1].variable_name == "sampleCounters"
    assert "RWTexture2DMSArray<int> signedLayers;" in generated_code
    assert "RWTexture2DMSArray<uint> sampleCounters;" in generated_code
    assert "iimage2DMSArray signedLayers;" not in generated_code
    assert "uimage2DMSArray sampleCounters;" not in generated_code


def test_vulkan_image_cube_array_uniform_emits_crossgl_resource():
    code = """
    layout(set = 0, binding = 0) uniform imageCubeArray cubes;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "imageCubeArray"
    assert ast.global_variables[0].variable_name == "cubes"
    assert "RWTextureCubeArray cubes;" in generated_code
    assert "imageCubeArray cubes;" not in generated_code


def test_vulkan_multisample_sampler_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform sampler2DMS msTex;
    layout(set = 0, binding = 1) uniform sampler2DMSArray msTexArray;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "sampler2DMS"
    assert ast.global_variables[0].variable_name == "msTex"
    assert ast.global_variables[1].data_type == "sampler2DMSArray"
    assert ast.global_variables[1].variable_name == "msTexArray"
    assert "Texture2DMS msTex;" in generated_code
    assert "Texture2DMSArray msTexArray;" in generated_code
    assert "sampler2DMS msTex;" not in generated_code
    assert "sampler2DMSArray msTexArray;" not in generated_code


def test_vulkan_multisample_image_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform image2DMS msImage;
    layout(set = 0, binding = 1) uniform image2DMSArray msImages;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "image2DMS"
    assert ast.global_variables[0].variable_name == "msImage"
    assert ast.global_variables[1].data_type == "image2DMSArray"
    assert ast.global_variables[1].variable_name == "msImages"
    assert "RWTexture2DMS msImage;" in generated_code
    assert "RWTexture2DMSArray msImages;" in generated_code
    assert "image2DMS msImage;" not in generated_code
    assert "image2DMSArray msImages;" not in generated_code


def test_vulkan_image_buffer_uniform_emits_crossgl_resource():
    code = """
    layout(set = 0, binding = 0) uniform imageBuffer data;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "imageBuffer"
    assert ast.global_variables[0].variable_name == "data"
    assert "RWBuffer data;" in generated_code
    assert "imageBuffer data;" not in generated_code


def test_vulkan_typed_image_buffer_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform iimageBuffer signedData;
    layout(set = 0, binding = 1) uniform uimageBuffer unsignedData;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "iimageBuffer"
    assert ast.global_variables[0].variable_name == "signedData"
    assert ast.global_variables[1].data_type == "uimageBuffer"
    assert ast.global_variables[1].variable_name == "unsignedData"
    assert "RWBuffer<int> signedData;" in generated_code
    assert "RWBuffer<uint> unsignedData;" in generated_code
    assert "iimageBuffer signedData;" not in generated_code
    assert "uimageBuffer unsignedData;" not in generated_code


def test_vulkan_sampler_buffer_uniform_emits_crossgl_resource():
    code = """
    layout(set = 0, binding = 0) uniform samplerBuffer data;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "samplerBuffer"
    assert ast.global_variables[0].variable_name == "data"
    assert "Buffer data;" in generated_code
    assert "samplerBuffer data;" not in generated_code


def test_vulkan_integer_sampler_buffer_uniforms_preserve_crossgl_resource_types():
    code = """
    layout(set = 0, binding = 0) uniform isamplerBuffer signedData;
    layout(set = 0, binding = 1) uniform usamplerBuffer unsignedData;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "isamplerBuffer"
    assert ast.global_variables[0].variable_name == "signedData"
    assert ast.global_variables[1].data_type == "usamplerBuffer"
    assert ast.global_variables[1].variable_name == "unsignedData"
    assert "isamplerBuffer signedData;" in generated_code
    assert "usamplerBuffer unsignedData;" in generated_code
    assert "Buffer<int> signedData;" not in generated_code
    assert "Buffer<uint> unsignedData;" not in generated_code


def test_vulkan_subpass_input_uniforms_emit_crossgl_resources():
    code = """
    layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput colorInput;
    layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInputMS msColorInput;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "subpassInput"
    assert ast.global_variables[0].variable_name == "colorInput"
    assert ast.global_variables[1].data_type == "subpassInputMS"
    assert ast.global_variables[1].variable_name == "msColorInput"
    assert (
        "subpassInput colorInput @set(0) @binding(0) @input_attachment_index(0);"
        in generated_code
    )
    assert (
        "subpassInputMS msColorInput @set(0) @binding(1) @input_attachment_index(1);"
        in generated_code
    )
    assert "Texture2D colorInput" not in generated_code
    assert "Texture2DMS msColorInput" not in generated_code


def test_vulkan_atomic_uint_uniform_emits_crossgl_resource():
    code = """
    layout(set = 0, binding = 0) uniform atomic_uint counter;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "atomic_uint"
    assert ast.global_variables[0].variable_name == "counter"
    assert "RWStructuredBuffer<uint> counter;" in generated_code
    assert "atomic_uint counter;" not in generated_code


def test_vulkan_standalone_sampler_uniforms_preserve_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform sampler compareSampler;
    uniform sampler samplers[4];
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "sampler"
    assert ast.global_variables[0].variable_name == "compareSampler"
    assert ast.global_variables[1].vtype == "sampler"
    assert ast.global_variables[1].name == "samplers[4]"
    assert "sampler compareSampler;" in generated_code
    assert "sampler samplers[4];" in generated_code
    assert "Texture2D compareSampler;" not in generated_code


def test_vulkan_shadow_sampler_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform sampler1DShadow shadowRamp;
    layout(set = 0, binding = 1) uniform sampler1DArrayShadow shadowRamps;
    layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;
    layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadowArray;
    layout(set = 0, binding = 4) uniform samplerCubeShadow shadowCube;
    layout(set = 0, binding = 5) uniform samplerCubeArrayShadow shadowCubeArray;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert [variable.data_type for variable in ast.global_variables] == [
        "sampler1DShadow",
        "sampler1DArrayShadow",
        "sampler2DShadow",
        "sampler2DArrayShadow",
        "samplerCubeShadow",
        "samplerCubeArrayShadow",
    ]
    assert "sampler1DShadow shadowRamp;" in generated_code
    assert "sampler1DArrayShadow shadowRamps;" in generated_code
    assert "sampler2DShadow shadowMap;" in generated_code
    assert "sampler2DArrayShadow shadowArray;" in generated_code
    assert "samplerCubeShadow shadowCube;" in generated_code
    assert "samplerCubeArrayShadow shadowCubeArray;" in generated_code
    assert "Texture1D shadowRamp;" not in generated_code
    assert "Texture1DArray shadowRamps;" not in generated_code
    assert "Texture2D shadowMap;" not in generated_code
    assert "TextureCube shadowCube;" not in generated_code


def test_vulkan_precision_declarations_do_not_emit_crossgl_resources():
    code = """
    precision highp float;
    precision mediump sampler2D;
    layout(set = 0, binding = 0) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(ast.global_variables) == 1
    assert ast.global_variables[0].data_type == "sampler2D"
    assert "Texture2D albedoTex;" in generated_code
    assert "precision" not in generated_code


def test_vulkan_layout_precision_after_in_codegen():
    code = """
    #version 310 es
    precision highp float;
    layout(location = 0) in highp vec2 vUV;
    layout(location = 0) out mediump vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float2 vUV @input @location(0) @highp;" in generated_code
    assert "float4 fragColor @output @location(0) @mediump;" in generated_code


def test_translate_api_accepts_spirv_layout_source(tmp_path):
    import crosstl

    shader_path = tmp_path / "layout_vertex.spvasm"
    shader_path.write_text(LAYOUT_SHADER, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="rust", format_output=False
    )

    assert "pub struct Particles" in generated_code
    assert (
        "static VIEW_PROJ: std::sync::LazyLock<Mat4<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());" in generated_code
    )
    assert (
        "static PARTICLES: std::sync::LazyLock<RWStructuredBuffer<Particles>> = "
        "std::sync::LazyLock::new(|| Default::default());" in generated_code
    )
    assert (
        "static POSITION: std::sync::LazyLock<Vec3<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());" in generated_code
    )
    assert (
        "gl_Position = Vec4::<f32>::new((*POSITION).x, (*POSITION).y, "
        "(*POSITION).z, 1.0);" in generated_code
    )
    assert '#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]' in generated_code


def test_translate_api_accepts_spirv_push_constant_layout_source(tmp_path):
    import crosstl

    shader_path = tmp_path / "push_constant_vertex.spvasm"
    shader_path.write_text(
        """
        layout(push_constant) uniform PushConstants {
            mat4 model;
            vec4 tint;
        } pc;
        void main() {
            gl_Position = pc.model * vec4(1.0, 0.0, 0.0, 1.0);
        }
        """,
        encoding="utf-8",
    )

    generated_cgl = crosstl.translate(
        str(shader_path), backend="cgl", format_output=False
    )
    generated_rust = crosstl.translate(
        str(shader_path), backend="rust", format_output=False
    )

    assert "cbuffer PushConstants @push_constant {" in generated_cgl
    assert "pc.model" not in generated_cgl
    assert "pub struct PushConstants" in generated_rust
    assert (
        "static MODEL: std::sync::LazyLock<Mat4<f32>> = "
        "std::sync::LazyLock::new(|| Default::default());" in generated_rust
    )
    assert '#[cfg_attr(feature = "crossgl_gpu", vertex_shader)]' in generated_rust


def test_vulkan_bitwise_shift_precedence_codegen():
    code = """
    void main() {
        int a = 1;
        int b = 2;
        int c = 3;
        int value = a & b << c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int value = (a & (b << c));" in generated_code
    assert "int value = ((a & b) << c);" not in generated_code


def test_vulkan_bitwise_or_xor_and_precedence_codegen():
    code = """
    void main() {
        int a = 1;
        int b = 2;
        int c = 3;
        int orAnd = a | b & c;
        int xorAnd = a ^ b & c;
        int orXor = a | b ^ c;
        int andEquality = a & b == c;
        int orRelational = a | b < c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int orAnd = (a | (b & c));" in generated_code
    assert "int orAnd = ((a | b) & c);" not in generated_code
    assert "int xorAnd = (a ^ (b & c));" in generated_code
    assert "int xorAnd = ((a ^ b) & c);" not in generated_code
    assert "int orXor = (a | (b ^ c));" in generated_code
    assert "int orXor = ((a | b) ^ c);" not in generated_code
    assert "int andEquality = (a & (b == c));" in generated_code
    assert "int andEquality = ((a & b) == c);" not in generated_code
    assert "int orRelational = (a | (b < c));" in generated_code
    assert "int orRelational = ((a | b) < c);" not in generated_code


def test_vulkan_hex_integer_literals_codegen():
    code = """
    void main() {
        uint mask = 0xFFu;
        uint upper = 0X10U;
        uint shifted = 0x10u << 2u;
        bool selected = (flags & 0x1u) == 0x1u;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "uint mask = 0xFF;" in generated_code
    assert "uint upper = 0X10;" in generated_code
    assert "uint shifted = (0x10 << 2);" in generated_code
    assert "bool selected = ((flags & 0x1) == 0x1);" in generated_code
    assert "xFFu" not in generated_code
    assert "X10U" not in generated_code


def test_struct_codegen():
    code = """
    struct VertexInput {
        vec4 position;
        vec4 color;
    };

    struct VertexOutput {
        vec4 position;
        vec4 color;
    };

    void main() {
        VertexOutput output;
        output.position = vec4(1.0, 0.0, 0.0, 1.0);
        output.color = vec4(0.0, 1.0, 0.0, 1.0);
        gl_Position = output.position;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "VertexOutput output;" in generated_code
    except SyntaxError:
        pytest.fail("Struct parsing or code generation not implemented.")


def test_if_codegen():
    code = """
    void main() {
        vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
        if (color.r > 0.5) {
            color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("If statement parsing or code generation not implemented.")


def test_for_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        for (int i = 0; i < 10; i++) {
            color.r += 0.1;
        }
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")


def test_for_structured_update_codegen():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; items[i]++) {
        }
        for (int i = 0; i < 4; object.field--) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); items[i]++) {" in generated_code
    assert "for (int i = 0; (i < 4); object.field--) {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_for_compound_update_codegen():
    code = """
    void main() {
        for (int i = 0; i < 4; i += 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); i += 1) {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_for_structured_assignment_update_codegen():
    code = """
    void main() {
        int value = 1;
        int items[4];
        for (int i = 0; i < 4; items[i] += value) {
        }
        for (int i = 0; i < 4; object.field = value) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); items[i] += value) {" in generated_code
    assert "for (int i = 0; (i < 4); object.field = value) {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_for_empty_clause_codegen():
    code = """
    void main() {
        int i = 0;
        for (; i < 4; i++) {
        }
        for (int j = 0; ; j++) {
            break;
        }
        for (int k = 0; k < 4; ) {
            k++;
        }
        for (; ; ) {
            break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (; (i < 4); i++) {" in generated_code
    assert "for (int j = 0; ; j++) {" in generated_code
    assert "for (int k = 0; (k < 4); ) {" in generated_code
    assert "for (; ; ) {" in generated_code
    assert "None" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_for_clause_comma_lists_codegen():
    code = """
    void main() {
        for (int i = 0, j = 1; i < 4; i++, j--) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0, j = 1; (i < 4); i++, j--) {" in generated_code
    assert "None" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_while_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        int i = 0;
        while (i < 10) {
            color.r += 0.1;
            i++;
        }
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_continue_codegen():
    code = """
    void main() {
        for (int i = 0; i < 4; i++) {
            if (i == 2) {
                continue;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "continue;" in generated_code
        assert "Unhandled statement type" not in generated_code
    except SyntaxError:
        pytest.fail("Continue statement parsing or code generation not implemented.")


def test_do_while_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        int i = 0;
        do {
            color.r += 0.1;
            i++;
            i--;
        } while (i < 10);
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "do {" in generated_code
        assert "i++;" in generated_code
        assert "i = i++;" not in generated_code
        assert "i--;" in generated_code
        assert "i = i--;" not in generated_code
        assert "} while ((i < 10));" in generated_code
        assert "Unhandled statement type" not in generated_code
    except SyntaxError:
        pytest.fail("Do-while loop parsing or code generation not implemented.")


def test_else_if_codegen():
    code = """
    void main() {
        float value = 0.7;
        vec4 color;

        if (value > 0.8) {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        } else if (value > 0.5) {
            color = vec4(0.0, 1.0, 0.0, 1.0);
        } else {
            color = vec4(0.0, 0.0, 1.0, 1.0);
        }

        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else-if statement parsing or code generation not implemented.")


def test_switch_case_codegen():
    code = """
    void main() {
        int value = 2;
        vec4 color;

        switch (value) {
            case 0:
                color = vec4(1.0, 0.0, 0.0, 1.0);
                break;
            case 1:
                color = vec4(0.0, 1.0, 0.0, 1.0);
                break;
            case 2:
                color = vec4(0.0, 0.0, 1.0, 1.0);
                break;
            default:
                color = vec4(0.5, 0.5, 0.5, 1.0);
                break;
        }

        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "switch (value) {" in generated_code
        assert "case 0:" in generated_code
        assert "case 1:" in generated_code
        assert "case 2:" in generated_code
        assert "default:" in generated_code
        assert generated_code.count("break;") == 4
        assert "break;\n                break;" not in generated_code
        assert "Unhandled statement type" not in generated_code
    except SyntaxError:
        pytest.fail("Switch-case parsing or code generation not implemented.")


def test_switch_fallthrough_codegen_does_not_insert_break():
    code = """
    void main() {
        int value = 0;
        vec4 color;

        switch (value) {
            case 0:
                color = vec4(1.0, 0.0, 0.0, 1.0);
            case 1:
                color = vec4(0.0, 1.0, 0.0, 1.0);
                break;
            default:
                color = vec4(0.5, 0.5, 0.5, 1.0);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("case 1:", 1)[0]
    default_block = generated_code.split("default:", 1)[1].split("}", 1)[0]
    assert "break;" not in case_zero_block
    assert "break;" not in default_block
    assert generated_code.count("break;") == 1
    assert "Unhandled statement type" not in generated_code


def test_switch_return_and_discard_codegen():
    code = """
    void main() {
        int value = 0;
        vec4 color;

        switch (value) {
            case 0:
                discard;
            case 1:
                return;
            default:
                color = vec4(0.5, 0.5, 0.5, 1.0);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "discard;" in generated_code
    assert "return;" in generated_code
    assert "default:" in generated_code
    assert "color = float4(0.5, 0.5, 0.5, 1.0);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_switch_loop_control_codegen_preserves_nesting_without_extra_breaks():
    code = """
    void main() {
        int hits = 0;
        for (int i = 0; i < 4; i++) {
            switch (i) {
                case 0:
                    hits = hits + 1;
                    continue;
                case 1:
                    while (hits < 3) {
                        hits = hits + 1;
                        break;
                    }
                    break;
                default:
                    hits = hits + 100;
                    break;
            }
            hits = hits + 1000;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("case 1:", 1)[0]
    case_one_block = generated_code.split("case 1:", 1)[1].split("default:", 1)[0]

    assert "continue;" in case_zero_block
    assert "break;" not in case_zero_block
    assert "while ((hits < 3)) {" in case_one_block
    assert case_one_block.count("break;") == 2
    assert "            }\n            hits = (hits + 1000);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_loop_inside_switch_case_codegen_keeps_following_case_statement():
    code = """
    void main() {
        int hits = 0;
        int value = 0;
        switch (value) {
            case 0:
                for (int j = 0; j < 2; j++) {
                    hits = hits + j;
                    break;
                }
                hits = hits + 10;
                break;
            default:
                break;
        }
        hits = hits + 1000;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("default:", 1)[0]

    assert "for (int j = 0; (j < 2); j++) {" in case_zero_block
    assert "hits = (hits + j);" in case_zero_block
    assert "hits = (hits + 10);" in case_zero_block
    assert case_zero_block.count("break;") == 2
    assert "        }\n        hits = (hits + 1000);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_bitwise_and_ops_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint mask = 3u;
        uint result = value & mask;  // Bitwise AND
        if (result == 1u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        # Print tokens for debugging
        print("Tokens for bitwise AND test:")
        for i, token in enumerate(tokens):
            print(f"{i}: {token}")

        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise AND operator parsing or code generation not implemented.")


def test_bitwise_or_ops_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint mask = 2u;
        uint result = value | mask;  // Bitwise OR
        if (result == 7u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR operator parsing or code generation not implemented.")


def test_bitwise_xor_ops_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint mask = 3u;
        uint result = value ^ mask;  // Bitwise XOR
        if (result == 6u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise XOR operator parsing or code generation not implemented.")


def test_bitwise_not_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint result = ~value;  // Bitwise NOT
        if (result != 5u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing or code generation not implemented.")


def test_logical_not_codegen():
    code = """
    void main() {
        bool disabled;
        if (!disabled) {
            discard;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (!disabled)" in generated_code
    assert "discard;" in generated_code


def test_bitwise_shift_ops_codegen():
    code = """
    void main() {
        uint value = 4u;
        uint result1 = value << 1;  // Left shift
        uint result2 = value >> 1;  // Right shift
        if (result1 == 8u && result2 == 2u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "uint result1 = (value << 1);" in generated_code
        assert "uint result2 = (value >> 1);" in generated_code
        assert "if (((result1 == 8) && (result2 == 2)))" in generated_code
        assert "((result1 == 8) && result2) == 2" not in generated_code
    except SyntaxError:
        pytest.fail(
            "Bitwise shift operators parsing or code generation not implemented."
        )


def test_const_local_declaration_codegen():
    code = """
    void main() {
        const float scale = 1.0;
        float value = scale;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const float scale = 1.0;" in generated_code
    assert "float value = scale;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_const_vector_declaration_codegen_maps_base_type():
    code = """
    void main() {
        const vec3 tint = vec3(1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const float3 tint = float3(1.0);" in generated_code
    assert "const vec3 tint" not in generated_code


def test_const_matrix_global_declaration_codegen_maps_base_type():
    code = """
    const mat4 VIEW = mat4(1.0);
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const float4x4 VIEW = float4x4(1.0);" in generated_code
    assert "const mat4 VIEW" not in generated_code


def test_const_global_declaration_codegen():
    code = """
    const int MAX_LIGHTS = 4;
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "const int MAX_LIGHTS = 4;" in generated_code


def test_double_dtype_codegen():
    code = """
    void main() {
        double value1 = 3.14159265358979323846;
        double value2 = 2.71828182845904523536;
        double result = value1 + value2;
        if (result > 5.0) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Double data type parsing or code generation not implemented.")


if __name__ == "__main__":
    pytest.main()
