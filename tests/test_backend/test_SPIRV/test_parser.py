from typing import List

import pytest

from crosstl.backend.common_ast import InitializerListNode
from crosstl.backend.SPIRV import VulkanParser
from crosstl.backend.SPIRV.VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    LayoutNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    UnaryOpNode,
    UniformNode,
    VariableNode,
    WhileNode,
)
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = VulkanParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    lexer = VulkanLexer(code)
    return lexer.tokenize()


def test_parse_debug_printf_string_literal_argument_from_vulkan_samples():
    code = """
    #extension GL_EXT_debug_printf : enable
    void main() {
        vec4 pos = vec4(1.0);
        debugPrintfEXT("Position = %v4f", pos);
    }
    """

    ast = parse_code(tokenize_code(code))

    main = next(function for function in ast.functions if function.name == "main")
    call = main.body[1]
    assert isinstance(call, FunctionCallNode)
    assert call.name == "debugPrintfEXT"
    assert call.args[0] == '"Position = %v4f"'


def test_parse_type_leading_array_suffix_from_vulkan_compute_samples():
    code = """
    float conv(in float[9] kernel, in float[9] data) {
        float[9] localKernel;
        return kernel[0] + data[0];
    }
    """

    ast = parse_code(tokenize_code(code))

    conv = next(function for function in ast.functions if function.name == "conv")
    assert conv.params[0].vtype == "float[9]"
    assert conv.params[0].name == "kernel"
    assert conv.params[1].vtype == "float[9]"
    assert isinstance(conv.body[0], VariableNode)
    assert conv.body[0].vtype == "float[9]"
    assert conv.body[0].name == "localKernel"


def test_parse_geometry_input_primitive_parameter_qualifier_from_vulkan_samples():
    code = """
    struct VSOutput {
        vec4 position;
    };
    struct triangle {
        vec4 position;
    };

    void geometry_main(triangle VSOutput input[3]) {
    }
    void typed(triangle value) {
    }
    """

    ast = parse_code(tokenize_code(code))

    geometry_main = next(
        function for function in ast.functions if function.name == "geometry_main"
    )
    typed = next(function for function in ast.functions if function.name == "typed")
    assert geometry_main.params[0].qualifiers == ["triangle"]
    assert geometry_main.params[0].vtype == "VSOutput"
    assert geometry_main.params[0].name == "input[3]"
    assert typed.params[0].qualifiers == []
    assert typed.params[0].vtype == "triangle"
    assert typed.params[0].name == "value"


def test_parse_struct_instance_declarator_from_vulkan_compute_samples():
    code = """
    struct ImageData {
        float avg[9];
    } imageData;
    """

    ast = parse_code(tokenize_code(code))

    assert ast.structs[0].name == "ImageData"
    assert ast.structs[0].members[0].vtype == "float"
    assert ast.structs[0].members[0].name == "avg[9]"


def test_parse_hlsl_attribute_and_semantic_declarations_from_vulkan_samples():
    code = """
    struct VSOutput
    {
        float4 Pos : SV_POSITION;
        [[vk::location(0)]] float2 UV : TEXCOORD0;
    };

    Texture2D textureColor : register(t1);
    SamplerState samplerColor : register(s1);

    cbuffer UBO : register(b0)
    {
        float blurScale;
    }

    [[vk::constant_id(0)]] const int blurdirection = 0;

    [numthreads(16, 1, 1)]
    bool frustumCheck(float4 pos)
    {
        return true;
    }

    float4 main([[vk::location(0)]] float2 inUV : TEXCOORD0) : SV_TARGET
    {
        VSOutput output = (VSOutput)0;
        return float4(0.0);
    }
    """

    ast = parse_code(tokenize_code(code))
    output = ast.structs[0]
    texture, sampler, cbuffer, constant = ast.global_variables
    main = ast.functions[1]

    assert output.members[0].name == "Pos"
    assert output.members[0].semantic == "SV_POSITION"
    assert output.members[1].name == "UV"
    assert output.members[1].semantic == "TEXCOORD0"

    assert texture.vtype == "Texture2D"
    assert texture.name == "textureColor"
    assert texture.semantic == "register(t1)"
    assert sampler.vtype == "SamplerState"
    assert sampler.semantic == "register(s1)"
    assert cbuffer.vtype == "cbuffer"
    assert cbuffer.name == "UBO"
    assert cbuffer.semantic == "register(b0)"
    assert constant.left.name == "blurdirection"

    assert ast.functions[0].name == "frustumCheck"
    assert main.return_type == "float4"
    assert main.semantic == "SV_TARGET"
    assert main.params[0].vtype == "float2"
    assert main.params[0].name == "inUV"
    assert main.params[0].semantic == "TEXCOORD0"
    assert isinstance(main.body[0], AssignmentNode)
    assert main.body[0].left.vtype == "VSOutput"
    assert isinstance(main.body[0].right, FunctionCallNode)
    assert main.body[0].right.name == "VSOutput"
    assert main.body[0].right.args == ["0"]


def test_parse_unsized_array_constructor_from_vulkan_samples():
    code = """
    vec2 triangle_positions[3] = vec2[](
        vec2(0.5, -0.5),
        vec2(0.5, 0.5),
        vec2(-0.5, 0.5)
    );
    """

    ast = parse_code(tokenize_code(code))

    assignment = ast.global_variables[0]
    assert isinstance(assignment, AssignmentNode)
    assert assignment.left.vtype == "vec2"
    assert assignment.left.name == "triangle_positions[3]"
    assert isinstance(assignment.right, FunctionCallNode)
    assert assignment.right.name == "vec2[]"


def test_parse_shader_object_interface_block_from_vulkan_samples():
    code = """
    layout (location = 0) in TerrainVertexData {
        vec2 uv;
        vec3 pos;
        vec3 normal;
        flat uint texture_index;
    } vertex_in;
    """

    ast = parse_code(tokenize_code(code))

    layout = ast.global_variables[0]
    assert isinstance(layout, LayoutNode)
    assert layout.layout_type == "IN"
    assert layout.block_name == "TerrainVertexData"
    assert layout.variable_name == "vertex_in"
    assert layout.struct_fields == [
        ("vec2", "uv"),
        ("vec3", "pos"),
        ("vec3", "normal"),
        ("uint", "texture_index"),
    ]


def test_parse_layout_qualified_block_members_and_texture_descriptor():
    code = """
    layout(set = 0, binding = 0) uniform texture2D Textures[];
    layout(push_constant) uniform Registers
    {
        layout(offset = 4) uint table_offset;
    } registers;
    """

    ast = parse_code(tokenize_code(code))

    descriptor = ast.global_variables[0]
    assert isinstance(descriptor, LayoutNode)
    assert descriptor.data_type == "texture2D"
    assert descriptor.variable_name == "Textures[]"

    push_constants = ast.global_variables[1]
    assert isinstance(push_constants, LayoutNode)
    assert push_constants.block_name == "Registers"
    assert push_constants.struct_fields == [("uint", "table_offset")]
    assert push_constants.variable_name == "registers"


def test_parse_local_multi_declaration_from_vulkan_compute_samples():
    code = """
    void main() {
        vec3 a, b = vec3(1.0), c;
        a = b + c;
    }
    """

    ast = parse_code(tokenize_code(code))

    main = next(function for function in ast.functions if function.name == "main")
    assert isinstance(main.body[0], VariableNode)
    assert main.body[0].vtype == "vec3"
    assert main.body[0].name == "a"
    assert isinstance(main.body[1], AssignmentNode)
    assert main.body[1].left.vtype == "vec3"
    assert main.body[1].left.name == "b"
    assert isinstance(main.body[2], VariableNode)
    assert main.body[2].vtype == "vec3"
    assert main.body[2].name == "c"


def test_parse_numeric_literal_postfix_swizzle_from_vulkan_samples():
    code = """
    void main() {
        vec3 value = 0u.xxx;
        float scalar = 1.0.xxx.x;
        vec4 zeroVector = 0.xxxx;
        float zeroScalar = 0.xxxx.x;
    }
    """

    ast = parse_code(tokenize_code(code))

    main = next(function for function in ast.functions if function.name == "main")
    vector_assignment = main.body[0]
    scalar_assignment = main.body[1]
    zero_vector_assignment = main.body[2]
    zero_scalar_assignment = main.body[3]
    assert isinstance(vector_assignment.right, MemberAccessNode)
    assert vector_assignment.right.object == "0"
    assert vector_assignment.right.member == "xxx"
    assert isinstance(scalar_assignment.right, MemberAccessNode)
    assert scalar_assignment.right.member == "x"
    assert isinstance(scalar_assignment.right.object, MemberAccessNode)
    assert isinstance(zero_vector_assignment.right, MemberAccessNode)
    assert zero_vector_assignment.right.object == "0"
    assert zero_vector_assignment.right.member == "xxxx"
    assert isinstance(zero_scalar_assignment.right, MemberAccessNode)
    assert zero_scalar_assignment.right.member == "x"
    assert isinstance(zero_scalar_assignment.right.object, MemberAccessNode)
    assert zero_scalar_assignment.right.object.object == "0"
    assert zero_scalar_assignment.right.object.member == "xxxx"


def test_parse_parenthesized_expression_postfix_swizzle_from_vulkan_samples():
    code = """
    void main() {
        vec3 eyes = (gl_Position).xyz;
    }
    """

    ast = parse_code(tokenize_code(code))

    main = next(function for function in ast.functions if function.name == "main")
    assignment = main.body[0]
    assert isinstance(assignment.right, MemberAccessNode)
    assert assignment.right.member == "xyz"
    assert isinstance(assignment.right.object, VariableNode)
    assert assignment.right.object.name == "gl_Position"


def test_parse_push_constant_in_mixed_layout_qualifiers():
    code = """
    layout(row_major, push_constant) uniform PushConstants {
        mat4 model;
    } pc;
    """

    ast = parse_code(tokenize_code(code))

    layout = ast.global_variables[0]
    assert isinstance(layout, LayoutNode)
    assert layout.push_constant is True
    assert layout.qualifiers == [("row_major", None)]
    assert layout.block_name == "PushConstants"
    assert layout.variable_name == "pc"


def test_parse_unbraced_if_body_from_vulkan_compute_samples():
    code = """
    void main() {
        uint index = 4u;
        if (index > 2u)
            return;
    }
    """

    ast = parse_code(tokenize_code(code))

    main = next(function for function in ast.functions if function.name == "main")
    if_stmt = main.body[1]
    assert isinstance(if_stmt, IfNode)
    assert isinstance(if_stmt.if_body[0], ReturnNode)


def test_parse_standalone_scoped_block_from_vulkan_samples():
    code = """
    void main() {
        {
            outColor = inColor;
        }
        gl_Position = inPos;
    }
    """

    ast = parse_code(tokenize_code(code))

    main = next(function for function in ast.functions if function.name == "main")
    assert len(main.body) == 2
    assert isinstance(main.body[0], AssignmentNode)
    assert isinstance(main.body[1], AssignmentNode)


def test_parse_ray_tracing_payload_layout_qualifier_from_glsl_samples():
    code = """
    struct RayPayload {
        vec3 color;
        float distance;
    };
    layout(location = 0) rayPayloadEXT RayPayload rayPayload;
    layout(location = 1) callableDataEXT vec3 outColor;
    hitAttributeEXT vec2 attribs;
    """

    ast = parse_code(tokenize_code(code))

    payload = ast.global_variables[0]
    callable_data = ast.global_variables[1]
    hit_attribute = ast.global_variables[2]
    assert isinstance(payload, LayoutNode)
    assert payload.data_type == "RayPayload"
    assert payload.variable_name == "rayPayload"
    assert payload.declaration_qualifiers == ["rayPayloadEXT"]
    assert callable_data.declaration_qualifiers == ["callableDataEXT"]
    assert hit_attribute.vtype == "hitAttributeEXT vec2"
    assert hit_attribute.name == "attribs"


def test_parse_mesh_shader_const_array_brace_initializers_from_glsl_samples():
    code = """
    const vec4[3] positions = {
        vec4(0.0, -1.0, 0.0, 1.0),
        vec4(-1.0, 1.0, 0.0, 1.0),
        vec4(1.0, 1.0, 0.0, 1.0)
    };
    void main() {
        float r = 0.1;
        float g = 0.2;
        float b = 0.3;
        float[3] rgb_values = { r, g, b };
    }
    """

    ast = parse_code(tokenize_code(code))

    positions = ast.global_variables[0]
    assert positions.left.vtype == "const vec4[3]"
    assert isinstance(positions.right, InitializerListNode)
    assert len(positions.right.elements) == 3

    rgb_values = ast.functions[0].body[3]
    assert rgb_values.left.vtype == "float[3]"
    assert isinstance(rgb_values.right, InitializerListNode)
    assert [element.name for element in rgb_values.right.elements] == ["r", "g", "b"]


def test_parse_template_style_tensor_layout_type_from_glsl_samples():
    code = """
    layout(set = 0, binding = 0) writeonly uniform tensorARM<float, 4> output_tensor;
    """

    ast = parse_code(tokenize_code(code))

    layout = ast.global_variables[0]
    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "tensorARM<float, 4>"
    assert layout.variable_name == "output_tensor"
    assert layout.declaration_qualifiers == ["writeonly"]


def test_parse_user_defined_return_type_and_const_parameter_from_glsl_samples():
    code = """
    struct Sphere {
        vec3 center;
        float radius;
    };
    struct Triangle {
        vec3 normal;
    };
    Triangle unpackTriangle(uint index, const Sphere sphere) {
        Triangle tri;
        return tri;
    }
    """

    ast = parse_code(tokenize_code(code))

    function = ast.functions[0]
    assert function.return_type == "Triangle"
    assert function.params[1].vtype == "Sphere"
    assert function.params[1].qualifiers == ["const"]


def test_parse_restrict_buffer_reference_local_declaration_from_glsl_samples():
    code = """
    void main() {
        restrict Position positions = registers.references.buffers[slice];
    }
    """

    ast = parse_code(tokenize_code(code))

    declaration = ast.functions[0].body[0]
    assert declaration.left.vtype == "restrict Position"
    assert declaration.left.name == "positions"


def test_parse_groupshared_hlsl_declarations_from_vulkan_samples():
    code = """
    struct DummyPayLoad {
        uint dummyData;
    };

    groupshared float4 sharedData[1024];
    groupshared DummyPayLoad dummyPayLoad;

    void main() {
        DispatchMesh(3, 1, 1, dummyPayLoad);
    }
    """

    ast = parse_code(tokenize_code(code))

    shared_data = ast.global_variables[0]
    payload = ast.global_variables[1]
    dispatch = ast.functions[0].body[0]
    assert shared_data.vtype == "groupshared float4"
    assert shared_data.name == "sharedData[1024]"
    assert payload.vtype == "groupshared DummyPayLoad"
    assert payload.name == "dummyPayLoad"
    assert isinstance(dispatch, FunctionCallNode)
    assert dispatch.name == "DispatchMesh"


def test_parse_empty_statement_after_block_from_glsl_samples():
    code = """
    void main() {
        if (sampledColor.a < 0.5) {
            discard;
        };
        outColor = vec4(0.0);
    }
    """

    ast = parse_code(tokenize_code(code))

    main = ast.functions[0]
    assert len(main.body) == 2
    assert isinstance(main.body[0], IfNode)
    assert isinstance(main.body[1], AssignmentNode)


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

SPIRV_TOOLS_FORWARD_POINTER_STRUCT_ASSEMBLY = """
; Reduced from Khronos SPIRV-Tools test/diff/diff_files/OpTypeForwardPointer_basic_src.spvasm.
OpCapability Kernel
OpCapability Addresses
OpCapability Linkage
OpMemoryModel Logical OpenCL
OpName %structptr "structptr"
OpTypeForwardPointer %structptr UniformConstant
%uint = OpTypeInt 32 0
%structt1 = OpTypeStruct %structptr %uint
%structt2 = OpTypeStruct %uint %structptr
%structptr = OpTypePointer UniformConstant %structt1
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


def test_spirv_assembly_location_decorated_interfaces_parse():
    tokens = tokenize_code(SPIRV_TOOLS_BASIC_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    input_layout = ast.global_variables[0]
    output_layout = ast.global_variables[1]
    builtin_layout = ast.global_variables[2]

    assert ast.spirv_assembly is True
    assert ast.spirv_entry_points == [
        {
            "execution_model": "Vertex",
            "id": "%22",
            "name": "main",
            "interface_ids": ["%4", "%14", "%19"],
        }
    ]
    assert ast.spirv_names["%4"] == "_ua_position"
    assert ast.spirv_decorations["%17"] == [("Block", [])]
    assert ast.spirv_member_decorations["%17"] == [("0", "BuiltIn", ["Position"])]
    assert ast.spirv_types["%2"]["name"] == "vec4"
    assert ast.spirv_types["%21"] == {
        "kind": "function",
        "return_type": "%20",
        "parameter_types": [],
    }
    assert len(ast.functions) == 1
    assert isinstance(ast.functions[0], FunctionNode)
    assert ast.functions[0].name == "main"
    assert ast.functions[0].return_type == "void"
    assert ast.functions[0].params == []
    assert ast.functions[0].spirv_id == "%22"
    assert ast.functions[0].spirv_function_control == "None"
    assert ast.functions[0].spirv_function_type_id == "%21"
    assert [
        opcode for _rid, opcode, _ops, _line in ast.functions[0].spirv_instructions
    ] == [
        "OpFunction",
        "OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    assert len(ast.global_variables) == 3
    assert isinstance(input_layout, LayoutNode)
    assert input_layout.spirv_id == "%4"
    assert input_layout.spirv_storage_class == "Input"
    assert input_layout.spirv_decorations == [("Location", ["0"])]
    assert input_layout.layout_type == "IN"
    assert input_layout.data_type == "vec4"
    assert input_layout.variable_name == "_ua_position"
    assert input_layout.qualifiers == [("location", "0")]

    assert isinstance(output_layout, LayoutNode)
    assert output_layout.spirv_id == "%14"
    assert output_layout.spirv_storage_class == "Output"
    assert output_layout.layout_type == "OUT"
    assert output_layout.data_type == "vec4"
    assert output_layout.variable_name == "ANGLEXfbPosition"
    assert output_layout.qualifiers == [("location", "0")]

    assert isinstance(builtin_layout, LayoutNode)
    assert builtin_layout.spirv_id == "%19.0"
    assert builtin_layout.spirv_storage_class == "Output"
    assert builtin_layout.layout_type == "OUT"
    assert builtin_layout.data_type == "vec4"
    assert builtin_layout.variable_name == "gl_Position"
    assert builtin_layout.qualifiers == [("builtin", "Position")]
    assert builtin_layout.spirv_decorations == [("BuiltIn", ["Position"])]


def test_spirv_assembly_matrix_interface_parse():
    tokens = tokenize_code(SPIRV_MATRIX_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    input_layout = ast.global_variables[0]

    assert ast.spirv_types["%mat4"]["kind"] == "matrix"
    assert ast.spirv_types["%mat4"]["name"] == "mat4"
    assert input_layout.spirv_id == "%model"
    assert input_layout.layout_type == "IN"
    assert input_layout.data_type == "mat4"
    assert input_layout.variable_name == "model"
    assert input_layout.qualifiers == [("location", "0")]


def test_spirv_assembly_push_constant_block_parse():
    tokens = tokenize_code(SPIRV_PUSH_CONSTANT_ASSEMBLY)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert layout.layout_type == "UNIFORM"
    assert layout.push_constant is True
    assert layout.block_name == "PushConstants"
    assert layout.variable_name == "pc"
    assert layout.struct_fields == [("mat4", "model"), ("vec4", "tint")]
    assert layout.spirv_storage_class == "PushConstant"


def test_spirv_assembly_uniform_block_parse():
    tokens = tokenize_code(SPIRV_UNIFORM_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert layout.layout_type == "UNIFORM"
    assert layout.push_constant is False
    assert layout.block_name == "Camera"
    assert layout.variable_name == "camera"
    assert layout.struct_fields == [("mat4", "viewProj"), ("vec4", "tint")]
    assert layout.qualifiers == [("set", "0"), ("binding", "2")]
    assert layout.spirv_storage_class == "Uniform"


def test_spirv_assembly_buffer_block_parse():
    tokens = tokenize_code(SPIRV_BUFFER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert layout.layout_type == "BUFFER"
    assert layout.push_constant is False
    assert layout.block_name == "Data"
    assert layout.variable_name == "data"
    assert layout.struct_fields == [("vec4", "value")]
    assert layout.qualifiers == [("set", "0"), ("binding", "1")]
    assert layout.spirv_storage_class == "Uniform"


def test_spirv_assembly_readonly_buffer_block_parse():
    tokens = tokenize_code(SPIRV_READONLY_BUFFER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert layout.layout_type == "BUFFER"
    assert layout.variable_name == "data"
    assert layout.declaration_qualifiers == ["readonly"]
    assert layout.spirv_decorations == [
        ("BufferBlock", []),
        ("DescriptorSet", ["0"]),
        ("Binding", ["1"]),
        ("NonWritable", []),
    ]


def test_spirv_assembly_runtime_array_buffer_block_parse():
    tokens = tokenize_code(SPIRV_RUNTIME_ARRAY_BUFFER_BLOCK_ASSEMBLY)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert layout.layout_type == "BUFFER"
    assert layout.block_name == "StorageBuffer"
    assert layout.variable_name == "storage"
    assert layout.struct_fields == [("uint", "header"), ("uint", "payload[]")]
    assert layout.qualifiers == [("set", "0"), ("binding", "4")]
    assert layout.spirv_storage_class == "Uniform"


def test_spirv_tools_array_length_parse():
    tokens = tokenize_code(SPIRV_TOOLS_ARRAY_LENGTH_ASSEMBLY)
    ast = parse_code(tokens)
    storage_layout = ast.global_variables[0]
    output_layout = ast.global_variables[1]
    assignment = ast.functions[0].body[0]

    assert ast.spirv_assembly is True
    assert storage_layout.layout_type == "BUFFER"
    assert storage_layout.block_name == "StorageBuffer"
    assert storage_layout.variable_name == "storage"
    assert storage_layout.struct_fields == [("uint", "header"), ("uint", "payload[]")]
    assert storage_layout.qualifiers == [("set", "0"), ("binding", "0")]
    assert storage_layout.spirv_storage_class == "StorageBuffer"
    assert output_layout.layout_type == "OUT"
    assert output_layout.data_type == "uint"
    assert output_layout.variable_name == "countOut"
    assert ast.spirv_types["%runtimearr_uint"]["kind"] == "runtime_array"

    assert ast.spirv_constant_types == {}
    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, VariableNode)
    assert assignment.left.name == "countOut"
    assert isinstance(assignment.right, FunctionCallNode)
    assert assignment.right.name == "spirvArrayLength"
    assert isinstance(assignment.right.args[0], VariableNode)
    assert assignment.right.args[0].name == "storage"
    assert assignment.right.args[1] == "1"


def test_spirv_assembly_specialization_constants_parse():
    tokens = tokenize_code(SPIRV_SPEC_CONSTANT_ASSEMBLY)
    ast = parse_code(tokens)
    max_lights = ast.global_variables[0]
    enable_shadows = ast.global_variables[1]

    assert ast.spirv_assembly is True
    assert max_lights.layout_type == "CONST"
    assert max_lights.qualifiers == [("constant_id", "0")]
    assert max_lights.spirv_id == "%max_lights"
    assert max_lights.spirv_decorations == [("SpecId", ["0"])]
    assert max_lights.declaration.left.vtype == "const uint"
    assert max_lights.declaration.left.name == "MAX_LIGHTS"
    assert max_lights.declaration.right == "4"

    assert enable_shadows.layout_type == "CONST"
    assert enable_shadows.qualifiers == [("constant_id", "1")]
    assert enable_shadows.declaration.left.vtype == "const bool"
    assert enable_shadows.declaration.left.name == "ENABLE_SHADOWS"
    assert enable_shadows.declaration.right == "true"


def test_spirv_assembly_numeric_id_specialization_constant_parse():
    tokens = tokenize_code(SPIRV_NUMERIC_ID_SPEC_CONSTANT_ASSEMBLY)
    ast = parse_code(tokens)
    constant = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert constant.layout_type == "CONST"
    assert constant.qualifiers == [("constant_id", "0")]
    assert constant.spirv_id == "%104"
    assert constant.declaration.left.vtype == "const uint"
    assert constant.declaration.left.name == "spec_constant_0"
    assert constant.declaration.right == "1"


def test_spirv_assembly_specialization_constant_composite_and_op_parse():
    tokens = tokenize_code(SPIRV_SPEC_CONSTANT_COMPOSITE_OP_ASSEMBLY)
    ast = parse_code(tokens)
    layouts = {
        layout.declaration.left.name: layout
        for layout in ast.global_variables
        if isinstance(layout, LayoutNode)
    }
    total = layouts["WORKGROUP_TOTAL"]
    workgroup_size = layouts["gl_WorkGroupSize"]

    assert ast.spirv_assembly is True
    assert ast.spirv_spec_constant_ids == [
        "%width",
        "%height",
        "%total",
        "%size",
    ]
    assert ast.spirv_constant_types["%size"] == "%v3uint"
    assert ast.spirv_constants["%total"] is total.declaration.right
    assert ast.structs[0].name == "SizedBlock"
    assert ast.structs[0].members[0].name == (
        "values[(WORKGROUP_WIDTH + WORKGROUP_HEIGHT)]"
    )

    assert total.layout_type == "CONST"
    assert total.qualifiers == []
    assert total.declaration.left.vtype == "const uint"
    assert isinstance(total.declaration.right, BinaryOpNode)
    assert total.declaration.right.left == "WORKGROUP_WIDTH"
    assert total.declaration.right.op == "+"
    assert total.declaration.right.right == "WORKGROUP_HEIGHT"

    assert workgroup_size.layout_type == "CONST"
    assert workgroup_size.qualifiers == [("builtin", "WorkgroupSize")]
    assert workgroup_size.spirv_decorations == [("BuiltIn", ["WorkgroupSize"])]
    assert workgroup_size.declaration.left.vtype == "const uvec3"
    assert isinstance(workgroup_size.declaration.right, FunctionCallNode)
    assert workgroup_size.declaration.right.name == "uvec3"
    assert workgroup_size.declaration.right.args == [
        "WORKGROUP_WIDTH",
        "WORKGROUP_HEIGHT",
        "1",
    ]


def test_glslang_web_comp_barrier_instructions_parse():
    tokens = tokenize_code(SPIRV_GLSLANG_WEB_COMP_BARRIER_ASSEMBLY)
    ast = parse_code(tokens)
    body = ast.functions[0].body

    assert ast.spirv_assembly is True
    assert [stmt.name for stmt in body[:3]] == [
        "spirvControlBarrier",
        "spirvMemoryBarrier",
        "spirvMemoryBarrier",
    ]
    assert body[0].args == ["2", "2", "264"]
    assert body[1].args == ["1", "3400"]
    assert body[2].args == ["2", "3400"]
    assert isinstance(body[3], ReturnNode)


def test_spirv_assembly_uniform_constant_resources_parse():
    tokens = tokenize_code(SPIRV_UNIFORM_CONSTANT_RESOURCE_ASSEMBLY)
    ast = parse_code(tokens)
    combined = ast.global_variables[0]
    linear_sampler = ast.global_variables[1]

    assert ast.spirv_assembly is True
    assert ast.spirv_types["%img"]["kind"] == "image"
    assert ast.spirv_types["%img"]["name"] == "sampler2D"
    assert ast.spirv_types["%sampled"]["kind"] == "sampled_image"
    assert ast.spirv_types["%sampled"]["name"] == "sampler2D"
    assert ast.spirv_types["%sampler"]["name"] == "sampler"

    assert combined.layout_type == "UNIFORM"
    assert combined.data_type == "sampler2D"
    assert combined.variable_name == "combinedTex"
    assert combined.qualifiers == [("set", "0"), ("binding", "0")]
    assert combined.spirv_storage_class == "UniformConstant"

    assert linear_sampler.layout_type == "UNIFORM"
    assert linear_sampler.data_type == "sampler"
    assert linear_sampler.variable_name == "linearSampler"
    assert linear_sampler.qualifiers == [("set", "0"), ("binding", "1")]
    assert linear_sampler.spirv_storage_class == "UniformConstant"


def test_spirv_assembly_storage_image_format_parse():
    tokens = tokenize_code(SPIRV_STORAGE_IMAGE_FORMAT_ASSEMBLY)
    ast = parse_code(tokens)
    storage_image = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert ast.spirv_types["%image"]["format"] == "R32ui"
    assert storage_image.layout_type == "UNIFORM"
    assert storage_image.data_type == "uimage2D"
    assert storage_image.variable_name == "storageImage"
    assert storage_image.qualifiers == [
        ("set", "0"),
        ("binding", "0"),
        ("r32ui", None),
    ]
    assert storage_image.declaration_qualifiers == ["readonly"]
    assert storage_image.spirv_storage_class == "UniformConstant"


def test_spirv_assembly_flat_location_interface_parse():
    tokens = tokenize_code(SPIRV_TOOLS_FLAT_LOCATION_ASSEMBLY)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert ast.spirv_assembly is True
    assert layout.layout_type == "IN"
    assert layout.data_type == "uint"
    assert layout.variable_name == "input_flat_u32"
    assert layout.qualifiers == [("location", "0")]
    assert layout.declaration_qualifiers == ["flat"]
    assert layout.spirv_decorations == [("Flat", []), ("Location", ["0"])]


def test_spirv_assembly_forward_pointer_structs_parse():
    tokens = tokenize_code(SPIRV_TOOLS_FORWARD_POINTER_STRUCT_ASSEMBLY)
    ast = parse_code(tokens)

    assert ast.spirv_assembly is True
    assert ast.global_variables == []
    assert [struct.name for struct in ast.structs] == ["structt1", "structt2"]
    assert all(isinstance(struct, StructNode) for struct in ast.structs)
    assert [(member.vtype, member.name) for member in ast.structs[0].members] == [
        ("structptr", "member0"),
        ("uint", "member1"),
    ]
    assert [(member.vtype, member.name) for member in ast.structs[1].members] == [
        ("uint", "member0"),
        ("structptr", "member1"),
    ]


def test_spirv_tools_gl_pervertex_access_chain_parse():
    tokens = tokenize_code(SPIRV_TOOLS_GLPERVERTEX_ACCESS_CHAIN_ASSEMBLY)
    ast = parse_code(tokens)

    assert ast.spirv_assembly is True
    assert [
        (node.layout_type, node.data_type, node.variable_name, node.qualifiers)
        for node in ast.global_variables
    ] == [
        ("IN", "vec4", "_ua_position", [("location", "0")]),
        ("OUT", "vec4", "gl_Position", [("builtin", "Position")]),
        ("OUT", "float", "gl_PointSize", [("builtin", "PointSize")]),
        ("OUT", "float", "gl_ClipDistance[8]", [("builtin", "ClipDistance")]),
        ("OUT", "float", "gl_CullDistance[8]", [("builtin", "CullDistance")]),
    ]
    assignment = ast.functions[0].body[0]
    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, VariableNode)
    assert assignment.left.name == "gl_Position"
    assert isinstance(assignment.right, VariableNode)
    assert assignment.right.name == "_ua_position"


def test_spirv_spec_vector_insert_dynamic_parse():
    tokens = tokenize_code(SPIRV_SPEC_VECTOR_INSERT_DYNAMIC_ASSEMBLY)
    ast = parse_code(tokens)
    assignment = ast.functions[0].body[0]

    assert ast.spirv_assembly is True
    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.right, FunctionCallNode)
    assert assignment.right.name == "spirvVectorInsertDynamic"
    assert [arg.name for arg in assignment.right.args] == [
        "inputVec",
        "insertValue",
        "index",
    ]


def test_spirv_spec_undef_parse():
    tokens = tokenize_code(SPIRV_SPEC_UNDEF_ASSEMBLY)
    ast = parse_code(tokens)
    module_assignment = ast.functions[0].body[0]
    body_assignment = ast.functions[0].body[1]
    undef = ast.spirv_constants["%undef"]

    assert ast.spirv_assembly is True
    assert ast.spirv_constant_types["%undef"] == "%v4float"
    assert isinstance(undef, FunctionCallNode)
    assert undef.name == "spirvUndef_vec4"
    assert undef.args == []
    assert isinstance(module_assignment, AssignmentNode)
    assert module_assignment.right is undef
    assert isinstance(body_assignment, AssignmentNode)
    assert isinstance(body_assignment.right, FunctionCallNode)
    assert body_assignment.right.name == "spirvUndef_vec4"
    assert body_assignment.right.args == []


def test_spirv_spec_constant_null_parse():
    tokens = tokenize_code(SPIRV_SPEC_CONSTANT_NULL_ASSEMBLY)
    ast = parse_code(tokens)
    module_assignment = ast.functions[0].body[0]
    body_assignment = ast.functions[0].body[1]
    null_value = ast.spirv_constants["%null_vec"]

    assert ast.spirv_assembly is True
    assert ast.spirv_constant_types["%null_vec"] == "%v4float"
    assert isinstance(null_value, FunctionCallNode)
    assert null_value.name == "spirvNull_vec4"
    assert null_value.args == []
    assert isinstance(module_assignment, AssignmentNode)
    assert module_assignment.right is null_value
    assert isinstance(body_assignment, AssignmentNode)
    assert isinstance(body_assignment.right, FunctionCallNode)
    assert body_assignment.right.name == "spirvNull_vec4"
    assert body_assignment.right.args == []


def test_spirv_spec_fragment_termination_parse():
    tokens = tokenize_code(SPIRV_SPEC_FRAGMENT_TERMINATION_ASSEMBLY)
    ast = parse_code(tokens)
    body = ast.functions[0].body

    assert ast.spirv_assembly is True
    assert [type(stmt) for stmt in body[:3]] == [
        DiscardNode,
        DiscardNode,
        DiscardNode,
    ]
    assert isinstance(body[3], ReturnNode)


def test_spirv_assembly_function_only_module_is_preserved():
    code = """
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpEntryPoint Vertex %main "main"
    %void = OpTypeVoid
    %fn = OpTypeFunction %void
    %main = OpFunction %void None %fn
    OpFunctionEnd
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.spirv_assembly is True
    assert ast.global_variables == []
    assert ast.structs == []
    assert ast.spirv_entry_points == [
        {
            "execution_model": "Vertex",
            "id": "%main",
            "name": "main",
            "interface_ids": [],
        }
    ]
    assert [function.name for function in ast.functions] == ["main"]
    assert ast.functions[0].return_type == "void"
    assert ast.functions[0].spirv_raw_instructions[-1]["opcode"] == "OpFunctionEnd"


def test_spirv_assembly_execution_mode_metadata_is_preserved():
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
    function = ast.functions[0]

    assert ast.spirv_execution_modes == {
        "%main": [
            {
                "opcode": "OpExecutionMode",
                "mode": "LocalSize",
                "operands": ["8", "4", "1"],
            }
        ]
    }
    assert function.spirv_execution_model == "GLCompute"
    assert function.spirv_entry_point == ast.spirv_entry_points[0]
    assert function.spirv_execution_modes == ast.spirv_execution_modes["%main"]


def test_spirv_assembly_function_parameters_parse():
    code = """
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    OpName %scale "scale"
    %float = OpTypeFloat 32
    %fn = OpTypeFunction %float %float
    %helper = OpFunction %float None %fn
    %scale = OpFunctionParameter %float
    %label = OpLabel
    OpReturnValue %scale
    OpFunctionEnd
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert function.return_type == "float"
    assert [
        (param.vtype, param.name, param.spirv_id, param.spirv_type_id)
        for param in function.params
    ] == [("float", "scale", "%scale", "%float")]
    assert "OpFunctionParameter" in [
        opcode for _rid, opcode, _ops, _line in function.spirv_instructions
    ]


def test_spirv_assembly_type_only_module_is_rejected():
    code = """
    OpCapability Shader
    OpMemoryModel Logical GLSL450
    %void = OpTypeVoid
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="only partially supported"):
        parse_code(tokens)


def test_mod_parsing():
    code = """

    void main() {
        int a = 10 % 3;  // Basic modulus
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_bitwise_not_parsing():
    code = """
    void main() {
        int a = 5;
        int b = ~a;  // Bitwise NOT
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing not implemented")


def test_float_suffix_literals_parse_as_numbers():
    code = """
    void main() {
        float direct = 2.0f;
        float bare_fraction = 1.f;
        float leading_dot = .5F;
        float exponent = 2.3283064365386963e-10;
        float suffixed_exponent = 6.0e+3f;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    values = [assignment.right for assignment in ast.functions[0].body]

    assert values == [
        "2.0",
        "1.",
        ".5",
        "2.3283064365386963e-10",
        "6.0e+3",
    ]


def test_function_parameter_qualifiers_parse():
    code = """
    void accumulate(in vec3 normal, inout float weight, out vec4 color) {
        color = vec4(normal, weight);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert [(p.vtype, p.name, p.qualifiers) for p in function.params] == [
        ("vec3", "normal", ["in"]),
        ("float", "weight", ["inout"]),
        ("vec4", "color", ["out"]),
    ]


def test_void_parameter_list_parses_as_empty_parameters():
    # Khronos glslang Test/spv.debuginfo.glsl.geom uses this entry point form.
    code = """
    void main(void) {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert function.return_type == "void"
    assert function.name == "main"
    assert function.params == []
    assert isinstance(function.body[0], ReturnNode)


def test_function_parameter_array_suffixes_parse():
    code = """
    void sampleResources(sampler2D textures[4], uimage2D outputs[2]) {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert [(p.vtype, p.name) for p in function.params] == [
        ("sampler2D", "textures[4]"),
        ("uimage2D", "outputs[2]"),
    ]


def test_struct_array_members_preserve_suffixes():
    code = """
    struct LightBlock {
        vec3 positions[4];
        float weights[4];
    };
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [(member.vtype, member.name) for member in members] == [
        ("vec3", "positions[4]"),
        ("float", "weights[4]"),
    ]


def test_continue_parsing():
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
        loop = ast.functions[0].body[0]
        condition = loop.body[0]

        assert isinstance(loop, ForNode)
        assert isinstance(condition, IfNode)
        assert isinstance(condition.if_body[0], ContinueNode)
    except SyntaxError:
        pytest.fail("Continue statement parsing not implemented")


def test_for_update_parses_structured_postfix_targets():
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
    array_loop = ast.functions[0].body[1]
    member_loop = ast.functions[0].body[2]

    assert isinstance(array_loop, ForNode)
    assert isinstance(array_loop.update, UnaryOpNode)
    assert array_loop.update.op == "POST_INCREMENT"
    assert isinstance(array_loop.update.operand, ArrayAccessNode)
    assert array_loop.update.operand.array.name == "items"
    assert array_loop.update.operand.index.name == "i"

    assert isinstance(member_loop.update, UnaryOpNode)
    assert member_loop.update.op == "POST_DECREMENT"
    assert isinstance(member_loop.update.operand, MemberAccessNode)
    assert member_loop.update.operand.object.name == "object"
    assert member_loop.update.operand.member == "field"


def test_for_update_parses_structured_prefix_target():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; ++items[i]) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[1]

    assert isinstance(loop.update, UnaryOpNode)
    assert loop.update.op == "PRE_INCREMENT"
    assert isinstance(loop.update.operand, ArrayAccessNode)


def test_for_update_parses_compound_assignment():
    code = """
    void main() {
        for (int i = 0; i < 4; i += 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop.update, AssignmentNode)
    assert loop.update.operator == "+="
    assert loop.update.left.name == "i"
    assert loop.update.right == "1"


def test_for_update_parses_structured_assignment_targets():
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
    array_loop = ast.functions[0].body[2]
    member_loop = ast.functions[0].body[3]

    assert isinstance(array_loop.update, AssignmentNode)
    assert array_loop.update.operator == "+="
    assert isinstance(array_loop.update.left, ArrayAccessNode)
    assert array_loop.update.left.array.name == "items"
    assert array_loop.update.left.index.name == "i"
    assert array_loop.update.right.name == "value"

    assert isinstance(member_loop.update, AssignmentNode)
    assert member_loop.update.operator == "="
    assert isinstance(member_loop.update.left, MemberAccessNode)
    assert member_loop.update.left.object.name == "object"
    assert member_loop.update.left.member == "field"
    assert member_loop.update.right.name == "value"


def test_for_parses_empty_clauses():
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
    loops = ast.functions[0].body[1:]

    init_empty_loop = loops[0]
    condition_empty_loop = loops[1]
    update_empty_loop = loops[2]
    all_empty_loop = loops[3]

    assert isinstance(init_empty_loop, ForNode)
    assert init_empty_loop.init is None
    assert isinstance(init_empty_loop.condition, BinaryOpNode)
    assert isinstance(init_empty_loop.update, UnaryOpNode)

    assert isinstance(condition_empty_loop.init, AssignmentNode)
    assert condition_empty_loop.condition is None
    assert isinstance(condition_empty_loop.update, UnaryOpNode)
    assert isinstance(condition_empty_loop.body[0], BreakNode)

    assert isinstance(update_empty_loop.init, AssignmentNode)
    assert isinstance(update_empty_loop.condition, BinaryOpNode)
    assert update_empty_loop.update is None

    assert all_empty_loop.init is None
    assert all_empty_loop.condition is None
    assert all_empty_loop.update is None


def test_for_clause_comma_lists_parse():
    code = """
    void main() {
        for (int i = 0, j = 1; i < 4; i++, j--) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop.init, list)
    assert len(loop.init) == 2
    assert all(isinstance(item, AssignmentNode) for item in loop.init)
    assert loop.init[0].left.vtype == "int"
    assert loop.init[0].left.name == "i"
    assert loop.init[1].left.vtype == "int"
    assert loop.init[1].left.name == "j"

    assert isinstance(loop.update, list)
    assert [update.op for update in loop.update] == [
        "POST_INCREMENT",
        "POST_DECREMENT",
    ]


def test_for_update_rejects_method_call_target():
    code = """
    void main() {
        for (int i = 0; i < 4; object.method()++) {
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Invalid update target: MethodCallNode"):
        parse_code(tokens)


def test_for_update_rejects_structured_postfix_trailing_assignment():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; items[i]++ = 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected RPAREN, got EQUALS"):
        parse_code(tokens)


def test_switch_parsing_preserves_cases_default_and_breaks():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            case 0:
                value = 2;
                break;
            case 1:
                value = 3;
            default:
                value = 4;
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[1]

    assert isinstance(switch, SwitchNode)
    assert len(switch.cases) == 3
    assert all(isinstance(case, CaseNode) for case in switch.cases)
    assert switch.cases[0].value == "0"
    assert isinstance(switch.cases[0].body[-1], BreakNode)
    assert switch.cases[1].value == "1"
    assert not isinstance(switch.cases[1].body[-1], BreakNode)
    assert switch.cases[2].value is None
    assert isinstance(switch.cases[2].body[-1], BreakNode)


def test_switch_rejects_duplicate_default_labels():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            default:
                value = 1;
            case 0:
                value = 2;
            default:
                value = 3;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="duplicate default"):
        parse_code(tokens)


def test_switch_node_preserves_empty_default_case():
    switch = SwitchNode("value", [], default_case=[])

    assert switch.default_case == []
    assert switch.default == []


def test_switch_parsing_preserves_return_and_discard_statements():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            case 0:
                discard;
            case 1:
                return;
            default:
                value = 4;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[1]

    assert isinstance(switch, SwitchNode)
    assert isinstance(switch.cases[0].body[0], DiscardNode)
    assert isinstance(switch.cases[1].body[0], ReturnNode)
    assert switch.cases[1].body[0].value is None


def test_switch_loop_control_parsing_keeps_nearest_nested_blocks():
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
    loop = ast.functions[0].body[1]
    switch = loop.body[0]
    case_zero, case_one, default_case = switch.cases

    assert isinstance(loop, ForNode)
    assert isinstance(switch, SwitchNode)
    assert isinstance(loop.body[1], AssignmentNode)
    assert isinstance(case_zero.body[-1], ContinueNode)
    assert isinstance(case_one.body[0], WhileNode)
    assert isinstance(case_one.body[0].body[-1], BreakNode)
    assert isinstance(case_one.body[-1], BreakNode)
    assert default_case.value is None
    assert isinstance(default_case.body[-1], BreakNode)


def test_loop_inside_switch_case_keeps_following_case_statement():
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
    switch = ast.functions[0].body[2]
    case_zero = switch.cases[0]

    assert isinstance(switch, SwitchNode)
    assert isinstance(case_zero.body[0], ForNode)
    assert isinstance(case_zero.body[0].body[-1], BreakNode)
    assert isinstance(case_zero.body[1], AssignmentNode)
    assert isinstance(case_zero.body[2], BreakNode)
    assert isinstance(ast.functions[0].body[3], AssignmentNode)


def test_break_at_function_scope_is_rejected():
    code = """
    void main() {
        break;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="break used outside loop or switch"):
        parse_code(tokens)


def test_continue_at_function_scope_is_rejected():
    code = """
    void main() {
        continue;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="continue used outside loop"):
        parse_code(tokens)


def test_continue_inside_switch_without_loop_is_rejected():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            case 0:
                continue;
            default:
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="continue used outside loop"):
        parse_code(tokens)


def test_loop_depth_does_not_leak_after_loop_inside_switch_case():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            case 0:
                for (int i = 0; i < 2; i++) {
                    continue;
                }
                continue;
            default:
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="continue used outside loop"):
        parse_code(tokens)


def test_break_inside_switch_without_loop_is_allowed():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            case 0:
                break;
            default:
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[1]

    assert isinstance(switch, SwitchNode)
    assert isinstance(switch.cases[0].body[-1], BreakNode)
    assert isinstance(switch.cases[1].body[-1], BreakNode)


def test_switch_rejects_statement_before_first_case():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            value = 4;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected CASE or DEFAULT"):
        parse_code(tokens)


def test_switch_rejects_unterminated_case():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            case 0:
                value = 4;
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Unterminated switch"):
        parse_code(tokens)


def test_layout_array_member_parsing():
    code = """
    layout(set = 0, binding = 0) uniform Bones {
        mat4 transforms[64];
        vec4 weights[4];
    } bones;
    void main() {
        gl_Position = transforms[0] * vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        assignment = ast.functions[0].body[0]

        assert ast.global_variables[0].struct_fields == [
            ("mat4", "transforms[64]"),
            ("vec4", "weights[4]"),
        ]
        assert isinstance(assignment.right.left, ArrayAccessNode)
        assert assignment.right.left.array.name == "transforms"
        assert assignment.right.left.index == "0"
    except SyntaxError:
        pytest.fail("Layout array member parsing not implemented")


def test_layout_custom_struct_member_type_parsing():
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
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.layout_type == "UNIFORM"
    assert layout.block_name == "Scene"
    assert layout.variable_name == "scene"
    assert layout.struct_fields == [("Light", "lights[4]"), ("mat4", "view")]


def test_custom_struct_uniform_type_parsing():
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
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.block_name is None
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "Light"
    assert layout.variable_name == "activeLight"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "Light"
    assert uniform.name == "fallbackLight"


def test_layout_resource_uniform_parsing():
    code = """
    layout(set = 0, binding = 1) uniform sampler2D albedoTex;
    void main() {
        vec4 color = texture(albedoTex, vec2(0.5, 0.5));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("set", "0"), ("binding", "1")]
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"
    assert layout.struct_fields == []


def test_layout_push_constant_block_parsing():
    code = """
    layout(push_constant) uniform PushConstants {
        mat4 model;
        vec4 tint;
    } pc;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.push_constant is True
    assert layout.qualifiers == []
    assert layout.layout_type == "UNIFORM"
    assert layout.block_name == "PushConstants"
    assert layout.variable_name == "pc"
    assert layout.struct_fields == [("mat4", "model"), ("vec4", "tint")]


def test_version_and_extension_directives_before_layout_parse():
    code = """
    #version 450
    #extension GL_EXT_nonuniform_qualifier : enable
    layout(set = 0, binding = 0) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert len(ast.global_variables) == 1
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "main"
    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("set", "0"), ("binding", "0")]
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"


def test_precision_declarations_before_layout_parse():
    code = """
    precision highp float;
    precision mediump sampler2D;
    layout(set = 0, binding = 0) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert len(ast.global_variables) == 1
    assert len(ast.functions) == 1
    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("set", "0"), ("binding", "0")]
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"


def test_layout_precision_qualifier_after_in_parsing():
    code = """
    #version 310 es
    precision highp float;
    layout(location = 0) in highp vec2 vUV;
    layout(location = 0) out mediump vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    input_layout = ast.global_variables[0]
    output_layout = ast.global_variables[1]

    assert isinstance(input_layout, LayoutNode)
    assert input_layout.declaration_qualifiers == ["highp"]
    assert input_layout.layout_type == "IN"
    assert input_layout.data_type == "vec2"
    assert input_layout.variable_name == "vUV"
    assert output_layout.declaration_qualifiers == ["mediump"]
    assert output_layout.layout_type == "OUT"
    assert output_layout.data_type == "vec4"
    assert output_layout.variable_name == "fragColor"


def test_layout_interpolation_qualifier_parsing():
    code = """
    layout(location = 0) flat in int faceID;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.declaration_qualifiers == ["flat"]
    assert layout.layout_type == "IN"
    assert layout.data_type == "int"
    assert layout.variable_name == "faceID"


def test_layout_component_and_index_qualifier_parsing():
    code = """
    layout(location = 1, component = 2) noperspective in vec4 color;
    layout(location = 0, index = 1) out vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    color = ast.global_variables[0]
    frag_color = ast.global_variables[1]

    assert isinstance(color, LayoutNode)
    assert color.qualifiers == [("location", "1"), ("component", "2")]
    assert color.declaration_qualifiers == ["noperspective"]
    assert color.layout_type == "IN"
    assert color.data_type == "vec4"
    assert color.variable_name == "color"
    assert isinstance(frag_color, LayoutNode)
    assert frag_color.qualifiers == [("location", "0"), ("index", "1")]
    assert frag_color.layout_type == "OUT"
    assert frag_color.data_type == "vec4"
    assert frag_color.variable_name == "fragColor"


def test_layout_readonly_buffer_qualifier_parsing():
    code = """
    layout(set = 0, binding = 0) readonly buffer Particles {
        vec4 pos[];
    } particles;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.declaration_qualifiers == ["readonly"]
    assert layout.layout_type == "BUFFER"
    assert layout.block_name == "Particles"
    assert layout.variable_name == "particles"
    assert layout.struct_fields == [("vec4", "pos[]")]


def test_layout_storage_image_format_and_access_qualifier_parsing():
    code = """
    layout(set = 0, binding = 0, r32ui) coherent readonly uniform uimage2D counters;
    layout(set = 0, binding = 1, rgba32f) writeonly uniform image2D outImage;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    counters = ast.global_variables[0]
    out_image = ast.global_variables[1]

    assert isinstance(counters, LayoutNode)
    assert counters.qualifiers == [("set", "0"), ("binding", "0"), ("r32ui", None)]
    assert counters.declaration_qualifiers == ["coherent", "readonly"]
    assert counters.layout_type == "UNIFORM"
    assert counters.data_type == "uimage2D"
    assert counters.variable_name == "counters"
    assert isinstance(out_image, LayoutNode)
    assert out_image.qualifiers == [
        ("set", "0"),
        ("binding", "1"),
        ("rgba32f", None),
    ]
    assert out_image.declaration_qualifiers == ["writeonly"]
    assert out_image.layout_type == "UNIFORM"
    assert out_image.data_type == "image2D"
    assert out_image.variable_name == "outImage"


def test_layout_identifier_qualifier_values_parse():
    code = """
    layout(set = MATERIAL_SET, binding = ALBEDO_BINDING) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [
        ("set", "MATERIAL_SET"),
        ("binding", "ALBEDO_BINDING"),
    ]
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"


def test_specialization_constant_layout_parsing():
    code = """
    layout(constant_id = 0) const uint a = 1;
    layout(constant_id = 1) const float scale = 3.0;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    first_layout = ast.global_variables[0]
    second_layout = ast.global_variables[1]

    assert isinstance(first_layout, LayoutNode)
    assert first_layout.qualifiers == [("constant_id", "0")]
    assert first_layout.layout_type == "CONST"
    assert isinstance(first_layout.declaration, AssignmentNode)
    assert isinstance(first_layout.declaration.left, VariableNode)
    assert first_layout.declaration.left.vtype == "const uint"
    assert first_layout.declaration.left.name == "a"
    assert first_layout.declaration.right == "1"

    assert isinstance(second_layout, LayoutNode)
    assert second_layout.qualifiers == [("constant_id", "1")]
    assert isinstance(second_layout.declaration, AssignmentNode)
    assert isinstance(second_layout.declaration.left, VariableNode)
    assert second_layout.declaration.left.vtype == "const float"
    assert second_layout.declaration.left.name == "scale"
    assert second_layout.declaration.right == "3.0"


def test_compute_local_size_layout_parsing():
    code = """
    layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [
        ("local_size_x", "8"),
        ("local_size_y", "4"),
        ("local_size_z", "1"),
    ]
    assert layout.layout_type == "IN"
    assert layout.data_type is None
    assert layout.variable_name is None


def test_fragment_early_tests_layout_parsing():
    code = """
    layout(early_fragment_tests) in;
    void main() {
        gl_FragColor = vec4(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("early_fragment_tests", None)]
    assert layout.layout_type == "IN"
    assert layout.data_type is None
    assert layout.variable_name is None


def test_standalone_postfix_update_parsing():
    code = """
    void main() {
        int i = 0;
        do {
            i++;
            i--;
        } while (i < 3);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[1]

    assert isinstance(loop, DoWhileNode)
    assert isinstance(loop.body[0], UnaryOpNode)
    assert loop.body[0].op == "POST_INCREMENT"
    assert isinstance(loop.body[1], UnaryOpNode)
    assert loop.body[1].op == "POST_DECREMENT"


def test_postfix_update_rejects_trailing_assignment():
    code = """
    void main() {
        int i = 0;
        i++ = 1;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected SEMICOLON, got EQUALS"):
        parse_code(tokens)


def test_plain_resource_uniform_parsing():
    code = """
    uniform sampler2D albedoTex;
    void main() {
        vec4 color = texture(albedoTex, vec2(0.5, 0.5));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    uniform = ast.global_variables[0]

    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "sampler2D"
    assert uniform.name == "albedoTex"


def test_one_dimensional_sampler_uniform_parsing():
    code = """
    layout(set = 0, binding = 0) uniform sampler1D ramp;
    uniform sampler1DArray ramps;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "sampler1D"
    assert layout.variable_name == "ramp"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "sampler1DArray"
    assert uniform.name == "ramps"


def test_atomic_uint_uniform_parsing():
    code = """
    layout(set = 0, binding = 0) uniform atomic_uint counter;
    uniform atomic_uint fallbackCounter;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "atomic_uint"
    assert layout.variable_name == "counter"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "atomic_uint"
    assert uniform.name == "fallbackCounter"


def test_standalone_sampler_uniform_parsing():
    code = """
    layout(set = 0, binding = 0) uniform sampler compareSampler;
    uniform sampler samplers[4];
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "sampler"
    assert layout.variable_name == "compareSampler"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "sampler"
    assert uniform.name == "samplers[4]"


def test_standalone_function_call_statement_consumes_semicolon():
    code = """
    void helper() {
    }

    void main() {
        helper();
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    call = ast.functions[1].body[0]
    following_statement = ast.functions[1].body[1]

    assert isinstance(call, FunctionCallNode)
    assert call.name == "helper"
    assert isinstance(following_statement, AssignmentNode)


def test_expression_statement_parses_unary_prefix_forms():
    code = """
    void main() {
        int value = 1;
        bool enabled;
        +value;
        !enabled;
        ++value;
        --value;
        (1 + 2);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    positive = ast.functions[0].body[2]
    logical_not = ast.functions[0].body[3]
    increment = ast.functions[0].body[4]
    decrement = ast.functions[0].body[5]
    parenthesized = ast.functions[0].body[6]

    assert isinstance(positive, UnaryOpNode)
    assert positive.op == "+"
    assert isinstance(logical_not, UnaryOpNode)
    assert logical_not.op == "!"
    assert isinstance(increment, UnaryOpNode)
    assert increment.op == "PRE_INCREMENT"
    assert isinstance(decrement, UnaryOpNode)
    assert decrement.op == "PRE_DECREMENT"
    assert parenthesized.op == "+"


def test_member_call_statement_is_not_parsed_as_variable():
    code = """
    void main() {
        image.store(1, value);
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    call = ast.functions[0].body[0]
    following_statement = ast.functions[0].body[1]

    assert isinstance(call, MethodCallNode)
    assert call.object.name == "image"
    assert call.method == "store"
    assert call.args[0] == "1"
    assert isinstance(call.args[1], VariableNode)
    assert call.args[1].name == "value"
    assert isinstance(following_statement, AssignmentNode)


def test_member_call_expression_and_array_receiver_parse_structurally():
    code = """
    void main() {
        int value = object.method();
        objects[0].store(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = ast.functions[0].body[0]
    call = ast.functions[0].body[1]

    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.right, MethodCallNode)
    assert assignment.right.object.name == "object"
    assert assignment.right.method == "method"
    assert isinstance(call, MethodCallNode)
    assert isinstance(call.object, ArrayAccessNode)
    assert call.object.array.name == "objects"
    assert call.object.index == "0"
    assert call.method == "store"


def test_member_access_assignment_lhs_is_structured():
    code = """
    void main() {
        color.r = 1.0;
        color.g += 0.5;
        objects[0].field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    scalar_assignment = ast.functions[0].body[0]
    compound_assignment = ast.functions[0].body[1]
    array_member_assignment = ast.functions[0].body[2]

    assert isinstance(scalar_assignment, AssignmentNode)
    assert isinstance(scalar_assignment.left, MemberAccessNode)
    assert scalar_assignment.left.object.name == "color"
    assert scalar_assignment.left.member == "r"

    assert isinstance(compound_assignment, AssignmentNode)
    assert isinstance(compound_assignment.left, MemberAccessNode)
    assert compound_assignment.left.object.name == "color"
    assert compound_assignment.left.member == "g"
    assert compound_assignment.operator == "+="

    assert isinstance(array_member_assignment, AssignmentNode)
    assert isinstance(array_member_assignment.left, MemberAccessNode)
    assert isinstance(array_member_assignment.left.object, ArrayAccessNode)
    assert array_member_assignment.left.object.array.name == "objects"
    assert array_member_assignment.left.object.index == "0"
    assert array_member_assignment.left.member == "field"


def test_assignment_expression_parsing_is_right_associative():
    code = """
    void main() {
        a = b = c;
        a += b = c;
        int value = a = b;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    simple = ast.functions[0].body[0]
    compound = ast.functions[0].body[1]
    declaration = ast.functions[0].body[2]

    assert isinstance(simple, AssignmentNode)
    assert simple.left.name == "a"
    assert isinstance(simple.right, AssignmentNode)
    assert simple.right.left.name == "b"
    assert simple.right.right.name == "c"

    assert isinstance(compound, AssignmentNode)
    assert compound.operator == "+="
    assert isinstance(compound.right, AssignmentNode)
    assert compound.right.left.name == "b"
    assert compound.right.right.name == "c"

    assert isinstance(declaration, AssignmentNode)
    assert declaration.left.vtype == "int"
    assert isinstance(declaration.right, AssignmentNode)
    assert declaration.right.left.name == "a"
    assert declaration.right.right.name == "b"


def test_typed_local_array_declaration_preserves_suffix():
    code = """
    void main() {
        float weights[4];
        weights[0] = 1.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.functions[0].body[0]
    assignment = ast.functions[0].body[1]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "float"
    assert declaration.name == "weights[4]"
    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, ArrayAccessNode)


def test_custom_type_local_declaration_preserves_type_and_name():
    code = """
    struct VertexOutput {
        vec4 position;
    };

    void main() {
        VertexOutput output;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.functions[0].body[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "VertexOutput"
    assert declaration.name == "output"


def test_const_local_declaration_parsing():
    code = """
    void main() {
        const float scale = 1.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.functions[0].body[0]

    assert isinstance(declaration, AssignmentNode)
    assert isinstance(declaration.left, VariableNode)
    assert declaration.left.vtype == "const float"
    assert declaration.left.name == "scale"
    assert declaration.right == "1.0"


def test_const_global_declaration_parsing():
    code = """
    const int MAX_LIGHTS = 4;
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.global_variables[0]

    assert isinstance(declaration, AssignmentNode)
    assert isinstance(declaration.left, VariableNode)
    assert declaration.left.vtype == "const int"
    assert declaration.left.name == "MAX_LIGHTS"
    assert declaration.right == "4"


def test_bitwise_and_shift_precedence_parsing():
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
    assignment = ast.functions[0].body[3]
    expression = assignment.right

    assert isinstance(expression, BinaryOpNode)
    assert expression.op == "&"
    assert isinstance(expression.right, BinaryOpNode)
    assert expression.right.op == "<<"


def test_bitwise_or_xor_and_precedence_parsing():
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
    or_and = ast.functions[0].body[3].right
    xor_and = ast.functions[0].body[4].right
    or_xor = ast.functions[0].body[5].right
    and_equality = ast.functions[0].body[6].right
    or_relational = ast.functions[0].body[7].right

    assert isinstance(or_and, BinaryOpNode)
    assert or_and.op == "|"
    assert isinstance(or_and.right, BinaryOpNode)
    assert or_and.right.op == "&"

    assert isinstance(xor_and, BinaryOpNode)
    assert xor_and.op == "^"
    assert isinstance(xor_and.right, BinaryOpNode)
    assert xor_and.right.op == "&"

    assert isinstance(or_xor, BinaryOpNode)
    assert or_xor.op == "|"
    assert isinstance(or_xor.right, BinaryOpNode)
    assert or_xor.right.op == "^"

    assert isinstance(and_equality, BinaryOpNode)
    assert and_equality.op == "&"
    assert isinstance(and_equality.right, BinaryOpNode)
    assert and_equality.right.op == "=="

    assert isinstance(or_relational, BinaryOpNode)
    assert or_relational.op == "|"
    assert isinstance(or_relational.right, BinaryOpNode)
    assert or_relational.right.op == "<"


def test_logical_and_keeps_equality_operands_grouped():
    code = """
    void main() {
        bool selected = result1 == 8u && result2 == 2u;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = ast.functions[0].body[0]
    expression = assignment.right

    assert isinstance(expression, BinaryOpNode)
    assert expression.op == "&&"
    assert isinstance(expression.left, BinaryOpNode)
    assert expression.left.op == "=="
    assert isinstance(expression.right, BinaryOpNode)
    assert expression.right.op == "=="


def test_hex_integer_literals_parse_as_single_numeric_values():
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
    mask, upper, shifted, selected = ast.functions[0].body

    assert mask.right == "0xFF"
    assert upper.right == "0X10"
    assert isinstance(shifted.right, BinaryOpNode)
    assert shifted.right.op == "<<"
    assert shifted.right.left == "0x10"
    assert shifted.right.right == "2"
    assert isinstance(selected.right, BinaryOpNode)
    assert selected.right.op == "=="
    assert selected.right.right == "0x1"


def test_half_float_literal_suffixes_parse_as_single_numeric_values():
    code = """
    void main() {
        f16vec4 color = f16vec4(0.95hf, 1.0HF, 2hf);
    }
    """

    ast = parse_code(tokenize_code(code))
    assignment = ast.functions[0].body[0]
    constructor = assignment.right

    assert isinstance(constructor, FunctionCallNode)
    assert constructor.args == ["0.95", "1.0", "2"]


def test_precision_qualified_local_declarations_parse():
    code = """
    void main() {
        highp vec4 world_w = vec4(1.0);
        mediump vec3 pos = world_w.xyz;
    }
    """

    ast = parse_code(tokenize_code(code))
    world_w, pos = ast.functions[0].body

    assert isinstance(world_w, AssignmentNode)
    assert world_w.left.vtype == "highp vec4"
    assert world_w.left.name == "world_w"
    assert isinstance(pos, AssignmentNode)
    assert pos.left.vtype == "mediump vec3"
    assert pos.left.name == "pos"


def test_layout_qualifier_constant_expressions_parse():
    code = """
    layout(local_size_x = numMeshInvocationsX, local_size_y = numMeshInvocationsY, local_size_z = 1) in;
    layout(triangles, max_vertices = 4 * numMeshInvocationsX * numMeshInvocationsY, max_primitives = 2 * numMeshInvocationsX * numMeshInvocationsY) out;
    """

    ast = parse_code(tokenize_code(code))
    local_size, mesh_output = ast.global_variables

    assert isinstance(local_size, LayoutNode)
    assert local_size.qualifiers == [
        ("local_size_x", "numMeshInvocationsX"),
        ("local_size_y", "numMeshInvocationsY"),
        ("local_size_z", "1"),
    ]
    assert isinstance(mesh_output, LayoutNode)
    assert mesh_output.qualifiers == [
        ("triangles", None),
        (
            "max_vertices",
            "4 * numMeshInvocationsX * numMeshInvocationsY",
        ),
        (
            "max_primitives",
            "2 * numMeshInvocationsX * numMeshInvocationsY",
        ),
    ]


def test_mesh_shader_task_payload_storage_qualifier_declaration():
    code = """
    struct SharedData {
        vec2 position;
    };

    taskPayloadSharedEXT SharedData sharedData;
    """

    ast = parse_code(tokenize_code(code))
    declaration = ast.global_variables[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "taskPayloadSharedEXT SharedData"
    assert declaration.name == "sharedData"


def test_vulkan_lexer_from_file_resolves_same_directory_include(tmp_path):
    include_file = tmp_path / "lighting.h"
    include_file.write_text(
        """
        struct Light {
            vec4 color;
        };
        """,
        encoding="utf-8",
    )
    shader_file = tmp_path / "lighting.frag"
    shader_file.write_text(
        """
        #version 450
        #include "lighting.h"

        layout(set = 0, binding = 0) uniform Lights {
            Light activeLight;
        } lights;
        """,
        encoding="utf-8",
    )

    ast = parse_code(VulkanLexer.from_file(str(shader_file)).tokenize())
    layout = ast.global_variables[0]

    assert ast.structs[0].name == "Light"
    assert isinstance(layout, LayoutNode)
    assert layout.struct_fields == [("Light", "activeLight")]


def test_unknown_identifier_statement_is_rejected_instead_of_dropped():
    code = """
    void main() {
        image.store 1;
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Unexpected token after identifier image"):
        parse_code(tokens)


@pytest.mark.parametrize(
    "statement",
    [
        "int value = 1 trailing;",
        "value += 1 trailing;",
        "value & 1 trailing;",
    ],
)
def test_malformed_identifier_statement_trailing_tokens_are_rejected(statement):
    code = f"""
    void main() {{
        int value = 1;
        {statement}
        int next = 2;
    }}
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected SEMICOLON, got IDENTIFIER"):
        parse_code(tokens)


def test_expression_statement_requires_semicolon():
    code = """
    void main() {
        (1 + 2)
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
        parse_code(tokens)


if __name__ == "__main__":
    pytest.main()
