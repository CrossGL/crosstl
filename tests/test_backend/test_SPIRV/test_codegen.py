from typing import List

import pytest

from crosstl.backend.SPIRV import VulkanCrossGLCodeGen
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser


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

    with pytest.raises(ValueError, match="Unsupported shader file type"):
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


def test_spirv_assembly_matrix_interface_codegen():
    tokens = tokenize_code(SPIRV_MATRIX_INTERFACE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4x4 model @input @location(0);" in generated_code
    assert "%model" not in generated_code
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


def test_spirv_assembly_uniform_constant_resources_codegen():
    tokens = tokenize_code(SPIRV_UNIFORM_CONSTANT_RESOURCE_ASSEMBLY)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Texture2D combinedTex @set(0) @binding(0);" in generated_code
    assert "sampler linearSampler @set(0) @binding(1);" in generated_code
    assert "%combined" not in generated_code
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


def test_translate_api_accepts_spirv_assembly_uniform_constant_resources(tmp_path):
    import crosstl

    shader_path = tmp_path / "resources.spvasm"
    shader_path.write_text(SPIRV_UNIFORM_CONSTANT_RESOURCE_ASSEMBLY, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="cgl", format_output=False
    )

    assert "Texture2D combinedTex @set(0) @binding(0);" in generated_code
    assert "sampler linearSampler @set(0) @binding(1);" in generated_code


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
    assert "Texture2D colorInput;" in generated_code
    assert "Texture2DMS msColorInput;" in generated_code
    assert "subpassInput colorInput;" not in generated_code
    assert "subpassInputMS msColorInput;" not in generated_code


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
        "static VIEW_PROJ: std::sync::LazyLock<float4x4> = "
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
        "static MODEL: std::sync::LazyLock<float4x4> = "
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
