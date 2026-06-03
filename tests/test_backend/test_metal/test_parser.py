from typing import List

import pytest

from crosstl.backend.common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    DiscardNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    MethodCallNode,
    RangeForNode,
    TextureSampleNode,
    VectorConstructorNode,
)
from crosstl.backend.Metal.MetalAst import BlockNode
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser


def tokenize_code(code: str) -> List:
    lexer = MetalLexer(code)
    return lexer.tokenize()


def parse_code(code: str):
    tokens = tokenize_code(code)
    parser = MetalParser(tokens)
    return parser.parse()


def parse_ok(code: str):
    ast = parse_code(code)
    assert ast is not None
    return ast


def parse_fails(code: str):
    with pytest.raises(SyntaxError):
        parse_code(code)


def iter_ast_nodes(node):
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_ast_nodes(value)
        return
    if isinstance(node, (list, tuple, set)):
        for value in node:
            yield from iter_ast_nodes(value)
        return
    yield node
    for value in getattr(node, "__dict__", {}).values():
        yield from iter_ast_nodes(value)


def test_parse_vertex_fragment_program():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexInput {
        float3 position [[attribute(0)]];
        float2 uv [[attribute(5)]];
    };

    struct VertexOutput {
        float4 position [[position]];
        float2 uv;
    };

    vertex VertexOutput vertex_main(VertexInput in [[stage_in]]) {
        VertexOutput out;
        out.position = float4(in.position, 1.0);
        out.uv = in.uv;
        return out;
    }

    fragment float4 fragment_main(VertexOutput in [[stage_in]]) {
        return float4(in.uv, 0.0, 1.0);
    }
    """
    parse_ok(code)


def test_parse_control_flow_constructs():
    code = """
    void main() {
        int i = 0;
        if (i < 1) {
            i++;
        } else if (i == 1) {
            i += 2;
        } else {
            i -= 1;
        }

        for (int j = 0; j < 4; j++) {
            if (j == 2) {
                continue;
            }
        }

        while (i < 10) {
            i++;
        }

        do {
            i--;
        } while (i > 0);

        switch (i) {
            case 0:
                i = 1;
                break;
            case 1:
                i = 2;
                break;
            default:
                i = 3;
                break;
        }

        return;
    }
    """
    parse_ok(code)


def test_parse_nested_unbraced_for_loops_from_public_msl_example():
    code = """
    fragment float4 shader_day53(float4 pixPos [[position]]) {
        const float PIXEL_SIZE = 40.0;
        float4 col = float4(0.0);
        for (int i = 0; i < int(PIXEL_SIZE); i++)
            for (int j = 0; j < int(PIXEL_SIZE); j++)
                col += float4(float(i + j));
        return col;
    }
    """
    ast = parse_ok(code)
    loops = [node for node in iter_ast_nodes(ast) if isinstance(node, ForNode)]

    assert len(loops) == 2
    assert isinstance(loops[0].body[0], ForNode)


def test_parse_resource_bindings_and_address_spaces():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Uniforms {
        float4x4 mvp;
    };

    kernel void compute_main(device float* data [[buffer(0)]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
        threadgroup float shared[16];
        float value = data[tid];
        data[tid] = (uniforms.mvp[0].x + value);
        shared[tid % 16] = value;
    }
    """
    parse_ok(code)


def test_parse_return_type_before_stage_qualifier_from_metal_cpp_sample():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct v2f
    {
        float4 position [[position]];
        half3 color;
    };

    v2f vertex vertexMain(uint vertexId [[vertex_id]],
                          device const float3* positions [[buffer(0)]],
                          device const float3* colors [[buffer(1)]])
    {
        v2f o;
        o.position = float4(positions[vertexId], 1.0);
        o.color = half3(colors[vertexId]);
        return o;
    }

    half4 fragment fragmentMain(v2f in [[stage_in]])
    {
        return half4(in.color, 1.0);
    }
    """
    ast = parse_ok(code)
    vertex, fragment = ast.functions

    assert vertex.return_type == "v2f"
    assert vertex.qualifier == "vertex"
    assert vertex.name == "vertexMain"
    assert fragment.return_type == "half4"
    assert fragment.qualifier == "fragment"
    assert fragment.name == "fragmentMain"


def test_parse_gnu_attribute_before_local_declaration():
    code = """
    void main(float b) {
        __attribute__((unused)) float zero = b;
    }
    """
    ast = parse_ok(code)

    assignment = ast.functions[0].body[0]
    variable = assignment.left
    assert variable.name == "zero"
    assert variable.vtype == "float"
    assert variable.attributes[0].name == "unused"
    assert variable.attributes[0].args == []


def test_parse_gnu_attribute_before_threadgroup_local_declaration():
    code = """
    void main() {
        __attribute__((unused)) threadgroup float scratch[16];
    }
    """
    ast = parse_ok(code)

    variable = ast.functions[0].body[0]
    assert variable.name == "scratch"
    assert variable.vtype == "float"
    assert variable.qualifiers == ["threadgroup"]
    assert variable.array_sizes == ["16"]
    assert variable.attributes[0].name == "unused"
    assert variable.attributes[0].args == []


def test_parse_gnu_attribute_before_global_constant_declaration():
    code = """
    __attribute__((unused)) constant int MathError_DivisionByZero = 0;
    """
    ast = parse_ok(code)

    constant = ast.global_variables[0]
    assert isinstance(constant, AssignmentNode)
    assert constant.left.name == "MathError_DivisionByZero"
    assert constant.left.vtype == "int"
    assert constant.left.qualifiers == ["constant"]
    assert constant.left.attributes[0].name == "unused"
    assert constant.left.attributes[0].args == []
    assert constant.right == "0"


def test_parse_gnu_attribute_before_global_constant_multiline_initializer():
    code = """
    __attribute__((unused)) constant float VALUES[2] = {
        1.0,
        2.0,
    };
    """
    ast = parse_ok(code)

    constant = ast.global_variables[0]
    assert isinstance(constant, AssignmentNode)
    assert constant.left.name == "VALUES"
    assert constant.left.qualifiers == ["constant"]
    assert constant.left.array_sizes == ["2"]
    assert constant.left.attributes[0].name == "unused"
    assert isinstance(constant.right, InitializerListNode)
    assert constant.right.elements == ["1.0", "2.0"]


def test_parse_gnu_attribute_after_function_return_type():
    code = """
    static inline float4 __attribute__((unused))
    helper(float4 value) {
        return value;
    }
    """
    ast = parse_ok(code)

    function = ast.functions[0]
    assert function.name == "helper"
    assert function.return_type == "float4"
    assert function.attributes[0].name == "unused"
    assert function.attributes[0].args == []


def test_parse_coherent_memory_qualifier_from_mlx_fence_kernel():
    code = """
    [[kernel]] void input_coherent(
        volatile coherent(system) device uint* input [[buffer(0)]],
        const constant uint& size [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      if (index < size) {
        input[index] = input[index];
      }
    }
    """
    ast = parse_ok(code)
    param = ast.functions[0].params[0]

    assert param.vtype == "uint*"
    assert param.name == "input"
    assert param.qualifiers == ["volatile", "coherent(system)", "device"]
    assert param.attributes[0].name == "buffer"
    assert param.attributes[0].args == ["0"]


def test_parse_scoped_atomic_thread_fence_call_from_mlx_kernel():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void fence() {
        metal::atomic_thread_fence(metal::mem_flags::mem_device);
    }
    """
    ast = parse_ok(code)

    call = ast.functions[0].body[0]
    assert isinstance(call, FunctionCallNode)
    assert call.name == "metal::atomic_thread_fence"
    assert call.args[0].name == "metal::mem_flags::mem_device"


def test_parse_arrays_and_indexing():
    code = """
    struct Data {
        float values[4];
    };

    void main() {
        Data d;
        float v = d.values[2];
        float arr[3];
        arr[1] = v;
    }
    """
    parse_ok(code)


def test_parse_imageblock_member_array_after_attribute_from_apple_sample():
    code = """
    struct TransparentFragmentValues {
        rgba8unorm<half4> colors [[raster_order_group(0)]] [kNumLayers];
        half depths [[raster_order_group(0)]] [kNumLayers];
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].vtype == "rgba8unorm<half4>"
    assert members[0].array_sizes[0].name == "kNumLayers"
    assert members[1].array_sizes[0].name == "kNumLayers"


def test_parse_typedef_struct_gnu_attribute_from_apple_deferred_sample():
    code = """
    typedef struct __attribute__ ((packed)) packed_float3 {
        float x;
        float y;
        float z;
    } packed_float3;
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "packed_float3"
    assert struct.typedef_tag == "packed_float3"
    assert struct.attributes[0].name == "packed"
    assert [(member.vtype, member.name) for member in struct.members] == [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
    ]


def test_parse_leading_global_scope_expression_from_apple_hdr_sample():
    code = """
    struct GaussSample {
        float2 offset;
        float weight;
    };

    constant GaussSample GaussKernelX[] = {
        {{-2.06278f, 0.f}, 0.05092f},
        {{ 0.53805f, 0.f}, 0.44908f}
    };
    constant size_t GAUSS_KERNEL_SIZE_X = sizeof(GaussKernelX) / sizeof(GaussKernelX[0]);

    half3 BlurredSampleX(float2 texCoords) {
        half3 finalColor = half3(0.f);

        for(uint32_t gaussSampleIdx = 0ul; gaussSampleIdx < ::GAUSS_KERNEL_SIZE_X; ++gaussSampleIdx) {
            constant GaussSample & gaussSample = ::GaussKernelX[gaussSampleIdx];
            finalColor += half3(gaussSample.weight);
        }

        return finalColor;
    }
    """
    ast = parse_ok(code)
    array_accesses = [
        node for node in iter_ast_nodes(ast) if isinstance(node, ArrayAccessNode)
    ]

    assert any(
        getattr(node.array, "name", None) == "GaussKernelX" for node in array_accesses
    )


def test_parse_fragment_output_color_and_depth_attributes():
    code = """
    struct FragmentOutput {
        float4 color0 [[color(0)]];
        float depth [[depth(any)]];
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].name == "color0"
    assert members[0].attributes[0].name == "color"
    assert members[0].attributes[0].args == ["0"]
    assert members[1].name == "depth"
    assert members[1].attributes[0].name == "depth"
    assert members[1].attributes[0].args == ["any"]


def test_parse_struct_member_default_initializers_from_metal4_basics_pbr():
    code = """
    typedef struct FragmentMaterial {
        float4 baseColor { 1.0f, 1.0f, 1.0f, 1.0f };
        float3 c_diff { 1.0f, 1.0f, 1.0f };
        float metalness = 1.0f;
        float perceptualRoughness = 1.0f;
    } FragmentMaterial;
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].name == "baseColor"
    assert isinstance(members[0].default_value, InitializerListNode)
    assert members[0].default_value.elements == ["1.0f", "1.0f", "1.0f", "1.0f"]
    assert members[1].name == "c_diff"
    assert isinstance(members[1].default_value, InitializerListNode)
    assert members[2].name == "metalness"
    assert members[2].default_value == "1.0f"
    assert members[3].name == "perceptualRoughness"
    assert members[3].default_value == "1.0f"


def test_parse_argument_buffer_array_of_device_pointers_from_apple_sample():
    code = """
    struct FragmentShaderArguments {
        array<texture2d<float>, AAPLNumTextureArguments> exampleTextures
            [[id(AAPLArgumentBufferIDExampleTextures)]];
        array<device float *, AAPLNumBufferArguments> exampleBuffers
            [[id(AAPLArgumentBufferIDExampleBuffers)]];
        array<uint32_t, AAPLNumBufferArguments> exampleConstants
            [[id(AAPLArgumentBufferIDExampleConstants)]];
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].vtype == "array<texture2d<float>,AAPLNumTextureArguments>"
    assert members[1].vtype == "array<device float*,AAPLNumBufferArguments>"
    assert members[1].attributes[0].name == "id"
    assert members[1].attributes[0].args == ["AAPLArgumentBufferIDExampleBuffers"]
    assert members[2].vtype == "array<uint32_t,AAPLNumBufferArguments>"


def test_parse_ternary_and_bitwise_expressions():
    code = """
    void main() {
        int a = 3;
        int b = 5;
        int c = (a > b) ? a : b;
        int d = (a & b) | (a ^ b);
        int e = ~a;
        int f = a << 1;
        int g = b >> 1;
    }
    """
    parse_ok(code)


def test_parse_argument_buffers_and_function_constants():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Args {
        float4x4 mvp;
    };

    constant Args& args [[buffer(0), id(3)]];

    constant int gMode [[function_constant(0)]];

    vertex float4 vertex_main(float3 pos [[attribute(0)]]) {
        if (gMode == 1) {
            return args.mvp * float4(pos, 1.0);
        }
        return float4(pos, 1.0);
    }
    """
    parse_ok(code)


def test_parse_argument_buffer_reference_array_parameter():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 my_fragment(
        constant texture2d<float> & texturesAB1 [[buffer(0)]],
        constant texture2d<float> & texturesAB2[10] [[buffer(1)]],
        array<texture2d<float>, 10> texturesArray [[texture(0)]]) {
        return float4(1.0);
    }
    """
    ast = parse_ok(code)
    params = ast.functions[0].params

    assert params[0].vtype == "texture2d<float>&"
    assert params[0].qualifiers == ["constant"]
    assert params[1].vtype == "texture2d<float>&"
    assert params[1].name == "texturesAB2"
    assert params[1].array_sizes == ["10"]
    assert params[1].attributes[0].name == "buffer"
    assert params[1].attributes[0].args == ["1"]
    assert params[2].vtype == "array<texture2d<float>,10>"
    assert params[2].name == "texturesArray"


def test_parse_shader_parameter_attribute_on_following_line_and_texture_arrays():
    code = """
    fragment float4 fragment_main(
        constant LightingModel &lighting_model
        [[buffer(3)]],
        array<texture2d<float>, MAX_SHADOW_CASCADES> shadow_maps [[texture(6)]],
        texture2d_array<float, access::read_write> irradiance_map [[texture(11)]]) {
        return float4(1.0);
    }
    """
    ast = parse_ok(code)
    params = ast.functions[0].params

    assert params[0].name == "lighting_model"
    assert params[0].vtype == "LightingModel&"
    assert params[0].qualifiers == ["constant"]
    assert params[0].attributes[0].name == "buffer"
    assert params[0].attributes[0].args == ["3"]
    assert params[1].name == "shadow_maps"
    assert params[1].vtype == "array<texture2d<float>,MAX_SHADOW_CASCADES>"
    assert params[1].attributes[0].name == "texture"
    assert params[1].attributes[0].args == ["6"]
    assert params[2].name == "irradiance_map"
    assert params[2].vtype == "texture2d_array<float,access::read_write>"
    assert params[2].attributes[0].name == "texture"
    assert params[2].attributes[0].args == ["11"]


def test_parse_fragment_entry_name_on_following_line():
    code = """
    fragment FragmentOutput
    fragment_main(FragmentInput input [[stage_in]]) {
        return FragmentOutput();
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.qualifier == "fragment"
    assert function.return_type == "FragmentOutput"
    assert function.name == "fragment_main"
    assert function.params[0].attributes[0].name == "stage_in"


def test_parse_defaulted_function_constant_preserves_attribute():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constant bool useFastPath [[function_constant(3)]] = true;

    fragment float4 fragment_main() {
        if (useFastPath) {
            return float4(1.0);
        }
        return float4(0.0);
    }
    """
    ast = parse_ok(code)
    constant = ast.global_variables[0]
    assert isinstance(constant, AssignmentNode)
    assert constant.left.name == "useFastPath"
    assert constant.left.vtype == "bool"
    assert "constant" in constant.left.qualifiers
    assert constant.left.attributes[0].name == "function_constant"
    assert constant.left.attributes[0].args == ["3"]
    assert constant.right == "true"


def test_parse_fragment_early_tests_attribute():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    [[early_fragment_tests]]
    fragment float4 fragment_main() {
        return float4(1.0);
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.qualifier == "fragment"
    assert function.attributes[0].name == "early_fragment_tests"
    assert function.attributes[0].args == []


def test_parse_access_qualified_textures_and_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(texture2d<float, access::read_write> tex [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        float4 c = tex.sample(samp, float2(0.5, 0.5));
        float4 r = tex.read(uint2(0, 0));
        tex.write(c, uint2(0, 0));
        float s = tex.sample_compare(samp, float2(0.5, 0.5), 0.5);
        float4 g = tex.gather(samp, float2(0.5, 0.5));
        return c + r + g + float4(s);
    }
    """
    parse_ok(code)


def test_parse_texture_method_ast_shapes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(texture2d<float, access::read_write> tex [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        float4 base = tex.sample(
            samp,
            float2(0.5, 0.5)
        );
        float4 mip = tex.sample(
            samp,
            float2(0.5, 0.5),
            1.0
        );
        float4 readValue = tex.read(uint2(0, 0));
        tex.write(
            base,
            uint2(0, 0)
        );
        float compareValue = tex.sample_compare_level(
            samp, float2(0.5, 0.5), 0.5, 0.0
        );
        float4 gathered = tex.gather_compare(
            samp,
            float2(0.5, 0.5),
            0.5
        );
        return base + mip + readValue + gathered + float4(compareValue);
    }
    """
    ast = parse_ok(code)
    nodes = list(iter_ast_nodes(ast))

    samples = [node for node in nodes if isinstance(node, TextureSampleNode)]
    assert len(samples) == 2
    assert samples[0].lod is None
    assert samples[1].lod is not None

    methods = [node.method for node in nodes if isinstance(node, MethodCallNode)]
    assert {"read", "write", "sample_compare_level", "gather_compare"}.issubset(
        set(methods)
    )
    assert "sample" not in methods


def test_parse_packed_and_simd_types():
    code = """
    struct PackedTypes {
        packed_float4 p4;
        packed_half2 ph2;
        packed_uint3 pu3;
        simd_float3 s3;
        simd_float4x4 s44;
    };
    """
    parse_ok(code)


def test_parse_ms_and_depth_textures():
    code = """
    fragment float4 fragment_main(texture2d_ms<float> msTex [[texture(0)]],
                                  depth2d_array<float> depthArr [[texture(1)]],
                                  texturecube_array<float> cubeArr [[texture(2)]],
                                  uint2 coord [[position]]) {
        float4 c = msTex.read(coord, 0);
        return c;
    }
    """
    parse_ok(code)


def test_parse_function_tables_and_acceleration_structures():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload {
        float3 color;
    };

    visible_function_table<void(Payload&)> vft [[buffer(0), id(1)]];
    intersection_function_table<void(Payload&)> ift [[buffer(1), id(2)]];
    acceleration_structure accel [[buffer(2)]];

    kernel void main(device float* outData [[buffer(3)]]) {
        // No-op
    }
    """
    parse_ok(code)


def test_parse_metal_namespace_types():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(metal::texture2d<float> tex [[texture(0)]],
                                  metal::sampler samp [[sampler(0)]]) {
        return tex.sample(samp, float2(0.5, 0.5));
    }
    """
    parse_ok(code)


def test_parse_scoped_return_type_prototype_from_msl_examples():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    metal::float2x2 rot(float radian);

    metal::float2x2 make_rot(float radian) {
        return metal::float2x2(
            cos(radian), -sin(radian),
            sin(radian), cos(radian)
        );
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["make_rot"]
    assert ast.functions[0].return_type == "metal::float2x2"


def test_parse_raytracing_qualifiers_and_types():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    intersection void isect(raytracing::ray r, intersector inter) {
        // no-op
    }

    anyhit void any_hit() { }
    closesthit void closest_hit() { }
    miss void miss_main() { }
    callable void callable_main() { }
    """
    parse_ok(code)


def test_parse_enum_and_typedef():
    code = """
    typedef int32_t MyInt;
    enum Mode { Off, On = 2, Auto };

    void main() {
        MyInt v = 1;
        Mode m = Auto;
    }
    """
    parse_ok(code)


def test_parse_scoped_enum_with_underlying_type_from_metal4_basics():
    code = """
    enum class BRDF : uint {
        Lambert,
        TrowbridgeReitz,
    };
    """
    ast = parse_ok(code)
    enum = ast.enums[0]

    assert enum.name == "BRDF"
    assert enum.is_scoped is True
    assert enum.underlying_type == "uint"
    assert enum.members == [("Lambert", None), ("TrowbridgeReitz", None)]


def test_parse_unscoped_enum_with_underlying_type_from_metal_splatter():
    code = """
    enum BufferIndex: int32_t
    {
        BufferIndexMeshPositions = 0,
        BufferIndexMeshGenerics  = 1,
        BufferIndexUniforms      = 2,
    };
    """
    ast = parse_ok(code)
    enum = ast.enums[0]

    assert enum.name == "BufferIndex"
    assert enum.is_scoped is False
    assert enum.underlying_type == "int32_t"
    assert enum.members[1] == ("BufferIndexMeshGenerics", "1")


def test_parse_anonymous_enum_from_metal_base_effect():
    code = """
    enum {
        VertexAttributePosition,
        VertexAttributeNormal,
        VertexAttributeColor,
        VertexAttributeTexCoord0,
    };
    """
    ast = parse_ok(code)
    enum = ast.enums[0]

    assert enum.name is None
    assert enum.underlying_type is None
    assert enum.members[0] == ("VertexAttributePosition", None)
    assert enum.members[-1] == ("VertexAttributeTexCoord0", None)


def test_parse_typedef_enum_with_tag_and_alias_from_satin_constants():
    code = """
    typedef enum ComputeBufferIndex {
        ComputeBufferUniforms = 0,
        ComputeBufferCustom0 = 1,
        ComputeBufferCustom1 = 2
    } ComputeBufferIndex;

    void main() {
        ComputeBufferIndex index = ComputeBufferCustom0;
    }
    """
    ast = parse_ok(code)

    enum = ast.enums[0]
    assert enum.name == "ComputeBufferIndex"
    assert enum.typedef_tag == "ComputeBufferIndex"
    assert enum.members[0][0] == "ComputeBufferUniforms"


def test_parse_top_level_material_chunk_statement_from_satin():
    code = """
    const float2 baseColorTexcoord =
        (uniforms.baseColorTexcoordTransform * float3(in.texcoord, 1.0)).xy;
    pixel.material.baseColor = baseColorMap.sample(baseColorSampler, baseColorTexcoord).rgb;
    pixel.material.baseColor *= uniforms.baseColor.rgb;
    """

    ast = parse_ok(code)

    assert len(ast.global_variables) == 1
    assert ast.global_variables[0].left.name == "baseColorTexcoord"


def test_parse_stage_keyword_names_in_helpers_and_parameters():
    code = """
    float intersection(float a, float b) { return max(a, b); }

    float fresnel(float3 eyeVector, float3 worldNormal, float amount = 3.0) {
        return pow(1.0 + dot(eyeVector, worldNormal), amount);
    }

    typedef struct {
        float time;
    } SpriteUniforms;

    vertex float4 spriteVertex(constant SpriteUniforms &compute [[buffer(1)]])
    {
        return float4(compute.time);
    }
    """
    ast = parse_ok(code)

    assert ast.functions[0].name == "intersection"
    assert ast.functions[1].params[2].name == "amount"
    assert ast.functions[1].params[2].default_value == "3.0"
    assert ast.functions[2].params[0].name == "compute"


def test_parse_object_data_reference_parameter_from_satin_mesh_shader():
    code = """
    struct Payload {
        uint indices[3];
    };

    [[object, max_total_threads_per_threadgroup(1)]]
    void customObject(
        object_data Payload &payload [[payload]],
        mesh_grid_properties mgp
    ) {
        payload.indices[0] = 0;
    }
    """
    ast = parse_ok(code)
    param = ast.functions[0].params[0]

    assert param.vtype == "Payload&"
    assert param.name == "payload"
    assert param.qualifiers == ["object_data"]
    assert param.attributes[0].name == "payload"


def test_parse_top_level_texture_parameter_fragment_from_satin_pbr_chunk():
    code = """
    texture2d<float> baseColorMap [[texture(PBRTextureBaseColor)]],
        texture2d<float> subsurfaceMap [[texture(PBRTextureSubsurface)]],
        texturecube<float> reflectionMap [[texture(PBRTextureReflection)]],
    """
    ast = parse_ok(code)

    assert ast.global_variables == []


def test_parse_sizeof_and_cast():
    code = """
    void main() {
        int a = sizeof(int);
        int b = alignof(float4);
        float3 v = (float3)(1.0);
    }
    """
    parse_ok(code)


def test_parse_pointer_member_access():
    code = """
    struct Uniforms {
        float4x4 mvp;
    };

    void main(constant Uniforms* uniforms) {
        float4 position = uniforms->mvp * float4(1.0);
    }
    """
    ast = parse_ok(code)
    member_accesses = [
        node for node in iter_ast_nodes(ast) if isinstance(node, MemberAccessNode)
    ]

    assert any(node.member == "mvp" and node.is_pointer for node in member_accesses)


def test_parse_single_statement_if_with_discard_fragment():
    code = """
    fragment half4 fragment_main(float4 color) {
        if (color.a < 0.5)
            discard_fragment();

        return half4(color);
    }
    """
    ast = parse_ok(code)
    if_nodes = [node for node in iter_ast_nodes(ast) if isinstance(node, IfNode)]

    assert len(if_nodes) == 1
    assert isinstance(if_nodes[0].if_body[0], DiscardNode)


def test_parse_alignas_and_static_assert():
    code = """
    alignas(16) float4 alignedValue;
    static_assert(1 == 1, "ok");

    void main() {
        alignas(float4) int v = 0;
        static_assert(sizeof(int) == 4);
    }
    """
    parse_ok(code)


def test_parse_using_alias():
    code = """
    using Index = uint;
    void main() {
        Index i = 0;
    }
    """
    parse_ok(code)


def test_parse_gpuimage_typedef_struct_and_parenthesized_identifier_expression():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    typedef struct {
        float reductionFactor;
    } LuminanceUniform;

    fragment half4 luminanceRangeReduction(
        texture2d<half> inputTexture [[texture(0)]],
        constant LuminanceUniform& uniform [[buffer(1)]]
    ) {
        half4 color;
        half luminanceRatio;
        return half4(half3((color.rgb) + (luminanceRatio)), color.w);
    }
    """
    parse_ok(code)


def test_parse_stage_attributes_and_trailing_const_pointer_qualifiers():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ShaderVertex {
        float4 position [[position]];
        float4 color;
    };

    [[vertex]] ShaderVertex main_vertex(
        device ShaderVertex const* const vertices [[buffer(0)]],
        uint vid [[vertex_id]]
    ) {
        return vertices[vid];
    }
    """
    ast = parse_ok(code)

    assert ast.functions[0].qualifier == "vertex"


def test_parse_keyword_named_sampler_and_buffer_from_apple_samples():
    code = """
    fragment half4 texturedQuadFragment(texture2d<half> texture [[texture(0)]],
                                        device float* values [[buffer(0)]],
                                        uint index) {
        constexpr sampler sampler(min_filter::linear, mag_filter::linear);
        device float* buffer = values;
        half4 color = texture.sample(sampler, float2(buffer[index]));
        return color;
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    sampler_decl = body[0]
    buffer_decl = body[1]
    assert sampler_decl.left.name == "sampler"
    assert sampler_decl.left.vtype == "sampler"
    assert buffer_decl.left.name == "buffer"
    assert buffer_decl.left.vtype == "float*"


def test_parse_function_prototypes_macro_expanded_empty_statements_and_comma_update():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    half lum(half3 c);

    half lum(half3 c) {
        return dot(c, half3(0.3, 0.59, 0.11));
    }

    kernel void reduce(device uint* values [[buffer(0)]], uint histSize) {
        uint residual = 4;
        uint residualStep = 2;
        for (uint i = 0; i < histSize && residual > 0; i += residualStep, residual--) {
            ; ;
            values[i]++;
        }
    }
    """
    parse_ok(code)


def test_parse_metalpetal_namespace_macro_qualifier_and_unsigned_int():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    namespace metalpetal {
        namespace yuv2rgbconvert {
            typedef struct {
                packed_float2 position;
            } Vertex;

            METAL_FUNC float helper(float value) {
                using namespace metalpetal::yuv2rgbconvert;
                return value;
            }

            vertex float4 colorConversionVertex(
                const device Vertex * vertices [[buffer(0)]],
                unsigned int vid [[vertex_id]]
            ) {
                return float4(float2(vertices[vid].position), helper(1.0), 1.0);
            }
        }
    }
    """
    parse_ok(code)


def test_parse_function_table_call_and_icb_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload { float3 color; };

    visible_function_table<void(Payload&)> vft [[buffer(0)]];
    indirect_command_buffer icb [[buffer(1)]];

    kernel void main(device float* outData [[buffer(2)]]) {
        Payload p;
        vft[0](p);
        icb.reset();
        icb.draw_primitives(3, 1, 0, 0);
    }
    """
    parse_ok(code)


def test_parse_icb_extended_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    indirect_command_buffer icb [[buffer(0)]];

    kernel void main() {
        icb.reset();
        icb.draw_primitives(3, 1, 0, 0);
        icb.draw_indexed_primitives(3, 1, 0, 0, 0);
        icb.draw_patches(3, 1, 0, 0);
        icb.compute_dispatch(uint3(1, 1, 1));
    }
    """
    parse_ok(code)


def test_parse_payload_and_hit_attributes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload { float3 color; };
    struct HitAttrib { float2 bary; };

    anyhit void any_hit(Payload& payload [[payload]],
                        HitAttrib attr [[hit_attribute]]) { }

    closesthit void closest_hit(Payload& payload [[payload]],
                                HitAttrib attr [[hit_attribute]]) { }
    """
    parse_ok(code)


def test_parse_mesh_object_io():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ObjPayload { float4 data; };

    object void object_main(threadgroup ObjPayload* payload [[payload]],
                            uint3 gid [[threadgroup_position_in_grid]]) { }

    mesh void mesh_main(threadgroup ObjPayload* payload [[payload]],
                        uint3 tid [[thread_position_in_threadgroup]]) { }
    """
    parse_ok(code)


def test_parse_mesh_output_functions():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    mesh void mesh_main() {
        SetMeshOutputCounts(64, 32);
        SetVertex(0, float3(0.0));
        SetPrimitive(0, 0);
    }
    """
    parse_ok(code)


def test_parse_atomic_operations():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void main() {
        atomic_int counter;
        int old = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
        old = atomic_exchange_explicit(counter, 0, memory_order_relaxed);
    }
    """
    parse_ok(code)


def test_parse_leading_decimal_float_literals_in_constexpr_arrays():
    code = """
    constexpr constant static float kvalues_mxfp4_f[4] = {0, .5f, 1.f, -.5f};
    """
    parse_ok(code)


def test_parse_as_type_template_call():
    code = """
    static inline float fp32_from_bits(uint32_t bits) {
        return as_type<float>(bits);
    }
    """
    ast = parse_ok(code)
    calls = [node for node in iter_ast_nodes(ast) if isinstance(node, FunctionCallNode)]

    assert any(node.name == "as_type<float>" for node in calls)


def test_parse_static_cast_from_apple_compute_sample():
    code = """
    kernel void process(uint2 gid [[thread_position_in_grid]]) {
        float2 p0 = static_cast<float2>(gid);
    }
    """
    ast = parse_ok(code)
    casts = [node for node in iter_ast_nodes(ast) if isinstance(node, CastNode)]

    assert len(casts) == 1
    assert casts[0].target_type == "float2"


def test_parse_template_dequantize_helper_from_llama_cpp():
    code = """
    template <typename type4x4>
    void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
        reg = (type4x4)(*src);
    }
    """
    ast = parse_ok(code)
    casts = [node for node in iter_ast_nodes(ast) if isinstance(node, CastNode)]

    assert [func.name for func in ast.functions] == ["dequantize_f32"]
    assert ast.functions[0].generics == ["type4x4"]
    assert ast.functions[0].params[0].vtype == "float4x4*"
    assert ast.functions[0].params[0].qualifiers == ["device", "const"]
    assert ast.functions[0].params[2].vtype == "type4x4&"
    assert ast.functions[0].params[2].qualifiers == ["thread"]
    assert any(node.target_type == "type4x4" for node in casts)


def test_parse_template_kernel_typename_without_space_from_llama_cpp():
    code = """
    template<typename T>
    kernel void kernel_memset(
            constant ggml_metal_kargs_memset & args,
            device T * dst,
            uint tpig [[thread_position_in_grid]]) {
        dst[tpig] = args.val;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["kernel_memset"]
    assert ast.functions[0].qualifier == "kernel"
    assert ast.functions[0].generics == ["T"]
    assert ast.functions[0].params[1].vtype == "T*"
    assert ast.functions[0].params[1].qualifiers == ["device"]


def test_parse_template_struct_from_mlx_arg_reduce():
    code = """
    template <typename U>
    struct IndexValPair {
        uint32_t index;
        U val;
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "IndexValPair"
    assert struct.generics == ["U"]
    assert struct.template_parameters == [("typename", "U")]
    assert [member.name for member in struct.members] == ["index", "val"]
    assert [member.vtype for member in struct.members] == ["uint32_t", "U"]


def test_skip_struct_constructor_with_initializer_list_and_trailing_semicolon_from_mlx():
    code = """
    struct complex64_t {
        float real;
        float imag;
        constexpr complex64_t(float real, float imag) : real(real), imag(imag) {};
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "complex64_t"
    assert [member.name for member in struct.members] == ["real", "imag"]


def test_parse_function_body_pragma_from_llama_cpp():
    code = """
    void quantize_q4_0(device const float* src, device block_q4_0& dst) {
        #pragma METAL fp math_mode(safe)
        float amax = 0.0f;
        dst.d = amax;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["quantize_q4_0"]
    assert len(ast.functions[0].body) == 2


def test_parse_comma_assignment_statement_from_llama_cpp():
    code = """
    void dequantize(device const float* values) {
        float dl = 0.0f;
        float ml = 0.0f;
        dl = values[0], ml = values[1];
    }
    """
    ast = parse_ok(code)
    statement = ast.functions[0].body[2]

    assert isinstance(statement, BinaryOpNode)
    assert statement.op == ","
    assert isinstance(statement.left, AssignmentNode)
    assert isinstance(statement.right, AssignmentNode)
    assert statement.left.left.name == "dl"
    assert statement.right.left.name == "ml"


def test_parse_braced_uchar_vector_constructor_from_llama_cpp():
    code = """
    static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
        return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                     : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                              uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
    }
    """
    ast = parse_ok(code)
    braced_constructors = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode)
        and node.name == "uchar2"
        and getattr(node, "is_braced_constructor", False)
    ]
    scalar_constructors = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, VectorConstructorNode) and node.type_name == "uchar"
    ]

    assert len(braced_constructors) == 2
    assert len(scalar_constructors) == 4


def test_parse_direct_list_declaration_initializers_from_book_of_shaders_metal():
    code = """
    constant float3 colorA { 0.000f, 0.129f, 0.647f };
    constant float3 colorB { 0.980f, 0.275f, 0.090f };

    fragment float4 fragment_main() {
        float3 color { colorA.x, colorB.y, 1.0f };
        float2 st { 0.0f, 1.0f }, offset { 1.0f, 0.0f };
        return float4(color + float3(st.x, offset.y, 0.0f), 1.0f);
    }
    """
    ast = parse_ok(code)

    assert [node.left.name for node in ast.global_variables] == ["colorA", "colorB"]
    assert all(
        isinstance(node.right, InitializerListNode) for node in ast.global_variables
    )
    assert [len(node.right.elements) for node in ast.global_variables] == [3, 3]

    body = ast.functions[0].body
    assert isinstance(body[0].right, InitializerListNode)
    assert [node.left.name for node in body[1:3]] == ["st", "offset"]
    assert all(isinstance(node.right, InitializerListNode) for node in body[1:3])


def test_parse_standalone_scoped_block_from_llama_cpp():
    code = """
    void FC_unary_op(device const float* src0, device float* dst, uint i0) {
        {
            if (i0 >= 4) {
                return;
            }

            const float x = src0[i0];
            dst[i0] = x;
        }
    }
    """
    ast = parse_ok(code)
    block = ast.functions[0].body[0]

    assert isinstance(block, BlockNode)
    assert len(block.statements) == 3
    assert isinstance(block.statements[0], IfNode)
    assert isinstance(block.statements[2], AssignmentNode)


def test_parse_decltype_template_typedef_and_explicit_instantiations_from_llama_cpp():
    code = """
    template [[host_name("kernel_unary_f32_f32")]]
    kernel void kernel_unary_impl(device const float* src, device float* dst) {
        dst[0] = erf_approx<float>(src[0]);
    }

    typedef decltype(kernel_unary_impl<float>) kernel_unary_t;
    template [[host_name("kernel_unary_f32_f32")]]
    kernel kernel_unary_t kernel_unary_impl<float>;
    """
    ast = parse_ok(code)

    assert ast.typedefs[0].name == "kernel_unary_t"
    assert ast.typedefs[0].alias_type == "decltype(kernel_unary_impl<float>)"
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "kernel_unary_impl"


def test_parse_pragma_and_type_trait_expression_from_llama_cpp():
    code = """
    void reduce(uint j, uint limit, device float* dst_row) {
        for (int i = 0; i < limit; i++) {
            _Pragma("clang loop unroll(full)")
            dst_row[i] = erf_approx<float>(dst_row[i]);
        }

        if (is_same<float4, T0>::value) {
            dst_row[0] = 0.0f;
        }
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert isinstance(body[0], ForNode)
    assert len(body[0].body) == 1
    assert isinstance(body[1], IfNode)


def test_parse_empty_for_condition_before_type_trait_from_llama_cpp():
    code = """
    void flash_attn_ext(uint iwg, uint sgitg) {
        for (int ic0 = iwg * NSG + sgitg; ; ic0 += NWG * NSG) {
            int ic = ic0;
            if (ic0 >= D) {
                break;
            }
        }

        if (is_same<float4, T0>::value) {
            return;
        }
    }
    """
    ast = parse_ok(code)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, ForNode)
    assert loop.condition is None
    assert isinstance(ast.functions[0].body[1], IfNode)


def test_parse_qualified_casts_and_range_designator_from_llama_cpp():
    code = """
    void load_block(device const void* src0, uint offset0, short r1ptg) {
        device const block_q1_0* block =
            (device const block_q1_0*)((device char*)src0 + offset0);
        float sumf[8] = {[0 ... r1ptg - 1] = 0.0f};
    }
    """
    ast = parse_ok(code)
    cast_assignment = ast.functions[0].body[0]
    range_assignment = ast.functions[0].body[1]
    initializer = range_assignment.right
    designator = initializer.elements[0]

    assert isinstance(cast_assignment.right, CastNode)
    assert cast_assignment.right.target_type == "block_q1_0*"
    assert designator.designators[0][0] == "range"
    assert designator.designators[0][1] == "0"
    assert designator.designators[0][2] == "r1ptg-1"


def test_parse_function_pointer_typedef_from_llama_cpp():
    code = """
    typedef void (im2col_t)(constant ggml_metal_kargs_im2col& args,
                            device const float* x,
                            device char* dst,
                            uint3 tgpig [[threadgroup_position_in_grid]]);
    """
    ast = parse_ok(code)

    assert ast.typedefs[0].name == "im2col_t"
    assert ast.typedefs[0].alias_type == "void"


def test_parse_struct_forward_declaration_from_mlx_complex_header():
    code = """
    struct complex64_t;

    void use_complex(thread complex64_t& value) {
        return;
    }
    """
    ast = parse_ok(code)

    assert ast.structs == []
    assert ast.functions[0].params[0].vtype == "complex64_t&"


def test_parse_multiline_macro_invocation_from_mlx_bf16_math_header():
    code = """
    #define instantiate_metal_math_funcs(itype, otype, ctype, mfast) \\
      METAL_FUNC otype abs(itype x) { \\
        return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast)); \\
      }

    namespace metal {
    instantiate_metal_math_funcs(
        bfloat16_t,
        bfloat16_t,
        float,
        __METAL_MAYBE_FAST_MATH__);
    }

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = 1.0f;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["real_kernel"]


def test_parse_metal_mesh_scoped_type_from_public_samples():
    code = """
    using TriangleMeshType = metal::mesh<VertexOut, PrimOut, 4, 2, topology::line>;

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = 1.0f;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["real_kernel"]


def test_parse_char_literal_initializer_from_public_metal_samples():
    code = """
    constant char PATTERNS[][2] = {
        { 'M', 'S' },
        { 'S', 'M' }
    };

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = 1.0f;
    }
    """
    ast = parse_ok(code)

    assert ast.global_variables[0].left.name == "PATTERNS"
    assert ast.global_variables[0].left.array_sizes == [None, "2"]
    assert [func.name for func in ast.functions] == ["real_kernel"]


def test_parse_member_template_disambiguator_from_mlx_arg_reduce():
    code = """
    void reduce(thread Reducer& op) {
        int best = 0;
        int vals[2] = {0, 1};
        best = op.template reduce_many<N_READS>(best, vals, 0);
    }
    """
    ast = parse_ok(code)

    calls = [node for node in iter_ast_nodes(ast) if isinstance(node, MethodCallNode)]
    assert calls[0].method == "reduce_many<N_READS>"


def test_parse_standalone_for_loop_macro_prefix_from_mlx_conv():
    code = """
    void load_tile(device float* out) {
        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < 4; ++cc) {
            out[cc] = float(cc);
        }
    }
    """
    ast = parse_ok(code)

    loops = [node for node in iter_ast_nodes(ast) if isinstance(node, ForNode)]
    assert len(loops) == 1


def test_parse_template_qualified_static_member_declaration_from_mlx_conv():
    code = """
    constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
    """
    ast = parse_ok(code)

    variable = ast.global_variables[0]
    assert variable.name == "WinogradTransforms<6,3,8>::wt_transform"
    assert len(variable.array_sizes) == 2


def test_parse_typename_qualified_threadgroup_type_from_mlx_gemv():
    code = """
    void load_tile() {
        threadgroup typename gemv_kernel::acc_type tgp_memory[4];
    }
    """
    ast = parse_ok(code)

    variable = ast.functions[0].body[0]
    assert variable.vtype == "gemv_kernel::acc_type"
    assert variable.name == "tgp_memory"
    assert variable.qualifiers == ["threadgroup"]


def test_parse_union_declaration_from_mlx_random():
    code = """
    union rbits {
        uint2 val;
        uchar4 bytes[2];
    };

    rbits make_bits() {
        rbits v;
        for (auto r : rotations[0]) {
            v.val.x += r;
        }
        return v;
    }
    """
    ast = parse_ok(code)

    union = ast.structs[0]
    assert union.name == "rbits"
    assert getattr(union, "aggregate_kind", None) == "union"
    assert [(member.vtype, member.name) for member in union.members] == [
        ("uint2", "val"),
        ("uchar4", "bytes"),
    ]
    assert union.members[1].array_sizes == ["2"]
    assert ast.functions[0].return_type == "rbits"
    assert ast.functions[0].body[0].vtype == "rbits"
    loop = ast.functions[0].body[1]
    assert isinstance(loop, RangeForNode)
    assert loop.vtype == "auto"
    assert loop.name == "r"


def test_parse_preprocessor_define():
    code = """
    #define FOO 1
    void main() {
        int x = FOO;
    }
    """
    parse_ok(code)


@pytest.mark.parametrize(
    "code",
    [
        "struct S { float x };",  # Missing semicolon in member declaration
        "struct S { float x;",  # Unclosed struct
        "float4 pos [[position];",  # Unterminated attribute brackets
        "vertex float4 main(float4 v [[stage_in]] { return v; }",  # Missing ')'
        "void main() { case 0: break; }",  # Case outside switch
        "void main() { switch (1) { case 0 return; } }",  # Missing ':'
    ],
)
def test_parse_invalid_syntax_cases(code):
    parse_fails(code)


if __name__ == "__main__":
    pytest.main()
