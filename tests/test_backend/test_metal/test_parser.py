from typing import List

import pytest

from crosstl.backend.common_ast import (
    DiscardNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    MethodCallNode,
    TextureSampleNode,
)
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
