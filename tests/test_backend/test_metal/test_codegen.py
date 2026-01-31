import re
import pytest
from typing import List
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter


def tokenize_code(code: str) -> List:
    """Tokenize Metal code."""
    lexer = MetalLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Parse tokens into an AST."""
    parser = MetalParser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Generate CrossGL code from an AST."""
    codegen = MetalToCrossGLConverter()
    return codegen.generate(ast_node)


def convert(code: str) -> str:
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    return generate_code(ast)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_codegen_emits_shader_and_stages():
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

    struct FragmentOutput {
        float4 color [[color(0)]];
    };

    vertex VertexOutput vertex_main(VertexInput in [[stage_in]]) {
        VertexOutput out;
        out.position = float4(in.position, 1.0);
        out.uv = in.uv;
        return out;
    }

    fragment FragmentOutput fragment_main(VertexOutput in [[stage_in]]) {
        FragmentOutput out;
        out.color = float4(in.uv, 0.0, 1.0);
        return out;
    }
    """
    result = convert(code)
    assert result.strip()
    assert "shader" in result
    assert "vertex" in result
    assert "fragment" in result
    assert "VertexInput" in result
    assert "VertexOutput" in result
    assert "FragmentOutput" in result
    assert re.search(r"gl_Position", result)
    assert re.search(r"gl_FragColor", result)


def test_codegen_attribute_mapping():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexInput {
        float3 position [[attribute(0)]];
        float3 normal [[attribute(1)]];
        float2 uv [[attribute(5)]];
    };

    struct VertexOutput {
        float4 position [[position]];
        float2 uv;
    };

    struct FragmentOutput {
        float4 color [[color(0)]];
    };

    vertex VertexOutput vertex_main(VertexInput in [[stage_in]],
                                    uint vid [[vertex_id]]) {
        VertexOutput out;
        out.position = float4(in.position, 1.0);
        out.uv = in.uv + float2(vid, vid);
        return out;
    }

    fragment FragmentOutput fragment_main(VertexOutput in [[stage_in]]) {
        FragmentOutput out;
        out.color = float4(in.uv, 0.0, 1.0);
        return out;
    }
    """
    result = convert(code)
    assert re.search(r"@\s*POSITION", result)
    assert re.search(r"@\s*NORMAL", result)
    assert re.search(r"@\s*TEXCOORD0", result)
    assert re.search(r"gl_Position", result)
    assert re.search(r"gl_FragColor", result)
    assert re.search(r"gl_VertexID", result)


def test_codegen_type_mapping_vectors_and_matrices():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Types {
        float2 uv;
        float3 normal;
        float4 position;
        float4x4 transform;
    };

    vertex float4 vertex_main() {
        Types t;
        t.uv = float2(0.0, 1.0);
        t.normal = float3(0.0, 1.0, 0.0);
        t.position = float4(t.normal, 1.0);
        t.transform = float4x4(1.0);
        return t.position;
    }
    """
    result = convert(code)
    assert "vec2" in result
    assert "vec3" in result
    assert "vec4" in result
    assert "mat4" in result


def test_codegen_texture_and_sampler_translation():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexOut {
        float4 position [[position]];
        float2 uv;
    };

    fragment float4 fragment_main(VertexOut in [[stage_in]],
                                  texture2d<float> albedo [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        float4 color = albedo.sample(samp, in.uv);
        return color;
    }
    """
    result = convert(code)
    assert re.search(r"sampler2d", result, re.IGNORECASE)
    assert "texture(" in result or "texture2D(" in result
    assert "albedo" in result


def test_codegen_control_flow_and_ops():
    code = """
    void main() {
        int i = 0;
        int sum = 0;
        for (int j = 0; j < 4; j++) {
            sum += j;
        }

        while (i < 10) {
            i++;
        }

        do {
            i--;
        } while (i > 0);

        if (sum > 4) {
            sum = sum & 0x1;
        } else {
            sum = sum | 0x2;
        }

        switch (sum) {
            case 0:
                sum = sum ^ 0x3;
                break;
            default:
                sum = ~sum;
                break;
        }

        if (sum == 0) {
            return;
        }

        sum <<= 1;
        sum >>= 1;
    }
    """
    result = convert(code)
    compact = normalize(result)
    assert "if" in compact
    assert "else" in compact
    assert "for" in compact
    assert "while" in compact
    assert "switch" in compact
    assert "case" in compact
    assert "break" in compact
    assert "return" in compact
    assert "&" in result
    assert "|" in result
    assert "^" in result
    assert "~" in result
    assert "<<" in result
    assert ">>" in result


def test_codegen_ternary_expression():
    code = """
    void main() {
        int a = 1;
        int b = 2;
        int c = (a > b) ? a : b;
    }
    """
    result = convert(code)
    assert "1" in result and "2" in result
    assert "?" in result or "if" in result


def test_codegen_arrays_and_indexing():
    code = """
    struct Data {
        float values[4];
    };

    void main() {
        Data d;
        float arr[3];
        d.values[0] = 1.0;
        d.values[1] = 2.0;
        arr[2] = d.values[1];
    }
    """
    result = convert(code)
    assert "values[0]" in result
    assert "values[1]" in result
    assert "arr[2]" in result


def test_codegen_compute_kernel():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(device float* data [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
        data[tid] = data[tid] * 2.0;
    }
    """
    result = convert(code)
    assert "compute" in result
    assert "compute_main" in result
    assert "data[" in result


def test_codegen_preserves_literals_and_swizzles():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexInput {
        float3 position [[attribute(0)]];
    };

    vertex float4 vertex_main(VertexInput in [[stage_in]]) {
        float3 p = in.position.xyz;
        float3 v = float3(3.14159, 2.71828, 1.61803);
        return float4(p + v, 1.0);
    }
    """
    result = convert(code)
    assert ".xyz" in result
    assert "3.14159" in result
    assert "2.71828" in result
    assert "1.61803" in result


def test_codegen_preserves_binding_attributes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VSOut {
        float4 position [[position]];
    };

    vertex VSOut vertex_main(float3 pos [[attribute(0)]],
                             constant float4x4& mvp [[buffer(0)]]) {
        VSOut out;
        out.position = mvp * float4(pos, 1.0);
        return out;
    }

    fragment float4 fragment_main(texture2d<float> tex [[texture(1)]],
                                  sampler samp [[sampler(0)]]) {
        return tex.sample(samp, float2(0.5, 0.5));
    }
    """
    result = convert(code)
    assert "@buffer(0)" in result
    assert "@texture(1)" in result
    assert "@sampler(0)" in result


def test_codegen_color_attribute_with_format():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Out {
        float4 color [[color(0, rgba8unorm)]];
    };

    fragment Out fragment_main() {
        Out out;
        out.color = float4(1.0, 0.0, 0.0, 1.0);
        return out;
    }
    """
    result = convert(code)
    assert "gl_FragColor" in result


def test_codegen_texture_read_write_and_compare():
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
    result = convert(code)
    assert "textureLoad" in result
    assert "textureStore" in result
    assert "textureCompare" in result
    assert "textureGather" in result


def test_codegen_compute_builtins():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(device float* data [[buffer(0)]],
                             uint3 tid [[thread_position_in_grid]],
                             uint3 lid [[thread_position_in_threadgroup]],
                             uint3 gid [[threadgroup_position_in_grid]],
                             uint tidx [[thread_index_in_threadgroup]]) {
        data[tid.x] = float(tidx);
    }
    """
    result = convert(code)
    assert "@gl_GlobalInvocationID" in result
    assert "@gl_LocalInvocationID" in result
    assert "@gl_WorkGroupID" in result
    assert "@gl_LocalInvocationIndex" in result


def test_codegen_packed_and_simd_types():
    code = """
    struct Types {
        packed_float4 p4;
        simd_float3 s3;
        simd_float4x4 m4;
    };
    """
    result = convert(code)
    assert "vec4 p4" in result
    assert "vec3 s3" in result
    assert "mat4 m4" in result


def test_codegen_function_constants_and_argument_buffers():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Args {
        float4x4 mvp;
    };

    constant Args& args [[buffer(0), id(3), argument_buffer]];
    constant int gMode [[function_constant(0)]];

    vertex float4 vertex_main(float3 pos [[attribute(0)]]) {
        if (gMode == 1) {
            return args.mvp * float4(pos, 1.0);
        }
        return float4(pos, 1.0);
    }
    """
    result = convert(code)
    assert "@buffer(0)" in result
    assert "@id(3)" in result
    assert "@argument_buffer" in result
    assert "@function_constant(0)" in result


def test_codegen_metal_namespace_types():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(metal::texture2d<float> tex [[texture(0)]],
                                  metal::sampler samp [[sampler(0)]]) {
        return tex.sample(samp, float2(0.5, 0.5));
    }
    """
    result = convert(code)
    assert "sampler2D" in result


def test_codegen_raytracing_qualifiers_output():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    intersection void isect(raytracing::ray r, intersector inter) { }
    anyhit void any_hit() { }
    closesthit void closest_hit() { }
    miss void miss_main() { }
    callable void callable_main() { }
    """
    result = convert(code)
    assert "isect" in result
    assert "any_hit" in result
    assert "closest_hit" in result
    assert "miss_main" in result
    assert "callable_main" in result


def test_codegen_enum_and_typedef():
    code = """
    typedef int32_t MyInt;
    enum Mode { Off, On = 2, Auto };

    void main() {
        MyInt v = 1;
        Mode m = Auto;
    }
    """
    result = convert(code)
    assert "typedef int MyInt;" in result
    assert "enum Mode" in result


def test_codegen_sizeof_and_cast():
    code = """
    void main() {
        int a = sizeof(int);
        int b = alignof(float4);
        float3 v = (float3)(1.0);
    }
    """
    result = convert(code)
    assert "sizeof(int)" in result
    assert "alignof(float4)" in result
    assert "(vec3)" in result or "(float3)" in result


def test_codegen_alignas_and_static_assert():
    code = """
    alignas(16) float4 alignedValue;
    static_assert(1 == 1, "ok");

    void main() {
        alignas(float4) int v = 0;
        static_assert(sizeof(int) == 4);
    }
    """
    result = convert(code)
    assert "alignas(16)" in result
    assert "static_assert(1 == 1" in result


def test_codegen_using_alias():
    code = """
    using Index = uint;
    void main() {
        Index i = 0;
    }
    """
    result = convert(code)
    assert "typedef uint Index;" in result


def test_codegen_function_table_call_and_icb_methods():
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
    result = convert(code)
    assert "vft[0](p)" in result
    assert "icb.reset()" in result
    assert "icb.draw_primitives" in result


def test_codegen_icb_extended_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    indirect_command_buffer icb [[buffer(0)]];

    kernel void main() {
        icb.reset();
        icb.draw_indexed_primitives(3, 1, 0, 0, 0);
        icb.draw_patches(3, 1, 0, 0);
        icb.compute_dispatch(uint3(1, 1, 1));
    }
    """
    result = convert(code)
    assert "icb.draw_indexed_primitives" in result
    assert "icb.draw_patches" in result
    assert "icb.compute_dispatch" in result


def test_codegen_payload_and_hit_attributes():
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
    result = convert(code)
    assert "@payload" in result
    assert "@hit_attribute" in result


def test_codegen_mesh_object_io():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ObjPayload { float4 data; };

    object void object_main(threadgroup ObjPayload* payload [[payload]],
                            uint3 gid [[threadgroup_position_in_grid]]) { }

    mesh void mesh_main(threadgroup ObjPayload* payload [[payload]],
                        uint3 tid [[thread_position_in_threadgroup]]) { }
    """
    result = convert(code)
    assert "object_main" in result
    assert "mesh_main" in result
    assert "@payload" in result
    assert "@gl_WorkGroupID" in result


def test_codegen_mesh_output_functions():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    mesh void mesh_main() {
        SetMeshOutputCounts(64, 32);
        SetVertex(0, float3(0.0));
        SetPrimitive(0, 0);
    }
    """
    result = convert(code)
    assert "SetMeshOutputCounts" in result
    assert "SetVertex" in result
    assert "SetPrimitive" in result


def test_codegen_preserves_preprocessor_define():
    code = """
    #define FOO 1
    void main() {
        int x = FOO;
    }
    """
    result = convert(code)
    assert "#define FOO 1" in result


if __name__ == "__main__":
    pytest.main()
