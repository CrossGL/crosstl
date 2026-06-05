import re
from typing import List

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.codegen.directx_codegen import (
    HLSLCodeGen as TranslatorHLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def tokenize_code(code: str) -> List:
    lexer = MetalLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    parser = MetalParser(tokens)
    return parser.parse()


def generate_code(ast_node):
    codegen = MetalToCrossGLConverter()
    return codegen.generate(ast_node)


def convert(code: str) -> str:
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    return generate_code(ast)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    parser = CrossGLParser(tokens)
    return parser.parse()


def test_codegen_preserves_variadic_pack_expansion_from_mlx_integral_constant():
    code = """
    template <typename T, typename... Us>
    METAL_FUNC constexpr auto sum(T x, Us... us) {
        return x + sum(us...);
    }
    """
    generated = convert(code)

    assert "Us... us" in generated
    assert "sum(us...)" in generated
    assert "post..." not in generated


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


def test_codegen_drops_static_branch_attribute_after_if_condition_from_blender_shader():
    # Reduced from:
    # Repo: https://github.com/blender/blender
    # Commit: e5fc656cdab0e682296f8dd024b942b548e788f4
    # Path: source/blender/gpu/shaders/gpu_shader_2D_widget_base.bsl.hh
    code = """
    void draw_widget(bool instanced) {
        if (instanced) [[static_branch]] {
            return;
        }
    }
    """
    crossgl = convert(code)

    assert "if (instanced)" in crossgl
    assert "static_branch" not in crossgl
    assert parse_crossgl(crossgl) is not None


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
    assert "texture(albedo, samp, in_.uv)" in result
    assert "albedo" in result


def test_codegen_gpuimage_typedef_struct_and_nested_constructor_expression():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    typedef struct
    {
        float rangeReduction;
    } RangeReductionUniform;

    struct SingleInputVertexIO {
        float4 position [[position]];
        float2 textureCoordinate;
    };

    fragment half4 luminanceRangeFragment(SingleInputVertexIO fragmentInput [[stage_in]],
                                          texture2d<half> inputTexture [[texture(0)]],
                                          constant RangeReductionUniform& uniform [[buffer(1)]]) {
        constexpr sampler quadSampler;
        half4 color = inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);
        half luminanceRatio = ((0.5 - color.r) * uniform.rangeReduction);
        return half4(half3((color.rgb) + (luminanceRatio)), color.w);
    }
    """
    result = convert(code)

    assert "struct RangeReductionUniform" in result
    assert "f16vec3(color.rgb + luminanceRatio)" in result
    assert (
        "float16 luminanceRatio = (0.5 - color.r) * uniform_.rangeReduction;" in result
    )


def test_codegen_texture_sample_preserves_explicit_sampler_roundtrip():
    code = """
    float4 sampleColor(texture2d<float> albedo, sampler linearSampler, float2 uv, float lod) {
        float4 base = albedo.sample(linearSampler, uv);
        float4 mip = albedo.sample(linearSampler, uv, lod);
        return base + mip;
    }
    """
    crossgl = convert(code)

    assert "texture(albedo, linearSampler, uv)" in crossgl
    assert "textureLod(albedo, linearSampler, uv, lod)" in crossgl
    assert "texture(albedo, uv)" not in crossgl
    assert "textureLod(albedo, uv, lod)" not in crossgl

    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "texture(albedo, uv)" in glsl
    assert "textureLod(albedo, uv, lod)" in glsl

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "Texture2D albedo" in hlsl
    assert "SamplerState linearSampler" in hlsl
    assert "albedo.Sample(linearSampler, uv)" in hlsl
    assert "albedo.SampleLevel(linearSampler, uv, lod)" in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "texture2d<float> albedo" in metal
    assert "sampler linearSampler" in metal
    assert "albedo.sample(linearSampler, uv)" in metal
    assert "albedo.sample(linearSampler, uv, level(lod))" in metal


def test_codegen_texture2d_array_sample_preserves_array_slice_from_filament_sdl():
    # Reduced from:
    # Repo: https://github.com/google/filament
    # Commit: 48881c840bca50da515f0df82b61c9a5b996b19a
    # Path: third_party/libsdl2/src/render/metal/SDL_shaders_metal.metal
    code = """
    struct CopyVertexOutput {
        float4 position [[position]];
        float2 texcoord;
    };

    fragment float4 SDL_YUV_fragment(CopyVertexOutput vert [[stage_in]],
                                     texture2d<float> texY [[texture(0)]],
                                     texture2d_array<float> texUV [[texture(1)]],
                                     sampler s [[sampler(0)]]) {
        float3 yuv;
        yuv.x = texY.sample(s, vert.texcoord).r;
        yuv.y = texUV.sample(s, vert.texcoord, 0).r;
        yuv.z = texUV.sample(s, vert.texcoord, 1).r;
        return float4(yuv, 1.0);
    }
    """
    crossgl = convert(code)

    assert "texture(texUV, s, vec3(vert.texcoord, 0)).r" in crossgl
    assert "texture(texUV, s, vec3(vert.texcoord, 1)).r" in crossgl
    assert "textureLod(texUV, s, vert.texcoord, 0)" not in crossgl
    assert "textureLod(texUV, s, vert.texcoord, 1)" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "texUV.sample(s, (float3(vert.texcoord, 0)).xy" in metal
    assert "uint((float3(vert.texcoord, 0)).z)" in metal
    assert "texUV.sample(s, (float3(vert.texcoord, 1)).xy" in metal
    assert "uint((float3(vert.texcoord, 1)).z)" in metal


def test_codegen_return_type_before_stage_qualifier_from_metal_cpp_sample():
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
    crossgl = convert(code)

    assert "vertex {" in crossgl
    assert "fragment {" in crossgl
    assert "v2f vertexMain" in crossgl
    assert "f16vec4 fragmentMain" in crossgl
    parse_crossgl(crossgl)


def test_codegen_gnu_attribute_between_function_qualifiers_and_return_type_from_spirv_cross():
    # Reduced from:
    # Repo: https://github.com/KhronosGroup/SPIRV-Cross
    # Path: reference/shaders-ue4/asm/vert/texture-buffer.asm.vert
    code = """
    static inline __attribute__((always_inline))
    uint2 spvTexelBufferCoord(uint tc) {
        return uint2(tc % 4096, tc / 4096);
    }

    uint2 main0(uint tc) {
        return spvTexelBufferCoord(tc);
    }
    """
    crossgl = convert(code)

    assert "uvec2 spvTexelBufferCoord(uint tc) @always_inline" in crossgl
    assert "return uvec2(tc % 4096, tc / 4096);" in crossgl
    assert "uvec2 main0(uint tc)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_reference_to_array_params_from_spirv_cross_reference():
    # Upstream repo: https://github.com/KhronosGroup/SPIRV-Cross
    # Commit: 9fbd8b789e351c2bb772cec570c1105962056b43
    # Path: reference/opt/shaders-msl/frag/array-lut-no-loop-variable.frag
    code = """
    #pragma clang diagnostic ignored "-Wmissing-prototypes"
    #include <metal_stdlib>
    using namespace metal;

    template<typename T, uint N>
    void spvArrayCopy(thread T (&dst)[N], thread const T (&src)[N]) {
        for (uint i = 0; i < N; dst[i] = src[i], i++);
    }
    """
    ast = parse_code(tokenize_code(code))
    fn = ast.functions[0]

    assert fn.name == "spvArrayCopy"
    assert fn.params[0].vtype == "T&"
    assert fn.params[0].name == "dst"
    assert fn.params[0].array_sizes[0].name == "N"
    assert fn.params[0].qualifiers == ["thread"]
    assert fn.params[1].vtype == "T&"
    assert fn.params[1].name == "src"
    assert fn.params[1].array_sizes[0].name == "N"
    assert fn.params[1].qualifiers == ["thread", "const"]

    crossgl = generate_code(ast)

    assert "void spvArrayCopy(thread T[N]& dst, thread T[N]& src)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_fragment_early_tests_attribute_becomes_stage_layout():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    [[early_fragment_tests]]
    fragment float4 fragment_main() {
        return float4(1.0);
    }
    """
    result = convert(code)

    assert "fragment {" in result
    assert "layout(early_fragment_tests) in;" in result
    assert "@early_fragment_tests" not in result
    assert result.index("layout(early_fragment_tests) in;") < result.index(
        "vec4 fragment_main"
    )
    parse_crossgl(result)


def test_codegen_host_name_attribute_uses_exported_entry_name():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    [[host_name("api_kernel")]]
    kernel void source_kernel(device float* data [[buffer(0)]]) {
        data[0] = 1.0;
    }
    """
    crossgl = convert(code)

    assert "void api_kernel(RWStructuredBuffer<float> data @buffer(0))" in crossgl
    assert "source_kernel" not in crossgl
    assert "@host_name" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_texture_sample_level_option_roundtrip():
    code = """
    float4 sampleLevel(texture2d<float> tex, sampler samp, float2 uv, float lod) {
        float4 mip = tex.sample(samp, uv, level(lod));
        return mip;
    }
    """
    crossgl = convert(code)

    assert "textureLod(tex, samp, uv, lod)" in crossgl
    assert "level(lod)" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample(samp, uv, level(lod))" in metal
    assert "level(level(" not in metal


def test_codegen_texture_sample_compare_level_option_roundtrip():
    code = """
    float sampleCompareLevel(depth2d<float> tex,
                             sampler samp,
                             float2 uv,
                             float depth,
                             float lod) {
        return tex.sample_compare(samp, uv, depth, level(lod));
    }
    """
    crossgl = convert(code)

    assert "textureCompareLod(tex, samp, uv, depth, lod)" in crossgl
    assert "level(lod)" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample_compare(samp, uv, depth, level(lod))" in metal
    assert "level(level(" not in metal


def test_codegen_texture_sample_compare_offset_and_gradient_options_roundtrip():
    code = """
    float sampleCompareOptions(depth2d<float> tex,
                               sampler samp,
                               float2 uv,
                               float depth,
                               float lod,
                               float2 ddx,
                               float2 ddy,
                               int2 offset) {
        float mip = tex.sample_compare(samp, uv, depth, level(lod));
        float shifted = tex.sample_compare(samp, uv, depth, offset);
        float gradient = tex.sample_compare(samp, uv, depth, gradient2d(ddx, ddy));
        float gradientShifted =
            tex.sample_compare(samp, uv, depth, gradient2d(ddx, ddy), offset);
        return mip + shifted + gradient + gradientShifted;
    }
    """
    crossgl = convert(code)

    assert "textureCompareLod(tex, samp, uv, depth, lod)" in crossgl
    assert "textureCompareOffset(tex, samp, uv, depth, offset)" in crossgl
    assert "textureCompareGrad(tex, samp, uv, depth, ddx, ddy)" in crossgl
    assert "textureCompareGradOffset(tex, samp, uv, depth, ddx, ddy, offset)" in crossgl
    assert "textureCompare(tex, samp, uv, depth, gradient2d(" not in crossgl
    assert "textureCompare(tex, samp, uv, depth, offset)" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample_compare(samp, uv, depth, level(lod))" in metal
    assert "tex.sample_compare(samp, uv, depth, offset)" in metal
    assert "tex.sample_compare(samp, uv, depth, gradient2d(ddx, ddy))" in metal
    assert "tex.sample_compare(samp, uv, depth, gradient2d(ddx, ddy), offset)" in metal
    assert "sample_compare(samp, uv, depth, level(level(" not in metal


def test_codegen_texture_sample_compare_array_options_roundtrip():
    code = """
    float sampleCompareArrayOptions(depth2d_array<float> tex,
                                    depthcube_array<float> cube,
                                    sampler samp,
                                    float2 uv,
                                    float3 dir,
                                    uint layer,
                                    float depth,
                                    float lod,
                                    float2 ddx,
                                    float2 ddy,
                                    float3 ddxCube,
                                    float3 ddyCube,
                                    int2 offset) {
        float base = tex.sample_compare(samp, uv, layer, depth);
        float shifted = tex.sample_compare(samp, uv, layer, depth, offset);
        float lodShifted =
            tex.sample_compare(samp, uv, layer, depth, level(lod), offset);
        float gradientShifted =
            tex.sample_compare(samp, uv, layer, depth, gradient2d(ddx, ddy), offset);
        float cubeGradient =
            cube.sample_compare(samp, dir, layer, depth, gradientcube(ddxCube, ddyCube));
        return base + shifted + lodShifted + gradientShifted + cubeGradient;
    }
    """
    crossgl = convert(code)

    assert "textureCompare(tex, samp, vec3(uv, layer), depth)" in crossgl
    assert "textureCompareOffset(tex, samp, vec3(uv, layer), depth, offset)" in crossgl
    assert (
        "textureCompareLodOffset(tex, samp, vec3(uv, layer), depth, lod, offset)"
        in crossgl
    )
    assert (
        "textureCompareGradOffset(tex, samp, vec3(uv, layer), depth, ddx, ddy, offset)"
        in crossgl
    )
    assert (
        "textureCompareGrad(cube, samp, vec4(dir, layer), depth, ddxCube, ddyCube)"
        in crossgl
    )
    assert "textureCompareOffset(tex, samp, vec3(uv, layer), depth)" not in crossgl
    assert "textureCompare(tex, samp, vec3(uv, layer), depth, level(" not in crossgl
    assert (
        "textureCompare(tex, samp, vec3(uv, layer), depth, gradient2d(" not in crossgl
    )
    assert (
        "textureCompare(cube, samp, vec4(dir, layer), depth, gradientcube("
        not in crossgl
    )

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert (
        "tex.sample_compare(samp, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), depth)" in metal
    )
    assert (
        "tex.sample_compare(samp, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), depth, offset)" in metal
    )
    assert (
        "tex.sample_compare(samp, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), depth, level(lod), offset)" in metal
    )
    assert (
        "tex.sample_compare(samp, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), depth, gradient2d(ddx, ddy), offset)" in metal
    )
    assert (
        "cube.sample_compare(samp, (float4(dir, layer)).xyz, "
        "uint((float4(dir, layer)).w), depth, gradientcube(ddxCube, ddyCube))" in metal
    )
    assert (
        "sample_compare(samp, (float3(uv, layer)).xy, uint((float3(uv, layer)).z), depth, level(level("
        not in metal
    )


def test_codegen_texture_sample_bias_and_gradient_options_roundtrip():
    code = """
    float4 sampleOptions(texture2d<float> tex,
                         sampler samp,
                         float2 uv,
                         float biasValue,
                         float2 ddx,
                         float2 ddy) {
        float4 biased = tex.sample(samp, uv, bias(biasValue));
        float4 gradient = tex.sample(samp, uv, gradient2d(ddx, ddy));
        return biased + gradient;
    }
    """
    crossgl = convert(code)

    assert "texture(tex, samp, uv, biasValue)" in crossgl
    assert "textureGrad(tex, samp, uv, ddx, ddy)" in crossgl
    assert "textureLod(tex, samp, uv, bias(" not in crossgl
    assert "textureLod(tex, samp, uv, gradient2d(" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample(samp, uv, bias(biasValue))" in metal
    assert "tex.sample(samp, uv, gradient2d(ddx, ddy))" in metal
    assert "level(bias(" not in metal
    assert "level(gradient2d(" not in metal


def test_codegen_texture_sample_min_lod_clamp_option_roundtrip():
    code = """
    float4 sparseSample(texture2d<float> colorMap,
                        sampler colorSampler,
                        float2 uv,
                        float firstTailMip) {
        return colorMap.sample(colorSampler, uv, min_lod_clamp(firstTailMip));
    }
    """
    crossgl = convert(code)

    assert "textureMinLodClamp(colorMap, colorSampler, uv, firstTailMip)" in crossgl
    assert "textureLod(colorMap" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "colorMap.sample(colorSampler, uv, min_lod_clamp(firstTailMip))" in metal
    assert "level(min_lod_clamp(" not in metal


def test_codegen_texture_sample_offset_options_roundtrip():
    code = """
    float4 sampleOffsetOptions(texture2d<float> tex,
                               texture3d<float> volume,
                               sampler samp,
                               float2 uv,
                               float3 uvw,
                               float lod,
                               float biasValue,
                               float2 ddx,
                               float2 ddy,
                               float3 ddx3,
                               float3 ddy3,
                               int2 offset,
                               int3 offset3) {
        float4 plain = tex.sample(samp, uv, offset);
        float4 biased = tex.sample(samp, uv, bias(biasValue), offset);
        float4 lodShifted = tex.sample(samp, uv, level(lod), offset);
        float4 gradShifted = tex.sample(samp, uv, gradient2d(ddx, ddy), offset);
        float4 volumeShifted = volume.sample(samp, uvw, offset3);
        float4 volumeGradShifted =
            volume.sample(samp, uvw, gradient3d(ddx3, ddy3), offset3);
        return plain + biased + lodShifted + gradShifted
            + volumeShifted + volumeGradShifted;
    }
    """
    crossgl = convert(code)

    assert "textureOffset(tex, samp, uv, offset)" in crossgl
    assert "textureOffset(tex, samp, uv, offset, biasValue)" in crossgl
    assert "textureLodOffset(tex, samp, uv, lod, offset)" in crossgl
    assert "textureGradOffset(tex, samp, uv, ddx, ddy, offset)" in crossgl
    assert "textureOffset(volume, samp, uvw, offset3)" in crossgl
    assert "textureGradOffset(volume, samp, uvw, ddx3, ddy3, offset3)" in crossgl
    assert "textureLod(tex, samp, uv, offset)" not in crossgl
    assert "textureLod(tex, samp, uv, lod)" not in crossgl
    assert "textureGrad(tex, samp, uv, ddx, ddy)" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample(samp, uv, offset)" in metal
    assert "tex.sample(samp, uv, bias(biasValue), offset)" in metal
    assert "tex.sample(samp, uv, level(lod), offset)" in metal
    assert "tex.sample(samp, uv, gradient2d(ddx, ddy), offset)" in metal
    assert "volume.sample(samp, uvw, offset3)" in metal
    assert "volume.sample(samp, uvw, gradient3d(ddx3, ddy3), offset3)" in metal
    assert "level(offset)" not in metal


def test_codegen_texture_method_descriptors():
    converter = MetalToCrossGLConverter()

    assert converter.texture_method_descriptor("read") == {
        "method": "read",
        "function": "textureLoad",
        "storage_operation": "read",
        "sampled_texture": False,
    }
    assert converter.texture_method_descriptor("write") == {
        "method": "write",
        "function": "textureStore",
        "storage_operation": "write",
        "sampled_texture": False,
    }
    assert converter.texture_method_descriptor("sample_compare") == {
        "method": "sample_compare",
        "function": "textureCompare",
        "storage_operation": None,
        "sampled_texture": True,
    }
    assert converter.texture_method_descriptor("sample_compare_level") == {
        "method": "sample_compare_level",
        "function": "textureCompareLod",
        "storage_operation": None,
        "sampled_texture": True,
    }
    assert converter.texture_method_descriptor("gather") == {
        "method": "gather",
        "function": "textureGather",
        "storage_operation": None,
        "sampled_texture": True,
    }
    assert converter.texture_method_descriptor("gather_compare") == {
        "method": "gather_compare",
        "function": "textureGatherCompare",
        "storage_operation": None,
        "sampled_texture": True,
    }
    assert converter.texture_method_descriptor("sample") is None
    assert converter.resource_method_descriptor("read") == {
        "method": "read",
        "function": "textureLoad",
        "storage_operation": "read",
        "sampled_texture": False,
        "resource": "texture_or_image",
        "operation": "load",
    }
    assert converter.resource_method_descriptor("write") == {
        "method": "write",
        "function": "textureStore",
        "storage_operation": "write",
        "sampled_texture": False,
        "resource": "texture_or_image",
        "operation": "store",
    }
    assert converter.resource_method_descriptor("sample_compare") == {
        "method": "sample_compare",
        "function": "textureCompare",
        "storage_operation": None,
        "sampled_texture": True,
        "resource": "texture",
        "operation": "sample_compare",
    }
    assert converter.resource_method_descriptor("gather_compare") == {
        "method": "gather_compare",
        "function": "textureGatherCompare",
        "storage_operation": None,
        "sampled_texture": True,
        "resource": "texture",
        "operation": "gather_compare",
    }
    assert converter.resource_method_descriptor("sample") is None


def test_codegen_binding_attributes_do_not_roundtrip_as_semantics():
    code = """
    float4 sampleBound(texture2d<float> tex [[texture(1)]], sampler samp [[sampler(2)]], float2 uv) {
        return tex.sample(samp, uv);
    }
    """
    crossgl = convert(code)

    assert "@texture(1)" in crossgl
    assert "@sampler(2)" in crossgl

    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "vec4 sampleBound(sampler2D tex, vec2 uv)" in glsl
    assert "tex texture" not in glsl

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "float4 sampleBound(Texture2D tex, SamplerState samp, float2 uv)" in hlsl
    assert ": texture" not in hlsl
    assert ": sampler" not in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "float4 sampleBound(texture2d<float> tex, sampler samp, float2 uv)" in metal
    assert "[[texture]]" not in metal
    assert "[[sampler]]" not in metal


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


def test_codegen_nested_unbraced_for_loops_from_public_msl_example():
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
    crossgl = convert(code)

    assert "for (int i = 0; i < int(PIXEL_SIZE); i++)" in crossgl
    assert "for (int j = 0; j < int(PIXEL_SIZE); j++)" in crossgl
    assert "col += vec4(float(i + j));" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_unbraced_do_while_body_from_msl_cxx14_statement_grammar():
    code = """
    kernel void normalize(device float* values [[buffer(0)]], uint count) {
        uint i = 0;
        do
            values[i++] = 0.0f;
        while (i < count);
    }
    """
    crossgl = convert(code)

    assert "do {" in crossgl
    assert "buffer_store(values, i++, 0.0f);" in crossgl
    assert "} while (i < count);" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_multi_declarator_for_header_from_mlx_conv_loader():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 6ea7a00d05d548219864d10ff6c013b7544b13ea
    # Path: mlx/backend/metal/kernels/steel/conv/loaders/loader_general.h
    code = """
    void load_unsafe(short n_rows, short TROWS) {
        for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
            short row = is;
        }
    }
    """
    crossgl = convert(code)

    assert "/* Unhandled expression: list */" not in crossgl
    assert "for (int16 i = 0, is = 0; i < n_rows;" in crossgl
    assert "is += TROWS" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_range_for_loop_from_mlx_random():
    code = """
    void mix_values() {
        for (auto r : rotations[0]) {
            value += r;
        }
    }
    """
    crossgl = convert(code)

    assert "for r in rotations[0] {" in crossgl
    assert "value += r;" in crossgl


def test_codegen_using_union_alias_from_mlx_cexpf_header_is_diagnostic_struct():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/cexpf.h
    code = """
    using ieee_float_shape_type = union {
      float value;
      uint32_t word;
    };

    void get_float_word(thread uint32_t& i, float d) {
      ieee_float_shape_type gf_u;
      gf_u.value = d;
      i = gf_u.word;
    }
    """
    crossgl = convert(code)

    assert (
        "// Metal union ieee_float_shape_type represented as struct-like layout; "
        "overlapping storage is not modeled"
    ) in crossgl
    assert "struct ieee_float_shape_type {" in crossgl
    assert "float value;" in crossgl
    assert "uint word;" in crossgl
    assert "using ieee_float_shape_type = union" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_template_struct_base_clause_from_mlx_type_traits():
    code = """
    namespace metal {
    template <typename T>
    struct is_empty : metal::bool_constant<__is_empty(T)> {};
    }
    """
    crossgl = convert(code)

    assert "struct is_empty {" in crossgl
    assert "metal::bool_constant" not in crossgl


def test_codegen_class_helper_data_member_from_public_metal_shader():
    # Reduced from:
    # Repo: https://github.com/imxieyi/SmallPT-Metal
    # Commit: 3cdd2f1272c891e9f98fe5cda1e785f085ab2dd8
    # Path: SmallPT/loki_header.metal
    code = """
    #include <metal_stdlib>
    using namespace metal;

    class Loki {
    private:
        thread float seed;
        unsigned TausStep(const unsigned z, const int s1, const int s2,
                          const int s3, const unsigned M);

    public:
        thread Loki(const unsigned seed1, const unsigned seed2 = 1);
        thread float rand();
    };

    float read_seed(thread Loki& rng) {
        return rng.seed;
    }
    """
    crossgl = convert(code)

    assert "struct Loki" in crossgl
    assert "thread float seed;" in crossgl
    assert "TausStep" not in crossgl
    assert "thread Loki(" not in crossgl
    assert "float rand(" not in crossgl
    assert "float read_seed(thread Loki& rng)" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_gnu_inline_unsigned_helpers_from_strelka_random_shader():
    # Reduced from:
    # Repo: https://github.com/arhix52/Strelka
    # Commit: 3eec7fa260e7d598911053f9e0f38054ce1c4f60
    # Path: src/render/metal/shaders/random.h
    code = """
    inline unsigned pcg_hash(unsigned seed) {
        unsigned state = seed * 747796405u + 2891336453u;
        return state;
    }

    template<unsigned int N>
    static __inline__ unsigned int tea(unsigned int val0, unsigned int val1) {
        unsigned int v0 = val0;
        return v0;
    }

    static __inline__ unsigned int lcg(thread unsigned int &prev) {
        prev = prev * 1664525u + 1013904223u;
        return prev & 0x00FFFFFF;
    }
    """
    crossgl = convert(code)

    assert "uint pcg_hash(uint seed)" in crossgl
    assert "uint state = seed * 747796405u + 2891336453u;" in crossgl
    assert "uint tea(uint val0, uint val1)" in crossgl
    assert "uint lcg(thread uint& prev)" in crossgl
    assert "__inline__" not in crossgl
    assert "unsigned" not in crossgl
    assert parse_crossgl(crossgl) is not None


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
    assert "RWStructuredBuffer<float> data @buffer(0)" in result
    assert "buffer_store(data, tid, buffer_load(data, tid) * 2.0);" in result


def test_codegen_device_buffer_parameters_use_structured_buffer_contract():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(device float* data [[buffer(0)]],
                             constant float* input [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
        float value = input[tid];
        data[tid] = value * 2.0;
    }
    """
    crossgl = convert(code)

    assert "RWStructuredBuffer<float> data @buffer(0)" in crossgl
    assert "StructuredBuffer<float> input @buffer(1)" in crossgl
    assert "float value = buffer_load(input, tid);" in crossgl
    assert "buffer_store(data, tid, value * 2.0);" in crossgl
    assert "data[tid]" not in crossgl
    assert "input[tid]" not in crossgl

    ast = parse_crossgl(crossgl)
    assert ast is not None

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "RWStructuredBuffer<float> data" in hlsl
    assert "StructuredBuffer<float> input" in hlsl
    assert "float value = input.Load(tid);" in hlsl
    assert "data.Store(tid, (value * 2.0));" in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "void compute_main(device float* data, const device float* input" in metal
    assert "float value = input[tid];" in metal
    assert "data[tid] = value * 2.0;" in metal


def test_roundtrip_scalar_thread_position_in_grid_from_apple_compute_sample():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void main(device const float* inA [[buffer(0)]],
                     device float* result [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
        result[index] = inA[index];
    }
    """
    crossgl = convert(code)

    assert "uint index @gl_GlobalInvocationID" in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)

    assert "uint index [[thread_position_in_grid]]" in metal
    assert "result[index] = inA[index];" in metal


def test_roundtrip_threads_per_threadgroup_from_apple_threadgroups_doc():
    # Reduced from Apple's "Creating threads and threadgroups" documentation.
    # https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void myKernel(
        uint2 threadgroupPositionInGrid [[ threadgroup_position_in_grid ]],
        uint2 threadPositionInThreadgroup [[ thread_position_in_threadgroup ]],
        uint2 threadsPerThreadgroup [[ threads_per_threadgroup ]]) {
        uint2 threadPositionInGrid =
            (threadgroupPositionInGrid * threadsPerThreadgroup) +
            threadPositionInThreadgroup;
    }
    """
    crossgl = convert(code)

    assert "uvec2 threadsPerThreadgroup @gl_WorkGroupSize" in crossgl
    assert "@threads_per_threadgroup" not in crossgl
    assert (
        "uvec2 threadPositionInGrid = "
        "threadgroupPositionInGrid * threadsPerThreadgroup "
        "+ threadPositionInThreadgroup;"
    ) in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "uint2 threadsPerThreadgroup [[threads_per_threadgroup]]" in metal


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


def test_codegen_preserves_leading_decimal_float_literals():
    code = """
    constexpr constant static float kvalues_mxfp4_f[4] = {0, .5f, 1.f, -.5f};
    """
    result = convert(code)

    assert "constant float[4] kvalues_mxfp4_f = {0, .5f, 1.f, (-.5f)};" in result
    assert "const constant" not in result
    assert "float(0, .5f" not in result
    assert ".5f" in result
    assert "1.f" in result
    assert "(-.5f)" in result

    metal = MetalCodeGen().generate(parse_crossgl(result))
    assert "constant float kvalues_mxfp4_f[4] = {0, 0.5, 1.0, -0.5};" in metal


def test_codegen_roundtrips_global_constant_half_vector_initializer():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constant half3 luminanceWeighting = half3(0.2126h, 0.7152h, 0.0722h);
    """
    crossgl = convert(code)

    assert (
        "constant f16vec3 luminanceWeighting = " "f16vec3(0.2126, 0.7152, 0.0722);"
    ) in crossgl
    assert "0.2126h" not in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert (
        "constant half3 luminanceWeighting = " "half3(0.2126, 0.7152, 0.0722);"
    ) in metal


def test_codegen_lowers_as_type_float_template_call():
    code = """
    static inline float fp32_from_bits(uint32_t bits) {
        return as_type<float>(bits);
    }
    """
    crossgl = convert(code)

    assert "asfloat(bits)" in crossgl
    assert "as_type<float>" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_lowers_static_cast_from_apple_compute_sample():
    code = """
    kernel void process(uint2 gid [[thread_position_in_grid]]) {
        float2 p0 = static_cast<float2>(gid);
    }
    """
    crossgl = convert(code)

    assert "vec2 p0 = (vec2)gid;" in crossgl
    assert "static_cast" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_scoped_atomic_thread_fence_from_mlx_kernel_roundtrips():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 6ea7a00d05d548219864d10ff6c013b7544b13ea
    # Path: mlx/backend/metal/kernels/fence.metal
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void fence() {
        metal::atomic_thread_fence(metal::mem_flags::mem_device,
                                   metal::memory_order_seq_cst,
                                   metal::thread_scope_system);
    }
    """
    crossgl = convert(code)

    assert "memoryBarrier();" in crossgl
    assert "metal_u3a_u3aatomic_thread_fence" not in crossgl
    assert "metal::atomic_thread_fence" not in crossgl
    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "threadgroup_barrier(mem_flags::mem_device);" in metal


def test_codegen_lowers_combined_threadgroup_barrier_flags_from_blender_builtin():
    # Reduced from:
    # Repo: https://github.com/blender/blender
    # Commit: b8e327c77fed04517e9a6ec8d306c8c3986d531b
    # Path: source/blender/gpu/shaders/gpu_shader_msl_builtin.msl
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void synchronize_all() {
        threadgroup_barrier(mem_flags::mem_threadgroup |
                            mem_flags::mem_device |
                            mem_flags::mem_texture);
    }
    """
    crossgl = convert(code)

    assert "allMemoryBarrier();" in crossgl
    assert "threadgroup_barrier" not in crossgl
    assert "mem_flags_u3a_u3a" not in crossgl
    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert (
        "threadgroup_barrier(mem_flags::mem_device | "
        "mem_flags::mem_threadgroup | mem_flags::mem_texture);"
    ) in metal


def test_codegen_scoped_variable_template_expression_from_mlx_gemv_masked():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/gemv_masked.h
    code = """
    using namespace metal;

    constant constexpr const bool has_operand_mask =
        !metal::is_same_v<op_mask_t, nomask_t>;
    constant constexpr const bool has_mul_operand_mask =
        has_operand_mask && !metal::is_same_v<op_mask_t, bool>;
    """
    crossgl = convert(code)

    assert "metal_u3a_u3ais_same_v_u3cop_mask_t_u2cnomask_t_u3e" in crossgl
    assert "metal_u3a_u3ais_same_v_u3cop_mask_t_u2cbool_u3e" in crossgl
    assert "is_same_v<" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_normalizes_generic_atomic_types_from_apple_msl_spec():
    # Provenance: Apple Metal Shading Language Specification, section 2.8
    # "Atomic Data Types", version 2025-10-23.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void atomics(device atomic<uint>* counters [[buffer(0)]],
                        device metal::atomic<float>* weights [[buffer(1)]],
                        device atomic<ulong>* totals [[buffer(2)]],
                        device atomic<bool>* flags [[buffer(3)]]) {
    }
    """
    crossgl = convert(code)

    assert "RWStructuredBuffer<atomic_uint> counters @buffer(0)" in crossgl
    assert "RWStructuredBuffer<atomic_float> weights @buffer(1)" in crossgl
    assert "RWStructuredBuffer<atomic_ulong> totals @buffer(2)" in crossgl
    assert "RWStructuredBuffer<atomic_bool> flags @buffer(3)" in crossgl
    assert "atomic<" not in crossgl
    assert "metal::atomic" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_unknown_struct_cast_uses_parseable_constructor_call():
    code = """
    struct Token {
        float value;
    };

    void f(float x) {
        Token t = (Token)x;
    }
    """
    crossgl = convert(code)

    assert "Token t = Token(x);" in crossgl
    assert "Token t = (Token)x;" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_ignores_function_body_pragma_from_llama_cpp():
    code = """
    void quantize_q4_0(device const float* src, device block_q4_0& dst) {
        #pragma METAL fp math_mode(safe)
        float amax = 0.0f;
        dst.d = amax;
    }
    """
    crossgl = convert(code)

    assert "#pragma" not in crossgl
    assert "float amax = 0.0f;" in crossgl
    assert "dst.d = amax;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_comma_assignment_statement_from_llama_cpp():
    code = """
    void dequantize(device const float* values) {
        float dl = 0.0f;
        float ml = 0.0f;
        dl = values[0], ml = values[1];
    }
    """
    crossgl = convert(code)

    assert "dl = values[0] , ml = values[1];" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_braced_uchar_vector_constructor_from_llama_cpp():
    code = """
    static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
        return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                     : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                              uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
    }
    """
    crossgl = convert(code)
    normalized = normalize(crossgl)

    assert "u8vec2 get_scale_min_k4_just2" in crossgl
    assert "u8vec2(uint8(q[j + 0 + k] & 63), uint8(q[j + 4 + k] & 63))" in crossgl
    assert "uchar2{" not in crossgl
    assert "return j < 4 ?" in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_standalone_scoped_block_from_llama_cpp():
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
    crossgl = convert(code)

    assert "void FC_unary_op" in crossgl
    assert re.search(r"\n\s+\{\n\s+if \(i0 >= 4\)", crossgl)
    assert "float x = src0[i0];" in crossgl
    assert "dst[i0] = x;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_statement_expression_block_from_angle_generated_shader():
    # Reduced from:
    # Repo: https://android.googlesource.com/platform/external/angle
    # Commit: 282a5fb4ad
    # Path: src/libANGLE/renderer/metal/shaders/mtl_internal_shaders_autogen.metal
    code = """
    void outputPrimitive(bool use16,
                         bool use32,
                         device ushort* out16,
                         device uint* out32,
                         thread uint& onOutIndex,
                         uint tmpIndex) {
        ({
            if (use16) {
                out16[(onOutIndex)] = tmpIndex;
            }
            if (use32) {
                out32[(onOutIndex)] = tmpIndex;
            }
            onOutIndex++;
        });
    }
    """
    crossgl = convert(code)

    assert re.search(r"\n\s+\{\n\s+if \(use16\)", crossgl)
    assert "out16[onOutIndex] = tmpIndex;" in crossgl
    assert "out32[onOutIndex] = tmpIndex;" in crossgl
    assert "onOutIndex++;" in crossgl
    assert "({" not in crossgl
    assert parse_crossgl(crossgl) is not None


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
    assert "image2D tex @texture(0) @readwrite" in result
    assert "imageLoad(tex, uvec2(0, 0))" in result
    assert "imageStore(tex, uvec2(0, 0), c);" in result
    assert "unsupported Metal storage texture sampled method: sample on tex" in result
    assert (
        "unsupported Metal storage texture sampled method: sample_compare on tex"
        in result
    )
    assert "unsupported Metal storage texture sampled method: gather on tex" in result
    assert "textureStore" not in result
    assert "texture(tex" not in result
    assert "textureCompare" not in result
    assert "textureGather" not in result


def test_codegen_metal_namespace_access_qualifiers_for_storage_textures():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ImagePack {
        metal::texture2d<uint, metal::access::read_write> image;
        metal::array<metal::texture2d<uint, metal::access::read>, 2> inputs;
    };

    kernel void compute_main(
        metal::texture2d<float, metal::access::read_write> image [[texture(0)]],
        constant ImagePack& pack [[buffer(0)]],
        uint2 tid [[thread_position_in_grid]]) {
        float4 color = image.read(tid);
        image.write(color, tid);
        uint oldValue = pack.image.read(tid).x;
        float4 inputValue = float4(pack.inputs[1].read(tid));
    }
    """
    result = convert(code)

    assert "image2D image @texture(0) @readwrite" in result
    assert "uimage2D image @readwrite" in result
    assert "uimage2D[2] inputs @readonly" in result
    assert "imageLoad(image, tid)" in result
    assert "imageStore(image, tid, color);" in result
    assert "uint oldValue = imageLoad(pack.image, tid).x;" in result
    assert "vec4 inputValue = vec4(imageLoad(pack.inputs[1], tid));" in result
    assert "unsupported Metal sampled texture write" not in result


def test_codegen_sampled_texture_write_emits_diagnostic():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment void fragment_main(texture2d<float> tex [[texture(0)]]) {
        tex.write(float4(1.0), uint2(0, 0));
    }
    """
    result = convert(code)

    assert "sampler2D tex @texture(0)" in result
    assert "unsupported Metal sampled texture write: write on tex" in result
    assert "textureStore" not in result
    assert "imageStore" not in result


def test_codegen_sampled_texture_reads_roundtrip_to_texel_fetch():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(texture2d<float> tex [[texture(0)]],
                                  texture2d_array<float> layers [[texture(1)]],
                                  texture2d_ms<float> msTex [[texture(2)]],
                                  texture2d_ms_array<float> msLayers [[texture(3)]],
                                  texture1d<float> line [[texture(4)]],
                                  texture1d_array<float> lineLayers [[texture(5)]],
                                  texture3d<float> volume [[texture(6)]],
                                  uint x,
                                  uint2 pixel,
                                  uint3 voxel,
                                  uint layer,
                                  uint lod,
                                  uint sample) {
        float4 a = tex.read(pixel, lod);
        float4 b = layers.read(pixel, layer, lod);
        float4 c = msTex.read(pixel, sample);
        float4 d = msLayers.read(pixel, layer, sample);
        float4 e = line.read(x, 0);
        float4 f = lineLayers.read(x, layer, 0);
        float4 g = volume.read(voxel, lod);
        return a + b + c + d + e + f + g;
    }
    """
    crossgl = convert(code)

    assert "vec4 a = texelFetch(tex, pixel, lod);" in crossgl
    assert "vec4 b = texelFetch(layers, uvec3(pixel, layer), lod);" in crossgl
    assert "vec4 c = texelFetch(msTex, pixel, sample);" in crossgl
    assert "vec4 d = texelFetch(msLayers, uvec3(pixel, layer), sample);" in crossgl
    assert "vec4 e = texelFetch(line, x, 0);" in crossgl
    assert "vec4 f = texelFetch(lineLayers, uvec2(x, layer), 0);" in crossgl
    assert "vec4 g = texelFetch(volume, voxel, lod);" in crossgl
    assert "textureLoad(" not in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "tex.read(pixel, lod)" in metal
    assert (
        "layers.read((uint3(pixel, layer)).xy, " "uint((uint3(pixel, layer)).z), lod)"
    ) in metal
    assert "msTex.read(pixel, uint(sample))" in metal
    assert (
        "msLayers.read((uint3(pixel, layer)).xy, "
        "uint((uint3(pixel, layer)).z), uint(sample))"
    ) in metal
    assert "line.read(uint(x), uint(0))" in metal
    assert (
        "lineLayers.read(uint((uint2(x, layer)).x), "
        "uint((uint2(x, layer)).y), uint(0))"
    ) in metal
    assert "volume.read(voxel, lod)" in metal
    assert "textureLoad(" not in metal


def test_codegen_access_qualified_1d_storage_textures():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(texture1d<float, access::read_write> line [[texture(0)]],
                             texture1d_array<uint, access::read_write> counters [[texture(1)]],
                             uint tid [[thread_position_in_grid]]) {
        float4 c = line.read(tid);
        line.write(c, tid);
        uint v = counters.read(tid, 0);
        counters.write(v, tid, 0);
    }
    """
    result = convert(code)
    assert "image1D line @texture(0) @readwrite" in result
    assert "uimage1DArray counters @texture(1) @readwrite" in result
    assert "vec4 c = imageLoad(line, tid);" in result
    assert "imageStore(line, tid, c);" in result
    assert "uint v = imageLoad(counters, uvec2(tid, 0));" in result
    assert "imageStore(counters, uvec2(tid, 0), v);" in result

    shader_ast = parse_crossgl(result)
    assert shader_ast is not None


def test_codegen_access_qualified_2d_3d_storage_textures():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(texture2d<uint, access::read_write> counters [[texture(0)]],
                             texture2d_array<float, access::read_write> layers [[texture(1)]],
                             texture3d<int, access::read_write> volume [[texture(2)]],
                             uint2 pixel [[thread_position_in_grid]]) {
        uint oldValue = counters.read(pixel).x;
        counters.write(uint4(oldValue), pixel);
        float4 c = layers.read(pixel, 1);
        layers.write(c, pixel, 1);
        int s = volume.read(uint3(pixel, 0)).x;
        volume.write(int4(s), uint3(pixel, 0));
    }
    """
    result = convert(code)
    assert "uimage2D counters @texture(0) @readwrite" in result
    assert "image2DArray layers @texture(1) @readwrite" in result
    assert "iimage3D volume @texture(2) @readwrite" in result
    assert "uint oldValue = imageLoad(counters, pixel).x;" in result
    assert "imageStore(counters, pixel, uvec4(oldValue));" in result
    assert "vec4 c = imageLoad(layers, uvec3(pixel, 1));" in result
    assert "imageStore(layers, uvec3(pixel, 1), c);" in result
    assert "int s = imageLoad(volume, uvec3(pixel, 0)).x;" in result
    assert "imageStore(volume, uvec3(pixel, 0), ivec4(s));" in result

    shader_ast = parse_crossgl(result)
    assert shader_ast is not None


def test_codegen_fixed_texture_arrays_lower_resource_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(array<texture2d<float>, 2> textures [[texture(0)]],
                                  array<texture2d<float, access::read_write>, 2> images [[texture(2)]],
                                  sampler samp [[sampler(0)]],
                                  uint index [[user(locn0)]]) {
        float4 c = textures[index].sample(samp, float2(0.5, 0.5));
        float4 r = images[index].read(uint2(1, 2));
        images[index].write(c, uint2(1, 2));
        return c + r;
    }
    """
    result = convert(code)

    assert "sampler2D[2] textures @texture(0)" in result
    assert "image2D[2] images @texture(2) @readwrite" in result
    assert "texture(textures[index], samp, vec2(0.5, 0.5))" in result
    assert "imageLoad(images[index], uvec2(1, 2))" in result
    assert "imageStore(images[index], uvec2(1, 2), c);" in result
    assert "array<texture" not in result
    assert "textureLoad(images" not in result
    assert "textureStore(images" not in result

    shader_ast = parse_crossgl(result)
    assert shader_ast is not None


def test_codegen_struct_member_storage_textures_lower_resource_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ImagePack {
        texture2d<uint, access::read_write> image;
        array<texture2d<uint, access::read_write>, 4> images;
        texture2d_array<float, access::read> layers;
    };

    uint touch(ImagePack pack, uint layer, uint2 pixel, float4 value) {
        uint oldValue = pack.image.read(pixel).x;
        pack.image.write(uint4(oldValue), pixel);
        uint arrayOld = pack.images[layer].read(pixel).x;
        pack.images[layer].write(uint4(arrayOld), pixel);
        float4 layered = pack.layers.read(pixel, layer);
        return oldValue + arrayOld + uint(layered.x);
    }
    """
    result = convert(code)

    assert "struct ImagePack" in result
    assert "uimage2D image @readwrite" in result
    assert "uimage2D[4] images @readwrite" in result
    assert "image2DArray layers @readonly" in result
    assert "uint oldValue = imageLoad(pack.image, pixel).x;" in result
    assert "imageStore(pack.image, pixel, uvec4(oldValue));" in result
    assert "uint arrayOld = imageLoad(pack.images[layer], pixel).x;" in result
    assert "imageStore(pack.images[layer], pixel, uvec4(arrayOld));" in result
    assert "vec4 layered = imageLoad(pack.layers, uvec3(pixel, layer));" in result
    assert "textureLoad(pack.image" not in result
    assert "textureLoad(pack.images" not in result
    assert "textureLoad(pack.layers" not in result
    assert "unsupported Metal sampled texture write" not in result

    metal = MetalCodeGen().generate(parse_crossgl(result))
    assert "texture2d<uint, access::read_write> image;" in metal
    assert "array<texture2d<uint, access::read_write>, 4> images;" in metal
    assert "texture2d_array<float, access::read> layers;" in metal
    assert "uint oldValue = pack.image.read(uint2(pixel)).x;" in metal
    assert "pack.image.write(uint4(oldValue), uint2(pixel));" in metal
    assert "uint arrayOld = pack.images[layer].read(uint2(pixel)).x;" in metal
    assert "pack.images[layer].write(uint4(arrayOld), uint2(pixel));" in metal
    assert "pack.layers.read" in metal
    assert ".x.x" not in metal
    assert "uint4(uint4(" not in metal
    assert "textureLoad(" not in metal
    assert "textureStore(" not in metal


def test_codegen_preserves_storage_texture_access_modes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(texture2d<uint, access::read> counters [[texture(0)]],
                             texture2d<float, access::write> outImage [[texture(1)]],
                             uint2 pixel [[thread_position_in_grid]]) {
        uint4 value = counters.read(pixel);
        outImage.write(float4(value), pixel);
    }
    """
    result = convert(code)

    assert "uimage2D counters @texture(0) @readonly" in result
    assert "image2D outImage @texture(1) @writeonly" in result
    assert "uvec4 value = imageLoad(counters, pixel);" in result
    assert "imageStore(outImage, pixel, vec4(value));" in result

    shader_ast = parse_crossgl(result)
    assert shader_ast is not None


def test_codegen_texture_query_methods_lower_to_crossgl_queries():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment int4 fragment_main(texture2d<float> tex [[texture(0)]],
                                texture2d_array<float> layers [[texture(1)]],
                                texture2d_ms<float> msTex [[texture(2)]],
                                float lod) {
        int2 size = int2(tex.get_width(uint(lod)), tex.get_height(uint(lod)));
        int3 layerSize = int3(layers.get_width(1),
                              layers.get_height(1),
                              layers.get_array_size());
        int levels = tex.get_num_mip_levels();
        int samples = msTex.get_num_samples();
        return int4(size.x + layerSize.z, size.y, levels, samples);
    }
    """
    crossgl = convert(code)

    assert "ivec2 size = textureSize(tex, uint(lod));" in crossgl
    assert "ivec3 layerSize = textureSize(layers, 1);" in crossgl
    assert "int levels = textureQueryLevels(tex);" in crossgl
    assert "int samples = textureSamples(msTex);" in crossgl
    assert ".get_width" not in crossgl
    assert ".get_height" not in crossgl
    assert ".get_array_size" not in crossgl
    assert ".get_num_mip_levels" not in crossgl
    assert ".get_num_samples" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None


def test_codegen_storage_texture_query_methods_lower_to_image_size():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void compute_main(texture2d<float, access::read_write> image [[texture(0)]],
                             texture3d<uint, access::read> volume [[texture(1)]],
                             uint3 tid [[thread_position_in_grid]]) {
        int2 imageSizeValue = int2(image.get_width(), image.get_height());
        int3 volumeSize = int3(volume.get_width(), volume.get_height(), volume.get_depth());
    }
    """
    crossgl = convert(code)

    assert "ivec2 imageSizeValue = imageSize(image);" in crossgl
    assert "ivec3 volumeSize = imageSize(volume);" in crossgl
    assert "image2D image @texture(0) @readwrite" in crossgl
    assert "uimage3D volume @texture(1) @readonly" in crossgl
    assert ".get_width" not in crossgl
    assert ".get_height" not in crossgl
    assert ".get_depth" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None


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


def test_codegen_simdgroup_indices_from_public_pmetal_kernel():
    # Reduced from:
    # Repo: https://github.com/Epistates/pmetal
    # Commit: 089171635d1b9c9b7a58b575cf7d522834022cd3
    # Path: crates/pmetal-metal/src/kernels/metal/fused_lora.metal
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void fused_lora_forward(device const half* x [[buffer(0)]],
                                   uint3 tid [[thread_position_in_threadgroup]],
                                   uint simd_lane_id [[thread_index_in_simdgroup]],
                                   uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
        device const half* x_row = x + tid.x;
        half value = *(x_row + simd_lane_id + simd_group_id);
    }
    """
    result = convert(code)

    assert "uint simd_lane_id @gl_SubgroupInvocationID" in result
    assert "uint simd_group_id @gl_SubgroupID" in result
    assert "@thread_index_in_simdgroup" not in result
    assert "@simdgroup_index_in_threadgroup" not in result
    assert parse_crossgl(result) is not None


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
        texture2d<float> albedo [[id(0)]];
        sampler linearSampler [[id(1)]];
        device float* weights [[id(2)]];
    };

    constant Args& args [[buffer(4), id(3), argument_buffer]];
    constant bool useFastPath [[function_constant(7)]];

    fragment float4 fragment_main(float2 uv [[stage_in]]) {
        if (useFastPath) {
            return args.albedo.sample(args.linearSampler, uv) * args.weights[0];
        }
        return float4(0.0);
    }
    """
    crossgl = convert(code)
    assert "@buffer(4)" in crossgl
    assert "@id(3)" in crossgl
    assert "@argument_buffer" in crossgl
    assert "@function_constant(7)" in crossgl
    assert "@id(0)" in crossgl
    assert "@id(1)" in crossgl
    assert "@id(2)" in crossgl

    regenerated = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "constant Args& args" in regenerated
    assert "[[buffer(4)]]" in regenerated
    assert "[[id(3)]]" in regenerated
    assert "[[argument_buffer]]" in regenerated
    assert "[[function_constant(7)]]" in regenerated
    assert "texture2d<float> albedo [[id(0)]];" in regenerated
    assert "sampler linearSampler [[id(1)]];" in regenerated
    assert "device float* weights [[id(2)]];" in regenerated
    assert "constant Args args" not in regenerated


def test_codegen_argument_buffer_array_of_device_pointers_from_apple_sample():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct FragmentShaderArguments {
        array<texture2d<float>, AAPLNumTextureArguments> exampleTextures
            [[id(AAPLArgumentBufferIDExampleTextures)]];
        array<device float *, AAPLNumBufferArguments> exampleBuffers
            [[id(AAPLArgumentBufferIDExampleBuffers)]];
        array<uint32_t, AAPLNumBufferArguments> exampleConstants
            [[id(AAPLArgumentBufferIDExampleConstants)]];
    };
    """
    crossgl = convert(code)

    assert "sampler2D[AAPLNumTextureArguments] exampleTextures" in crossgl
    assert "device float*[AAPLNumBufferArguments] exampleBuffers" in crossgl
    assert "@id(AAPLArgumentBufferIDExampleBuffers)" in crossgl
    assert "uint[AAPLNumBufferArguments] exampleConstants" in crossgl
    assert "devicefloat" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_argument_buffer_reference_array_parameter_roundtrips():
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
    crossgl = convert(code)

    assert "constant sampler2D& texturesAB1 @buffer(0)" in crossgl
    assert "constant sampler2D& texturesAB2[10] @buffer(1)" in crossgl
    assert "sampler2D[10] texturesArray @texture(0)" in crossgl
    assert "sampler2D&[10] texturesAB2" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_defaulted_function_constant_preserves_attribute():
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
    result = convert(code)
    assert "constant bool useFastPath @function_constant(3) = true;" in result
    assert "if (useFastPath)" in result


def test_codegen_mlx_steel_const_function_constants_from_fft():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: e9e20fa69184bd38cc0ca12bd9a854c059e59588
    # Path: mlx/backend/metal/kernels/fft.h
    code = """
    #include <metal_common>
    #include "mlx/backend/metal/kernels/steel/defines.h"
    using namespace metal;

    STEEL_CONST bool inv_ [[function_constant(0)]];
    STEEL_CONST int elems_per_thread_ [[function_constant(2)]];

    float2 apply_fft_flag(float2 value) {
        if (inv_) {
            value.y = -value.y;
        }
        return value * float(elems_per_thread_);
    }
    """
    crossgl = convert(code)

    assert "constant bool inv_ @function_constant(0);" in crossgl
    assert "constant int elems_per_thread_ @function_constant(2);" in crossgl
    assert "value.y = (-value.y);" in crossgl
    assert "STEEL_CONST" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_skips_mlx_decltype_kernel_template_id_instantiation():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Path: mlx/backend/metal/jit/indexing.h
    code = """
    template <typename T>
    [[kernel]] void slice_update_op_impl(device T* out [[buffer(0)]]) {
        out[0] = T(0);
    }

    [[kernel]] decltype(slice_update_op_impl<float>) slice_update_op_impl<float>;
    """
    crossgl = convert(code)

    assert "slice_update_op_impl" in crossgl
    assert "decltype" not in crossgl
    assert "slice_update_op_impl<float>" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_sanitizes_crossgl_keyword_identifiers_from_real_msl():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    typedef struct {
        float brightness;
    } BrightnessUniform;

    fragment half4 brightnessFragment(VertexOut in [[stage_in]],
                                      texture2d<half> texture [[texture(0)]],
                                      constant BrightnessUniform& uniform [[buffer(1)]]) {
        half4 color = texture.sample(linearSampler, in.textureCoordinate);
        return half4(color.rgb + uniform.brightness, color.a);
    }
    """
    crossgl = convert(code)

    assert "VertexOut in_" in crossgl
    assert "sampler2D texture_" in crossgl
    assert "BrightnessUniform& uniform_" in crossgl
    assert "in.textureCoordinate" not in crossgl
    assert "uniform.brightness" not in crossgl

    ast = parse_crossgl(crossgl)
    assert ast is not None


def test_codegen_omits_global_constexpr_sampler_argument_for_roundtrip():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constexpr sampler sampler2d(coord::normalized, filter::linear);

    fragment half4 second_passthrough(QuadVertexOut in [[stage_in]],
                                      texture2d<float, access::sample> texture [[texture(0)]]) {
        float4 const color = texture.sample(sampler2d, in.texCoords);
        return half4(half3(color.rgb), 1);
    }
    """
    crossgl = convert(code)

    assert "texture(texture_, in_.texCoords)" in crossgl
    assert "sampler2d" not in crossgl
    assert "texture(texture_, sampler2d" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "texture_.sample(sampler(" in metal


def test_roundtrip_local_constexpr_sampler_options_from_apple_texture_sample():
    code = """
    struct RasterizerData {
        float4 position [[position]];
        float2 textureCoordinate;
    };

    fragment float4 samplingShader(RasterizerData in [[stage_in]],
                                   texture2d<half> colorTexture [[texture(0)]]) {
        constexpr sampler textureSampler (mag_filter::linear, min_filter::linear);
        const half4 colorSample = colorTexture.sample(textureSampler, in.textureCoordinate);
        return float4(colorSample);
    }
    """
    crossgl = convert(code)

    assert "_u3a_u3a" not in crossgl
    assert "sampler(mag_filter::linear, min_filter::linear)" in crossgl
    assert "@ stage_entry" in crossgl
    assert "vec4 samplingShader" in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "_u3a_u3a" not in metal
    assert "mag_filter::linear" in metal
    assert "min_filter::linear" in metal
    assert "fragment float4 samplingShader(" in metal
    assert "texture2d<float> colorTexture [[texture(0)]]" in metal
    assert "fragment void fragment_main()" not in metal


def test_codegen_keyword_named_sampler_from_apple_filter_sample():
    code = """
    struct RasterizerData {
        float4 position [[position]];
        float2 texCoord;
    };

    fragment half4 texturedQuadFragment(RasterizerData in [[stage_in]],
                                        texture2d<half> texture [[texture(0)]],
                                        constant float& mipmapBias [[buffer(0)]]) {
        constexpr sampler sampler(min_filter::linear,
                                  mag_filter::linear,
                                  mip_filter::linear);
        half4 color = texture.sample(sampler, in.texCoord, level(mipmapBias));
        return color;
    }
    """
    crossgl = convert(code)

    assert "const sampler sampler_ = sampler(" in crossgl
    assert "textureLod(texture_, sampler_, in_.texCoord, mipmapBias)" in crossgl
    assert " texture.sample(sampler" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_sanitizes_unicode_identifiers_for_crossgl_parse():
    code = """
    void main() {
        float const 𝛂 = 1.0;
        float value = 𝛂;
    }
    """
    crossgl = convert(code)

    assert "_u1d6c2" in crossgl
    assert "𝛂" not in crossgl
    assert parse_crossgl(crossgl) is not None


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
    # Source inspiration: Apple WWDC20 "Get to know Metal function pointers"
    # shows visible_function_table<T> resources passed through buffer bindings and
    # invoked through indexed table calls.
    # https://developer.apple.com/videos/play/wwdc2020/10013/
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
    assert "visible_function_table vft @buffer(0);" in result
    assert "visible_function_table<" not in result
    assert "vft[0](p)" in result
    assert "icb.reset()" in result
    assert "icb.draw_primitives" in result
    assert parse_crossgl(result) is not None


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


def test_codegen_expands_preprocessor_define():
    code = """
    #define FOO 1
    void main() {
        int x = FOO;
    }
    """
    result = convert(code)
    assert "#define FOO 1" not in result
    assert "int x = 1;" in result


def test_codegen_threadgroup_memory_and_barrier():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void main(device float* data [[buffer(0)]],
                     threadgroup float* sharedMem [[threadgroup(0)]],
                     uint tid [[thread_index_in_threadgroup]]) {
        sharedMem[tid] = data[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        data[tid] = sharedMem[tid];
    }
    """
    result = convert(code)
    assert "workgroupBarrier();" in result
    assert "@threadgroup" in result or "threadgroup" in result


def test_codegen_preserves_lambda_callback_from_mlx_fp_quantized_nax():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/fp_quantized_nax.h
    code = """
    void run(bool is_unaligned_sm) {
      dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        if constexpr (kAlignedM.value) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      });
    }
    """
    result = convert(code)
    compact = normalize(result)

    assert "dispatch_bool" in compact
    assert "[&](auto kAlignedM)" in compact
    assert "if (kAlignedM.value)" in compact
    assert "workgroupBarrier();" in compact
    assert "Unhandled expression" not in compact


def test_codegen_preserves_native_address_space_qualifiers():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload {
        float value;
    };

    void update(threadgroup Payload& scratch,
                device float values[],
                constant uint& count,
                thread float& localValue) {
        scratch.value = values[count] + localValue;
    }

    kernel void main(device float* outData [[buffer(0)]],
                     constant float* inData [[buffer(1)]],
                     uint tid [[thread_index_in_threadgroup]]) {
        threadgroup Payload scratch;
        thread float localValue = inData[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch.value = localValue;
        outData[tid] = scratch.value;
    }
    """
    crossgl = convert(code)

    assert "void update(threadgroup Payload& scratch" in crossgl
    assert "device float[] values" in crossgl
    assert "constant uint& count" in crossgl
    assert "thread float& localValue" in crossgl
    assert "threadgroup Payload scratch;" in crossgl
    assert "thread float localValue = buffer_load(inData, tid);" in crossgl
    assert "RWStructuredBuffer<float> outData @buffer(0)" in crossgl
    assert "StructuredBuffer<float> inData @buffer(1)" in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "void update(threadgroup Payload& scratch" in metal
    assert "device float values[]" in metal
    assert "constant uint& count" in metal
    assert "thread float& localValue" in metal
    assert "threadgroup Payload scratch;" in metal
    assert "float localValue = inData[tid];" in metal
    assert "unsupported Metal address-space call" not in metal


if __name__ == "__main__":
    pytest.main()
