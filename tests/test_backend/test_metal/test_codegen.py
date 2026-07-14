import re
import shutil
import subprocess
from typing import List

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalAtomicFenceLoweringError,
    MetalBuiltinOverloadResolutionError,
    MetalCallableAliasLoweringError,
    MetalCallableLoweringError,
    MetalSizeofResolutionError,
    MetalStageEntryArrayResourceError,
    MetalStaticConstantResolutionError,
    MetalStructMethodCallResolutionError,
    MetalTemplateArgumentResolutionError,
    MetalToCrossGLConverter,
    MetalWideVectorLoweringError,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.translator.ast import ResourceMemoryQualifierNode
from crosstl.translator.codegen.directx_codegen import (
    HLSLCodeGen as TranslatorHLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
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


def convert_without_preprocessing(code: str, file_path=None) -> str:
    tokens = MetalLexer(code, preprocess=False).tokenize()
    ast = MetalParser(tokens, file_path=file_path).parse()
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

    assert "Us us" in generated
    assert "sum(us)" in generated
    assert "post..." not in generated
    assert parse_crossgl(generated) is not None


def test_codegen_keeps_dependent_enable_if_return_type_from_tinygrad_metal():
    # Reduced from tinygrad/tinygrad
    # extra/thunder/metal/include/ops/group/memory/tile/shared_to_register.metal.
    code = """
    template<typename RT, typename ST>
    METAL_FUNC static typename metal::enable_if<
        ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(),
        void>::type
    load(thread RT &dst, threadgroup const ST &src, const int threadIdx) {
        return;
    }
    """
    generated = convert(code)

    assert (
        "type load(inout thread RT dst, threadgroup ST& src, int threadIdx)"
        in generated
    )
    assert "return;" in generated
    assert parse_crossgl(generated) is not None


def test_codegen_preserves_hex_float_literals_from_msl_cxx_base():
    # MSL is C++14 based, so hexadecimal floating literals use a p/P exponent.
    code = """
    kernel void main(device float* out [[buffer(0)]]) {
        float tiny = 0x1.0p-14f;
        half one = 0x1p+0h;
        float separated = 0x1'0.8p+2f;
        out[0] = tiny + float(one) + separated;
    }
    """
    generated = convert(code)

    assert "float tiny = 0x1.0p-14f;" in generated
    assert "float16 one = 0x1p+0h;" in generated
    assert "float separated = 0x10.8p+2f;" in generated
    assert parse_crossgl(generated) is not None


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


def test_codegen_fragment_front_facing_attribute_uses_parseable_crossgl_builtin():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(bool isFrontFace [[front_facing]]) {
        return isFrontFace ? float4(1.0) : float4(0.0);
    }
    """
    result = convert(code)

    assert "bool isFrontFace @gl_FrontFacing" in result
    assert "@gl_IsFrontFace" not in result
    assert "@front_facing" not in result
    assert parse_crossgl(result) is not None


def test_codegen_fragment_sample_mask_parameter_uses_input_builtin():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(uint coverage [[sample_mask]]) {
        return float4(float(coverage), 0.0, 0.0, 1.0);
    }
    """
    result = convert(code)

    assert "uint coverage @gl_SampleMaskIn" in result
    assert re.search(r"uint coverage @gl_SampleMask(?!In)\b", result) is None
    assert parse_crossgl(result) is not None

    regenerated = MetalCodeGen().generate(parse_crossgl(result))

    assert "uint coverage [[sample_mask]]" in regenerated


def test_codegen_fragment_barycentric_attribute_uses_canonical_builtin():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct FragmentInput {
        float4 position [[position]];
        float3 barycentricCoords [[barycentric_coord, center_no_perspective]];
    };

    fragment float4 fragment_main(FragmentInput input [[stage_in]]) {
        return float4(input.barycentricCoords, 1.0);
    }
    """
    result = convert(code)

    assert "vec3 barycentricCoords @gl_BaryCoordNoPerspEXT;" in result
    assert "@barycentric_coord" not in result
    assert "@center_no_perspective" not in result
    assert parse_crossgl(result) is not None

    regenerated = MetalCodeGen().generate(parse_crossgl(result))

    assert (
        "float3 barycentricCoords [[barycentric_coord]] "
        "[[center_no_perspective]];" in regenerated
    )


def test_codegen_trailing_return_type_helper_from_msl_cxx14_grammar():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    auto remap(float value) -> float {
        return value * 2.0;
    }

    fragment float4 fragment_main() {
        return float4(remap(1.0));
    }
    """
    result = convert(code)

    assert "float remap(float value)" in result
    assert "auto remap" not in result
    assert "return vec4(remap(1.0));" in result
    assert parse_crossgl(result) is not None


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


def test_codegen_normalizes_64_bit_integer_vectors_for_opengl():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    ulong2 unsigned_pair(ulong first, ulong second) {
        return ulong2(first, second);
    }

    long3 signed_triple(long value) {
        return long3(value, value + 1, value + 2);
    }
    """

    crossgl = convert(code)
    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "u64vec2 unsigned_pair(uint64 first, uint64 second)" in crossgl
    assert "return u64vec2(first, second);" in crossgl
    assert "i64vec3 signed_triple(int64 value)" in crossgl
    assert "return i64vec3(value, value + 1, value + 2);" in crossgl
    assert "u64vec2 unsigned_pair(uint64_t first, uint64_t second)" in glsl
    assert "return u64vec2(first, second);" in glsl
    assert "i64vec3 signed_triple(int64_t value)" in glsl
    assert "return i64vec3(value, (value + 1), (value + 2));" in glsl
    assert "#extension GL_ARB_gpu_shader_int64 : require" in glsl
    assert "ulong2" not in crossgl
    assert "long3" not in crossgl


def test_codegen_xhalf_vectors_lower_before_opengl_generation():
    code = """
    struct Camera {
        float4x4 invViewMatrix;
    };

    struct Input {
        float3 position;
    };

    struct Output {
        xhalf3 viewDir;
    };

    vertex Output main_vertex(Input in [[stage_in]],
                              constant Camera& camera [[buffer(0)]]) {
        Output out;
        out.viewDir = (xhalf3)normalize(camera.invViewMatrix[3].xyz - in.position);
        return out;
    }
    """

    crossgl = convert(code)
    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "f16vec3 viewDir;" in crossgl
    assert "f16vec3(normalize" in crossgl
    assert "xhalf" not in crossgl
    assert "out vec3 viewDir;" in glsl
    assert "viewDir = vec3(normalize" in glsl
    assert "xhalf" not in glsl
    assert "f16vec3" not in glsl


def test_codegen_stage_input_reference_struct_lowers_to_flat_opengl_inputs(tmp_path):
    code = """
    struct VertexInput {
        float3 position [[attribute(0)]];
        float3 normal [[attribute(1)]];
    };

    vertex float4 main_vertex(const VertexInput& input [[stage_in]]) {
        return float4(input.position + input.normal, 1.0);
    }
    """

    crossgl = convert(code)
    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "in VertexInput" not in glsl
    assert "in struct" not in glsl
    assert "struct VertexInput" not in glsl
    assert "layout(location = 0) in vec3 position;" in glsl
    assert "layout(location = 1) in vec3 normal;" in glsl
    assert "gl_Position = vec4((position + normal), 1.0);" in glsl

    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "metal_stage_input_reference.vert"
    source.write_text(glsl, encoding="utf-8")
    subprocess.run([glslang, "-S", "vert", str(source)], check=True)


def test_codegen_packed_integer_vertex_storage_types_do_not_leak_metal_names():
    code = """
    struct VertexInput {
        packed_float3 position;
        packed_uchar4 color;
        packed_short2 joints;
        metal::packed_ushort2 uv;
    };

    packed_uchar4 makeColor(VertexInput in) {
        return packed_uchar4(in.color);
    }

    metal::packed_ushort2 makeUv(VertexInput in) {
        return metal::packed_ushort2(in.uv);
    }
    """
    crossgl = convert(code)

    assert "vec3 position;" in crossgl
    assert "u8vec4 color;" in crossgl
    assert "i16vec2 joints;" in crossgl
    assert "u16vec2 uv;" in crossgl
    assert "u8vec4 makeColor(VertexInput in_)" in crossgl
    assert "return u8vec4(in_.color);" in crossgl
    assert "u16vec2 makeUv(VertexInput in_)" in crossgl
    assert "return u16vec2(in_.uv);" in crossgl
    assert "metal_u3a" not in crossgl
    for raw_type in (
        "packed_uchar4",
        "packed_short2",
        "packed_ushort2",
    ):
        assert raw_type not in crossgl
    parse_crossgl(crossgl)


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


def test_codegen_texture2d_array_alias_preserves_sample_slice():
    code = """
    using ColorArray = texture2d_array<float>;

    float4 sampleSlice(ColorArray tex, sampler s, float2 uv) {
        return tex.sample(s, uv, 2);
    }
    """
    crossgl = convert(code)

    assert "typedef sampler2DArray ColorArray;" not in crossgl
    assert "vec4 sampleSlice(sampler2DArray tex, sampler s, vec2 uv)" in crossgl
    assert "texture(tex, s, vec3(uv, 2))" in crossgl
    assert "textureLod(tex, s, uv, 2)" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_resource_alias_references_preserve_array_family_lowering():
    code = """
    using CubeArray = texturecube_array<float>;
    typedef texture_buffer<uint> UIntBuffer;
    using DepthArray = depth2d_array<float>;

    float4 sampleCube(thread CubeArray& tex, sampler s, float3 dir, uint layer) {
        return tex.sample(s, dir, layer);
    }

    uint4 readBuffer(thread UIntBuffer& tex, uint index) {
        return tex.read(index);
    }

    float compareDepth(thread DepthArray& tex,
                       sampler s,
                       float2 uv,
                       uint layer,
                       float depth) {
        return tex.sample_compare(s, uv, layer, depth);
    }
    """
    crossgl = convert(code)

    assert "typedef samplerCubeArray CubeArray;" not in crossgl
    assert "typedef usamplerBuffer UIntBuffer;" not in crossgl
    assert "typedef sampler2DArrayShadow DepthArray;" not in crossgl
    assert "texture(tex, s, vec4(dir, layer))" in crossgl
    assert "texelFetch(tex, index)" in crossgl
    assert "texelFetch(tex, index, 0)" not in crossgl
    assert "textureCompare(tex, s, vec3(uv, layer), depth)" in crossgl
    assert "textureLod(tex, s, dir, layer)" not in crossgl
    assert "textureCompareOffset(tex, s, uv, layer, depth)" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_storage_texture_alias_uses_image_read_write():
    code = """
    typedef texture2d<float, access::read_write> RWColor;

    float4 readWrite(RWColor tex, uint2 p, float4 value) {
        tex.write(value, p);
        return tex.read(p);
    }
    """
    crossgl = convert(code)

    assert "typedef image2D RWColor;" not in crossgl
    assert "vec4 readWrite(image2D tex @readwrite, uvec2 p, vec4 value)" in crossgl
    assert "imageStore(tex, p, value);" in crossgl
    assert "return imageLoad(tex, p);" in crossgl
    assert "unsupported Metal sampled texture write" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_scoped_access_mode_expression_from_blender_shader():
    # Reduced from:
    # Repo: https://github.com/blender/blender
    # Commit: 5711482b0608efd82006c6d9e230cf0b3e657cc1
    # Path: source/blender/draw/engines/eevee/shaders/eevee_depth_of_field_resolve.bsl.hh
    code = """
    kernel void copy_read(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          uint2 tid [[thread_position_in_grid]]) {
        auto readMode = access::read;
        auto readWriteMode = metal::access::read_write;
        float4 value = src.read(tid);
        dst.write(value, tid);
    }
    """
    crossgl = convert(code)

    assert "auto readMode = access_u3a_u3aread;" in crossgl
    assert "auto readWriteMode = metal_u3a_u3aaccess_u3a_u3aread_write;" in crossgl
    assert "image2D src @texture(0) @readonly" in crossgl
    assert "image2D dst @texture(1) @writeonly" in crossgl
    assert "imageLoad(src, tid)" in crossgl
    assert "imageStore(dst, tid, value);" in crossgl
    parse_crossgl(crossgl)


def test_codegen_access_qualified_texture_buffer_uses_image_buffer():
    code = """
    typedef texture_buffer<uint, access::read_write> RWCounterBuffer;

    float4 readLine(texture_buffer<float, access::read> line, uint index) {
        return line.read(index);
    }

    void writeSigned(texture_buffer<int, access::write> outLine,
                     uint index,
                     int4 value) {
        outLine.write(value, index);
    }

    uint4 updateCounter(RWCounterBuffer counters, uint index, uint4 value) {
        counters.write(value, index);
        return counters.read(index);
    }
    """
    crossgl = convert(code)

    assert "typedef uimageBuffer RWCounterBuffer;" not in crossgl
    assert "vec4 readLine(imageBuffer line @readonly, uint index)" in crossgl
    assert "void writeSigned(iimageBuffer outLine @writeonly" in crossgl
    assert (
        "uvec4 updateCounter(uimageBuffer counters @readwrite, "
        "uint index, uvec4 value)"
    ) in crossgl
    assert "return imageLoad(line, index);" in crossgl
    assert "imageStore(outLine, index, value);" in crossgl
    assert "imageStore(counters, index, value);" in crossgl
    assert "return imageLoad(counters, index);" in crossgl
    assert "unsupported Metal sampled texture write" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_integer_texture_element_aliases_preserve_sampler_and_image_family():
    code = """
    using UShortColor = texture2d<ushort>;
    typedef metal::texture2d<short, metal::access::read_write> SignedStorage;

    ushort4 sampleUnsigned(UShortColor tex, sampler s, float2 uv) {
        return tex.sample(s, uv);
    }

    short4 readSigned(SignedStorage tex, uint2 p) {
        return tex.read(p);
    }

    void writeUnsigned(texture2d<ushort, access::write> tex,
                       uint2 p,
                       ushort4 value) {
        tex.write(value, p);
    }
    """
    crossgl = convert(code)

    assert "typedef usampler2D UShortColor;" not in crossgl
    assert "typedef iimage2D SignedStorage;" not in crossgl
    assert "u16vec4 sampleUnsigned(usampler2D tex, sampler s, vec2 uv)" in crossgl
    assert "i16vec4 readSigned(iimage2D tex @readwrite, uvec2 p)" in crossgl
    assert "void writeUnsigned(uimage2D tex @writeonly" in crossgl
    assert "texture(tex, s, uv)" in crossgl
    assert "return imageLoad(tex, p);" in crossgl
    assert "imageStore(tex, p, value);" in crossgl
    assert "texture2d<ushort>" not in crossgl
    assert "i16vec4 readSigned(image2D tex @readwrite" not in crossgl
    parse_crossgl(crossgl)


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


def test_codegen_uint_vertex_id_roundtrips_to_opengl_builtin_int():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex {
        float4 position [[position]];
        float4 color;
    };

    vertex Vertex vertex_main(const device Vertex *vertices [[buffer(0)]],
                              uint vid [[vertex_id]]) {
        return vertices[vid];
    }

    fragment float4 fragment_main(Vertex inVertex [[stage_in]]) {
        return inVertex.color;
    }
    """
    crossgl = convert(code)

    assert "int vid @gl_VertexID" in crossgl
    assert "uint vid @gl_VertexID" not in crossgl
    ast = parse_crossgl(crossgl)
    generated = GLSLCodeGen().generate(ast)

    assert "uint vid" not in generated
    assert "verticesBuffer" in generated


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


def test_codegen_drops_scoped_vendor_function_attribute_from_spirv_cross_quantize():
    # Reduced from:
    # Repo: https://github.com/KhronosGroup/SPIRV-Cross
    # Commit: 146679ff8255a6068518685599d7fb8761d1b570
    # Path: reference/shaders-msl/asm/comp/quantize.asm.comp
    code = """
    #include <metal_stdlib>
    using namespace metal;

    template<typename F>
    [[clang::optnone]]
    F spvQuantizeToF16(F fval) {
        return F(fval);
    }

    kernel void main0(device float& out [[buffer(0)]]) {
        out = spvQuantizeToF16(out);
    }
    """
    crossgl = convert(code)

    assert "spvQuantizeToF16" in crossgl
    assert "clang::optnone" not in crossgl
    assert parse_crossgl(crossgl) is not None


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

    assert "void spvArrayCopy(inout thread T[N] dst, thread T[N] src)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_writable_c_array_parameter_preserves_aliasing():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    METAL_FUNC void fill(
        thread float values[4],
        thread const float source[4]) {
        values[0] = source[0];
    }
    """

    crossgl = convert(code)

    assert "void fill(inout thread float[4] values, thread float[4] source)" in crossgl
    assert (
        "void fill(inout float values[4], float source[4])"
        in TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    )
    assert (
        "void fill(inout float values[4], float source[4])"
        in GLSLCodeGen().generate(parse_crossgl(crossgl))
    )
    assert (
        "void fill(thread float values[4], thread float source[4])"
        in MetalCodeGen().generate(parse_crossgl(crossgl))
    )


def test_codegen_struct_method_receiver_directions_reach_native_targets():
    source = """
    struct NestedState { int value; };

    struct State {
        int scalar;
        int values[2];
        NestedState nested;

        void mutate(int amount) {
            scalar += amount;
            values[1] = amount;
            nested.value += amount;
        }

        int total() const {
            return scalar + values[1] + nested.value;
        }
    };

    int apply(thread State& state) {
        state.mutate(3);
        return state.total();
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(source))
    ast = parse_crossgl(crossgl)
    generated_targets = (
        crossgl,
        TranslatorHLSLCodeGen().generate(ast),
        GLSLCodeGen().generate(ast),
    )

    assert "int State__total(in thread State& self)" in normalize(crossgl)

    for generated in generated_targets:
        generated = normalize(generated)
        mutable_signature = re.search(r"\bvoid State__mutate\s*\(([^)]*)\)", generated)
        const_signature = re.search(r"\bint State__total\s*\(([^)]*)\)", generated)

        assert mutable_signature is not None
        assert "inout" in mutable_signature.group(1).split(",", 1)[0].split()
        assert const_signature is not None
        assert "inout" not in const_signature.group(1).split(",", 1)[0].split()
        assert "self.scalar += amount;" in generated
        assert "self.values[1] = amount;" in generated
        assert "self.nested.value += amount;" in generated


def test_codegen_rebinds_lowered_struct_sibling_overload_with_resources():
    source = """
    struct ReadWriter_float2_float2 {
        const device float* input;
        device float* output;
        int bias;

        ReadWriter_float2_float2(
            const device float* input_, device float* output_, int bias_)
            : input(input_), output(output_), bias(bias_) {}

        float post_in(float elem) const {
            return elem + input[0] + float(bias);
        }

        int post_in(int elem) const {
            return elem + int(input[0]) + bias;
        }

        float load(float elem) const {
            return post_in(elem);
        }
    };

    kernel void k(
        const device float* input [[buffer(0)]],
        device float* output [[buffer(1)]]) {
        ReadWriter_float2_float2 rw(input, output, 2);
        output[0] = rw.load(1.0f);
    }
    """

    lowered = MetalPreprocessor().preprocess(source)
    assert "return post_in(elem);" in lowered

    crossgl = convert(lowered)
    load_body = crossgl.rsplit("ReadWriter_float2_float2__load", 1)[1].split("}", 1)[0]

    assert (
        "ReadWriter_float2_float2__post_in(self, crosstl_ptr_input, "
        "crosstl_ptr_output, elem)" in load_body
    )
    assert "return post_in(elem);" not in load_body
    assert (
        "float ReadWriter_float2_float2__load(in thread "
        "ReadWriter_float2_float2& self" in normalize(crossgl)
    )


def test_codegen_does_not_rebind_global_call_from_lowered_struct_method():
    lowered = """
    struct Reader { int bias; };

    float post_in(float elem) {
        return elem * 2.0f;
    }

    float Reader__post_in(thread const Reader& self, float elem) {
        return elem + float(self.bias);
    }

    float Reader__load(thread const Reader& self, float elem) {
        return post_in(elem);
    }
    """

    crossgl = convert(lowered)
    load_body = crossgl.split("Reader__load", 1)[1].split("}", 1)[0]

    assert "return post_in(elem);" in load_body
    assert "Reader__post_in(self" not in load_body


def test_codegen_does_not_rebind_lexically_shadowed_callable():
    lowered = """
    struct Reader { int bias; };
    struct Callable { int tag; };

    float Reader__post_in(thread const Reader& self, float elem) {
        return elem + float(self.bias);
    }

    float Reader__load(
        thread const Reader& self, Callable post_in, float elem) {
        return post_in(elem);
    }
    """

    crossgl = convert(lowered)
    load_body = crossgl.split("Reader__load", 1)[1].split("}", 1)[0]

    assert "return post_in(elem);" in load_body
    assert "Reader__post_in(self" not in load_body


def test_codegen_fails_closed_for_ambiguous_lowered_sibling_overload():
    lowered = """
    struct Reader { int bias; };

    long Reader__post_in(thread const Reader& self, long elem) {
        return elem;
    }

    int64_t Reader__post_in(thread const Reader& self, int64_t elem) {
        return elem;
    }

    long Reader__load(thread const Reader& self, long elem) {
        return post_in(elem);
    }
    """

    with pytest.raises(MetalStructMethodCallResolutionError) as exc_info:
        convert(lowered)

    error = exc_info.value
    assert error.owner == "Reader"
    assert error.method_name == "post_in"
    assert error.argument_types == ("int64",)
    assert error.reason == "multiple exact overloads remain after type matching"
    assert error.candidates == (
        "Reader__post_in(long)",
        "Reader__post_in(int64_t)",
    )
    assert "qualify the intended call" in str(error)


def test_codegen_rebinds_nested_lowered_sibling_calls():
    source = """
    struct Reader {
        const device float* input;

        Reader(const device float* input_) : input(input_) {}

        float post_in(float elem) const {
            return elem + input[0];
        }

        int post_in(int elem) const {
            return elem + int(input[0]);
        }

        float load(float elem) const {
            return float(post_in(post_in(int(elem)))) + post_in(elem);
        }
    };

    kernel void k(
        const device float* input [[buffer(0)]],
        device float* output [[buffer(1)]]) {
        Reader reader(input);
        output[0] = reader.load(1.0f);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(source))
    load_body = normalize(crossgl.rsplit("Reader__load", 1)[1].split("}", 1)[0])

    assert (
        "Reader__post_in(self, crosstl_ptr_input, "
        "Reader__post_in(self, crosstl_ptr_input, int(elem)))" in load_body
    )
    assert "Reader__post_in(self, crosstl_ptr_input, elem)" in load_body
    assert "post_in(post_in" not in load_body


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


def test_codegen_depth_array_sample_compare_bias_option_from_msl_spec_is_diagnostic():
    # Provenance: Apple Metal Shading Language Specification, section 6.13.10
    # "2D depth texture array", version 2025-10-23. The documented overload is
    # sample_compare(sampler, float2 coord, uint array, float compare, lod_options).
    code = """
    float sampleBiasedShadow(depth2d_array<float> shadowMap,
                             sampler shadowSampler,
                             float2 uv,
                             uint layer,
                             float compare,
                             float lodBias) {
        return shadowMap.sample_compare(
            shadowSampler, uv, layer, compare, bias(lodBias));
    }
    """
    crossgl = convert(code)

    assert (
        "0.0 /* unsupported Metal depth compare lod option: bias on shadowMap */"
        in crossgl
    )
    assert "textureCompare(shadowMap, shadowSampler, uv, layer, compare" not in crossgl
    assert "bias(lodBias)" not in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "return 0.0;" in metal


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


def test_codegen_texture_sample_namespace_qualified_options_roundtrip():
    code = """
    float4 sampleScopedOptions(texture2d<float> tex,
                               sampler samp,
                               float2 uv,
                               float lod,
                               float biasValue,
                               float2 ddx,
                               float2 ddy,
                               float firstTailMip) {
        float4 mip = tex.sample(samp, uv, metal::level(lod));
        float4 biased = tex.sample(samp, uv, metal::bias(biasValue));
        float4 gradient = tex.sample(samp, uv, metal::gradient2d(ddx, ddy));
        float4 clamped = tex.sample(samp, uv, metal::min_lod_clamp(firstTailMip));
        return mip + biased + gradient + clamped;
    }
    """
    crossgl = convert(code)

    assert "textureLod(tex, samp, uv, lod)" in crossgl
    assert "texture(tex, samp, uv, biasValue)" in crossgl
    assert "textureGrad(tex, samp, uv, ddx, ddy)" in crossgl
    assert "textureMinLodClamp(tex, samp, uv, firstTailMip)" in crossgl
    assert "metal_u3a_u3a" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample(samp, uv, level(lod))" in metal
    assert "tex.sample(samp, uv, bias(biasValue))" in metal
    assert "tex.sample(samp, uv, gradient2d(ddx, ddy))" in metal
    assert "tex.sample(samp, uv, min_lod_clamp(firstTailMip))" in metal


def test_codegen_msl_relational_namespace_intrinsics_import_to_crossgl():
    # Reduced from Metal Shading Language Specification table 6.3 relational
    # functions, where these intrinsics are declared in the metal namespace.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float classify(float value, float3 values) {
        bool nanValue = metal::isnan(value);
        bool infValue = metal::isinf(value);
        bool finiteValue = metal::isfinite(value);
        bool3 nanMask = metal::isnan(values);
        return (nanValue || infValue || !finiteValue || any(nanMask)) ? 1.0 : 0.0;
    }
    """
    crossgl = convert(code)

    assert "bool nanValue = isnan(value);" in crossgl
    assert "bool infValue = isinf(value);" in crossgl
    assert "bool finiteValue = isfinite(value);" in crossgl
    assert "bvec3 nanMask = isnan(values);" in crossgl
    assert "metal_u3a_u3a" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_msl_relational_reductions_and_ordering_import_to_crossgl():
    # Apple Metal Shading Language Specification, relational functions:
    # all/any, isnormal, isordered, and isunordered are defined in <metal_relational>.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    bool classify(float3 left, float3 right, bool3 mask) {
        bool everyMask = metal::all(mask);
        bool anyMask = metal::any(mask);
        bool3 normalMask = metal::isnormal(left);
        bool3 orderedMask = metal::isordered(left, right);
        bool3 unorderedMask = metal::isunordered(left, right);
        return everyMask
            || anyMask
            || metal::any(normalMask)
            || metal::all(orderedMask)
            || metal::any(unorderedMask);
    }
    """
    crossgl = convert(code)

    assert "bool everyMask = all(mask);" in crossgl
    assert "bool anyMask = any(mask);" in crossgl
    assert "bvec3 normalMask = isnormal(left);" in crossgl
    assert "bvec3 orderedMask = isordered(left, right);" in crossgl
    assert "bvec3 unorderedMask = isunordered(left, right);" in crossgl
    assert "|| any(normalMask)" in crossgl
    assert "|| all(orderedMask)" in crossgl
    assert "|| any(unorderedMask)" in crossgl
    assert "metal_u3a_u3a" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_multi_declarator_struct_members_from_cxx_msl_headers():
    code = """
    struct KernelParams {
        float scale = 1.0, bias = 0.0;
        uint2 extent, stride;
    };

    float2 apply(KernelParams params) {
        return float2(params.scale, params.bias)
            + float2(params.extent.x, params.stride.y);
    }
    """
    crossgl = convert(code)

    assert "float scale;" in crossgl
    assert "float bias;" in crossgl
    assert "uvec2 extent;" in crossgl
    assert "uvec2 stride;" in crossgl
    assert "vec2 apply(KernelParams params)" in crossgl
    parse_crossgl(crossgl)


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


def test_codegen_texture_gather_component_selector_from_msl_spec():
    # Metal texture gather overloads accept component::x/y/z/w selectors.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float4 gatherRed(texture2d<float> tex, sampler samp, float2 uv) {
        return tex.gather(samp, uv, component::x);
    }
    """
    crossgl = convert(code)

    assert "textureGather(tex, samp, uv, 0)" in crossgl
    assert "component_u3a_u3ax" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_texture_gather_offset_and_array_slice_overloads_roundtrip():
    # The MSL gather overloads pass array slices separately from the coordinate.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float4 gatherOffset(texture2d<float> tex,
                        sampler samp,
                        float2 uv,
                        int2 offset) {
        return tex.gather(samp, uv, offset, component::z);
    }

    float4 gatherArray(texture2d_array<float> tex,
                       sampler samp,
                       float2 uv,
                       uint layer,
                       int2 offset) {
        return tex.gather(samp, uv, layer, offset, component::w);
    }

    float4 gatherCubeArray(texturecube_array<float> tex,
                           sampler samp,
                           float3 dir,
                           uint layer) {
        return tex.gather(samp, dir, layer, component::y);
    }
    """
    crossgl = convert(code)

    assert "textureGatherOffset(tex, samp, uv, offset, 2)" in crossgl
    assert "textureGatherOffset(tex, samp, vec3(uv, layer), offset, 3)" in crossgl
    assert "textureGather(tex, samp, vec4(dir, layer), 1)" in crossgl
    assert "textureGather(tex, samp, uv, layer" not in crossgl
    assert "component_u3a_u3a" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.gather(samp, uv, offset, component::z)" in metal
    assert (
        "tex.gather(samp, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), offset, component::w)" in metal
    )
    assert (
        "tex.gather(samp, (float4(dir, layer)).xyz, "
        "uint((float4(dir, layer)).w), component::y)" in metal
    )
    assert "textureGather" not in metal


def test_codegen_depth_gather_compare_array_slice_overloads_roundtrip():
    # MSL depth array gather_compare overloads carry array slice before compare.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float4 gatherDepthArray(depth2d_array<float> shadowMap,
                            sampler shadowSampler,
                            float2 uv,
                            uint layer,
                            float compare,
                            int2 offset) {
        float4 base = shadowMap.gather_compare(
            shadowSampler, uv, layer, compare);
        float4 shifted = shadowMap.gather_compare(
            shadowSampler, uv, layer, compare, offset);
        return base + shifted;
    }

    float4 gatherDepthCubeArray(depthcube_array<float> shadowMap,
                                sampler shadowSampler,
                                float3 dir,
                                uint layer,
                                float compare) {
        return shadowMap.gather_compare(shadowSampler, dir, layer, compare);
    }
    """
    crossgl = convert(code)

    assert (
        "textureGatherCompare(shadowMap, shadowSampler, vec3(uv, layer), compare)"
        in crossgl
    )
    assert (
        "textureGatherCompareOffset("
        "shadowMap, shadowSampler, vec3(uv, layer), compare, offset)" in crossgl
    )
    assert (
        "textureGatherCompare("
        "shadowMap, shadowSampler, vec4(dir, layer), compare)" in crossgl
    )
    assert "textureGatherCompare(shadowMap, shadowSampler, uv, layer" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert (
        "shadowMap.gather_compare("
        "shadowSampler, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), compare)" in metal
    )
    assert (
        "shadowMap.gather_compare("
        "shadowSampler, (float3(uv, layer)).xy, "
        "uint((float3(uv, layer)).z), compare, offset)" in metal
    )
    assert (
        "shadowMap.gather_compare("
        "shadowSampler, (float4(dir, layer)).xyz, "
        "uint((float4(dir, layer)).w), compare)" in metal
    )
    assert "textureGatherCompare(" not in metal


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

    assert "vec4 pixPos @gl_FragCoord" in crossgl
    assert "@gl_Position" not in crossgl
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


def test_codegen_comma_separated_pointer_declarators_keep_own_suffixes():
    code = """
    void main() {
        thread float *a, *b;
        thread float *c, d;
    }
    """
    crossgl = convert(code)

    assert "thread float* a;" in crossgl
    assert "thread float* b;" in crossgl
    assert "thread float* c;" in crossgl
    assert "thread float d;" in crossgl
    assert "thread float** b;" not in crossgl
    assert "thread float* d;" not in crossgl
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
    assert "float read_seed(inout thread Loki rng)" in crossgl
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
    assert "uint lcg(inout thread uint prev)" in crossgl
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
                             uint3 tid [[thread_position_in_grid]]) {
        float value = input[tid.x];
        data[tid.x] = value * 2.0;
    }
    """
    crossgl = convert(code)

    assert "RWStructuredBuffer<float> data @buffer(0)" in crossgl
    assert "StructuredBuffer<float> input @buffer(1)" in crossgl
    assert "float value = buffer_load(input, tid.x);" in crossgl
    assert "buffer_store(data, tid.x, value * 2.0);" in crossgl
    assert "data[tid.x]" not in crossgl
    assert "input[tid.x]" not in crossgl

    ast = parse_crossgl(crossgl)
    assert ast is not None

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "RWStructuredBuffer<float> data" in hlsl
    assert "StructuredBuffer<float> input" in hlsl
    assert "float value = input.Load(tid.x);" in hlsl
    assert "data[tid.x] = (value * 2.0);" in hlsl
    assert "data.Store(" not in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "kernel void compute_main(device float* data" in metal
    assert "const device float* input" in metal
    assert "float value = input[tid.x];" in metal
    assert "data[tid.x] = value * 2.0;" in metal


def test_codegen_stage_entry_arrays_lower_to_non_conflicting_resources(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;

    void copy_pair(constant const int offsets[2], device float scratch[2]) {
        scratch[offsets[0]] = scratch[offsets[1]];
    }

    kernel void array_resources(
        constant const int strides[3],
        device float values[],
        device const float* source [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        uint index = uint(strides[gid % 3]);
        values[gid] = source[index] + values[gid];
    }
    """
    crossgl = convert(code)

    assert "StructuredBuffer<int> strides @buffer(1)" in crossgl
    assert "RWStructuredBuffer<float> values @buffer(2)" in crossgl
    assert "StructuredBuffer<float> source @buffer(0)" in crossgl
    assert "uint index = uint(buffer_load(strides, gid % 3));" in crossgl
    assert (
        "buffer_store(values, gid, buffer_load(source, index) + "
        "buffer_load(values, gid));"
    ) in crossgl
    assert (
        "void copy_pair(constant int[2] offsets, " "inout device float[2] scratch)"
    ) in crossgl
    assert "scratch[offsets[0]] = scratch[offsets[1]];" in crossgl
    assert "StructuredBuffer<int> offsets" not in crossgl
    assert "RWStructuredBuffer<float> scratch" not in crossgl

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    spirv = VulkanSPIRVCodeGen().generate(ast)

    assert "StructuredBuffer<int> strides : register(t1);" in hlsl
    assert "RWStructuredBuffer<float> values : register(u2);" in hlsl
    assert "StructuredBuffer<float> source : register(t0);" in hlsl
    assert (
        "void copy_pair(StructuredBuffer<int> offsets, "
        "inout RWStructuredBuffer<float> scratch)"
    ) in hlsl
    assert "const int*" not in hlsl
    assert "float*" not in hlsl
    assert "uint index = uint(strides.Load((gid % 3)));" in hlsl
    assert "values[gid] = (source.Load(index) + values.Load(gid));" in hlsl

    assert (
        "layout(std430, binding = 1) readonly buffer stridesBuffer "
        "{ int strides[]; };"
    ) in glsl
    assert (
        "layout(std430, binding = 2) buffer valuesBuffer { float values[]; };" in glsl
    )
    assert (
        "layout(std430, binding = 0) readonly buffer sourceBuffer "
        "{ float source[]; };"
    ) in glsl
    assert "void copy_pair(int offsets[2], inout float scratch[2])" in glsl
    assert "uint index = uint(strides[(gid % 3)]);" in glsl
    assert "values[gid] = (source[index] + values[gid]);" in glsl

    for resource_name, binding in (("source", 0), ("strides", 1), ("values", 2)):
        resource_id_match = re.search(rf'OpName (%\d+) "{resource_name}"', spirv)
        assert resource_id_match is not None
        resource_id = resource_id_match.group(1)
        assert f"OpDecorate {resource_id} DescriptorSet 0" in spirv
        assert f"OpDecorate {resource_id} Binding {binding}" in spirv
    assert spirv.count(" BufferBlock") == 3
    assert spirv.count(" NonWritable") == 2
    assert "OpTypeArray" in spirv
    assert "WARNING" not in spirv

    glslang = shutil.which("glslangValidator")
    dxc = shutil.which("dxc")
    hlsl_path = tmp_path / "stage-entry-arrays.hlsl"
    hlsl_path.write_text(hlsl, encoding="utf-8")
    if dxc is not None:
        subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "stage-entry-arrays.dxil"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    elif glslang is not None:
        subprocess.run(
            [
                glslang,
                "-D",
                "-V",
                "-S",
                "comp",
                "-e",
                "CSMain",
                str(hlsl_path),
                "-o",
                str(tmp_path / "stage-entry-arrays-hlsl.spv"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    if glslang is not None:
        glsl_path = tmp_path / "stage-entry-arrays.comp"
        glsl_path.write_text(glsl, encoding="utf-8")
        subprocess.run(
            [glslang, "-S", "comp", str(glsl_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        assembly_path = tmp_path / "stage-entry-arrays.spvasm"
        binary_path = tmp_path / "stage-entry-arrays.spv"
        assembly_path.write_text(spirv, encoding="utf-8")
        subprocess.run(
            [
                spirv_as,
                "--target-env",
                "vulkan1.1",
                str(assembly_path),
                "-o",
                str(binary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [spirv_val, "--target-env", "vulkan1.1", str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_codegen_reports_multidimensional_stage_entry_array_resource():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void invalid_array_resource(constant int values[2][3]) {
    }
    """

    with pytest.raises(MetalStageEntryArrayResourceError) as exc_info:
        convert(code)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-entry-array-resource-invalid"
    )
    assert diagnostic.missing_capabilities == (
        "metal.stage-entry-array-resource-lowering",
    )
    assert diagnostic.parameter_name == "values"
    assert diagnostic.array_dimensions == ("2", "3")
    assert diagnostic.reason == "multidimensional-parameter-array"


def test_codegen_address_of_device_buffer_element_preserves_lvalue():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    void load_one(device const float* src, thread float& dst) {
        dst = src[0];
    }

    kernel void repro(
        device const float* values [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        float result = 0.0f;
        load_one(&values[gid], result);
        output[gid] = result;
    }
    """

    crossgl = convert(code)

    assert "load_one((&values[gid]), result);" in crossgl
    assert "&buffer_load(values, gid)" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_readonly_device_helper_parameters_for_hlsl(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;

    void copy_one(const device float* src, device float* dst, uint index) {
        dst[index] = src[index];
    }

    kernel void copy_kernel(
        device const float* src [[buffer(0)]],
        device float* dst [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
        copy_one(src, dst, index);
    }
    """

    crossgl = convert(code)

    assert (
        "void copy_one(const device float* src, device float* dst, uint index)"
        in crossgl
    )
    assert "StructuredBuffer<float> src @buffer(0)" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert (
        "void copy_one(StructuredBuffer<float> src, "
        "RWStructuredBuffer<float> dst, uint index)" in hlsl
    )
    assert "StructuredBuffer<float> src : register(t0);" in hlsl
    assert "copy_one(src, dst, index);" in hlsl

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "readonly-device-helper.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_directx_codegen_lowers_native_metal_entry_buffer_parameters_to_resources():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void matmul(device float* A [[buffer(0)]],
                       constant float* B [[buffer(1)]],
                       device float* X [[buffer(2)]],
                       uint3 id [[thread_position_in_grid]]) {
        X[id.x] = A[id.x] + B[id.x];
    }
    """
    ast = parse_code(tokenize_code(code))

    hlsl = TranslatorHLSLCodeGen().generate(ast)

    assert "RWStructuredBuffer<float> A : register(u0);" in hlsl
    assert "StructuredBuffer<float> B : register(t1);" in hlsl
    assert "RWStructuredBuffer<float> X : register(u2);" in hlsl
    assert "void CSMain(uint3 id : SV_DispatchThreadID)" in hlsl
    assert "float* A" not in hlsl
    assert "float* B" not in hlsl
    assert "float* X" not in hlsl
    assert "thread_position_in_grid" not in hlsl


def test_glsl_codegen_lowers_native_metal_entry_buffer_parameters_to_resources():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        uint row_dim_x;
        uint col_dim_x;
        uint inner_dim;
    };

    kernel void matmul(constant Params* params [[buffer(0)]],
                       constant float* A [[buffer(1)]],
                       constant float* B [[buffer(2)]],
                       device float* X [[buffer(3)]]) {
        X[0] = A[0] + B[0];
    }
    """
    ast = parse_code(tokenize_code(code))

    glsl = GLSLCodeGen().generate(ast)

    assert "layout(std140, binding = 0) uniform Params" in glsl
    assert "uint row_dim_x;" in glsl
    assert "} params;" in glsl
    assert "layout(std430, binding = 1) readonly buffer ABuffer" in glsl
    assert "float A[];" in glsl
    assert "layout(std430, binding = 2) readonly buffer BBuffer" in glsl
    assert "float B[];" in glsl
    assert "layout(std430, binding = 3) buffer XBuffer" in glsl
    assert "float X[];" in glsl
    assert "void main()" in glsl
    assert "constant Params* params" not in glsl
    assert "constant float* A" not in glsl
    assert "constant float* B" not in glsl
    assert "device float* X" not in glsl


def test_translate_metal_constant_buffer_member_access_qualifies_glsl_uniform_block(
    tmp_path,
):
    metal = """
    #include <metal_stdlib>
    using namespace metal;

    struct Camera {
        float4x4 invViewMatrix;
    };

    struct VertexOut {
        float4 position [[position]];
        float3 viewDir;
    };

    vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                                 constant Camera& camera [[buffer(0)]]) {
        VertexOut out;
        out.position = float4(0.0, 0.0, 0.0, 1.0);
        out.viewDir = camera.invViewMatrix[3].xyz;
        return out;
    }
    """
    shader_path = tmp_path / "apple_mesh_viewdir.metal"
    shader_path.write_text(metal, encoding="utf-8")

    glsl = crosstl.translate(
        str(shader_path),
        backend="opengl",
        format_output=False,
    )

    assert "layout(std140, binding = 0) uniform Camera" in glsl
    assert "} camera;" in glsl
    assert "viewDir = camera.invViewMatrix[3].xyz;" in glsl
    assert "viewDir = invViewMatrix[3].xyz;" not in glsl

    glslang = shutil.which("glslangValidator")
    if glslang:
        glsl_path = tmp_path / "apple_mesh_viewdir.vert"
        glsl_path.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [glslang, "-S", "vert", str(glsl_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_codegen_mlx_multi_entry_opengl_resource_bindings_do_not_overlap():
    # Reduced from MLX-generated multi-entry Metal kernels where unrelated entry
    # parameters reuse names and Metal buffer indices across kernels.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void scaled_dot_product_attention(
        device float* out_ [[buffer(1)]],
        constant uint* bmask [[buffer(11)]],
        constant int* x_shape [[buffer(8)]],
        uint tid [[thread_position_in_grid]]) {
        out_[tid] = float(bmask[0] + uint(x_shape[0]));
    }

    kernel void quantized(
        device float* out_ [[buffer(2)]],
        constant uint* bmask [[buffer(13)]],
        constant float* input [[buffer(0)]],
        constant float* raders_b_q [[buffer(4)]],
        constant int* x_shape [[buffer(9)]],
        uint tid [[thread_position_in_grid]]) {
        out_[tid] = input[tid] + float(bmask[0]) + raders_b_q[0] + float(x_shape[0]);
    }

    kernel void fence(
        device float* timestamp [[buffer(0)]],
        constant float* w_q [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
        timestamp[tid] = w_q[0];
    }
    """
    crossgl = convert(code)
    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))

    buffer_bindings = [
        int(binding)
        for binding in re.findall(r"layout\(std430, binding = (\d+)\)", glsl)
    ]
    assert len(buffer_bindings) == len(set(buffer_bindings))
    assert {0, 1, 2, 4, 8, 9, 11, 13}.issubset(buffer_bindings)

    assert "layout(std430, binding = 3) buffer timestampBuffer" in glsl
    assert "layout(std430, binding = 5) readonly buffer w_qBuffer" in glsl
    assert "scaled_dot_product_attention_bmask[0]" not in glsl
    assert "quantized_bmask[0]" in glsl
    assert "quantized_out[tid]" in glsl
    assert "quantized_x_shape[0]" in glsl


def test_codegen_pointer_return_buffer_selector_reparses_from_compiler_fixture():
    # Reduced from local CrossGL-Compiler build artifact:
    # build/test-metal-storage-buffer-nonuniform-descriptor-array.cglb/backend/metal/
    # MetalStorageBufferNonUniformDescriptorArrayShader.metal.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    device float* cgl_select_compute_values(int descriptorIndex,
                                            device float* values_0,
                                            device float* values_1) {
        if (descriptorIndex < 0 || descriptorIndex >= 2) {
            return values_0;
        }
        switch (descriptorIndex) {
        case 0:
            return values_0;
        case 1:
            return values_1;
        default:
            return values_0;
        }
    }

    kernel void compute_main(device float* values_0 [[buffer(0)]],
                             device float* values_1 [[buffer(1)]],
                             device int* descriptors [[buffer(4)]]) {
        int descriptor = descriptors[0];
        float first = cgl_select_compute_values(
            descriptor, values_0, values_1)[0];
        values_0[1] = first;
    }
    """
    crossgl = convert(code)

    assert "RWStructuredBuffer<float> cgl_select_compute_values" in crossgl
    assert "return values_0;" in crossgl
    assert "return values_1;" in crossgl
    assert "/* Unhandled expression: ReturnNode */" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_struct_pointer_return_buffer_selector_reparses_from_compiler_fixture():
    # Reduced from local CrossGL-Compiler build artifact:
    # build/test-metal-mixed-resource-descriptor-array.cglb/backend/metal/
    # MixedResourceDescriptorArrayShader.metal.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Particle {
        float3 position;
        float mass;
    };

    device Particle* cgl_select_compute_particles(int descriptorIndex,
                                                  device Particle* particles_0,
                                                  device Particle* particles_1) {
        switch (descriptorIndex) {
        case 0:
            return particles_0;
        case 1:
            return particles_1;
        default:
            return particles_0;
        }
    }

    kernel void compute_main(device Particle* particles_0 [[buffer(0)]],
                             device Particle* particles_1 [[buffer(1)]]) {
        float mass = cgl_select_compute_particles(
            1, particles_0, particles_1)[0].mass;
        particles_0[1].mass = mass;
    }
    """
    crossgl = convert(code)

    assert "RWStructuredBuffer<Particle> cgl_select_compute_particles" in crossgl
    assert "return particles_0;" in crossgl
    assert "return particles_1;" in crossgl
    assert "/* Unhandled expression: ReturnNode */" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_buffer_pointer_typedef_resource_resolves_element_contract():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    typedef texture2d<float> ColorTexture;

    fragment float4 sample_alias(
        constant ColorTexture* textures [[buffer(0)]],
        sampler linearSampler [[sampler(0)]]) {
        return textures[0].sample(linearSampler, float2(0.5));
    }
    """
    crossgl = convert(code)

    assert "StructuredBuffer<sampler2D> textures @buffer(0)" in crossgl
    assert "StructuredBuffer<ColorTexture>" not in crossgl
    assert "texture(buffer_load(textures, 0), linearSampler, vec2(0.5))" in crossgl
    assert parse_crossgl(crossgl) is not None


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


@pytest.mark.parametrize(
    ("source_type", "target_type", "crossgl_source", "crossgl_target"),
    [
        pytest.param("uint64_t", "uint2", "uint64", "uvec2", id="uint64-to-uint2"),
        pytest.param("uint2", "uint64_t", "uvec2", "uint64", id="uint2-to-uint64"),
        pytest.param("double", "int2", "double", "ivec2", id="double-to-int2"),
    ],
)
def test_codegen_preserves_as_type_target_shape_for_equal_width_reshape(
    source_type, target_type, crossgl_source, crossgl_target
):
    # MLX materializes subgroup shuffles that carry 64-bit values through two
    # 32-bit lanes. A family-only asuint/asint alias inherits the source shape.
    code = f"""
    static inline {target_type} reshape_bits({source_type} value) {{
        return as_type<{target_type}>(value);
    }}
    """
    crossgl = convert(code)

    assert f"{crossgl_target} reshape_bits({crossgl_source} value)" in crossgl
    assert f"return as_type<{crossgl_target}>(value);" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_outer_bitcast_shape_for_shuffle_and_fill_call():
    code = """
    static inline uint64_t shuffle_and_fill(uint64_t value, uint64_t fill) {
        return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
            as_type<uint2>(value), as_type<uint2>(fill), ushort(1)));
    }
    """
    crossgl = convert(code)

    assert "return as_type<uint64>(" in crossgl
    assert crossgl.count("as_type<uvec2>") == 2
    assert "WaveShuffleAndFillUp(" in crossgl
    assert "simd_shuffle_and_fill_up" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_keeps_same_shape_as_type_uint_alias():
    code = """
    static inline uint uint_from_bits(float value) {
        return as_type<uint>(value);
    }
    """
    crossgl = convert(code)

    assert "return asuint(value);" in crossgl
    assert "as_type<uint>" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_lowers_msl_bit_builtins_from_apple_spec():
    # Apple Metal Shading Language Specification, "Integer Functions":
    # popcount(T x), reverse_bits(T x).
    code = """
    #include <metal_stdlib>
    using namespace metal;

    uint4 bitOps(uint4 mask, uint value) {
        uint4 counts = popcount(mask);
        uint reversed = metal::reverse_bits(value);
        return counts + uint4(reversed);
    }
    """
    crossgl = convert(code)

    assert "uvec4 counts = bitCount(mask);" in crossgl
    assert "uint reversed = bitfieldReverse(value);" in crossgl
    assert "popcount" not in crossgl
    assert "reverse_bits" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_user_defined_bit_builtin_names():
    code = """
    uint popcount(uint value) {
        return value + 1;
    }

    uint reverse_bits(uint value) {
        return value + 2;
    }

    uint callUserHelpers(uint value) {
        return popcount(value) + reverse_bits(value);
    }
    """
    crossgl = convert(code)

    assert "uint popcount(uint value)" in crossgl
    assert "uint reverse_bits(uint value)" in crossgl
    assert "return popcount(value) + reverse_bits(value);" in crossgl
    assert "bitCount" not in crossgl
    assert "bitfieldReverse" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_lowers_static_cast_from_apple_compute_sample():
    code = """
    kernel void process(uint2 gid [[thread_position_in_grid]]) {
        float2 p0 = static_cast<float2>(gid);
    }
    """
    crossgl = convert(code)

    assert "vec2 p0 = vec2(gid);" in crossgl
    assert "static_cast" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_scoped_atomic_thread_fence_from_mlx_kernel_roundtrips():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 4367c73b60541ddd5a266ce4644fd93d20223b6e
    # Path: mlx/backend/metal/kernels/fence.metal
    code = """
    #pragma METAL internals : enable

    #ifndef __METAL_MEMORY_SCOPE_SYSTEM__
    #define __METAL_MEMORY_SCOPE_SYSTEM__ 3
    #endif
    namespace metal {
    constexpr constant metal::thread_scope thread_scope_system =
        static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
    }

    #include <metal_atomic>

    [[kernel]] void fence(
        volatile coherent(system) device uint* timestamp [[buffer(0)]],
        constant uint& value [[buffer(1)]]) {
        timestamp[0] = value;
        metal::atomic_thread_fence(metal::mem_flags::mem_device,
                                   metal::memory_order_seq_cst,
                                   metal::thread_scope_system);
    }
    """
    crossgl = convert(code)

    assert (
        "atomicThreadFence(mem_device, memory_order_seq_cst, " "thread_scope_system);"
    ) in crossgl
    assert (
        "volatile coherent(system) RWStructuredBuffer<uint> timestamp @buffer(0)"
        in crossgl
    )
    assert "memoryBarrier();" not in crossgl
    assert "constant thread_scope thread_scope_system" not in crossgl
    assert "metal_u3a_u3aatomic_thread_fence" not in crossgl
    assert "metal::atomic_thread_fence" not in crossgl
    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert (
        "metal::atomic_thread_fence(metal::mem_flags::mem_device, "
        "metal::memory_order_seq_cst, metal::thread_scope_system);"
    ) in metal
    assert "volatile coherent(system) device uint* timestamp [[buffer(0)]]" in metal
    assert "device uint* volatile" not in metal
    assert "threadgroup_barrier(mem_flags::mem_device);" not in metal
    assert "#pragma METAL internals : enable" in metal
    assert "__METAL_MEMORY_SCOPE_SYSTEM__ 3" in metal


def test_codegen_resource_memory_qualifiers_survive_aliases_without_leaking():
    code = """
    using FencePointer = volatile coherent(system) device atomic_uint*;

    [[kernel]] void qualify(
        FencePointer aliased [[buffer(0)]],
        volatile device uint* volatile_only [[buffer(1)]],
        coherent(device) device uint* device_scoped [[buffer(2)]],
        device uint* plain [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
      uint local = index;
    }
    """

    crossgl = convert(code)
    assert (
        "volatile coherent(system) RWStructuredBuffer<atomic_uint> aliased "
        "@buffer(0)" in crossgl
    )
    assert "volatile RWStructuredBuffer<uint> volatile_only @buffer(1)" in crossgl
    assert (
        "coherent(device) RWStructuredBuffer<uint> device_scoped @buffer(2)" in crossgl
    )
    assert "RWStructuredBuffer<uint> plain @buffer(3)" in crossgl
    assert "volatile uint index" not in crossgl
    assert "coherent(device) uint index" not in crossgl

    shared_ast = parse_crossgl(crossgl)
    function = next(iter(shared_ast.stages.values())).entry_point
    aliased, volatile_only, device_scoped, plain, index = function.parameters

    assert all(
        isinstance(qualifier, ResourceMemoryQualifierNode)
        for qualifier in aliased.resource_qualifiers
    )
    assert [
        (qualifier.kind, qualifier.scope) for qualifier in aliased.resource_qualifiers
    ] == [("volatile", None), ("coherent", "system")]
    assert [str(qualifier) for qualifier in volatile_only.resource_qualifiers] == [
        "volatile"
    ]
    assert [str(qualifier) for qualifier in device_scoped.resource_qualifiers] == [
        "coherent(device)"
    ]
    assert plain.resource_qualifiers == []
    assert index.resource_qualifiers == []
    assert function.body.statements[0].resource_qualifiers == []

    metal = MetalCodeGen().generate(shared_ast)
    assert (
        "volatile coherent(system) device atomic_uint* aliased [[buffer(0)]]" in metal
    )
    assert "volatile device uint* volatile_only [[buffer(1)]]" in metal
    assert "coherent(device) device uint* device_scoped [[buffer(2)]]" in metal
    assert "device uint* plain [[buffer(3)]]" in metal
    assert "uint index [[thread_position_in_grid]]" in metal
    assert "device atomic_uint* volatile" not in metal
    assert "uint local = index;" in metal
    assert "volatile uint local" not in metal
    assert "coherent(device) uint local" not in metal


def test_codegen_scopes_local_resource_alias_qualifiers_and_shadowing():
    code = """
    void aliases(uint index) {
      using FencePointer = volatile coherent(system) device uint*;
      FencePointer qualified = nullptr;
      {
        using FencePointer = device uint*;
        FencePointer shadowed = nullptr;
      }
      FencePointer restored = nullptr;
    }

    void later(uint index) {
      uint local = index;
    }
    """

    crossgl = convert(code)

    assert crossgl.count("volatile coherent(system) device uint*") == 2
    assert "device uint* shadowed = nullptr;" in crossgl
    assert "volatile device uint* shadowed" not in crossgl
    assert "coherent(system) device uint* shadowed" not in crossgl
    assert "uint local = index;" in crossgl
    assert "volatile uint local" not in crossgl
    assert "coherent(system) uint local" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_atomic_thread_fence_contract_matrix():
    code = """
    kernel void fence_contracts() {
        metal::atomic_thread_fence(metal::mem_flags::mem_threadgroup,
                                   metal::memory_order_relaxed,
                                   metal::thread_scope_threadgroup);
        metal::atomic_thread_fence(metal::mem_flags::mem_device,
                                   metal::memory_order_acquire,
                                   metal::thread_scope_device);
        metal::atomic_thread_fence(metal::mem_flags::mem_texture,
                                   metal::memory_order_release,
                                   metal::thread_scope_threadgroup);
        metal::atomic_thread_fence(
            metal::mem_flags::mem_device | metal::mem_flags::mem_threadgroup,
            metal::memory_order_acq_rel,
            metal::thread_scope_device);
        metal::atomic_thread_fence(metal::mem_flags::mem_device,
                                   metal::memory_order_seq_cst,
                                   metal::thread_scope_system);
    }
    """

    crossgl = convert(code)
    expected_contracts = (
        "atomicThreadFence(mem_threadgroup, memory_order_relaxed, "
        "thread_scope_threadgroup);",
        "atomicThreadFence(mem_device, memory_order_acquire, " "thread_scope_device);",
        "atomicThreadFence(mem_texture, memory_order_release, "
        "thread_scope_threadgroup);",
        "atomicThreadFence(mem_device | mem_threadgroup, "
        "memory_order_acq_rel, thread_scope_device);",
        "atomicThreadFence(mem_device, memory_order_seq_cst, " "thread_scope_system);",
    )
    for contract in expected_contracts:
        assert contract in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert metal.count("metal::atomic_thread_fence(") == len(expected_contracts)
    assert "metal::memory_order_relaxed" in metal
    assert "metal::memory_order_acquire" in metal
    assert "metal::memory_order_release" in metal
    assert "metal::memory_order_acq_rel" in metal
    assert "metal::memory_order_seq_cst" in metal
    assert "metal::thread_scope_threadgroup" in metal
    assert "metal::thread_scope_device" in metal
    assert "metal::thread_scope_system" in metal
    assert "threadgroup_barrier(" not in metal


def test_codegen_rejects_unknown_atomic_thread_fence_order():
    code = """
    kernel void fence() {
        metal::atomic_thread_fence(metal::mem_flags::mem_device,
                                   metal::memory_order_consume,
                                   metal::thread_scope_device);
    }
    """

    with pytest.raises(MetalAtomicFenceLoweringError) as exc_info:
        convert(code)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-atomic-fence-unsupported"
    )
    assert diagnostic.missing_capabilities == (
        "metal.atomic-thread-fence-contract-lowering",
    )
    assert diagnostic.reason == "unsupported-memory-order"
    assert diagnostic.memory_flags == "mem_device"
    assert diagnostic.memory_order == "memory_order_consume"
    assert diagnostic.thread_scope == "thread_scope_device"


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


def test_codegen_lowers_simdgroup_barrier_from_apple_silicon_sync_sample():
    # Reduced from Apple's WWDC20 "Bring your Metal app to Apple silicon Macs"
    # threadgroup synchronization sample.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void kernelMain(uint tid [[thread_index_in_threadgroup]],
                           uint simd_size [[threads_per_simdgroup]],
                           device uint* res [[buffer(0)]]) {
        threadgroup uint buf[64];
        buf[tid] = initBuffer(tid);

        if (simd_size == 64u)
            simdgroup_barrier(mem_flags::mem_threadgroup);
        else
            threadgroup_barrier(mem_flags::mem_threadgroup);

        uint index = (tid < 32) ? tid + 32 : tid - 32;
        res[tid] = buf[tid] + buf[index];
    }
    """
    crossgl = convert(code)

    assert crossgl.count("workgroupBarrier();") == 2
    assert "simdgroup_barrier" not in crossgl
    assert "threadgroup_barrier" not in crossgl
    assert "mem_flags" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_lowers_execution_only_threadgroup_barrier_from_mlx_gemv():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void reduce(device float* out [[buffer(0)]]) {
        threadgroup_barrier(mem_flags::mem_none);
        threadgroup_barrier(mem_flags::mem_none | mem_flags::mem_threadgroup);
        out[0] = 1.0f;
    }
    """
    crossgl = convert(code)

    assert "workgroupExecutionBarrier();" in crossgl
    assert "workgroupBarrier();" in crossgl
    assert "threadgroup_barrier" not in crossgl
    assert "mem_none" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_execution_only_threadgroup_barrier_reaches_native_targets():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void synchronize() {
        threadgroup_barrier(mem_flags::mem_none);
    }
    """
    ast = parse_crossgl(convert(code))

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    metal = MetalCodeGen().generate(ast)
    spirv = VulkanSPIRVCodeGen().generate(ast)

    assert "GroupMemoryBarrierWithGroupSync();" in hlsl
    assert "barrier();" in glsl
    assert "threadgroup_barrier(mem_flags::mem_none);" in metal
    control_barrier = re.search(r"OpControlBarrier %\d+ %\d+ (%\d+)", spirv)
    assert control_barrier is not None
    assert re.search(
        rf"{re.escape(control_barrier.group(1))} = OpConstant %\d+ 0", spirv
    )
    assert "OpMemoryBarrier" not in spirv
    for generated in (hlsl, glsl, metal, spirv):
        assert "workgroupExecutionBarrier" not in generated


def test_codegen_preserves_standalone_void_cast_operand_evaluation():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    uint observe(uint value) {
        return value + 1u;
    }

    kernel void discard_values(uint gid [[thread_position_in_grid]]) {
        (void)gid;
        (void)observe(gid);
    }
    """
    crossgl = convert(code)

    assert "(void)" not in crossgl
    assert "gid;" in crossgl
    assert "observe(gid);" in crossgl

    ast = parse_crossgl(crossgl)
    generated_targets = (
        TranslatorHLSLCodeGen().generate(ast),
        GLSLCodeGen().generate(ast),
        MetalCodeGen().generate(ast),
        VulkanSPIRVCodeGen().generate(ast),
    )
    for generated in generated_targets:
        assert "unknown function 'void'" not in generated
        assert "void(gid)" not in generated
    assert "OpFunctionCall" in generated_targets[-1]


def test_static_template_sibling_helper_reaches_native_targets(tmp_path):
    code = """
    template <typename T, int N>
    struct BlockKernel {
      template <typename U = T>
      static void load(const device T* src, thread U dst[N]) {
        for (int i = 0; i < N; ++i) {
          dst[i] = static_cast<U>(src[i]);
        }
      }

      static void run(const device T* src, device T* out) {
        thread T values[N];
        load<T>(src, values);
        out[0] = values[0];
      }
    };

    [[kernel]] void k(
        const device float* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      BlockKernel<float, 4>::run(src, out);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(code))

    helper_name = "BlockKernel_float_4__load__float"
    assert helper_name in crossgl
    assert "load_u3cfloat_u3e" not in crossgl
    ast = parse_crossgl(crossgl)
    generated_targets = (
        TranslatorHLSLCodeGen().generate(ast),
        GLSLCodeGen().generate(ast),
        MetalCodeGen().generate(ast),
        VulkanSPIRVCodeGen().generate(ast),
    )
    for generated in generated_targets:
        assert "cannot lower unknown function" not in generated
        assert "load_u3cfloat_u3e" not in generated
    assert helper_name in generated_targets[0]
    glsl_helper_name = "BlockKernel_float_4_load_float__glsl_src_src_float"
    assert (
        f"void {glsl_helper_name}(inout float dst[4], int src_offset);"
        in generated_targets[1]
    )
    assert (
        f"void {glsl_helper_name}(inout float dst[4], int src_offset) {{"
        in generated_targets[1]
    )
    assert f"{glsl_helper_name}(values, int(src_offset));" in generated_targets[1]
    assert "dst[i] = float(src[(src_offset + i)]);" in generated_targets[1]
    assert helper_name in generated_targets[2]
    assert "OpLoad" in generated_targets[3]
    assert "OpStore" in generated_targets[3]
    assert "WARNING" not in generated_targets[3]

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        source = tmp_path / "static-template-sibling-helper.comp"
        output = tmp_path / "static-template-sibling-helper.spv"
        source.write_text(generated_targets[1], encoding="utf-8")
        result = subprocess.run(
            [
                glslang,
                "-V",
                "--target-env",
                "vulkan1.1",
                "-S",
                "comp",
                str(source),
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "alias_declaration",
    [
        "using ConcreteKernel = BlockKernel<float, 4>;",
        "typedef BlockKernel<float, 4> ConcreteKernel;",
    ],
)
def test_codegen_materializes_static_template_helper_through_concrete_alias(
    alias_declaration,
):
    code = """
    template <typename T, int N>
    struct BlockKernel {
      template <typename U = T>
      static void load(const device T* src, thread U dst[N]) {
        for (int i = 0; i < N; ++i) {
          dst[i] = static_cast<U>(src[i]);
        }
      }

      static void run(const device T* src, device T* out) {
        thread T values[N];
        load(src, values);
        out[0] = values[0];
      }
    };

    [[kernel]] void k(
        const device float* src [[buffer(0)]],
        device float* out [[buffer(1)]]) {
      __ALIAS_DECLARATION__
      ConcreteKernel::run(src, out);
    }
    """.replace("__ALIAS_DECLARATION__", alias_declaration)

    crossgl = convert(MetalPreprocessor().preprocess(code))

    run_name = "BlockKernel_float_4__run"
    helper_name = "BlockKernel_float_4__load__float"
    assert re.search(rf"\bvoid\s+{helper_name}\s*\(", crossgl)
    assert re.search(rf"\b{helper_name}\s*\(src,\s*values\);", crossgl)
    assert re.search(rf"\bvoid\s+{run_name}\s*\(", crossgl)
    assert re.search(rf"\b{run_name}\s*\(src,\s*out_\);", crossgl)
    assert not re.search(r"(?<![\w:])run\s*\(", crossgl)


def test_codegen_binds_reused_concrete_alias_to_nearest_specialization():
    code = """
    template <typename T, int N>
    struct BlockKernel {
      static void run(device T* out) {
        out[0] = T(N);
      }
    };

    [[kernel]] void float_kernel(device float* out [[buffer(0)]]) {
      using ConcreteKernel = BlockKernel<float, 4>;
      ConcreteKernel::run(out);
    }

    [[kernel]] void int_kernel(device int* out [[buffer(0)]]) {
      using ConcreteKernel = BlockKernel<int, 2>;
      ConcreteKernel::run(out);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(code))

    assert re.search(r"\bBlockKernel_float_4__run\s*\(out_\);", crossgl)
    assert re.search(r"\bBlockKernel_int_2__run\s*\(out_\);", crossgl)
    assert not re.search(r"(?<![\w:])run\s*\(", crossgl)


def test_codegen_restores_outer_alias_after_nested_shadowing():
    code = """
    template <typename T, int N>
    struct BlockKernel {
      static void run(device T* out) {
        out[0] = T(N);
      }
    };

    [[kernel]] void k(
        device float* float_out [[buffer(0)]],
        device int* int_out [[buffer(1)]]) {
      using ConcreteKernel = BlockKernel<float, 4>;
      {
        using ConcreteKernel = BlockKernel<int, 2>;
        ConcreteKernel::run(int_out);
      }
      ConcreteKernel::run(float_out);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(code))

    assert re.search(r"\bBlockKernel_int_2__run\s*\(int_out\);", crossgl)
    assert re.search(r"\bBlockKernel_float_4__run\s*\(float_out\);", crossgl)


def test_codegen_rewrites_static_call_through_chained_alias():
    code = """
    template <typename T>
    struct BlockKernel {
      static void run(device T* out) {
        out[0] = T(1);
      }
    };

    [[kernel]] void k(device float* out [[buffer(0)]]) {
      using ConcreteKernel = BlockKernel<float>;
      using KernelAlias = ConcreteKernel;
      KernelAlias::run(out);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(code))

    assert re.search(r"\bBlockKernel_float__run\s*\(out_\);", crossgl)
    assert "KernelAlias::run" not in crossgl


def test_codegen_local_alias_shadows_concrete_struct_name():
    code = """
    struct FloatKernel {
      static void run(device int* out) {
        out[0] = 1;
      }
    };

    struct IntKernel {
      static void run(device int* out) {
        out[0] = 2;
      }
    };

    [[kernel]] void k(device int* out [[buffer(0)]]) {
      using FloatKernel = IntKernel;
      using KernelAlias = FloatKernel;
      FloatKernel::run(out);
      KernelAlias::run(out);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(code))

    assert len(re.findall(r"\bIntKernel__run\s*\(out_\);", crossgl)) == 2
    assert not re.search(r"\bFloatKernel__run\s*\(out_\);", crossgl)


@pytest.mark.parametrize(
    "alias_like_text",
    [
        "// using FloatKernel = IntKernel;",
        "/* using FloatKernel = IntKernel; */",
        'const char* note = "using FloatKernel = IntKernel;";',
    ],
)
def test_codegen_ignores_alias_text_in_comments_and_literals(alias_like_text):
    code = """
    struct FloatKernel {
      static void run(device int* out) {
        out[0] = 1;
      }
    };

    struct IntKernel {
      static void run(device int* out) {
        out[0] = 2;
      }
    };

    [[kernel]] void k(device int* out [[buffer(0)]]) {
      __ALIAS_LIKE_TEXT__
      FloatKernel::run(out);
    }
    """.replace("__ALIAS_LIKE_TEXT__", alias_like_text)

    preprocessed = MetalPreprocessor().preprocess(code)

    assert "FloatKernel__run(out);" in preprocessed
    assert "IntKernel__run(out);" not in preprocessed


def test_codegen_rewrites_static_alias_call_inside_materialized_method():
    code = """
    template <typename T>
    struct Helper {
      static void run(device T* out) {
        out[0] = T(1);
      }
    };

    template <typename T>
    struct Wrapper {
      static void run(device T* out) {
        using HelperType = Helper<T>;
        HelperType::run(out);
      }
    };

    [[kernel]] void k(device float* out [[buffer(0)]]) {
      using ConcreteWrapper = Wrapper<float>;
      ConcreteWrapper::run(out);
    }
    """

    crossgl = convert(MetalPreprocessor().preprocess(code))

    assert re.search(r"\bHelper_float__run\s*\(out_\);", crossgl)
    assert re.search(r"\bWrapper_float__run\s*\(out_\);", crossgl)
    assert "HelperType::run" not in crossgl


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

    # Metal atomics lower to their plain element type; atomicity is carried by the
    # atomic_* intrinsics, and the GLSL/DirectX/SPIR-V backends store atomics as
    # the underlying scalar (an atomic-wrapped buffer element is not valid).
    assert "RWStructuredBuffer<uint> counters @buffer(0)" in crossgl
    assert "RWStructuredBuffer<float> weights @buffer(1)" in crossgl
    assert "RWStructuredBuffer<uint64> totals @buffer(2)" in crossgl
    assert "RWStructuredBuffer<bool> flags @buffer(3)" in crossgl
    assert "atomic<" not in crossgl
    assert "metal::atomic" not in crossgl
    assert "atomic_uint" not in crossgl
    assert "atomic_float" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_lowers_metal_uniform_values_to_annotated_payload_types():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void uniform_values(device ulong* out [[buffer(0)]]) {
        const uniform<int> signed_value = make_uniform(-4);
        const metal::uniform<uint> unsigned_value = metal::make_uniform(5u);
        const uniform<ulong> wide_value = make_uniform(6ul);
        const metal::uniform<int2> lanes = metal::make_uniform(int2(1, 2));
        const uniform<int> expression_value = make_uniform(1 + 2) * 3;
        out[0] = ulong(signed_value + int(unsigned_value) + lanes.x
            + expression_value) + wide_value;
    }
    """
    crossgl = convert(code)

    assert "const int signed_value @uniform_value = (-4);" in crossgl
    assert "const uint unsigned_value @uniform_value = 5u;" in crossgl
    assert "const uint64 wide_value @uniform_value = 6u;" in crossgl
    assert "const ivec2 lanes @uniform_value = ivec2(1, 2);" in crossgl
    assert "const int expression_value @uniform_value = (1 + 2) * 3;" in crossgl
    assert "make_uniform" not in crossgl
    assert "uniform_<" not in crossgl
    assert "metal::uniform" not in crossgl

    parsed = parse_crossgl(crossgl)
    entry_point = next(iter(parsed.stages.values())).entry_point
    declarations = {
        statement.name: statement
        for statement in entry_point.body.statements
        if getattr(statement, "name", None)
        in {
            "signed_value",
            "unsigned_value",
            "wide_value",
            "lanes",
            "expression_value",
        }
    }
    assert set(declarations) == {
        "signed_value",
        "unsigned_value",
        "wide_value",
        "lanes",
        "expression_value",
    }
    for declaration in declarations.values():
        assert "uniform_value" in {
            attribute.name for attribute in declaration.attributes
        }


def test_codegen_metal_uniform_values_reach_targets_without_fallbacks():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void uniform_stride(device int* out [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
        const uniform<int> stride = make_uniform(16);
        out[gid] = stride + int(gid);
    }
    """
    ast = parse_crossgl(convert(code))

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    spirv = VulkanSPIRVCodeGen().generate(ast)

    for generated in (hlsl, glsl, spirv):
        assert "make_uniform" not in generated
        assert "uniform_value" not in generated
        assert "uniform_<" not in generated
    assert "cannot lower unknown function" not in spirv
    assert "Unknown type" not in spirv


def test_codegen_does_not_lower_user_defined_make_uniform_function():
    code = """
    int make_uniform(int value) {
        return value + 1;
    }

    kernel void custom_uniform(device int* out [[buffer(0)]]) {
        int value = make_uniform(16);
        int scoped_value = helpers::make_uniform(32);
        out[0] = value + scoped_value;
    }
    """
    crossgl = convert(code)

    assert "int make_uniform(int value)" in crossgl
    assert "int value = make_uniform(16);" in crossgl
    assert "int scoped_value = make_uniform(32);" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_lowers_metal_device_atomics_to_crossgl_intrinsics():
    # Metal explicit atomics map to CrossGL atomic intrinsics: the trailing
    # memory_order argument is dropped, and a `&buffer[i]` target lowers to the
    # buffer element subscript the GLSL/DirectX/SPIR-V backends expect.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void scatter(device atomic_int* out [[buffer(0)]],
                        device const int* idx [[buffer(1)]],
                        device const int* val [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
        atomic_fetch_add_explicit(&out[idx[i]], val[i], memory_order_relaxed);
        atomic_fetch_max_explicit(&out[0], val[i], memory_order_relaxed);
        atomic_exchange_explicit(&out[1], val[i], memory_order_relaxed);
    }
    """
    crossgl = convert(code)

    assert "atomicAdd(out_[buffer_load(idx, i)], buffer_load(val, i));" in crossgl
    assert "atomicMax(out_[0], buffer_load(val, i));" in crossgl
    assert "atomicExchange(out_[1], buffer_load(val, i));" in crossgl
    # The memory_order argument and the Metal spelling must not survive.
    assert "atomic_fetch" not in crossgl
    assert "memory_order" not in crossgl
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


def test_codegen_fragment_stencil_output_from_angle_generated_shader():
    # Reduced from:
    # Repo: https://android.googlesource.com/platform/external/angle
    # Path: src/libANGLE/renderer/metal/shaders/mtl_internal_shaders_autogen.metal
    code = """
    struct FragmentStencilOut {
        uint32_t stencil [[stencil]];
    };

    fragment FragmentStencilOut blitStencilFS() {
        FragmentStencilOut output;
        output.stencil = 7u;
        return output;
    }
    """
    crossgl = convert(code)

    assert "uint stencil @gl_FragStencilRefEXT;" in crossgl
    assert "@stencil" not in crossgl
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


@pytest.mark.parametrize("depth_mode", ["less", "greater"])
def test_codegen_fragment_output_attributes_canonicalize_high_mrt_and_depth_modes(
    depth_mode,
):
    code = f"""
    #include <metal_stdlib>
    using namespace metal;

    struct Out {{
        float4 color [[color(7, rgba16float)]];
        float depth [[depth({depth_mode})]];
    }};

    fragment Out fragment_main() {{
        Out out;
        out.color = float4(1.0, 0.0, 0.0, 1.0);
        out.depth = 0.5;
        return out;
    }}
    """
    result = convert(code)

    assert "vec4 color @gl_FragColor7;" in result
    assert "float depth @gl_FragDepth;" in result
    assert "@color(7" not in result
    assert f"@depth({depth_mode})" not in result
    assert parse_crossgl(result) is not None


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
    assert "tex.read(uint2(pixel), uint(lod))" in metal
    assert (
        "layers.read(uint2((uint3(pixel, layer)).xy), "
        "uint((uint3(pixel, layer)).z), uint(lod))"
    ) in metal
    assert "msTex.read(uint2(pixel), uint(sample))" in metal
    assert (
        "msLayers.read(uint2((uint3(pixel, layer)).xy), "
        "uint((uint3(pixel, layer)).z), uint(sample))"
    ) in metal
    assert "line.read(uint(x), uint(0))" in metal
    assert (
        "lineLayers.read(uint((uint2(x, layer)).x), "
        "uint((uint2(x, layer)).y), uint(0))"
    ) in metal
    assert "volume.read(uint3(voxel), uint(lod))" in metal
    assert "textureLoad(" not in metal


def test_codegen_sampled_cube_texture_read_from_msl_spec_is_diagnostic():
    # Apple Metal Shading Language Specification, section 5.10.6:
    # texturecube read overloads carry coord, face, and lod separately. CrossGL
    # has no face-aware cube texel-fetch form, so do not lower face as LOD.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    float4 readCube(texturecube<float> tex, uint2 coord, uint face, uint lod) {
        return tex.read(coord, face, lod);
    }

    float4 readCubeArray(texturecube_array<float> tex,
                         uint2 coord,
                         uint face,
                         uint layer,
                         uint lod) {
        return tex.read(coord, face, layer, lod);
    }
    """
    crossgl = convert(code)

    diagnostic = (
        "vec4(0.0) /* unsupported Metal sampled cube texture read: "
        "read on tex requires face-aware texel fetch */"
    )
    assert crossgl.count(diagnostic) == 2
    assert "texelFetch(tex, coord, face)" not in crossgl
    assert "texelFetch(tex, vec4(coord, face), layer)" not in crossgl
    assert parse_crossgl(crossgl) is not None


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


def test_codegen_kernel_texture_read_write_roundtrips_as_stage_entry():
    # Mirrors the public Metal image-processing idiom documented by Apple and
    # Metal by Example: compute kernels reading and writing texture2d resources.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void convertToGrayscale(
        texture2d<half, access::read> inTexture [[texture(0)]],
        texture2d<half, access::write> outTexture [[texture(1)]],
        uint2 gid [[thread_position_in_grid]]) {
        half4 color = inTexture.read(gid);
        half gray = dot(color.rgb, half3(0.299h, 0.587h, 0.114h));
        outTexture.write(half4(gray, gray, gray, color.a), gid);
    }
    """
    crossgl = convert(code)

    assert "@ stage_entry" in crossgl
    assert "image2D inTexture @texture(0) @readonly" in crossgl
    assert "image2D outTexture @texture(1) @writeonly" in crossgl
    assert crossgl.count("@rgba16f @metal_texture_element_half") == 2
    assert "imageLoad(inTexture, gid)" in crossgl
    assert "imageStore(outTexture, gid" in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)

    assert "kernel void convertToGrayscale(" in metal
    assert "texture2d<half, access::read> inTexture [[texture(0)]]" in metal
    assert "texture2d<half, access::write> outTexture [[texture(1)]]" in metal
    assert "inTexture.read(uint2(gid))" in metal
    assert "outTexture.write(half4(gray, gray, gray, color.a), uint2(gid))" in metal
    assert "kernel void kernel_main()" not in metal


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
                                   uint simd_group_id [[simdgroup_index_in_threadgroup]],
                                   uint simd_size [[threads_per_simdgroup]]) {
        device const half* x_row = x + tid.x;
        half value = *(x_row + simd_lane_id + simd_group_id + simd_size);
    }
    """
    result = convert(code)

    assert "uint simd_lane_id @gl_SubgroupInvocationID" in result
    assert "uint simd_group_id @gl_SubgroupID" in result
    assert "uint simd_size @gl_SubgroupSize" in result
    assert "@thread_index_in_simdgroup" not in result
    assert "@simdgroup_index_in_threadgroup" not in result
    assert "@threads_per_simdgroup" not in result
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


def test_codegen_function_constant_parameter_strips_global_namespace_qualifier_from_apple_hdr_sample():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constant uint32_t kExposureModeIndex [[function_constant(AAPLFunctionConstantIndexExposureType)]];

    fragment half4 BloomSetup(
        texture2d<half> logLuminanceIn [[texture(1), function_constant(::kExposureModeIndex)]]) {
        return half4(logLuminanceIn.get_width());
    }
    """
    crossgl = convert(code)

    assert "@function_constant(kExposureModeIndex)" in crossgl
    assert "@function_constant(::kExposureModeIndex)" not in crossgl
    assert parse_crossgl(crossgl) is not None


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


def test_codegen_preserves_materialx_out_parameter_qualifier_from_xcode_genmsl():
    # Reduced from Xcode's bundled MaterialX MSL library:
    # /Applications/Xcode.app/.../USDLib_FormatLoaderProxy_Xcode.framework/
    # Resources/libraries/stdlib/genmsl/mx_burn_float.metal
    code = """
    void mx_burn_float(float fg, float bg, float mixval, out float result) {
        if (abs(fg) < M_FLOAT_EPS) {
            result = 0.0;
            return;
        }
        result = mixval * (1.0 - ((1.0 - bg) / fg)) + ((1.0 - mixval) * bg);
    }
    """
    crossgl = convert(code)

    assert "out float result" in crossgl
    assert "float out" not in crossgl
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


def test_codegen_sanitizes_crossgl_keyword_generic_type_argument_from_tinygrad():
    code = """
    template<typename T, typename layout>
    void consume(thread rt_base<T, layout>& src) {
        return;
    }
    """
    crossgl = convert(code)

    assert "inout thread rt_base<T,layout_> src" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_template_helper_generics_for_crossgl_specialization():
    code = """
    template <typename T, typename U>
    METAL_FUNC T ceildiv(T N, U M) {
        return (N + M - 1) / M;
    }
    """
    crossgl = convert(code)

    assert "generic<T, U> T ceildiv(T N, U M)" in normalize(crossgl)
    ast = parse_crossgl(crossgl)
    function = ast.functions[0]
    assert function.name == "ceildiv"
    assert [param.name for param in function.generic_params] == ["T", "U"]


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


def test_codegen_omits_global_constexpr_sampler_array_argument_for_roundtrip():
    # Apple MSL supports arrays of samplers declared in program scope.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constexpr array<sampler, 2> samplers = {
        sampler(address::clamp_to_zero),
        sampler(coord::pixel)
    };

    fragment float4 sampleIndexedSampler(texture2d<float> tex, float2 uv) {
        return tex.sample(samplers[0], uv);
    }
    """
    crossgl = convert(code)

    assert "texture(tex, uv)" in crossgl
    assert "sampler[2]" not in crossgl
    assert "samplers" not in crossgl

    ast = parse_crossgl(crossgl)
    metal = MetalCodeGen().generate(ast)
    assert "tex.sample(sampler(" in metal


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


def test_codegen_raytracing_namespace_alias_resolves_to_crossgl_type():
    code = """
    #include <metal_stdlib>
    using namespace metal;
    namespace rt = raytracing;

    intersection void isect(rt::ray r, intersector inter) { }
    """

    result = convert(code)

    assert "raytracing;" not in result
    assert "rt::ray" not in result
    assert "void isect(ray r, intersector inter)" in result
    parse_crossgl(result)


def test_codegen_preserves_ray_and_object_address_space_payloads_from_msl_spec():
    # Apple Metal Shading Language Specification, section 4 Address Spaces:
    # ray_data and object_data are address-space attributes for payload references.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct RayPayload {
        float value;
    };

    struct ObjectPayload {
        float value;
    };

    intersection void intersectPayload(ray_data RayPayload& payload [[payload]],
                                       object_data ObjectPayload& objectPayload [[payload]]) {
        payload.value = objectPayload.value;
    }
    """
    crossgl = convert(code)

    assert "inout RayPayload payload @ray_data @payload" in crossgl
    assert "inout ObjectPayload objectPayload @object_data @payload" in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "ray_data RayPayload& payload [[payload]]" in metal
    assert "object_data ObjectPayload& objectPayload [[payload]]" in metal
    assert "thread RayPayload& payload" not in metal
    assert "thread ObjectPayload& objectPayload" not in metal


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


def test_codegen_size_t_typedef_uses_parseable_crossgl_alias_from_moltenvk():
    # Reduced from:
    # Repo: https://github.com/KhronosGroup/MoltenVK
    # Commit: 5843e5da2e1f561261cb06a2f859ad39663d054f
    # Path: MoltenVK/MoltenVK/Commands/MVKCommandPipelineStateFactoryShaderSource.h
    code = """
    #include <metal_stdlib>
    using namespace metal;

    typedef size_t VkDeviceSize;

    typedef enum : uint32_t {
        VK_FORMAT_BC1_RGB_UNORM_BLOCK = 131,
    } VkFormat;
    """
    crossgl = convert(code)

    assert "typedef u64 VkDeviceSize;" in crossgl
    assert "typedef uint64 VkDeviceSize;" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_omits_user_type_aliases_from_public_metal_headers():
    # Reduced from MetalPetal's MTIShaderLib.h: CrossGL can use the user type
    # directly, but cannot parse a typedef whose source is another user type.
    code = """
    typedef MTIVertex VertexIn;

    struct MTIVertex {
        float4 position;
    };

    fragment float4 passthrough(VertexIn in [[stage_in]]) {
        return in.position;
    }
    """
    crossgl = convert(code)

    assert "typedef MTIVertex VertexIn;" not in crossgl
    assert "VertexIn in_" in crossgl
    parse_crossgl(crossgl)


def test_codegen_omits_namespace_self_alias_from_mlx_logging_header():
    # Reduced from MLX logging.h: `using os_log = metal::os_log` maps to the
    # same CrossGL type/name pair and should not emit `typedef os_log os_log`.
    code = """
    namespace mlx {
    using os_log = metal::os_log;

    struct os_log {
    };
    }
    """
    crossgl = convert(code)

    assert "typedef os_log os_log;" not in crossgl
    assert "struct os_log" in crossgl
    parse_crossgl(crossgl)


def test_codegen_preserves_type_template_metadata_for_crossgl_generics():
    code = """
    template <typename U>
    struct Limits {
        U max;
    };
    """
    crossgl = convert(code)

    assert "generic<U> struct Limits" in crossgl
    parse_crossgl(crossgl)


def test_codegen_sanitizes_scoped_types_from_public_shader_headers():
    # Reduced from Blender and tinygrad shader headers that reference C++
    # namespace-scoped helper types in typedefs and function signatures.
    code = """
    typedef gbuffer::ClosurePacking ClosurePacking;
    typedef ducks::rv_layout::align align_l;

    void consume(shader::Type type, gbuffer::Header header) {
        type = shader::Type::float_t;
        header.value = 1u;
    }
    """
    crossgl = convert(code)

    assert "typedef ClosurePacking ClosurePacking;" not in crossgl
    assert "typedef align align_l;" not in crossgl
    assert "shader::Type" not in crossgl
    assert "gbuffer::Header" not in crossgl
    assert "void consume(Type type, Header header)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_local_enum_declaration_from_metal_function_body():
    code = """
    fragment float4 local_enum_frag(float4 color [[stage_in]]) {
        enum Mode { ModeA = 0, ModeB = 1 };
        Mode mode = ModeA;
        return mode == ModeA ? color : float4(0.0);
    }
    """
    crossgl = convert(code)

    assert "enum Mode {" in crossgl
    assert "ModeA = 0" in crossgl
    assert "ModeB = 1" in crossgl
    assert "Mode mode = ModeA;" in crossgl
    assert "return mode == ModeA ? color : vec4(0.0);" in crossgl
    assert "Unhandled statement type: EnumNode" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_anonymous_enum_uses_synthetic_name_from_metal_base_effect():
    code = """
    enum {
        VertexAttributePosition,
        VertexAttributeNormal,
        VertexAttributeColor,
        VertexAttributeTexCoord0,
    };

    vertex float4 vertex_main(float4 position [[attribute(VertexAttributePosition)]]) {
        return position;
    }
    """
    crossgl = convert(code)

    assert "enum MetalAnonymousEnum0" in crossgl
    assert "enum None" not in crossgl
    assert "VertexAttributeTexCoord0" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_sizeof_and_cast():
    code = """
    using Scalar = half;

    void main() {
        int a = sizeof(int);
        int b = alignof(float4);
        int c = sizeof(float3);
        int d = sizeof(Scalar);
        int e = sizeof(metal::array<float3, 2>);
        float3 v = (float3)(1.0);
    }
    """
    result = convert(code)
    assert "int a = 4;" in result
    assert "alignof(float4)" in result
    assert "int c = 16;" in result
    assert "int d = 2;" in result
    assert "int e = 32;" in result
    assert "sizeof(" not in result
    assert "vec3(1.0)" in result


def test_codegen_rejects_sizeof_known_aggregate_without_layout_contract():
    code = """
    struct alignas(16) Payload {
        float value;
    };

    int payload_size() {
        return sizeof(Payload);
    }
    """

    with pytest.raises(MetalSizeofResolutionError) as error:
        convert(code)

    assert error.value.operand == "Payload"
    assert "aggregate object layout" in error.value.reason


def test_codegen_folds_exact_metal_struct_layouts():
    code = """
    struct ComplexValue {
        float real;
        float imag;
    };

    struct PaddedValue {
        char tag;
        float3 value;
        short tail;
    };

    struct NestedValue {
        ComplexValue value;
        half lanes[3];
        static constexpr int count = 3;
    };

    union ValueBits {
        float value;
        uint words[2];
    };

    int complex_size() { return sizeof(ComplexValue); }
    int padded_size() { return sizeof(PaddedValue); }
    int nested_size() { return sizeof(NestedValue); }
    int union_size() { return sizeof(ValueBits); }
    """

    crossgl = convert(code)

    assert crossgl.count("return 8;") == 2
    assert "return 48;" in crossgl
    assert "return 16;" in crossgl
    assert "sizeof(" not in crossgl


def test_codegen_sizeof_local_type_alias_does_not_leak_between_functions():
    crossgl = convert("""
        int first() {
            using Scalar = half;
            return sizeof(Scalar);
        }

        int second() {
            return sizeof(Scalar);
        }
        """)

    assert "return 2;" in crossgl
    assert "return sizeof(Scalar);" in crossgl


def test_codegen_resolves_function_local_typedef_uses():
    crossgl = convert("""
        template <typename T>
        struct Limits {
            static constexpr T finite_min = T(-1);
        };

        template <>
        struct Limits<float> {
            static constexpr float finite_min = -3.0f;
        };

        float compute(float scale) {
            typedef float U;
            U value = U(scale);
            U minimum = Limits<U>::finite_min;
            return value + minimum;
        }
        """)

    assert "float value = float(scale);" in crossgl
    assert "float minimum = ((-3.0f));" in crossgl
    assert "typedef float U;" not in crossgl
    assert "Limits_u3cU_u3e" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_scopes_nested_function_local_typedefs():
    crossgl = convert("""
        float convert_value(float value, bool use_integer) {
            typedef float U;
            if (use_integer) {
                typedef int U;
                U inner = U(value);
            }
            U outer = U(value);
            return outer;
        }
        """)

    assert "int inner = int(value);" in crossgl
    assert "float outer = float(value);" in crossgl
    assert "typedef" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_isolates_function_local_typedefs_between_siblings():
    crossgl = convert("""
        float first(float value) {
            typedef float U;
            U converted = U(value);
            return converted;
        }

        int second(float value) {
            typedef int U;
            U converted = U(value);
            return converted;
        }
        """)

    assert "float converted = float(value);" in crossgl
    assert "int converted = int(value);" in crossgl
    assert "typedef" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_switch_typedef_scope_spans_cases():
    crossgl = convert("""
        int select_value(int mode) {
            switch (mode) {
                case 0:
                    typedef int CaseValue;
                    break;
                case 1:
                    CaseValue value = CaseValue(3);
                    return value;
                default:
                    return 0;
            }
        }
        """)

    assert "int value = int(3);" in crossgl
    assert "typedef" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_parses_materialized_owner_alias_pointer_casts():
    source = """
        template <typename T>
        struct Tile {
            using elem_type = T;
            elem_type values[4];

            thread elem_type* elems() {
                return (thread elem_type*)values;
            }

            const thread elem_type* const_elems() const {
                return reinterpret_cast<const thread elem_type*>(values);
            }
        };

        Tile<float> float_tile;
        Tile<int> int_tile;
        """

    crossgl = convert(MetalPreprocessor().preprocess(source))
    float_helper = crossgl.split("Tile_float__elems", 1)[1].split("}", 1)[0]
    float_const_helper = crossgl.split("Tile_float__const_elems", 1)[1].split("}", 1)[0]
    int_helper = crossgl.split("Tile_int__elems", 1)[1].split("}", 1)[0]
    int_const_helper = crossgl.split("Tile_int__const_elems", 1)[1].split("}", 1)[0]

    assert "return (float*)self.values;" in float_helper
    assert "return (float*)self.values;" in float_const_helper
    assert "return (int*)self.values;" in int_helper
    assert "return (int*)self.values;" in int_const_helper
    assert "elem_type" not in float_helper
    assert "elem_type" not in float_const_helper
    assert "elem_type" not in int_helper
    assert "elem_type" not in int_const_helper
    assert "thread float*" not in crossgl
    assert "thread int*" not in crossgl
    strict_ast = CrossGLParser(
        CrossGLLexer(crossgl).get_tokens(), strict_function_bodies=True
    ).parse()
    assert strict_ast is not None


def test_codegen_parses_materialized_struct_alias_template_vectors():
    source = """
        struct BaseFrag {
            static constexpr short kElems = 4;

            template <typename U>
            using dtype_frag_t = typename metal::vec<U, kElems>;

            template <typename T>
            static dtype_frag_t<T> make(T value) {
                dtype_frag_t<T> result;
                result[0] = value;
                return result;
            }
        };

        kernel void vector_alias(
            device float* output [[buffer(0)]],
            uint gid [[thread_position_in_grid]]) {
            BaseFrag::dtype_frag_t<float> value = BaseFrag::make<float>(1.0f);
            output[gid] = value[0];
        }
        """

    crossgl = convert(MetalPreprocessor().preprocess(source))

    assert "vec4 value = BaseFrag__make__float(1.0f);" in crossgl
    assert "vec4 BaseFrag__make__float(float value)" in crossgl
    assert "vec4 result;" in crossgl
    assert "dtype_frag_t" not in crossgl
    strict_ast = CrossGLParser(
        CrossGLLexer(crossgl).get_tokens(), strict_function_bodies=True
    ).parse()
    assert strict_ast is not None


def test_codegen_preserves_non_wide_array_return_initializer(tmp_path):
    source = """
        metal::array<bool, 2> make_pair(bool x, bool y) {
            return {x, y};
        }

        kernel void use_pair(
            device uint* output [[buffer(0)]],
            uint gid [[thread_position_in_grid]]) {
            metal::array<bool, 2> value = make_pair(true, false);
            output[gid] = uint(value[0]);
        }
        """

    crossgl = convert(source)

    assert "bool[2] make_pair(bool x, bool y)" in crossgl
    assert "return {x, y};" in crossgl
    ast = parse_crossgl(crossgl)
    spirv = VulkanSPIRVCodeGen().generate(ast)
    assert "OpTypeArray" in spirv
    assert "OpCompositeConstruct" in spirv
    assert "OpReturnValue" in spirv
    assert "WARNING" not in spirv

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        assembly_path = tmp_path / "array-return.spvasm"
        binary_path = tmp_path / "array-return.spv"
        assembly_path.write_text(spirv, encoding="utf-8")
        subprocess.run(
            [
                spirv_as,
                "--target-env",
                "vulkan1.1",
                str(assembly_path),
                "-o",
                str(binary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [spirv_val, "--target-env", "vulkan1.1", str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_codegen_lowers_concrete_wide_vectors_to_aggregate_helpers():
    source = """
        using Wide = metal::vec<float, 8>;

        Wide make_wide(float base) {
            return Wide(
                base,
                base + 1.0f,
                base + 2.0f,
                base + 3.0f,
                base + 4.0f,
                base + 5.0f,
                base + 6.0f,
                base + 7.0f);
        }

        void accumulate(thread Wide& value, float delta) {
            value += Wide(delta);
            value[3] = value.s7;
        }

        float read_lane(Wide value, uint lane) {
            return value[lane] + value.s2;
        }

        kernel void use_wide_vector(
            device float* output [[buffer(0)]],
            uint gid [[thread_position_in_grid]]) {
            Wide splat = Wide(1.0f);
            Wide full = make_wide(float(gid));
            accumulate(full, 2.0f);
            output[gid] = read_lane(full, gid & 7u) + splat.s0;
        }
        """

    crossgl = convert(source)

    aggregate = "CrossGLMetalVector_float_8"
    assert crossgl.count(f"struct {aggregate}") == 1
    assert "float lanes[8];" in crossgl
    assert f"{aggregate} {aggregate}_splat(float value)" in crossgl
    assert f"{aggregate} {aggregate}_make(" in crossgl
    for lane in range(8):
        assert f"result.lanes[{lane}] = value;" in crossgl
        assert f"result.lanes[{lane}] = value{lane};" in crossgl
    assert (
        f"return {aggregate}_make(base, base + 1.0f, base + 2.0f, "
        "base + 3.0f, base + 4.0f, base + 5.0f, base + 6.0f, "
        "base + 7.0f);"
    ) in crossgl
    assert f"{aggregate} make_wide(float base)" in crossgl
    assert f"void accumulate(inout thread {aggregate} value, float delta)" in crossgl
    assert (
        f"{aggregate}_add_assign_vector(value, {aggregate}_splat(delta));"
    ) in crossgl
    assert "value.lanes[3] = value.lanes[7];" in crossgl
    assert "return value.lanes[lane] + value.lanes[2];" in crossgl
    assert f"{aggregate} splat = {aggregate}_splat(1.0f);" in crossgl
    assert "accumulate(full, 2.0f);" in crossgl
    assert "gid & 7u" in crossgl
    assert "vec<float, 8>" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_wide_vector_aggregate_reaches_native_targets(tmp_path):
    source = """
        using Wide = metal::vec<float, 8>;

        void add_bias(thread Wide& value, float bias) {
            value += Wide(bias);
        }

        kernel void store_wide_lane(
            device float* out_buffer [[buffer(0)]],
            uint gid [[thread_position_in_grid]]) {
            Wide value = Wide(
                0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f);
            add_bias(value, 1.0f);
            out_buffer[gid] = value[gid & 7u];
        }
        """

    crossgl = convert(source)
    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    spirv = VulkanSPIRVCodeGen().generate(ast)

    aggregate = "CrossGLMetalVector_float_8"
    for generated in (hlsl, glsl):
        assert f"struct {aggregate}" in generated
        assert "float lanes[8]" in generated
        assert "vec<float, 8>" not in generated
    assert f"{aggregate}_add_assign_vector" in hlsl
    assert f"{aggregate}_add_assign_vector" in glsl
    assert f"void add_bias(inout {aggregate} value, float bias)" in hlsl
    assert f"void add_bias(inout {aggregate} value, float bias)" in glsl
    assert "add_bias(value, 1.0);" in hlsl
    assert "add_bias(value, 1.0);" in glsl
    assert "OpTypeStruct" in spirv
    assert "OpTypeArray" in spirv
    assert "OpAccessChain" in spirv
    assert "OpFAdd" in spirv
    assert "OpStore" in spirv
    assert "WARNING" not in spirv

    dxc = shutil.which("dxc")
    glslang = shutil.which("glslangValidator")
    hlsl_path = tmp_path / "wide-vector.hlsl"
    hlsl_path.write_text(hlsl, encoding="utf-8")
    if dxc is not None:
        subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    elif glslang is not None:
        subprocess.run(
            [
                glslang,
                "-D",
                "-V",
                "-S",
                "comp",
                "-e",
                "CSMain",
                str(hlsl_path),
                "-o",
                str(tmp_path / "wide-vector-hlsl.spv"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    if glslang is not None:
        glsl_path = tmp_path / "wide-vector.comp"
        glsl_path.write_text(glsl, encoding="utf-8")
        subprocess.run(
            [glslang, "-S", "comp", str(glsl_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        assembly_path = tmp_path / "wide-vector.spvasm"
        binary_path = tmp_path / "wide-vector.spv"
        assembly_path.write_text(spirv, encoding="utf-8")
        subprocess.run(
            [
                spirv_as,
                "--target-env",
                "vulkan1.1",
                str(assembly_path),
                "-o",
                str(binary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [spirv_val, "--target-env", "vulkan1.1", str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_codegen_rejects_multi_lane_wide_vector_member_selector():
    source = """
        using Wide = metal::vec<float, 8>;

        float2 select_lanes(Wide value) {
            return value.xy;
        }
        """

    with pytest.raises(MetalWideVectorLoweringError) as error:
        convert(source)

    assert error.value.vector_type == "metal::vec<float,8>"
    assert error.value.operation == "member-access"
    assert "member selector 'xy' is not a single lane" in error.value.reason


def test_codegen_rejects_wide_vector_constructor_from_narrow_vector():
    source = """
        using Wide = metal::vec<float, 8>;

        Wide widen(float4 value) {
            return Wide(value);
        }
        """

    with pytest.raises(MetalWideVectorLoweringError) as error:
        convert(source)

    assert error.value.vector_type == "metal::vec<float,8>"
    assert error.value.operation == "constructor"
    assert "requires a scalar or one matching wide vector" in error.value.reason


@pytest.mark.parametrize(
    ("expression", "operation"),
    [
        ("-value", "-"),
        ("metal::abs(value)", "call metal::abs"),
    ],
)
def test_codegen_rejects_unsupported_wide_vector_operations(expression, operation):
    source = f"""
        using Wide = metal::vec<float, 8>;

        Wide apply(Wide value) {{
            return {expression};
        }}
        """

    with pytest.raises(MetalWideVectorLoweringError) as error:
        convert(source)

    assert error.value.vector_type == "metal::vec<float,8>"
    assert error.value.operation == operation
    assert "no semantics-preserving aggregate" in error.value.reason


def test_codegen_wide_vector_pointer_indexing_selects_the_pointee_first():
    source = """
        using Wide = metal::vec<float, 8>;

        float read_lane(thread Wide* values, uint row, uint lane) {
            return values[row][lane];
        }
        """

    crossgl = convert(source)

    assert "return values[row].lanes[lane];" in crossgl
    assert "values.lanes[row]" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_wide_vector_compound_assignment_evaluates_lvalue_once():
    source = """
        using Wide = metal::vec<float, 8>;

        void update(thread Wide* values, thread uint& index) {
            values[index++] += Wide(1.0f);
        }
        """

    crossgl = convert(source)

    statement = next(
        line for line in crossgl.splitlines() if "add_assign_vector(values" in line
    )
    assert statement.count("index++") == 1
    assert "add_assign_vector(values[index++]" in statement
    assert parse_crossgl(crossgl) is not None


def test_codegen_wide_vector_braced_initializer_zero_fills_remaining_lanes():
    source = """
        using Wide = metal::vec<float, 8>;

        Wide make_value() {
            return Wide{1.0f};
        }
        """

    crossgl = convert(source)

    assert (
        "return CrossGLMetalVector_float_8_make(" "1.0f, 0, 0, 0, 0, 0, 0, 0);"
    ) in crossgl
    assert "CrossGLMetalVector_float_8_splat(1.0f)" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_wide_vector_nested_arithmetic_retains_aggregate_type():
    source = """
        using Wide = metal::vec<float, 8>;

        Wide add_twice(Wide value, float bias) {
            return (value + bias) + bias;
        }
        """

    crossgl = convert(source)

    helper = "CrossGLMetalVector_float_8_add_vector_scalar"
    assert f"return {helper}({helper}(value, bias), bias);" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_rejects_abi_visible_wide_vector_storage():
    source = """
        using Wide = metal::vec<float, 8>;

        kernel void store_wide(
            device Wide* output [[buffer(0)]],
            uint gid [[thread_position_in_grid]]) {
            output[gid] = Wide(1.0f);
        }
        """

    with pytest.raises(MetalWideVectorLoweringError) as error:
        convert(source)

    assert error.value.operation == "resource-layout"
    assert "does not preserve Metal ABI alignment" in error.value.reason


def test_codegen_avoids_wide_vector_helper_name_collisions():
    source = """
        float CrossGLMetalVector_float_8_splat(float value) {
            return value;
        }

        using Wide = metal::vec<float, 8>;

        Wide make_value(float value) {
            return Wide(value);
        }
        """

    crossgl = convert(source)

    assert "struct CrossGLMetalVector_float_8_1" in crossgl
    assert "CrossGLMetalVector_float_8_1_splat(value)" in crossgl
    assert crossgl.count("CrossGLMetalVector_float_8_splat(float value)") == 1
    assert parse_crossgl(crossgl) is not None


def test_codegen_avoids_wide_vector_helper_name_collisions_with_local_variables():
    source = """
        using Wide = metal::vec<float, 8>;

        Wide make_value(float value) {
            float CrossGLMetalVector_float_8_splat = value;
            return Wide(CrossGLMetalVector_float_8_splat);
        }
        """

    crossgl = convert(source)

    assert "struct CrossGLMetalVector_float_8_1" in crossgl
    assert (
        "CrossGLMetalVector_float_8_1_splat(" "CrossGLMetalVector_float_8_splat)"
    ) in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_sizeof_dependent_typename_from_tinygrad_tile_copy():
    code = """
    template<typename ST>
    METAL_FUNC void load(threadgroup ST &dst) {
        constexpr const int elem_per_memcpy =
            sizeof(read_vector) / sizeof(typename ST::dtype);
        return;
    }
    """
    result = convert(code)

    assert "sizeof(ST::dtype)" in result
    assert parse_crossgl(result) is not None


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


def test_codegen_visible_function_table_using_signature_alias_from_apple_wwdc():
    # Apple WWDC20 uses function-signature aliases to keep visible function table
    # declarations readable.
    # https://developer.apple.com/videos/play/wwdc2020/10013/
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Light { uint index; };
    struct Lighting { float3 color; };
    struct Material { uint index; };
    struct TriangleIntersectionData { float3 normal; };

    using LightingFunction = Lighting(Light, TriangleIntersectionData);
    using MaterialFunction = float3(Material, Lighting, TriangleIntersectionData);

    kernel void shade(
        visible_function_table<LightingFunction> lightingFunctions [[buffer(1)]],
        visible_function_table<MaterialFunction> materialFunctions [[buffer(2)]],
        device float3* output [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
        Light light;
        Material material;
        TriangleIntersectionData triangleIntersection;
        Lighting lighting = lightingFunctions[light.index](light, triangleIntersection);
        output[tid] = materialFunctions[material.index](
            material, lighting, triangleIntersection);
    }
    """
    result = convert(code)

    assert "visible_function_table lightingFunctions @buffer(1)" in result
    assert "visible_function_table materialFunctions @buffer(2)" in result
    assert "lightingFunctions[light.index](light, triangleIntersection)" in result
    assert "materialFunctions[material.index]" in result
    assert "typedef Lighting LightingFunction;" not in result
    assert "typedef vec3 MaterialFunction;" not in result
    assert parse_crossgl(result) is not None


def test_codegen_materializes_callable_alias_without_runtime_typedef():
    code = """
    typedef void (*RadixFunc)(thread float2*, thread float2*);

    void radix2(thread float2* values, thread float2* scratch) {
        values[0] = scratch[0];
    }

    void radix4(thread float2* values, thread float2* scratch) {
        values[0] = scratch[0] + scratch[1];
    }

    template <int Radix, RadixFunc Function>
    void apply_radix(thread float2* values, thread float2* scratch) {
        Function(values, scratch);
    }

    kernel void fft(device float2* out [[buffer(0)]]) {
        float2 values[2];
        float2 scratch[2];
        apply_radix<2, radix2>(values, scratch);
        apply_radix<4, radix4>(values, scratch);
        out[0] = values[0];
    }
    """
    result = convert(code)

    assert "typedef void RadixFunc;" not in result
    assert "void RadixFunc;" not in result
    assert "apply_radix_2_radix2(values, scratch);" in result
    assert "apply_radix_4_radix4(values, scratch);" in result
    assert "radix2(values, scratch);" in result
    assert "radix4(values, scratch);" in result
    assert parse_crossgl(result) is not None


def test_codegen_callable_alias_materializations_reach_directx_and_opengl(
    tmp_path,
):
    code = """
    typedef float (*UnaryFunc)(float);

    float plus_one(float value) {
        return value + 1.0;
    }

    float times_two(float value) {
        return value * 2.0;
    }

    template <UnaryFunc Function>
    float apply(float value) {
        return Function(value);
    }

    kernel void callable_kernel(device float* out [[buffer(0)]]) {
        out[0] = apply<plus_one>(1.0) + apply<times_two>(2.0);
    }
    """
    ast = parse_crossgl(convert(code))
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)

    for artifact in (hlsl, glsl):
        assert "apply_plus_one(1.0)" in artifact
        assert "apply_times_two(2.0)" in artifact
        assert "return plus_one(value);" in artifact
        assert "return times_two(value);" in artifact
        assert "UnaryFunc" not in artifact
        assert "Function(" not in artifact

    glslang = shutil.which("glslangValidator")
    if glslang:
        hlsl_path = tmp_path / "callable_alias.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        hlsl_result = subprocess.run(
            [
                glslang,
                "-D",
                "-S",
                "comp",
                "-e",
                "CSMain",
                "-V",
                str(hlsl_path),
                "-o",
                str(tmp_path / "callable_alias_hlsl.spv"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert hlsl_result.returncode == 0, hlsl_result.stdout + hlsl_result.stderr

        glsl_path = tmp_path / "callable_alias.comp"
        glsl_path.write_text(glsl, encoding="utf-8")
        glsl_result = subprocess.run(
            [
                glslang,
                "-S",
                "comp",
                "-V",
                str(glsl_path),
                "-o",
                str(tmp_path / "callable_alias_glsl.spv"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert glsl_result.returncode == 0, glsl_result.stdout + glsl_result.stderr

    dxc = shutil.which("dxc")
    if dxc:
        hlsl_path = tmp_path / "callable_alias.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        dxc_result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "callable_alias.dxil"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert dxc_result.returncode == 0, dxc_result.stdout + dxc_result.stderr


def test_codegen_rejects_runtime_callable_alias_value():
    code = """
    typedef void (*RadixFunc)(thread float2*, thread float2*);
    typedef RadixFunc SelectedRadixFunc;

    void invoke(SelectedRadixFunc function,
                thread float2* values,
                thread float2* scratch) {
        function(values, scratch);
    }
    """

    with pytest.raises(MetalCallableAliasLoweringError) as excinfo:
        convert(code)

    error = excinfo.value
    assert error.alias_name == "SelectedRadixFunc"
    assert error.signature == (
        "void (*SelectedRadixFunc)(thread float2*, thread float2*)"
    )
    assert error.usage == "VariableNode 'function' vtype"
    assert error.project_diagnostic_code == (
        "project.translate.metal-callable-alias-unsupported"
    )
    assert error.missing_capabilities == ("metal.runtime-callable-alias-lowering",)
    assert error.source_location["line"] == 3


def test_codegen_callable_alias_does_not_match_shadowing_value_names():
    code = """
    typedef void (*RadixFunc)(thread float2*, thread float2*);

    void consume_scalar(int RadixFunc) {
        int value = RadixFunc;
    }

    void ordinary_void_function(thread float2* value) {
        value[0] = float2(0.0);
    }
    """
    result = convert(code)

    assert "void consume_scalar(int RadixFunc)" in result
    assert "void ordinary_void_function(thread vec2* value)" in result
    assert "typedef void RadixFunc;" not in result
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


def test_codegen_lowers_dispatch_bool_callback_from_mlx_fp_quantized_nax():
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

    assert "dispatch_bool" not in compact
    assert "[&]" not in compact
    assert "if ((!is_unaligned_sm))" in compact
    assert "if (true)" in compact
    assert "if (false)" in compact
    assert "workgroupBarrier();" in compact
    assert "Unhandled expression" not in compact
    assert parse_crossgl(result) is not None


def test_codegen_sanitizes_template_id_value_expression_from_mlx_gemm_gather_nax():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 8f0e8b14e0fc028df8618684583af9bef44647b8
    # Path: mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.h
    code = """
    void run(bool is_unaligned_sm) {
      dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        auto do_gemm = gemm_loop<
            T,
            SM,
            kAlignedM.value,
            AccumType>;
        if constexpr (kAlignedM.value) {
          do_gemm();
        }
      });
    }
    """
    result = convert(code)
    compact = normalize(result)

    assert "gemm_loop_u3cT_u2cSM_u2ctrue_u2cAccumType_u3e" in compact
    assert "gemm_loop_u3cT_u2cSM_u2cfalse_u2cAccumType_u3e" in compact
    assert "gemm_loop<" not in compact
    assert "if (true)" in compact
    assert "if (false)" in compact
    assert "Unhandled expression" not in compact
    assert parse_crossgl(result) is not None


def test_codegen_lowers_nested_dispatch_bool_callbacks():
    code = """
    void run(bool align_m, bool align_n, device uint* out) {
      dispatch_bool(align_m, [&](auto kAlignedM) {
        dispatch_bool(align_n, [&](auto kAlignedN) {
          if constexpr (kAlignedM.value && kAlignedN.value) {
            out[0] = 1;
          }
        });
      });
    }
    """

    result = convert(code)
    compact = normalize(result)

    assert compact.count("if (align_m)") == 1
    assert compact.count("if (align_n)") == 2
    assert "if (true && true)" in compact
    assert "if (true && false)" in compact
    assert "if (false && true)" in compact
    assert "if (false && false)" in compact
    assert "dispatch_bool" not in compact
    assert parse_crossgl(result) is not None


def test_codegen_rejects_dispatch_bool_callback_without_parameter():
    code = """
    void run(bool condition) {
      dispatch_bool(condition, [&]() {});
    }
    """

    with pytest.raises(MetalCallableLoweringError) as error:
        convert(code)

    assert error.value.project_diagnostic_code == (
        "project.translate.metal-callable-unsupported"
    )
    assert "exactly one integral-constant parameter" in error.value.reason


@pytest.mark.parametrize(
    ("callback", "reason"),
    [
        ("[=](auto flag) { out[0] = 1u; }", "reference-default capture"),
        ("[&](auto flag) { return; }", "callback-local return statements"),
    ],
)
def test_codegen_rejects_unsafe_dispatch_bool_callback_shapes(callback, reason):
    code = f"""
    void run(bool condition, device uint* out) {{
      dispatch_bool(condition, {callback});
    }}
    """

    with pytest.raises(MetalCallableLoweringError) as error:
        convert(code)

    assert reason in error.value.reason


def test_codegen_preserves_dispatch_bool_call_with_named_functor():
    code = """
    void run(bool condition, Handler handler) {
      dispatch_bool(condition, handler);
    }
    """

    result = convert(code)

    assert "dispatch_bool(condition, handler);" in result
    assert parse_crossgl(result) is not None


def test_codegen_nested_dispatch_bool_parameter_shadowing_uses_inner_value():
    code = """
    void run(bool outer, bool inner, device uint* out) {
      dispatch_bool(outer, [&](auto flag) {
        dispatch_bool(inner, [&](auto flag) {
          if constexpr (flag.value) {
            out[0] = 1u;
          }
        });
      });
    }
    """

    compact = normalize(convert(code))

    assert compact.count("if (true)") == 2
    assert compact.count("if (false)") == 2


def test_codegen_rejects_callback_helper_without_semantic_lowering():
    code = """
    void run(device uint* out) {
      apply_callback([&](auto value) {
        out[0] = value.value ? 1u : 0u;
      });
    }
    """

    with pytest.raises(MetalCallableLoweringError) as error:
        convert(code)

    assert error.value.helper == "apply_callback"
    assert error.value.capture == "&"
    assert error.value.enclosing_function == "run"
    assert error.value.source_location["line"] == 3
    assert "invocation count" in error.value.suggested_action


def test_codegen_names_unnamed_template_tag_parameters_from_mlx_nax():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/steel/attn/nax.h
    code = """
    template <
        typename CTile,
        typename ATile,
        typename BTile,
        bool transpose_a,
        bool transpose_b>
    void tile_matmad_nax(thread CTile& C,
                         thread ATile& A,
                         bool_constant<transpose_a>,
                         thread BTile& B,
                         bool_constant<transpose_b>) {
        const short TM = transpose_a ? 1 : 2;
        C.val = TM;
    }
    """
    result = convert(code)

    assert "bool_constant<transpose_a> _unnamed_param_2" in result
    assert "bool_constant<transpose_b> _unnamed_param_4" in result
    assert parse_crossgl(result) is not None


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

    assert "void update(inout threadgroup Payload scratch" in crossgl
    assert "inout device float[] values" in crossgl
    assert "constant uint& count" in crossgl
    assert "inout thread float localValue" in crossgl
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


def test_codegen_reference_return_helper_reparses_from_pytorch_linalg():
    # Reduced from pytorch/pytorch aten/src/ATen/native/mps/kernels/LinearAlgebra.metal.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    template<bool upper>
    float& get_ref(device float* A, uint row, uint col, uint N) {
        return A[row * N + col];
    }

    kernel void factorDiagonalBlock(device float* A [[buffer(0)]],
                                    constant uint& N [[buffer(1)]]) {
        uint row = 0;
        uint col = 0;
        get_ref<true>(A, row, col, N) = 1.0f;
    }
    """
    crossgl = convert(code)

    assert "float get_ref(device float* A" in crossgl
    assert "float& get_ref" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_preserves_threadgroup_imageblock_local_pointer_roundtrip():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct TransparentFragmentValues {
        half4 color;
    };

    kernel void resolve(
        imageblock<TransparentFragmentValues> blockData [[threadgroup_imageblock]],
        ushort2 localThreadID [[thread_position_in_threadgroup]]) {
        threadgroup_imageblock TransparentFragmentValues* fragmentValues =
            blockData.data(localThreadID);
        half4 color = fragmentValues->color;
    }
    """
    crossgl = convert(code)

    assert "threadgroup_imageblock TransparentFragmentValues* fragmentValues" in crossgl

    metal = MetalCodeGen().generate(parse_crossgl(crossgl))
    assert "threadgroup_imageblock TransparentFragmentValues* fragmentValues" in metal
    assert "thread TransparentFragmentValues* fragmentValues" not in metal


def test_codegen_lowers_metal_simd_group_intrinsics_to_crossgl_wave_ops():
    # Metal SIMD-group (wave) intrinsics must lower to canonical CrossGL Wave*
    # ops. Otherwise they leak unchanged into the IR and the DirectX/SPIR-V
    # backends emit uncompilable or silently-defaulted code. Mirrors the SIMD
    # reductions that MLX metal kernels rely on (reduce / softmax / gemv).
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void wave_ops(
        device const float* in [[buffer(0)]],
        device float* outf [[buffer(1)]],
        device uint* outu [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
        float v = in[gid];
        uint u = outu[gid];
        outf[gid] = simd_sum(v) + simd_product(v) + simd_min(v) + simd_max(v)
            + simd_prefix_exclusive_sum(v) + simd_prefix_exclusive_product(v)
            + simd_broadcast_first(v) + simd_broadcast(v, 0u)
            + simd_shuffle_and_fill_up(v, 0.0f, 1u);
        outu[gid] = simd_and(u) + simd_or(u) + simd_xor(u) + simd_ballot(v > 0.0);
        if (simd_all(v > 0.0) && simd_any(v < 0.0)) {
            outf[gid] = v;
        }
    }
    """
    generated = convert(code)

    for token in (
        "WaveActiveSum(v)",
        "WaveActiveProduct(v)",
        "WaveActiveMin(v)",
        "WaveActiveMax(v)",
        "WavePrefixSum(v)",
        "WavePrefixProduct(v)",
        "WaveReadLaneFirst(v)",
        "WaveReadLaneAt(v",
        "WaveShuffleAndFillUp(v, 0.0f, 1u)",
        "WaveActiveBitAnd(u)",
        "WaveActiveBitOr(u)",
        "WaveActiveBitXor(u)",
        "WaveActiveAllTrue(",
        "WaveActiveAnyTrue(",
        "WaveActiveBallot(",
    ):
        assert token in generated, f"missing {token} in:\n{generated}"

    # The raw Metal spellings must not leak into the CrossGL IR.
    for leaked in (
        "simd_sum",
        "simd_product",
        "simd_min",
        "simd_max",
        "simd_prefix_exclusive",
        "simd_broadcast",
        "simd_shuffle_and_fill_up",
        "simd_and",
        "simd_or",
        "simd_xor",
        "simd_all",
        "simd_any",
        "simd_ballot",
    ):
        assert leaked not in generated, f"leaked {leaked} in:\n{generated}"

    assert parse_crossgl(generated) is not None


def test_codegen_binds_metal_simd_intrinsics_by_source_signature(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct complex64_t {
        float real;
        float imag;
    };

    uint64_t simd_shuffle_down(uint64_t data, uint delta) {
        return data;
    }

    bool simd_shuffle_down(bool data, uint delta) {
        return simd_shuffle_down(uint(data), delta) != 0u;
    }

    complex64_t simd_shuffle_down(complex64_t data, uint delta) {
        return {
            simd_shuffle_down(data.real, delta),
            simd_shuffle_down(data.imag, delta)
        };
    }

    float simd_shuffle(float data, uint lane) {
        return data;
    }

    kernel void shuffle_values(
        device float* out_buffer [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        uint delta = 1u;
        bool flag = true;
        uint64_t wide = uint64_t(gid);
        out_buffer[gid] = simd_shuffle_down(float(gid), delta);
        out_buffer[gid] += simd_shuffle(float(gid), delta);
        out_buffer[gid] += metal::simd_shuffle(float(gid), ushort(delta));
        flag = simd_shuffle_down(flag, delta);
        wide = simd_shuffle_down(wide, delta);
    }
    """

    generated = convert(code)

    assert generated.count("WaveShuffleDown(data.real, delta)") == 1
    assert generated.count("WaveShuffleDown(data.imag, delta)") == 1
    assert "WaveShuffleDown(uint(data), delta)" in generated
    assert "WaveShuffleDown(float(gid), delta)" in generated
    assert "simd_shuffle_down(flag, delta)" in generated
    assert "simd_shuffle_down(wide, delta)" in generated
    assert "simd_shuffle(float(gid), delta)" in generated
    assert "WaveReadLaneAt(float(gid), uint16(delta))" in generated
    ast = parse_crossgl(generated)
    assert ast is not None

    glsl = GLSLCodeGen().generate(ast)
    assert "subgroupShuffleDown(data.real, delta)" in glsl
    assert "subgroupShuffleDown(data.imag, delta)" in glsl
    assert "subgroupShuffleDown(uint(data), delta)" in glsl
    assert "subgroupShuffleDown(float(gid), delta)" in glsl
    assert "simd_shuffle(float(gid), delta)" in glsl
    assert "subgroupShuffle(float(gid)," in glsl

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        source = tmp_path / "metal-simd-overloads.comp"
        output = tmp_path / "metal-simd-overloads.spv"
        source.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [
                glslang,
                "--target-env",
                "opengl",
                "--target-env",
                "spirv1.3",
                "-S",
                "comp",
                str(source),
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_codegen_mlx_gemv_materialized_array_shuffle_uses_builtin_overload():
    # Regression for #1504, reduced from MLX GEMV: the concrete helper's local
    # float array and for-loop delta must not bind unrelated user overloads.
    code = """
    bool simd_shuffle_down(bool value, ushort delta) {
        return value;
    }

    uint64_t simd_shuffle_down(uint64_t value, ushort delta) {
        return value;
    }

    template <typename T>
    T shuffle_local(T value, uint index) {
        T result[1] = {value};
        for (ushort sn = 1; sn > 0; sn >>= 1) {
            result[index] = simd_shuffle_down(result[index], sn);
        }
        return result[index];
    }

    float use_shuffles(float value, bool flag, uint64_t wide, ushort delta) {
        float shuffled = shuffle_local<float>(value, 0u);
        flag = simd_shuffle_down(flag, delta);
        wide = simd_shuffle_down(wide, delta);
        return shuffled;
    }
    """

    generated = convert(code)
    normalized = normalize(generated)

    assert (
        "float shuffle_local_float(float value, uint index) { "
        "float[1] result = {value}; "
        "for (uint16 sn = 1; sn > 0; sn >>= 1) { "
        "result[index] = WaveShuffleDown(result[index], sn); } "
        "return result[index]; }"
    ) in normalized
    assert "flag = simd_shuffle_down(flag, delta);" in generated
    assert "wide = simd_shuffle_down(wide, delta);" in generated
    assert parse_crossgl(generated) is not None


def test_codegen_mlx_gemvt_materialized_lane_expression_uses_builtin_overload(
    tmp_path,
):
    code = """
    struct complex64_t {
        float real;
        float imag;
    };

    bool simd_shuffle_down(bool value, ushort delta) {
        return value;
    }

    uint64_t simd_shuffle_down(uint64_t value, ushort delta) {
        return value;
    }

    complex64_t simd_shuffle_down(complex64_t value, uint delta) {
        return value;
    }

    complex64_t preserve_complex_shuffle(complex64_t value, ushort sm) {
        return simd_shuffle_down(value, 4 * sm);
    }

    template <typename T, int SN>
    T shuffle_scaled(T value, uint index) {
        T result[1] = {value};
        for (ushort sm = 1; sm > 0; sm >>= 1) {
            result[index] = simd_shuffle_down(result[index], SN * sm);
        }
        return result[index];
    }

    kernel void use_shuffles(
        device float* out_buffer [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out_buffer[gid] = shuffle_scaled<float, 4>(float(gid), 0u);
    }
    """

    generated = convert(code)
    normalized = normalize(generated)

    assert (
        "float shuffle_scaled_float_4(float value, uint index) { "
        "float[1] result = {value}; "
        "for (uint16 sm = 1; sm > 0; sm >>= 1) { "
        "result[index] = WaveShuffleDown(result[index], 4 * sm); } "
        "return result[index]; }"
    ) in normalized
    assert (
        "complex64_t preserve_complex_shuffle(complex64_t value, uint16 sm) { "
        "return simd_shuffle_down(value, 4 * sm); }"
    ) in normalized
    ast = parse_crossgl(generated)
    assert ast is not None

    glsl = GLSLCodeGen().generate(ast)
    assert "subgroupShuffleDown(result[index], (4 * int(sm)))" in glsl
    assert "simd_shuffle_down(result[index]" not in glsl
    assert "return simd_shuffle_down(value" in glsl

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        source = tmp_path / "metal-materialized-lane-expression.comp"
        output = tmp_path / "metal-materialized-lane-expression.spv"
        source.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [
                glslang,
                "--target-env",
                "opengl",
                "--target-env",
                "spirv1.3",
                "-S",
                "comp",
                str(source),
                "-o",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_codegen_reports_ambiguous_metal_builtin_user_overloads():
    code = """
    char simd_shuffle_down(char data, uint delta) {
        return data;
    }

    int8_t simd_shuffle_down(int8_t data, uint delta) {
        return data;
    }

    kernel void shuffle_value(device char* output [[buffer(0)]]) {
        char value = 1;
        uint delta = 1u;
        output[0] = simd_shuffle_down(value, delta);
    }
    """

    with pytest.raises(MetalBuiltinOverloadResolutionError) as exc_info:
        convert(code)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-builtin-overload-ambiguous"
    )
    assert diagnostic.missing_capabilities == ("metal.builtin-overload-resolution",)
    assert diagnostic.function_name == "simd_shuffle_down"
    assert diagnostic.argument_types == ("int8", "uint")
    assert set(diagnostic.candidates) == {
        "simd_shuffle_down(char, uint)",
        "simd_shuffle_down(int8_t, uint)",
    }


def test_metal_simd_reductions_reach_backend_wave_instructions():
    # End to end: Metal simd_* reductions become real subgroup instructions on
    # the DirectX and SPIR-V (Vulkan) backends instead of leaking into output.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void wave_reduce(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        float v = in[gid];
        ushort lane = ushort(gid & 31u);
        out[gid] = simd_sum(v) + simd_prefix_exclusive_sum(v)
            + simd_broadcast(v, lane);
    }
    """
    ast = parse_crossgl(convert(code))

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "WaveActiveSum(" in hlsl
    assert "WavePrefixSum(" in hlsl
    # simd_broadcast carries a ushort lane (min16uint); DirectX must accept it.
    assert "WaveReadLaneAt(" in hlsl
    assert "simd_sum" not in hlsl
    assert "simd_broadcast" not in hlsl

    spirv = VulkanSPIRVCodeGen().generate(ast)
    assert "OpGroupNonUniformFAdd" in spirv
    # WaveReadLaneAt lowers to a dynamic-lane shuffle, not a constant broadcast.
    assert "OpGroupNonUniformShuffle" in spirv
    assert "cannot lower unknown function" not in spirv


if __name__ == "__main__":
    pytest.main()


def test_codegen_resolves_local_conditional_t_alias_to_integer_pack_type():
    # Reduced from mlx quantized `affine_quantize`: an uninstantiated generic
    # kernel aliases its packed accumulator type through metal::conditional_t.
    # The alias must resolve to an integer type so the bit-packing math stays
    # integer-typed instead of the alias defaulting to float (which produces an
    # invalid float bitwise operation in the SPIR-V backend).
    code = """
    #include <metal_stdlib>
    using namespace metal;

    template <typename T, const int bits>
    [[kernel]] void pack(
        device uint8_t* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        using OutType = metal::conditional_t<bits == 5, uint64_t, uint32_t>;
        OutType output = 0;
        output |= 7;
        out[gid] = output & 0xff;
    }
    """
    result = convert(code)
    assert "uint output = 0" in result
    assert "OutType" not in result
    assert parse_crossgl(result) is not None


def test_codegen_declares_bitwise_value_template_parameter_as_int():
    # Reduced from mlx quantized: the `bits` non-type template parameter drives
    # the power-of-two idiom `bits & (bits - 1)`. In an uninstantiated generic
    # kernel it has no declaration and would default to float, so it is declared
    # as an integer placeholder to keep the bitwise math integer-typed.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    template <typename T, const int bits>
    [[kernel]] void quant(
        device uint* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        int power_of_2 = (bits & (bits - 1)) == 0;
        out[gid] = power_of_2;
    }
    """
    result = convert(code)
    assert "int bits = 0;" in result
    assert parse_crossgl(result) is not None


def test_codegen_leaves_array_extent_value_template_parameter_undeclared():
    # A value template parameter used only as an array extent (as in mlx fft's
    # `threadgroup float2 shared[tg_mem_size]`) must not be turned into an
    # injected runtime local; doing so would disturb array sizing. Only
    # parameters consumed by bitwise/shift operators receive the placeholder.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    template <int tg_mem_size, typename T>
    [[kernel]] void fftlike(
        device T* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        threadgroup float shared_mem[tg_mem_size];
        out[gid] = shared_mem[0];
    }
    """
    result = convert(code)
    assert "int tg_mem_size = 0;" not in result


def test_codegen_keeps_unresolved_struct_template_local_alias_uninlined():
    # A body-local alias whose target still contains a generic argument cannot
    # be mapped to a concrete CrossGL struct and must not become a scalar.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    template <typename T>
    struct ReadWriter { T value; };

    template <typename T>
    [[kernel]] void rw(
        device float* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        using read_writer_t = ReadWriter<T>;
        read_writer_t r;
        out[gid] = r.value;
    }
    """
    result = convert(code)
    # The struct alias resolves to a non-scalar type, so it is not inlined to a
    # primitive (which would otherwise misdeclare the variable as `uint r`).
    assert "uint r " not in result
    assert "float r " not in result


def test_codegen_inlines_local_alias_to_materialized_struct():
    code = """
    struct ReadWriter_float {
      int value;
      ReadWriter_float(int value_) : value(value_) {}
    };

    kernel void rw(device int* out [[buffer(0)]]) {
      using read_writer_t = ReadWriter_float;
      read_writer_t writer = read_writer_t(3);
      out[0] = writer.value;
    }
    """

    result = convert(code)

    assert "ReadWriter_float writer = ReadWriter_float(3);" in result
    assert "read_writer_t" not in result
    assert parse_crossgl(result) is not None


def test_codegen_preserves_static_struct_integer_constants():
    code = """
    struct Tile {
        static constant constexpr const int WIDTH = 4 / 2;
        float values[WIDTH];
    };
    """

    result = convert(code)
    assert "static constant int WIDTH = 4 / 2;" in result

    parsed = parse_crossgl(result)
    tile = next(struct for struct in parsed.structs if struct.name == "Tile")
    width = next(member for member in tile.members if member.name == "WIDTH")
    assert {attribute.name for attribute in width.attributes} >= {
        "static",
        "constant",
    }
    assert width.default_value.operator == "/"


def test_codegen_materializes_qualified_static_constant_initializers(tmp_path):
    code = """
    typedef bfloat bfloat16_t;

    struct Values {
        static constexpr constant int base = 40;
        static constexpr constant int answer = base + 2;
    };

    template <typename T>
    struct Limits {
        static constexpr constant T max = T(13);
    };

    template <>
    struct Limits<float> {
        static constexpr constant float max =
            metal::numeric_limits<float>::infinity();
        static constexpr constant float finite_max =
            metal::numeric_limits<float>::max();
    };

    template <>
    struct Limits<bfloat16_t> {
        static constexpr constant bfloat16_t max =
            metal::numeric_limits<bfloat16_t>::infinity();
    };

    int answer() {
        return Values::answer;
    }

    float clamp_float(float value) {
        return min(value, Limits<float>::max);
    }

    bfloat16_t max_bfloat() {
        return Limits<bfloat16_t>::max;
    }

    float alias_max() {
        using FloatLimits = Limits<float>;
        return FloatLimits::max;
    }

    kernel void use_limits(
        device float* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = clamp_float(float(gid));
    }
    """

    crossgl = convert(code)

    assert "return 42;" in crossgl
    assert "return min(value, (asfloat(0x7f800000u)));" in crossgl
    assert "return min(value, 13);" not in crossgl
    assert "return (bfloat16(asfloat(0x7f800000u)));" in crossgl
    assert "FloatLimits" not in crossgl
    assert "static constant float finite_max = asfloat(0x7f7fffffu);" in crossgl
    assert "_u3a_u3a" not in crossgl

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    spirv = VulkanSPIRVCodeGen().generate(ast)

    assert "asfloat(2139095040u)" in hlsl
    assert "return half(asfloat(2139095040u));" in hlsl
    assert "bfloat16(" not in hlsl
    assert "uintBitsToFloat(2139095040u)" in glsl
    assert "OpConstant" in spirv and " 2139095040" in spirv
    assert "OpBitcast" in spirv
    assert "WARNING" not in spirv

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "static-constant.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl_path = tmp_path / "static-constant.comp"
        glsl_path.write_text(glsl, encoding="utf-8")
        subprocess.run(
            [
                glslang,
                "--target-env",
                "opengl",
                "--target-env",
                "spirv1.3",
                "-S",
                "comp",
                str(glsl_path),
                "-o",
                str(tmp_path / "static-constant-opengl.spv"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        spirv_path = tmp_path / "static-constant.spvasm"
        binary_path = tmp_path / "static-constant.spv"
        spirv_path.write_text(spirv, encoding="utf-8")
        subprocess.run(
            [
                spirv_as,
                "--target-env",
                "vulkan1.1",
                str(spirv_path),
                "-o",
                str(binary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [spirv_val, "--target-env", "vulkan1.1", str(binary_path)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_codegen_materializes_static_constant_through_decltype_expression_owner():
    crossgl = convert("""
        struct Tile {
            static constexpr int count = 2;
        };

        struct State {
            Tile tile;
        };

        int tile_count(State self) {
            return decltype(self.tile)::count;
        }
        """)

    assert "return 2;" in crossgl
    assert "decltype" not in crossgl
    assert "_u3a_u3a" not in crossgl


def test_codegen_rejects_uninferable_decltype_static_constant_owner():
    code = """
    struct Tile {
        static constexpr int count = 2;
    };

    int tile_count() {
        return decltype(missing_tile)::count;
    }
    """

    with pytest.raises(MetalStaticConstantResolutionError) as error:
        convert(code)

    assert error.value.owner == "decltype(missing_tile)"
    assert error.value.member == "count"
    assert "infer" in error.value.reason


def test_codegen_rejects_decltype_owner_without_static_constant():
    code = """
    struct Tile {
        int count;
    };

    struct State {
        Tile tile;
    };

    int tile_count(State self) {
        return decltype(self.tile)::count;
    }
    """

    with pytest.raises(MetalStaticConstantResolutionError) as error:
        convert(code)

    assert error.value.owner == "Tile"
    assert error.value.member == "count"
    assert "no compile-time static member" in error.value.reason


def test_codegen_rejects_cyclic_static_constant_initializers():
    code = """
    struct Cycle {
        static constexpr int first = second;
        static constexpr int second = first;
    };

    int value() {
        return Cycle::first;
    }
    """

    with pytest.raises(
        ValueError,
        match=r"initializer dependency chain is cyclic .*Cycle::first",
    ):
        convert(code)


def test_codegen_does_not_inline_mutable_static_struct_members():
    crossgl = convert("""
        struct Counter {
            static int value = 1;
        };

        int read_counter() {
            return Counter::value;
        }
        """)

    assert "return Counter_u3a_u3avalue;" in crossgl
    assert "return 1;" not in crossgl


def test_codegen_resolves_unique_namespaced_static_constant_owner():
    crossgl = convert("""
        namespace Numeric {
        struct Limits {
            static constexpr float max =
                metal::numeric_limits<float>::infinity();
        };
        }

        using namespace Numeric;

        float qualified_max() {
            return Numeric::Limits::max;
        }

        float imported_max() {
            return Limits::max;
        }
        """)

    assert crossgl.count("return (asfloat(0x7f800000u));") == 2
    assert "_u3a_u3amax" not in crossgl


def test_codegen_rejects_ambiguous_namespaced_static_constant_owner():
    code = """
    namespace First {
    struct Limits {
        static constexpr float max = 1.0;
    };
    }

    namespace Second {
    struct Limits {
        static constexpr float max = 2.0;
    };
    }

    using namespace First;
    using namespace Second;

    float max_value() {
        return Limits::max;
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"Cannot materialize Metal static constant Limits::max: "
            r"multiple visible struct declarations"
        ),
    ):
        convert(code)


def test_const_for_loop_nonzero_indices_reach_vulkan_stores(tmp_path):
    source = tmp_path / "const_for_loop.metal"
    source.write_text("""
        template <typename T, T v>
        struct integral_constant { static constexpr T value = v; };
        template <int Value>
        using Int = integral_constant<int, Value>;
        template <int start, int stop, int step, typename F>
        constexpr void const_for_loop(F f) {
          if constexpr (start < stop) {
            constexpr auto index = Int<start>{};
            f(index);
            const_for_loop<start + step, stop, step, F>(f);
          }
        }
        template <typename T, T lhs, typename U, U rhs>
        constexpr auto operator*(
            integral_constant<T, lhs>, integral_constant<U, rhs>) {
          return integral_constant<decltype(lhs * rhs), lhs * rhs>{};
        }

        struct Writer {
          template <typename Row, typename Col>
          static void store(device int* out, Row row, Col col) {
            out[row.value + col.value] = row.value + col.value;
          }
        };
        struct Tile {
          template <typename U>
          void run(device U* out) {
            const_for_loop<0, 2, 1>([&](auto row) {
              const_for_loop<0, 2, 1>([&](auto col) {
                Writer::store(out, row * Int<4>{}, col * Int<1>{});
              });
            });
          }
        };
        kernel void write_indices(device int* out [[buffer(0)]]) {
          Tile tile;
          tile.run(out);
        }
        """)

    spirv = crosstl.translate(str(source), backend="vulkan", format_output=False)

    assert spirv.count("OpStore") == 4
    constant_ids = {
        int(value): result_id
        for result_id, value in re.findall(
            r"^(%\d+) = OpConstant %\d+ (-?\d+)$", spirv, re.MULTILINE
        )
    }
    assert {0, 1, 4} <= constant_ids.keys()
    for left, right in ((0, 0), (0, 1), (4, 0), (4, 1)):
        operation = re.compile(
            rf"^%\d+ = OpIAdd %\d+ {re.escape(constant_ids[left])} "
            rf"{re.escape(constant_ids[right])}$",
            re.MULTILINE,
        )
        assert len(operation.findall(spirv)) == 2
    assert "WARNING" not in spirv

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        return
    assembly = tmp_path / "const_for_loop.spvasm"
    binary = tmp_path / "const_for_loop.spv"
    assembly.write_text(spirv)
    subprocess.run(
        [spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_metal_cooperative_matrix_operations_round_trip_through_shared_ir():
    source = """
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;

    kernel void matrix_roundtrip(
        device float* input [[buffer(0)]],
        device float* output [[buffer(1)]]) {
      simdgroup_matrix<float, 8, 8> left;
      simdgroup_matrix<float, 8, 8> right;
      simdgroup_matrix<float, 8, 8> accumulator;
      simdgroup_load(left, input, 8);
      left.thread_elements()[0] = 1.0f;
      simdgroup_matrix<float, 8, 8> product = left * right;
      simdgroup_matrix<float, 8, 8> sum = left + right;
      simdgroup_matrix<float, 8, 8> difference = left - right;
      simdgroup_matrix<float, 8, 8> negated = -left;
      simdgroup_load(left, input, 8, 0, true);
      simdgroup_multiply_accumulate(accumulator, left, right, product);
      simdgroup_store(accumulator, output, 8, 0, true);
    }
    """

    crossgl = convert(source)

    assert "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified>" in crossgl
    assert "cooperative_matrix_load(left, input, 8);" in crossgl
    assert "cooperative_matrix_element(left, 0) = 1.0f;" in crossgl
    assert "cooperative_matrix_multiply(left, right)" in crossgl
    assert "cooperative_matrix_add(left, right)" in crossgl
    assert "cooperative_matrix_subtract(left, right)" in crossgl
    assert "cooperative_matrix_negate(left)" in crossgl
    assert (
        "cooperative_matrix_multiply_accumulate(accumulator, left, right, product);"
        in crossgl
    )
    assert "cooperative_matrix_store(output, accumulator, 8, 0, true);" in crossgl

    regenerated = MetalCodeGen().generate(parse_crossgl(crossgl))

    assert "#include <metal_simdgroup_matrix>" in regenerated
    assert "simdgroup_matrix<float, 8, 8> left;" in regenerated
    assert "left.thread_elements()[0] = 1.0;" in regenerated
    assert "simdgroup_matrix<float, 8, 8> product = (left * right);" in regenerated
    assert "simdgroup_matrix<float, 8, 8> sum = (left + right);" in regenerated
    assert "simdgroup_matrix<float, 8, 8> difference = (left - right);" in regenerated
    assert "simdgroup_matrix<float, 8, 8> negated = (-left);" in regenerated
    assert "cooperative_matrix_load(left, input, 8, 0, true);" in crossgl
    assert "cooperative_matrix_store(output, accumulator, 8, 0, true);" in crossgl
    assert "simdgroup_load(left, input, 8, 0, true);" in regenerated
    assert (
        "simdgroup_multiply_accumulate(accumulator, left, right, product);"
        in regenerated
    )
    assert "simdgroup_store(accumulator, output, 8, 0, true);" in regenerated
    assert "cooperative_matrix_" not in regenerated


def test_codegen_materializes_defaulted_bool_and_explicit_specialization():
    code = """
    template <bool UseAlternate = false>
    float select_value(float primary, float alternate) {
        return UseAlternate ? alternate : primary;
    }

    float select_both(float primary, float alternate) {
        return select_value(primary, alternate)
            + select_value<true>(primary, alternate);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "float select_value(float primary, float alternate)" in crossgl
    assert "return false ? alternate : primary;" in crossgl
    assert "float select_value_true(float primary, float alternate)" in crossgl
    assert "return true ? alternate : primary;" in crossgl
    assert "select_value(primary, alternate)" in crossgl
    assert "select_value_true(primary, alternate)" in crossgl
    assert not re.search(r"\bUseAlternate\b", crossgl)

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert not re.search(r"\bUseAlternate\b", hlsl)
    assert "select_value_true" in hlsl
    assert "false ? alternate : primary" in hlsl
    assert "true ? alternate : primary" in hlsl


def test_codegen_materializes_pinned_mlx_perform_fft_bool_specializations():
    # Reduced from mlx/backend/metal/kernels/fft.h at
    # 4367c73b60541ddd5a266ce4644fd93d20223b6e.
    code = """
    template <bool rader = false>
    int perform_fft(int value) {
        return rader ? value + 11 : value + 2;
    }

    int run_fft_modes(int value) {
        return perform_fft(value) + perform_fft<true>(value);
    }
    """

    crossgl = convert_without_preprocessing(code)
    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    for output in (crossgl, hlsl):
        assert not re.search(r"\brader\b", output)
        assert "perform_fft" in output
        assert "perform_fft_true" in output
        assert "false ?" in output
        assert "true ?" in output
        assert "value + 11" in output
        assert "value + 2" in output


def test_codegen_propagates_nested_value_template_specializations():
    code = """
    template <bool Inner = false>
    float leaf(float value) {
        return Inner ? value : -value;
    }

    template <bool Outer = false>
    float branch(float value) {
        return leaf<Outer>(value);
    }

    float run_branches(float value) {
        return branch(value) + branch<true>(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "return leaf_false(value);" in crossgl
    assert "return leaf_true(value);" in crossgl
    assert "return false ? value : (-value);" in crossgl
    assert "return true ? value : (-value);" in crossgl
    assert "branch(value) + branch_true(value)" in crossgl
    assert not re.search(r"\b(?:Inner|Outer)\b", crossgl)
    assert parse_crossgl(crossgl) is not None


def test_codegen_resolves_later_value_default_from_earlier_binding_in_extent():
    code = """
    template <int Base = 2, int Width = Base + 1>
    int extent_value(int index) {
        int values[Width];
        values[0] = Base;
        return values[index] + Width;
    }

    int run_extent() {
        return extent_value(0);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "int[3] values;" in crossgl
    assert "values[0] = 2;" in crossgl
    assert "return values[index] + 3;" in crossgl
    assert not re.search(r"\b(?:Base|Width)\b", crossgl)
    assert parse_crossgl(crossgl) is not None


def test_codegen_folds_owner_dependent_constexpr_static_member_chains():
    code = """
    template <typename Scalar, int Bits, int Word = 8>
    inline constexpr short pack_factor() {
        return Word / Bits;
    }

    template <int Bits, int Word = 8>
    inline constexpr short bytes_per_pack() {
        constexpr int power_of_two = (Bits & (Bits - 1)) == 0;
        return power_of_two ? (Word / 8) : (Bits == 5 ? 5 : 3);
    }

    template <typename T, int Bits>
    struct Loader {
        static constexpr short factor = pack_factor<T, Bits>();
        static constexpr short bytes = bytes_per_pack<Bits>();
        static constexpr short reads = 32 / factor;
    };

    kernel void load_factors(device int* out [[buffer(0)]]) {
        Loader<float, 2> two_bit;
        Loader<float, 3> three_bit;
        Loader<float, 4> four_bit;
        out[0] = two_bit.reads;
        out[1] = three_bit.bytes;
        out[2] = four_bit.reads;
    }
    """

    crossgl = convert(code)
    compact = normalize(crossgl)

    assert "struct Loader_float_2 { static int16 factor = 4;" in compact
    assert "static int16 bytes = 1;" in compact
    assert "static int16 reads = 8;" in compact
    assert "struct Loader_float_3 { static int16 factor = 2;" in compact
    assert "static int16 bytes = 3;" in compact
    assert "struct Loader_float_4 { static int16 factor = 2;" in compact
    assert "static int16 reads = 16;" in compact
    assert "pack_factor_float_2_8" not in crossgl
    assert "pack_factor_float_4_8" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_owner_constexpr_diagnostic_names_static_member_context():
    code = """
    template <int Bits, int Word = 8>
    constexpr short pack_factor() {
        return Word / Bits;
    }

    struct Loader_4 {
        static constexpr short factor = pack_factor<RuntimeBits>();
    };
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code, file_path="owner-constexpr.metal")

    diagnostic = exc_info.value
    assert diagnostic.owner == "Loader_4"
    assert diagnostic.member == "factor"
    assert diagnostic.function_name == "pack_factor"
    assert diagnostic.parameter_name == "Bits"
    assert diagnostic.argument_expression == "RuntimeBits"
    assert diagnostic.requested_specialization == "pack_factor<RuntimeBits>"
    assert diagnostic.reason == "remains dependent on RuntimeBits"
    assert diagnostic.source_location["file"] == "owner-constexpr.metal"
    assert "while resolving Loader_4::factor" in str(diagnostic)


def test_codegen_reuses_equivalent_owner_constexpr_helper_requests():
    code = """
    template <int Bits, int Word = 8>
    constexpr short pack_factor() {
        return Word / Bits;
    }

    struct LeftLoader {
        static constexpr short factor = pack_factor<4>();
    };

    struct RightLoader {
        static constexpr short factor = pack_factor<4>();
    };
    """
    ast = MetalParser(MetalLexer(code, preprocess=False).tokenize()).parse()
    converter = MetalToCrossGLConverter()

    crossgl = converter.generate(ast)

    assert len(converter.constexpr_helper_values) == 1
    assert crossgl.count("static int16 factor = 2;") == 2
    assert "pack_factor_4_8" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_rejects_cyclic_owner_constexpr_helper_requests():
    code = """
    template <int Bits = 4>
    constexpr short first_factor() {
        return second_factor<Bits>();
    }

    template <int Bits = 4>
    constexpr short second_factor() {
        return first_factor<Bits>();
    }

    struct Loader_4 {
        static constexpr short factor = first_factor<4>();
    };
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code, file_path="cyclic-owner.metal")

    diagnostic = exc_info.value
    assert diagnostic.owner == "Loader_4"
    assert diagnostic.member == "factor"
    assert diagnostic.function_name == "first_factor"
    assert diagnostic.requested_specialization == "first_factor<4>"
    assert diagnostic.reason == "has a cyclic constexpr helper dependency"
    assert diagnostic.source_location["file"] == "cyclic-owner.metal"


def test_codegen_value_template_substitution_respects_lexical_shadowing():
    code = """
    template <bool Flag = false>
    float shadowed(float value) {
        float selected = Flag ? value : -value;
        {
            bool Flag = true;
            if (Flag) {
                selected += 2.0f;
            }
        }
        return Flag ? selected : -selected;
    }

    float run_shadowed(float value) {
        return shadowed(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "float selected = false ? value : (-value);" in crossgl
    assert "bool Flag = true;" in crossgl
    assert "if (Flag)" in crossgl
    assert "return false ? selected : (-selected);" in crossgl
    assert "generic<Flag>" not in crossgl
    assert parse_crossgl(crossgl) is not None


@pytest.mark.parametrize(
    ("code", "argument_kind", "parameter_name", "expression", "reason", "message"),
    [
        (
            "template<typename T, int Count = T::extent> "
            "int count_value() { return Count; } "
            "int run() { return count_value(); }",
            "default",
            "Count",
            "T::extent",
            "remains dependent on T, extent",
            "Cannot materialize Metal function template 'count_value' for call "
            "'count_value(...)': default argument for 'Count' (T::extent) "
            "remains dependent on T, extent",
        ),
        (
            "template<bool Flag = false> int choose() { return Flag; } "
            "int run(bool Runtime) { return choose<Runtime>(); }",
            "explicit",
            "Flag",
            "Runtime",
            "remains dependent on Runtime",
            "Cannot materialize Metal function template 'choose' for call "
            "'choose<Runtime>': explicit argument for 'Flag' (Runtime) "
            "remains dependent on Runtime",
        ),
        (
            "template<int Count, bool Enabled = false> "
            "int count_value() { return Enabled ? Count : Count; } "
            "int run() { return count_value<>(); }",
            "missing",
            "Count",
            None,
            "was not supplied and has no declaration default",
            "Cannot materialize Metal function template 'count_value' for call "
            "'count_value<>': required argument for 'Count' was not supplied "
            "and has no declaration default",
        ),
    ],
)
def test_codegen_value_template_resolution_diagnostics_are_structured(
    code, argument_kind, parameter_name, expression, reason, message
):
    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code, file_path="template-diagnostic.metal")

    diagnostic = exc_info.value
    assert str(diagnostic) == message
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-template-argument-unresolved"
    )
    assert diagnostic.missing_capabilities == (
        "metal.value-template-argument-materialization",
    )
    assert diagnostic.argument_kind == argument_kind
    assert diagnostic.parameter_name == parameter_name
    assert diagnostic.argument_expression == expression
    assert diagnostic.reason == reason
    assert diagnostic.source_location["file"] == "template-diagnostic.metal"
    assert diagnostic.source_location["line"] == 1
    assert diagnostic.source_location["column"] == 1
    assert diagnostic.default_expression == (
        expression if argument_kind == "default" else None
    )
    assert diagnostic.explicit_argument == (
        expression if argument_kind == "explicit" else None
    )


@pytest.mark.parametrize("template_first", [True, False])
def test_codegen_same_name_template_and_overload_is_declaration_order_independent(
    template_first,
):
    template = """
    template <bool Alternate = false>
    float pick(float value) {
        return Alternate ? value : -value;
    }
    """
    overload = """
    int pick(int value) {
        return value + 1;
    }
    """
    declarations = template + overload if template_first else overload + template
    code = declarations + """
    float run_picks() {
        return pick(1.0f) + float(pick(1));
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "float pick(float value)" in crossgl
    assert "return false ? value : (-value);" in crossgl
    assert "int pick(int value)" in crossgl
    assert "return value + 1;" in crossgl
    assert not re.search(r"\bAlternate\b", crossgl)
    assert parse_crossgl(crossgl) is not None


def test_codegen_fails_closed_for_ambiguous_same_name_value_templates():
    code = """
    template <bool Alternate = false>
    float pick(float value) {
        return Alternate ? value : -value;
    }

    template <int Offset = 0>
    float pick(float value) {
        return value + Offset;
    }

    float run_pick() {
        return pick(1.0f);
    }
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code)

    diagnostic = exc_info.value
    assert diagnostic.argument_kind == "overload"
    assert diagnostic.function_name == "pick"
    assert diagnostic.reason == (
        "does not identify one unique value-template declaration"
    )
    assert "pick<Alternate>(float)" in diagnostic.argument_expression
    assert "pick<Offset>(float)" in diagnostic.argument_expression


def test_codegen_allocates_specialization_name_around_unrelated_collision():
    code = """
    float leaf_false(float value) {
        return 99.0f;
    }

    template <bool Inner>
    float leaf(float value) {
        return Inner ? value : -value;
    }

    template <bool Outer = false>
    float branch(float value) {
        return leaf<Outer>(value);
    }

    float run_collision(float value) {
        return leaf_false(value) + branch(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "float leaf_false(float value)" in crossgl
    assert "return 99.0f;" in crossgl
    assert "float leaf_false_2(float value)" in crossgl
    assert "return leaf_false_2(value);" in crossgl
    assert "return leaf_false(value) + branch(value);" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_mixed_type_value_template_keeps_inferred_type_parameter():
    code = """
    template <typename T, bool Alternate = false>
    T mixed(T value) {
        return Alternate ? value : -value;
    }

    float run_mixed(float value) {
        return mixed(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "generic<T> T mixed(T value)" in crossgl
    assert "return false ? value : (-value);" in crossgl
    assert not re.search(r"\bAlternate\b", crossgl)
    assert parse_crossgl(crossgl) is not None


def test_codegen_fails_closed_for_explicit_mixed_type_value_arguments():
    code = """
    template <typename T, bool Alternate = false>
    T mixed(T value) {
        return Alternate ? value : -value;
    }

    float run_mixed(float value) {
        return mixed<float, true>(value);
    }
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code, file_path="mixed-template.metal")

    diagnostic = exc_info.value
    assert diagnostic.argument_kind == "explicit_type"
    assert diagnostic.parameter_name == "T"
    assert diagnostic.argument_expression == "float"
    assert diagnostic.explicit_argument == "float"
    assert diagnostic.reason == (
        "cannot be preserved by the value-template specialization identity"
    )
    assert diagnostic.source_location["file"] == "mixed-template.metal"
    assert diagnostic.source_location["line"] == 2
    assert str(diagnostic) == (
        "Cannot materialize Metal function template 'mixed' for call "
        "'mixed<float,true>': explicit type argument for 'T' (float) cannot be "
        "preserved by the value-template specialization identity"
    )
