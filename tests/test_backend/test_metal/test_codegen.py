import re
import shutil
import subprocess
from typing import List

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.Metal.MetalAst import CastNode as MetalCastNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalAddressProvenanceError,
    MetalAliasTemplateResolutionError,
    MetalAtomicFenceLoweringError,
    MetalAutoTypeInferenceError,
    MetalBuiltinOverloadResolutionError,
    MetalBuiltinResultTypeResolutionError,
    MetalCallableAliasLoweringError,
    MetalCallableLoweringError,
    MetalConstructorContractError,
    MetalCooperativeMatrixFragmentLoweringError,
    MetalFunctionLocalTypeResolutionError,
    MetalIndexedComponentTypeResolutionError,
    MetalOutOfLineCallOperatorLoweringError,
    MetalSizeofResolutionError,
    MetalSourceOverloadResolutionError,
    MetalStageEntryArrayResourceError,
    MetalStandardLibraryWrapperLoweringError,
    MetalStaticConstantResolutionError,
    MetalStructAliasResolutionError,
    MetalStructMethodCallResolutionError,
    MetalTemplateArgumentResolutionError,
    MetalToCrossGLConverter,
    MetalWideVectorLoweringError,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)
from crosstl.project import reflect_target_host_interface, translate_project
from crosstl.translator.ast import ArrayAccessNode as CrossGLArrayAccessNode
from crosstl.translator.ast import AssignmentNode as CrossGLAssignmentNode
from crosstl.translator.ast import BinaryOpNode as CrossGLBinaryOpNode
from crosstl.translator.ast import CooperativeMatrixType
from crosstl.translator.ast import FunctionCallNode as CrossGLFunctionCallNode
from crosstl.translator.ast import IdentifierNode as CrossGLIdentifierNode
from crosstl.translator.ast import LiteralNode as CrossGLLiteralNode
from crosstl.translator.ast import MemberAccessNode as CrossGLMemberAccessNode
from crosstl.translator.ast import ResourceMemoryQualifierNode
from crosstl.translator.ast import TernaryOpNode as CrossGLTernaryOpNode
from crosstl.translator.ast import UnaryOpNode as CrossGLUnaryOpNode
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


def assert_opengl_compute_validates_if_available(glsl, tmp_path, stem):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    source_path = tmp_path / f"{stem}.comp"
    binary_path = tmp_path / f"{stem}.spv"
    source_path.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        [
            glslang,
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(source_path),
            "-o",
            str(binary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    spirv_val = shutil.which("spirv-val")
    if spirv_val is not None:
        result = subprocess.run(
            [spirv_val, "--target-env", "opengl4.5", str(binary_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def resolve_directx_numthreads_entry(source_path):
    interface = reflect_target_host_interface(source_path, target="directx")
    assert interface is not None
    entries = []
    for entry in interface.get("entryPoints", []):
        execution_config = entry.get("executionConfig")
        if (
            entry.get("stage") == "compute"
            and isinstance(execution_config, dict)
            and "numthreads" in execution_config
        ):
            entries.append(entry)
    assert len(entries) == 1, interface
    entry_name = entries[0].get("name")
    assert isinstance(entry_name, str) and entry_name, interface
    return entry_name


def find_crossgl_function(shader, name):
    functions = list(shader.functions)
    functions.extend(stage.entry_point for stage in shader.stages.values())
    return next(function for function in functions if function.name == name)


def crossgl_local_initializers(function):
    return {
        statement.name: statement.initial_value
        for statement in function.body.statements
        if getattr(statement, "initial_value", None) is not None
    }


def crossgl_expression_tree(expression):
    if isinstance(expression, CrossGLIdentifierNode):
        return expression.name
    if isinstance(expression, CrossGLLiteralNode):
        return expression.value
    if isinstance(expression, CrossGLAssignmentNode):
        return (
            "assignment",
            expression.operator,
            crossgl_expression_tree(expression.left),
            crossgl_expression_tree(expression.right),
        )
    if isinstance(expression, CrossGLBinaryOpNode):
        return (
            "binary",
            expression.op,
            crossgl_expression_tree(expression.left),
            crossgl_expression_tree(expression.right),
        )
    if isinstance(expression, CrossGLTernaryOpNode):
        return (
            "conditional",
            crossgl_expression_tree(expression.condition),
            crossgl_expression_tree(expression.true_expr),
            crossgl_expression_tree(expression.false_expr),
        )
    if isinstance(expression, CrossGLFunctionCallNode):
        return (
            "call",
            crossgl_expression_tree(expression.function),
            tuple(crossgl_expression_tree(arg) for arg in expression.arguments),
        )
    if isinstance(expression, CrossGLMemberAccessNode):
        return (
            "member",
            crossgl_expression_tree(expression.object),
            expression.member,
        )
    if isinstance(expression, CrossGLArrayAccessNode):
        return (
            "subscript",
            crossgl_expression_tree(expression.array),
            crossgl_expression_tree(expression.index),
        )
    if isinstance(expression, CrossGLUnaryOpNode):
        return (
            "postfix" if expression.is_postfix else "prefix",
            expression.op,
            crossgl_expression_tree(expression.operand),
        )
    raise AssertionError(f"Unexpected CrossGL expression: {expression!r}")


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
        "type load(inout thread RT dst, in threadgroup ST src, int threadIdx)"
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

    assert "void spvArrayCopy(inout thread T[N] dst, in thread T[N] src)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_reference_parameters_preserve_readonly_direction(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload { float value; };

    float read_scalar(const thread float& value) { return value; }
    float2 read_vector(const thread float2& value) { return value; }
    float read_payload(const thread Payload& value) { return value.value; }
    float read_temporary() { return read_scalar(1.0); }

    void write_scalar(thread float& value) { value = 1.0; }
    void write_vector(thread float2& value) { value.x = 1.0; }
    void write_payload(thread Payload& value) { value.value = 1.0; }

    kernel void apply(
        device float* output [[buffer(0)]],
        uint tid [[thread_position_in_grid]]) {
      float scalar = float(tid);
      float2 vector = float2(scalar);
      Payload payload;
      payload.value = scalar;
      write_scalar(scalar);
      write_vector(vector);
      write_payload(payload);
      output[tid] = read_scalar(scalar) + read_vector(vector).x
          + read_payload(payload) + read_temporary();
    }
    """

    crossgl = convert(code)

    assert "float read_scalar(in thread float value)" in crossgl
    assert "vec2 read_vector(in thread vec2 value)" in crossgl
    assert "float read_payload(in thread Payload value)" in crossgl
    assert "void write_scalar(inout thread float value)" in crossgl
    assert "void write_vector(inout thread vec2 value)" in crossgl
    assert "void write_payload(inout thread Payload value)" in crossgl
    assert "return read_scalar(1.0);" in crossgl

    ast = parse_crossgl(crossgl)
    hlsl_source = TranslatorHLSLCodeGen().generate(ast)
    glsl_source = GLSLCodeGen().generate(ast)
    metal_source = MetalCodeGen().generate(ast)
    hlsl = normalize(hlsl_source)
    glsl = normalize(glsl_source)
    metal = normalize(metal_source)

    assert "float read_scalar(in float value)" in hlsl
    assert "float2 read_vector(in float2 value)" in hlsl
    assert "float read_payload(in Payload value)" in hlsl
    assert "float read_scalar(float value)" in glsl
    assert "vec2 read_vector(vec2 value)" in glsl
    assert "float read_payload(Payload value)" in glsl
    assert "float read_scalar(const thread float& value)" in metal
    assert "float2 read_vector(const thread float2& value)" in metal
    assert "float read_payload(const thread Payload& value)" in metal
    for generated in (hlsl, glsl):
        assert "void write_scalar(inout float value)" in generated
        assert "void write_vector(inout" in generated
        assert "void write_payload(inout Payload value)" in generated
        assert "return read_scalar(1.0);" in generated

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "readonly-reference.hlsl"
        hlsl_path.write_text(hlsl_source, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-WX", "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_opengl_compute_validates_if_available(
        glsl_source, tmp_path, "readonly-reference"
    )

    xcrun = shutil.which("xcrun")
    if xcrun is not None:
        metal_path = tmp_path / "readonly-reference.metal"
        air_path = tmp_path / "readonly-reference.air"
        metal_path.write_text(metal_source, encoding="utf-8")
        result = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(metal_path),
                "-o",
                str(air_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


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
        (crossgl, "State__"),
        (TranslatorHLSLCodeGen().generate(ast), "State__"),
        (GLSLCodeGen().generate(ast), "State_"),
    )

    assert "int State__total(in thread State self)" in normalize(crossgl)

    for generated, method_prefix in generated_targets:
        generated = normalize(generated)
        mutable_signature = re.search(
            rf"\bvoid {method_prefix}mutate\s*\(([^)]*)\)", generated
        )
        const_signature = re.search(
            rf"\bint {method_prefix}total\s*\(([^)]*)\)", generated
        )

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
        "ReadWriter_float2_float2 self" in normalize(crossgl)
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


def test_codegen_binds_pinned_mlx_inverse_complex_call_operator_bodies():
    # Reduced from MLX unary_ops.h at
    # 4367c73b60541ddd5a266ce4644fd93d20223b6e.
    source = """
    struct complex64_t { float real; float imag; };
    struct Log {};
    struct Sqrt {};

    struct ArcCos {
      complex64_t operator()(complex64_t declared_value) const;
    };
    struct ArcSin {
      complex64_t operator()(complex64_t declared_value);
    };
    struct ArcTan {
      complex64_t operator()(complex64_t declared_value);
    };

    complex64_t ArcCos::operator()(complex64_t x) const {
      auto i = complex64_t{0.0, 1.0};
      auto y = Log__operator_call__temporary(
          x + i * Sqrt__operator_call__temporary(1.0 - x * x));
      return {y.imag, -y.real};
    }
    complex64_t ArcSin::operator()(complex64_t x) {
      auto i = complex64_t{0.0, 1.0};
      auto y = Log__operator_call__temporary(
          i * x + Sqrt__operator_call__temporary(1.0 - x * x));
      return {y.imag, -y.real};
    }
    complex64_t ArcTan::operator()(complex64_t x) {
      auto i = complex64_t{0.0, 1.0};
      auto ix = i * x;
      return (1.0 / complex64_t{0.0, 2.0}) *
          Log__operator_call__temporary((1.0 + ix) / (1.0 - ix));
    }

    complex64_t Log__operator_call__temporary(complex64_t x) { return x; }
    complex64_t Sqrt__operator_call__temporary(complex64_t x) { return x; }

    complex64_t ArcCos__operator_call__complex64_t(
        thread const ArcCos& self, complex64_t scalar_value) {
      return metal::precise::acos(scalar_value);
    }
    complex64_t ArcSin__operator_call__complex64_t(
        thread ArcSin& self, complex64_t scalar_value) {
      return metal::precise::asin(scalar_value);
    }
    complex64_t ArcTan__operator_call__complex64_t(
        thread ArcTan& self, complex64_t scalar_value) {
      return metal::precise::atan(scalar_value);
    }

    kernel void inverse_complex(
        device complex64_t* output [[buffer(0)]], complex64_t value) {
      ArcCos acos_op;
      ArcSin asin_op;
      ArcTan atan_op;
      output[0] = ArcCos__operator_call__complex64_t(acos_op, value);
      output[1] = ArcSin__operator_call__complex64_t(asin_op, value);
      output[2] = ArcTan__operator_call__complex64_t(atan_op, value);
    }
    """

    crossgl = convert_without_preprocessing(source, "mlx-unary-inverse.metal")
    normalized = normalize(crossgl)

    acos_body = normalized.split("ArcCos__operator_call__complex64_t", 1)[1].split(
        "}", 1
    )[0]
    asin_body = normalized.split("ArcSin__operator_call__complex64_t", 1)[1].split(
        "}", 1
    )[0]
    atan_body = normalized.split("ArcTan__operator_call__complex64_t", 1)[1].split(
        "}", 1
    )[0]

    assert "Log__operator_call__temporary" in acos_body
    assert "Sqrt__operator_call__temporary(1.0 - scalar_value * scalar_value)" in (
        acos_body
    )
    assert "Log__operator_call__temporary" in asin_body
    assert "Sqrt__operator_call__temporary(1.0 - scalar_value * scalar_value)" in (
        asin_body
    )
    assert "Log__operator_call__temporary" in atan_body
    assert "acos(scalar_value)" not in crossgl
    assert "asin(scalar_value)" not in crossgl
    assert "atan(scalar_value)" not in crossgl
    assert "ArcCos_u3a_u3aoperator_u28_u29" not in crossgl
    assert "ArcSin_u3a_u3aoperator_u28_u29" not in crossgl
    assert "ArcTan_u3a_u3aoperator_u28_u29" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_keeps_out_of_line_call_operator_without_materialized_helper():
    source = """
    struct Increment {
      int operator()(int declared_value) const;
    };

    int Increment::operator()(int value) const {
      return value + 1;
    }
    """

    crossgl = convert_without_preprocessing(source, "increment.metal")

    assert "Increment_u3a_u3aoperator_u28_u29" in crossgl
    assert "return value + 1;" in crossgl
    assert "Increment__operator_call" not in crossgl


def test_codegen_rejects_ambiguous_out_of_line_call_operator_helpers():
    source = """struct Increment {
    int operator()(int value) const;
};

int Increment::operator()(int value) const {
    return value + 1;
}

int Increment__operator_call__int(
    thread const Increment& self, int value) {
    return value;
}
int Increment__operator_call__int(
    thread const Increment& self, int value) {
    return value + 2;
}
"""
    tokens = MetalLexer(source, preprocess=False).tokenize()
    ast = MetalParser(tokens, file_path="ambiguous-helper.metal").parse()

    with pytest.raises(MetalOutOfLineCallOperatorLoweringError) as exc_info:
        generate_code(ast)

    error = exc_info.value
    assert error.project_diagnostic_code == (
        "project.translate.metal-out-of-line-call-operator-unresolved"
    )
    assert error.owner == "Increment"
    assert error.reason == "multiple lowered helpers match the declaration contract"
    assert error.source_location["file"] == "ambiguous-helper.metal"
    assert error.source_location["line"] == 5
    assert error.declaration_location["line"] == 2
    assert [location["line"] for location in error.candidate_locations] == [9, 13]


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


def test_codegen_using_union_alias_from_mlx_cexpf_header_retains_layout_contract():
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
        "// Metal union ieee_float_shape_type retains overlapping storage through "
        "layout metadata"
    ) in crossgl
    assert "@union_layout(4, 4, little_endian, metal)" in crossgl
    assert "struct ieee_float_shape_type {" in crossgl
    assert "@union_member_layout(0, 4, 4) float value;" in crossgl
    assert "@union_member_layout(0, 4, 4) uint word;" in crossgl
    assert "using ieee_float_shape_type = union" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_mlx_random_union_retains_member_array_layout_contract():
    crossgl = convert("""
        union rbits {
          uint2 val;
          uchar4 bytes[2];
        };
        """)

    assert "@union_layout(8, 8, little_endian, metal)" in crossgl
    assert "@union_member_layout(0, 8, 8) uvec2 val;" in crossgl
    assert "@union_member_layout(0, 8, 4) u8vec4[2] bytes;" in crossgl
    parsed = parse_crossgl(crossgl)
    union = parsed.structs[0]
    assert [attribute.name for attribute in union.attributes] == ["union_layout"]
    assert [
        attribute.name for member in union.members for attribute in member.attributes
    ] == ["union_member_layout", "union_member_layout"]


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


def test_codegen_preserves_conditional_and_assignment_binary_operand_trees():
    code = """
    int grouped(bool choose, int a, int b, int c) {
        int left_conditional = (choose ? a : b) + c;
        int right_conditional = a + (choose ? b : c);
        int left_assignment = (a = b) + c;
        int right_assignment = a + (b = c);
        int ordinary = a + b * c;
        return ordinary;
    }
    """

    parsed = parse_crossgl(convert(code))
    initializers = crossgl_local_initializers(find_crossgl_function(parsed, "grouped"))
    trees = {
        name: crossgl_expression_tree(expression)
        for name, expression in initializers.items()
    }

    assert trees == {
        "left_conditional": (
            "binary",
            "+",
            ("conditional", "choose", "a", "b"),
            "c",
        ),
        "right_conditional": (
            "binary",
            "+",
            "a",
            ("conditional", "choose", "b", "c"),
        ),
        "left_assignment": (
            "binary",
            "+",
            ("assignment", "=", "a", "b"),
            "c",
        ),
        "right_assignment": (
            "binary",
            "+",
            "a",
            ("assignment", "=", "b", "c"),
        ),
        "ordinary": (
            "binary",
            "+",
            "a",
            ("binary", "*", "b", "c"),
        ),
    }


def test_codegen_preserves_nested_conditional_parser_associativity():
    code = """
    int nested(bool outer, bool inner, bool fallback, int a, int b, int c) {
        int nested_condition = (outer ? inner : fallback) ? a : b;
        int nested_true = outer ? (inner ? a : b) : c;
        int nested_false = outer ? a : (inner ? b : c);
        int true_assignment = outer ? (a = b) : c;
        int false_assignment = outer ? a : (b = c);
        int assignment_condition = (outer = inner) ? a : b;
        return nested_condition;
    }
    """

    parsed = parse_crossgl(convert(code))
    initializers = crossgl_local_initializers(find_crossgl_function(parsed, "nested"))
    trees = {
        name: crossgl_expression_tree(expression)
        for name, expression in initializers.items()
    }

    assert trees == {
        "nested_condition": (
            "conditional",
            ("conditional", "outer", "inner", "fallback"),
            "a",
            "b",
        ),
        "nested_true": (
            "conditional",
            "outer",
            ("conditional", "inner", "a", "b"),
            "c",
        ),
        "nested_false": (
            "conditional",
            "outer",
            "a",
            ("conditional", "inner", "b", "c"),
        ),
        "true_assignment": (
            "conditional",
            "outer",
            ("assignment", "=", "a", "b"),
            "c",
        ),
        "false_assignment": (
            "conditional",
            "outer",
            "a",
            ("assignment", "=", "b", "c"),
        ),
        "assignment_condition": (
            "conditional",
            ("assignment", "=", "outer", "inner"),
            "a",
            "b",
        ),
    }


def test_codegen_preserves_conditional_structured_buffer_pointer_offset_tree():
    code = """
    kernel void shifted_mask(
        constant int* mask_strides [[buffer(0)]],
        device int* output [[buffer(1)]],
        bool has_output_mask,
        uint gid [[thread_position_in_grid]]) {
        const constant int* lhs_mask_strides =
            mask_strides + (has_output_mask ? 2 : 0);
        output[gid] = lhs_mask_strides[0];
    }
    """

    parsed = parse_crossgl(convert(code))
    initializers = crossgl_local_initializers(
        find_crossgl_function(parsed, "shifted_mask")
    )

    assert crossgl_expression_tree(initializers["lhs_mask_strides"]) == (
        "binary",
        "+",
        "mask_strides",
        ("conditional", "has_output_mask", 2, 0),
    )


def test_codegen_preserves_conditional_postfix_base_trees():
    code = """
    struct Pair {
        int value;
        int values[2];
    };

    int low_precedence_bases(bool choose, Pair left, Pair right) {
        int member = (choose ? left : right).value;
        int element = (choose ? left.values : right.values)[1];
        return member + element;
    }
    """

    parsed = parse_crossgl(convert(code))
    initializers = crossgl_local_initializers(
        find_crossgl_function(parsed, "low_precedence_bases")
    )

    assert crossgl_expression_tree(initializers["member"]) == (
        "member",
        ("conditional", "choose", "left", "right"),
        "value",
    )
    assert crossgl_expression_tree(initializers["element"]) == (
        "subscript",
        (
            "conditional",
            "choose",
            ("member", "left", "values"),
            ("member", "right", "values"),
        ),
        1,
    )


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
        "void copy_one(StructuredBuffer<float> src, int64_t src_offset, "
        "RWStructuredBuffer<float> dst, int64_t dst_offset, uint index)" in hlsl
    )
    assert "StructuredBuffer<float> src : register(t0);" in hlsl
    assert "dst[uint((dst_offset + index))] = src[uint((src_offset + index))];" in hlsl
    assert "copy_one(src, int64_t(0), dst, int64_t(0), index);" in hlsl

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
    assert "quantized_out_[tid]" in glsl
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


def test_codegen_elides_pure_standalone_void_casts():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void discard_values(
        const device uint* values [[buffer(0)]],
        uint3 position [[thread_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]]) {
        (void)lid;
        (void)values[lid];
        (void)position.x;
        (void)((uint)lid);
    }
    """
    metal_ast = parse_code(tokenize_code(code))
    kernel = next(
        function
        for function in metal_ast.functions
        if function.name == "discard_values"
    )

    assert len(kernel.body) == 4
    assert all(
        isinstance(statement, MetalCastNode) and statement.target_type == "void"
        for statement in kernel.body
    )

    crossgl = convert(code)

    assert "(void)" not in crossgl
    crossgl_ast = parse_crossgl(crossgl)
    assert find_crossgl_function(crossgl_ast, "discard_values").body.statements == []

    hlsl = TranslatorHLSLCodeGen().generate(crossgl_ast)
    assert not re.search(r"(?m)^\s*(?:lid|position\.x);$", hlsl)


def test_codegen_preserves_effectful_standalone_void_casts_once():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    uint observe(uint value) {
        return value + 1u;
    }

    kernel void discard_values(
        const device uint* values [[buffer(0)]],
        uint lid [[thread_index_in_threadgroup]]) {
        uint counter = lid;
        (void)observe(lid);
        (void)values[observe(lid)];
        (void)(counter = lid + 1u);
        (void)++counter;
        (void)counter--;
    }
    """
    crossgl = convert(code)

    assert "(void)" not in crossgl
    assert crossgl.count("observe(lid)") == 2
    assert crossgl.count("observe(lid);") == 1
    assert crossgl.count("counter = lid + 1u;") == 1
    assert crossgl.count("++counter") == 1
    assert crossgl.count("counter--;") == 1

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)

    assert hlsl.count("observe(lid)") == 2
    assert hlsl.count("observe(lid);") == 1
    assert len(re.findall(r"counter\s*=\s*\(?lid \+ 1u\)?;", hlsl)) == 1
    assert len(re.findall(r"(?:\+\+counter|counter\+\+)", hlsl)) == 1
    assert len(re.findall(r"(?:--counter|counter--)", hlsl)) == 1


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
    glsl_helper_name = "BlockKernel_float_4_load_float_glsl_src_src_float"
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


def test_codegen_hoists_function_local_typedef_struct_for_project_targets(tmp_path):
    crossgl = convert("""
        kernel void copy_words(
            const device uint* input [[buffer(0)]],
            device uint* output [[buffer(1)]]) {
            constexpr int word_count = 1 + 1;
            using Word = uint32_t;
            typedef struct {
                Word values[word_count];
            } WordBlock;
            thread WordBlock local =
                *((const device WordBlock*)input);
            output[0] = local.values[0];
        }
        """)

    canonical = "MetalLocal_copy_words_WordBlock"
    assert f"struct {canonical} {{" in crossgl
    assert "uint[2] values;" in crossgl
    assert f"thread {canonical} local" in crossgl
    assert f"({canonical}*)input" in crossgl
    assert "local.values[0]" in crossgl
    assert "WordBlock" not in crossgl.replace(canonical, "")
    assert "Word values" not in crossgl

    shader = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(shader)
    glsl = GLSLCodeGen().generate(shader)
    assert f"struct {canonical}" in hlsl
    assert f"struct {canonical}" in glsl
    assert "uint values[2];" in hlsl
    assert "uint values[2];" in glsl
    assert "local.values[0]" in hlsl
    assert "local.values[0]" in glsl
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()
    assert_opengl_compute_validates_if_available(glsl, tmp_path, "local-typedef-struct")


def test_codegen_hoists_named_local_struct_with_qualified_pointer_cast():
    crossgl = convert("""
        kernel void copy_words(
            const device uint* input [[buffer(0)]],
            device uint* output [[buffer(1)]]) {
            constexpr int word_count = 2;
            struct WordBlock {
                uint values[word_count];
            };
            thread WordBlock local =
                *reinterpret_cast<const device WordBlock*>(input);
            output[0] = local.values[0];
        }
        """)

    canonical = "MetalLocal_copy_words_WordBlock"
    assert f"struct {canonical} {{" in crossgl
    assert "uint[2] values;" in crossgl
    assert f"thread {canonical} local = (*({canonical}*)input);" in crossgl
    assert "const device WordBlock" not in crossgl
    strict_ast = CrossGLParser(
        CrossGLLexer(crossgl).get_tokens(), strict_function_bodies=True
    ).parse()
    assert strict_ast is not None


def test_codegen_resolves_alias_chain_to_named_local_struct():
    crossgl = convert("""
        uint read_word() {
            struct WordBlock {
                uint layout;
            };
            using BlockAlias = WordBlock;
            typedef struct WordBlock ElaboratedAlias;
            thread BlockAlias direct;
            thread ElaboratedAlias elaborated;
            return direct.layout + elaborated.layout;
        }
        """)

    canonical = "MetalLocal_read_word_WordBlock"
    assert f"struct {canonical}" in crossgl
    assert "uint layout_;" in crossgl
    assert f"thread {canonical} direct" in crossgl
    assert f"thread {canonical} elaborated" in crossgl
    assert "return direct.layout_ + elaborated.layout_;" in crossgl
    assert "BlockAlias" not in crossgl
    assert "ElaboratedAlias" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_colliding_local_aggregate_member_names():
    crossgl = convert("""
        uint read_word() {
            struct Names {
                uint layout;
                uint layout_;
            };
            thread Names names;
            names.layout = 3;
            names.layout_ = 7;
            return names.layout + names.layout_;
        }
        """)

    assert "uint layout_;" in crossgl
    assert "uint layout__2;" in crossgl
    assert "names.layout_ = 3;" in crossgl
    assert "names.layout__2 = 7;" in crossgl
    assert "return names.layout_ + names.layout__2;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_colliding_module_aggregate_member_names():
    crossgl = convert("""
        struct Names {
            uint layout;
            uint layout_;
        };

        uint read_word(thread Names& names) {
            names.layout = 3;
            names.layout_ = 7;
            return names.layout + names.layout_;
        }
        """)

    assert "uint layout_;" in crossgl
    assert "uint layout__2;" in crossgl
    assert "names.layout_ = 3;" in crossgl
    assert "names.layout__2 = 7;" in crossgl
    assert "return names.layout_ + names.layout__2;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_colliding_members_through_nested_access():
    crossgl = convert("""
        struct Names {
            uint layout;
            uint layout_;
        };

        struct Wrapper {
            Names names;
        };

        uint read_word(thread Wrapper& wrapper) {
            wrapper.names.layout = 3;
            wrapper.names.layout_ = 7;
            return wrapper.names.layout + wrapper.names.layout_;
        }
        """)

    assert "wrapper.names.layout_ = 3;" in crossgl
    assert "wrapper.names.layout__2 = 7;" in crossgl
    assert "return wrapper.names.layout_ + wrapper.names.layout__2;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_colliding_members_in_constructor_factory():
    crossgl = convert("""
        struct Names {
            uint layout;
            uint layout_;

            Names(uint first, uint second)
                : layout(first), layout_(second) {}
        };

        uint read_word() {
            thread Names names(3, 7);
            return names.layout + names.layout_;
        }
        """)

    assert "crosstl_ctor_value.layout_ = uint(first);" in crossgl
    assert "crosstl_ctor_value.layout__2 = uint(second);" in crossgl
    assert "return names.layout_ + names.layout__2;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_colliding_members_in_designated_initializer():
    crossgl = convert("""
        struct Names {
            uint layout;
            uint layout_;
        };

        uint read_word() {
            thread Names names = {.layout = 3, .layout_ = 7};
            return names.layout + names.layout_;
        }
        """)

    assert "Names{layout_: 3, layout__2: 7}" in crossgl
    assert "return names.layout_ + names.layout__2;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_materializes_local_typedef_struct_per_value_specialization():
    crossgl = convert("""
        template <int Width>
        [[kernel]] void make_block(device uint* output [[buffer(0)]]) {
            constexpr int word_count = Width + 1;
            typedef struct {
                uint words[word_count];
            } Block;
            thread Block local;
            output[0] = local.words[0];
        }

        template [[host_name("make_block_2")]] [[kernel]]
        decltype(make_block<2>) make_block<2>;
        template [[host_name("make_block_4")]] [[kernel]]
        decltype(make_block<4>) make_block<4>;
        """)

    assert "struct MetalLocal_make_block_2_Block" in crossgl
    assert "struct MetalLocal_make_block_4_Block" in crossgl
    assert "uint[3] words;" in crossgl
    assert "uint[5] words;" in crossgl
    assert "thread MetalLocal_make_block_2_Block local" in crossgl
    assert "thread MetalLocal_make_block_4_Block local" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_folds_transitive_constexpr_local_struct_extent():
    crossgl = convert("""
        template <int WordSize = 8, int Bits = 4>
        constexpr short get_pack_factor() {
            return WordSize / Bits;
        }

        template <int WordSize = 8>
        constexpr short get_bytes_per_pack() {
            return WordSize / 8;
        }

        [[kernel]] void make_block_float_16_4() {
            constexpr int pack_factor = get_pack_factor<32, 4>();
            constexpr int bytes_per_pack = get_bytes_per_pack();
            constexpr int width = 16 / pack_factor;
            using Word = uint32_t;
            typedef struct {
                Word values[width * bytes_per_pack];
            } Block;
            thread Block local;
        }
        """)

    assert "struct MetalLocal_make_block_float_16_4_Block" in crossgl
    assert "uint[2] values;" in crossgl
    assert "thread MetalLocal_make_block_float_16_4_Block local" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_prefers_constexpr_non_template_zero_argument_overload_for_extent():
    crossgl = convert("""
        template <int Width = 2>
        constexpr int block_width() {
            return Width;
        }

        constexpr int block_width() {
            return 3;
        }

        uint read_word() {
            constexpr int width = block_width();
            typedef struct {
                uint values[width];
            } Block;
            thread Block local;
            return local.values[0];
        }
        """)

    assert "struct MetalLocal_read_word_Block" in crossgl
    assert "uint[3] values;" in crossgl
    assert "const int width = 3;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_isolates_same_named_local_structs_between_sibling_functions():
    crossgl = convert("""
        uint first() {
            typedef struct { uint values[2]; } Block;
            thread Block local;
            return local.values[0];
        }

        float second() {
            typedef struct { float values[3]; } Block;
            thread Block local;
            return local.values[0];
        }
        """)

    assert "struct MetalLocal_first_Block" in crossgl
    assert "uint[2] values;" in crossgl
    assert "struct MetalLocal_second_Block" in crossgl
    assert "float[3] values;" in crossgl
    assert "thread MetalLocal_first_Block local" in crossgl
    assert "thread MetalLocal_second_Block local" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_binds_named_local_struct_tag_and_typedef_alias():
    crossgl = convert("""
        uint read_word() {
            typedef struct WordTag {
                uint value;
            } WordAlias;
            thread WordTag tagged;
            thread WordAlias aliased;
            return tagged.value + aliased.value;
        }
        """)

    canonical = "MetalLocal_read_word_WordAlias"
    assert f"struct {canonical}" in crossgl
    assert f"thread {canonical} tagged" in crossgl
    assert f"thread {canonical} aliased" in crossgl
    assert "WordTag" not in crossgl
    assert "WordAlias" not in crossgl.replace(canonical, "")
    assert parse_crossgl(crossgl) is not None


def test_codegen_restores_shadowed_local_struct_after_nested_block():
    crossgl = convert("""
        uint select_word() {
            typedef struct { uint values[2]; } Block;
            thread Block outer;
            {
                typedef struct { float values[3]; } Block;
                thread Block inner;
                inner.values[0] = 1.0f;
            }
            thread Block restored;
            return outer.values[0] + restored.values[1];
        }
        """)

    outer = "MetalLocal_select_word_Block"
    inner = "MetalLocal_select_word_Block_2"
    assert f"struct {outer}" in crossgl
    assert f"struct {inner}" in crossgl
    assert f"thread {outer} outer" in crossgl
    assert f"thread {inner} inner" in crossgl
    assert f"thread {outer} restored" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_rejects_unresolved_function_local_struct_dependency():
    with pytest.raises(MetalFunctionLocalTypeResolutionError) as exc_info:
        convert("""
            template <typename T>
            void prepare() {
                using Element = T;
                typedef struct {
                    Element value;
                } LocalValue;
                thread LocalValue local;
            }
            """)

    diagnostic = exc_info.value
    assert diagnostic.function_name == "prepare"
    assert diagnostic.type_name == "LocalValue"
    assert diagnostic.unresolved_dependencies == ("Element",)
    assert diagnostic.source_location is not None


def test_codegen_rejects_unresolved_function_local_struct_extent():
    with pytest.raises(MetalFunctionLocalTypeResolutionError) as exc_info:
        convert("""
            template <int Width>
            void prepare() {
                typedef struct {
                    uint values[Width + 1];
                } LocalValue;
                thread LocalValue local;
            }
            """)

    diagnostic = exc_info.value
    assert diagnostic.function_name == "prepare"
    assert diagnostic.type_name == "LocalValue"
    assert diagnostic.unresolved_dependencies == ("Width",)
    assert diagnostic.source_location is not None


@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        (
            "typedef union { uint bits; float value; } LocalValue;",
            "union storage semantics",
        ),
        (
            "class LocalValue { uint value; };",
            "class storage semantics",
        ),
        (
            "struct LocalValue : BaseValue { uint value; };",
            "base classes",
        ),
        (
            "struct LocalValue { using Word = uint; Word value; };",
            "aggregate-scoped type aliases",
        ),
        (
            "struct LocalValue { LocalValue() {} uint value; };",
            "constructors or call operators",
        ),
        (
            "struct LocalValue { uint operator()() { return 0; } uint value; };",
            "constructors or call operators",
        ),
        (
            "typedef struct { uint value = 7; } LocalValue;",
            "default initializer",
        ),
        (
            "typedef struct alignas(16) { uint value; } LocalValue;",
            "alignment or attributes",
        ),
        (
            "typedef struct { volatile uint value; } LocalValue;",
            "unsupported qualifiers",
        ),
        (
            "typedef struct { alignas(16) uint value; } LocalValue;",
            "member 'value' has alignment or attributes",
        ),
        (
            "typedef struct { uint value; } LocalValue[2];",
            "typedef declarator changes",
        ),
        (
            "typedef struct { uint value; } *LocalValue;",
            "typedef declarator changes",
        ),
        (
            "struct LocalValue { uint value; } local;",
            "trailing object declaration",
        ),
        (
            "struct LocalValue {};",
            "empty aggregates",
        ),
    ],
)
def test_codegen_rejects_local_aggregate_semantics_lost_by_hoisting(
    declaration, reason
):
    with pytest.raises(MetalFunctionLocalTypeResolutionError) as exc_info:
        convert(f"""
            uint read_word() {{
                {declaration}
                return 0;
            }}
            """)

    diagnostic = exc_info.value
    assert diagnostic.function_name == "read_word"
    assert diagnostic.type_name == "LocalValue"
    assert reason in diagnostic.reason
    assert diagnostic.source_location is not None


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


def test_codegen_elides_specialized_constexpr_assertions_from_targets(tmp_path):
    source = """
    template <int Word, int Bits>
    constexpr int get_pack_factor() {
        constexpr int factor = Word / Bits;
        return factor;
    }

    template <int GroupSize, int Bits>
    [[kernel]] void checked_pack(
        device int* out [[buffer(0)]]) {
        constexpr int reads =
            GroupSize / (get_pack_factor<8, Bits>() * 4);
        static_assert(
            reads * get_pack_factor<8, Bits>() <= GroupSize,
            "Pack reads must fit the group.");
        out[0] = reads;
    }

    template [[host_name("checked_pack_16_4")]] [[kernel]]
    decltype(checked_pack<16, 4>) checked_pack<16, 4>;
    """
    source_path = tmp_path / "checked-pack.metal"
    source_path.write_text(source, encoding="utf-8")

    outputs = {
        target: crosstl.translate(str(source_path), backend=target, format_output=False)
        for target in ("directx", "opengl")
    }

    for generated in outputs.values():
        assert "static_assert" not in generated
        assert "reads" in generated
    HLSLParser(HLSLLexer(outputs["directx"]).tokenize()).parse()
    assert_opengl_compute_validates_if_available(
        outputs["opengl"],
        tmp_path,
        "checked-pack-constexpr-assertion",
    )


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
    assert "in constant uint count" in crossgl
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


def test_codegen_direct_reference_accessor_preserves_storage_lvalue(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Tile {
        static constexpr short width = 2;
        float values[4];
        constexpr thread float& at(short row, short col) {
            return values[row * width + col];
        }
    };

    kernel void write_kernel(device float* out [[buffer(0)]]) {
        Tile tile;
        tile.at(1, 1) = 73.25f;
        out[0] = tile.values[3];
    }
    """

    crossgl = convert(code)
    assert "Tile__at" not in crossgl
    assert "tile.values[1 * 2 + 1] = 73.25f;" in crossgl

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    for generated in (hlsl, glsl):
        assert "tile.values[((1 * 2) + 1)] = 73.25;" in generated
        assert "tile.values[3]" in generated
        assert "Tile__at" not in generated

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl_path = tmp_path / "direct-reference-accessor.glsl"
        glsl_path.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [
                glslang,
                "-S",
                "comp",
                "-G",
                str(glsl_path),
                "-o",
                str(tmp_path / "direct-reference-accessor.spv"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "direct-reference-accessor.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "direct-reference-accessor.dxil"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_codegen_const_implicit_reference_accessor_reaches_target_backends(tmp_path):
    # Reduced from MLX steel/gemm/mma.h at the pinned integration revision.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Fragment {
        float lanes[2];
    };

    struct FragmentOps {
        static void store(
            const thread Fragment& fragment,
            device float* out) {
            out[0] = fragment.lanes[0];
        }
    };

    struct MMATile {
        static constexpr short width = 2;
        Fragment val_frags[4];

        constexpr thread Fragment& frag_at(short i, short j) {
            return val_frags[i * width + j];
        }

        constexpr const thread Fragment& frag_at(short i, short j) const {
            return val_frags[i * width + j];
        }

        void store(device float* out, short i, short j) const {
            FragmentOps::store(frag_at(i, j), out);
        }
    };

    kernel void store_fragment(device float* out [[buffer(0)]]) {
        MMATile tile;
        tile.val_frags[3].lanes[0] = 73.25f;
        tile.store(out, 1, 1);
    }
    """

    crossgl = convert(code)
    assert "frag_at" not in crossgl
    assert re.search(
        r"FragmentOps__store\(self\.val_frags\[[^]]*i[^]]*j[^]]*\], out_?\);",
        crossgl,
    )

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    for generated in (hlsl, glsl):
        assert "frag_at" not in generated
        assert re.search(
            r"FragmentOps\w*\(self\.val_frags\[[^]]*i[^]]*j[^]]*\]",
            generated,
        )

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl_path = tmp_path / "const-implicit-reference-accessor.glsl"
        glsl_path.write_text(glsl, encoding="utf-8")
        result = subprocess.run(
            [
                glslang,
                "-S",
                "comp",
                "-G",
                str(glsl_path),
                "-o",
                str(tmp_path / "const-implicit-reference-accessor.spv"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "const-implicit-reference-accessor.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_2",
                "-enable-16bit-types",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "const-implicit-reference-accessor.dxil"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_codegen_nested_const_reference_alias_reaches_native_targets(tmp_path):
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Tile {
        static constexpr int width = 2;
        float2 val_frags[4];

        constexpr thread float2& frag_at(int i, int j) {
            return val_frags[i * width + j];
        }

        constexpr const thread float2& frag_at(int i, int j) const {
            return val_frags[i * width + j];
        }
    };

    struct StoreLoop {
        Tile Ctile;

        void store(thread float2& stored, int i, int j) const {
            thread const auto& accum = Ctile.frag_at(i, j);
            for (int k = 0; k < 2; ++k) {
                stored[k] = accum[k];
            }
        }
    };

    kernel void store_fragment(
        device const StoreLoop* loops [[buffer(0)]],
        device float2* out [[buffer(1)]]) {
        StoreLoop loop = loops[0];
        float2 stored;
        loop.store(stored, 1, 1);
        out[0] = stored;
    }
    """

    crossgl = convert(code)
    assert "frag_at" not in crossgl
    assert "accum" not in crossgl
    assert (
        "CrossGLMetalVectorIndex_vec2_set(stored, k, "
        "CrossGLMetalVectorIndex_vec2_get("
        "self.Ctile.val_frags[i * 2 + j], k))" in crossgl
    )

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    for generated, store_call in (
        (hlsl, "StoreLoop__store"),
        (glsl, "StoreLoop_store"),
    ):
        assert "frag_at" not in generated
        assert "accum" not in generated
        assert "val_frags[4]" in generated
        assert "CrossGLMetalVectorIndex_vec2_set(stored," in generated
        assert "CrossGLMetalVectorIndex_vec2_get(self.Ctile.val_frags[" in generated
        assert f"{store_call}(loop, stored" in generated

    dxc = shutil.which("dxc")
    glslang = shutil.which("glslangValidator")
    spirv_val = shutil.which("spirv-val")
    hlsl_path = tmp_path / "nested-const-reference-alias.hlsl"
    hlsl_path.write_text(hlsl, encoding="utf-8")
    if dxc is not None:
        subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_2",
                "-enable-16bit-types",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "nested-const-reference-alias.dxil"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    elif glslang is not None:
        hlsl_spirv_path = tmp_path / "nested-const-reference-alias-hlsl.spv"
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
                str(hlsl_spirv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if spirv_val is not None:
            subprocess.run(
                [spirv_val, str(hlsl_spirv_path)],
                check=True,
                capture_output=True,
                text=True,
            )

    if glslang is not None:
        glsl_path = tmp_path / "nested-const-reference-alias.comp"
        glsl_spirv_path = tmp_path / "nested-const-reference-alias-glsl.spv"
        glsl_path.write_text(glsl, encoding="utf-8")
        subprocess.run(
            [
                glslang,
                "--target-env",
                "opengl",
                "-S",
                "comp",
                str(glsl_path),
                "-o",
                str(glsl_spirv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if spirv_val is not None:
            subprocess.run(
                [spirv_val, "--target-env", "opengl4.5", str(glsl_spirv_path)],
                check=True,
                capture_output=True,
                text=True,
            )


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


def mlx_materialized_auto_local_overload_source(kidx_initializer="2 * index.x"):
    return """
    int64_t elem_to_loc_int64_t(
        int64_t elem,
        constant const int* shape,
        constant const int64_t* strides,
        int ndim) {
      return elem + int64_t(shape[0]) + strides[0] + ndim;
    }

    int64_t elem_to_loc_int64_t(
        uint3 elem,
        constant const int* shape,
        constant const int64_t* strides,
        int ndim) {
      return int64_t(elem.x) + int64_t(shape[0]) + strides[0] + ndim;
    }

    kernel void use_indices(
        device int64_t* out_values [[buffer(0)]],
        constant const int* shape [[buffer(1)]],
        constant const int64_t* strides [[buffer(2)]],
        constant const int& ndim [[buffer(3)]],
        uint2 index [[thread_position_in_grid]]) {
      auto cast_index = uint(index.x);
      auto next_index = cast_index + 1;
      auto vector_index = index + uint2(1u);
      auto kidx = KIDX_INITIALIZER;
      auto first = elem_to_loc_int64_t(kidx, shape, strides, ndim);
      auto second = elem_to_loc_int64_t(kidx + 1, shape, strides, ndim);
      auto vector_value = elem_to_loc_int64_t(
          uint3(vector_index.x, 0u, 0u), shape, strides, ndim);
      out_values[0] = first + int64_t(next_index);
      out_values[1] = second;
      out_values[2] = vector_value;
    }
    """.replace("KIDX_INITIALIZER", kidx_initializer)


def test_codegen_infers_mlx_auto_locals_before_resource_overload_binding(tmp_path):
    crossgl = convert(mlx_materialized_auto_local_overload_source())
    normalized = normalize(crossgl)

    assert "uint cast_index = uint(index.x);" in normalized
    assert "uint next_index = cast_index + 1;" in normalized
    assert "uvec2 vector_index = index + uvec2(1u);" in normalized
    assert "uint kidx = 2 * index.x;" in normalized
    assert (
        "int64 first = elem_to_loc_int64_t(kidx, shape, strides, ndim);" in normalized
    )
    assert (
        "int64 second = elem_to_loc_int64_t(kidx + 1, shape, strides, ndim);"
        in normalized
    )
    assert "int64 elem_to_loc_int64_t(int64 elem" in normalized
    assert "int64 elem_to_loc_int64_t__metal_overload_2(uvec3 elem" in normalized
    assert (
        "elem_to_loc_int64_t__metal_overload_2("
        "uvec3(vector_index.x, 0u, 0u), shape, strides, ndim)" in normalized
    )

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    scalar_calls = re.findall(r"int64_t (?:first|second) = ([A-Za-z_]\w*)\(", glsl)
    assert len(scalar_calls) == 2
    assert len(set(scalar_calls)) == 1
    assert "metal_overload" not in scalar_calls[0]
    assert re.search(r"int64_t vector_value = [A-Za-z_]\w*metal_overload_2\w*\(", glsl)
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-auto-local-overload"
    )


def test_codegen_reports_unresolved_auto_local_resource_overload():
    source = mlx_materialized_auto_local_overload_source("external_index_value(index)")

    with pytest.raises(MetalSourceOverloadResolutionError) as exc_info:
        convert(source)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-source-overload-unresolved"
    )
    assert diagnostic.missing_capabilities == ("metal.source-overload-resolution",)
    assert diagnostic.function_name == "elem_to_loc_int64_t"
    assert diagnostic.argument_types == ("<unknown>", "int*", "int64_t*", "int")
    assert diagnostic.reason == "one or more argument types could not be inferred"
    assert set(diagnostic.candidates) == {
        "elem_to_loc_int64_t(int64_t, constant const int*, "
        "constant const int64_t*, int)",
        "elem_to_loc_int64_t(uint3, constant const int*, "
        "constant const int64_t*, int)",
    }


def metal_materialized_callable_array_auto_source():
    return """
    struct DivMod {
      template <typename T>
      metal::array<T, 2> operator()(T x, T y) {
        return {x / y, x % y};
      }
    };

    kernel void aggregate_auto(
        device int* quotients [[buffer(0)]],
        device int* remainders [[buffer(1)]]) {
      auto out = DivMod{}(7, 3);
      quotients[0] = out[0];
      remainders[0] = out[1];
    }
    """


def test_codegen_infers_materialized_callable_fixed_array_auto_local(tmp_path):
    source_path = tmp_path / "materialized_callable_array.metal"
    source_path.write_text(
        metal_materialized_callable_array_auto_source(), encoding="utf-8"
    )

    direct = crosstl.translate(str(source_path), backend="cgl", format_output=False)
    report = translate_project(tmp_path, targets=["cgl"], output_dir="out")
    payload = report.to_json()
    assert payload["summary"]["translatedCount"] == 1, payload
    assert payload["summary"]["failedCount"] == 0, payload
    project = (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")

    expected_declaration = "int[2] out_ = DivMod__operator_call__int__temporary(7, 3);"
    assert expected_declaration in normalize(direct)
    assert expected_declaration in normalize(project)

    shader = parse_crossgl(direct)
    hlsl = TranslatorHLSLCodeGen().generate(shader)
    assert "int2 out_ = DivMod__operator_call__int__temporary(7, 3);" in hlsl
    assert "int out_[2] = DivMod__operator_call__int__temporary" not in hlsl

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "materialized_callable_array.hlsl"
        binary_output = tmp_path / "materialized_callable_array.dxil"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(binary_output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    spirv = VulkanSPIRVCodeGen().generate(shader)
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        spirv_path = tmp_path / "materialized_callable_array.spvasm"
        binary_path = tmp_path / "materialized_callable_array.spv"
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


def test_codegen_auto_aggregate_uses_selected_overload_return_identity():
    source = """
    using IntPair = metal::array<int, 2>;

    struct Record {
      float value;
    };

    IntPair select_value(int value) {
      return {value, value + 1};
    }

    Record select_value(float value) {
      return {value};
    }

    kernel void select_aggregates(device int* output [[buffer(0)]]) {
      auto pair = select_value(4);
      auto record = select_value(4.0f);
      output[0] = pair[1] + int(record.value);
    }
    """

    normalized = normalize(convert_without_preprocessing(source))
    assert "int[2] pair = select_value(4);" in normalized
    assert "Record record = select_value(4.0f);" in normalized


def test_codegen_auto_pointer_arithmetic_preserves_device_pointer_type():
    source = """
    float read_offset(const device float* input, uint offset) {
      const auto cursor = input + offset;
      const auto nested = 1u + cursor;
      return nested[0];
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "const device float* cursor = input + offset;" in normalized
    assert "const device float* nested = 1u + cursor;" in normalized
    assert "return nested[0];" in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_arithmetic_preserves_writable_pointer_type():
    source = """
    void write_offset(device float* output, uint offset, float value) {
      auto cursor = output + offset;
      cursor += 1u;
      cursor[0] = value;
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "device float* cursor = output + offset;" in normalized
    assert "cursor += 1u;" in normalized
    assert "cursor[0] = value;" in normalized
    assert "const device float* cursor" not in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_arithmetic_preserves_threadgroup_address_space():
    source = """
    float read_threadgroup(threadgroup float* values, uint offset) {
      auto cursor = values - offset;
      return cursor[0];
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "threadgroup float* cursor = values - offset;" in normalized
    assert "return cursor[0];" in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_constructor_binding_preserves_cast_and_array_qualifiers():
    source = """
    struct Loader {
      Loader(
          device const uint8_t* weights,
          threadgroup float* shared,
          ushort lane) {}
    };

    void build_loader(
        device const void* raw_weights,
        device void* mutable_raw_weights,
        uint lane) {
      auto weights = (device const uint8_t*)raw_weights;
      auto mutable_weights = (device uint8_t*)mutable_raw_weights;
      threadgroup float fixed_shared[32];
      constexpr int BK_padded = 36;
      constexpr int BN_padded = 40;
      threadgroup float dependent_shared[
          true ? 32 * BK_padded : 32 * BN_padded];
      Loader fixed_loader(weights + lane, fixed_shared, lane);
      Loader dependent_loader(weights + lane, dependent_shared, lane);
      Loader const_loader(mutable_weights + lane, fixed_shared, lane);
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "const device uint8* weights = (uint8*)raw_weights;" in normalized
    assert (
        "Loader fixed_loader = crosstl_ctor_Loader_1("
        "weights + lane, fixed_shared, lane);" in normalized
    )
    assert (
        "Loader dependent_loader = crosstl_ctor_Loader_1("
        "weights + lane, dependent_shared, lane);" in normalized
    )
    assert (
        "Loader const_loader = crosstl_ctor_Loader_1("
        "mutable_weights + lane, fixed_shared, lane);" in normalized
    )
    assert parse_crossgl(crossgl) is not None


def test_codegen_constructor_factory_initializes_const_value_members(tmp_path):
    source = """
    struct Loader {
      const int src_ld;
      const int tile_stride;
      const int lane;

      Loader(int src_ld_, int lane_)
          : src_ld(src_ld_),
            tile_stride(src_ld * 2),
            lane(lane_) {}
    };

    kernel void build_loader(
        device int* output [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
      Loader loader(int(gid), int(gid));
      output[gid] = loader.src_ld + loader.tile_stride + loader.lane;
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    src_assignment = "crosstl_ctor_value.src_ld = int(src_ld_);"
    stride_assignment = (
        "crosstl_ctor_value.tile_stride = int(crosstl_ctor_value.src_ld * 2);"
    )
    lane_assignment = "crosstl_ctor_value.lane = int(lane_);"
    assert src_assignment in normalized
    assert stride_assignment in normalized
    assert lane_assignment in normalized
    assert normalized.index(src_assignment) < normalized.index(stride_assignment)
    assert normalized.index(stride_assignment) < normalized.index(lane_assignment)

    shader = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(shader)
    glsl = GLSLCodeGen().generate(shader)
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "const-member-constructor.hlsl"
        binary_path = tmp_path / "const-member-constructor.dxil"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(binary_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_opengl_compute_validates_if_available(
        glsl,
        tmp_path,
        "const-member-constructor",
    )


def test_codegen_constructor_factory_initializes_fixed_member_arrays(tmp_path):
    source = """
    struct FragmentTile {
      using frag_type = metal::vec<float, 2>;
      frag_type val_frags[3] = {frag_type(1.25f)};
      float weights[2] = {2};

      explicit FragmentTile(uint lane) {}
    };

    kernel void read_fragments(
        device float* output [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
      FragmentTile tile(gid);
      output[gid] = tile.val_frags[0][0]
          + tile.val_frags[1][0]
          + tile.val_frags[2][1]
          + tile.weights[0]
          + tile.weights[1];
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "vec2[3] val_frags;" in normalized
    assert "crosstl_ctor_value.val_frags[0] = vec2(1.25f);" in normalized
    assert "crosstl_ctor_value.val_frags =" not in normalized
    assert (
        "for (int crosstl_ctor_value_val_frags_index = 1; "
        "crosstl_ctor_value_val_frags_index < 3; "
        "crosstl_ctor_value_val_frags_index++)" in normalized
    )
    assert (
        "crosstl_ctor_value.val_frags[crosstl_ctor_value_val_frags_index] "
        "= vec2(0);" in normalized
    )
    assert "float[2] weights;" in normalized
    assert "crosstl_ctor_value.weights[0] = float(2);" in normalized
    assert (
        "crosstl_ctor_value.weights[crosstl_ctor_value_weights_index] = 0;"
        in normalized
    )

    shader = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(shader)
    glsl = GLSLCodeGen().generate(shader)
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "fixed-member-array-constructor.hlsl"
        binary_path = tmp_path / "fixed-member-array-constructor.dxil"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(binary_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_opengl_compute_validates_if_available(
        glsl,
        tmp_path,
        "fixed-member-array-constructor",
    )


@pytest.mark.parametrize(
    ("member_declaration", "reason"),
    [
        pytest.param(
            "int values[2][2] = {{1, 2}, {3, 4}};",
            "member array 'values' is multidimensional",
            id="multidimensional",
        ),
        pytest.param(
            "int values[1] = {1, 2};",
            "member array 'values' initializer contains more elements than its extent",
            id="excess-elements",
        ),
    ],
)
def test_codegen_constructor_factory_rejects_unrepresentable_member_arrays(
    member_declaration,
    reason,
):
    source = f"""
    struct Tile {{
      {member_declaration}
      Tile() {{}}
    }};

    void build_tile() {{
      Tile tile;
    }}
    """

    with pytest.raises(MetalConstructorContractError) as raised:
        convert_without_preprocessing(source)

    assert raised.value.owner == "Tile"
    assert raised.value.reason == reason


@pytest.mark.parametrize(
    ("actual_qualifiers", "parameter_qualifiers"),
    [
        pytest.param("thread const", "device const", id="address-space"),
        pytest.param("device const", "device", id="cv-removal"),
    ],
)
def test_codegen_constructor_binding_rejects_pointer_qualifier_mismatch(
    actual_qualifiers, parameter_qualifiers
):
    source = f"""
    struct Loader {{
      Loader({parameter_qualifiers} uint8_t* weights) {{}}
    }};

    void build_loader(void* raw_weights) {{
      auto weights = ({actual_qualifiers} uint8_t*)raw_weights;
      Loader loader(weights);
    }}
    """

    with pytest.raises(MetalConstructorContractError) as raised:
        convert_without_preprocessing(source)

    assert raised.value.owner == "Loader"
    assert raised.value.argument_types == ("uint8_t*",)
    assert raised.value.reason == (
        "no source-compatible constructor matches the inferred types"
    )


def test_codegen_auto_pointer_dereference_infers_pointee_type():
    source = """
    void read_offset(
        const device float* input,
        device float* output,
        uint offset) {
      const auto cursor = input + offset;
      auto first = *cursor;
      auto second = *(cursor + 1u);
      output[0] = first + second;
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "const device float* cursor = input + offset;" in normalized
    assert "float first = (*cursor);" in normalized
    assert "float second = (*(cursor + 1u));" in normalized
    assert "float* first" not in normalized
    assert "float* second" not in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_from_index_preserves_device_provenance():
    source = """
    float read_indexed(const device float* values, uint index, uint offset) {
      auto cursor = &values[index];
      const auto nested = cursor + offset;
      return nested[0];
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "const device float* cursor = (&values[index]);" in normalized
    assert "const device float* nested = cursor + offset;" in normalized
    assert "return nested[0];" in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_from_index_preserves_writable_storage():
    source = """
    void write_indexed(device float* values, uint index, float value) {
      auto cursor = &values[index];
      cursor[1] = value;
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "device float* cursor = (&values[index]);" in normalized
    assert "cursor[1] = value;" in normalized
    assert "const device float* cursor" not in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_from_nested_local_index_preserves_allocation():
    source = """
    void address_local(uint row, uint column) {
      threadgroup volatile float shared_values[4][8];
      int private_values[4][8];
      auto shared_cursor = &shared_values[row][column];
      auto private_cursor = &private_values[row][column];
      shared_cursor[0] = private_cursor[0];
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert (
        "volatile threadgroup float* shared_cursor = "
        "(&shared_values[row][column]);" in normalized
    )
    assert "thread int* private_cursor = (&private_values[row][column]);" in normalized
    assert "shared_cursor[0] = private_cursor[0];" in normalized
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_from_index_evaluates_offset_once():
    source = """
    uint next_index(thread uint& index) {
      return index++;
    }

    float read_once(const device float* values, thread uint& index) {
      auto cursor = &values[next_index(index)];
      return cursor[0];
    }
    """

    crossgl = convert_without_preprocessing(source)
    declaration = next(line for line in crossgl.splitlines() if "cursor =" in line)

    assert declaration.count("next_index(index)") == 1
    assert "const device float* cursor = (&values[next_index(index)]);" in normalize(
        declaration
    )
    assert parse_crossgl(crossgl) is not None


def test_codegen_auto_pointer_from_index_reaches_portable_target_offsets(tmp_path):
    source = """
    kernel void indexed_alias(
        const device float* values [[buffer(0)]],
        device float* results [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
      auto cursor = &values[gid];
      results[gid] = cursor[1];
    }
    """

    crossgl = convert_without_preprocessing(source)
    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "int64_t cursor_offset = int64_t(gid);" in hlsl
    assert "results[gid] = values[uint((cursor_offset + 1))];" in hlsl
    assert "int cursor_offset = int(gid);" in glsl
    assert "results[gid] = values[(cursor_offset + 1)];" in glsl

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "metal-indexed-address.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-indexed-address"
    )


@pytest.mark.parametrize(
    ("initializer", "operand_kind"),
    [
        ("&make_value()[index]", "vector subscript"),
        ("&value.xy", "vector swizzle"),
    ],
)
def test_codegen_rejects_auto_pointer_without_addressable_storage(
    initializer, operand_kind
):
    source = f"""
    float4 make_value() {{
      return float4(1.0f);
    }}

    void invalid_address(float4 value, uint index) {{
      auto pointer = {initializer};
    }}
    """

    with pytest.raises(MetalAddressProvenanceError) as exc_info:
        convert_without_preprocessing(source)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-address-provenance-unresolved"
    )
    assert diagnostic.missing_capabilities == ("metal.address-provenance-inference",)
    assert diagnostic.operand_kind == operand_kind


def test_project_reports_unresolved_metal_address_provenance(tmp_path):
    source = """
    float4 make_value() {
      return float4(1.0f);
    }

    kernel void invalid_address(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      auto pointer = &make_value()[index];
      output[index] = pointer[0];
    }
    """
    (tmp_path / "invalid_address.metal").write_text(source, encoding="utf-8")

    payload = translate_project(tmp_path, targets=["cgl"], output_dir="out").to_json()

    assert payload["summary"]["translatedCount"] == 0, payload
    assert payload["summary"]["failedCount"] == 1, payload
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.metal-address-provenance-unresolved": 1
    }
    assert payload["summary"]["missingCapabilityCounts"] == {
        "metal.address-provenance-inference": 1
    }


def test_codegen_reports_ambiguous_selected_auto_return_type(tmp_path):
    source = """
    auto ambiguous_result(bool use_integer) {
      if (use_integer) {
        return 1;
      }
      return 1.0f;
    }

    kernel void use_ambiguous_result(device int* output [[buffer(0)]]) {
      auto result = ambiguous_result(true);
      output[0] = result;
    }
    """

    with pytest.raises(MetalAutoTypeInferenceError) as exc_info:
        convert_without_preprocessing(source)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-auto-type-unresolved"
    )
    assert diagnostic.missing_capabilities == ("metal.auto-local-type-inference",)
    assert diagnostic.variable_name == "result"
    assert diagnostic.callable_name == "ambiguous_result"
    assert diagnostic.return_type == "auto"
    assert diagnostic.reason == "the selected callable return type is auto"

    source_path = tmp_path / "ambiguous_auto.metal"
    source_path.write_text(source, encoding="utf-8")
    payload = translate_project(tmp_path, targets=["cgl"], output_dir="out").to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert payload["summary"]["failedCount"] == 1, payload
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.metal-auto-type-unresolved": 1
    }
    assert payload["summary"]["missingCapabilityCounts"] == {
        "metal.auto-local-type-inference": 1
    }


def test_codegen_reports_unbound_selected_template_return_type():
    source = """
    template <typename Result>
    Result make_result(int value) {
      return Result(value);
    }

    template <typename Result>
    void use_result(device int* output) {
      auto result = make_result(1);
      output[0] = result;
    }
    """

    with pytest.raises(MetalAutoTypeInferenceError) as exc_info:
        convert_without_preprocessing(source)

    diagnostic = exc_info.value
    assert diagnostic.variable_name == "result"
    assert diagnostic.callable_name == "make_result"
    assert diagnostic.return_type == "Result"
    assert diagnostic.unresolved_parameters == ("Result",)
    assert diagnostic.reason == (
        "the selected callable return type remains dependent on Result"
    )


def test_codegen_infers_selected_template_return_from_argument_type():
    source = """
    template <typename T>
    T identity(T value) {
      return value;
    }

    template <typename T>
    void use_identity(device T* output, T value) {
      auto result = identity(value);
      output[0] = result;
    }
    """

    normalized = normalize(convert_without_preprocessing(source))
    assert "T result = identity(value);" in normalized


def test_codegen_preserves_resource_qualifiers_during_source_overload_binding(
    tmp_path,
):
    source = """
    int64_t pick(int64_t value, constant const int* data) {
      return value + data[0];
    }

    int64_t pick(int64_t value, device int* data) {
      return value + data[0];
    }

    int64_t pick(uint3 value, constant const int* data) {
      return int64_t(value.x) + data[0];
    }

    kernel void run_picks(
        device int64_t* out_values [[buffer(0)]],
        constant const int* data [[buffer(1)]],
        device int* mutable_data [[buffer(2)]],
        uint3 index [[thread_position_in_grid]]) {
      auto kidx = 2 * index.x;
      auto scalar = pick(kidx, data);
      auto device_scalar = pick(kidx, mutable_data);
      auto vector = pick(index, data);
      out_values[0] = scalar + device_scalar + vector;
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "int64 pick(int64 value, constant int* data)" in normalized
    assert "int64 pick__metal_overload_2(int64 value, device int* data)" in normalized
    assert "int64 pick__metal_overload_3(uvec3 value, constant int* data)" in normalized
    assert "int64 scalar = pick(kidx, data);" in normalized
    assert (
        "int64 device_scalar = pick__metal_overload_2(kidx, mutable_data);"
        in normalized
    )
    assert "int64 vector = pick__metal_overload_3(index, data);" in normalized

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-qualified-source-overload"
    )


def test_codegen_infers_nested_metal_builtin_results_across_targets(tmp_path):
    source = """
    struct complex64_t {
      float real;
      float imag;
    };

    float consume(float value) { return value; }
    half consume(half value) { return value; }
    float3 consume(float3 value) { return value; }
    half3 consume(half3 value) { return value; }
    complex64_t consume(complex64_t value) { return value; }

    kernel void nested_builtin_results(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      float scalar = float(index) * 0.5f;
      half half_scalar = half(index) * 0.5h;
      float3 vector_value = float3(scalar, scalar + 1.0f, scalar + 2.0f);
      bool condition = (index & 1u) != 0u;
      output[index * 8u] = consume(metal::exp(scalar));
      output[index * 8u + 1u] = consume(
          condition ? metal::exp(scalar) : scalar);
      output[index * 8u + 2u] = consume(metal::exp(scalar) + scalar);
      output[index * 8u + 3u] = consume(metal::exp(vector_value)).x;
      output[index * 8u + 4u] = float(consume(metal::exp(half_scalar)));
      output[index * 8u + 5u] = consume(
          metal::clamp(vector_value, 0.0f, 1.0f)).x;
      output[index * 8u + 6u] = consume(
          metal::step(0.0f, vector_value)).x;
      output[index * 8u + 7u] = consume(
          metal::mix(vector_value, float3(1.0f), 0.5f)).x;
    }
    """
    repo = tmp_path / "nested-builtin-results"
    repo.mkdir()
    source_path = repo / "nested.metal"
    source_path.write_text(source, encoding="utf-8")

    crossgl = crosstl.translate(str(source_path), backend="cgl", format_output=False)
    normalized = normalize(crossgl)
    assert "consume__metal_overload_1(exp(scalar))" in normalized
    assert "consume__metal_overload_1(condition ? exp(scalar) : scalar)" in normalized
    assert "consume__metal_overload_1(exp(scalar) + scalar)" in normalized
    assert "consume__metal_overload_3(exp(vector_value)).x" in normalized
    assert "consume__metal_overload_2(exp(half_scalar))" in normalized
    assert "consume__metal_overload_3(clamp(vector_value, 0.0f, 1.0f)).x" in normalized
    assert "consume__metal_overload_3(step(0.0f, vector_value)).x" in normalized
    assert (
        "consume__metal_overload_3(mix(vector_value, vec3(1.0f), 0.5f)).x" in normalized
    )
    assert "consume__metal_overload_5(exp(" not in normalized

    direct_outputs = {
        target: crosstl.translate(str(source_path), backend=target, format_output=False)
        for target in ("directx", "opengl")
    }
    report = translate_project(
        repo,
        targets=["directx", "opengl"],
        output_dir="out",
    )
    payload = report.to_json()
    assert payload["summary"]["translatedCount"] == 2, payload
    assert payload["summary"]["failedCount"] == 0, payload
    artifacts = {artifact["target"]: artifact for artifact in payload["artifacts"]}
    for target, direct in direct_outputs.items():
        project = (repo / artifacts[target]["path"]).read_text(encoding="utf-8")
        for generated in (direct, project):
            assert "<unknown>" not in generated
            assert "metal::" not in generated
            assert "exp(" in generated

    hlsl = direct_outputs["directx"]
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()
    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "nested-builtin-results.hlsl"
        binary_path = tmp_path / "nested-builtin-results.dxil"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_2",
                "-enable-16bit-types",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(binary_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert_opengl_compute_validates_if_available(
        direct_outputs["opengl"], tmp_path, "nested-builtin-results"
    )


def test_codegen_prefers_qualified_bfloat_builtin_overload_result():
    overloads = """
    typedef bfloat bfloat16_t;

    struct complex64_t {
      float real;
      float imag;
    };

    float consume(float value) { return value; }
    bfloat16_t consume(bfloat16_t value) { return value; }
    complex64_t consume(complex64_t value) { return value; }
    """
    standard_source = overloads + """
        float standard_exp_result(bfloat16_t value) {
          return consume(metal::exp(value));
        }
    """
    custom_source = overloads + """
        float exp(float value) { return value; }

        namespace metal {
          bfloat16_t exp(bfloat16_t value) { return value; }
        }

        using namespace metal;

        bfloat16_t custom_exp_result(bfloat16_t value) {
          return consume(metal::exp(value));
        }

        bfloat16_t custom_unqualified_exp_result(bfloat16_t value) {
          return consume(exp(value));
        }
        """
    reference_source = """
        float consume(float value) { return value; }
        half consume(half value) { return value; }

        float converted_sincos_result(half value) {
          float cosine;
          return consume(metal::sincos(value, cosine));
        }
        """
    fast_source = overloads + """
        float fast_exp_result(half value) {
          return consume(metal::fast::exp(value));
        }

        float precise_exp_result(half value) {
          return consume(metal::precise::exp(value));
        }
        """

    standard = normalize(convert_without_preprocessing(standard_source))
    custom = normalize(convert_without_preprocessing(custom_source))
    reference = normalize(convert_without_preprocessing(reference_source))
    fast = normalize(convert_without_preprocessing(fast_source))
    assert "return consume__metal_overload_1(exp(value));" in standard
    assert "return consume__metal_overload_2(exp__metal_overload_2(value));" in custom
    assert (
        custom.count("return consume__metal_overload_2(exp__metal_overload_2(value));")
        == 2
    )
    assert "return consume__metal_overload_1(sincos(value, cosine));" in reference
    assert "return consume__metal_overload_1(exp(value));" in fast
    assert "consume__metal_overload_2(exp(value))" not in fast
    assert "consume__metal_overload_3(exp(value))" not in standard
    assert "consume__metal_overload_3(exp(value))" not in custom


def test_codegen_composes_precise_bfloat_extension_with_standard_float_builtin():
    source = """
    typedef bfloat bfloat16_t;

    namespace metal {
    namespace precise {
      bfloat16_t sqrt(bfloat16_t value) { return value; }
    }
    }

    struct complex64_t {
      float real;
      float imag;

      template <typename T>
      complex64_t(T value) : real(float(value)), imag(0.0f) {}
    };

    bfloat16_t extension_path(bfloat16_t value) {
      auto result = metal::precise::sqrt(value);
      return result;
    }

    complex64_t standard_path(float value) {
      return (complex64_t)metal::precise::sqrt(value);
    }
    """

    normalized = normalize(convert_without_preprocessing(source))

    assert "bfloat16 result = sqrt(value);" in normalized
    assert "complex64_t crosstl_ctor_complex64_t_1_float(float value)" in normalized
    assert "return crosstl_ctor_complex64_t_1_float(sqrt(value));" in normalized
    assert "<unknown>" not in normalized


def test_codegen_keeps_metal_stdlib_wrappers_as_non_emitted_builtin_metadata():
    source = """
    typedef bfloat bfloat16_t;

    namespace metal {
    METAL_FUNC bfloat16_t exp(bfloat16_t value) {
      return bfloat16_t(__metal_exp(float(value)));
    }

    METAL_FUNC bfloat16_t simd_max(bfloat16_t value) {
      return bfloat16_t(__metal_simd_max(float(value)));
    }

    namespace fast {
    METAL_FUNC bfloat16_t exp(bfloat16_t value) {
      return bfloat16_t(__metal_fast_math(float(value), __METAL_EXP));
    }
    }

    namespace precise {
    METAL_FUNC bfloat16_t exp(bfloat16_t value) {
      return bfloat16_t(__metal_precise_math(float(value), __METAL_EXP));
    }
    }
    }

    kernel void wrapper_results(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      bfloat16_t value = bfloat16_t(float(index));
      auto standard_result = metal::exp(value);
      auto fast_result = metal::fast::exp(value);
      auto precise_result = metal::precise::exp(value);
      auto simd_result = metal::simd_max(value);
      output[index] = float(
          standard_result + fast_result + precise_result + simd_result);
    }
    """

    crossgl = convert_without_preprocessing(source)
    shader = parse_crossgl(crossgl)
    generated_targets = {
        "crossgl": crossgl,
        "directx": TranslatorHLSLCodeGen().generate(shader),
        "opengl": GLSLCodeGen().generate(shader),
    }
    expected_result_types = {
        "crossgl": "bfloat16",
        "directx": "uint",
        "opengl": "float",
    }

    for target, generated in generated_targets.items():
        normalized = normalize(generated)
        result_type = expected_result_types[target]
        for result_name in (
            "standard_result",
            "fast_result",
            "precise_result",
            "simd_result",
        ):
            assert f"{result_type} {result_name}" in normalized
        assert generated.count("exp(") == 3
        assert "simd_max" not in generated
        assert "__metal_" not in generated
        assert "__METAL_" not in generated
        assert "<unknown>" not in generated

    assert "bfloat16 simd_result = bfloat16(WaveActiveMax(float(value)));" in normalize(
        crossgl
    )
    assert (
        "uint simd_result = __crossgl_bfloat16_from_float("
        "float(WaveActiveMax(__crossgl_bfloat16_to_float(uint(value)))));"
        in normalize(generated_targets["directx"])
    )
    assert "float simd_result = float(subgroupMax(float(value)));" in normalize(
        generated_targets["opengl"]
    )


@pytest.mark.parametrize("namespace", ["fast", "precise"])
def test_codegen_canonicalizes_qualified_log10_stdlib_wrapper(namespace):
    source = f"""
    typedef bfloat bfloat16_t;

    namespace metal {{
    namespace {namespace} {{
    METAL_FUNC bfloat16_t log10(bfloat16_t value) {{
      return bfloat16_t(__metal_log10(float(value), true));
    }}
    }}
    }}

    kernel void wrapper_result(
        device float* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {{
      bfloat16_t value = bfloat16_t(float(index));
      auto result = metal::{namespace}::log10(value);
      output[index] = float(result);
    }}
    """

    crossgl = convert_without_preprocessing(source)
    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "bfloat16 result = log10(value);" in normalize(crossgl)
    assert (
        "uint result = __crossgl_bfloat16_from_float("
        "float(log10(__crossgl_bfloat16_to_float(uint(value)))));" in normalize(hlsl)
    )
    for generated in (crossgl, hlsl):
        assert generated.count("log10(") == 1
        assert "__metal_log10" not in generated
        assert "metal::" not in generated
        assert "<unknown>" not in generated


@pytest.mark.parametrize("namespace", ["fast", "precise"])
def test_codegen_canonicalizes_qualified_rint_stdlib_wrapper(namespace):
    source = f"""
    typedef bfloat bfloat16_t;

    namespace metal {{
    namespace {namespace} {{
    METAL_FUNC bfloat16_t rint(bfloat16_t value) {{
      return bfloat16_t(__metal_rint(float(value), true));
    }}
    }}
    }}

    float4 rounded_values(float4 values, bfloat16_t value) {{
      auto wrapper_result = metal::{namespace}::rint(value);
      float4 vector_result = metal::{namespace}::rint(values);
      return vector_result + float4(float(wrapper_result));
    }}
    """

    crossgl = convert_without_preprocessing(source)
    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)

    assert "bfloat16 wrapper_result = rint(value);" in normalize(crossgl)
    assert "vec4 vector_result = rint(values);" in normalize(crossgl)
    assert (
        "uint wrapper_result = __crossgl_bfloat16_from_float("
        "float(round(__crossgl_bfloat16_to_float(uint(value)))));" in normalize(hlsl)
    )
    assert "float4 vector_result = round(values);" in normalize(hlsl)
    assert "float wrapper_result = roundEven(value);" in normalize(glsl)
    assert "vec4 vector_result = roundEven(values);" in normalize(glsl)
    for generated in (crossgl, hlsl, glsl):
        assert "__metal_rint" not in generated
        assert "metal::" not in generated
        assert "<unknown>" not in generated


def test_codegen_canonicalizes_qualified_copysign_without_shadowing_user_code():
    builtin_source = """
    float4 copy_signs(
        float magnitude,
        float sign_source,
        float4 magnitudes,
        float4 sign_sources) {
      auto scalar_result = metal::copysign(magnitude, sign_source);
      auto vector_result = metal::copysign(magnitudes, sign_sources);
      return vector_result + float4(scalar_result);
    }
    """
    user_source = """
    namespace metal {
    float copysign(float magnitude, float sign_source) {
      return magnitude + sign_source;
    }
    }

    float copy_sign(float magnitude, float sign_source) {
      return metal::copysign(magnitude, sign_source);
    }
    """

    builtin = normalize(convert_without_preprocessing(builtin_source))
    user = normalize(convert_without_preprocessing(user_source))

    assert "float scalar_result = copysign(magnitude, sign_source);" in builtin
    assert "vec4 vector_result = copysign(magnitudes, sign_sources);" in builtin
    assert "float copysign(float magnitude, float sign_source)" in user
    assert "return magnitude + sign_source;" in user
    assert "return copysign(magnitude, sign_source);" in user


def test_codegen_reports_called_unsupported_metal_stdlib_wrapper():
    source = """
    namespace metal {
    METAL_FUNC float implementation_only(float value) {
      return __metal_unrepresentable(value);
    }
    }

    float use_wrapper(float value) {
      return metal::implementation_only(value);
    }
    """

    with pytest.raises(MetalStandardLibraryWrapperLoweringError) as exc_info:
        convert_without_preprocessing(
            source,
            file_path="unsupported-stdlib-wrapper.metal",
        )

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-stdlib-wrapper-unsupported"
    )
    assert diagnostic.missing_capabilities == (
        "metal.standard-library-wrapper-lowering",
    )
    assert diagnostic.function_name == "metal::implementation_only"
    assert diagnostic.implementation_intrinsics == ("__metal_unrepresentable",)
    assert diagnostic.source_location["file"] == "unsupported-stdlib-wrapper.metal"
    assert diagnostic.source_location["line"] == 9


def test_codegen_rejects_mismatched_metal_wave_wrapper_intrinsic():
    source = """
    typedef bfloat bfloat16_t;

    namespace metal {
    METAL_FUNC bfloat16_t simd_max(bfloat16_t value) {
      return bfloat16_t(__metal_simd_min(float(value)));
    }
    }

    bfloat16_t use_wrapper(bfloat16_t value) {
      return metal::simd_max(value);
    }
    """

    with pytest.raises(MetalStandardLibraryWrapperLoweringError) as exc_info:
        convert_without_preprocessing(
            source,
            file_path="mismatched-wave-wrapper.metal",
        )

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-stdlib-wrapper-unsupported"
    )
    assert diagnostic.function_name == "metal::simd_max"
    assert diagnostic.implementation_intrinsics == ("__metal_simd_min",)
    assert diagnostic.source_location["file"] == "mismatched-wave-wrapper.metal"
    assert diagnostic.source_location["line"] == 11


def test_codegen_rejects_metal_wave_wrapper_with_unknown_intrinsic():
    source = """
    typedef bfloat bfloat16_t;

    namespace metal {
    METAL_FUNC bfloat16_t simd_max(bfloat16_t value) {
      return bfloat16_t(
          __metal_simd_max(float(value)) + __metal_unrepresentable(float(value)));
    }
    }

    bfloat16_t use_wrapper(bfloat16_t value) {
      return metal::simd_max(value);
    }
    """

    with pytest.raises(MetalStandardLibraryWrapperLoweringError) as exc_info:
        convert_without_preprocessing(
            source,
            file_path="unknown-wave-wrapper-intrinsic.metal",
        )

    diagnostic = exc_info.value
    assert diagnostic.implementation_intrinsics == (
        "__metal_simd_max",
        "__metal_unrepresentable",
    )
    assert diagnostic.source_location["file"] == (
        "unknown-wave-wrapper-intrinsic.metal"
    )
    assert diagnostic.source_location["line"] == 12


def test_codegen_keeps_unqualified_bfloat_math_user_shadow():
    source = """
    typedef bfloat bfloat16_t;

    int abs(float value) {
      return int(value) + 1;
    }

    int use_shadow(bfloat16_t value) {
      auto result = abs(value);
      return result;
    }
    """

    normalized = normalize(convert_without_preprocessing(source))

    assert "int abs(float value)" in normalized
    assert "int result = abs(value);" in normalized
    assert "bfloat16 result = abs(value);" not in normalized


def test_codegen_reports_ambiguous_unqualified_metal_builtin_result_type():
    source = """
    float consume(float value) { return value; }
    half consume(half value) { return value; }

    float unresolved_builtin(int value) {
      return consume(exp(value));
    }
    """

    with pytest.raises(MetalBuiltinResultTypeResolutionError) as exc_info:
        convert_without_preprocessing(
            source,
            file_path="unqualified-builtin-result.metal",
        )

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-builtin-result-unresolved"
    )
    assert diagnostic.missing_capabilities == ("metal.builtin-result-type-inference",)
    assert diagnostic.function_name == "exp"
    assert diagnostic.argument_types == ("int",)
    assert diagnostic.reason == (
        "multiple builtin signatures remain viable after type matching"
    )
    assert set(diagnostic.candidates) == {
        "half exp(half)",
        "float exp(float)",
        "double exp(double)",
    }
    assert diagnostic.source_location["file"] == "unqualified-builtin-result.metal"
    assert diagnostic.source_location["line"] == 6
    assert diagnostic.source_location["column"] == 25


@pytest.mark.parametrize(
    ("parameter_type", "argument", "reason"),
    [
        (
            "float",
            "external_value(value)",
            "one or more builtin argument types could not be inferred",
        ),
        (
            "complex64_t",
            "value",
            "no builtin signature matches the inferred argument types",
        ),
        (
            "int",
            "value",
            "multiple builtin signatures remain viable after type matching",
        ),
    ],
)
def test_codegen_reports_unresolved_metal_builtin_result_type(
    parameter_type, argument, reason
):
    source = f"""
    struct complex64_t {{
      float real;
      float imag;
    }};

    float consume(float value) {{ return value; }}
    half consume(half value) {{ return value; }}
    complex64_t consume(complex64_t value) {{ return value; }}

    float unresolved_builtin({parameter_type} value) {{
      return float(consume(metal::exp({argument})));
    }}
    """

    with pytest.raises(MetalBuiltinResultTypeResolutionError) as exc_info:
        convert_without_preprocessing(source, file_path="builtin-result.metal")

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-builtin-result-unresolved"
    )
    assert diagnostic.missing_capabilities == ("metal.builtin-result-type-inference",)
    assert diagnostic.function_name == "metal::exp"
    assert diagnostic.reason == reason
    assert len(diagnostic.candidates) <= diagnostic.maximum_candidates
    assert diagnostic.source_location["file"] == "builtin-result.metal"
    assert diagnostic.source_location["line"] == 12
    assert diagnostic.source_location["column"] == 28
    if parameter_type == "int":
        assert set(diagnostic.candidates) == {
            "half metal::exp(half)",
            "float metal::exp(float)",
            "double metal::exp(double)",
        }
    if "external_value" in argument:
        assert diagnostic.argument_types == ("<unknown>",)


def test_codegen_fails_closed_for_equally_ranked_source_numeric_overloads():
    source = """
    int64_t pick(int64_t value, constant const int* data) {
      return value + data[0];
    }

    int64_t pick(float value, constant const int* data) {
      return int64_t(value) + data[0];
    }

    int64_t pick(uint3 value, constant const int* data) {
      return int64_t(value.x) + data[0];
    }

    kernel void run_pick(
        device int64_t* out_values [[buffer(0)]],
        constant const int* data [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      auto value = pick(index, data);
      out_values[0] = value;
    }
    """

    with pytest.raises(MetalSourceOverloadResolutionError) as exc_info:
        convert_without_preprocessing(source)

    diagnostic = exc_info.value
    assert diagnostic.argument_types == ("uint", "int*")
    assert diagnostic.reason == (
        "multiple source-compatible overloads remain after type matching"
    )
    assert set(diagnostic.candidates) == {
        "pick(int64_t, constant const int*)",
        "pick(float, constant const int*)",
    }


def test_codegen_reserves_transported_names_and_resets_converter_state():
    collision_source = """
    template <int Offset>
    int pick__metal_overload(int value) {
      return value + Offset;
    }

    int64_t pick(int64_t value, constant int* data) {
      return value + data[0];
    }

    int64_t pick(uint3 value, constant int* data) {
      return int64_t(value.x) + data[0];
    }

    kernel void run_collision(
        device int64_t* out_values [[buffer(0)]],
        constant int* data [[buffer(1)]],
        uint3 index [[thread_position_in_grid]]) {
      out_values[0] = pick(index, data)
          + pick__metal_overload<2>(int(index.x));
    }
    """
    clean_source = """
    int64_t pick(int64_t value, constant int* data) {
      return value + data[0];
    }

    int64_t pick(uint3 value, constant int* data) {
      return int64_t(value.x) + data[0];
    }

    kernel void run_clean(
        device int64_t* out_values [[buffer(0)]],
        constant int* data [[buffer(1)]],
        uint3 index [[thread_position_in_grid]]) {
      out_values[0] = pick(index, data);
    }
    """
    converter = MetalToCrossGLConverter()

    collision_ast = MetalParser(
        MetalLexer(collision_source, preprocess=False).tokenize()
    ).parse()
    collision_crossgl = converter.generate(collision_ast)
    collision_normalized = normalize(collision_crossgl)
    assert "int pick__metal_overload_2_2(int value)" in collision_normalized
    assert (
        "pick__metal_overload_2(index, data)"
        " + pick__metal_overload_2_2(int(index.x))" in collision_normalized
    )
    assert parse_crossgl(collision_crossgl) is not None

    clean_ast = MetalParser(
        MetalLexer(clean_source, preprocess=False).tokenize()
    ).parse()
    clean_crossgl = converter.generate(clean_ast)
    clean_normalized = normalize(clean_crossgl)
    assert "int64 pick__metal_overload_2(uvec3 value" in clean_normalized
    assert "pick__metal_overload_2(index, data)" in clean_normalized
    assert "pick__metal_overload_2_2" not in clean_normalized
    assert parse_crossgl(clean_crossgl) is not None


def test_codegen_leaves_portably_distinct_source_overloads_native():
    source = """
    float choose(float value) {
      return value + 1.0f;
    }

    int choose(int value) {
      return value - 1;
    }

    kernel void run_choose(
        device float* values [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      values[index] = choose(float(index)) + float(choose(int(index)));
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)
    assert "float choose(float value)" in normalized
    assert "int choose(int value)" in normalized
    assert "metal_overload" not in normalized
    assert "choose(float(index)) + float(choose(int(index)))" in normalized


def test_mlx_random_auto_local_overload_matches_direct_and_project_opengl(
    tmp_path,
):
    # Reduced from MLX 4367c73b60541ddd5a266ce4644fd93d20223b6e,
    # mlx/backend/metal/kernels/random.metal and utils.h.
    source = """
    template <typename IdxT = int64_t>
    IdxT elem_to_loc(
        IdxT elem,
        constant const int* shape,
        constant const int64_t* strides,
        int ndim) {
      return elem + IdxT(shape[0]) + strides[0] + ndim;
    }

    template <typename IdxT = int64_t>
    IdxT elem_to_loc(
        uint3 elem,
        constant const int* shape,
        constant const int64_t* strides,
        int ndim) {
      return IdxT(elem.x) + IdxT(shape[0]) + strides[0] + ndim;
    }

    kernel void rbits(
        device int64_t* out_values [[buffer(0)]],
        constant const int* key_shape [[buffer(1)]],
        constant const int64_t* key_strides [[buffer(2)]],
        constant const int& ndim [[buffer(3)]],
        uint2 index [[thread_position_in_grid]]) {
      auto kidx = 2 * index.x;
      auto first = elem_to_loc(kidx, key_shape, key_strides, ndim);
      auto second = elem_to_loc(kidx + 1, key_shape, key_strides, ndim);
      out_values[0] = first;
      out_values[1] = second;
    }
    """
    repo = tmp_path / "mlx-reduced"
    repo.mkdir()
    source_path = repo / "random.metal"
    source_path.write_text(source, encoding="utf-8")

    direct = crosstl.translate(str(source_path), backend="opengl", format_output=False)
    report = translate_project(repo, targets=["opengl"], output_dir="out")
    payload = report.to_json()
    assert payload["summary"]["translatedCount"] == 1, payload
    assert payload["summary"]["failedCount"] == 0, payload
    artifact = payload["artifacts"][0]
    project = (repo / artifact["path"]).read_text(encoding="utf-8")

    helper_pattern = r"int64_t (?:first|second) = ([A-Za-z_]\w*)\("
    direct_helpers = re.findall(helper_pattern, direct)
    project_helpers = re.findall(helper_pattern, project)
    assert len(direct_helpers) == len(project_helpers) == 2
    assert len(set(direct_helpers)) == len(set(project_helpers)) == 1
    assert direct_helpers[0] == project_helpers[0]
    assert "metal_overload" not in direct_helpers[0]
    assert_opengl_compute_validates_if_available(
        direct, tmp_path, "mlx-random-auto-local-overload"
    )


def test_mlx_distinct_float16_and_bfloat16_overloads_match_project_hlsl(tmp_path):
    # Reduced from MLX 4367c73b60541ddd5a266ce4644fd93d20223b6e,
    # mlx/backend/metal/kernels/binary_two.metal.
    source = """
    typedef bfloat bfloat16_t;

    struct Divide {};

    half Divide__operator_call(thread Divide& self, half x, half y) {
      return x / y;
    }

    bfloat16_t Divide__operator_call(
        thread Divide& self, bfloat16_t x, bfloat16_t y) {
      return x - y;
    }

    kernel void exercise_collapsed_overloads(
        device half* output [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      Divide operation;
      half half_value = half(index + 2u);
      bfloat16_t bfloat_value = bfloat16_t(index + 2u);
      output[index * 2u] = Divide__operator_call(
          operation, half_value, half(2.0h));
      output[index * 2u + 1u] = half(Divide__operator_call(
          operation, bfloat_value, bfloat16_t(2.0h)));
    }
    """
    repo = tmp_path / "mlx-reduced"
    repo.mkdir()
    source_path = repo / "binary_two.metal"
    source_path.write_text(source, encoding="utf-8")

    direct = crosstl.translate(str(source_path), backend="directx", format_output=False)
    report = translate_project(repo, targets=["directx"], output_dir="out")
    payload = report.to_json()
    assert payload["summary"]["translatedCount"] == 1, payload
    assert payload["summary"]["failedCount"] == 0, payload
    artifact = payload["artifacts"][0]
    project_path = repo / artifact["path"]
    project = project_path.read_text(encoding="utf-8")

    helper_types = {
        "Divide__operator_call__metal_overload_1": "float16_t",
        "Divide__operator_call__metal_overload_2": "uint",
    }
    for helper, scalar_type in helper_types.items():
        definition_pattern = (
            rf"{scalar_type} {helper}"
            rf"\(inout Divide self, {scalar_type} x, {scalar_type} y\) \{{"
        )
        assert re.search(definition_pattern, direct) is not None
        assert re.search(definition_pattern, project) is not None
        assert f"{helper}(operation," in direct
    assert " Divide__operator_call(inout Divide self," not in direct
    assert artifact["bfloat16Lowering"]["status"] == "exact"
    assert artifact["bfloat16Lowering"]["approximationUsed"] is False

    HLSLParser(HLSLLexer(direct).tokenize()).parse()
    source_output = tmp_path / "mlx-collapsed-overloads.hlsl"
    source_output.write_text(direct, encoding="utf-8")
    direct_entry_point = resolve_directx_numthreads_entry(source_output)
    project_entry_point = resolve_directx_numthreads_entry(project_path)
    assert direct_entry_point == project_entry_point
    assert direct_entry_point != "exercise_collapsed_overloads"

    dxc = shutil.which("dxc")
    if dxc is not None:
        binary_output = tmp_path / "mlx-collapsed-overloads.dxil"
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_2",
                "-enable-16bit-types",
                "-E",
                direct_entry_point,
                str(source_output),
                "-Fo",
                str(binary_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert binary_output.stat().st_size > 0


def test_mlx_random_vector_subscripts_keep_scalar_component_contract(tmp_path):
    # Reduced from MLX 4367c73b60541ddd5a266ce4644fd93d20223b6e,
    # mlx/backend/metal/kernels/random.metal::threefry2x32_hash.
    source = """
    struct rbits {
        uint2 val;
    };

    uint select_component(uint value) {
        return value;
    }

    uint4 select_component(uint4 value) {
        return value;
    }

    kernel void threefry2x32_hash(
        device uint* out_values [[buffer(0)]],
        uint lane [[thread_position_in_grid]]) {
        uint2 key = uint2(11u, 17u);
        uint2 count = uint2(5u, 7u);
        uint4 ks = uint4(key.x, key.y, key.x ^ key.y ^ 0x1BD11BDAu, 0u);
        rbits v;
        v.val.x = count.x + ks[0];
        auto constant_component = ks[1];
        auto runtime_component = ks[lane];
        auto selected_call = select_component(ks[lane]);
        uint4 assigned = ks;
        assigned[0] = constant_component;
        assigned[lane] += runtime_component;
        if ((lane & 1u) == 0u) {
            ++assigned[lane];
            assigned[lane]++;
        }
        out_values[0] = v.val.x + selected_call + assigned.x;
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)
    helper = "CrossGLMetalVectorIndex_uvec4"

    assert f"v.val.x = count.x + {helper}_get(ks, 0);" in normalized
    assert f"uint constant_component = {helper}_get(ks, 1);" in normalized
    assert f"uint runtime_component = {helper}_get(ks, lane);" in normalized
    assert (
        f"uint selected_call = select_component({helper}_get(ks, lane));" in normalized
    )
    assert f"{helper}_set(assigned, 0, constant_component);" in normalized
    assert f"{helper}_add_assign(assigned, lane, runtime_component);" in normalized
    assert f"{helper}_pre_increment(assigned, lane);" in normalized
    assert f"{helper}_post_increment(assigned, lane);" in normalized
    assert f"uvec4({helper}_get" not in normalized
    assert "uvec4(ks[" not in normalized

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert f"uint constant_component = {helper}_get(ks, 1);" in glsl
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "mlx-random-vector-components"
    )


@pytest.mark.parametrize(
    ("vector_type", "mapped_vector", "mapped_component"),
    [
        ("float4", "vec4", "float"),
        ("int4", "ivec4", "int"),
        ("uint4", "uvec4", "uint"),
        ("bool4", "bvec4", "bool"),
        ("short4", "i16vec4", "int16"),
        ("ushort4", "u16vec4", "uint16"),
        ("char4", "i8vec4", "int8"),
        ("uchar4", "u8vec4", "uint8"),
        ("long4", "i64vec4", "int64"),
        ("ulong4", "u64vec4", "uint64"),
        ("half4", "f16vec4", "float16"),
        ("bfloat4", "bfloat16vec4", "bfloat16"),
    ],
)
def test_codegen_vector_subscript_component_scalar_families(
    vector_type, mapped_vector, mapped_component
):
    source_scalar = vector_type[:-1]
    source = f"""
    {source_scalar} extract_component({vector_type} value, uint lane) {{
        auto selected = value[lane];
        return selected;
    }}
    """

    crossgl = normalize(convert(source))
    helper = f"CrossGLMetalVectorIndex_{mapped_vector}_get"

    assert f"{mapped_component} {helper}({mapped_vector} value, uint lane)" in crossgl
    assert f"{mapped_component} selected = {helper}(value, lane);" in crossgl


def test_codegen_nested_array_vector_matrix_and_swizzle_types():
    source = """
    void inspect_components(
        thread uint4 vectors[2],
        thread float3x3 matrices[2],
        uint row,
        uint column,
        uint lane) {
        auto vector_value = vectors[row];
        auto vector_lane = vectors[row][lane];
        auto scalar_swizzle = vectors[row].x;
        auto vector_swizzle = vectors[row].xy;
        auto matrix_column = matrices[row][column];
        auto matrix_lane = matrices[row][column][lane];
    }
    """

    crossgl = normalize(convert(source))

    assert "uvec4 vector_value = vectors[row];" in crossgl
    assert (
        "uint vector_lane = CrossGLMetalVectorIndex_uvec4_get("
        "vectors[row], lane);" in crossgl
    )
    assert "uint scalar_swizzle = vectors[row].x;" in crossgl
    assert "uvec2 vector_swizzle = vectors[row].xy;" in crossgl
    assert "vec3 matrix_column = matrices[row][column];" in crossgl
    assert (
        "float matrix_lane = CrossGLMetalVectorIndex_vec3_get("
        "matrices[row][column], lane);" in crossgl
    )
    assert "CrossGLMetalVectorIndex_uvec4_get(vectors, row)" not in crossgl
    assert "CrossGLMetalVectorIndex_vec3_get(matrices[row], column)" not in crossgl


def test_codegen_auto_struct_return_retains_nested_array_vector_type():
    source = """
    struct LaneBits {
        uchar4 bytes[2];
    };

    LaneBits make_bits(const thread uint2& key) {
        LaneBits result;
        result.bytes[0] = uchar4(key.x);
        result.bytes[1] = uchar4(key.y);
        return result;
    }

    uint read_byte(uint2 key, uint lane) {
        auto bits = make_bits(key);
        auto selected = bits.bytes[0][lane];
        return uint(selected);
    }
    """

    crossgl = normalize(convert(source))

    assert "LaneBits bits = make_bits(key);" in crossgl
    assert (
        "uint8 selected = CrossGLMetalVectorIndex_u8vec4_get("
        "bits.bytes[0], lane);" in crossgl
    )
    assert "CrossGLMetalVectorIndex_u8vec4_get(bits.bytes, 0)" not in crossgl


def test_codegen_type_alias_member_array_keeps_component_type():
    source = """
    struct Tile {
        float2 values[2];
    };

    void read_alias(uint lane) {
        using tile_t = Tile;
        tile_t tile;
        tile.values[0][lane] = 1.0f;
    }
    """

    crossgl = normalize(convert(source))

    assert "CrossGLMetalVectorIndex_vec2_set(tile.values[0], lane, 1.0f);" in crossgl


def test_codegen_struct_scoped_alias_member_array_keeps_component_type(tmp_path):
    source = """
    struct Tile {
        using frag_type = metal::vec<float, 2>;
        frag_type val_frags[4];
    };

    kernel void read_fragment(
        device float* output [[buffer(0)]],
        uint2 index [[thread_position_in_grid]]) {
        Tile tile;
        tile.val_frags[0] = metal::vec<float, 2>(1.0f);
        output[index.x] = tile.val_frags[index.x & 3u][index.y & 1u];
    }
    """

    crossgl = convert_without_preprocessing(source)
    normalized = normalize(crossgl)

    assert "vec2[4] val_frags;" in normalized
    assert (
        "CrossGLMetalVectorIndex_vec2_get("
        "tile.val_frags[index.x & 3u], index.y & 1u)" in normalized
    )

    shader = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(shader)
    glsl = GLSLCodeGen().generate(shader)
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()
    assert "frag_type" not in hlsl
    assert "frag_type" not in glsl

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "struct-scoped-alias-member-array.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(hlsl_path),
                "-Fo",
                str(tmp_path / "struct-scoped-alias-member-array.dxil"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_opengl_compute_validates_if_available(
        glsl,
        tmp_path,
        "struct-scoped-alias-member-array",
    )


@pytest.mark.parametrize(
    ("second_alias", "reason"),
    [
        (
            "",
            "a visible declaration does not define exactly one concrete alias",
        ),
        (
            "using frag_type = metal::vec<float, 4>;",
            "visible declarations define conflicting concrete alias targets",
        ),
    ],
)
def test_codegen_rejects_non_unique_struct_member_alias_owner(second_alias, reason):
    source = f"""
    struct Tile {{
        using frag_type = metal::vec<float, 2>;
        frag_type val_frags[4];
    }};

    struct Tile {{
        {second_alias}
        frag_type val_frags[4];
    }};
    """

    with pytest.raises(MetalStructAliasResolutionError) as exc_info:
        convert_without_preprocessing(source, file_path="struct-member-alias.metal")

    diagnostic = exc_info.value
    assert diagnostic.owner == "Tile"
    assert diagnostic.alias_name == "frag_type"
    assert diagnostic.reason == reason
    assert diagnostic.source_location["file"] == "struct-member-alias.metal"


def test_codegen_does_not_capture_qualified_static_call_as_struct_alias():
    source = """
    struct Fragment {
        using frag_type = metal::vec<float, 2>;
    };

    void load_fragment(thread Fragment::frag_type& value) {
        Fragment:: load(value);
    }
    """

    crossgl = convert_without_preprocessing(source)

    assert "load(value);" in crossgl


def test_codegen_resource_vector_components_preserve_buffer_indexing(tmp_path):
    source = """
    kernel void resource_components(
        device uint4* values [[buffer(0)]],
        device uint* out_values [[buffer(1)]],
        uint2 index [[thread_position_in_grid]]) {
        auto selected = values[index.x][index.y];
        values[index.x][index.y] = selected + 1u;
        values[index.x][0] += 2u;
        out_values[0] = selected;
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)
    helper = "CrossGLMetalVectorIndex_uvec4"

    assert (
        f"uint selected = {helper}_get(buffer_load(values, index.x), index.y);"
        in normalized
    )
    assert (
        f"{helper}_set_resource(values, index.x, index.y, selected + 1u);" in normalized
    )
    assert f"{helper}_add_assign_resource(values, index.x, 0, 2u);" in normalized
    assert (
        f"uint {helper}_set_resource(device uvec4* data, "
        "uint element_index, uint lane, uint selected)" in normalized
    )
    assert "buffer_load(data, element_index)" in normalized
    assert "buffer_store(data, element_index, value)" in normalized
    assert "buffer_load(buffer_load(values" not in normalized

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert f"{helper}_set_resource_glsl_data_values_uvec4" in glsl
    assert "values[(data_offset + int(element_index))] = value;" in glsl
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-resource-vector-components"
    )


@pytest.mark.parametrize(
    ("vector_type", "component_type", "mapped_vector", "mapped_component"),
    [
        ("simd_short4", "short", "i16vec4", "int16"),
        ("vector_uchar4", "uchar", "u8vec4", "uint8"),
        ("packed_bfloat4", "bfloat", "bfloat16vec4", "bfloat16"),
    ],
)
def test_codegen_named_vector_aliases_use_canonical_component_helpers(
    vector_type, component_type, mapped_vector, mapped_component
):
    source = f"""
    {component_type} extract_alias_component({vector_type} value, uint lane) {{
        return value[lane];
    }}
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)

    assert (
        f"{mapped_component} CrossGLMetalVectorIndex_{mapped_vector}_get("
        f"{mapped_vector} value, uint lane)" in normalized
    )
    assert f"{mapped_component} extract_alias_component({mapped_vector} value" in (
        normalized
    )
    assert vector_type not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_narrow_and_bool_component_updates_restore_lane_types(tmp_path):
    source = """
    kernel void narrow_updates(
        device int* out_values [[buffer(0)]],
        uint lane [[thread_position_in_grid]]) {
        char4 signed_values = char4(127);
        ushort4 unsigned_values = ushort4(65535u);
        bool4 flags = bool4(true);
        int signed_divisor = 300;
        uint unsigned_divisor = 70000u;
        int flag_mask = 2;
        auto signed_result = (signed_values[lane] /= signed_divisor);
        auto unsigned_result = (unsigned_values[lane] %= unsigned_divisor);
        auto flag_result = (flags[lane] &= flag_mask);
        auto signed_before = signed_values[lane]++;
        auto unsigned_after = ++unsigned_values[lane];
        out_values[0] = int(signed_result) + int(unsigned_result)
            + int(flag_result) + int(signed_before) + int(unsigned_after);
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)

    assert (
        "int8 CrossGLMetalVectorIndex_i8vec4_div_assign("
        "inout i8vec4 value, uint lane, int right)" in normalized
    )
    assert (
        "int8 original = value.x; int computed = int(original) / right; "
        "int8 updated = computed; value.x = updated; return updated;" in normalized
    )
    assert (
        "uint16 CrossGLMetalVectorIndex_u16vec4_mod_assign("
        "inout u16vec4 value, uint lane, uint right)" in normalized
    )
    assert (
        "bool CrossGLMetalVectorIndex_bvec4_bit_and_assign("
        "inout bvec4 value, uint lane, int right)" in normalized
    )
    assert (
        "bool original = value.x; int computed = int(original) & right; "
        "bool updated = computed; value.x = updated; return updated;" in normalized
    )
    assert "int8 updated = computed; value.x = updated; return original;" in normalized

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "int updated = bitfieldExtract(computed, 0, 8);" in glsl
    assert "uint updated = bitfieldExtract(computed, 0, 16);" in glsl
    assert "bool updated = (computed != 0);" in glsl
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-narrow-vector-component-updates"
    )


def test_codegen_component_updates_accept_enum_and_ptrdiff_contracts():
    source = """
    enum ResourceIndex : uint {
        First = 0,
        Delta = 2,
    };

    void update_local(thread uint4& values, uint lane, ptrdiff_t delta) {
        enum LocalDelta { LocalStep = 1 };
        values[lane] += LocalStep;
        values[lane] += delta;
    }

    kernel void update_resource(
        device uint4* values [[buffer(0)]],
        uint lane [[thread_position_in_grid]]) {
        values[First][lane] += Delta;
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)
    helper = "CrossGLMetalVectorIndex_uvec4_add_assign"

    assert f"uint {helper}(inout uvec4 value, uint lane, int64 right)" in normalized
    assert f"uint {helper}(inout uvec4 value, uint lane, uint right)" in normalized
    assert f"{helper}(values, lane, LocalStep);" in normalized
    assert f"{helper}(values, lane, delta);" in normalized
    assert f"{helper}_resource(values, First, lane, Delta);" in normalized
    assert (
        f"uint {helper}_resource(device uvec4* data, "
        "uint element_index, uint lane, uint right)" in normalized
    )

    assert parse_crossgl(crossgl) is not None


def test_codegen_resource_component_updates_evaluate_operands_once(tmp_path):
    source = """
    uint next_index(thread uint& value) {
        return value++;
    }

    uint next_lane(thread uint& value) {
        return value++;
    }

    uint next_value(thread uint& value) {
        return value++;
    }

    kernel void resource_update_results(
        device uint4* values [[buffer(0)]],
        device uint* out_values [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        uint element = gid;
        uint lane = 0u;
        uint payload = 1u;
        auto assigned = (
            values[next_index(element)][next_lane(lane)] = next_value(payload));
        auto compounded = (
            values[next_index(element)][next_lane(lane)] += next_value(payload));
        auto prefixed = ++values[next_index(element)][next_lane(lane)];
        auto postfixed = values[next_index(element)][next_lane(lane)]++;
        out_values[0] = assigned + compounded + prefixed + postfixed;
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)
    helper = "CrossGLMetalVectorIndex_uvec4"

    assert (
        f"{helper}_set_resource(values, next_index(element), "
        "next_lane(lane), next_value(payload))" in normalized
    )
    assert (
        f"{helper}_add_assign_resource(values, next_index(element), "
        "next_lane(lane), next_value(payload))" in normalized
    )
    assert (
        f"{helper}_pre_increment_resource(values, next_index(element), "
        "next_lane(lane))" in normalized
    )
    assert (
        f"{helper}_post_increment_resource(values, next_index(element), "
        "next_lane(lane))" in normalized
    )
    assert normalized.count("next_index(element)") == 4
    assert normalized.count("next_lane(lane)") == 4
    assert normalized.count("next_value(payload)") == 2

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-resource-vector-update-results"
    )


def test_codegen_nested_resource_vector_paths_load_and_store_root_once(tmp_path):
    source = """
    struct Payload {
        uint4 lanes[2];
        float3x3 matrix;
    };

    uint next_element(thread uint& value) {
        return value++;
    }

    uint next_path(thread uint& value) {
        return value++;
    }

    uint next_lane(thread uint& value) {
        return value++;
    }

    kernel void nested_resource_components(
        device Payload* values [[buffer(0)]],
        device uint* out_values [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
        uint element = gid;
        uint path = 0u;
        uint lane = 0u;
        auto selected = values[next_element(element)]
            .lanes[next_path(path)][next_lane(lane)];
        values[next_element(element)].lanes[next_path(path)][next_lane(lane)]
            += 2u;
        auto matrix_value = values[next_element(element)]
            .matrix[next_path(path)][next_lane(lane)];
        values[next_element(element)].matrix[next_path(path)][next_lane(lane)]
            = matrix_value + 1.0;
        out_values[0] = selected + uint(matrix_value);
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)

    assert (
        "CrossGLMetalVectorIndex_uvec4_get(buffer_load(values, "
        "next_element(element)).lanes[next_path(path)], next_lane(lane))" in normalized
    )
    assert (
        "CrossGLMetalVectorIndex_uvec4_add_assign_resource_lanes_index("
        "values, next_element(element), next_path(path), next_lane(lane), 2u)"
        in normalized
    )
    assert (
        "CrossGLMetalVectorIndex_vec3_get(buffer_load(values, "
        "next_element(element)).matrix[next_path(path)], next_lane(lane))" in normalized
    )
    assert (
        "CrossGLMetalVectorIndex_vec3_set_resource_matrix_index("
        "values, next_element(element), next_path(path), next_lane(lane), "
        "matrix_value + 1.0)" in normalized
    )
    assert normalized.count("next_element(element)") == 4
    assert normalized.count("next_path(path)") == 4
    assert normalized.count("next_lane(lane)") == 4
    assert "buffer_load(values[next_element" not in normalized
    assert "buffer_load(values, next_element(element)).lanes" in normalized
    assert "buffer_load(values, next_element(element)).matrix" in normalized

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert_opengl_compute_validates_if_available(
        glsl, tmp_path, "metal-nested-resource-vector-components"
    )


def test_codegen_reports_unresolved_indexed_component_type():
    source = """
    kernel void unresolved_component(
        device uint* out_values [[buffer(0)]],
        uint lane [[thread_position_in_grid]]) {
        auto selected = external_vector()[lane];
        out_values[0] = selected;
    }
    """

    with pytest.raises(MetalIndexedComponentTypeResolutionError) as exc_info:
        convert(source)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-indexed-component-type-unresolved"
    )
    assert diagnostic.missing_capabilities == (
        "metal.indexed-component-type-inference",
    )
    assert diagnostic.base_type is None
    assert diagnostic.access_kind == "subscript"
    assert diagnostic.index_expression == "lane"
    assert diagnostic.reason == (
        "the indexed base expression type could not be inferred"
    )


@pytest.mark.parametrize(
    ("return_type", "statement"),
    [
        ("void", "uint selected = external_vector()[lane];"),
        ("uint", "return external_vector()[lane];"),
    ],
)
def test_codegen_reports_unresolved_indexed_rvalue_with_expected_type(
    return_type, statement
):
    source = f"""
    {return_type} unresolved_rvalue(uint lane) {{
        {statement}
    }}
    """

    with pytest.raises(MetalIndexedComponentTypeResolutionError) as exc_info:
        convert(source)

    diagnostic = exc_info.value
    assert diagnostic.base_type is None
    assert diagnostic.access_kind == "subscript"
    assert diagnostic.index_expression == "lane"
    assert diagnostic.reason == (
        "the indexed base expression type could not be inferred"
    )


@pytest.mark.parametrize(
    "statement",
    [
        "external_vector()[lane] = 1u;",
        "external_vector()[lane] += 1u;",
        "++external_vector()[lane];",
        "external_vector()[lane]++;",
    ],
)
def test_codegen_reports_unresolved_indexed_update_target_type(statement):
    source = f"""
    void unresolved_update(uint lane) {{
        {statement}
    }}
    """

    with pytest.raises(MetalIndexedComponentTypeResolutionError) as exc_info:
        convert(source)

    diagnostic = exc_info.value
    assert diagnostic.base_type is None
    assert diagnostic.access_kind == "subscript"
    assert diagnostic.index_expression == "lane"
    assert diagnostic.reason == (
        "the indexed base expression type could not be inferred"
    )


def test_codegen_reports_user_aggregate_subscript_result_type():
    source = """
    struct IndexedTable {};

    void unresolved_aggregate(IndexedTable table, uint lane) {
        auto selected = table[lane];
    }
    """

    with pytest.raises(MetalIndexedComponentTypeResolutionError) as exc_info:
        convert(source)

    diagnostic = exc_info.value
    assert diagnostic.base_type == "IndexedTable"
    assert diagnostic.access_kind == "aggregate-subscript"
    assert diagnostic.index_expression == "lane"
    assert diagnostic.reason == (
        "the user-defined aggregate subscript result type cannot be inferred"
    )


def test_codegen_resolves_equivalent_duplicate_concrete_struct_alias_for_indexing():
    source = """
    namespace mlx {
    namespace steel {
    struct Fragment_float {
        typedef metal::vec<float, 2> frag_type;
    };
    }
    }

    namespace mlx {
    namespace steel {
    struct Fragment_float {
        typedef metal::vec<float, 2> frag_type;
    };
    }
    }

    using namespace mlx::steel;

    float read_lane(Fragment_float::frag_type value, ushort lane) {
        auto selected = value[lane];
        return selected;
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)
    assert "float read_lane(vec2 value, uint16 lane)" in normalized
    assert (
        "float selected = CrossGLMetalVectorIndex_vec2_get(value, lane);" in normalized
    )
    parse_crossgl(crossgl)


def test_codegen_keeps_duplicate_struct_aliases_isolated_by_qualified_namespace():
    source = """
    namespace Left {
    struct Fragment {
        typedef metal::vec<float, 2> frag_type;
    };
    struct Fragment {
        typedef metal::vec<float, 2> frag_type;
    };
    }

    namespace Right {
    struct Fragment {
        typedef metal::vec<float, 4> frag_type;
    };
    struct Fragment {
        typedef metal::vec<float, 4> frag_type;
    };
    }

    float read_left(Left::Fragment::frag_type value, ushort lane) {
        return value[lane];
    }

    float read_right(Right::Fragment::frag_type value, ushort lane) {
        return value[lane];
    }
    """

    crossgl = convert(source)
    normalized = normalize(crossgl)
    assert "float read_left(vec2 value, uint16 lane)" in normalized
    assert "float read_right(vec4 value, uint16 lane)" in normalized
    assert "CrossGLMetalVectorIndex_vec2_get(value, lane)" in normalized
    assert "CrossGLMetalVectorIndex_vec4_get(value, lane)" in normalized
    parse_crossgl(crossgl)


def test_equivalent_duplicate_struct_alias_reaches_project_targets(tmp_path):
    source = """
    struct Fragment_float {
        typedef metal::vec<float, 2> frag_type;
    };

    struct Fragment_float {
        typedef metal::vec<float, 2> frag_type;
    };

    kernel void alias_kernel(
        device float* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        Fragment_float::frag_type values =
            Fragment_float::frag_type(float(gid));
        out[gid] = values[gid & 1u];
    }
    """
    repo = tmp_path / "equivalent-struct-alias"
    repo.mkdir()
    source_path = repo / "alias.metal"
    source_path.write_text(source, encoding="utf-8")

    report = translate_project(
        repo,
        targets=["directx", "opengl"],
        output_dir="out",
        format_output=False,
    )
    payload = report.to_json()
    assert payload["summary"]["translatedCount"] == 2, payload
    assert payload["summary"]["failedCount"] == 0, payload
    artifacts = {artifact["target"]: artifact for artifact in payload["artifacts"]}
    outputs = {
        target: (repo / artifact["path"]).read_text(encoding="utf-8")
        for target, artifact in artifacts.items()
    }

    HLSLParser(HLSLLexer(outputs["directx"]).tokenize()).parse()
    dxc = shutil.which("dxc")
    if dxc is not None:
        source_output = tmp_path / "equivalent-struct-alias.hlsl"
        binary_output = tmp_path / "equivalent-struct-alias.dxil"
        source_output.write_text(outputs["directx"], encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                "CSMain",
                str(source_output),
                "-Fo",
                str(binary_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert binary_output.stat().st_size > 0

    assert_opengl_compute_validates_if_available(
        outputs["opengl"],
        tmp_path,
        "equivalent-struct-alias",
    )


def test_codegen_resolves_equivalent_chained_struct_alias_with_static_width():
    source = """
    namespace mlx {
    namespace steel {
    struct Fragment_float {
        static constexpr int kElemsPerFrag = 1 + 1;
        using storage_type = metal::vec<float, kElemsPerFrag>;
        using frag_type = storage_type;
    };
    }
    }

    namespace mlx {
    namespace steel {
    struct Fragment_float {
        static constexpr int kElemsPerFrag = 2;
        using storage_type = metal::vec<float, kElemsPerFrag>;
        using frag_type = storage_type;
    };
    }
    }

    using namespace mlx::steel;

    float read_lane(Fragment_float::frag_type value, ushort lane) {
        return value[lane];
    }
    """

    crossgl = convert(source)
    assert "float read_lane(vec2 value, uint16 lane)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_resolves_equivalent_struct_array_alias_with_static_extent():
    source = """
    struct Fragment_float {
        static constexpr int kWidth = 1 + 1;
        typedef float lane_type[kWidth];
    };

    struct Fragment_float {
        static constexpr int kWidth = 2;
        typedef float lane_type[kWidth];
    };

    float read_lane(Fragment_float::lane_type value, ushort lane) {
        return value[lane];
    }
    """

    crossgl = convert_without_preprocessing(source)
    assert "float read_lane(float[2] value, uint16 lane)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_rejects_constexpr_only_duplicate_without_materialization_provenance():
    source = """
    struct BaseMMAFrag_float_8_8 {
        static constexpr int kFragRows = 8;
        static constexpr int kFragCols = 8;
        static constexpr int kElemsPerFrag = 2;
        static constexpr int kElemRows = 1;
        static constexpr int kElemCols = 2;
    };

    struct BaseMMAFrag_float_8_8 {
        static constexpr int kElemsPerFrag = 1 + 1;
        typedef metal::vec<float, kElemsPerFrag> frag_type;
    };

    struct BaseMMAFrag_float_8_8 {
        static constexpr int kElemsPerFrag = 2;
        typedef metal::vec<float, kElemsPerFrag> frag_type;
    };

    float read_lane(BaseMMAFrag_float_8_8::frag_type value, ushort lane) {
        return value[lane];
    }
    """

    with pytest.raises(MetalStructAliasResolutionError) as exc_info:
        convert_without_preprocessing(source, file_path="unproven-owner-fragment.metal")

    diagnostic = exc_info.value
    assert diagnostic.owner == "BaseMMAFrag_float_8_8"
    assert diagnostic.alias_name == "frag_type"
    assert diagnostic.reason == (
        "a visible declaration does not define exactly one concrete alias"
    )
    assert diagnostic.source_location["file"] == "unproven-owner-fragment.metal"


def test_codegen_rejects_missing_struct_alias_on_runtime_owner_fragment():
    source = """
    struct Fragment_float {
        static constexpr int width = 2;
        float runtime_value;
    };

    struct Fragment_float {
        typedef metal::vec<float, 2> frag_type;
    };

    float read_lane(Fragment_float::frag_type value, ushort lane) {
        return value[lane];
    }
    """

    with pytest.raises(MetalStructAliasResolutionError) as exc_info:
        convert_without_preprocessing(source, file_path="runtime-owner-fragment.metal")

    diagnostic = exc_info.value
    assert diagnostic.owner == "Fragment_float"
    assert diagnostic.alias_name == "frag_type"
    assert diagnostic.reason == (
        "a visible declaration does not define exactly one concrete alias"
    )
    assert diagnostic.source_location["file"] == "runtime-owner-fragment.metal"


def test_codegen_rejects_conflicting_duplicate_concrete_struct_alias():
    source = """
    namespace mlx {
    namespace steel {
    struct Fragment_float {
        typedef metal::vec<float, 2> frag_type;
    };
    }
    }

    namespace mlx {
    namespace steel {
    struct Fragment_float {
        typedef metal::vec<float, 4> frag_type;
    };
    }
    }

    using namespace mlx::steel;

    float read_lane(Fragment_float::frag_type value, ushort lane) {
        return value[lane];
    }
    """

    with pytest.raises(MetalStructAliasResolutionError) as exc_info:
        convert_without_preprocessing(source, file_path="duplicate-alias.metal")

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-struct-alias-unresolved"
    )
    assert diagnostic.owner == "mlx::steel::Fragment_float"
    assert diagnostic.alias_name == "frag_type"
    assert diagnostic.requested_signature == ("mlx::steel::Fragment_float::frag_type")
    assert diagnostic.reason == (
        "visible declarations define conflicting concrete alias targets"
    )
    assert diagnostic.candidate_identities == (
        "mlx::steel::Fragment_float::frag_type=metal::vec<float,2>",
        "mlx::steel::Fragment_float::frag_type=metal::vec<float,4>",
    )
    assert str(diagnostic).endswith(
        "candidates are "
        "mlx::steel::Fragment_float::frag_type=metal::vec<float,2>, "
        "mlx::steel::Fragment_float::frag_type=metal::vec<float,4>"
    )
    assert len(diagnostic.candidate_locations) == 2
    assert diagnostic.source_location["file"] == "duplicate-alias.metal"


@pytest.mark.parametrize(
    ("first_alias", "second_alias"),
    [
        (
            "typedef device const float* lane_type;",
            "typedef threadgroup const float* lane_type;",
        ),
        ("typedef float* lane_type;", "typedef float** lane_type;"),
        ("typedef float lane_type[2];", "typedef float lane_type[4];"),
    ],
)
def test_codegen_rejects_duplicate_struct_alias_qualifier_or_shape_conflicts(
    first_alias, second_alias
):
    source = f"""
    struct Fragment_float {{
        {first_alias}
    }};

    struct Fragment_float {{
        {second_alias}
    }};

    void consume(Fragment_float::lane_type value) {{}}
    """

    with pytest.raises(MetalStructAliasResolutionError) as exc_info:
        convert_without_preprocessing(source, file_path="alias-shape.metal")

    diagnostic = exc_info.value
    assert diagnostic.owner == "Fragment_float"
    assert diagnostic.alias_name == "lane_type"
    assert diagnostic.reason == (
        "visible declarations define conflicting concrete alias targets"
    )
    assert len(diagnostic.candidate_identities) == 2
    assert len(diagnostic.candidate_locations) == 2


def test_mlx_materialized_collapsed_member_overloads_reach_project_targets(tmp_path):
    # Reduced from MLX 4367c73b60541ddd5a266ce4644fd93d20223b6e,
    # mlx/backend/metal/kernels/binary_ops.h and binary_two.metal.
    source = """
    typedef bfloat bfloat16_t;

    struct FloorDivide {
      template <typename T>
      T operator()(T x, T y) {
        return x / y;
      }

      template <>
      half operator()(half x, half y) {
        return trunc(x / y);
      }

      template <>
      bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {
        return x - y;
      }
    };

    kernel void exercise_collapsed_member_overloads(
        device half* values [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      FloorDivide operation;
      half half_value = half(index + 2u);
      bfloat16_t bfloat_value = bfloat16_t(index + 2u);
      values[index * 4u] = operation(half_value, half(2.0h));
      values[index * 4u + 1u] = half(operation(
          bfloat_value, bfloat16_t(2.0h)));
      values[index * 4u + 2u] = FloorDivide{}(half_value, half(2.0h));
      values[index * 4u + 3u] = half(FloorDivide{}(
          bfloat_value, bfloat16_t(2.0h)));
    }
    """
    repo = tmp_path / "mlx-materialized-overloads"
    repo.mkdir()
    source_path = repo / "binary_two.metal"
    source_path.write_text(source, encoding="utf-8")

    intermediate = crosstl.translate(
        str(source_path), backend="cgl", format_output=False
    )
    normalized_intermediate = normalize(intermediate)
    half_helper = "FloorDivide__operator_call__metal_overload_1"
    bfloat_helper = "FloorDivide__operator_call__metal_overload_2"
    half_wrapper = f"{half_helper}__temporary"
    bfloat_wrapper = f"{bfloat_helper}__temporary"
    assert (
        f"float16 {half_helper}(inout thread FloorDivide self, "
        "float16 x, float16 y) { return trunc(x / y); }" in normalized_intermediate
    )
    assert (
        f"bfloat16_t {bfloat_helper}(inout thread FloorDivide self, "
        "bfloat16_t x, bfloat16_t y) { return x - y; }" in normalized_intermediate
    )
    assert f"return {half_helper}(self, x, y);" in normalized_intermediate
    assert f"return {bfloat_helper}(self, x, y);" in normalized_intermediate
    for helper in (half_helper, bfloat_helper, half_wrapper, bfloat_wrapper):
        assert f"{helper}(" in intermediate
    assert "FloorDivide__operator_call(inout thread FloorDivide" not in intermediate

    direct_outputs = {
        target: crosstl.translate(str(source_path), backend=target, format_output=False)
        for target in ("directx", "opengl")
    }
    report = translate_project(
        repo,
        targets=["directx", "opengl"],
        output_dir="out",
    )
    payload = report.to_json()
    assert payload["summary"]["translatedCount"] == 2, payload
    assert payload["summary"]["failedCount"] == 0, payload
    artifacts = {artifact["target"]: artifact for artifact in payload["artifacts"]}

    for target, direct in direct_outputs.items():
        project = (repo / artifacts[target]["path"]).read_text(encoding="utf-8")
        source_helpers = {half_helper, bfloat_helper}
        source_wrappers = {half_wrapper, bfloat_wrapper}
        if target == "directx":
            expected_helpers = source_helpers
            expected_wrappers = source_wrappers
            helper_types = {half_helper: "float16_t", bfloat_helper: "uint"}
            for helper, scalar_type in helper_types.items():
                definition_pattern = (
                    rf"{scalar_type} {helper}"
                    rf"\(inout FloorDivide self, {scalar_type} x, "
                    rf"{scalar_type} y\) \{{"
                )
                assert re.search(definition_pattern, direct) is not None
                assert re.search(definition_pattern, project) is not None
            assert " FloorDivide__operator_call(inout FloorDivide self," not in direct
        else:
            expected_helpers = {name.replace("__", "_") for name in source_helpers}
            expected_wrappers = {name.replace("__", "_") for name in source_wrappers}
            unspecialized_helper = "FloorDivide_operator_call"
            helper_pattern = "|".join(
                re.escape(name) for name in sorted(expected_helpers)
            )
            definition_pattern = (
                rf"float ({helper_pattern})"
                r"\(inout FloorDivide self, float x, float y\) \{"
            )
            assert set(re.findall(definition_pattern, direct)) == expected_helpers
            assert set(re.findall(definition_pattern, project)) == expected_helpers
            assert f"float {unspecialized_helper}(" not in direct
        for helper in (*expected_helpers, *expected_wrappers):
            assert f"{helper}(" in direct
            assert f"{helper}(" in project

    hlsl = direct_outputs["directx"]
    HLSLParser(HLSLLexer(hlsl).tokenize()).parse()
    dxc = shutil.which("dxc")
    if dxc is not None:
        source_output = tmp_path / "materialized-collapsed-overloads.hlsl"
        binary_output = tmp_path / "materialized-collapsed-overloads.dxil"
        source_output.write_text(hlsl, encoding="utf-8")
        result = subprocess.run(
            [
                dxc,
                "-T",
                "cs_6_2",
                "-enable-16bit-types",
                "-E",
                "CSMain",
                str(source_output),
                "-Fo",
                str(binary_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert binary_output.stat().st_size > 0

    assert_opengl_compute_validates_if_available(
        direct_outputs["opengl"],
        tmp_path,
        "materialized-collapsed-overloads",
    )


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

    assert "ReadWriter_float writer = crosstl_ctor_ReadWriter_float_1(3);" in result
    assert "crosstl_ctor_value.value = int(value_);" in result
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
    assert "return __crossgl_bfloat16_from_float(float(asfloat(2139095040u)));" in hlsl
    assert "return half(asfloat(2139095040u));" not in hlsl
    assert "// CrossGL exact bfloat16 lowering:" in hlsl
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


def test_codegen_resolves_equivalent_duplicate_concrete_static_constant_owner():
    crossgl = convert("""
        namespace mlx {
        namespace steel {
        struct BaseMMAFrag_float_8_8 {
            static constexpr int kFragRows = 8;
            static constexpr int kFragCols = 8;
        };
        }
        }

        namespace mlx {
        namespace steel {
        struct BaseMMAFrag_float_8_8 {
            static constexpr int kFragCols = 4 + 4;
            static constexpr int kFragRows = kFragCols;
            static constexpr int kElemsPerFrag = 2;
        };
        }
        }

        using namespace mlx::steel;

        int fragment_rows() {
            return BaseMMAFrag_float_8_8::kFragRows;
        }
        """)

    assert "return 8;" in crossgl
    assert "BaseMMAFrag_float_8_8_u3a_u3akFragRows" not in crossgl


def test_codegen_rejects_conflicting_duplicate_concrete_static_constant_owner():
    code = """
    namespace mlx {
    namespace steel {
    struct BaseMMAFrag_float_8_8 {
        static constexpr int kFragRows = 8;
    };
    }
    }

    namespace mlx {
    namespace steel {
    struct BaseMMAFrag_float_8_8 {
        static constexpr int kFragRows = 16;
    };
    }
    }

    using namespace mlx::steel;

    int fragment_rows() {
        return BaseMMAFrag_float_8_8::kFragRows;
    }
    """

    with pytest.raises(MetalStaticConstantResolutionError) as error:
        convert(code)

    assert error.value.owner == "BaseMMAFrag_float_8_8"
    assert error.value.member == "kFragRows"
    assert "conflicting compile-time values" in error.value.reason
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-static-constant-unresolved"
    )


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

    with pytest.raises(MetalStaticConstantResolutionError) as error:
        convert(code)

    assert error.value.owner == "Limits"
    assert error.value.member == "max"
    assert "multiple visible struct declarations" in error.value.reason
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-static-constant-unresolved"
    )


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


def cooperative_matrix_fragment_transfer_source(rows=8, columns=8):
    return f"""
    using fragment_type = metal::vec<float, 2>;
    using matrix_type = metal::simdgroup_matrix<float, {rows}, {columns}>;

    kernel void transfer_fragment(
        device fragment_type* fragments [[buffer(0)]]) {{
      matrix_type matrix;
      fragments[0] = reinterpret_cast<const thread fragment_type&>(
          matrix.thread_elements());
    }}
    """


def generate_cooperative_matrix_fragment_transfer(converter, rows=8, columns=8):
    source = cooperative_matrix_fragment_transfer_source(rows, columns)
    return converter.generate(parse_code(tokenize_code(source)))


def test_metal_cooperative_matrix_fragment_mapping_is_not_inferred():
    crossgl = generate_cooperative_matrix_fragment_transfer(MetalToCrossGLConverter())

    assert (
        "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
        "metal_thread_elements,32,2,metal_thread_elements_reference_view>"
    ) in crossgl
    assert "tile_4x4_row_pair" not in crossgl


def test_metal_cooperative_matrix_fragment_mapping_emits_configured_metadata():
    converter = MetalToCrossGLConverter(
        cooperative_matrix_fragment_mapping="tile_4x4_row_pair",
        cooperative_matrix_fragment_mapping_provenance="source_coordinate_helper",
    )

    crossgl = generate_cooperative_matrix_fragment_transfer(converter)

    assert (
        "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
        "metal_thread_elements,32,2,metal_thread_elements_reference_view,"
        "tile_4x4_row_pair,source_coordinate_helper>"
    ) in crossgl


def cooperative_matrix_fragment_contract_propagation_source():
    return """
    float preserve_fragment_matrix(float value) {
      return value;
    }

    metal::simdgroup_matrix<float, 8, 8> preserve_fragment_matrix(
        thread metal::simdgroup_matrix<float, 8, 8>& matrix) {
      matrix.thread_elements()[0] = matrix.thread_elements()[1];
      return matrix;
    }

    kernel void propagate_fragment_contract(
        device float2* fragments [[buffer(0)]]) {
      metal::simdgroup_matrix<float, 8, 8> left;
      metal::simdgroup_matrix<float, 8, 8> right;
      metal::simdgroup_matrix<float, 8, 8> accumulator;
      reinterpret_cast<thread float2&>(left.thread_elements()) = fragments[0];
      fragments[1] = reinterpret_cast<const thread float2&>(
          left.thread_elements());
      right.thread_elements()[0] = left.thread_elements()[1];
      metal::simdgroup_matrix<float, 8, 8> prior =
          preserve_fragment_matrix(accumulator);
      simdgroup_multiply_accumulate(accumulator, left, right, prior);
    }
    """


def generate_cooperative_matrix_fragment_contract_propagation(converter):
    source = cooperative_matrix_fragment_contract_propagation_source()
    return converter.generate(parse_code(tokenize_code(source)))


def assert_cooperative_matrix_fragment_contract(crossgl, expected_type):
    emitted_types = re.findall(
        r"CooperativeMatrix<float,8,8,[^>]+>",
        crossgl,
    )

    assert emitted_types
    assert set(emitted_types) == {expected_type}
    assert (
        "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified>" not in crossgl
    )
    assert "cooperative_matrix_element(matrix, 0)" in crossgl
    assert "cooperative_matrix_element(matrix, 1)" in crossgl
    assert (
        "cooperative_matrix_multiply_accumulate(accumulator, left, right, prior);"
        in crossgl
    )


def test_metal_cooperative_matrix_fragment_contract_propagates_configured_mapping():
    converter = MetalToCrossGLConverter(
        cooperative_matrix_fragment_mapping="tile_4x4_row_pair",
        cooperative_matrix_fragment_mapping_provenance="source_coordinate_helper",
    )

    crossgl = generate_cooperative_matrix_fragment_contract_propagation(converter)

    assert_cooperative_matrix_fragment_contract(
        crossgl,
        "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
        "metal_thread_elements,32,2,metal_thread_elements_reference_view,"
        "tile_4x4_row_pair,source_coordinate_helper>",
    )


def test_metal_cooperative_matrix_fragment_contract_propagates_without_mapping():
    crossgl = generate_cooperative_matrix_fragment_contract_propagation(
        MetalToCrossGLConverter()
    )

    assert_cooperative_matrix_fragment_contract(
        crossgl,
        "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
        "metal_thread_elements,32,2,metal_thread_elements_reference_view>",
    )
    assert "tile_4x4_row_pair" not in crossgl
    assert "source_coordinate_helper" not in crossgl


@pytest.mark.parametrize(
    ("mapping", "provenance"),
    [
        ("tile_4x4_row_pair", None),
        (None, "source_coordinate_helper"),
    ],
)
def test_metal_cooperative_matrix_fragment_mapping_requires_provenance_pair(
    mapping, provenance
):
    with pytest.raises(ValueError, match="must be configured together"):
        MetalToCrossGLConverter(
            cooperative_matrix_fragment_mapping=mapping,
            cooperative_matrix_fragment_mapping_provenance=provenance,
        )


def test_metal_cooperative_matrix_fragment_mapping_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown cooperative-matrix fragment"):
        MetalToCrossGLConverter(
            cooperative_matrix_fragment_mapping="unknown_profile",
            cooperative_matrix_fragment_mapping_provenance="project_configuration",
        )


def test_metal_cooperative_matrix_fragment_mapping_rejects_contract_mismatch():
    converter = MetalToCrossGLConverter(
        cooperative_matrix_fragment_mapping="tile_4x4_row_pair",
        cooperative_matrix_fragment_mapping_provenance="source_coordinate_helper",
    )

    with pytest.raises(MetalCooperativeMatrixFragmentLoweringError) as exc_info:
        generate_cooperative_matrix_fragment_transfer(converter, rows=4, columns=4)

    assert "has no registered contract for 4x4" in exc_info.value.reason


def test_codegen_lowers_whole_cooperative_matrix_fragment_transfers_once(tmp_path):
    source = """
    using fragment_type = metal::vec<float, 2>;
    using matrix_type = metal::simdgroup_matrix<float, 8, 8>;

    uint matrix_index() {
      return 0u;
    }

    kernel void transfer_fragments(
        device fragment_type* fragments [[buffer(0)]]) {
      matrix_type matrices[2];
      reinterpret_cast<thread fragment_type&>(
          matrices[matrix_index()].thread_elements()) = fragments[0];
      fragments[1] = reinterpret_cast<const thread fragment_type&>(
          matrices[matrix_index()].thread_elements());
    }
    """

    crossgl = convert(source)

    write_helper = "_crosstl_metal_cooperative_matrix_fragment_write_float_8_8_2"
    read_helper = "_crosstl_metal_cooperative_matrix_fragment_read_float_8_8_2"
    assert ".thread_elements()" not in crossgl
    assert "(vec2&)" not in crossgl
    assert crossgl.count("cooperative_matrix_element(matrix, 0)") == 2
    assert crossgl.count("cooperative_matrix_element(matrix, 1)") == 2
    assert (
        f"{write_helper}(buffer_load(fragments, 0), matrices[matrix_index()]);"
        in crossgl
    )
    assert (
        f"buffer_store(fragments, 1, {read_helper}(matrices[matrix_index()]));"
        in crossgl
    )
    assert crossgl.count("matrices[matrix_index()]") == 2
    assert crossgl.count("buffer_load(fragments, 0)") == 1
    assert (
        "CooperativeMatrix<float,8,8,subgroup,unspecified,unspecified,"
        "metal_thread_elements,32,2,metal_thread_elements_reference_view>"
    ) in crossgl

    regenerated = MetalCodeGen().generate(parse_crossgl(crossgl))
    xcrun = shutil.which("xcrun")
    if xcrun is not None:
        metal_path = tmp_path / "cooperative-matrix-fragment-transfer.metal"
        air_path = tmp_path / "cooperative-matrix-fragment-transfer.air"
        metal_path.write_text(regenerated, encoding="utf-8")
        result = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(metal_path),
                "-o",
                str(air_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_codegen_whole_cooperative_matrix_fragment_helpers_round_trip(tmp_path):
    source = """
    using fragment_type = metal::vec<float, 2>;
    using matrix_type = metal::simdgroup_matrix<float, 8, 8>;

    void copy_fragment(
        const thread matrix_type& matrix,
        thread fragment_type& fragment) {
      fragment = reinterpret_cast<const thread fragment_type&>(
          matrix.thread_elements());
    }
    """

    crossgl = convert(source)
    shader = parse_crossgl(crossgl)
    regenerated = MetalCodeGen().generate(shader)
    matrix_types = [
        node for node in shader.walk() if isinstance(node, CooperativeMatrixType)
    ]

    assert ".thread_elements()" not in crossgl
    assert "reinterpret_cast" not in crossgl
    assert "cooperative_matrix_element(matrix, 0)" in crossgl
    assert "cooperative_matrix_element(matrix, 1)" in crossgl
    assert "matrix.thread_elements()[0]" in regenerated
    assert "matrix.thread_elements()[1]" in regenerated
    assert "reinterpret_cast" not in regenerated
    assert any(
        matrix_type.fragment_layout == "metal_thread_elements"
        and matrix_type.subgroup_size == 32
        and matrix_type.elements_per_lane == 2
        and matrix_type.fragment_provenance == "metal_thread_elements_reference_view"
        for matrix_type in matrix_types
    )

    xcrun = shutil.which("xcrun")
    if xcrun is not None:
        metal_path = tmp_path / "cooperative-matrix-fragment.metal"
        air_path = tmp_path / "cooperative-matrix-fragment.air"
        metal_path.write_text(regenerated, encoding="utf-8")
        result = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(metal_path),
                "-o",
                str(air_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("cast_type", "reason"),
    [
        (
            "float3&",
            "matrix shape 8x8 cannot be distributed into 3 elements per lane "
            "with an integral SIMD-group width",
        ),
        (
            "half2&",
            "fragment component type 'half' does not match matrix component "
            "type 'float'",
        ),
        ("float2", "the cast target is not an lvalue reference"),
    ],
)
def test_codegen_rejects_incompatible_whole_cooperative_matrix_fragment_writes(
    cast_type, reason
):
    source = f"""
    void invalid_fragment_write(
        thread metal::simdgroup_matrix<float, 8, 8>& matrix,
        thread float2& fragment) {{
      reinterpret_cast<thread {cast_type}>(matrix.thread_elements()) = fragment;
    }}
    """

    with pytest.raises(MetalCooperativeMatrixFragmentLoweringError) as exc_info:
        convert_without_preprocessing(source, file_path="fragment_write.metal")

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-cooperative-matrix-fragment-unsupported"
    )
    assert diagnostic.missing_capabilities == (
        "metal.cooperative-matrix-fragment-lowering",
    )
    assert diagnostic.matrix_type == "metal::simdgroup_matrix<float,8,8>"
    assert diagnostic.fragment_type == cast_type
    assert diagnostic.direction == "write"
    assert reason in diagnostic.reason
    assert diagnostic.qualifiers == ("thread",)
    assert diagnostic.source_location["file"] == "fragment_write.metal"
    assert diagnostic.source_location["line"] == 2


@pytest.mark.parametrize(
    ("template_parameter", "matrix_type", "fragment_type", "reason"),
    [
        (
            "int Rows",
            "metal::simdgroup_matrix<float, Rows, 8>",
            "float2",
            "row or column count remains dependent",
        ),
        (
            "int Width",
            "metal::simdgroup_matrix<float, 8, 8>",
            "metal::vec<float, Width>",
            "fragment width remains dependent",
        ),
    ],
)
def test_codegen_rejects_dependent_whole_cooperative_matrix_fragment_shapes(
    template_parameter, matrix_type, fragment_type, reason
):
    source = f"""
    template <{template_parameter}>
    void invalid_fragment_read(
        thread {matrix_type}& matrix,
        thread {fragment_type}& fragment) {{
      fragment = reinterpret_cast<thread {fragment_type}&>(
          matrix.thread_elements());
    }}
    """

    with pytest.raises(MetalCooperativeMatrixFragmentLoweringError) as exc_info:
        convert_without_preprocessing(source, file_path="dependent_fragment.metal")

    diagnostic = exc_info.value
    assert diagnostic.direction == "read"
    assert reason in diagnostic.reason
    assert diagnostic.source_location["file"] == "dependent_fragment.metal"
    assert diagnostic.source_location["line"] == 2


def test_codegen_rejects_conflicting_whole_cooperative_matrix_fragment_contracts():
    source = """
    void conflicting_fragment_contracts(
        thread metal::simdgroup_matrix<float, 8, 8>& matrix,
        thread float2& narrow_fragment,
        thread float4& wide_fragment) {
      narrow_fragment = reinterpret_cast<thread float2&>(
          matrix.thread_elements());
      wide_fragment = reinterpret_cast<thread float4&>(
          matrix.thread_elements());
    }
    """

    with pytest.raises(MetalCooperativeMatrixFragmentLoweringError) as exc_info:
        convert_without_preprocessing(source, file_path="conflicting_fragment.metal")

    diagnostic = exc_info.value
    assert diagnostic.direction == "read"
    assert (
        "conflicting whole-fragment contracts: subgroup_size=32, "
        "elements_per_lane=2 and subgroup_size=16, elements_per_lane=4"
        in diagnostic.reason
    )
    assert diagnostic.source_location["file"] == "conflicting_fragment.metal"


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


def test_codegen_folds_constexpr_helpers_in_materialized_function_bodies():
    code = """
    template <int Bits, int Word = 32>
    constexpr int get_pack_factor() {
        constexpr int factor = Word / Bits;
        return factor;
    }

    template <int Bits = 4, int Word = 32>
    int load_pack() {
        constexpr int explicit_factor = get_pack_factor<Bits, Word>();
        constexpr int default_factor = get_pack_factor<Bits>();
        return explicit_factor + default_factor +
            get_pack_factor<Bits, Word>();
    }

    int run_packs() {
        return load_pack<4>() + load_pack<8>();
    }
    """
    ast = MetalParser(MetalLexer(code, preprocess=False).tokenize()).parse()
    converter = MetalToCrossGLConverter()

    crossgl = converter.generate(ast)
    compact = normalize(crossgl)

    assert "int load_pack_4_32()" in crossgl
    assert "int explicit_factor = 8;" in crossgl
    assert "int default_factor = 8;" in crossgl
    assert "return explicit_factor + default_factor + 8;" in crossgl
    assert "int load_pack_8_32()" in crossgl
    assert "int explicit_factor = 4;" in crossgl
    assert "int default_factor = 4;" in crossgl
    assert "return load_pack_4_32() + load_pack_8_32();" in crossgl
    assert len(converter.constexpr_helper_values) == 2
    assert "get_pack_factor_4_32" not in compact
    assert "get_pack_factor_8_32" not in compact
    assert parse_crossgl(crossgl) is not None


def test_codegen_selects_constexpr_helper_overload_in_materialized_body():
    code = """
    template <int Scale>
    constexpr int scaled_value(int value) {
        return value * Scale;
    }

    template <int Scale>
    constexpr int scaled_value(float value) {
        return int(value) + Scale;
    }

    template <int Scale = 3>
    int load_scaled() {
        constexpr int value = scaled_value<Scale>(4);
        return value;
    }

    int run_scaled() {
        return load_scaled<3>();
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "int value = 12;" in crossgl
    assert "scaled_value_3" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_preserves_runtime_constexpr_calls_in_materialized_returns():
    code = """
    template <int Scale = 2>
    constexpr int scaled_runtime_value(int value) {
        int scaled = value * Scale;
        return scaled;
    }

    template <int Scale = 2>
    int forward_scaled_value(int value) {
        return scaled_runtime_value<Scale>(value);
    }

    int run_scaled_value(int value) {
        return forward_scaled_value<2>(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "return scaled_runtime_value_2(value);" in crossgl
    assert "int scaled_runtime_value_2(int value)" in crossgl
    assert "int scaled = value * 2;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_materialized_constexpr_diagnostic_respects_shadowing():
    code = """
    template <int Bits, int Word = 32>
    constexpr int get_pack_factor() {
        return Word / Bits;
    }

    template <int Bits = 4>
    int load_factor(int runtime_bits) {
        {
            int Bits = runtime_bits;
            constexpr int factor = get_pack_factor<Bits>();
            return factor;
        }
    }

    int run_factor() {
        return load_factor<4>(4);
    }
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(
            code,
            file_path="materialized-shadow.metal",
        )

    diagnostic = exc_info.value
    assert diagnostic.enclosing_function == "load_factor"
    assert diagnostic.enclosing_specialization == "load_factor_4"
    assert diagnostic.nested_helper == "get_pack_factor<Bits>"
    assert diagnostic.function_name == "get_pack_factor"
    assert diagnostic.parameter_name == "Bits"
    assert diagnostic.requested_specialization == "get_pack_factor<Bits>"
    assert diagnostic.reason == "remains dependent on Bits"
    assert diagnostic.source_location["file"] == "materialized-shadow.metal"
    assert "in specialization 'load_factor_4'" in str(diagnostic)


@pytest.mark.parametrize(
    ("helper_declarations", "expected_kind", "expected_reason"),
    [
        (
            """
            template <int Bits>
            constexpr int resolve_factor() {
                int factor = 32 / Bits;
                return factor;
            }
            """,
            "constexpr_body",
            "contains statements outside the supported constexpr local "
            "declaration and return subset",
        ),
        (
            """
            template <int Bits>
            constexpr int resolve_factor() {
                return next_factor<Bits>();
            }

            template <int Bits>
            constexpr int next_factor() {
                return resolve_factor<Bits>();
            }
            """,
            "constexpr_body",
            "has a cyclic constexpr helper dependency",
        ),
    ],
)
def test_codegen_materialized_constexpr_failures_are_structured(
    helper_declarations,
    expected_kind,
    expected_reason,
):
    code = helper_declarations + """
        template <int Bits = 4>
        int load_factor() {
            constexpr int factor = resolve_factor<Bits>();
            return factor;
        }

        int run_factor() {
            return load_factor<4>();
        }
        """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(
            code,
            file_path="materialized-constexpr.metal",
        )

    diagnostic = exc_info.value
    assert diagnostic.enclosing_function == "load_factor"
    assert diagnostic.enclosing_specialization == "load_factor_4"
    assert diagnostic.argument_kind == expected_kind
    assert diagnostic.reason == expected_reason
    assert diagnostic.source_location["file"] == "materialized-constexpr.metal"


def test_codegen_rejects_ambiguous_materialized_constexpr_helper_overload():
    code = """
    template <int Bits>
    constexpr int resolve_factor() {
        return 32 / Bits;
    }

    template <int Bits>
    constexpr short resolve_factor() {
        return 16 / Bits;
    }

    template <int Bits = 4>
    int load_factor() {
        constexpr int factor = resolve_factor<Bits>();
        return factor;
    }

    int run_factor() {
        return load_factor<4>();
    }
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code)

    diagnostic = exc_info.value
    assert diagnostic.enclosing_function == "load_factor"
    assert diagnostic.enclosing_specialization == "load_factor_4"
    assert diagnostic.argument_kind == "overload"
    assert diagnostic.reason == (
        "does not identify one unique value-template declaration"
    )


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


def test_codegen_materializes_explicit_mixed_type_value_arguments():
    code = """
    template <typename T, bool Alternate = false>
    T mixed(T value) {
        return Alternate ? T(1) : -value;
    }

    float run_mixed(float value) {
        return mixed<float, true>(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "float mixed_float_true(float value)" in crossgl
    assert "return true ? float(1) : (-value);" in crossgl
    assert "return mixed_float_true(value);" in crossgl
    assert "generic<T>" not in crossgl
    assert not re.search(r"\b(?:T|Alternate)\b", crossgl)
    assert parse_crossgl(crossgl) is not None


def test_codegen_materializes_transitive_mixed_template_graph(tmp_path):
    code = """
    template <typename T, int Width>
    T leaf(T value) {
        return value + T(Width);
    }

    template <typename T, int Width>
    T middle(T value) {
        return leaf<T, Width>(value);
    }

    kernel void run_materialized(
        device float* out [[buffer(0)]],
        uint gid [[thread_position_in_grid]]) {
        out[gid] = middle<float, 4>(out[gid])
            + middle<float, 4>(out[gid])
            + middle<float, 8>(out[gid]);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert crossgl.count("float leaf_float_4(float value)") == 1
    assert crossgl.count("float middle_float_4(float value)") == 1
    assert crossgl.count("float leaf_float_8(float value)") == 1
    assert crossgl.count("float middle_float_8(float value)") == 1
    assert crossgl.index("float leaf_float_4") < crossgl.index("float middle_float_4")
    assert crossgl.index("float leaf_float_8") < crossgl.index("float middle_float_8")
    assert crossgl.index("float middle_float_8") < crossgl.index("run_materialized")
    assert crossgl.count("middle_float_4(buffer_load(out_, gid))") == 2
    assert not re.search(r"\b(?:T|Width)\b", crossgl)

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    assert "middle_float_4(out_.Load(gid))" in hlsl
    assert "middle_float_8(out_.Load(gid))" in hlsl
    assert "middle_float_4(out_[gid])" in glsl
    assert "middle_float_8(out_[gid])" in glsl

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "transitive-materialization.hlsl"
        hlsl_path.write_text(hlsl, encoding="utf-8")
        subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl_path = tmp_path / "transitive-materialization.comp"
        glsl_path.write_text(glsl, encoding="utf-8")
        subprocess.run(
            [
                glslang,
                "--target-env",
                "opengl",
                "-S",
                "comp",
                str(glsl_path),
                "-o",
                str(tmp_path / "transitive-materialization.spv"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def test_codegen_preserves_overload_selection_in_transitive_materialization():
    code = """
    template <int Width>
    float leaf(float value) {
        return value + float(Width);
    }

    template <int Width>
    int leaf(int value) {
        return value + Width;
    }

    template <typename T, int Width>
    T middle(T value) {
        return leaf<Width>(value);
    }

    float run_materialized(float value) {
        return middle<float, 4>(value);
    }
    """

    crossgl = convert_without_preprocessing(code)

    assert "float leaf_4_float(float value)" in crossgl
    assert "return leaf_4_float(value);" in crossgl
    assert "int leaf_4_int" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_rejects_partial_transitive_template_request():
    code = """
    template <typename T, int Width>
    T leaf(T value) {
        return value;
    }

    template <typename T, int Width>
    T middle(T value) {
        return leaf<T>(value);
    }

    float run_materialized(float value) {
        return middle<float, 4>(value);
    }
    """

    with pytest.raises(MetalTemplateArgumentResolutionError) as exc_info:
        convert_without_preprocessing(code, file_path="partial-template.metal")

    diagnostic = exc_info.value
    assert diagnostic.enclosing_function == "middle"
    assert diagnostic.enclosing_specialization == "middle_float_4"
    assert diagnostic.nested_helper == "leaf<T>"
    assert diagnostic.function_name == "leaf"
    assert diagnostic.parameter_name == "Width"
    assert diagnostic.argument_kind == "missing"
    assert diagnostic.reason == "was not supplied and has no declaration default"
    assert diagnostic.source_location["file"] == "partial-template.metal"


def test_codegen_rejects_recursive_transitive_template_graph():
    code = """
    template <typename T, int Width>
    T first(T value) {
        return second<T, Width>(value);
    }

    template <typename T, int Width>
    T second(T value) {
        return first<T, Width>(value);
    }

    float run_materialized(float value) {
        return first<float, 4>(value);
    }
    """

    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        convert_without_preprocessing(code)

    diagnostic = exc_info.value
    assert diagnostic.caller_specialization == "second_float_4"
    assert diagnostic.callee_template == "first"
    assert diagnostic.requested_arguments == ("float", "4")
    assert diagnostic.requested_signature == "first<float, 4>"
    assert str(diagnostic) == (
        "Metal template specialization graph is recursive: "
        "first_float_4 -> second_float_4 -> first_float_4"
    )


def test_codegen_honors_transitive_template_specialization_budget():
    code = """
    template <typename T, int Width>
    T leaf(T value) {
        return value + T(Width);
    }

    float run_materialized(float value) {
        return leaf<float, 4>(value) + leaf<float, 8>(value);
    }
    """
    tokens = MetalLexer(code, preprocess=False).tokenize()
    ast = MetalParser(
        tokens,
        file_path="template-budget.metal",
        max_template_specializations=1,
        template_specialization_limit_source="focused test budget",
    ).parse()

    with pytest.raises(MetalTemplateSpecializationError) as exc_info:
        MetalToCrossGLConverter().generate(ast)

    diagnostic = exc_info.value
    assert diagnostic.limit == 1
    assert diagnostic.limit_source == "focused test budget"
    assert diagnostic.unique_specialization_count == 2
    assert diagnostic.caller_specialization == "run_materialized"
    assert diagnostic.callee_template == "leaf"
    assert diagnostic.requested_arguments == ("float", "8")
    assert diagnostic.requested_signature == "leaf<float, 8>"


def test_codegen_materializes_ordered_variadic_alias_and_dependent_chain():
    crossgl = convert_without_preprocessing("""
        template <typename First, typename Second>
        struct select_second {
          using selected = Second;
          using type = selected;
        };

        template <typename... Ts>
        using second_t = typename select_second<Ts...>::type;

        template <typename... Ts>
        using chained_t = second_t<Ts...>;

        template <typename T>
        struct Box {};

        template <typename... Ts>
        struct Bundle {};

        template <typename... Ts>
        using boxed_t = Bundle<Box<Ts>...>;

        template <typename... Ts>
        struct make_void { using type = void; };

        template <typename... Ts>
        using void_t = typename make_void<Ts...>::type;

        chained_t<int, float> choose(chained_t<int, float> value) {
          return value;
        }

        boxed_t<int, float> ordered(boxed_t<int, float> value) {
          return value;
        }

        void_t<> empty_pack() { return; }
        """)

    normalized = normalize(crossgl)
    assert "float choose(float value)" in normalized
    assert (
        "Bundle<Box<int>,Box<float>> ordered(Bundle<Box<int>,Box<float>> value)"
        in normalized
    )
    assert "void empty_pack()" in normalized
    for alias_name in ("second_t", "chained_t", "boxed_t", "void_t"):
        assert alias_name not in crossgl
    assert "using " not in crossgl
    assert "template <" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_resolves_alias_templates_with_namespace_and_shadowing():
    crossgl = convert_without_preprocessing("""
        namespace Left {
        template <typename... Ts>
        using Pick = int;
        int left(Pick<float> value) { return value; }
        }

        namespace Right {
        template <typename... Ts>
        using Pick = float;
        float right(Pick<int> value) { return value; }
        }

        using namespace Right;
        float imported(Pick<int> value) { return value; }
        float qualified(Left::Pick<float> value) { return float(value); }

        template <template <typename...> class Pick>
        void shadowed(Pick<float> value) {}
        """)

    normalized = normalize(crossgl)
    assert "int left(int value)" in normalized
    assert "float right(float value)" in normalized
    assert "float imported(float value)" in normalized
    assert "float qualified(int value)" in normalized
    assert "generic<Pick> void shadowed(Pick<float> value)" in normalized


def test_codegen_reports_recursive_variadic_alias_dependency():
    code = """
    template <typename... Ts>
    using Loop = Loop<Ts...>;
    void consume(Loop<int> value) {}
    """
    tokens = MetalLexer(code, preprocess=False).tokenize()
    ast = MetalParser(tokens, file_path="recursive-alias.metal").parse()

    with pytest.raises(MetalAliasTemplateResolutionError) as exc_info:
        MetalToCrossGLConverter().generate(ast)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.metal-alias-template-unresolved"
    )
    assert diagnostic.alias_name == "Loop"
    assert diagnostic.requested_signature == "Loop<int>"
    assert diagnostic.dependency_chain == ("Loop<int>", "Loop<int>")
    assert diagnostic.source_location["file"] == "recursive-alias.metal"
    assert diagnostic.source_location["line"] == 2


def test_codegen_reports_symbolic_variadic_pack_use():
    code = """
    template <typename... Ts>
    struct make_void { using type = void; };
    template <typename... Ts>
    using void_t = typename make_void<Ts...>::type;
    template <typename... Ts>
    void consume(void_t<Ts...> value) {}
    """
    tokens = MetalLexer(code, preprocess=False).tokenize()
    ast = MetalParser(tokens, file_path="unresolved-pack.metal").parse()

    with pytest.raises(MetalAliasTemplateResolutionError) as exc_info:
        MetalToCrossGLConverter().generate(ast)

    diagnostic = exc_info.value
    assert diagnostic.alias_name == "void_t"
    assert diagnostic.parameter_pack == "Ts"
    assert "unresolved pack or template parameter" in diagnostic.reason
    assert diagnostic.source_location["file"] == "unresolved-pack.metal"


def test_codegen_enforces_chained_alias_materialization_budget():
    code = """
    template <typename T>
    using Inner = T;
    template <typename T>
    using Outer = Inner<T>;
    void consume(Outer<int> value) {}
    """
    tokens = MetalLexer(code, preprocess=False).tokenize()
    ast = MetalParser(
        tokens,
        file_path="alias-budget.metal",
        max_template_specializations=1,
        template_specialization_limit_source="focused alias budget",
    ).parse()

    with pytest.raises(MetalAliasTemplateResolutionError) as exc_info:
        MetalToCrossGLConverter().generate(ast)

    diagnostic = exc_info.value
    assert diagnostic.alias_name == "Inner"
    assert diagnostic.limit == 1
    assert diagnostic.limit_source == "focused alias budget"
    assert diagnostic.required_work_items == 2
    assert diagnostic.source_location["file"] == "alias-budget.metal"


def test_codegen_removes_alias_template_syntax_from_project_targets(tmp_path):
    crossgl = convert_without_preprocessing("""
        template <typename First, typename Second>
        struct select_second {
          using selected = Second;
          using type = selected;
        };
        template <typename... Ts>
        using second_t = typename select_second<Ts...>::type;
        template <typename... Ts>
        using result_t = second_t<Ts...>;

        kernel void alias_pack_kernel(
            device result_t<int, float>* values [[buffer(0)]],
            uint gid [[thread_position_in_grid]]) {
          values[gid] = float(gid);
        }
        """)
    shader = parse_crossgl(crossgl)
    targets = {
        "directx": TranslatorHLSLCodeGen().generate(shader),
        "opengl": GLSLCodeGen().generate(shader),
        "vulkan": VulkanSPIRVCodeGen().generate(shader),
    }

    for generated in (crossgl, *targets.values()):
        assert "template <" not in generated
        assert "using " not in generated
        assert "second_t" not in generated
        assert "result_t" not in generated
        assert "Ts..." not in generated
    assert "RWStructuredBuffer<float> values" in targets["directx"]
    assert "float values[]" in targets["opengl"]
    assert "OpTypeFloat 32" in targets["vulkan"]

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "alias-pack.hlsl"
        hlsl_path.write_text(targets["directx"], encoding="utf-8")
        subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        glsl_path = tmp_path / "alias-pack.comp"
        glsl_path.write_text(targets["opengl"], encoding="utf-8")
        subprocess.run(
            [
                glslang,
                "--target-env",
                "opengl",
                "-S",
                "comp",
                str(glsl_path),
                "-o",
                str(tmp_path / "alias-pack-opengl.spv"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is not None and spirv_val is not None:
        spirv_path = tmp_path / "alias-pack.spvasm"
        binary_path = tmp_path / "alias-pack.spv"
        spirv_path.write_text(targets["vulkan"], encoding="utf-8")
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


def test_stateless_compile_time_global_reaches_native_targets(tmp_path):
    # Reduced from MLX 4367c73b60541ddd5a266ce4644fd93d20223b6e,
    # mlx/backend/metal/kernels/metal_3_0/binary.metal.
    source = """
    struct Logger {
      constexpr Logger(constant char*, constant char*) constant {}

      template <typename... Args>
      void log_debug(constant char*, Args...) const {}

      template <typename... Args>
      void log_debug(constant char*, Args...) const constant {}
    };

    constant Logger logger("mlx", "binary_ops");

    int bump(device int* output) {
      output[0] += 1;
      return output[0];
    }

    kernel void compute(device int* output [[buffer(0)]]) {
      logger.log_debug("value=%d", bump(output));
      logger.log_debug("plain=%d", output[0]);
    }
    """
    source_path = tmp_path / "stateless-global.metal"
    source_path.write_text(source, encoding="utf-8")

    targets = {
        target: crosstl.translate(str(source_path), backend=target, format_output=False)
        for target in ("directx", "opengl")
    }
    report = translate_project(
        tmp_path, targets=["directx", "opengl"], output_dir="out"
    ).to_json()

    assert report["summary"]["translatedCount"] == 2, report
    assert report["summary"]["failedCount"] == 0, report
    for generated in targets.values():
        assert "logger" not in generated
        assert "log_debug" not in generated
        assert (
            len(re.findall(r"(?m)^\s*bump(?:_[A-Za-z0-9_]+)?\([^;]*\);\s*$", generated))
            == 1
        )

    HLSLParser(HLSLLexer(targets["directx"]).tokenize()).parse()
    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "stateless-global.hlsl"
        hlsl_path.write_text(targets["directx"], encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    assert_opengl_compute_validates_if_available(
        targets["opengl"], tmp_path, "stateless-global"
    )


def test_project_reports_unsafe_stateless_global_blocker(tmp_path):
    source = """
    struct Logger {
      constexpr Logger() constant {}
      void log_debug(constant char*) const constant { printf("real log"); }
    };

    constant Logger logger;

    kernel void compute() {
      logger.log_debug("message");
    }
    """
    (tmp_path / "unsafe-stateless-global.metal").write_text(source, encoding="utf-8")

    payload = translate_project(
        tmp_path, targets=["directx"], output_dir="out"
    ).to_json()

    assert payload["summary"]["translatedCount"] == 0, payload
    assert payload["summary"]["failedCount"] == 1, payload
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.metal-stateless-global-unsafe": 1
    }
    assert payload["summary"]["missingCapabilityCounts"] == {
        "metal.stateless-global-elision": 1
    }
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["location"]["line"] == 10
    assert diagnostic["location"]["column"] == 7
    assert diagnostic["details"]["statelessGlobal"]["globalName"] == "logger"
    assert diagnostic["details"]["statelessGlobal"]["typeName"] == "Logger"
    assert diagnostic["details"]["statelessGlobal"]["methodName"] == "log_debug"
    assert diagnostic["details"]["statelessGlobal"]["reason"] == ("method-has-effects")
    assert diagnostic["details"]["statelessGlobal"]["effect"] == "function-call"


def test_stateful_compile_time_global_reaches_existing_directx_lowering(tmp_path):
    source = """
    struct Config {
      int value;
    };

    constant Config config = {7};

    kernel void compute(device int* output [[buffer(0)]]) {
      output[0] = config.value;
    }
    """
    source_path = tmp_path / "stateful-global.metal"
    source_path.write_text(source, encoding="utf-8")

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    assert "static const Config config = {7};" in generated
    assert "output[0] = config.value;" in generated
    HLSLParser(HLSLLexer(generated).tokenize()).parse()

    dxc = shutil.which("dxc")
    if dxc is not None:
        hlsl_path = tmp_path / "stateful-global.hlsl"
        hlsl_path.write_text(generated, encoding="utf-8")
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", str(hlsl_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
