import textwrap

import pytest

from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

VERTEX_GLSL = textwrap.dedent("""
    #version 450 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec2 uv;
    layout(location = 0) out vec2 vUV;
    uniform mat4 uMVP;

    void main() {
        vUV = uv;
        gl_Position = uMVP * vec4(position, 1.0);
    }
    """).strip()

FRAGMENT_GLSL = textwrap.dedent("""
    #version 450 core
    layout(location = 0) in vec2 vUV;
    layout(location = 0) out vec4 fragColor;
    uniform sampler2D uTexture;

    void main() {
        vec4 color = texture(uTexture, vUV);
        if (color.a < 0.1) {
            discard;
        }
        fragColor = color;
    }
    """).strip()

CONTROL_FLOW_GLSL = textwrap.dedent("""
    #version 450 core
    layout(location = 0) in vec3 position;

    void main() {
        int i = 0;
        for (int j = 0; j < 4; j++) {
            if (j == 2) {
                continue;
            }
            i += j;
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
            default:
                break;
        }

        gl_Position = vec4(position, 1.0);
    }
    """).strip()

STRUCT_ARRAY_GLSL = textwrap.dedent("""
    #version 450 core
    struct Light {
        vec3 position;
        vec3 color;
        float intensity;
    };

    uniform Light lights[4];
    layout(location = 0) out vec3 vColor;

    void main() {
        vec3 color = lights[0].color * lights[0].intensity;
        vColor = color;
        gl_Position = vec4(1.0);
    }
    """).strip()

INTERFACE_BLOCK_GLSL = textwrap.dedent("""
    #version 450 core
    layout(std140, binding = 0) uniform Globals {
        mat4 mvp;
        vec4 baseColor;
    };

    in VertexIn {
        vec3 position;
        vec2 uv;
    } vin;

    out VertexOut {
        vec4 color;
    } vout;

    void main() {
        vout.color = vec4(vin.position, 1.0);
        gl_Position = mvp * vec4(vin.position, 1.0);
    }
    """).strip()
COMPUTE_GLSL = textwrap.dedent("""
    #version 430
    layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    layout(binding = 0, rgba8) uniform writeonly image2D outImage;

    void main() {
        ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
        imageStore(outImage, coord, vec4(1.0, 0.0, 0.0, 1.0));
    }
    """).strip()


def generate_crossgl(code: str, shader_type: str = "vertex") -> str:
    tokens = GLSLLexer(code).tokenize()
    ast = GLSLParser(tokens, shader_type).parse()
    generator = GLSLToCrossGLConverter(shader_type=shader_type)
    return generator.generate(ast)


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    parser = CrossGLParser(tokens)
    return parser.parse()


def assert_roundtrip(code: str, shader_type: str, expected_stage: ShaderStage) -> str:
    output = generate_crossgl(code, shader_type)
    assert isinstance(output, str)
    assert output.strip()
    assert "#version" in output
    shader_ast = parse_crossgl(output)
    assert expected_stage in shader_ast.stages
    return output


def test_codegen_vertex_roundtrip():
    output = assert_roundtrip(VERTEX_GLSL, "vertex", ShaderStage.VERTEX)
    lowered = output.lower()
    assert "shader" in lowered
    assert "vertex" in lowered
    for name in ["position", "uv", "vUV", "uMVP"]:
        assert name in output


def test_codegen_fragment_roundtrip():
    output = assert_roundtrip(FRAGMENT_GLSL, "fragment", ShaderStage.FRAGMENT)
    lowered = output.lower()
    assert "shader" in lowered
    assert "fragment" in lowered
    for name in ["vUV", "fragColor", "uTexture"]:
        assert name in output


def test_codegen_default_function_argument_from_glslang_default_args():
    code = textwrap.dedent("""
        #version 450

        void foo(int n, int x = 2)
        {
        }

        void main()
        {
            foo(6);
            foo(8, 3);
        }
    """).strip()

    output = assert_roundtrip(code, "compute", ShaderStage.COMPUTE)

    assert "int x = 2" in output


def test_codegen_macro_declaration_prefixes_from_filament_sources():
    code = textwrap.dedent("""
        #version 450

        LAYOUT_LOCATION(LOCATION_POSITION) ATTRIBUTE vec4 position;

        struct PostProcessVertexInputs {
            vec2 normalizedUV;
            DECLARE_FIELD(MATERIAL_SLOT) highp vec4 material;
        };

        void initPostProcessMaterialVertex(out PostProcessVertexInputs inputs) {
            inputs.normalizedUV = position.xy;
        }

        void main() {
            gl_Position = position;
        }
    """).strip()

    output = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "vec4 position" in output
    assert "vec4 material" in output


def test_codegen_resource_function_descriptors():
    converter = GLSLToCrossGLConverter(shader_type="fragment")

    assert "texture2D" not in converter.function_map
    assert "textureCube" not in converter.function_map
    assert converter.resource_function_descriptor("texture2D") == {
        "name": "texture2D",
        "function": "texture",
        "resource": "texture",
        "operation": "sample",
    }
    assert converter.resource_function_descriptor("textureLod") == {
        "name": "textureLod",
        "function": "textureLod",
        "resource": "texture",
        "operation": "sample_lod",
    }
    assert converter.resource_function_descriptor("texelFetch") == {
        "name": "texelFetch",
        "function": "texelFetch",
        "resource": "texture",
        "operation": "fetch",
    }
    assert converter.resource_function_descriptor("textureCompare") == {
        "name": "textureCompare",
        "function": "textureCompare",
        "resource": "texture",
        "operation": "compare",
    }
    assert converter.resource_function_descriptor("textureQueryLod") == {
        "name": "textureQueryLod",
        "function": "textureQueryLod",
        "resource": "texture",
        "operation": "query_lod",
    }
    assert converter.resource_function_descriptor("textureSamples") == {
        "name": "textureSamples",
        "function": "textureSamples",
        "resource": "texture",
        "operation": "query_samples",
    }
    assert converter.resource_function_descriptor("imageLoad") == {
        "name": "imageLoad",
        "function": "imageLoad",
        "resource": "image",
        "operation": "load",
    }
    assert converter.resource_function_descriptor("imageStore") == {
        "name": "imageStore",
        "function": "imageStore",
        "resource": "image",
        "operation": "store",
    }
    assert converter.resource_function_descriptor("imageAtomicAdd") == {
        "name": "imageAtomicAdd",
        "function": "imageAtomicAdd",
        "resource": "image",
        "operation": "atomic",
    }
    assert converter.resource_function_descriptor("imageSize") == {
        "name": "imageSize",
        "function": "imageSize",
        "resource": "image",
        "operation": "query_size",
    }
    assert converter.resource_function_descriptor("imageSamples") == {
        "name": "imageSamples",
        "function": "imageSamples",
        "resource": "image",
        "operation": "query_samples",
    }
    assert converter.resource_function_descriptor("mix") is None


def test_codegen_texture_intrinsics_use_canonical_crossgl_resources():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec2 uv;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2D tex;

        void main() {
            vec4 base = texture(tex, uv);
            vec4 legacy = texture2D(tex, uv);
            vec4 mip = textureLod(tex, uv, 1.0);
            vec4 grad = textureGrad(tex, uv, vec2(1.0), vec2(1.0));
            vec4 gathered = textureGather(tex, uv);
            fragColor = base + legacy + mip + grad + gathered;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "sampler2D tex;" in crossgl
    assert "Texture2D tex;" not in crossgl
    assert "texture(tex, input.uv)" in crossgl
    assert "textureLod(tex, input.uv, 1.0)" in crossgl
    assert "textureGrad(tex, input.uv, vec2(1.0), vec2(1.0))" in crossgl
    assert "textureGather(tex, input.uv)" in crossgl
    assert "sample(tex" not in crossgl
    assert "texture2D(" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "layout(binding = 0) uniform sampler2D tex;" in glsl
    assert "texture(tex, uv)" in glsl
    assert "textureLod(tex, uv, 1.0)" in glsl
    assert "textureGrad(tex, uv, vec2(1.0), vec2(1.0))" in glsl
    assert "textureGather(tex, uv)" in glsl


def test_codegen_array_of_arrays_return_type_from_glslang_spv_aofa():
    code = textwrap.dedent("""
        #version 430

        in float infloat;
        out float outfloat;

        float[4][7] foo(float a[5][7])
        {
            float r[7];
            r = a[2];
            return float[4][7](a[0], a[1], r, a[3]);
        }

        void main()
        {
            float u[][7];
            u[2][2] = infloat;
            outfloat = foo(u)[1][2];
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float[4][7] foo(float a[5][7])" in crossgl
    assert "return { a[0], a[1], r, a[3] };" in crossgl
    assert "float u[][7];" in crossgl


def test_codegen_hex_float_literals_import_to_parseable_crossgl():
    code = textwrap.dedent("""
        #version 450
        void main() {
            float exposure = 0x1.8p+1;
            float bias = 0x1p-2;
            gl_Position = vec4(exposure + bias);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "float exposure = 0x1.8p+1;" in crossgl
    assert "float bias = 0x1p-2;" in crossgl
    parse_crossgl(crossgl)


def test_codegen_scientific_float_literals_import_to_parseable_crossgl():
    code = textwrap.dedent("""
        #version 450
        void main() {
            float tiny = 1e-3;
            float large = 2.0E+4;
            gl_Position = vec4(tiny + large);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)
    shader_ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "float tiny = 1e-3;" in crossgl
    assert "float large = 2.0E+4;" in crossgl
    assert "float tiny = 0.001;" in glsl
    assert "float large = 20000.0;" in glsl
    assert "gl_Position" in glsl


def test_codegen_vulkan_separate_texture_sampler_uniforms_are_resources():
    code = textwrap.dedent("""
        #version 450
        #extension GL_EXT_nonuniform_qualifier : require

        layout(set = 0, binding = 0) uniform texture2D Textures[];
        layout(set = 1, binding = 0) uniform sampler ImmutableSampler;

        layout(location = 0) in vec2 in_uv;
        layout(location = 0) out vec4 out_frag_color;

        void main() {
            out_frag_color = texture(sampler2D(Textures[0], ImmutableSampler), in_uv);
        }
    """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "texture2D Textures[] @binding(0);" in crossgl
    assert "sampler ImmutableSampler @binding(0);" in crossgl
    assert "cbuffer Uniforms" not in crossgl


def test_codegen_vulkan_subpass_inputs_are_resources():
    code = textwrap.dedent("""
        #version 450
        layout(input_attachment_index = 0, set = 0, binding = 0)
        uniform subpassInput colorInput;
        layout(input_attachment_index = 1, set = 0, binding = 1)
        uniform usubpassInputMS idInput;
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = subpassLoad(colorInput) + vec4(subpassLoad(idInput, 0));
        }
    """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "subpassInput colorInput @binding(0) @input_attachment_index(0);" in crossgl
    assert "usubpassInputMS idInput @binding(1) @input_attachment_index(1);" in crossgl
    assert "subpassLoad(colorInput)" in crossgl
    assert "subpassLoad(idInput, 0)" in crossgl
    assert "cbuffer Uniforms" not in crossgl


def test_codegen_comma_separated_for_updates():
    code = textwrap.dedent("""
        #version 460
        void main() {
            uint plane_index = 0;
            for (uint i = 0; i < 3; ++i, ++plane_index) {
                plane_index += i;
            }
        }
    """).strip()

    crossgl = generate_crossgl(code, "compute")

    assert "for (uint i = 0; (i < 3); (++i), (++plane_index))" in crossgl


def test_codegen_logical_xor_normalizes_to_boolean_inequality():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) out vec4 fragColor;

        void main() {
            bool a = true;
            bool b = false;
            bool c = a ^^ b;
            fragColor = c ? vec4(1.0) : vec4(0.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "bool c = (a != b)" in crossgl
    assert "^^" not in crossgl


def test_codegen_parenthesized_comma_assignment_expression():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) out vec4 fragColor;

        void main() {
            int a = 0;
            int b = (a = 1, a + 2);
            fragColor = vec4(float((a = 3, b)));
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "int b = (a = 1, (a + 2));" in crossgl
    assert "float((a = 3, b))" in crossgl


def test_codegen_unnamed_function_parameters_get_stable_names():
    code = textwrap.dedent("""
        #version 400 core

        void ftd(int, float, double) {}

        void main() {
            ftd(1, 1.0, 2.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "void ftd(int _param0, float _param1, double _param2)" in crossgl
    assert "ftd(1, 1.0, 2.0)" in crossgl


def test_codegen_query_intrinsics_use_resource_descriptors():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec2 uv;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2D tex;
        layout(binding = 0, rgba32f) uniform image2D img;

        void main() {
            ivec2 texSize = textureSize(tex, 0);
            int levels = textureQueryLevels(tex);
            vec2 lod = textureQueryLod(tex, uv);
            int sampleCount = textureSamples(tex);
            ivec2 imgSize = imageSize(img);
            int imgSamples = imageSamples(img);
            imageStore(img, ivec2(0), vec4(texSize, levels + sampleCount + imgSamples));
            fragColor = vec4(imgSize, lod);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "textureSize(tex, 0)" in crossgl
    assert "textureQueryLevels(tex)" in crossgl
    assert "textureQueryLod(tex, input.uv)" in crossgl
    assert "textureSamples(tex)" in crossgl
    assert "imageSize(img)" in crossgl
    assert "imageSamples(img)" in crossgl
    assert (
        "imageStore(img, ivec2(0), vec4(texSize, ((levels + sampleCount) + imgSamples)))"
        in crossgl
    )


def test_codegen_resource_array_functions_keep_array_receivers():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) out vec4 fragColor;
        uniform sampler2D textures[2];
        layout(binding = 0, rgba32f) uniform image2D images[2];

        void main() {
            int index = 1;
            vec4 c = texture(textures[index], vec2(0.5));
            vec4 r = imageLoad(images[index], ivec2(1, 2));
            imageStore(images[index], ivec2(1, 2), c + r);
            fragColor = c + r;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "sampler2D textures[2];" in crossgl
    assert "image2D images[2] @binding(0) @rgba32f;" in crossgl
    assert "@rgba32f[2]" not in crossgl
    assert "texture(textures[index], vec2(0.5))" in crossgl
    assert "imageLoad(images[index], ivec2(1, 2))" in crossgl
    assert "imageStore(images[index], ivec2(1, 2), (c + r))" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "layout(binding = 0) uniform sampler2D textures[2];" in glsl
    assert "layout(rgba32f, binding = 0) uniform image2D images[2];" in glsl


def test_codegen_multisample_compare_roundtrip_uses_translator_diagnostics():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec2 uv;
        layout(location = 1) in vec3 uvLayer;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2DMS msTex;
        uniform sampler2DMSArray msArray;

        void main() {
            float cmp = textureCompare(msTex, uv, 0.5);
            vec4 gathered = textureGatherCompare(msArray, uvLayer, 0.5);
            fragColor = vec4(cmp) + gathered;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "sampler2DMS msTex;" in crossgl
    assert "sampler2DMSArray msArray;" in crossgl
    assert "Texture2DMS" not in crossgl
    assert "textureCompare(msTex, input.uv, 0.5)" in crossgl
    assert "textureGatherCompare(msArray, input.uvLayer, 0.5)" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert (
        "unsupported GLSL multisample texture comparison: textureCompare on sampler2DMS"
        in glsl
    )
    assert (
        "unsupported GLSL multisample texture gather comparison: textureGatherCompare on sampler2DMSArray"
        in glsl
    )
    assert "texture(msTex" not in glsl
    assert "textureGather(msArray" not in glsl


def test_codegen_control_flow_roundtrip():
    output = assert_roundtrip(CONTROL_FLOW_GLSL, "vertex", ShaderStage.VERTEX)
    lowered = output.lower()
    assert "for" in lowered
    assert "while" in lowered
    assert "switch" in lowered


def test_codegen_glslang_while_condition_declaration_roundtrip():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec4 color;
        layout(location = 0) out vec4 fragColor;

        void main() {
            float d = 0.5;
            while (bool test = color.y < d) {
                fragColor = color;
                d -= 0.25;
            }
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "while (true)" in crossgl
    assert "bool test = (input.color.y < d);" in crossgl
    assert "if (!test)" in crossgl


def test_codegen_reserved_helper_name_from_glslang_struct_sample_roundtrip():
    code = textwrap.dedent("""
        #version 450 core
        struct S {
            vec4 a;
        };
        layout(location = 0) out vec4 fragColor;

        S compute(vec4 value) {
            return S(value);
        }

        void main() {
            S result = compute(vec4(1.0));
            fragColor = result.a;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "S compute_(vec4 value)" in crossgl
    assert "compute_(vec4(1.0))" in crossgl
    assert "S compute(" not in crossgl
    assert " compute(vec4" not in crossgl
    assert "S{value}" in crossgl


def test_codegen_do_while_continue_roundtrip_preserves_condition_check():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) out vec4 fragColor;

        void main() {
            int i = 0;
            do {
                i++;
                if (i < 2) {
                    continue;
                }
                fragColor = vec4(float(i));
            } while (i < 3);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "do {" in crossgl
    assert "continue;" in crossgl
    assert "} while ((i < 3));" in crossgl
    assert "while (true)" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "do {" in glsl
    assert "continue;" in glsl
    assert "} while ((i < 3));" in glsl
    assert "while (true)" not in glsl


def test_codegen_structs_and_arrays_roundtrip():
    output = assert_roundtrip(STRUCT_ARRAY_GLSL, "vertex", ShaderStage.VERTEX)
    assert "Light" in output
    assert "lights" in output
    assert "vColor" in output


def test_codegen_const_gather_offset_arrays_roundtrip():
    code = textwrap.dedent("""
        #version 460 core
        uniform sampler2D tex;
        uniform sampler2DArrayShadow shadowArray;
        layout(location = 0) out vec4 fragColor;

        void main() {
            const ivec2 offsets[4] = {
                ivec2(-1, 0),
                ivec2(1, 0),
                ivec2(0, -1),
                ivec2(0, 1),
            };
            const ivec2 ctorOffsets[4] = ivec2[4](
                ivec2(-1, -1),
                ivec2(1, -1),
                ivec2(-1, 1),
                ivec2(1, 1)
            );
            const ivec2 nested[2][2] = {
                { ivec2(-1, 0), ivec2(1, 0) },
                { ivec2(0, -1), ivec2(0, 1) },
            };
            vec2 uv = vec2(0.5, 0.5);
            vec3 uvLayer = vec3(uv, 0.0);
            ivec2 selected = nested[1][0];
            fragColor = textureGatherOffsets(tex, uv, offsets, 0)
                + textureGatherOffsets(tex, uv, ctorOffsets, 1)
                + textureGatherCompareOffsets(shadowArray, uvLayer, 0.4, ctorOffsets)
                + vec4(selected, 0, 0) * 0.0;
        }
        """).strip()

    crossgl = generate_crossgl(code, "fragment")
    assert "ArrayAccessNode" not in crossgl
    assert "InitializerListNode" not in crossgl
    assert "const ivec2 offsets[4] = {" in crossgl
    assert "const ivec2 ctorOffsets[4] = {" in crossgl
    assert "const ivec2 nested[2][2] = {" in crossgl
    assert "ivec2 selected = nested[1][0]" in crossgl
    assert "textureGatherOffsets(tex, uv, offsets, 0)" in crossgl
    assert "textureGatherOffsets(tex, uv, ctorOffsets, 1)" in crossgl
    assert (
        "textureGatherCompareOffsets(shadowArray, uvLayer, 0.4, ctorOffsets)" in crossgl
    )

    shader_ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(shader_ast)
    assert "textureGatherOffsets(" not in glsl
    assert "textureGatherCompareOffsets(" not in glsl
    assert "textureGatherOffset(tex" in glsl
    assert "textureGatherOffset(shadowArray" in glsl
    assert "const ivec2 nested[2][2] = {" in glsl
    assert "ivec2 selected = nested[1][0];" in glsl


def test_codegen_global_array_type_constants_roundtrip_parse():
    code = textwrap.dedent("""
        #version 450
        const vec2[3] positions = vec2[]
        (
            vec2(-1, -1),
            vec2(-1,  3),
            vec2( 3, -1)
        );

        const vec2[3] uv = vec2[]
        (
            vec2(0, 0),
            vec2(0, 2),
            vec2(2, 0)
        );

        layout(location = 0) out vec2 outUV;

        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0f, 1.0f);
            outUV = uv[gl_VertexIndex];
        }
    """).strip()

    crossgl = generate_crossgl(code, "vertex")

    assert "const vec2[3] positions = {" in crossgl
    assert "const vec2[3] uv = {" in crossgl
    assert "const vec2 positions[3]" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_compute_roundtrip():
    output = assert_roundtrip(COMPUTE_GLSL, "compute", ShaderStage.COMPUTE)
    lowered = output.lower()
    assert "compute" in lowered
    assert "outImage" in output


def test_codegen_invalid_glsl_raises():
    code = "void main() { float x = 1.0 return x; }"
    with pytest.raises(SyntaxError):
        generate_crossgl(code, "vertex")


def test_codegen_interface_block_roundtrip():
    output = assert_roundtrip(INTERFACE_BLOCK_GLSL, "vertex", ShaderStage.VERTEX)
    assert "struct VertexIn" in output
    assert "struct VertexOut" in output
    assert "struct Globals" in output
    assert "cbuffer Uniforms" in output


def test_codegen_uniform_struct_specifier_uses_uniform_buffer():
    code = textwrap.dedent("""
        precision mediump float;
        uniform struct S {
            float field;
        } s;

        void main() {
            gl_FragColor = vec4(0.0, s.field, 0.0, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "struct S" in crossgl
    assert "cbuffer Uniforms" in crossgl
    assert "S s;" in crossgl
    assert "\n    S s;\n\n    fragment" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_local_struct_with_mixed_array_declarators_from_khronos_webgl():
    code = textwrap.dedent("""
        precision mediump float;
        void main() {
            struct S {
                float field;
            };
            S s1[2], s2;
            s1[0].field = 1.0;
            gl_FragColor = vec4(0.0, s1[0].field, 0.0, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "struct S" in crossgl
    assert "S s1[2];" in crossgl
    assert "S s2;" in crossgl
    assert "s1[0].field = 1.0;" in crossgl
    parse_crossgl(crossgl)


def test_codegen_push_constant_interface_block_preserves_attribute():
    code = textwrap.dedent("""
        #version 450
        layout(push_constant, std430) uniform MVPUniform {
            mat4 model;
            mat4 view_proj;
        } mvp_uniform;

        void main() {
            gl_Position = mvp_uniform.view_proj * vec4(1.0);
        }
        """).strip()

    crossgl = generate_crossgl(code, "vertex")

    assert "cbuffer MVPUniform @push_constant {" in crossgl
    assert "mat4 model;" in crossgl
    assert "mat4 view_proj;" in crossgl
    assert "cbuffer Uniforms" not in crossgl
    assert "mvp_uniform.view_proj" not in crossgl
    assert "gl_Position = (view_proj * vec4(1.0));" in crossgl
    parse_crossgl(crossgl)


def test_codegen_multidimensional_interface_and_parameter_arrays_roundtrip():
    code = textwrap.dedent("""
        #version 460 core
        in VertexIn {
            vec3 positions[2][3];
            ivec2 ids[2][2];
        } vin[2][3];

        out VertexOut {
            vec4 colors[2][3];
        } vout;

        struct PatchData {
            vec4 control[2][3];
        };

        vec4 pickColor(in vec4 table[2][3], inout PatchData patches[2][2], int i, int j) {
            patches[0][1].control[i][j] = table[i][j];
            return patches[0][1].control[i][j];
        }

        void main() {
            PatchData patches[2][2];
            vec4 table[2][3];
            vout.colors[1][2] = pickColor(table, patches, 1, 2);
            gl_Position = vec4(vin[1][2].positions[0][1], 1.0);
        }
        """).strip()

    crossgl = generate_crossgl(code, "vertex")
    assert "@glsl_interface_instance(vin) @glsl_interface_array(2, 3)" in crossgl
    assert "vec3 positions[2][3];" in crossgl
    assert "ivec2 ids[2][2];" in crossgl
    assert "vec4 colors[2][3];" in crossgl
    assert (
        "vec4 pickColor(in vec4 table[2][3], inout PatchData patches[2][2]" in crossgl
    )
    assert "PatchData patches[2][2];" in crossgl
    assert "vec4 table[2][3];" in crossgl

    shader_ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(shader_ast)
    assert "} vin[2][3];" in glsl
    assert "vec3 positions[2][3];" in glsl
    assert "ivec2 ids[2][2];" in glsl
    assert "vec4 colors[2][3];" in glsl
    assert "vec4 pickColor(vec4 table[2][3], inout PatchData patches[2][2]" in glsl
    assert "vec4[2][3] table" not in glsl
    assert "PatchData[2][2] patches" not in glsl


def test_codegen_compute_atomics_and_barriers():
    code = textwrap.dedent("""
        #version 450 core
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0, rgba8) uniform coherent image2D img;
        layout(std430, binding = 1) buffer Counter { uint value; } counter;

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            imageAtomicAdd(img, coord, 1);
            atomicAdd(counter.value, 1);
            memoryBarrier();
            barrier();
        }
    """).strip()

    output = assert_roundtrip(code, "compute", ShaderStage.COMPUTE)
    for expected in ["imageAtomicAdd", "atomicAdd", "memoryBarrier", "barrier"]:
        assert expected in output


def test_codegen_inserts_default_version_when_missing():
    code = textwrap.dedent("""
        layout(location = 0) in vec3 position;
        void main() {
            gl_Position = vec4(position, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code, "vertex")
    shader_ast = parse_crossgl(crossgl)
    assert ShaderStage.VERTEX in shader_ast.stages
    glsl_code = GLSLCodeGen().generate(shader_ast)
    assert glsl_code.lstrip().startswith("#version 450 core")


if __name__ == "__main__":
    pytest.main()
