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
    assert "vec4 pickColor(vec4 table[2][3], PatchData patches[2][2]" in glsl
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
