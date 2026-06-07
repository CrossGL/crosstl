import textwrap

import pytest

from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
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


def test_codegen_layout_qualifier_with_newline_before_parens_from_glsl_grammar():
    # GLSL 4.60 layout-qualifier is "layout ( ... )"; newlines are whitespace.
    code = textwrap.dedent("""
        #version 450 core

        layout
        (location = 0) in vec3 position;

        void main()
        {
            gl_Position = vec4(position, 1.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "vec3 position @location(0);" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "layout(location = 0) in vec3 position;" in glsl


def test_codegen_fragment_roundtrip():
    output = assert_roundtrip(FRAGMENT_GLSL, "fragment", ShaderStage.FRAGMENT)
    lowered = output.lower()
    assert "shader" in lowered
    assert "fragment" in lowered
    for name in ["vUV", "fragColor", "uTexture"]:
        assert name in output


def test_codegen_fragment_output_array_roundtrip_uses_direct_declaration():
    # Fragment output arrays are a common MRT pattern; they cannot be modeled as
    # a scalar fragment return value without corrupting indexed writes.
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) out vec4 fragColor[2];

        void main() {
            fragColor[0] = vec4(1.0);
            fragColor[1] = vec4(0.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "out vec4 fragColor[2] @location(0);" in crossgl
    assert "void main()" in crossgl
    assert "vec4 main()" not in crossgl
    assert "vec4 fragColor;" not in crossgl
    assert "return fragColor;" not in crossgl
    assert "fragColor[0] = vec4(1.0);" in crossgl
    assert "fragColor[1] = vec4(0.0);" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "layout(location = 0) out vec4 fragColor[2];" in glsl
    assert "fragColor[0] = vec4(1.0);" in glsl
    assert "fragColor[1] = vec4(0.0);" in glsl


def test_codegen_shadertoy_main_image_synthesizes_fragment_entrypoint():
    # Shadertoy-style fragments provide mainImage instead of GLSL main.
    code = textwrap.dedent("""
        #version 300 es
        precision highp float;
        uniform vec3 iResolution;
        uniform sampler2D iChannel0;

        void mainImage(out vec4 fragColor, in vec2 fragCoord)
        {
            vec2 uv = fragCoord / iResolution.xy;
            fragColor = texture(iChannel0, uv);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "void mainImage(out vec4 fragColor, in vec2 fragCoord)" in crossgl
    assert "vec4 main() @ gl_FragColor" in crossgl
    assert "mainImage(shadertoyFragColor, gl_FragCoord.xy);" in crossgl
    assert "return shadertoyFragColor;" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "void main()" in glsl
    assert "mainImage(shadertoyFragColor, gl_FragCoord.xy);" in glsl
    assert "fragColor = shadertoyFragColor;" in glsl


def test_codegen_block_scope_precision_statement_from_glslang_precision_frag():
    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/precision.frag, which declares precision defaults inside nested blocks.
    code = textwrap.dedent("""
        #version 100
        precision highp int;

        void main()
        {
            precision lowp int;
            lowp int sum = 0;

            {
                precision highp int;
                int level2_high = sum;
            }

            do {
                if (true) {
                    precision mediump int;
                    int level4_medium = sum;
                }
            } while (false);

            gl_FragColor = vec4(float(sum));
        }
    """).strip()

    output = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "int sum @lowp = 0;" in output
    assert "int level2_high = sum;" in output
    assert "int level4_medium = sum;" in output
    assert "precision lowp int" not in output
    assert "precision mediump int" not in output


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


def test_codegen_for_init_custom_struct_declaration_from_glsl_460_grammar():
    # Reduced from Khronos GLSL 4.60.8 grammar:
    # for_init_statement accepts declaration_statement, including user types.
    code = textwrap.dedent("""
        #version 460

        struct Cursor {
            int value;
        };

        void main()
        {
            for (Cursor cursor = Cursor(0); cursor.value < 2; cursor.value++)
            {
                cursor.value += 1;
            }
        }
    """).strip()

    output = assert_roundtrip(code, "compute", ShaderStage.COMPUTE)

    assert (
        "for (Cursor cursor = Cursor{0}; (cursor.value < 2); (cursor.value++))"
        in output
    )
    assert "cursor.value += 1;" in output


def test_codegen_interface_block_nested_struct_member_from_glsl_460_grammar():
    # Reduced from Khronos GLSL 4.60.8 grammar:
    # member_declaration -> type_specifier struct_declarator_list ';',
    # where type_specifier_nonarray can be a struct_specifier.
    code = textwrap.dedent("""
        #version 460

        layout(std140, binding = 0) uniform Scene {
            struct Light {
                vec4 position;
                vec4 color;
            } light;
        };

        void main()
        {
            gl_Position = light.position;
        }
    """).strip()

    output = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "struct Light" in output
    assert "struct Scene" in output
    assert output.index("struct Light") < output.index("struct Scene")
    assert "Light light;" in output
    assert "gl_Position = light.position;" in output


def test_codegen_local_function_prototype_from_glslang_scope_vert():
    code = textwrap.dedent("""
        #version 110

        void helper() {
        }

        void main() {
            void helper();
            helper();
            gl_Position = vec4(1.0);
        }
    """).strip()

    output = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert output.count("void helper()") == 1
    assert "void helper();" not in output
    assert "helper();" in output
    assert "gl_Position = vec4(1.0);" in output


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


def test_codegen_precision_qualified_custom_type_local_from_filament_fsr():
    # Reduced from google/filament@6221f22a79597006b98e329f7267ee59f8ff354c
    # filament/src/materials/fsr/ffx_fsr1_mobile.fs.
    code = textwrap.dedent("""
        #version 450

        AF3 FsrEasuSampleF(highp AF2 p);

        void main()
        {
            highp AF2 pp = AF2(0.5);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "AF3 FsrEasuSampleF(AF2 p @highp)" in crossgl
    assert "AF2 pp @highp = AF2(0.5);" in crossgl


def test_codegen_explicit_typecast_from_glslang_nv_extension():
    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/spv.nv.explicittypecast.frag, which uses GL_NV_explicit_typecast.
    code = textwrap.dedent("""
        #version 460
        #extension GL_NV_explicit_typecast : enable

        float func(float a, vec2 b)
        {
            return dot(b, vec2(a));
        }

        void main()
        {
            float f_0;
            uint u_0;
            vec4 v4_0;
            uvec4 u4_0;

            f_0 = (float) u_0;
            v4_0 = (vec4) u4_0;
            func((float)u_0, (vec2)v4_0);
        }
    """).strip()

    output = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "f_0 = float(u_0);" in output
    assert "v4_0 = vec4(u4_0);" in output
    assert "func(float(u_0), vec2(v4_0));" in output


def test_codegen_two_argument_atan_from_khronos_docs_maps_to_crossgl_atan2():
    # Khronos GLSL docs define atan(y, x) as the quadrant-aware form; CrossGL
    # uses atan2 for that portable builtin spelling.
    code = textwrap.dedent("""
        #version 450

        layout(location = 0) in vec2 direction;
        layout(location = 0) out vec4 fragColor;

        void main()
        {
            float angle = atan(direction.y, direction.x);
            float slope = atan(direction.y);
            fragColor = vec4(angle, slope, 0.0, 1.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float angle = atan2(input.direction.y, input.direction.x);" in crossgl
    assert "float slope = atan(input.direction.y);" in crossgl
    assert "atan(input.direction.y, input.direction.x)" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "float angle = atan(direction.y, direction.x);" in glsl
    assert "float slope = atan(direction.y);" in glsl


def test_codegen_user_defined_two_argument_atan_is_preserved():
    code = textwrap.dedent("""
        #version 450

        layout(location = 0) in vec2 direction;
        layout(location = 0) out vec4 fragColor;

        float atan(float y, float x)
        {
            return y + x;
        }

        void main()
        {
            float angle = atan(direction.y, direction.x);
            fragColor = vec4(angle, 0.0, 0.0, 1.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float atan(float y, float x)" in crossgl
    assert "float angle = atan(input.direction.y, input.direction.x);" in crossgl
    assert "atan2(" not in crossgl


def test_codegen_user_defined_one_argument_atan_does_not_shadow_builtin_two_argument_atan():
    code = textwrap.dedent("""
        #version 450

        layout(location = 0) in vec2 direction;
        layout(location = 0) out vec4 fragColor;

        float atan(float y)
        {
            return y + 1.0;
        }

        void main()
        {
            float custom = atan(direction.y);
            float angle = atan(direction.y, direction.x);
            fragColor = vec4(angle, custom, 0.0, 1.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float atan(float y)" in crossgl
    assert "float custom = atan(input.direction.y);" in crossgl
    assert "float angle = atan2(input.direction.y, input.direction.x);" in crossgl


def test_codegen_inversesqrt_from_khronos_spec_preserves_glsl_spelling():
    # Khronos GLSL 4.60 specifies the builtin as inversesqrt(x). Keeping that
    # spelling lets CrossGL reparse and regenerate valid native GLSL.
    code = textwrap.dedent("""
        #version 450

        layout(location = 0) out vec4 fragColor;

        void main()
        {
            float scalarInv = inversesqrt(4.0);
            vec3 vectorInv = inversesqrt(vec3(4.0, 9.0, 16.0));
            fragColor = vec4(vectorInv * scalarInv, 1.0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float scalarInv = inversesqrt(4.0);" in crossgl
    assert "vec3 vectorInv = inversesqrt(vec3(4.0, 9.0, 16.0));" in crossgl
    assert "inverseSqrt(" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "float scalarInv = inversesqrt(4.0);" in glsl
    assert "vec3 vectorInv = inversesqrt(vec3(4.0, 9.0, 16.0));" in glsl
    assert "inverseSqrt(" not in glsl


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
    assert converter.resource_function_descriptor("texture2DLod") == {
        "name": "texture2DLod",
        "function": "textureLod",
        "resource": "texture",
        "operation": "sample_lod",
    }
    assert converter.resource_function_descriptor("textureCubeLod") == {
        "name": "textureCubeLod",
        "function": "textureLod",
        "resource": "texture",
        "operation": "sample_lod",
    }
    assert converter.resource_function_descriptor("texture2DGrad") == {
        "name": "texture2DGrad",
        "function": "textureGrad",
        "resource": "texture",
        "operation": "sample_grad",
    }
    assert converter.resource_function_descriptor("texture2DLodOffset") == {
        "name": "texture2DLodOffset",
        "function": "textureLodOffset",
        "resource": "texture",
        "operation": "sample_lod",
    }
    assert converter.resource_function_descriptor("texture2DProj") == {
        "name": "texture2DProj",
        "function": "textureProj",
        "resource": "texture",
        "operation": "sample_projected",
    }
    assert converter.resource_function_descriptor("textureProjOffset") == {
        "name": "textureProjOffset",
        "function": "textureProjOffset",
        "resource": "texture",
        "operation": "sample_projected",
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


def test_codegen_gles_precision_qualified_sampler_roundtrip():
    # Reduced from common ES texture shader idioms: precision defaults,
    # precision-qualified sampler uniforms, and texture() sampling.
    code = textwrap.dedent("""
        #version 300 es
        precision mediump float;
        precision lowp sampler2D;

        layout(location = 0) in highp vec2 coord;
        layout(location = 0) out lowp vec4 fragColor;
        uniform highp sampler2D texSampler;

        void main()
        {
            lowp vec4 col = texture(texSampler, coord);
            fragColor = col;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "sampler2D texSampler @highp;" in crossgl
    assert "vec2 coord @location(0) @highp;" in crossgl
    assert "vec4 col @lowp = texture(texSampler, input.coord);" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "precision lowp sampler2D;" in glsl
    assert "layout(binding = 0) uniform highp sampler2D texSampler;" in glsl
    assert "layout(location = 0) in highp vec2 coord;" in glsl
    assert "lowp vec4 col = texture(texSampler, coord);" in glsl


def test_codegen_native_shadow_texture_imports_compare_helpers():
    code = textwrap.dedent("""
        #version 460 core
        layout(location = 0) in vec2 uv;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2DShadow shadowMap;

        void main() {
            float cmp = texture(shadowMap, vec3(uv, 0.5));
            vec3 uvz = vec3(uv, 0.75);
            float lodOffset = textureLodOffset(
                shadowMap,
                uvz,
                0.0,
                ivec2(1, -1)
            );
            fragColor = vec4(cmp + lodOffset);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float cmp = textureCompare(shadowMap, input.uv, 0.5);" in crossgl
    assert (
        "float lodOffset = textureCompareLodOffset("
        "shadowMap, uvz.xy, uvz.z, 0.0, ivec2(1, (-1)));"
    ) in crossgl
    assert "texture(shadowMap" not in crossgl
    assert "textureLodOffset(shadowMap" not in crossgl

    shader_ast = parse_crossgl(crossgl)

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "shadowMap.SampleCmp(shadowMapSampler, input.uv, 0.5)" in hlsl
    assert (
        "shadowMap.SampleCmpLevel(" "shadowMapSampler, uvz.xy, uvz.z, 0.0, int2(1, -1))"
    ) in hlsl
    assert "shadowMap.Sample(" not in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "shadowMap.sample_compare" in metal
    assert "shadowMap.sample(" not in metal

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "texture(shadowMap, vec3(uv, 0.5))" in glsl
    assert (
        "textureLodOffset(shadowMap, vec3(uvz.xy, uvz.z), 0.0, ivec2(1, (-1)))" in glsl
    )
    assert "textureCompare(" not in glsl
    assert "textureCompareLodOffset(" not in glsl


def test_codegen_native_projected_shadow_texture_imports_compare_helpers():
    # Vulkan GLSL built-ins expose projected shadow comparisons through
    # textureProj* overloads on sampler2DShadow with the reference packed in P.z.
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec4 uvRefQ;
        layout(location = 1) in vec2 ddx;
        layout(location = 2) in vec2 ddy;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2DShadow shadowMap;

        void main() {
            const ivec2 offset = ivec2(1, -1);
            float projected = textureProj(shadowMap, uvRefQ);
            float projectedOffset = textureProjOffset(shadowMap, uvRefQ, offset);
            float projectedLod = textureProjLod(shadowMap, uvRefQ, 0.0);
            float projectedLodOffset = textureProjLodOffset(
                shadowMap,
                uvRefQ,
                0.0,
                offset
            );
            float projectedGrad = textureProjGrad(shadowMap, uvRefQ, ddx, ddy);
            float projectedGradOffset = textureProjGradOffset(
                shadowMap,
                uvRefQ,
                ddx,
                ddy,
                offset
            );
            fragColor = vec4(
                projected
                + projectedOffset
                + projectedLod
                + projectedLodOffset
                + projectedGrad
                + projectedGradOffset
            );
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "textureCompareProj(shadowMap, input.uvRefQ.xyw, input.uvRefQ.z)" in crossgl
    assert (
        "textureCompareProjOffset(shadowMap, input.uvRefQ.xyw, input.uvRefQ.z, offset)"
        in crossgl
    )
    assert (
        "textureCompareProjLod(shadowMap, input.uvRefQ.xyw, input.uvRefQ.z, 0.0)"
        in crossgl
    )
    assert (
        "textureCompareProjLodOffset(shadowMap, input.uvRefQ.xyw, input.uvRefQ.z, 0.0, offset)"
        in crossgl
    )
    assert (
        "textureCompareProjGrad(shadowMap, input.uvRefQ.xyw, input.uvRefQ.z, input.ddx, input.ddy)"
        in crossgl
    )
    assert (
        "textureCompareProjGradOffset(shadowMap, input.uvRefQ.xyw, input.uvRefQ.z, input.ddx, input.ddy, offset)"
        in crossgl
    )
    assert "float projected = textureProj(" not in crossgl
    assert "float projectedLod = textureProjLod(" not in crossgl

    shader_ast = parse_crossgl(crossgl)

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "shadowMap.SampleCmp(" in hlsl
    assert "shadowMap.SampleCmpLevel(" in hlsl
    assert "shadowMap.SampleCmpGrad(" in hlsl
    assert "shadowMap.Sample(" not in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "shadowMap.sample_compare" in metal
    assert "shadowMap.sample(" not in metal

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "texture(shadowMap, vec3(uvRefQ.xyw.xy / uvRefQ.xyw.z, uvRefQ.z))" in glsl
    assert (
        "textureLodOffset(shadowMap, vec3(uvRefQ.xyw.xy / uvRefQ.xyw.z, uvRefQ.z), 0.0, offset)"
        in glsl
    )
    assert "textureCompareProj" not in glsl


def test_codegen_legacy_shadow2d_imports_compare_helpers_from_openmw_pcf():
    # Reduced from s-ilent/shadows_fragment.glsl, which samples a
    # sampler2DShadow through shadow2D(...).r for PCF taps. GLSL 1.40 also
    # deprecates the dimension-suffixed texture names in favor of texture().
    code = textwrap.dedent("""
        #version 120
        varying vec2 uv;
        uniform sampler2DShadow shadowMap;

        void main() {
            vec3 uvz = vec3(uv, 0.5);
            float base = shadow2D(shadowMap, uvz).r;
            float lod = shadow2DLod(shadowMap, vec3(uv, 0.75), 0.0).r;
            gl_FragColor = vec4(base + lod);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "float base = textureCompare(shadowMap, uvz.xy, uvz.z);" in crossgl
    assert "float lod = textureCompareLod(shadowMap, input.uv, 0.75, 0.0);" in crossgl
    assert "shadow2D(" not in crossgl
    assert "shadow2DLod(" not in crossgl
    assert "textureCompare(shadowMap, uvz.xy, uvz.z).r" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "float base = texture(shadowMap, vec3(uvz.xy, uvz.z));" in glsl
    assert "float lod = textureLod(shadowMap, vec3(uv, 0.75), 0.0);" in glsl
    assert "shadow2D(" not in glsl
    assert "shadow2DLod(" not in glsl
    assert "textureCompare(" not in glsl


def test_codegen_native_cube_array_shadow_texture_imports_separate_compare_reference():
    # GLSL 4.60.8 declares the cube-array shadow overload as
    # texture(samplerCubeArrayShadow, vec4 P, float compare), unlike the packed
    # coordinate/reference forms used by 2D and cube shadow samplers.
    code = textwrap.dedent("""
        #version 460 core
        layout(location = 0) in vec4 cubeLayer;
        layout(location = 0) out vec4 fragColor;
        uniform samplerCubeArrayShadow shadowCubeArray;

        void main() {
            float lit = texture(shadowCubeArray, cubeLayer, 0.5);
            fragColor = vec4(lit);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert (
        "float lit = textureCompare(shadowCubeArray, input.cubeLayer, 0.5);" in crossgl
    )
    assert "texture(shadowCubeArray" not in crossgl

    shader_ast = parse_crossgl(crossgl)

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert (
        "shadowCubeArray.SampleCmp(shadowCubeArraySampler, input.cubeLayer, 0.5)"
        in hlsl
    )
    assert "shadowCubeArray.Sample(" not in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert (
        "shadowCubeArray.sample_compare("
        "sampler(mag_filter::linear, min_filter::linear), "
        "input.cubeLayer.xyz, uint(input.cubeLayer.w), 0.5)" in metal
    )
    assert "shadowCubeArray.sample(" not in metal

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "texture(shadowCubeArray, cubeLayer, 0.5)" in glsl
    assert "textureCompare(" not in glsl


def test_codegen_legacy_lod_grad_texture_intrinsics_from_bgfx_examples():
    # Reduced from bkaradzic/bgfx@6e0d61bf examples that use texture2DLod,
    # textureCubeLod, texture2DGrad, and texture2DLodOffset.
    code = textwrap.dedent("""
        #version 130
        varying vec2 vUV;
        uniform sampler2D s_texColor;
        uniform samplerCube s_texCube;

        void main() {
            vec4 lod = texture2DLod(s_texColor, vUV, 0.0);
            vec4 cube = textureCubeLod(s_texCube, vec3(1.0), 1.0);
            vec4 grad = texture2DGrad(s_texColor, vUV, vec2(1.0), vec2(1.0));
            vec4 offset = texture2DLodOffset(s_texColor, vUV, 0.0, ivec2(1));
            gl_FragColor = lod + cube + grad + offset;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "textureLod(s_texColor, input.vUV, 0.0)" in crossgl
    assert "textureLod(s_texCube, vec3(1.0), 1.0)" in crossgl
    assert "textureGrad(s_texColor, input.vUV, vec2(1.0), vec2(1.0))" in crossgl
    assert "textureLodOffset(s_texColor, input.vUV, 0.0, ivec2(1))" in crossgl
    assert "texture2DLod(" not in crossgl
    assert "textureCubeLod(" not in crossgl
    assert "texture2DGrad(" not in crossgl
    assert "texture2DLodOffset(" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "textureLod(s_texColor, vUV, 0.0)" in glsl
    assert "textureLod(s_texCube, vec3(1.0), 1.0)" in glsl
    assert "textureGrad(s_texColor, vUV, vec2(1.0), vec2(1.0))" in glsl
    assert "textureLodOffset(s_texColor, vUV, 0.0, ivec2(1))" in glsl


def test_codegen_projected_texture_offset_from_glslang_non_const_offset():
    # Reduced from KhronosGroup/glslang Test/spv.textureoffset_non_const.vert.
    code = textwrap.dedent("""
        #version 450 core
        #extension GL_EXT_texture_offset_non_const : enable

        layout(location = 4) in vec2 a_in0;
        layout(location = 10) in ivec2 offsetValue;
        layout(location = 0) out vec4 v_color0;
        layout(binding = 0) uniform sampler2D u_sampler;
        layout(binding = 1) uniform texture2D u_texture;
        layout(binding = 2) uniform sampler u_linear;

        void main()
        {
            v_color0 = textureProjOffset(
                u_sampler, vec3(a_in0, 1.0), offsetValue);
            v_color0 += textureProjOffset(
                sampler2D(u_texture, u_linear), vec3(a_in0, 1.0), offsetValue);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert (
        "textureProjOffset(u_sampler, vec3(input.a_in0, 1.0), input.offsetValue)"
        in crossgl
    )
    assert (
        "textureProjOffset("
        "u_texture, u_linear, vec3(input.a_in0, 1.0), input.offsetValue)" in crossgl
    )
    assert "sampler2D(u_texture, u_linear)" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_vertex_clip_distance_builtin_from_sascha_willems_offscreen():
    # Reduced from SaschaWillems/Vulkan@180be3f9
    # shaders/glsl/offscreen/phong.vert.
    code = textwrap.dedent("""
        #version 450
        layout(location = 0) in vec3 inPos;

        void main()
        {
            vec4 clipPlane = vec4(0.0, 0.0, 0.0, 0.0);
            gl_Position = vec4(inPos, 1.0);
            gl_ClipDistance[0] = dot(vec4(inPos, 1.0), clipPlane);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "float gl_ClipDistance[] @ gl_ClipDistance;" in crossgl
    assert "output.gl_ClipDistance[0] = dot(" in crossgl
    assert not any(
        line.lstrip().startswith("gl_ClipDistance[0] =")
        for line in crossgl.splitlines()
    )


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


def test_codegen_double_float_suffix_literals_from_glslang_numeral_reparse():
    # Reduced from KhronosGroup/glslang Test/numeral.frag. GLSL 4.60.8
    # section 4.1.4 defines lf/LF as double-precision floating suffixes.
    code = textwrap.dedent("""
        #version 400
        layout(location = 0) out double outValue;

        void main() {
            double gf1 = 1.0lf;
            double gf2 = 2.Lf;
            double gf3 = .3e1lF;
            double gf4 = .4e1LF;
            outValue = gf1 + gf2 + gf3 + gf4;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "lf" not in crossgl.lower()
    assert "double gf1 = 1.0;" in crossgl
    assert "double gf2 = 2.;" in crossgl
    assert "double gf3 = .3e1;" in crossgl
    assert "double gf4 = .4e1;" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "double gf1 = 1.0;" in glsl
    assert "double gf2 = 2.0;" in glsl
    assert "double gf3 = 3.0;" in glsl
    assert "double gf4 = 4.0;" in glsl
    assert "outValue = (((gf1 + gf2) + gf3) + gf4);" in glsl
    assert "fragColor = outValue;" in glsl


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

    assert "texture2D Textures[] @set(0) @binding(0);" in crossgl
    assert "sampler ImmutableSampler @set(1) @binding(0);" in crossgl
    assert "cbuffer Uniforms" not in crossgl


def test_codegen_combined_sampler_constructor_from_glslang_register_autoassign():
    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/spv.glsl.register.autoassign.frag, which samples separate texture and
    # sampler resources through sampler1D(g_tTex1, g_sSamp1).
    code = textwrap.dedent("""
        #version 450

        uniform layout(binding = 0) sampler g_sSamp1;
        uniform layout(binding = 1) texture1D g_tTex1;
        out vec4 FragColor;

        void main()
        {
            FragColor = texture(sampler1D(g_tTex1, g_sSamp1), 0.1);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "sampler g_sSamp1 @binding(0);" in crossgl
    assert "texture1D g_tTex1 @binding(1);" in crossgl
    assert "texture(g_tTex1, g_sSamp1, 0.1)" in crossgl
    assert "sampler1D(g_tTex1, g_sSamp1)" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "texture(sampler1D(g_tTex1, g_sSamp1), 0.1)" in glsl

    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/spv.sampledImageBlock.frag, which fetches from sampler2D(tex0, samp0).
    fetch_code = textwrap.dedent("""
        #version 450

        layout(binding = 0) uniform texture2D tex0;
        layout(binding = 1) uniform sampler samp0;
        layout(location = 0) out vec4 FragColor;

        void main()
        {
            FragColor = texelFetch(sampler2D(tex0, samp0), ivec2(0), 0);
        }
    """).strip()

    fetch_crossgl = assert_roundtrip(fetch_code, "fragment", ShaderStage.FRAGMENT)

    assert "texture2D tex0 @binding(0);" in fetch_crossgl
    assert "sampler samp0 @binding(1);" in fetch_crossgl
    assert "texelFetch(tex0, ivec2(0), 0)" in fetch_crossgl
    assert "texelFetch(tex0, samp0" not in fetch_crossgl

    fetch_glsl = GLSLCodeGen().generate(parse_crossgl(fetch_crossgl))
    assert "texelFetch(tex0, ivec2(0), 0)" in fetch_glsl


def test_codegen_vulkan_descriptor_sets_preserved_on_resources():
    # Reduced from Khronos Vulkan GLSL examples that use descriptor
    # layout(set = <set-index>, binding = <binding-index>) on resources.
    code = textwrap.dedent("""
        #version 450
        #extension GL_KHR_vulkan_glsl : enable

        layout(set = 1, binding = 2) uniform sampler2D colorTex;
        layout(set = 3, binding = 7, r32ui) coherent readonly uniform uimage2D counters;
        layout(set = 4, binding = 8, rgba32f) writeonly uniform image2D outImage;

        layout(location = 0) out vec4 fragColor;

        void main() {
            fragColor = texture(colorTex, vec2(0.5)) + vec4(imageLoad(counters, ivec2(0)));
            imageStore(outImage, ivec2(0), fragColor);
        }
    """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "sampler2D colorTex @set(1) @binding(2);" in crossgl
    assert (
        "uimage2D counters @set(3) @binding(7) @r32ui @coherent @readonly;" in crossgl
    )
    assert "image2D outImage @set(4) @binding(8) @rgba32f @writeonly;" in crossgl
    assert "cbuffer Uniforms" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_external_yuv_sampler_uniforms_are_resources_from_glslang():
    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/300samplerExternalYUV.frag.
    code = textwrap.dedent("""
        #version 300 es
        #extension GL_EXT_YUV_target : enable

        uniform __samplerExternal2DY2YEXT sExt;
        uniform highp __samplerExternal2DY2YEXT highExt;
        layout(location = 0) out vec4 fragColor;

        void main() {
            fragColor = texture(sExt, vec2(0.2)) + texture(highExt, vec2(0.2));
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "__samplerExternal2DY2YEXT sExt;" in crossgl
    assert "__samplerExternal2DY2YEXT highExt @highp;" in crossgl
    assert "cbuffer Uniforms" not in crossgl
    assert "texture(sExt, vec2(0.2))" in crossgl
    assert "texture(highExt, vec2(0.2))" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "uniform __samplerExternal2DY2YEXT sExt;" in glsl
    assert "uniform highp __samplerExternal2DY2YEXT highExt;" in glsl
    assert "texture(sExt, vec2(0.2))" in glsl
    assert "texture(highExt, vec2(0.2))" in glsl


def test_codegen_external_oes_sampler_roundtrips_to_glsl():
    # Common in Android/OpenGL ES camera and video texture shaders.
    code = textwrap.dedent("""
        #version 300 es
        #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump float;

        uniform samplerExternalOES cameraTexture;
        layout(location = 0) in vec2 vTexCoord;
        layout(location = 0) out vec4 fragColor;

        void main() {
            fragColor = texture(cameraTexture, vTexCoord);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "samplerExternalOES cameraTexture;" in crossgl
    assert "cbuffer Uniforms" not in crossgl
    assert "texture(cameraTexture, input.vTexCoord)" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "#extension GL_OES_EGL_image_external_essl3 : require" in glsl
    assert "uniform samplerExternalOES cameraTexture;" in glsl
    assert "texture(cameraTexture, vTexCoord)" in glsl


def test_codegen_subroutine_metadata_from_khronos_shader_subroutine():
    code = textwrap.dedent("""
        #version 400 core
        subroutine vec4 ColorFunc();
        layout(index = 2) subroutine(ColorFunc) vec4 redColor()
        {
            return vec4(1.0, 0.0, 0.0, 1.0);
        }
        layout(location = 1) subroutine uniform ColorFunc materialColor;
        out vec4 outColor;

        void main()
        {
            outColor = materialColor();
        }
        """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "@subroutine vec4 ColorFunc()" in crossgl
    assert "@index(2) @subroutine(ColorFunc) vec4 redColor()" in crossgl
    assert "ColorFunc materialColor @location(1) @subroutine;" in crossgl
    assert "cbuffer Uniforms" not in crossgl
    assert "outColor = materialColor();" in crossgl
    parse_crossgl(crossgl)


def test_codegen_subroutine_type_list_with_newlines_from_khronos_syntax():
    # Khronos GLSL 4.60 subroutine examples use subroutine(typeName) as a
    # qualifier; newlines are whitespace between the qualifier tokens.
    code = textwrap.dedent("""
        #version 400 core
        subroutine vec4 ColorFunc(vec3 color);

        subroutine
        (
            ColorFunc
        )
        vec4 redColor(vec3 color)
        {
            return vec4(color.r, 0.0, 0.0, 1.0);
        }

        subroutine uniform ColorFunc materialColor;
        out vec4 outColor;

        void main()
        {
            outColor = materialColor(vec3(1.0));
        }
        """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "@subroutine vec4 ColorFunc(vec3 color)" in crossgl
    assert "@subroutine(ColorFunc) vec4 redColor(vec3 color)" in crossgl
    assert "ColorFunc materialColor @subroutine;" in crossgl
    assert "outColor = materialColor(vec3(1.0));" in crossgl
    parse_crossgl(crossgl)


def test_codegen_nonuniform_ext_qualifier_from_glslang_is_preserved():
    # Reduced from KhronosGroup/glslang Test/spv.nonuniform.frag.
    code = textwrap.dedent("""
        #version 450
        #extension GL_EXT_nonuniform_qualifier : enable

        layout(location=0) nonuniformEXT in vec4 nu_inv4;
        nonuniformEXT float nu_gf;
        layout(location=1) in nonuniformEXT flat int nu_ii;

        nonuniformEXT int foo(nonuniformEXT int nupi, nonuniformEXT out int f)
        {
            return nupi;
        }

        void main()
        {
            nonuniformEXT int nu_li;
            int a = foo(nu_li, nu_li);
            nu_li = nonuniformEXT(a) + nonuniformEXT(a * 2);
        }
    """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "vec4 nu_inv4 @location(0) @nonuniformEXT;" in crossgl
    assert "flat int nu_ii @location(1) @nonuniformEXT;" in crossgl
    assert "float nu_gf @nonuniformEXT;" in crossgl
    assert "int foo(int nupi @nonuniformEXT, out int f @nonuniformEXT)" in crossgl
    assert "int nu_li @nonuniformEXT;" in crossgl
    assert "nonuniformEXT(a)" in crossgl
    parse_crossgl(crossgl)


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

    assert (
        "subpassInput colorInput @set(0) @binding(0) @input_attachment_index(0);"
        in crossgl
    )
    assert (
        "usubpassInputMS idInput @set(0) @binding(1) @input_attachment_index(1);"
        in crossgl
    )
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


def test_codegen_type_only_atomic_uint_layout_default_from_glslang_spec_examples():
    # Reduced from KhronosGroup/glslang Test/specExamples.vert.
    code = textwrap.dedent("""
        #version 430

        layout (binding = 2, offset = 4) uniform atomic_uint;
        layout (binding = 2) uniform atomic_uint bar;

        void main()
        {
        }
    """).strip()

    crossgl = generate_crossgl(code, "vertex")

    assert "layout(binding = 2, offset = 4) uniform atomic_uint;" in crossgl


def test_codegen_for_condition_declaration_from_glslang_debuginfo_declaration():
    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/spv.debuginfo.declaration.glsl.frag.
    code = textwrap.dedent("""
        #version 460

        out vec4 outColor;

        void main() {
            int y = 0;
            for (int x = 50; bool test = x < 53; ) {
                y += x;
                x += 1;
            }
            outColor = vec4(y);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "for (int x = 50; ; )" in crossgl
    assert "bool test = (x < 53);" in crossgl
    assert "if (!test)" in crossgl
    assert "for (int x = 50; test; )" not in crossgl


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


def test_codegen_do_while_single_statement_from_glsl_spec_grammar():
    # Reduced from Khronos GLSL 4.60.8 grammar: DO statement WHILE (...).
    code = textwrap.dedent("""
        #version 460

        void main()
        {
            int i = 17;
            do
                int i = 4;
            while (i == 0);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "compute", ShaderStage.COMPUTE)

    assert "do {" in crossgl
    assert "int i = 4;" in crossgl
    assert "} while ((i == 0));" in crossgl


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


def test_codegen_invariant_builtin_redeclaration_from_glslang_150_vert():
    # Reduced from KhronosGroup/glslang@98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Test/150.vert.
    code = textwrap.dedent("""
        #version 150 core

        in vec4 iv4;

        invariant gl_Position;

        void main()
        {
            gl_Position = iv4;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "vec4 gl_Position @invariant @ gl_Position;" in crossgl
    assert " gl_Position @invariant;" not in crossgl
    assert "output.gl_Position = input.iv4;" in crossgl


def test_codegen_vertex_builtin_point_size_write_from_glslang_150_vert():
    # Reduced from KhronosGroup/glslang Test/150.vert, which writes gl_PointSize
    # without a separate redeclaration.
    code = textwrap.dedent("""
        #version 150 core

        in vec4 iv4;
        uniform float ps;

        void main()
        {
            gl_Position = iv4;
            gl_PointSize = ps;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "vertex", ShaderStage.VERTEX)

    assert "float gl_PointSize @ gl_PointSize;" in crossgl
    assert "output.gl_PointSize = ps;" in crossgl
    assert "\n        gl_PointSize = ps;" not in crossgl


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


def test_codegen_local_custom_type_array_declaration_from_glslang_struct_deref():
    code = textwrap.dedent("""
        #version 140

        uniform sampler2D samp2D;
        in vec2 coord;

        struct s0 {
            int i;
        };

        struct s1 {
            int i;
            float f;
            s0 s0_1;
        };

        s1 foo1;

        void main()
        {
            s1[10] locals1Array;
            locals1Array[6] = foo1;
            gl_FragColor = vec4(locals1Array[6].f) * texture(samp2D, coord);
        }
        """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "s1 locals1Array[10];" in crossgl
    assert "locals1Array[6] = foo1;" in crossgl
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


def test_codegen_arrayed_descriptor_uniform_block_preserves_instance_metadata():
    code = textwrap.dedent("""
        #version 450
        struct Foo { vec4 v; };
        layout(set = 2, binding = 4) uniform UBO { Foo foo; } ubos[2];
        layout(location = 0) out vec4 FragColor;

        void main() {
            FragColor = ubos[1].foo.v;
        }
        """).strip()

    crossgl = generate_crossgl(code, "fragment")

    assert "struct UBO" in crossgl
    assert "Foo foo;" in crossgl
    assert "uniform UBO ubos[2] @set(2) @binding(4);" in crossgl
    assert "cbuffer Uniforms" not in crossgl
    assert "ubos[1].foo.v" in crossgl

    shader_ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "uniform UBO {" in glsl
    assert "Foo foo;" in glsl
    assert "} ubos[2];" in glsl
    assert "FragColor = ubos[1].foo.v;" in glsl


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


def test_codegen_atomic_counter_memory_barrier_imports_as_crossgl_barrier():
    # Khronos GLSL 4.60 defines memoryBarrierAtomicCounter() as the atomic-counter
    # specific member of the same memory-barrier family covered by memoryBarrier().
    code = textwrap.dedent("""
        #version 450 core
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(binding = 0, offset = 0) uniform atomic_uint counter;

        void main() {
            uint value = atomicCounterIncrement(counter);
            memoryBarrierAtomicCounter();
            barrier();
            atomicCounterAdd(counter, value);
        }
    """).strip()

    crossgl = assert_roundtrip(code, "compute", ShaderStage.COMPUTE)

    assert "memoryBarrier();" in crossgl
    assert "memoryBarrierAtomicCounter(" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "AllMemoryBarrier();" in hlsl
    assert "memoryBarrierAtomicCounter(" not in hlsl


def test_codegen_block_preprocessor_injection_directive_from_godot_tex_blit():
    # Reduced from godotengine/godot@070dc9897ea1b84ab2a7ec04b9bc1b94f38a0eaf
    # servers/rendering/renderer_rd/shaders/tex_blit.glsl, which has a
    # Godot code-injection directive inside main().
    code = textwrap.dedent("""
        #version 450
        layout(location = 0) out vec4 out_color0;

        void main()
        {
            vec4 color0 = vec4(0.0);

        #CODE : BLIT

            color0 = vec4(1.0);
            out_color0 = color0;
        }
    """).strip()

    crossgl = assert_roundtrip(code, "fragment", ShaderStage.FRAGMENT)

    assert "#CODE" not in crossgl
    assert "vec4 color0 = vec4(0.0);" in crossgl
    assert "color0 = vec4(1.0);" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "#CODE" not in glsl
    assert "vec4 color0 = vec4(0.0);" in glsl
    assert "color0 = vec4(1.0);" in glsl


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
