import shutil
import subprocess
from types import SimpleNamespace

import pytest

import crosstl
import crosstl.translator.codegen as codegen
from crosstl.formatter import format_shader_code
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.test_backend.test_SPIRV.test_codegen import (
    SPIRV_TOOLS_GLPERVERTEX_ACCESS_CHAIN_ASSEMBLY,
)

WEBGL_SHADER = """
shader WebGLSmoke {
    vertex {
        vec4 main(vec3 position @ POSITION) @ gl_Position {
            return vec4(position, 1.0);
        }
    }
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
}
"""

WEBGL_MIXED_STAGE_SHADER = """
shader WebGLMixedStages {
    vertex {
        vec4 main(vec3 position @ POSITION) @ gl_Position {
            return vec4(position, 1.0);
        }
    }
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(1.0);
        }
    }
    compute {
        void main() {
            return;
        }
    }
}
"""


def parse_shader(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_webgl_backend_is_target_only():
    spec = codegen.get_backend("webgl2")

    assert spec is not None
    assert spec.name == "webgl"
    assert spec.source_registry_name is None
    assert "webgl" not in codegen.source_backend_names()
    assert codegen.normalize_backend_name("target.webgl.glsl") == "webgl"
    assert codegen.get_backend_extension("glsl-es") == ".webgl.glsl"
    assert isinstance(codegen.get_codegen("essl"), WebGLCodeGen)


def test_webgl_codegen_emits_glsl_es_header_and_default_precision():
    generated = WebGLCodeGen().generate(parse_shader(WEBGL_SHADER))

    assert generated.startswith("#version 300 es\n")
    assert generated.index("#version 300 es") < generated.index(
        "precision highp float;"
    )
    assert "precision highp float;\n" in generated
    assert "precision highp int;\n" in generated
    assert "#version 450 core" not in generated
    assert "layout(location = 0) out vec4 fragColor;" in generated


def test_webgl_codegen_preserves_explicit_precision_qualifiers():
    shader = """
    precision mediump float;
    precision lowp int;
    shader WebGLPrecision {
        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "precision mediump float;" in generated
    assert "precision lowp int;" in generated
    assert "precision highp float;" not in generated
    assert "precision highp int;" not in generated


def test_webgl_codegen_omits_es300_binding_layouts_for_resources():
    shader = """
    shader WebGLResourceBindings {
        sampler2D colorTex @binding(2);
        cbuffer MaterialData @binding(1) {
            float tint;
        };
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return texture(colorTex, uv) * tint;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "binding =" not in generated
    assert "uniform sampler2D colorTex;" in generated
    assert "layout(std140) uniform MaterialData" in generated


def test_webgl_codegen_emits_stage_local_cbuffers():
    shader = """
    shader WebGLStageLocalCBuffer {
        vertex {
            cbuffer LocalParams {
                float scale;
            };
            vec4 main(vec3 position @ POSITION) @ gl_Position {
                return vec4(position * scale, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "layout(std140) uniform LocalParams" in generated
    assert "float scale;" in generated
    assert "(position * scale)" in generated


def test_webgl_codegen_maps_separate_texture_resources_to_samplers():
    shader = """
    shader WebGLSeparateTexture {
        texture2D colorTex;
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return texture(colorTex, uv);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "uniform sampler2D colorTex;" in generated
    assert "uniform texture2D colorTex;" not in generated


@pytest.mark.parametrize("qualifier", ("coherent", "volatile", "restrict"))
def test_webgl_codegen_rejects_sampler_memory_qualifiers(qualifier):
    shader = f"""
    shader WebGLSamplerMemoryQualifier {{
        {qualifier} sampler2D colorTex;

        fragment {{
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {{
                return texture(colorTex, uv);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support resource memory qualifier\\(s\\) "
            rf"'{qualifier}' on sampled resource 'colorTex'"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_omits_fragment_input_location_layouts():
    shader = """
    shader WebGLFragmentInput {
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return vec4(uv, 0.0, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "in vec2 uv;" in generated
    assert "layout(location = 5) in vec2 uv;" not in generated


def test_webgl_codegen_omits_vertex_output_location_layouts():
    shader = """
    shader WebGLVertexOutput {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate_program(
        parse_shader(shader),
        target_stage="vertex",
    )

    assert "layout(location = 5) out vec2 out_uv;" not in generated
    assert "out vec2 out_uv;" in generated


@pytest.mark.parametrize("qualifier", ("noperspective", "sample"))
def test_webgl_codegen_rejects_desktop_interpolation_qualifiers(qualifier):
    shader = f"""
    shader WebGLBadInterpolation {{
        struct VSInput {{
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        }};
        struct VSOutput {{
            vec4 position @ gl_Position;
            vec2 uv @{qualifier} @TEXCOORD0;
        }};
        vertex {{
            VSOutput main(VSInput input) {{
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support interpolation qualifier "
            rf"'{qualifier}' on 'uv'"
        ),
    ):
        WebGLCodeGen().generate_program(
            parse_shader(shader),
            target_stage="vertex",
        )


def test_webgl_codegen_aligns_split_stage_varying_names_without_locations():
    shader = """
    shader WebGLLinkedVaryings {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };
        vertex {
            VSOutput main(VSInput input) {
                return VSOutput(vec4(input.position, 1.0), input.uv);
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """
    ast = parse_shader(shader)
    generator = WebGLCodeGen()

    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")
    combined_code = WebGLCodeGen().generate(ast)

    assert "out vec2 out_uv;" in vertex_code
    assert "in vec2 out_uv;" in fragment_code
    assert "fragColor = vec4(out_uv, 0.0, 1.0);" in fragment_code
    assert "layout(location = 5) out vec2 out_uv;" not in vertex_code
    assert "layout(location = 5) in vec2 out_uv;" not in fragment_code
    assert "out vec2 out_uv;" in combined_code
    assert "in vec2 out_uv;" in combined_code
    assert "in_out_uv" not in combined_code


def test_webgl_codegen_stage_builtin_parameter_aliases_to_glsl_builtin():
    shader = """
    shader WebGLBuiltinParam {
        fragment {
            vec4 main(vec4 coord @ gl_FragCoord) @ gl_FragColor {
                return coord;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "in vec4 coord;" not in generated
    assert "fragColor = gl_FragCoord;" in generated


def test_webgl_codegen_stage_builtin_parameter_semantics_validate_type():
    shader = """
    shader WebGLBadBuiltinParam {
        fragment {
            vec4 main(float coord @ gl_FragCoord) @ gl_FragColor {
                return vec4(coord);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="gl_FragCoord parameter 'coord' must be vec4",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_casts_texture_size_for_float_vector_arithmetic():
    shader = """
    shader WebGLTextureSizeCast {
        sampler2D shadowMap;
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
                return vec4(texelSize, 0.0, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "vec2 texelSize = (1.0 / vec2(textureSize(shadowMap, 0)));" in generated


def test_webgl_codegen_lowers_projected_texture_sampling():
    shader = """
    shader WebGLProjectedTexture {
        sampler2D colorTex;

        fragment {
            vec4 main(vec3 uvq @ TEXCOORD0) @ gl_FragColor {
                return textureProj(colorTex, uvq);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "uniform sampler2D colorTex;" in generated
    assert "fragColor = textureProj(colorTex, uvq);" in generated


def test_webgl_codegen_lowers_texel_fetch():
    shader = """
    shader WebGLTexelFetch {
        sampler2D colorTex;

        fragment {
            vec4 main(ivec2 pixel @ TEXCOORD0) @ gl_FragColor {
                return texelFetch(colorTex, pixel, 0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "flat in ivec2 pixel;" in generated
    assert "fragColor = texelFetch(colorTex, pixel, 0);" in generated


def test_webgl_codegen_lowers_dynamic_sampler_array_helper_call():
    shader = """
    shader WebGLDynamicSamplerArray {
        const int MAP_COUNT = 2;

        float sampleShadow(sampler2D shadowMap, vec2 uv) {
            return texture(shadowMap, uv).r;
        }

        fragment {
            uniform sampler2D shadowMaps[MAP_COUNT];
            uniform int shadowIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                float shadow = 0.0;
                shadow = sampleShadow(shadowMaps[shadowIndex], uv);
                return vec4(shadow);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "sampleShadow(shadowMaps[shadowIndex]" not in generated
    assert "float shadow = 0.0;" in generated
    assert "switch (shadowIndex)" in generated
    assert "case 0:" in generated
    assert "shadow = sampleShadow(shadowMaps[0], uv);" in generated
    assert "case 1:" in generated
    assert "shadow = sampleShadow(shadowMaps[1], uv);" in generated
    assert "default:" in generated
    assert "shadow = 0.0;" in generated


def test_webgl_codegen_lowers_nested_dynamic_sampler_array_helper_call():
    shader = """
    shader WebGLNestedDynamicSamplerArray {
        const int MAP_COUNT = 2;

        vec4 sampleColor(sampler2D colorMap, vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                vec4 color = sampleColor(colorMaps[colorIndex], uv) + vec4(0.25);
                return color;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "sampleColor(colorMaps[colorIndex]" not in generated
    assert "vec4 crossgl_dynamic_sampler_value;" in generated
    assert "switch (colorIndex)" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[0], uv);" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[1], uv);" in generated
    assert "vec4 color;" in generated
    assert "color = (crossgl_dynamic_sampler_value + vec4(0.25));" in generated


def test_webgl_codegen_lowers_nested_dynamic_sampler_array_assignment():
    shader = """
    shader WebGLNestedDynamicSamplerAssignment {
        const int MAP_COUNT = 2;

        vec4 sampleColor(sampler2D colorMap, vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                vec4 color = vec4(0.0);
                color = sampleColor(colorMaps[colorIndex], uv) * 0.5;
                return color;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "sampleColor(colorMaps[colorIndex]" not in generated
    assert "switch (colorIndex)" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[0], uv);" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[1], uv);" in generated
    assert "color = (crossgl_dynamic_sampler_value * 0.5);" in generated


def test_webgl_codegen_lowers_direct_dynamic_sampler_array_texture_return():
    shader = """
    shader WebGLDirectDynamicSamplerReturn {
        const int MAP_COUNT = 2;

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return texture(colorMaps[colorIndex], uv);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "switch (colorIndex)" in generated
    assert "fragColor = texture(colorMaps[0], uv);" in generated
    assert "fragColor = texture(colorMaps[1], uv);" in generated
    assert "fragColor = vec4(0.0);" in generated
    assert "return;" in generated


def test_webgl_codegen_lowers_nested_direct_dynamic_sampler_array_texture_call():
    shader = """
    shader WebGLNestedDirectDynamicSampler {
        const int MAP_COUNT = 2;

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                vec4 color = texture(colorMaps[colorIndex], uv) + vec4(0.25);
                return color;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "vec4 crossgl_dynamic_sampler_value;" in generated
    assert "switch (colorIndex)" in generated
    assert "crossgl_dynamic_sampler_value = texture(colorMaps[0], uv);" in generated
    assert "crossgl_dynamic_sampler_value = texture(colorMaps[1], uv);" in generated
    assert "vec4 color;" in generated
    assert "color = (crossgl_dynamic_sampler_value + vec4(0.25));" in generated


def test_webgl_codegen_dynamic_sampler_array_ternary_return_keeps_diagnostic():
    shader = """
    shader WebGLDynamicSamplerTernary {
        const int MAP_COUNT = 2;

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return colorIndex > 0 ? texture(colorMaps[colorIndex], uv) : vec4(1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "switch (colorIndex)" not in generated
    assert (
        "unsupported WebGL dynamic sampler array expression: dynamic sampler arrays "
        "cannot be lifted from ternary or short-circuit expressions"
    ) in generated
    assert (
        "fragColor = /* unsupported WebGL dynamic sampler array expression:"
        in generated
    )
    assert "fragColor = vec4(0.0);" not in generated
    assert "return;" in generated


def test_webgl_codegen_lowers_direct_dynamic_shadow_compare_sampler_array():
    shader = """
    shader WebGLDynamicShadowCompareArray {
        const int MAP_COUNT = 2;

        fragment {
            uniform sampler2DShadow shadowMaps[MAP_COUNT];
            uniform int shadowIndex;

            vec4 main(vec2 uv @ TEXCOORD0, float depth @ TEXCOORD1) @ gl_FragColor {
                float shadow = textureCompare(shadowMaps[shadowIndex], uv, depth);
                return vec4(shadow);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "textureCompare(shadowMaps[shadowIndex]" not in generated
    assert "textureCompare(shadowMaps[0]" not in generated
    assert "switch (shadowIndex)" in generated
    assert "shadow = texture(shadowMaps[0], vec3(uv, depth));" in generated
    assert "shadow = texture(shadowMaps[1], vec3(uv, depth));" in generated
    assert "shadow = 0.0;" in generated


def test_webgl_codegen_dynamic_sampler_array_texture_offset_keeps_diagnostic():
    shader = """
    shader WebGLDynamicTextureOffsetArray {
        const int MAP_COUNT = 2;

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;
            uniform ivec2 dynamicOffset;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return textureOffset(colorMaps[colorIndex], uv, dynamicOffset);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "textureOffset(colorMaps[colorIndex]" not in generated
    assert "textureOffset(colorMaps[0]" not in generated
    assert "switch (colorIndex)" in generated
    assert (
        "unsupported GLSL texture offset: textureOffset texel offsets must be "
        "compile-time integer constants"
    ) in generated
    assert "fragColor = vec4(0.0);" in generated


def test_webgl_codegen_lowers_dynamic_sampler_array_stage_return():
    shader = """
    shader WebGLDynamicSamplerReturn {
        const int MAP_COUNT = 2;

        vec4 sampleColor(sampler2D colorMap, vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return sampleColor(colorMaps[colorIndex], uv);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "return sampleColor(colorMaps[colorIndex]" not in generated
    assert "switch (colorIndex)" in generated
    assert "fragColor = sampleColor(colorMaps[0], uv);" in generated
    assert "fragColor = sampleColor(colorMaps[1], uv);" in generated
    assert "fragColor = vec4(0.0);" in generated
    assert "return;" in generated


def test_webgl_codegen_lowers_nested_dynamic_sampler_array_stage_return():
    shader = """
    shader WebGLNestedDynamicSamplerReturn {
        const int MAP_COUNT = 2;

        vec4 sampleColor(sampler2D colorMap, vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            uniform sampler2D colorMaps[MAP_COUNT];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return sampleColor(colorMaps[colorIndex], uv) + vec4(0.25);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "sampleColor(colorMaps[colorIndex]" not in generated
    assert "vec4 crossgl_dynamic_sampler_value;" in generated
    assert "switch (colorIndex)" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[0], uv);" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[1], uv);" in generated
    assert "fragColor = (crossgl_dynamic_sampler_value + vec4(0.25));" in generated
    assert "return (crossgl_dynamic_sampler_value" not in generated


def test_webgl_codegen_lowers_dynamic_sampler_array_if_condition():
    shader = """
    shader WebGLDynamicSamplerArrayIfCondition {
        vec4 sampleColor(sampler2D colorMap, vec2 uv) {
            return texture(colorMap, uv);
        }

        fragment {
            uniform sampler2D colorMaps[2];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                if (sampleColor(colorMaps[colorIndex], uv).r > 0.5) {
                    return vec4(1.0);
                }
                return vec4(0.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "sampleColor(colorMaps[colorIndex]" not in generated
    assert "vec4 crossgl_dynamic_sampler_value;" in generated
    assert "switch (colorIndex)" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[0], uv);" in generated
    assert "crossgl_dynamic_sampler_value = sampleColor(colorMaps[1], uv);" in generated
    assert "if ((crossgl_dynamic_sampler_value.r > 0.5)) {" in generated


def test_webgl_codegen_lowers_direct_dynamic_sampler_array_if_condition():
    shader = """
    shader WebGLDirectDynamicSamplerArrayIfCondition {
        fragment {
            uniform sampler2D colorMaps[2];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                if (texture(colorMaps[colorIndex], uv).r > 0.5) {
                    return vec4(1.0);
                }
                return vec4(0.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "vec4 crossgl_dynamic_sampler_value;" in generated
    assert "switch (colorIndex)" in generated
    assert "crossgl_dynamic_sampler_value = texture(colorMaps[0], uv);" in generated
    assert "crossgl_dynamic_sampler_value = texture(colorMaps[1], uv);" in generated
    assert "if ((crossgl_dynamic_sampler_value.r > 0.5)) {" in generated


def test_webgl_codegen_lowers_dynamic_sampler_array_while_condition():
    shader = """
    shader WebGLDynamicSamplerArrayWhileCondition {
        fragment {
            uniform sampler2D colorMaps[2];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                float threshold = 0.25;
                while (texture(colorMaps[colorIndex], uv).r > threshold) {
                    threshold = threshold + 0.25;
                }
                return vec4(threshold);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "while (true) {" in generated
    assert "switch (colorIndex)" in generated
    assert "if (!((crossgl_dynamic_sampler_value.r > threshold))) {" in generated
    assert "break;" in generated


def test_webgl_codegen_lowers_dynamic_sampler_array_for_condition():
    shader = """
    shader WebGLDynamicSamplerArrayForCondition {
        fragment {
            uniform sampler2D colorMaps[2];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                float total = 0.0;
                for (int i = 0; texture(colorMaps[colorIndex], uv).r > 0.5; i = i + 1) {
                    total = total + float(i);
                }
                return vec4(total);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "for (int i = 0; ; i = (i + 1)) {" in generated
    assert "switch (colorIndex)" in generated
    assert "if (!((crossgl_dynamic_sampler_value.r > 0.5))) {" in generated
    assert "break;" in generated


def test_webgl_codegen_lowers_dynamic_sampler_array_do_while_condition():
    shader = """
    shader WebGLDynamicSamplerArrayDoWhileCondition {
        fragment {
            uniform sampler2D colorMaps[2];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                vec4 color = vec4(0.0);
                do {
                    color = color + vec4(0.25);
                } while (texture(colorMaps[colorIndex], uv).r > 0.5);
                return color;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "texture(colorMaps[colorIndex]" not in generated
    assert "while (true) {" in generated
    assert "color = (color + vec4(0.25));" in generated
    assert "switch (colorIndex)" in generated
    assert "if (!((crossgl_dynamic_sampler_value.r > 0.5))) {" in generated
    assert "break;" in generated


def test_webgl_codegen_rejects_dynamic_sampler_array_do_while_condition_with_continue():
    shader = """
    shader WebGLDynamicSamplerArrayDoWhileContinue {
        fragment {
            uniform sampler2D colorMaps[2];
            uniform int colorIndex;

            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                do {
                    continue;
                } while (texture(colorMaps[colorIndex], uv).r > 0.5);
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="dynamic sampler array do-while condition",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_aliases_format_as_glsl():
    assert format_shader_code("void main(){}", "webgl") == format_shader_code(
        "void main(){}", "glsl"
    )
    assert format_shader_code("void main(){}", "shader.webgl.glsl")


def test_translate_crossgl_to_webgl(tmp_path):
    source_path = tmp_path / "shader.cgl"
    source_path.write_text(WEBGL_SHADER, encoding="utf-8")

    generated = crosstl.translate(
        str(source_path), backend="webgl", format_output=False
    )

    assert "#version 300 es" in generated
    assert "precision highp float;" in generated


def test_webgl_array_declarations_and_access_emit_glsl_es_syntax():
    shader = """
    shader WebGLArrays {
        vertex {
            vec4 main(vec3 position @ POSITION) @ gl_Position {
                float weights[3];
                weights[0] = 0.25;
                weights[1] = 0.50;
                weights[2] = 1.00;
                float offset = weights[1] + weights[2];
                return vec4(position.x + offset, position.y, position.z, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "#version 300 es" in generated
    assert "float weights[3];" in generated
    assert "weights[0] = 0.25;" in generated
    assert "float offset = (weights[1] + weights[2]);" in generated


def test_webgl_bitwise_operations_emit_glsl_es_integer_operators():
    shader = """
    shader WebGLBitwise {
        fragment {
            vec4 main() @ gl_FragColor {
                int flags = 3;
                int masked = (flags & 1) | (flags ^ 2);
                int shifted = masked << 1;
                return vec4(float(shifted), 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "int masked = ((flags & 1) | (flags ^ 2));" in generated
    assert "int shifted = (masked << 1);" in generated
    assert "fragColor = vec4(float(shifted), 0.0, 0.0, 1.0);" in generated


def test_webgl_control_flow_emits_if_for_while_and_continue():
    shader = """
    shader WebGLControlFlow {
        fragment {
            vec4 main() @ gl_FragColor {
                float total = 0.0;
                for (int i = 0; i < 3; i = i + 1) {
                    if (i == 1) {
                        continue;
                    }
                    total = total + float(i);
                }
                while (total < 3.0) {
                    total = total + 1.0;
                }
                return vec4(total, 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "for (int i = 0; (i < 3); i = (i + 1)) {" in generated
    assert "if ((i == 1)) {" in generated
    assert "continue;" in generated
    assert "while ((total < 3.0)) {" in generated


def test_webgl_match_literal_and_wildcard_lowers_to_switch():
    shader = """
    shader WebGLMatch {
        fragment {
            vec4 main() @ gl_FragColor {
                int mode = 1;
                int value = 0;
                match mode {
                    0 => { value = 1; }
                    1 => { value = 2; }
                    _ => { value = 3; }
                }
                return vec4(float(value), 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "switch (mode)" in generated
    assert "case 0:" in generated
    assert "case 1:" in generated
    assert "default:" in generated
    assert "fragColor = vec4(float(value), 0.0, 0.0, 1.0);" in generated


def test_webgl_function_declarations_and_calls_emit_glsl_es_helpers():
    shader = """
    shader WebGLFunctions {
        float brighten(float value) {
            return value + 0.25;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                float value = brighten(0.5);
                return vec4(value, value, value, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "float brighten(float value) {" in generated
    assert "return (value + 0.25);" in generated
    assert "float value = brighten(0.5);" in generated


def test_webgl_struct_declarations_and_construction_emit_glsl_es_members():
    shader = """
    shader WebGLStructs {
        struct Material {
            vec3 albedo;
            float roughness;
        };

        fragment {
            vec4 main() @ gl_FragColor {
                Material material;
                material.albedo = vec3(0.1, 0.2, 0.3);
                material.roughness = 0.5;
                return vec4(material.albedo * material.roughness, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "struct Material {" in generated
    assert "vec3 albedo;" in generated
    assert "Material material;" in generated
    assert "material.albedo = vec3(0.1, 0.2, 0.3);" in generated
    assert "fragColor = vec4((material.albedo * material.roughness), 1.0);" in generated


def test_webgl_vector_and_matrix_expressions_emit_glsl_es_arithmetic():
    shader = """
    shader WebGLVectorMatrix {
        vertex {
            vec4 main(vec3 position @ POSITION) @ gl_Position {
                mat4 transform = mat4(1.0);
                vec4 local = vec4(position, 1.0);
                vec4 projected = transform * local;
                return projected + vec4(0.0, 0.0, 0.0, 0.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "mat4 transform = mat4(1.0);" in generated
    assert "vec4 local = vec4(position, 1.0);" in generated
    assert "vec4 projected = (transform * local);" in generated
    assert "gl_Position = (projected + vec4(0.0, 0.0, 0.0, 0.0));" in generated


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "return_value", "expected"),
    (
        ("vertex", "float", "gl_Position", "1.0", "must be vec4"),
        ("fragment", "vec4", "gl_FragDepth", "vec4(1.0)", "must be scalar float"),
    ),
)
def test_webgl_codegen_return_semantics_validate_builtin_types(
    stage,
    return_type,
    semantic,
    return_value,
    expected,
):
    shader = f"""
    shader WebGLBadReturnSemantic {{
        {stage} {{
            {return_type} main() @ {semantic} {{
                return {return_value};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=rf"WebGL {stage} stage function 'main' return semantic '{semantic}' {expected}",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_struct_semantics_validate_builtin_types():
    shader = """
    shader WebGLBadStructSemantic {
        struct VSOutput {
            float position @ gl_Position;
        };

        vertex {
            VSOutput main() {
                VSOutput output;
                output.position = 1.0;
                return output;
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL vertex stage struct 'VSOutput' member 'position' "
            "semantic 'gl_Position' must be vec4"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_non_webgl_stages():
    shader = """
    shader WebGLNoCompute {
        compute {
            void main() {
                return;
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="WebGL target does not support shader stage\\(s\\): compute",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_filters_unsupported_stages_when_graphics_stages_exist():
    generated = WebGLCodeGen().generate(parse_shader(WEBGL_MIXED_STAGE_SHADER))

    assert "#version 300 es" in generated
    assert "// Vertex Shader" in generated
    assert "// Fragment Shader" in generated
    assert "// Compute Shader" not in generated


def test_webgl_codegen_explicit_graphics_stage_ignores_unsupported_siblings():
    generated = WebGLCodeGen().generate_program(
        parse_shader(WEBGL_MIXED_STAGE_SHADER),
        target_stage="fragment",
    )

    assert "#version 300 es" in generated
    assert "// Fragment Shader" in generated
    assert "// Vertex Shader" not in generated
    assert "// Compute Shader" not in generated


@pytest.mark.parametrize("buffer_type", ("StructuredBuffer", "RWStructuredBuffer"))
def test_webgl_codegen_rejects_storage_buffer_resources(buffer_type):
    shader = f"""
    shader WebGLNoStorageBuffer {{
        {buffer_type}<int> values;

        fragment {{
            vec4 main() @ gl_FragColor {{
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support storage buffer resource "
            rf"'values' \({buffer_type}\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_glsl_buffer_blocks():
    shader = """
    shader WebGLNoGlslBufferBlock {
        layout(std430, binding = 0) buffer DataBlock {
            float values[];
        } data;

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="WebGL target does not support GLSL buffer block resource 'data'",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_storage_image_resources():
    shader = """
    shader WebGLNoStorageImage {
        image2D target;

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support storage image resource "
            r"'target' \(image2D\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_atomic_counter_resources():
    shader = """
    shader WebGLNoAtomicCounter {
        atomic_uint counter;

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support atomic counter resource "
            r"'counter' \(atomic_uint\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "member_type,resource_kind,diagnostic_type",
    (
        ("sampler2D", "sampled", "sampler2D"),
        ("texture2D", "sampled", "sampler2D"),
        ("image2D", "storage image", "image2D"),
        ("atomic_uint", "atomic counter", "atomic_uint"),
    ),
)
def test_webgl_codegen_rejects_opaque_resource_members_in_cbuffers(
    member_type,
    resource_kind,
    diagnostic_type,
):
    shader = f"""
    shader WebGLNoOpaqueCBufferMember {{
        cbuffer MaterialData {{
            {member_type} resourceMember;
            float factor;
        }};

        fragment {{
            vec4 main() @ gl_FragColor {{
                return vec4(factor);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"WebGL target does not support {resource_kind} resource member "
            rf"'resourceMember' in constant buffer 'MaterialData' "
            rf"\({diagnostic_type}\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_opaque_resource_members_in_constant_buffers():
    shader = """
    shader WebGLNoOpaqueConstantBufferMember {
        struct MaterialData {
            sampler2D colorTex;
            float factor;
        };
        ConstantBuffer<MaterialData> material @binding(0);

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support sampled resource member "
            r"'colorTex' in constant buffer 'material' \(sampler2D\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_opaque_resource_members_in_uniform_blocks():
    shader = """
    shader WebGLNoOpaqueUniformBlockMember {
        struct MaterialData {
            texture2D colorTex;
            vec4 tint;
        };
        uniform MaterialData material @binding(0);

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support sampled resource member "
            r"'colorTex' in uniform block 'material' \(sampler2D\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_opaque_resource_members_in_interface_blocks():
    shader = """
    shader WebGLNoOpaqueInterfaceBlockMember {
        @glsl_interface_block(in) @glsl_interface_instance(fragmentInput)
        struct FragmentInputBlock {
            sampler2D colorTex;
            vec2 uv;
        };

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support sampled resource member "
            r"'colorTex' in interface block 'FragmentInputBlock' \(sampler2D\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_preserves_regular_uniform_blocks_and_sampler_uniforms():
    shader = """
    shader WebGLRegularUniformResources {
        struct MaterialData {
            vec4 tint;
            float factor;
        };
        uniform MaterialData material @binding(0);
        uniform sampler2D colorTex;

        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return texture(colorTex, uv) * material.tint * material.factor;
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "layout(std140) uniform MaterialData" in generated
    assert "vec4 tint;" in generated
    assert "float factor;" in generated
    assert "uniform sampler2D colorTex;" in generated
    assert "resource member" not in generated


@pytest.mark.parametrize(
    "resource_type,diagnostic_type",
    (
        ("sampler1D", "sampler1D"),
        ("samplerBuffer", "samplerBuffer"),
        ("sampler2DMS", "sampler2DMS"),
        ("samplerCubeArray", "samplerCubeArray"),
        ("Texture1D", "sampler1D"),
        ("Texture2DMS", "sampler2DMS"),
        ("TextureCubeArray", "samplerCubeArray"),
    ),
)
def test_webgl_codegen_rejects_sampled_resource_types_outside_glsl_es300(
    resource_type,
    diagnostic_type,
):
    shader = f"""
    shader WebGLNoUnsupportedSampler {{
        {resource_type} colorTex;

        fragment {{
            vec4 main() @ gl_FragColor {{
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support sampled resource "
            rf"'colorTex' \({diagnostic_type}\)"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "declaration,diagnostic_type",
    (
        ("double value = 1.0;", "double"),
        ("dvec2 value = dvec2(1.0);", "dvec2"),
        ("dmat2x2 value = dmat2x2(1.0);", "dmat2x2"),
    ),
)
def test_webgl_codegen_rejects_64_bit_float_types(declaration, diagnostic_type):
    shader = f"""
    shader WebGLNoFloat64 {{
        fragment {{
            vec4 main() @ gl_FragColor {{
                {declaration}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support 64-bit floating-point type "
            rf"'{diagnostic_type}'"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_storage_image_intrinsics():
    shader = """
    shader WebGLNoStorageImageIntrinsic {
        sampler2D colorTex;

        fragment {
            vec4 main() @ gl_FragColor {
                return imageLoad(colorTex, ivec2(0, 0));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="WebGL target does not support storage image intrinsic 'imageLoad'",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "call",
    (
        "textureGather(colorTex, uv)",
        "textureGatherOffset(colorTex, uv, ivec2(0, 0))",
        "textureGatherOffsets(colorTex, uv, ivec2(-1, 0), ivec2(0, -1), ivec2(1, 0), ivec2(0, 1))",
        "textureGatherCompare(colorTex, uv, 0.5)",
        "textureGatherCompareOffset(colorTex, uv, 0.5, ivec2(0, 0))",
        "textureGatherCompareOffsets(colorTex, uv, 0.5, ivec2(-1, 0), ivec2(0, -1), ivec2(1, 0), ivec2(0, 1))",
    ),
)
def test_webgl_codegen_rejects_glsl_es_310_texture_gather_intrinsics(call):
    shader = f"""
    shader WebGLNoTextureGather {{
        sampler2D colorTex;

        fragment {{
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target requires GLSL ES 3.00 and does not support "
            "texture gather intrinsic"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "call",
    (
        "float(textureQueryLevels(colorTex))",
        "textureQueryLod(colorTex, uv).x",
    ),
)
def test_webgl_codegen_rejects_desktop_texture_query_intrinsics(call):
    shader = f"""
    shader WebGLNoDesktopTextureQuery {{
        sampler2D colorTex;

        fragment {{
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {{
                return vec4({call}, 0.0, 0.0, 1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target requires GLSL ES 3.00 and does not support "
            "texture query intrinsic"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_atomics():
    shader = """
    shader WebGLNoAtomics {
        fragment {
            vec4 main() @ gl_FragColor {
                uint value = 0u;
                uint oldValue = atomicAdd(value, 1u);
                return vec4(float(oldValue));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="WebGL target does not support atomic operation 'atomicAdd'",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_codegen_rejects_wave_intrinsics():
    shader = """
    shader WebGLNoWave {
        fragment {
            vec4 main() @ gl_FragColor {
                uint total = WaveActiveSum(1u);
                return vec4(float(total));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support wave/subgroup intrinsic " "'WaveActiveSum'"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "call",
    ("workgroupBarrier()", "memoryBarrier()", "GroupMemoryBarrierWithGroupSync()"),
)
def test_webgl_codegen_rejects_synchronization_intrinsics(call):
    shader = f"""
    shader WebGLNoSynchronization {{
        fragment {{
            vec4 main() @ gl_FragColor {{
                {call};
                return vec4(1.0);
            }}
        }}
    }}
    """

    func_name = call.split("(", 1)[0]
    with pytest.raises(
        ValueError,
        match=(
            "WebGL target does not support synchronization intrinsic " rf"'{func_name}'"
        ),
    ):
        WebGLCodeGen().generate(parse_shader(shader))


def test_webgl_translate_spirv_builtin_position_omits_gl_pervertex(tmp_path):
    shader_path = tmp_path / "glpervertex.vert.spvasm"
    shader_path.write_text(
        SPIRV_TOOLS_GLPERVERTEX_ACCESS_CHAIN_ASSEMBLY, encoding="utf-8"
    )

    generated = crosstl.translate(
        str(shader_path), backend="webgl", format_output=False
    )

    assert generated.startswith("#version 300 es\n")
    assert "gl_PerVertex" not in generated
    assert "gl_ClipDistance" not in generated
    assert "gl_CullDistance" not in generated
    assert "layout(location = 0) in vec4 _ua_position;" in generated
    assert "gl_Position = _ua_position;" in generated

    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator not available")

    output_path = tmp_path / "glpervertex.vert.webgl.glsl"
    output_path.write_text(generated, encoding="utf-8")
    result = subprocess.run(
        [glslang, "-S", "vert", str(output_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    ("declaration", "assignment", "builtin"),
    (
        (
            "float clipDistance[1] @output @gl_ClipDistance;",
            "clipDistance[0] = 1.0;",
            "gl_ClipDistance",
        ),
        (
            "float cullDistance[1] @output @gl_CullDistance;",
            "cullDistance[0] = 1.0;",
            "gl_CullDistance",
        ),
    ),
)
def test_webgl_codegen_rejects_clip_and_cull_distance_writes(
    declaration, assignment, builtin
):
    shader = f"""
    shader WebGLUnsupportedDistances {{
        vertex {{
            {declaration}

            void main() {{
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                {assignment}
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=rf"WebGL target does not support vertex built-in output '{builtin}'",
    ):
        WebGLCodeGen().generate(parse_shader(shader))


@pytest.mark.parametrize(
    "stage",
    (
        "compute",
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "ray_generation",
    ),
)
def test_webgl_codegen_rejects_non_webgl_stage_targets(stage):
    ast = SimpleNamespace(functions=[], stages={})

    with pytest.raises(
        ValueError,
        match=rf"WebGL target does not support shader stage\(s\): {stage}",
    ):
        WebGLCodeGen().generate_program(ast, target_stage=stage)
