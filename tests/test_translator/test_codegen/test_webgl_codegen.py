from types import SimpleNamespace

import pytest

import crosstl
import crosstl.translator.codegen as codegen
from crosstl.formatter import format_shader_code
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

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
