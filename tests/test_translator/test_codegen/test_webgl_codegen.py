import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import crosstl
import crosstl.translator.codegen as codegen
from crosstl.formatter import format_shader_code
from crosstl.translator.ast import BinaryOpNode
from crosstl.translator.codegen.webgl_codegen import (
    WebGLArithmeticConversionError,
    WebGLBooleanCompoundAssignmentError,
    WebGLCodeGen,
    WebGLStructConstructionError,
)
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


def assert_webgl_stage_validates_if_available(generated, tmp_path, name, stage="frag"):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    output_path = tmp_path / f"{name}.{stage}.webgl.glsl"
    output_path.write_text(generated, encoding="utf-8")
    result = subprocess.run(
        [glslang, "-S", stage, str(output_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout


def test_webgl_backend_is_target_only():
    spec = codegen.get_backend("webgl2")

    assert spec is not None
    assert spec.name == "webgl"
    assert spec.source_registry_name is None
    assert "webgl" not in codegen.source_backend_names()
    assert codegen.normalize_backend_name("target.webgl.glsl") == "webgl"
    assert codegen.get_backend_extension("glsl-es") == ".webgl.glsl"
    assert isinstance(codegen.get_codegen("essl"), WebGLCodeGen)


def test_webgl_codegen_emits_glsl_es_header_and_default_precision(tmp_path):
    generated = WebGLCodeGen().generate(parse_shader(WEBGL_SHADER))

    assert generated.startswith("#version 300 es\n")
    assert generated.index("#version 300 es") < generated.index(
        "precision highp float;"
    )
    assert "precision highp float;\n" in generated
    assert "precision highp int;\n" in generated
    assert "#version 450 core" not in generated
    assert "layout(location = 0) out vec4 fragColor;" in generated
    _assert_webgl_supported_arithmetic_conversions(tmp_path)
    _assert_webgl_unavailable_integer_width_diagnostic()
    _assert_webgl_boolean_arithmetic_compound_assignments(tmp_path)
    _assert_webgl_boolean_arithmetic_compound_diagnostic()


def _assert_webgl_supported_arithmetic_conversions(tmp_path):
    shader = r"""
    shader WebGLArithmeticConversions {
        int addNarrow(uint16_t left, int16_t right) {
            return left + right;
        }

        ivec3 addNarrowVector(ushort3 left, short3 right) {
            return left + right;
        }

        uvec3 addMixed(ivec3 left, uint right) {
            return left + right;
        }

        bvec3 compareMixed(ivec3 left, uint right) {
            return left < right;
        }

        uvec3 chooseMixed(bool condition, int left, uvec3 right) {
            return condition ? left : right;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                int narrow = addNarrow(uint16_t(3u), int16_t(-2));
                ivec3 narrowVector = addNarrowVector(
                    ushort3(1u),
                    short3(-2)
                );
                uvec3 mixed = addMixed(ivec3(narrow), 4u);
                bvec3 ordered = compareMixed(ivec3(narrow), 5u);
                uvec3 selected = chooseMixed(ordered.x, narrow, mixed);
                return vec4(selected, 1.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate_stage(parse_shader(shader), "fragment")

    assert "return (int(left) + right);" in generated
    assert "return (ivec3(left) + right);" in generated
    assert "return (uvec3(left) + right);" in generated
    assert "return lessThan(uvec3(left), uvec3(right));" in generated
    assert "return (condition ? uvec3(left) : right);" in generated
    assert "#extension GL_ARB_gpu_shader_int64" not in generated
    assert_webgl_stage_validates_if_available(
        generated, tmp_path, "arithmetic_conversions", "frag"
    )


def _assert_webgl_unavailable_integer_width_diagnostic():
    shader = r"""
    shader WebGLWideArithmetic {
        int64_t multiply(uint left, int64_t right) {
            return left * right;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(1.0);
            }
        }
    }
    """
    ast = parse_shader(shader)
    source_location = {"line": 4, "column": 25}
    codegen = WebGLCodeGen()
    expression = next(
        node for node in codegen.walk_ast(ast) if isinstance(node, BinaryOpNode)
    )
    expression.source_location = source_location

    with pytest.raises(WebGLArithmeticConversionError) as exc_info:
        codegen.generate_stage(ast, "fragment")

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.webgl-arithmetic-conversion-invalid"
    )
    assert diagnostic.missing_capabilities == ("webgl.arithmetic-conversion-lowering",)
    assert diagnostic.operator == "*"
    assert diagnostic.operand_types == ("uint", "int64_t")
    assert diagnostic.attempted_common_type == "int64_t"
    assert diagnostic.common_type == "int64_t"
    assert diagnostic.reason == "target-integer-width-unsupported"
    assert diagnostic.source_location == source_location


def _assert_webgl_boolean_arithmetic_compound_assignments(tmp_path):
    shader = r"""
    shader WebGLBooleanArithmeticCompoundAssignments {
        struct Flags {
            bool enabled;
        };

        bool powerBool(bool base, bool exp) {
            bool result = true;
            while (exp) {
                if (exp & 1) {
                    result *= base;
                }
                exp >>= 1;
                base *= base;
            }
            return result;
        }

        bool3 updateMask(bool3 value, bool3 factor) {
            value *= factor;
            value >>= 1;
            return value;
        }

        bool nextFlag(inout uint calls) {
            calls += 1u;
            return true;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                uint calls = 0u;
                Flags flags;
                flags.enabled = false;
                flags.enabled *= nextFlag(calls);
                bool value = powerBool(true, false);
                bool3 mask = updateMask(
                    bool3(true, false, true),
                    bool3(false, true, true)
                );
                return vec4(value ? 1.0 : 0.0);
            }
        }
    }
    """

    generated = WebGLCodeGen().generate_stage(parse_shader(shader), "fragment")

    assert "if (((int(exp) & 1) != 0))" in generated
    assert "result = ((int(result) * int(base)) != 0);" in generated
    assert "exp = ((int(exp) >> 1) != 0);" in generated
    assert "base = ((int(base) * int(base)) != 0);" in generated
    assert "value = notEqual((ivec3(value) * ivec3(factor)), ivec3(0));" in generated
    assert "value = notEqual((ivec3(value) >> 1), ivec3(0));" in generated
    assert generated.count("nextFlag(calls)") == 1
    assert (
        "flags.enabled = ((int(flags.enabled) * int(nextFlag(calls))) != 0);"
        in generated
    )
    assert "result *= base;" not in generated
    assert "exp >>= 1;" not in generated
    assert_webgl_stage_validates_if_available(
        generated,
        tmp_path,
        "boolean_arithmetic_compound_assignments",
        "frag",
    )


def _assert_webgl_boolean_arithmetic_compound_diagnostic():
    shader = r"""
    shader WebGLSideEffectingBooleanCompoundAssignmentIndex {
        uint nextIndex(inout uint calls) {
            calls += 1u;
            return 0u;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                uint calls = 0u;
                bool values[2] = {true, false};
                values[nextIndex(calls)] *= false;
                return vec4(1.0);
            }
        }
    }
    """

    with pytest.raises(WebGLBooleanCompoundAssignmentError) as exc_info:
        WebGLCodeGen().generate_stage(parse_shader(shader), "fragment")

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.webgl-boolean-compound-assignment-invalid"
    )
    assert diagnostic.missing_capabilities == (
        "webgl.boolean-compound-assignment-lowering",
    )
    assert diagnostic.operator == "*="
    assert diagnostic.target_type == "bool"
    assert diagnostic.reason == "lvalue-side-effects"


def test_webgl_codegen_sanitizes_reserved_identifiers_and_collisions():
    shader = r"""
    shader WebGLReservedIdentifiers {
        float gl_helper(float __arg, float _arg) {
            float local__value = __arg;
            float local_value = _arg;
            return local__value + local_value;
        }

        vertex {
            vec4 main(
                vec3 position__value @ POSITION,
                vec3 position_value @ NORMAL
            ) @ gl_Position {
                float result__value = gl_helper(
                    position__value.x,
                    position_value.x
                );
                float result_value = result__value;
                return vec4(
                    position__value + position_value + vec3(result_value),
                    1.0
                );
            }
        }
    }
    """

    generated = WebGLCodeGen().generate_stage(parse_shader(shader), "vertex")

    assert "__" not in generated
    assert "layout(location = 0) in vec3 position_value_2;" in generated
    assert "layout(location = 1) in vec3 position_value;" in generated
    assert "float crossgl_gl_helper(float arg, float _arg)" in generated
    assert generated.count("float crossgl_gl_helper(float arg, float _arg)") == 2
    assert "crossgl_gl_helper(position_value_2.x, position_value.x)" in generated
    assert "float local_value_2 = arg;" in generated


def test_webgl_codegen_sanitizes_webgl_reserved_prefixes():
    shader = r"""
    shader WebGLReservedPrefixes {
        float crossgl_webgl_helper(float value) { return value; }

        float webgl_helper(float _webgl_arg) {
            float webgl__value = _webgl_arg;
            return webgl__value;
        }

        vertex {
            vec4 main(vec3 webgl_position @ POSITION) @ gl_Position {
                return vec4(webgl_position, webgl_helper(1.0));
            }
        }
    }
    """

    generated = WebGLCodeGen().generate_stage(parse_shader(shader), "vertex")

    assert "layout(location = 0) in vec3 crossgl_webgl_position;" in generated
    assert "float crossgl_webgl_helper_2(float crossgl_webgl_arg)" in generated
    assert "float crossgl_webgl_value = crossgl_webgl_arg;" in generated
    assert "crossgl_webgl_helper_2(1.0)" in generated
    assert re.search(r"\b(?:webgl_|_webgl_)[A-Za-z0-9_]*", generated) is None
    assert "__" not in generated


@pytest.mark.parametrize(
    ("source_name", "emitted_name"),
    (
        ("texture_a_", "texture_a_"),
        ("texture_a__", "texture_a"),
        ("texture_a___", "texture_a"),
    ),
)
def test_webgl_trailing_underscore_resource_names_share_glsl_mapping(
    tmp_path, source_name, emitted_name
):
    shader = f"""
    shader WebGLTrailingUnderscore {{
        sampler2D {source_name} @binding(0);

        fragment {{
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {{
                return texture({source_name}, uv);
            }}
        }}
    }}
    """

    codegen = WebGLCodeGen()
    generated = codegen.generate_stage(parse_shader(shader), "fragment")

    assert codegen.glsl_sanitized_identifier_base(source_name) == emitted_name
    assert codegen.glsl_module_identifier_names[source_name] == emitted_name
    assert f"uniform sampler2D {emitted_name};" in generated
    assert f"texture({emitted_name}, uv)" in generated
    assert generated.count(emitted_name) == 2
    assert_webgl_stage_validates_if_available(
        generated,
        tmp_path,
        f"trailing_underscore_{len(source_name) - len(source_name.rstrip('_'))}",
    )


def test_webgl_structure_copy_and_registered_scalar_conversion_validate(tmp_path):
    shader = """
    shader WebGLStructureConversions {
        struct Pair {
            float first;
            float second;
        };

        struct complex64_t {
            float real;
            float imag;
        };

        Pair copyPair(Pair value) {
            return Pair(value);
        }

        complex64_t promote(float value) {
            return value;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                Pair pair = Pair(1.0, 2.0);
                Pair copied = Pair(pair);
                complex64_t promoted = complex64_t(3.0);
                complex64_t returned = promote(4.0);
                return vec4(
                    copied.first,
                    copied.second,
                    promoted.real,
                    returned.imag
                );
            }
        }
    }
    """

    generated = WebGLCodeGen().generate_stage(parse_shader(shader), "fragment")

    assert "return value;" in generated
    assert "Pair copied = pair;" in generated
    assert "complex64_t promoted = complex64_t(float(3.0), 0.0);" in generated
    assert "return complex64_t(float(value), 0.0);" in generated
    assert "Pair(value)" not in generated
    assert "Pair(pair)" not in generated
    assert_webgl_stage_validates_if_available(
        generated,
        tmp_path,
        "webgl_structure_conversions",
    )


def test_webgl_registered_scalar_conversion_rejects_destination_shape():
    shader = """
    shader InvalidWebGLStructureConversion {
        struct complex64_t {
            float real;
            int imag;
        };

        complex64_t promote(float value) {
            return value;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                complex64_t value = promote(1.0);
                return vec4(value.real);
            }
        }
    }
    """

    with pytest.raises(WebGLStructConstructionError) as exc_info:
        WebGLCodeGen().generate_stage(parse_shader(shader), "fragment")

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.webgl-struct-construction-unsupported"
    )
    assert diagnostic.destination_type == "complex64_t"
    assert diagnostic.source_type == "float"
    assert diagnostic.conversion_kind == "contextual-scalar-conversion"
    assert diagnostic.reason == "destination-shape-mismatch"


def test_webgl_codegen_reuses_fixed_array_for_in_contract():
    shader = """
    shader WebGLFixedArrayForIn {
        constant uint[2] values = {3u, 5u};
        fragment {
            vec4 main() @ gl_FragColor {
                uint total = 0u;
                for value in values {
                    total += value;
                }
                return vec4(float(total));
            }
        }
    }
    """

    generated = WebGLCodeGen().generate(parse_shader(shader))

    assert "uint value_crossgl_iterable[2] = values;" in generated
    assert "value_crossgl_index < 2" in generated
    assert "uint value = value_crossgl_iterable[value_crossgl_index];" in generated

    counted_shader = """
    shader WebGLIntegerBoundForIn {
        int helper(int limit) {
            int total = 0;
            for value in limit {
                total += value;
            }
            return total;
        }
    }
    """

    counted = WebGLCodeGen().generate(parse_shader(counted_shader))

    assert "for (int value = 0; value < limit; ++value)" in counted


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
        "textureGatherCompare(colorTex, uv, 0.5)",
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
