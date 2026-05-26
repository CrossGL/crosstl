import pytest

import crosstl.translator
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def generate_crossgl(code: str, shader_type: str):
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(
        parse_glsl(code, shader_type)
    )


def test_parse_ssbo_layout():
    code = """
    #version 450 core
    layout(std430, binding = 2) buffer DataBlock {
        vec4 values[];
    } dataBlock;
    void main() {
        vec4 v = dataBlock.values[0];
    }
    """
    ast = parse_glsl(code, "vertex")
    assert ast is not None


def test_parse_layout_locations_and_components():
    code = """
    #version 450 core
    layout(location = 1, component = 2) in vec4 color;
    layout(location = 0, index = 1) out vec4 fragColor;
    void main() {
        fragColor = color;
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_precision_qualifiers_on_variables():
    code = """
    #version 300 es
    precision mediump float;
    mediump vec2 uv;
    void main() {
        uv = vec2(0.0);
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_early_fragment_tests_layout():
    code = """
    #version 450 core
    layout(early_fragment_tests) in;
    layout(location = 0) out vec4 fragColor;
    void main() {
        fragColor = vec4(1.0);
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_ray_shader_record_layout_roundtrips_without_binding():
    code = """
    #version 460 core
    #extension GL_EXT_ray_tracing : require

    struct RayPayload {
        vec4 color;
    };

    layout(binding = 0) uniform accelerationStructureEXT topLevelAS;
    layout(location = 0) rayPayloadEXT RayPayload rayPayload;
    layout(shaderRecordEXT, std430) buffer ShaderRecordData {
        uint materialIndex;
    } shaderRecord;

    void main() {
        rayPayload.color = vec4(float(shaderRecord.materialIndex));
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsNoneEXT,
            0xff,
            0,
            1,
            0,
            vec3(0.0),
            0.001,
            vec3(0.0, 0.0, 1.0),
            1000.0,
            0
        );
    }
    """

    crossgl = generate_crossgl(code, "ray_generation")

    assert "accelerationStructureEXT topLevelAS @binding(0);" in crossgl
    assert "RayPayload rayPayload @location(0) @rayPayloadEXT;" in crossgl
    assert (
        "ShaderRecordData shaderRecord @glsl_buffer_block(shaderRecordEXT, std430);"
        in crossgl
    )
    assert "RayGenerationInput" not in crossgl
    assert "RayGenerationOutput" not in crossgl
    assert "void main()" in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert glsl.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in glsl
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in glsl
    assert "layout(location = 0) rayPayloadEXT RayPayload rayPayload;" in glsl
    assert "layout(shaderRecordEXT, std430) buffer ShaderRecordData" in glsl
    assert "layout(shaderRecordEXT, std430, binding" not in glsl
    assert "rayPayload.color = vec4(float(shaderRecord.materialIndex));" in glsl
    assert "traceRayEXT(" in glsl


def test_parse_ray_shader_record_layout_rejects_binding():
    code = """
    #version 460 core
    #extension GL_EXT_ray_tracing : require

    layout(shaderRecordEXT, binding = 3) buffer ShaderRecordData {
        uint materialIndex;
    } shaderRecord;

    void main() { }
    """

    with pytest.raises(
        ValueError,
        match="shaderRecordEXT buffer blocks cannot declare binding layout qualifiers",
    ):
        generate_crossgl(code, "ray_generation")


if __name__ == "__main__":
    pytest.main()
