import textwrap

import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen

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
