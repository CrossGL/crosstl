import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

GEOMETRY_GLSL = """
#version 450 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vColor[];
out vec3 gColor;

void main() {
    for (int i = 0; i < 3; i++) {
        gColor = vColor[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def generate_crossgl(code: str, shader_type: str):
    ast = parse_glsl(code, shader_type)
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(ast)


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def test_parse_geometry_shader():
    ast = parse_glsl(GEOMETRY_GLSL, "geometry")
    assert ast is not None
    assert ast.functions


def test_codegen_geometry_roundtrip():
    output = generate_crossgl(GEOMETRY_GLSL, "geometry")
    assert "geometry" in output.lower()
    assert "EmitVertex" in output
    shader_ast = parse_crossgl(output)
    assert ShaderStage.GEOMETRY in shader_ast.stages


if __name__ == "__main__":
    pytest.main()
