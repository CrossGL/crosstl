import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

TESSELLATION_CONTROL_GLSL = """
#version 450 core
layout(vertices = 3) out;

in vec3 vPosition[];
out vec3 tcPosition[];

void main() {
    tcPosition[gl_InvocationID] = vPosition[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    gl_TessLevelOuter[0] = 1.0;
    gl_TessLevelOuter[1] = 1.0;
    gl_TessLevelOuter[2] = 1.0;
    gl_TessLevelInner[0] = 1.0;
}
"""


TESSELLATION_EVAL_GLSL = """
#version 450 core
layout(triangles, equal_spacing, cw) in;

in vec3 tcPosition[];

void main() {
    vec3 p = (tcPosition[0] + tcPosition[1] + tcPosition[2]) / 3.0;
    gl_Position = vec4(p, 1.0);
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


def test_parse_tess_control_shader():
    ast = parse_glsl(TESSELLATION_CONTROL_GLSL, "tessellation_control")
    assert ast is not None


def test_parse_tess_eval_shader():
    ast = parse_glsl(TESSELLATION_EVAL_GLSL, "tessellation_evaluation")
    assert ast is not None


def test_codegen_tess_control_roundtrip():
    output = generate_crossgl(TESSELLATION_CONTROL_GLSL, "tessellation_control")
    assert "tessellation_control" in output.lower()
    shader_ast = parse_crossgl(output)
    assert ShaderStage.TESSELLATION_CONTROL in shader_ast.stages


def test_codegen_tess_eval_roundtrip():
    output = generate_crossgl(TESSELLATION_EVAL_GLSL, "tessellation_evaluation")
    assert "tessellation_evaluation" in output.lower()
    shader_ast = parse_crossgl(output)
    assert ShaderStage.TESSELLATION_EVALUATION in shader_ast.stages


if __name__ == "__main__":
    pytest.main()
