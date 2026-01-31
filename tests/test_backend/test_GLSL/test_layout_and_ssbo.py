import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


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


if __name__ == "__main__":
    pytest.main()
