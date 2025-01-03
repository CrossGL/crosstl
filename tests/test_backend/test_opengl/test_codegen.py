from crosstl.backend.Opengl.OpenglLexer import GLSLLexer
import pytest
from typing import List
from crosstl.backend.Opengl.OpenglParser import GLSLParser
from crosstl.backend.Opengl.openglCrossglCodegen import GLSLToCrossGLConverter


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code.

    Args:
        code (str): The code to tokenize
    Returns:
        List: The list of tokens generated from the lexer


    """
    lexer = GLSLLexer(code)
    return lexer.tokenize()


def parse_code(Tokens: List, shader_type="vertex") -> List:
    """Helper function to parse code.

    Args:
        Tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser


    """
    parser = GLSLParser(Tokens, shader_type)
    return parser.parse()


def generate_code(ast: List, shader_type="vertex") -> str:
    """Helper function to generate code.

    Args:
        ast (List): The abstract syntax tree to generate code from
    Returns:
        str: The code generated from the ast


    """
    converter = GLSLToCrossGLConverter(shader_type=shader_type)
    return converter.generate(ast)


def test_input_output():
    code = """
    #version 450
    // Vertex shader

    layout(location = 0) in vec3 position;
    layout(location = 0) in vec3 color;
    out vec2 vUV;

    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
        vUV = position.xy * 10.0;
    }

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens, "vertex")
        code = generate_code(ast, "vertex")
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement():
    code = """
    #version 450
    // Fragment shader
    layout(location = 0) in vec3 position;
    out vec2 vUV;

    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
        float noise = perlinNoise(vUV);
        if (noise > 0.5) {
            float fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    }

    layout(location = 0) in vec2 vUV;
    out vec2 color;

    void main() {
        color = position.xy * 10.0;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for_statement():
    code = """
    #version 450
    // Vertex shader
    layout(location = 0) in vec3 position;
    out vec2 vUV;

    void main() {
        vUV = position.xy * 10.0;
        for (int i = 0; i < 10; i = i + 1) {
            vUV = vec2(0.0, 0.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens, "vertex")
        code = generate_code(ast, "vertex")
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_statement():
    code = """
    #version 450
    // Vertex shader
    layout(location = 0) in vec3 position;
    out vec2 vUV;

    void main() {
        vUV = position.xy * 10.0;
        if (vUV.x > vUV.y) {
            vUV = vec2(0.0, 0.0);
        }
        else {
            vUV = vec2(1.0, 1.0);
        }
        gl_Position = vec4(position, 1.0);
    }
    
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens, "vertex")
        code = generate_code(ast, "vertex")
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call():
    code = """
    #version 450
    // Fragment shader
    
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """

    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens, "fragment")
        code = generate_code(ast, "fragment")
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_double_dtype_codegen():
    code = """
    double ComputeArea(double radius) {
        double pi = 3.14159265359;
        double area = pi * radius * radius;
        return area;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("double tokenization not implemented")


if __name__ == "__main__":
    pytest.main()
