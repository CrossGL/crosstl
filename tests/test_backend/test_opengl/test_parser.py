from crosstl.backend.Opengl.OpenglLexer import GLSLLexer
import pytest
from typing import List
from crosstl.backend.Opengl.OpenglParser import GLSLParser


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code.

    Args:
        code (str): The code to tokenize
    Returns:
        List: The list of tokens generated from the lexer


    """
    lexer = GLSLLexer(code)
    return lexer.tokens


def parse_code(Tokens: List, shader_type="vertex") -> List:
    """Helper function to parse code.

    Args:
        Tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser


    """
    parser = GLSLParser(Tokens, shader_type)
    return parser.parse()


def test_input_output():
    code = """
    layout(location = 0) in vec3 position;
    out vec2 vUV;
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens, "vertex")
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement():
    code = """
    #version 450
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        if (noise > 0.5) {
            float fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    }

    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    layout(location = 0) in vec3 position;
    out vec2 vUV;

    void main() {
        vUV = position.xy * 10.0;
    }
    
    """
    try:
        tokens = tokenize_code(code)
        print(parse_code(tokens))
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for_statement():
    code = """
    #version 450
    // Vertex shader
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
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
        print(parse_code(tokens, "vertex"))
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_statement():
    code = """
    #version 450
    // Vertex shader
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
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
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens, "vertex")
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_if_statement():
    code = """
    #version 450
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        if (noise > 0.75) {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        else if (noise > 0.5) {
            fragColor = vec4(0.0, 1.0, 0.0, 1.0);
        }
        else {
            fragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens, "fragment")
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call():
    code = """
    #version 450
    // Vertex shader
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    layout(location = 0) in vec3 position;
    out vec2 vUV;

    void main() {
        gl_Position = vec4(position, 1.0);
        vUV = position.xy * 10.0;
        float noise = perlinNoise(vUV);

    }
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens, "vertex")
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_double_dtype_tokenization():
    code = """
    double ComputeArea(double radius) {
        double pi = 3.14159265359;
        double area = pi * radius * radius;
        return area;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("double tokenization not implemented")


def test_mod_parsing():
    code = """
    void main() {
        int a = 10 % 3;  // Basic modulus
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


if __name__ == "__main__":
    pytest.main()
