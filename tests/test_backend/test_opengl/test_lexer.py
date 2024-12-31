from crosstl.backend.Opengl.OpenglLexer import GLSLLexer
import pytest
from typing import List


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code.

    Args:
        code (str): The code to tokenize
    Returns:
        List: The list of tokens generated from the lexer


    """
    lexer = GLSLLexer(code)
    return lexer.tokens


def test_input_output_tokenization():
    code = """
    #version 330 core
    layout(location = 0) in vec3 position;
    out vec2 vUV;
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;
    """
    try:
        print(tokenize_code(code))
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement_tokenization():
    code = """
    if (a > b) {
        return a;
    } else {
        return b;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_else_condition_tokenization():
    code_complex_if_else = """
    if ((a + b) > (c * d)) {
        return (a - c) / d;
    } else if ((a - b) < c) {
        return a + b;
    }
    else {
        return;
    }
    """
    try:
        tokenize_code(code_complex_if_else)
    except SyntaxError:
        pytest.fail("Complex if-else condition parsing not implemented.")


def test_for_statement_tokenization():
    code = """
    for (int i = 0; i < 10; i = i + 1) {
        sum += i;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_statement_tokenization():
    code = """
    if (a > b) {
        return a;
    } else {
        return 0;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call_tokenization():
    code = """
    float perlinNoise(float2 p) {
        return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    }
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
        fragColor = vec4(color, 1.0);
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_double_dtype_tokenization():
    code = """
    double ComputeArea(double radius) {
        double pi = 3.14159265359;
        double area = pi * radius * radius;
        return area;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("double tokenization not implemented")


def test_unsigned_int_dtype_tokenization():
    code = """
    double ComputeArea(double radius) {
        uint a = 3;
        uint b = 4;
        
        uint ans = a + b;
        return ans;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("unsigned integer parsing not implemented.")


if __name__ == "__main__":
    pytest.main()
