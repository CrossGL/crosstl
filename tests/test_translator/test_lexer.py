from crosstl.src.translator.lexer import Lexer
import pytest
from typing import List


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokens


def test_input_output_tokenization():
    code = """
    input vec3 position;
    output vec2 vUV;
    input vec2 vUV;
    output vec4 fragColor;
    """
    try:
        tokenize_code(code)
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


def test_else_if_statement_tokenization():
    code = """
    if (!a) {
        return b;
    } 
    if (!b) {
        return a;
    } else if (a < b) {
        return b;
    } else if (a > b) {
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
    shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            gl_Position = vec4(position, 1.0);
        }
    }

    // Perlin Noise Function
    float perlinNoise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            float noise = perlinNoise(vUV);
            float height = noise * 10.0;
            vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
            fragColor = vec4(color, 1.0);
            }
        }
    }

    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_bitwise_operator_tokenization():
    code = """
    int a = 60; // 60 = 0011 1100
    int b = 13; // 13 = 0000 1101
    int c = 0;
    c = a & b; // 12 = 0000 1100
    c = a | b; // 61 = 0011 1101
    c = a ^ b; // 49 = 0011 0001
    c = ~a; // -61 = 1100 0011
    c = a << 2; // 240 = 1111 0000
    c = a >> 2; // 15 = 0000 1111
      """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Bitwise operator tokenization not implemented.")


def test_data_types_tokenization():
    code = """
    int a;
    uint b;
    float c;
    double d;
    bool e;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Data types tokenization not implemented.")


def test_operators_tokenization():
    code = """
    int a;
    a = 2 + 1;
    a = a -  2;
    a = a / 1;
    a = a * 2;
    a = a % 2;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Operators tokenization not implemented.")


def test_logical_operators_tokenization():
    code = """
    if (0.8 > 0.7 || 0.6 > 0.7) {    
        return 0;
    } else if(0.8 > 0.7 && 0.8> 0.7) {        
        return 1;  
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Logical Operators tokenization not implemented.")


def test_assignment_shift_operators():
    code = """
    a >>= 1;
    b <<= 1;
        """

    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Shift operators tokenization failed.")


def test_assignment_operators_tokenization():
    code = """
    int a = 1;
    a += 1;
    a *= 2;
    a /= a;
    a -= -1;
    a %= 2;
    a &= 1;
    a |= 1;
    a ^= 1;
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Assignment operators tokenization not implemented.")


def test_const_tokenization():
    code = """
    const int a;
    """

    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Const keyword tokenization failed")
