from crosstl.src.translator.lexer import Lexer
import pytest
from typing import List


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokenize()


def test_input_output_tokenization():
    code = """
    input vec3 position;
    output vec2 vUV;
    input vec2 vUV;
    output vec4 fragColor;
    """
    try:
        tokens = tokenize_code(code)
        assert tokens is not None
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
        tokens = tokenize_code(code)
        assert tokens is not None
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for_statement_tokenization():
    code = """
    for (int i = 0; i < 10; i = i + 1) {
        sum += i;
    }
    """
    try:
        tokens = tokenize_code(code)
        assert tokens is not None
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
        tokens = tokenize_code(code)
        assert tokens is not None
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
        tokens = tokenize_code(code)
        assert tokens is not None
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_bitwise_operators_tokenization():
    code = """
    int a = 5 & 3;
    int b = 5 | 3;
    int c = 5 ^ 3;
    int d = ~5;
    int e = 5 << 1;
    int f = 5 >> 1;
    """
    try:
        tokens = tokenize_code(code)
        expected_tokens = [
            ("INT", "int"),
            ("IDENTIFIER", "a"),
            ("EQUALS", "="),
            ("NUMBER", "5"),
            ("BITWISE_AND", "&"),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("INT", "int"),
            ("IDENTIFIER", "b"),
            ("EQUALS", "="),
            ("NUMBER", "5"),
            ("BITWISE_OR", "|"),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("INT", "int"),
            ("IDENTIFIER", "c"),
            ("EQUALS", "="),
            ("NUMBER", "5"),
            ("BITWISE_XOR", "^"),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("INT", "int"),
            ("IDENTIFIER", "d"),
            ("EQUALS", "="),
            ("BITWISE_NOT", "~"),
            ("NUMBER", "5"),
            ("SEMICOLON", ";"),
            ("INT", "int"),
            ("IDENTIFIER", "e"),
            ("EQUALS", "="),
            ("NUMBER", "5"),
            ("SHIFT_LEFT", "<<"),
            ("NUMBER", "1"),
            ("SEMICOLON", ";"),
            ("INT", "int"),
            ("IDENTIFIER", "f"),
            ("EQUALS", "="),
            ("NUMBER", "5"),
            ("SHIFT_RIGHT", ">>"),
            ("NUMBER", "1"),
            ("SEMICOLON", ";"),
        ]
        assert tokens == expected_tokens
    except SyntaxError:
        pytest.fail("Bitwise operators tokenization failed.")


def test_assignment_operators_tokenization():
    code = """
    int a = 5;
    a &= 3;
    a |= 3;
    a ^= 3;
    a %= 3;
    a <<= 1;
    a >>= 1;
    """
    try:
        tokens = tokenize_code(code)
        expected_tokens = [
            ("INT", "int"),
            ("IDENTIFIER", "a"),
            ("EQUALS", "="),
            ("NUMBER", "5"),
            ("SEMICOLON", ";"),
            ("IDENTIFIER", "a"),
            ("ASSIGN_AND", "&="),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("IDENTIFIER", "a"),
            ("ASSIGN_OR", "|="),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("IDENTIFIER", "a"),
            ("ASSIGN_XOR", "^="),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("IDENTIFIER", "a"),
            ("ASSIGN_MOD", "%="),
            ("NUMBER", "3"),
            ("SEMICOLON", ";"),
            ("IDENTIFIER", "a"),
            ("ASSIGN_SHIFT_LEFT", "<<="),
            ("NUMBER", "1"),
            ("SEMICOLON", ";"),
            ("IDENTIFIER", "a"),
            ("ASSIGN_SHIFT_RIGHT", ">>="),
            ("NUMBER", "1"),
            ("SEMICOLON", ";"),
        ]
        assert tokens == expected_tokens
    except SyntaxError:
        pytest.fail("Assignment operators tokenization failed.")


def test_data_types_and_const_tokenization():
    code = """
    const double pi = 3.14159;
    unsigned int maxVal = 255;
    """
    try:
        tokens = tokenize_code(code)
        expected_tokens = [
            ("CONST", "const"),
            ("DOUBLE", "double"),
            ("IDENTIFIER", "pi"),
            ("EQUALS", "="),
            ("FLOAT_NUMBER", "3.14159"),
            ("SEMICOLON", ";"),
            ("UNSIGNED_INT", "unsigned int"),
            ("IDENTIFIER", "maxVal"),
            ("EQUALS", "="),
            ("NUMBER", "255"),
            ("SEMICOLON", ";"),
        ]
        assert tokens == expected_tokens
    except SyntaxError:
        pytest.fail("Data types and const tokenization failed.")
