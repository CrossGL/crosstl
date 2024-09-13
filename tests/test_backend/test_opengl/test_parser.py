from crosstl.src.backend.Opengl.OpenglLexer import GLSLLexer
import pytest
from typing import List
from crosstl.src.backend.Opengl.OpenglParser import GLSLParser


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code.

    Args:
        code (str): The code to tokenize
    Returns:
        List: The list of tokens generated from the lexer


    """
    lexer = GLSLLexer(code)
    return lexer.tokens


def parse_code(Tokens: List) -> List:
    """Helper function to parse code.

    Args:
        Tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser


    """
    parser = GLSLParser(Tokens)
    return parser.parse()


def test_input_output():
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
    }

    // Fragment shader

    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement():
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
    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        if (noise > 0.5) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
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
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        for (int i = 0; i < 10; i = i + 1) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
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
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;
    void main() {
        float noise = perlinNoise(vUV);
        if (noise > 0.5) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        else {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_if_statement():
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
        parse_code(tokens)
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
        vUV = position.xy * 10.0;
        float noise = perlinNoise(vUV);

    }
    // Fragment shader
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_bitwise_and():
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
        float noise = perlinNoise(vUV);

    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        int x = 3&4;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise operations failed")


def test_bitwise_xor():
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
        float noise = perlinNoise(vUV);

    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        int x = 3^4;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise operations failed")


def test_bitwise_or():
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
        float noise = perlinNoise(vUV);

    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        int x = 3 | 4;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise operations failed")


def test_uint():
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
        float noise = perlinNoise(vUV);
    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        unsigned int x = 6;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Unsigned int operation failed")


def test_double():
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
        float noise = perlinNoise(vUV);
    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        double x = 6.94;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Double operation failed")


def test_assign_and():
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
        float noise = perlinNoise(vUV);
    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        int x= 7;
        x&=9;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Assign and operation failed")


def test_assign_or():
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
        float noise = perlinNoise(vUV);
    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        int x= 7;
        x|=9;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Assign or operation failed")


def test_assign_xor():
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
        float noise = perlinNoise(vUV);
    }
    // Fragment shader
    in vec2 vUV;
    layout(location = 0) out vec4 fragColor;

    void main() {
        float noise = perlinNoise(vUV);
        float height = noise * 10.0;
        int x= 7;
        x^=9;
        vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Assign xor operation failed")
