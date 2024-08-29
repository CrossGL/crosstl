from crosstl.src.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.src.translator.parser import Parser
from crosstl.src.translator.codegen.directx_codegen import HLSLCodeGen


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.tokens


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = HLSLCodeGen()
    return codegen.generate(ast_node)


def test_input_output():
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

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            fragColor = vec4(color, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_statement():
    code = """
    shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            if (vUV.x > 0.5) {
                vUV.x = 0.5;
            }

            gl_Position = vec4(position, 1.0);
        }
    }

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
                if (vUV.x > 0.5) {
                fragColor = vec4(1.0, 1.0, 1.0, 1.0);
                }
            fragColor = vec4(color, 1.0);
            }
        }
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
    shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            for (int i = 0; i < 10; i = i + 1) {
                vUV = vec2(0.0, 0.0);
            }
            gl_Position = vec4(position, 1.0);
        }
    }

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            for (int i = 0; i < 10; i = i + 1) {
                vec3 color = vec3(1.0, 1.0, 1.0);
            }
            fragColor = vec4(color, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_statement():
    code = """
    shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            if (vUV.x > 0.5) {
                vUV.x = 0.5;
            } else {
                vUV.x = 0.0;
            }
            gl_Position = vec4(position, 1.0);
        }
    }

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            if (vUV.x > 0.5) {
            fragColor = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
            fragColor = vec4(color, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_if_statement():
    code = """
    shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV = position.xy * 10.0;
            if (vUV.x < 0.5) {
                vUV.x = 0.25;
            }
            if (vUV.x < 0.25) {
                vUV.x = 0.0;
            } else if (vUV.x < 0.75) {
                vUV.x = 0.5;
            } else if (vUV.x < 1.0) {
                vUV.x = 0.75;
            } else {
                vUV.x = 0.0;
            }
            gl_Position = vec4(position, 1.0);
        }
    }

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            if (vUV.x > 0.75) {
                fragColor = vec4(1.0, 1.0, 1.0, 1.0);
            } else if (vUV.x > 0.5) {
                fragColor = vec4(0.5, 0.5, 0.5, 1.0);
            } else {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
            fragColor = vec4(color, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call():
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
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")
