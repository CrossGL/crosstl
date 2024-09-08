from crosstl.src.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.src.translator.parser import Parser


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
        parse_code(tokens)
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
        parse_code(tokens)
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
        parse_code(tokens)
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
        parse_code(tokens)
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
            } else if (vUV.x > 0.25) {
                fragColor = vec4(0.25, 0.25, 0.25, 1.0);
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
        parse_code(tokens)
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_assign_shift_right():
    code = """
    shader PerlinNoise {
    vertex {
        input vec3 position;
        output vec2 vUV;

        void main() {
            vUV >>= 1;
            vUV = position.xy * 10.0;
            gl_Position = vec4(position, 1.0);
        }
    }

    // Fragment Shader
    fragment {
        input vec2 vUV;
        output vec4 fragColor;

        void main() {
            double noise = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
            double height = noise * 10.0;
            uint a >>= 1;
            uint b = 2;
            vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
            fragColor = vec4(color, 1.0);
            }
        }
    }
    """

    try:
        tokens = tokenize_code(code)
        print(parse_code(tokens))
    except SyntaxError as e:
        pytest.fail(f"Failed to parse ASSIGN_SHIFT_RIGHT token: {e}")


def test_logical_operators():
    code = """
        shader LightControl {
        vertex {
            input vec3 position;
            output float isLightOn;
            void main() {
                // Using logical AND and logical OR operators
                if ((position.x > 0.3 && position.z < 0.7) || position.y > 0.5) {
                    isLightOn = 1.0;
                } else {
                    isLightOn = 0.0;
                }
                // Set the vertex position
                gl_Position = vec4(position, 1.0);
            }
        }
        fragment {
            input float isLightOn;
            output vec4 fragColor;
            void main() {
                if (isLightOn == 1.0) {
                    fragColor = vec4(1.0, 1.0, 0.0, 1.0);  // Light is on
                } else {
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // Light is off
                }
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_var_assignment():
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
            double noise = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
            double height = noise * 10.0;
            uint a = 1;
            uint b = 2;
            vec3 color = vec3(height / 10.0, 1.0 - height / 10.0, 0.0);
            fragColor = vec4(color, 1.0);
            }
        }
    }

    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Variable assignment parsing not implemented.")


def test_assign_ops():

    code = """
            shader LightControl {
                vertex {
                    input vec3 position;
                    output int lightStatus;

                    void main() {
                        int xStatus = int(position.x * 10.0);
                        int yStatus = int(position.y * 10.0);
                        int zStatus = int(position.z * 10.0);

                        xStatus |= yStatus;
                        yStatus &= zStatus;
                        zStatus %= xStatus;
                        lightStatus = xStatus;
                        lightStatus ^= zStatus;

                        gl_Position = vec4(position, 1.0);
                    }
                }

                fragment {
                    input int lightStatus;
                    output vec4 fragColor;

                    void main() {
                        if (lightStatus > 0) {
                            fragColor = vec4(1.0, 1.0, 0.0, 1.0); 
                        } else {
                            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                        }
                    }
                }
            }
        """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Assignment Operator parsing not implemented.")


def test_bitwise_operators():
    code = """
        shader LightControl {
        vertex {
            input vec3 position;
            output int isLightOn;
            void main() {        
                    isLightOn = 2 >> 1;
            }
        }
        fragment {
            input int isLightOn;
            output vec4 fragColor;
            void main() {
                isLightOn = isLightOn << 1;
            }
        }
    }

    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise Shift not working")
