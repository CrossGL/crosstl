import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from typing import List


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


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct codegen not implemented.")


def test_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else {
                bloom = 0.0;
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
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
        pytest.fail("if statement codegen not implemented.")


def test_for_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            for (int i = 0; i < 10; i++) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            }
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
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
        pytest.fail("for statement codegen not implemented.")


def test_else_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else if (input.texCoord.x < 0.5) {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                output.color = vec4(0.5, 0.5, 0.5, 1.0);

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }
}

fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else if (bloom < 0.5) {
                bloom = 0.0;
            } else {
                bloom = 0.5;
            }
            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
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
        pytest.fail("else if codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    float result = add(1.0, 2.0);
                }
                
                float add(float a, float b) {
                    return a + b;
                }
            }
            """,
            "add(1.0, 2.0)",
        )
    ],
)
def test_function_call(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)
    
    assert expected_output in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a |= 2;
                }
            }
            """,
            "a |= 2",
        )
    ],
)
def test_assignment_or_operator(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)
    
    assert expected_output in generated_code


def test_assignment_modulus_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 10;
                a %= 3;  // Assignment modulus operator
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a %= 3" in generated_code or "a = a % 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment modulus operator codegen not implemented.")


def test_assignment_xor_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 5;
                a ^= 3;  // Assignment XOR operator
            }
        }
    }

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a ^= 3" in generated_code or "a = a ^ 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment XOR operator codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a <<= 2;
                    a >>= 1;
                }
            }
            """,
            ["a <<= 2", "a >>= 1"],
        )
    ],
)
def test_assignment_shift_operators(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)
    
    for output in expected_output:
        assert output in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a | b;
                    int d = a & b;
                    int e = a ^ b;
                }
            }
            """,
            ["a | b", "a & b", "a ^ b"],
        )
    ],
)
def test_bitwise_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)
    
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_and_operator():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    sampler2D iChannel0;
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            // Use bitwise AND on texture coordinates (for testing purposes)
            output.color = vec4(float(int(input.texCoord.x * 100.0) & 15), 
                                float(int(input.texCoord.y * 100.0) & 15), 
                                0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Simple fragment shader to display the result of the AND operation
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise AND codegen not implemented")


def test_double_data_type():
    code = """
    shader DoubleShader {
        struct VSInput {
            double texCoord @ TEXCOORD0;
        };

        struct VSOutput {
            double color @ COLOR;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = input.texCoord * 2.0;
                return output;
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "double" in generated_code
    except SyntaxError:
        pytest.fail("Double data type not supported.")


# Test the codegen for the shift operators("<<", ">>")
@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a << b;
                    int d = a >> b;
                }
            }
            """,
            ["a << b", "a >> b"],
        )
    ],
)
def test_shift_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)
    
    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_or_operator():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    sampler2D iChannel0;
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            // Use bitwise OR on texture coordinates (for testing purposes)
            output.color = vec4(float(int(input.texCoord.x * 100.0) | 15), 
                                float(int(input.texCoord.y * 100.0) | 15), 
                                0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Simple fragment shader to display the result of the AND operation
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR codegen not implemented")


def test_directx_array_handling(array_test_data):
    """Test the DirectX code generator's handling of array types and array access."""
    code = """
    shader main {
    struct Particle {
        vec3 position;
        vec3 velocity;
    };

    struct Material {
        float values[4];  // Fixed-size array
        vec3 colors[];    // Dynamic array
    };

    cbuffer Constants {
        float weights[8];
        int indices[10];
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            
            // Array access in various forms
            float value = weights[2];
            int index = indices[5];
            
            // Array member access
            Material material;
            float x = material.values[0];
            vec3 color = material.colors[index];
            
            // Nested array access
            Particle particles[10];
            vec3 pos = particles[3].position;
            
            // Array access in expressions
            float sum = weights[0] + weights[1] + weights[2];
            
            return output;
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        
        # Use the fixture data for verification
        for expected in array_test_data["hlsl"]["array_type_declarations"]:
            assert expected in generated_code or expected.replace("[", "<").replace("]", ">") in generated_code
            
        for expected in array_test_data["hlsl"]["array_access"]:
            assert expected in generated_code
        
    except SyntaxError as e:
        pytest.fail(f"DirectX array codegen failed: {e}")


if __name__ == "__main__":
    pytest.main()
