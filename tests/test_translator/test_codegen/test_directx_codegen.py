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


def test_function_call():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vec4 addColor(vec4 color1, vec4 color2) {
        return color1 + color2;
    }

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
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
        pytest.fail("function call codegen not implemented.")


def test_assignment_or_operator():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            vUV.x |= 3.0;  // OR assignment operator
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
            // Apply bloom to the texture color
            vUV.x |= 3.0;  // OR assignment operator
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
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("OR operator codegen not implemented.")


def test_assignment_modulus_operator():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            uint a %= b;
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
            // Apply bloom to the texture color
            uint a %= b;
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
        pytest.fail("assignment modulus codegen not implemented.")


def test_assignment_modulus_operator():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            uint a ^= b;
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
            // Apply bloom to the texture color
            uint a ^= b;
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
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("assignment modulus codegen not implemented.")


def test_assignment_xor_operator():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            uint a ^= b;
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
            // Apply bloom to the texture color
            uint a ^= b;
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
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("assignment modulus codegen not implemented.")


def test_assignment_shift_operators():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            uint a >>= 1;
            uint b <<= 2;
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
            // Apply bloom to the texture color
            uint a >>= 1;
            uint b <<= 2;
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
        pytest.fail("Assignment shift codegen not implemented.")


def test_bitwise_operators():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            isLightOn = 2 & 1;
            isLightOn = 2 | 1;
            isLightOn = 2 ^ 1;
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
            // Apply bloom to the texture color
            isLightOn = 2 & 1;
            isLightOn = 2 | 1;
            isLightOn = 2 ^ 1;
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
        pytest.fail("Bitwise Shift codegen not implemented")


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
def test_shift_operators():
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
            output.color = addColor(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 1.0));
            uint a = 1 << 1;
            uint b = 2 >> 1;
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
            // Apply bloom to the texture color
            uint a = 1 << 1;
            uint b = 2 >> 1;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
    """

    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Shift right assignment operator codegen not implemented.")


if __name__ == "__main__":
    pytest.main()
