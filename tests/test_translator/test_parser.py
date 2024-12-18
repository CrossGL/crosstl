from crosstl.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.translator.parser import Parser


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


def test_struct_tokenization():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("if statement parsing not implemented.")


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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("for parsing not implemented.")


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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("else if statement parsing not implemented.")


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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("function call parsing not implemented.")


def test_assign_shift_right():
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
            uint b = 2;
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
            uint b = 2;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """

    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError as e:
        pytest.fail(f"Failed to parse ASSIGN_SHIFT_RIGHT token: {e}")


def test_logical_operators():
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
            // Using logical AND and logical OR operators
            if ((position.x > 0.3 && position.z < 0.7) || position.y > 0.5) {
                isLightOn = 1.0;
            } else {
                isLightOn = 0.0;
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
            // Apply bloom to the texture color
            // Using logical AND and logical OR operators
            if ((position.x > 0.3 && position.z < 0.7) || position.y > 0.5) {
                isLightOn = 1.0;
            } else {
                isLightOn = 0.0;
            }
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
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
            double noise = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
            double height = noise * 10.0;
            uint a = 1;
            uint b = 2;
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
            double noise = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
            double height = noise * 10.0;
            uint a = 1;
            uint b = 2;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
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
            int xStatus = int(position.x * 10.0);
            int yStatus = int(position.y * 10.0);
            int zStatus = int(position.z * 10.0);

            xStatus |= yStatus;
            yStatus &= zStatus;
            zStatus %= xStatus;
            lightStatus = xStatus;
            lightStatus ^= zStatus;
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
            int xStatus = int(position.x * 10.0);
            int yStatus = int(position.y * 10.0);
            int zStatus = int(position.z * 10.0);

            xStatus |= yStatus;
            yStatus &= zStatus;
            zStatus %= xStatus;
            lightStatus = xStatus;
            lightStatus ^= zStatus;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
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
            isLightOn = 2 >> 1;
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
            isLightOn = isLightOn << 1;
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
"""
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise Shift not working")


def test_xor_operator():
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
            vUV.x = float(int(position.x) ^ 3);  // XOR with 3
            vUV.y = float(int(position.y) ^ 5);  // XOR with 5
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
            vUV.x = float(int(position.x) ^ 3);  // XOR with 3
            vUV.y = float(int(position.y) ^ 5);  // XOR with 5
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);
            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise XOR not working")


def test_and_operator():
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
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise AND not working")


if __name__ == "__main__":
    pytest.main()
