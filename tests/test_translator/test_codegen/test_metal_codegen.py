from crosstl.translator.lexer import Lexer
import pytest
from typing import List
from crosstl.translator.parser import Parser
from crosstl.translator.codegen.metal_codegen import MetalCodeGen


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
    codegen = MetalCodeGen()
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
        pytest.fail("assignment shift codegen not implemented.")


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
            // Apply bloom to the texture color
            isLightOn = 2 >> 1;
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
        pytest.fail("Bitwise Shift codegen not implemented.")


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
            // Use bitwise AND on texture coordinates (for testing purposes)
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


def test_metal_texture_types():
    """Test proper conversion of sampler types to Metal texture types."""
    code = """
    shader main {
        sampler2D albedoMap;
        sampler2D environmentMap;
        sampler2D depthMap;
        
        struct VSOutput {
            vec2 texCoord;
            vec4 position @ gl_Position;
        };
        
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 albedo = texture(albedoMap, input.texCoord);
                vec3 normal = normalize(vec3(0.0, 1.0, 0.0));
                vec4 reflection = texture(environmentMap, normalize(normal));
                float depth = texture(depthMap, input.texCoord).r;
                
                return albedo * depth;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        
        # Verify proper Metal texture types
        assert "texture2d<float> albedoMap" in generated_code
        assert "texture2d<float> environmentMap" in generated_code
        assert "texture2d<float> depthMap" in generated_code
        
        # Verify sampling operations - metal_codegen uses the texture function, not .sample
        assert "texture( albedoMap" in generated_code
        assert "normalize" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal texture type conversion failed: {e}")


def test_metal_attributes_semantics():
    """Test the conversion of CrossGL semantics to Metal attributes."""
    code = """
    shader main {
        struct VSInput {
            vec3 position @ POSITION;
            vec3 normal @ NORMAL;
            vec2 texCoord @ TEXCOORD0;
        };
        
        struct VSOutput {
            vec4 position @ gl_Position;
            vec3 worldNormal;
            vec2 texCoord;
        };
        
        struct FSOutput {
            vec4 color @ gl_FragColor;
        };
        
        vertex {
            VSOutput main(VSInput input, int vertexID @ gl_VertexID) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.worldNormal = input.normal;
                output.texCoord = input.texCoord;
                return output;
            }
        }
        
        fragment {
            FSOutput main(VSOutput input) {
                FSOutput output;
                output.color = vec4(normalize(input.worldNormal) * 0.5 + 0.5, 1.0);
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        
        # Check Metal attributes for vertex inputs
        assert "[[attribute(0)]]" in generated_code  # POSITION
        assert "[[attribute(1)]]" in generated_code  # NORMAL
        assert "[[attribute(5)]]" in generated_code  # TEXCOORD0
        
        # Check Metal attributes for vertex outputs
        assert "[[position]]" in generated_code  # gl_Position
        
        # Check Metal attributes for fragment outputs
        assert "[[color(0)]]" in generated_code  # gl_FragColor
        
        # Check vertex ID
        assert "[[vertex_id]]" in generated_code or "[[stage_in]]" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal attributes/semantics conversion failed: {e}")


def test_metal_vector_type_conversions():
    """Test proper conversion of vector types to Metal types."""
    code = """
    shader main {
        struct TestTypes {
            vec2 vec2Field;
            vec3 vec3Field;
            vec4 vec4Field;
            ivec2 ivec2Field;
            ivec3 ivec3Field;
            ivec4 ivec4Field;
            mat2 mat2Field;
            mat3 mat3Field;
            mat4 mat4Field;
        };
        
        vertex {
            TestTypes main() {
                TestTypes output;
                output.vec2Field = vec2(1.0, 2.0);
                output.vec3Field = vec3(1.0, 2.0, 3.0);
                output.vec4Field = vec4(1.0, 2.0, 3.0, 4.0);
                output.ivec2Field = ivec2(1, 2);
                output.ivec3Field = ivec3(1, 2, 3);
                output.ivec4Field = ivec4(1, 2, 3, 4);
                
                // Set matrix fields
                output.mat2Field = mat2(1.0, 0.0, 0.0, 1.0);
                output.mat3Field = mat3(1.0);
                output.mat4Field = mat4(1.0);
                
                return output;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        
        # Check vector type conversions
        assert "float2 vec2Field" in generated_code
        assert "float3 vec3Field" in generated_code
        assert "float4 vec4Field" in generated_code
        assert "int2 ivec2Field" in generated_code
        assert "int3 ivec3Field" in generated_code
        assert "int4 ivec4Field" in generated_code
        
        # Check matrix type conversions
        assert "float2x2 mat2Field" in generated_code
        assert "float3x3 mat3Field" in generated_code
        assert "float4x4 mat4Field" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal vector type conversion failed: {e}")


def test_metal_texture_sampling():
    """Test the proper conversion of texture sampling operations to Metal."""
    code = """
    shader main {
        sampler2D colorMap;
        sampler2D normalMap;
        sampler2D envMap;
        
        struct VSOutput {
            vec2 texCoord;
            vec3 normal;
            vec3 viewDir;
        };
        
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                // Basic sampling
                vec4 color = texture(colorMap, input.texCoord);
                
                // Component selection after sampling
                vec3 normalTS = texture(normalMap, input.texCoord).rgb;
                
                // Sampling with direction
                vec3 reflectDir = normalize(input.normal);
                vec4 reflectionColor = texture(envMap, input.texCoord);
                
                // Combined result
                return color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        
        # Check texture declarations
        assert "texture2d<float> colorMap" in generated_code
        assert "texture2d<float> normalMap" in generated_code
        assert "texture2d<float> envMap" in generated_code
        
        # Check sampling operations - metal_codegen uses the texture function, not .sample
        assert "texture( colorMap" in generated_code
        assert "texture( normalMap" in generated_code
        assert "texture( envMap" in generated_code
        
        # Check component selection
        assert ".rgb" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal texture sampling conversion failed: {e}")


def test_metal_constant_buffer():
    """Test the conversion of constant buffers to Metal."""
    code = """
    shader main {
        struct MaterialParams {
            vec4 baseColor;
            float metallic;
            float roughness;
            vec2 textureScale;
        };
        
        MaterialParams material;
        
        struct VSOutput {
            vec2 texCoord;
        };
        
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec2 scaledTexCoord = input.texCoord * material.textureScale;
                vec4 finalColor = material.baseColor;
                finalColor.a = finalColor.a * (1.0 - material.roughness);
                finalColor.rgb = finalColor.rgb * material.metallic;
                return finalColor;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        
        # Check material struct declaration
        assert "struct MaterialParams" in generated_code
        
        # Check member access
        assert "material.baseColor" in generated_code
        assert "material.metallic" in generated_code
        assert "material.roughness" in generated_code
        assert "material.textureScale" in generated_code
    except SyntaxError as e:
        pytest.fail(f"Metal constant buffer conversion failed: {e}")


if __name__ == "__main__":
    pytest.main()
