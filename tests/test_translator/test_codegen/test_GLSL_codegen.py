import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from typing import List


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.get_tokens()


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
    codegen = GLSLCodeGen()
    return codegen.generate(ast_node)


def test_input_output():
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
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


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
            for (int i = 0; i < 10; i=i+1) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
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
            for (int i = 0; i < 10; i++) {
                if (i > 0.5) {
                    bloom = 0.5;
                } else {
                    bloom = 0.0;
                }
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
        pytest.fail("Struct parsing not implemented.")


def test_else_statement():
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
        pytest.fail("Struct parsing not implemented.")


def test_codegen_emits_version_when_absent_in_crossgl():
    shader = """
    shader simple {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        vertex {
            VSOut main() {
                VSOut o;
                o.position = vec4(0.0, 0.0, 0.0, 1.0);
                return o;
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)
    glsl_code = GLSLCodeGen().generate(ast)
    assert glsl_code.lstrip().startswith("#version 450 core")


def test_geometry_stage_entrypoint():
    code = """
    shader geom {
        geometry {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "void main()" in generated


def test_glsl_texture_and_atomic_builtins():
    code = """
    shader main {
        compute {
            void main() {
                sampler2D tex;
                image2D img;
                atomic_uint counter;
                vec4 g = textureGatherOffset(tex, vec2(0.5), ivec2(1));
                uint v = imageAtomicAdd(img, ivec2(0, 0), 1);
                uint c = atomicCounterIncrement(counter);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "textureGatherOffset" in generated
    assert "imageAtomicAdd" in generated
    assert "atomicCounterIncrement" in generated


def test_glsl_wave_and_mesh_intrinsics():
    code = """
    shader main {
        compute {
            void main() {
                uint v;
                uint sum = WaveActiveSum(v);
                SetMeshOutputCounts(64, 32);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "WaveActiveSum" in generated
    assert "SetMeshOutputCounts" in generated


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
            } else if (bloom < 0.5) {
                bloom = 0.1;
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
        pytest.fail("Struct parsing not implemented.")


def test_function_call():
    code = """
    shader perlinNoise {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    vec2 perlinNoise(vec2 uv) {
        return vec2(0.0, 0.0);
    }
    
    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            vec2 noise = perlinNoise(input.texCoord);
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
            vec2 noise = perlinNoise(input.color.xy);
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
        pytest.fail("Struct parsing not implemented.")


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
            int x = 2;
            x >>= 1;
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
            int x = 2;
            x <<= 1;

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
        pytest.fail("Struct parsing not implemented.")


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
            int x = 2;
            x = x & 1;
            x = x | 1;
            x = x + 1;

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
            int x = 2+ 2;
            x = x + 1;
            x = x + 1;
            x = x + 1;

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
        pytest.fail("Bitwise Shift parsing not implemented.")


def test_opengl_array_handling(array_test_data):
    """Test the OpenGL code generator's handling of array types and array access."""
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
        for expected in array_test_data["glsl"]["array_type_declarations"]:
            assert expected in generated_code

        for expected in array_test_data["glsl"]["array_access"]:
            assert expected in generated_code

    except SyntaxError as e:
        pytest.fail(f"OpenGL array codegen failed: {e}")


@pytest.mark.parametrize(
    "array_shader, expected_outputs",
    [
        (
            """
            shader TestArrayShader {
                struct DataArray {
                    float values[4];
                };
                
                void main() {
                    DataArray data;
                    data.values[2] = 1.0;
                    
                    float arr[5];
                    arr[0] = 1.0;
                }
            }
            """,
            ["float values[4]", "data.values[2]", "arr[0]"],
        )
    ],
)
def test_array_handling(array_shader, expected_outputs):
    try:
        ast = crosstl.translator.parse(array_shader)
        code_gen = GLSLCodeGen()
        generated_code = code_gen.generate(ast)

        for expected in expected_outputs:
            assert expected in generated_code
    except Exception as e:
        pytest.fail(f"OpenGL array codegen failed: {e}")


def test_opengl_local_array_declarations_use_glsl_order():
    shader = """
    shader TestShader {
        void main() {
            vec3 localColors[4];
            float weights[8];
            localColors[0] = vec3(1.0, 0.0, 0.0);
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "vec3 localColors[4];" in generated_code
    assert "float weights[8];" in generated_code
    assert "vec3[4] localColors" not in generated_code
    assert "float[8] weights" not in generated_code


def test_opengl_array_parameters_use_glsl_order():
    shader = """
    shader TestShader {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "float accumulate(float weights[4], vec3 colors[2])" in generated_code
    assert "float[4] weights" not in generated_code
    assert "vec3[2] colors" not in generated_code


def test_opengl_non_resource_arrays_preserve_expression_sizes():
    shader = """
    shader ArrayExpressionSizes {
        struct Payload {
            vec3 colors[(2 + 1) * 2];
            float weights[+6];
        };

        float accumulate(float values[(2 + 1) * 2], vec3 normals[+6]) {
            float localWeights[(2 + 1) * 2];
            vec3 localNormals[+6];
            return values[2] + normals[2].x + localWeights[2] + localNormals[2].x;
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "vec3 colors[((2 + 1) * 2)];" in generated_code
    assert "float weights[(+6)];" in generated_code
    assert (
        "float accumulate(float values[((2 + 1) * 2)], vec3 normals[(+6)])"
        in generated_code
    )
    assert "float localWeights[((2 + 1) * 2)];" in generated_code
    assert "vec3 localNormals[(+6)];" in generated_code
    assert "float values[]" not in generated_code
    assert "float localWeights[]" not in generated_code


def test_opengl_sampler_globals_are_uniform_resources():
    shader = """
    shader TextureShader {
        sampler2D colorMap;
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
            vec3 normal;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 color = texture(colorMap, input.uv);
                vec4 env = texture(envMap, input.normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCube envMap;" in generated_code
    assert "layout(std140, binding = 0) sampler2D" not in generated_code
    assert "out vec2 uv;" in generated_code
    assert "out vec3 normal;" in generated_code
    assert "VectorType(" not in generated_code


def test_opengl_texture_array_resources_and_indexed_sampling():
    shader = """
    shader TextureArrayShader {
        sampler2D textures[4];
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
            vec3 normal;
            int layer;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 color = texture(textures[input.layer], input.uv);
                vec4 env = texture(envMap, input.normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform samplerCube envMap;" in generated_code
    assert "texture(textures[input.layer], input.uv)" in generated_code
    assert "texture(envMap, input.normal)" in generated_code


def test_opengl_fixed_texture_array_keeps_declared_size_with_constant_indices():
    shader = """
    shader FixedArrayConstantIndex {
        const int LAYER = 2;
        sampler2D textures[6];
        sampler samplers[6];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[6], sampler samplers[6], vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv) + texture(textures[1 + 2], samplers[1 + 2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 2;" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[6];" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[6], vec2 uv)" in generated_code
    assert "texture(textures[LAYER], uv)" in generated_code
    assert "texture(textures[(1 + 2)], uv)" in generated_code
    assert "sampleLayer(textures, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" not in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code
    assert "samplers[(1 + 2)]" not in generated_code


def test_opengl_fixed_texture_array_resolves_constant_declared_size_for_bindings():
    shader = """
    shader ConstSizedResourceArrays {
        const int BASE_COUNT = 2;
        const int TEXTURE_COUNT = BASE_COUNT * 3;
        sampler2D textures[TEXTURE_COUNT];
        sampler samplers[TEXTURE_COUNT];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[TEXTURE_COUNT], sampler samplers[TEXTURE_COUNT], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE_COUNT = 2;" in generated_code
    assert "const int TEXTURE_COUNT = (BASE_COUNT * 3);" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2D textures[TEXTURE_COUNT];"
        in generated_code
    )
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[TEXTURE_COUNT], vec2 uv)" in generated_code
    )
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleLayer(textures, input.uv)" in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code


def test_opengl_fixed_texture_array_resolves_inline_declared_size_expression_for_bindings():
    shader = """
    shader ExprSizedResourceArrays {
        sampler2D textures[2 * 3];
        sampler samplers[2 * 3];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[2 * 3], sampler samplers[2 * 3], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[(2 * 3)];" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[(2 * 3)], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleLayer(textures, input.uv)" in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "[None]" not in generated_code


def test_opengl_fixed_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
    shader = """
    shader ParenthesizedSizedResourceArrays {
        sampler2D textures[(2 + 1) * 2];
        sampler samplers[(2 + 1) * 2];
        sampler2D unaryTextures[+6];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[(2 + 1) * 2], sampler samplers[(2 + 1) * 2], sampler2D unaryTextures[+6], vec2 uv) {
            return texture(textures[2], samplers[2], uv) + texture(unaryTextures[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, unaryTextures, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2D textures[((2 + 1) * 2)];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform sampler2D unaryTextures[(+6)];" in generated_code
    )
    assert "layout(binding = 12) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[((2 + 1) * 2)], sampler2D unaryTextures[(+6)], vec2 uv)"
        in generated_code
    )
    assert "sampleLayer(textures, unaryTextures, input.uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "texture(unaryTextures[2], uv)" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "[None]" not in generated_code


def test_opengl_texture_array_helper_parameter_keeps_sampler_array():
    shader = """
    shader TextureArrayHelper {
        sampler2D textures[4];

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv) {
            return texture(textures[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, input.layer, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated_code
    )
    assert "texture(textures[layer], uv)" in generated_code
    assert "sampleLayer(textures, input.layer, input.uv)" in generated_code


def test_opengl_texture_array_helper_parameter_filters_indexed_sampler_array():
    shader = """
    shader SamplerArrayHelper {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], int layer, vec2 uv) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated_code
    )
    assert "texture(textures[layer], uv)" in generated_code
    assert "sampleLayer(textures, input.layer, input.uv)" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[layer]" not in generated_code


def test_opengl_unsized_texture_array_infers_helper_size_and_filters_sampler_array():
    shader = """
    shader UnsizedSampledResourceArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 color = texture(textures[2], samplers[2], uv);
            vec4 other = texture(textures[1], samplers[1], uv);
            return color + other;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[3];" in generated_code
    assert "layout(binding = 3) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[3], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "texture(textures[1], uv)" in generated_code
    assert "sampleLayer(textures, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[2]" not in generated_code


def test_opengl_unsized_texture_array_infers_transitive_helper_size():
    shader = """
    shader MultiHopUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleDeep(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 high = texture(textures[4], samplers[4], uv);
            vec4 low = texture(textures[1], samplers[1], uv);
            return high + low;
        }

        vec4 sampleMid(sampler2D textures[], sampler samplers[], vec2 uv) {
            return sampleDeep(textures, samplers, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleMid(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[5];" in generated_code
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleDeep(sampler2D textures[5], vec2 uv)" in generated_code
    assert "vec4 sampleMid(sampler2D textures[5], vec2 uv)" in generated_code
    assert "texture(textures[4], uv)" in generated_code
    assert "texture(textures[1], uv)" in generated_code
    assert "sampleDeep(textures, uv)" in generated_code
    assert "sampleMid(textures, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[4]" not in generated_code


def test_opengl_unsized_texture_array_preserves_dynamic_indexing():
    shader = """
    shader MixedIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int layer, vec2 uv) {
            vec4 dynamicColor = texture(textures[layer], samplers[layer], uv);
            vec4 fixedColor = texture(textures[3], samplers[3], uv);
            return dynamicColor + fixedColor;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated_code
    )
    assert "texture(textures[layer], uv)" in generated_code
    assert "texture(textures[3], uv)" in generated_code
    assert "sampleLayer(textures, input.layer, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[layer]" not in generated_code


def test_opengl_unsized_texture_array_ignores_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int layer, vec2 uv) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[-1], samplers[-1], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    dynamic_code = GLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = GLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "layout(binding = 0) uniform sampler2D textures[];" in dynamic_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" in dynamic_code
    assert "vec4 sampleLayer(sampler2D textures[], int layer, vec2 uv)" in dynamic_code
    assert "texture(textures[layer], uv)" in dynamic_code
    assert "layout(binding = 0) uniform sampler2D textures[1];" not in dynamic_code
    assert "layout(binding = 2) uniform sampler2D afterTexture;" not in dynamic_code
    assert "sampler samplers" not in dynamic_code
    assert "samplers[layer]" not in dynamic_code

    assert "layout(binding = 0) uniform sampler2D textures[];" in negative_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" in negative_code
    assert "vec4 sampleLayer(sampler2D textures[], vec2 uv)" in negative_code
    assert "texture(textures[(-1)], uv)" in negative_code
    assert "layout(binding = 0) uniform sampler2D textures[0];" not in negative_code
    assert "layout(binding = 0) uniform sampler2D afterTexture;" not in negative_code
    assert "sampler samplers" not in negative_code


def test_opengl_unsized_texture_array_infers_constant_expression_size():
    shader = """
    shader ExprIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[1 + 2], samplers[1 + 2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[(1 + 2)], uv)" in generated_code
    assert "sampleLayer(textures, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[(1 + 2)]" not in generated_code


def test_opengl_unsized_texture_array_infers_named_constant_size():
    shader = """
    shader ConstIndexedUnsizedSampledResources {
        const int BASE = 1;
        const int LAYER = BASE + 2;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE = 1;" in generated_code
    assert "const int LAYER = (BASE + 2);" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[LAYER], uv)" in generated_code
    assert "sampleLayer(textures, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code


def test_opengl_unsized_texture_array_ignores_shadowed_constant_name():
    shader = """
    shader ShadowedConstIndex {
        const int LAYER = 3;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int LAYER, vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 3;" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[];" in generated_code
    assert "layout(binding = 1) uniform sampler2D afterTexture;" in generated_code
    assert (
        "vec4 sampleLayer(sampler2D textures[], int LAYER, vec2 uv)" in generated_code
    )
    assert "texture(textures[LAYER], uv)" in generated_code
    assert "sampleLayer(textures, input.layer, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" not in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code


def test_opengl_texture_array_helper_operation_variants_filter_sampler_array():
    shader = """
    shader TextureArrayOps {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
            int layer;
        };

        vec4 sampleOps(sampler2D textures[4], sampler samplers[4], int layer, vec2 uv, ivec2 pixel) {
            vec4 lodColor = textureLod(textures[layer], samplers[layer], uv, 2.0);
            vec4 gradColor = textureGrad(textures[layer], samplers[layer], uv, vec2(0.1), vec2(0.2));
            vec4 gathered = textureGather(textures[layer], samplers[layer], uv);
            vec4 fetched = texelFetch(textures[layer], pixel, 0);
            return lodColor + gradColor + gathered + fetched;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleOps(textures, samplers, input.layer, input.uv, input.pixel);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert (
        "vec4 sampleOps(sampler2D textures[4], int layer, vec2 uv, ivec2 pixel)"
        in generated_code
    )
    assert "textureLod(textures[layer], uv, 2.0)" in generated_code
    assert "textureGrad(textures[layer], uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "textureGather(textures[layer], uv)" in generated_code
    assert "texelFetch(textures[layer], pixel, 0)" in generated_code
    assert "sampleOps(textures, input.layer, input.uv, input.pixel)" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[layer]" not in generated_code


def test_opengl_array_texture_types_and_shadow_compare_coordinates():
    shader = """
    shader ArrayTextureTypes {
        sampler2DArray colorArray;
        sampler2DArrayShadow shadowArray;
        samplerCubeShadow cubeShadow;
        sampler arraySampler;
        sampler shadowSampler;

        struct VSOutput {
            vec3 uvLayer;
            ivec3 pixelLayer;
            vec3 direction;
            float depth;
        };

        vec4 sampleArray(sampler2DArray tex, sampler s, vec3 uvLayer, ivec3 pixelLayer) {
            vec4 color = texture(tex, s, uvLayer);
            vec4 lodColor = textureLod(tex, s, uvLayer, 1.0);
            vec4 gradColor = textureGrad(tex, s, uvLayer, vec2(0.1), vec2(0.2));
            vec4 gathered = textureGather(tex, s, uvLayer);
            vec4 fetched = texelFetch(tex, pixelLayer, 0);
            return color + lodColor + gradColor + gathered + fetched;
        }

        float sampleShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth) {
            return textureCompare(tex, s, uvLayer, depth);
        }

        float sampleCubeShadow(samplerCubeShadow tex, sampler s, vec3 direction, float depth) {
            return textureCompare(tex, s, direction, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float shadow = sampleShadowArray(shadowArray, shadowSampler, input.uvLayer, input.depth);
                float cube = sampleCubeShadow(cubeShadow, shadowSampler, input.direction, input.depth);
                return sampleArray(colorArray, arraySampler, input.uvLayer, input.pixelLayer) * shadow * cube;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DArray colorArray;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert "layout(binding = 2) uniform samplerCubeShadow cubeShadow;" in generated_code
    assert (
        "vec4 sampleArray(sampler2DArray tex, vec3 uvLayer, ivec3 pixelLayer)"
        in generated_code
    )
    assert "texture(tex, uvLayer)" in generated_code
    assert "textureLod(tex, uvLayer, 1.0)" in generated_code
    assert "textureGrad(tex, uvLayer, vec2(0.1), vec2(0.2))" in generated_code
    assert "textureGather(tex, uvLayer)" in generated_code
    assert "texelFetch(tex, pixelLayer, 0)" in generated_code
    assert (
        "float sampleShadowArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth)"
        in generated_code
    )
    assert "texture(tex, vec4(uvLayer, depth))" in generated_code
    assert (
        "float sampleCubeShadow(samplerCubeShadow tex, vec3 direction, float depth)"
        in generated_code
    )
    assert "texture(tex, vec4(direction, depth))" in generated_code
    assert "sampler s" not in generated_code
    assert "arraySampler" not in generated_code
    assert "shadowSampler" not in generated_code
    assert "vec3(uvLayer, depth)" not in generated_code
    assert "vec3(direction, depth)" not in generated_code


def test_opengl_cube_array_and_multisample_texture_types():
    shader = """
    shader CubeMsResources {
        samplerCubeArray cubeArray;
        samplerCubeArrayShadow cubeArrayShadow;
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler cubeSampler;
        sampler shadowSampler;

        struct VSOutput {
            vec4 cubeLayer;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
            float depth;
        };

        vec4 sampleCubeArray(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
            return texture(tex, s, cubeLayer) + textureLod(tex, s, cubeLayer, 2.0);
        }

        float sampleCubeArrayShadow(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureCompare(tex, s, cubeLayer, depth);
        }

        vec4 fetchMs(sampler2DMS tex, sampler2DMSArray texArray, ivec2 pixel, ivec3 pixelLayer, int sampleIndex) {
            return texelFetch(tex, pixel, sampleIndex) + texelFetch(texArray, pixelLayer, sampleIndex);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float shadow = sampleCubeArrayShadow(cubeArrayShadow, shadowSampler, input.cubeLayer, input.depth);
                return sampleCubeArray(cubeArray, cubeSampler, input.cubeLayer) * shadow + fetchMs(msTex, msArray, input.pixel, input.pixelLayer, input.sampleIndex);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeArrayShadow;"
        in generated_code
    )
    assert "layout(binding = 2) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 3) uniform sampler2DMSArray msArray;" in generated_code
    assert (
        "vec4 sampleCubeArray(samplerCubeArray tex, vec4 cubeLayer)" in generated_code
    )
    assert "texture(tex, cubeLayer)" in generated_code
    assert "textureLod(tex, cubeLayer, 2.0)" in generated_code
    assert (
        "float sampleCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "texture(tex, cubeLayer, depth)" in generated_code
    assert (
        "vec4 fetchMs(sampler2DMS tex, sampler2DMSArray texArray, ivec2 pixel, ivec3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "texelFetch(tex, pixel, sampleIndex)" in generated_code
    assert "texelFetch(texArray, pixelLayer, sampleIndex)" in generated_code
    assert "cubeSampler" not in generated_code
    assert "shadowSampler" not in generated_code
    assert "vec4(cubeLayer, depth)" not in generated_code


def test_opengl_storage_image_load_store():
    shader = """
    shader StorageImages {
        image2D outputImage;
        image3D volumeImage;
        image2DArray layerImage;

        vec4 touchImages(image2D outImg, image3D volume, image2DArray layers, ivec2 pixel, ivec3 voxel, ivec3 pixelLayer) {
            vec4 color = imageLoad(outImg, pixel);
            vec4 volumeColor = imageLoad(volume, voxel);
            vec4 layerColor = imageLoad(layers, pixelLayer);
            imageStore(outImg, pixel, color + layerColor);
            imageStore(volume, voxel, volumeColor);
            imageStore(layers, pixelLayer, color);
            return color + volumeColor + layerColor;
        }

        compute {
            void main() {
                vec4 result = touchImages(outputImage, volumeImage, layerImage, ivec2(0, 1), ivec3(0, 1, 2), ivec3(3, 4, 5));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba32f, binding = 0) uniform image2D outputImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert (
        "vec4 touchImages(image2D outImg, image3D volume, image2DArray layers, ivec2 pixel, ivec3 voxel, ivec3 pixelLayer)"
        in generated_code
    )
    assert "vec4 color = imageLoad(outImg, pixel);" in generated_code
    assert "vec4 volumeColor = imageLoad(volume, voxel);" in generated_code
    assert "vec4 layerColor = imageLoad(layers, pixelLayer);" in generated_code
    assert "imageStore(outImg, pixel, (color + layerColor));" in generated_code
    assert "imageStore(volume, voxel, volumeColor);" in generated_code
    assert "imageStore(layers, pixelLayer, color);" in generated_code


def test_opengl_storage_image_format_layouts():
    shader = """
    shader ImageFormats {
        sampler2D sampledColor;
        image2D colorOut;
        image3D volumeOut;
        image2DArray layerOut;
        iimage2D signedOut;
        iimage3D signedVolumeOut;
        iimage2DArray signedLayerOut;
        uimage2D unsignedOut;
        uimage3D unsignedVolumeOut;
        uimage2DArray unsignedLayerOut;
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D sampledColor;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image2D colorOut;" in generated_code
    assert "layout(rgba32f, binding = 2) uniform image3D volumeOut;" in generated_code
    assert (
        "layout(rgba32f, binding = 3) uniform image2DArray layerOut;" in generated_code
    )
    assert "layout(r32i, binding = 4) uniform iimage2D signedOut;" in generated_code
    assert (
        "layout(r32i, binding = 5) uniform iimage3D signedVolumeOut;" in generated_code
    )
    assert (
        "layout(r32i, binding = 6) uniform iimage2DArray signedLayerOut;"
        in generated_code
    )
    assert "layout(r32ui, binding = 7) uniform uimage2D unsignedOut;" in generated_code
    assert (
        "layout(r32ui, binding = 8) uniform uimage3D unsignedVolumeOut;"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 9) uniform uimage2DArray unsignedLayerOut;"
        in generated_code
    )


def test_opengl_explicit_storage_image_format_layouts():
    shader = """
    shader ExplicitImageFormats {
        image2D halfColor @rgba16f;
        image3D normalizedVolume @ rgba8;
        image2DArray layerColor @format(rgba16f);
        iimage2D signedImage @ r32i;
        uimage2D counters @r32ui;
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba16f, binding = 0) uniform image2D halfColor;" in generated_code
    assert (
        "layout(rgba8, binding = 1) uniform image3D normalizedVolume;" in generated_code
    )
    assert (
        "layout(rgba16f, binding = 2) uniform image2DArray layerColor;"
        in generated_code
    )
    assert "layout(r32i, binding = 3) uniform iimage2D signedImage;" in generated_code
    assert "layout(r32ui, binding = 4) uniform uimage2D counters;" in generated_code


def test_opengl_explicit_scalar_image_formats():
    shader = """
    shader ExplicitScalarImageFormats {
        image2D scalarFloat @r32f;
        image3D signedVolume @ r32i;
        image2DArray unsignedLayers @format(r32ui);

        float touchFloat(image2D image @r32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int touchSigned(image3D image @r32i, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint touchUnsigned(image2DArray image @format(r32ui), ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchFloat(scalarFloat, ivec2(0, 1), 0.5);
                int b = touchSigned(signedVolume, ivec3(1, 2, 3), -4);
                uint c = touchUnsigned(unsignedLayers, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32f, binding = 0) uniform image2D scalarFloat;" in generated_code
    assert "layout(r32i, binding = 1) uniform iimage3D signedVolume;" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2DArray unsignedLayers;"
        in generated_code
    )
    assert "float touchFloat(image2D image, ivec2 pixel, float value)" in generated_code
    assert "int touchSigned(iimage3D image, ivec3 voxel, int value)" in generated_code
    assert (
        "uint touchUnsigned(uimage2DArray image, ivec3 pixelLayer, uint value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "imageStore(image, pixel, vec4((oldValue + value)));" in generated_code
    assert "int oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "imageStore(image, voxel, ivec4((oldValue + value)));" in generated_code
    assert "uint oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixelLayer, uvec4((oldValue + value)));" in generated_code


def test_opengl_explicit_rg_image_formats():
    shader = """
    shader ExplicitRGImageFormats {
        image2D rgFloat @rg32f;
        image3D rgSigned @format(rg32i);
        image2DArray rgUnsigned @ rg32ui;

        vec2 touchFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        ivec2 touchSigned(image3D image @rg32i, ivec3 voxel, ivec2 value) {
            ivec2 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec2 touchUnsigned(image2DArray image @rg32ui, ivec3 pixelLayer, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rg32f, binding = 0) uniform image2D rgFloat;" in generated_code
    assert "layout(rg32i, binding = 1) uniform iimage3D rgSigned;" in generated_code
    assert (
        "layout(rg32ui, binding = 2) uniform uimage2DArray rgUnsigned;"
        in generated_code
    )
    assert "vec2 touchFloat(image2D image, ivec2 pixel, vec2 value)" in generated_code
    assert (
        "ivec2 touchSigned(iimage3D image, ivec3 voxel, ivec2 value)" in generated_code
    )
    assert (
        "uvec2 touchUnsigned(uimage2DArray image, ivec3 pixelLayer, uvec2 value)"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert "ivec2 oldValue = imageLoad(image, voxel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(image, pixelLayer).xy;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, voxel, ivec4((oldValue + value), 0, 0));" in generated_code
    )
    assert (
        "imageStore(image, pixelLayer, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert "layout(rg32i, binding = 1) uniform image3D" not in generated_code
    assert "layout(rg32ui, binding = 2) uniform image2DArray" not in generated_code
    assert "imageLoad(image, pixel).x;" not in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code


def test_opengl_explicit_narrow_rg_image_formats():
    shader = """
    shader ExplicitNarrowRGImageFormats {
        image2D rg8Float @rg8;
        image2D rg8Snorm @rg8_snorm;
        image3D rg16Float @format(rg16);
        image2D rg16Snorm @rg16_snorm;
        image2DArray rg16Half @ rg16f;
        image2D rg8Signed @rg8i;
        image3D rg16Signed @format(rg16i);
        image2D rg8Unsigned @rg8ui;
        image2DArray rg16Unsigned @format(rg16ui);

        vec2 touchFloat(image2D image @rg8, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchHalf(image2DArray image @rg16f, ivec3 pixelLayer, vec2 value) {
            vec2 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        ivec2 touchSigned(image3D image @rg16i, ivec3 voxel, ivec2 value) {
            ivec2 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec2 touchUnsigned(image2D image @rg8ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rg8, binding = 0) uniform image2D rg8Float;" in generated_code
    assert "layout(rg8_snorm, binding = 1) uniform image2D rg8Snorm;" in generated_code
    assert "layout(rg16, binding = 2) uniform image3D rg16Float;" in generated_code
    assert (
        "layout(rg16_snorm, binding = 3) uniform image2D rg16Snorm;" in generated_code
    )
    assert "layout(rg16f, binding = 4) uniform image2DArray rg16Half;" in generated_code
    assert "layout(rg8i, binding = 5) uniform iimage2D rg8Signed;" in generated_code
    assert "layout(rg16i, binding = 6) uniform iimage3D rg16Signed;" in generated_code
    assert "layout(rg8ui, binding = 7) uniform uimage2D rg8Unsigned;" in generated_code
    assert (
        "layout(rg16ui, binding = 8) uniform uimage2DArray rg16Unsigned;"
        in generated_code
    )
    assert "vec2 touchFloat(image2D image, ivec2 pixel, vec2 value)" in generated_code
    assert (
        "vec2 touchHalf(image2DArray image, ivec3 pixelLayer, vec2 value)"
        in generated_code
    )
    assert (
        "ivec2 touchSigned(iimage3D image, ivec3 voxel, ivec2 value)" in generated_code
    )
    assert (
        "uvec2 touchUnsigned(uimage2D image, ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert "vec2 oldValue = imageLoad(image, pixelLayer).xy;" in generated_code
    assert "ivec2 oldValue = imageLoad(image, voxel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, pixelLayer, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, voxel, ivec4((oldValue + value), 0, 0));" in generated_code
    )
    assert (
        "imageStore(image, pixel, uvec4((oldValue + value), 0u, 0u));" in generated_code
    )
    assert "layout(rg8i, binding = 5) uniform image2D" not in generated_code
    assert "layout(rg8ui, binding = 7) uniform image2D" not in generated_code
    assert "imageLoad(image, pixel).x;" not in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code


def test_opengl_explicit_rgba_float_image_formats():
    shader = """
    shader ExplicitRGBAFloatFormats {
        image2D rgba8Color @rgba8;
        image2D rgba8Snorm @rgba8_snorm;
        image3D rgba16Color @format(rgba16);
        image2D rgba16Snorm @rgba16_snorm;
        image2DArray rgba16Half @ rgba16f;
        image3D rgba32Float @format(rgba32f);

        vec4 touchColor(image2D image @rgba8, ivec2 pixel, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec4 touchHalf(image2DArray image @rgba16f, ivec3 pixelLayer, vec4 value) {
            vec4 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        vec4 touchFloat(image3D image @rgba32f, ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        vec4 typedOverride(iimage2D image @rgba16f, ivec2 pixel, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba8, binding = 0) uniform image2D rgba8Color;" in generated_code
    assert (
        "layout(rgba8_snorm, binding = 1) uniform image2D rgba8Snorm;" in generated_code
    )
    assert "layout(rgba16, binding = 2) uniform image3D rgba16Color;" in generated_code
    assert (
        "layout(rgba16_snorm, binding = 3) uniform image2D rgba16Snorm;"
        in generated_code
    )
    assert (
        "layout(rgba16f, binding = 4) uniform image2DArray rgba16Half;"
        in generated_code
    )
    assert "layout(rgba32f, binding = 5) uniform image3D rgba32Float;" in generated_code
    assert "vec4 touchColor(image2D image, ivec2 pixel, vec4 value)" in generated_code
    assert (
        "vec4 touchHalf(image2DArray image, ivec3 pixelLayer, vec4 value)"
        in generated_code
    )
    assert "vec4 touchFloat(image3D image, ivec3 voxel, vec4 value)" in generated_code
    assert (
        "vec4 typedOverride(image2D image, ivec2 pixel, vec4 value)" in generated_code
    )
    assert "vec4 oldValue = imageLoad(image, pixel);" in generated_code
    assert "vec4 oldValue = imageLoad(image, pixelLayer);" in generated_code
    assert "vec4 oldValue = imageLoad(image, voxel);" in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" in generated_code
    assert "imageStore(image, pixelLayer, (oldValue + value));" in generated_code
    assert "imageStore(image, voxel, (oldValue + value));" in generated_code
    assert "vec4 typedOverride(iimage2D image" not in generated_code
    assert "imageLoad(image, pixel).x" not in generated_code
    assert "imageLoad(image, pixel).xy" not in generated_code
    assert "imageStore(image, pixel, vec4(" not in generated_code


def test_opengl_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader FormattedImageArrays {
        image2D counters @r32ui[2];
        image2D rgPairs @rg16f[3];
        image3D rgbaVolumes @rgba16f[2];
        image2D afterCounters @r32ui;
        sampler2D sampled;

        uint touchCounters(image2D images[2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec4 touchVolumes(image3D images[2] @rgba16f, ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(images[1], voxel);
            imageStore(images[0], voxel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
                vec4 c = touchVolumes(rgbaVolumes, ivec3(1, 2, 3), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[2];" in generated_code
    assert "layout(rg16f, binding = 2) uniform image2D rgPairs[3];" in generated_code
    assert (
        "layout(rgba16f, binding = 5) uniform image3D rgbaVolumes[2];" in generated_code
    )
    assert (
        "layout(r32ui, binding = 7) uniform uimage2D afterCounters;" in generated_code
    )
    assert "layout(binding = 8) uniform sampler2D sampled;" in generated_code
    assert (
        "uint touchCounters(uimage2D images[2], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert (
        "vec4 touchVolumes(image3D images[2], ivec3 voxel, vec4 value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[1], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "vec4 oldValue = imageLoad(images[1], voxel);" in generated_code
    assert "imageStore(images[0], voxel, (oldValue + value));" in generated_code
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 2) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[1], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_formatted_image_arrays_preserve_expression_sizes():
    shader = """
    shader ExprFormattedImageArrays {
        image2D counters @r32ui[(1 + 1) * 2];
        image2D rgPairs @rg16f[+3];
        image2D afterCounters @r32ui;
        sampler2D sampled;

        uint touchCounters(image2D images[(1 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[+3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[((1 + 1) * 2)];"
        in generated_code
    )
    assert "layout(rg16f, binding = 4) uniform image2D rgPairs[(+3)];" in generated_code
    assert (
        "layout(r32ui, binding = 7) uniform uimage2D afterCounters;" in generated_code
    )
    assert "layout(binding = 8) uniform sampler2D sampled;" in generated_code
    assert (
        "uint touchCounters(uimage2D images[((1 + 1) * 2)], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[(+3)], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[2], pixel).x;" in generated_code
    assert "imageStore(images[1], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "uint a = touchCounters(counters, ivec2(1, 2), 3);" in generated_code
    assert "vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));" in generated_code
    assert (
        "layout(r32ui, binding = 4) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 4) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[2], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_unsized_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader UnsizedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[0], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[0], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[];" in generated_code
    assert "layout(rg16f, binding = 1) uniform image2D rgPairs[];" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[], ivec2 pixel, vec2 value)" in generated_code
    )
    assert "uint oldValue = imageLoad(images[0], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[0], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 1) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[0], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[0], pixel).x;" not in generated_code


def test_opengl_formatted_image_arrays_infer_named_constant_size():
    shader = """
    shader ConstSizedFormattedImageArrays {
        const int COUNT = 3;
        const int LAYER = COUNT - 1;
        image2D counters @r32ui[COUNT];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[COUNT] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int COUNT = 3;" in generated_code
    assert "const int LAYER = (COUNT - 1);" in generated_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[COUNT];" in generated_code
    )
    assert "layout(rg16f, binding = 3) uniform image2D rgPairs[3];" in generated_code
    assert (
        "layout(r32ui, binding = 6) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[COUNT], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairs(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert "imageStore(images[1], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[LAYER], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "uint a = touchCounters(counters, ivec2(1, 2), 3);" in generated_code
    assert "vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));" in generated_code
    assert "layout(rg16f, binding = 3) uniform image2D rgPairs[];" not in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "layout(r32ui, binding = 0) uniform image2D counters" not in generated_code
    assert "layout(rg16f, binding = 3) uniform uimage2D rgPairs" not in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[LAYER], pixel).x;" not in generated_code


def test_opengl_formatted_image_arrays_ignore_shadowed_local_constant():
    shader = """
    shader ShadowedImageConstIndex {
        const int LAYER = 3;
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            int LAYER = 0;
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 3;" in generated_code
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[];" in generated_code
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "uint a = touchCounters(counters, ivec2(1, 2), 3);" in generated_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[4];" not in generated_code
    )
    assert (
        "layout(r32ui, binding = 4) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code


def test_opengl_formatted_image_arrays_infer_transitive_helper_size():
    shader = """
    shader TransitiveFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCountersDeep(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint touchCountersMid(image2D images[] @r32ui, ivec2 pixel, uint value) {
            return touchCountersDeep(images, pixel, value);
        }

        vec2 touchPairsDeep(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairsMid(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            return touchPairsDeep(images, pixel, value);
        }

        compute {
            void main() {
                uint a = touchCountersMid(counters, ivec2(1, 2), 3);
                vec2 b = touchPairsMid(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[4];" in generated_code
    assert "layout(rg16f, binding = 4) uniform image2D rgPairs[3];" in generated_code
    assert (
        "layout(r32ui, binding = 7) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCountersDeep(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint touchCountersMid(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "vec2 touchPairsDeep(image2D images[3], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "vec2 touchPairsMid(image2D images[3], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "imageStore(images[1], pixel, uvec4((oldValue + value)));" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "return touchCountersDeep(images, pixel, value);" in generated_code
    assert "return touchPairsDeep(images, pixel, value);" in generated_code
    assert "uint a = touchCountersMid(counters, ivec2(1, 2), 3);" in generated_code
    assert "vec2 b = touchPairsMid(rgPairs, ivec2(2, 3), vec2(0.5));" in generated_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[];" not in generated_code
    )
    assert "layout(rg16f, binding = 4) uniform image2D rgPairs[];" not in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "uint oldValue = imageLoad(images[3], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_explicit_scalar_float_image_formats():
    shader = """
    shader ExplicitScalarFloatFormats {
        image2D normalized8 @r8;
        image3D normalized16 @format(r16);
        image2DArray halfLayers @ r16f;
        image2D signedNormalized @r8_snorm;

        float touchR8(image2D image @r8, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        float touchR16(image3D image @r16, ivec3 voxel, float value) {
            float oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        float touchR16f(image2DArray image @r16f, ivec3 pixelLayer, float value) {
            float oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r8, binding = 0) uniform image2D normalized8;" in generated_code
    assert "layout(r16, binding = 1) uniform image3D normalized16;" in generated_code
    assert (
        "layout(r16f, binding = 2) uniform image2DArray halfLayers;" in generated_code
    )
    assert (
        "layout(r8_snorm, binding = 3) uniform image2D signedNormalized;"
        in generated_code
    )
    assert "float touchR8(image2D image, ivec2 pixel, float value)" in generated_code
    assert "float touchR16(image3D image, ivec3 voxel, float value)" in generated_code
    assert (
        "float touchR16f(image2DArray image, ivec3 pixelLayer, float value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "float oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "float oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixel, vec4((oldValue + value)));" in generated_code
    assert "imageStore(image, voxel, vec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixelLayer, vec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code
    assert "imageLoad(image, pixel);" not in generated_code


def test_opengl_explicit_narrow_integer_image_formats():
    shader = """
    shader ExplicitNarrowIntegerFormats {
        image2D signed8 @r8i;
        image3D signed16 @format(r16i);
        image2D unsigned8 @ r8ui;
        image2DArray unsigned16 @format(r16ui);

        int loadSigned8(image2D image @r8i, ivec2 pixel, int value) {
            int oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int loadSigned16(image3D image @r16i, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint loadUnsigned8(image2D image @r8ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint loadUnsigned16(image2DArray image @r16ui, ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r8i, binding = 0) uniform iimage2D signed8;" in generated_code
    assert "layout(r16i, binding = 1) uniform iimage3D signed16;" in generated_code
    assert "layout(r8ui, binding = 2) uniform uimage2D unsigned8;" in generated_code
    assert (
        "layout(r16ui, binding = 3) uniform uimage2DArray unsigned16;" in generated_code
    )
    assert "int loadSigned8(iimage2D image, ivec2 pixel, int value)" in generated_code
    assert "int loadSigned16(iimage3D image, ivec3 voxel, int value)" in generated_code
    assert (
        "uint loadUnsigned8(uimage2D image, ivec2 pixel, uint value)" in generated_code
    )
    assert (
        "uint loadUnsigned16(uimage2DArray image, ivec3 pixelLayer, uint value)"
        in generated_code
    )
    assert "int oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "int oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "uint oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixel, ivec4((oldValue + value)));" in generated_code
    assert "imageStore(image, voxel, ivec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixel, uvec4((oldValue + value)));" in generated_code
    assert "imageStore(image, pixelLayer, uvec4((oldValue + value)));" in generated_code
    assert "layout(r8i, binding = 0) uniform image2D" not in generated_code
    assert "layout(r8ui, binding = 2) uniform image2D" not in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" not in generated_code


def test_opengl_explicit_integer_image_formats_use_integer_image_types():
    shader = """
    shader ExplicitAtomicFormats {
        image2D unsignedCounters @r32ui;
        image2D signedCounters @r32i;
        image3D unsignedVolume @format(r32ui);
        image2DArray signedLayers @format(r32i);

        uint addUnsigned(image2D image @r32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(image, pixel, value);
        }

        int minSigned(image2D image @r32i, ivec2 pixel, int value) {
            return imageAtomicMin(image, pixel, value);
        }

        uint swapVolume(image3D image @r32ui, ivec3 voxel, uint expected, uint value) {
            return imageAtomicCompSwap(image, voxel, expected, value);
        }

        int exchangeLayer(image2DArray image @r32i, ivec3 pixelLayer, int value) {
            return imageAtomicExchange(image, pixelLayer, value);
        }

        compute {
            void main() {
                uint a = addUnsigned(unsignedCounters, ivec2(0, 0), 1);
                int b = minSigned(signedCounters, ivec2(1, 0), -1);
                uint c = swapVolume(unsignedVolume, ivec3(0, 1, 2), 3, 4);
                int d = exchangeLayer(signedLayers, ivec3(2, 3, 4), 5);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(r32ui, binding = 0) uniform uimage2D unsignedCounters;"
        in generated_code
    )
    assert (
        "layout(r32i, binding = 1) uniform iimage2D signedCounters;" in generated_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage3D unsignedVolume;" in generated_code
    )
    assert (
        "layout(r32i, binding = 3) uniform iimage2DArray signedLayers;"
        in generated_code
    )
    assert "uint addUnsigned(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert "int minSigned(iimage2D image, ivec2 pixel, int value)" in generated_code
    assert (
        "uint swapVolume(uimage3D image, ivec3 voxel, uint expected, uint value)"
        in generated_code
    )
    assert (
        "int exchangeLayer(iimage2DArray image, ivec3 pixelLayer, int value)"
        in generated_code
    )
    assert "return imageAtomicAdd(image, pixel, value);" in generated_code
    assert "return imageAtomicMin(image, pixel, value);" in generated_code
    assert (
        "return imageAtomicCompSwap(image, voxel, expected, value);" in generated_code
    )
    assert "return imageAtomicExchange(image, pixelLayer, value);" in generated_code


def test_opengl_explicit_vector_integer_image_formats():
    shader = """
    shader ExplicitVectorIntegerFormats {
        image2D unsignedColor @rgba32ui;
        image3D signedVolume @format(rgba32i);
        image2DArray unsignedLayers @ rgba16ui;

        uvec4 touchUnsigned(image2D image @rgba32ui, ivec2 pixel, uvec4 value) {
            uvec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        ivec4 touchSigned(image3D image @format(rgba32i), ivec3 voxel, ivec4 value) {
            ivec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec4 touchLayers(image2DArray image @rgba16ui, ivec3 pixelLayer, uvec4 value) {
            uvec4 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(rgba32ui, binding = 0) uniform uimage2D unsignedColor;"
        in generated_code
    )
    assert (
        "layout(rgba32i, binding = 1) uniform iimage3D signedVolume;" in generated_code
    )
    assert (
        "layout(rgba16ui, binding = 2) uniform uimage2DArray unsignedLayers;"
        in generated_code
    )
    assert (
        "uvec4 touchUnsigned(uimage2D image, ivec2 pixel, uvec4 value)"
        in generated_code
    )
    assert (
        "ivec4 touchSigned(iimage3D image, ivec3 voxel, ivec4 value)" in generated_code
    )
    assert (
        "uvec4 touchLayers(uimage2DArray image, ivec3 pixelLayer, uvec4 value)"
        in generated_code
    )
    assert "uvec4 oldValue = imageLoad(image, pixel);" in generated_code
    assert "ivec4 oldValue = imageLoad(image, voxel);" in generated_code
    assert "uvec4 oldValue = imageLoad(image, pixelLayer);" in generated_code
    assert "imageStore(image, pixel, (oldValue + value));" in generated_code
    assert "imageStore(image, voxel, (oldValue + value));" in generated_code
    assert "imageStore(image, pixelLayer, (oldValue + value));" in generated_code
    assert "layout(rgba32ui, binding = 0) uniform image2D" not in generated_code
    assert "layout(rgba32i, binding = 1) uniform image3D" not in generated_code
    assert "imageLoad(image, pixel).x" not in generated_code
    assert "uvec4((oldValue + value))" not in generated_code
    assert "ivec4((oldValue + value))" not in generated_code


def test_opengl_integer_image_atomic_add():
    shader = """
    shader AtomicImages {
        uimage2D counters;
        iimage2D signedCounters;

        uint addCounter(uimage2D image, ivec2 pixel, uint value) {
            uint previous = imageAtomicAdd(image, pixel, value);
            return previous;
        }

        int addSignedCounter(iimage2D image, ivec2 pixel, int value) {
            int previous = imageAtomicAdd(image, pixel, value);
            return previous;
        }

        compute {
            void main() {
                uint oldValue = addCounter(counters, ivec2(0, 1), 2);
                int oldSigned = addSignedCounter(signedCounters, ivec2(2, 3), -1);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters;" in generated_code
    assert (
        "layout(r32i, binding = 1) uniform iimage2D signedCounters;" in generated_code
    )
    assert "uint addCounter(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert (
        "int addSignedCounter(iimage2D image, ivec2 pixel, int value)" in generated_code
    )
    assert "uint previous = imageAtomicAdd(image, pixel, value);" in generated_code
    assert "int previous = imageAtomicAdd(image, pixel, value);" in generated_code


def test_opengl_integer_image_atomic_operations():
    shader = """
    shader AtomicOps {
        uimage2D counters;
        iimage2D signedCounters;

        uint unsignedOps(uimage2D image, ivec2 pixel, uint value) {
            uint minValue = imageAtomicMin(image, pixel, value);
            uint maxValue = imageAtomicMax(image, pixel, value);
            uint andValue = imageAtomicAnd(image, pixel, value);
            uint orValue = imageAtomicOr(image, pixel, value);
            uint xorValue = imageAtomicXor(image, pixel, value);
            uint exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + maxValue + andValue + orValue + xorValue + exchanged;
        }

        int signedOps(iimage2D image, ivec2 pixel, int value) {
            int minValue = imageAtomicMin(image, pixel, value);
            int maxValue = imageAtomicMax(image, pixel, value);
            int exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + maxValue + exchanged;
        }

        compute {
            void main() {
                uint unsignedResult = unsignedOps(counters, ivec2(0, 1), 3);
                int signedResult = signedOps(signedCounters, ivec2(2, 3), -4);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    for operation in [
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
    ]:
        assert f"{operation}(image, pixel, value)" in generated_code

    assert "uint unsignedOps(uimage2D image, ivec2 pixel, uint value)" in generated_code
    assert "int signedOps(iimage2D image, ivec2 pixel, int value)" in generated_code


def test_opengl_integer_image_atomic_compare_swap():
    shader = """
    shader AtomicCompareSwap {
        uimage2D counters;
        iimage2D signedCounters;

        uint compareUnsigned(uimage2D image, ivec2 pixel, uint expected, uint replacement) {
            uint previous = imageAtomicCompSwap(image, pixel, expected, replacement);
            return previous;
        }

        int compareSigned(iimage2D image, ivec2 pixel, int expected, int replacement) {
            int previous = imageAtomicCompSwap(image, pixel, expected, replacement);
            return previous;
        }

        compute {
            void main() {
                uint oldValue = compareUnsigned(counters, ivec2(0, 1), 2, 3);
                int oldSigned = compareSigned(signedCounters, ivec2(2, 3), -1, 4);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "uint compareUnsigned(uimage2D image, ivec2 pixel, uint expected, uint replacement)"
        in generated_code
    )
    assert (
        "int compareSigned(iimage2D image, ivec2 pixel, int expected, int replacement)"
        in generated_code
    )
    assert (
        "uint previous = imageAtomicCompSwap(image, pixel, expected, replacement);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicCompSwap(image, pixel, expected, replacement);"
        in generated_code
    )


def test_opengl_integer_image_dimension_atomics():
    shader = """
    shader TypedImageDimensions {
        uimage3D volumeCounters;
        iimage3D signedVolumeCounters;
        iimage2DArray layerCounters;
        uimage2DArray unsignedLayerCounters;

        uint touchVolume(uimage3D image, ivec3 voxel, uint value) {
            uint oldValue = imageAtomicAdd(image, voxel, value);
            uint swapped = imageAtomicCompSwap(image, voxel, oldValue, value);
            return oldValue + swapped;
        }

        int touchLayers(iimage2DArray image, ivec3 pixelLayer, int value) {
            int oldValue = imageAtomicMin(image, pixelLayer, value);
            int swapped = imageAtomicCompSwap(image, pixelLayer, oldValue, value);
            return oldValue + swapped;
        }

        compute {
            void main() {
                uint volumeResult = touchVolume(volumeCounters, ivec3(0, 1, 2), 3);
                int layerResult = touchLayers(layerCounters, ivec3(4, 5, 6), -7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(r32ui, binding = 0) uniform uimage3D volumeCounters;" in generated_code
    )
    assert (
        "layout(r32i, binding = 1) uniform iimage3D signedVolumeCounters;"
        in generated_code
    )
    assert (
        "layout(r32i, binding = 2) uniform iimage2DArray layerCounters;"
        in generated_code
    )
    assert (
        "layout(r32ui, binding = 3) uniform uimage2DArray unsignedLayerCounters;"
        in generated_code
    )
    assert "uint touchVolume(uimage3D image, ivec3 voxel, uint value)" in generated_code
    assert (
        "int touchLayers(iimage2DArray image, ivec3 pixelLayer, int value)"
        in generated_code
    )
    assert "uint oldValue = imageAtomicAdd(image, voxel, value);" in generated_code
    assert (
        "uint swapped = imageAtomicCompSwap(image, voxel, oldValue, value);"
        in generated_code
    )
    assert "int oldValue = imageAtomicMin(image, pixelLayer, value);" in generated_code
    assert (
        "int swapped = imageAtomicCompSwap(image, pixelLayer, oldValue, value);"
        in generated_code
    )


def test_opengl_integer_image_scalar_load_store():
    shader = """
    shader IntegerImageLoadStore {
        uimage2D counters;
        iimage3D signedVolume;
        uimage2DArray layerCounters;

        uint touch2D(uimage2D image, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int touch3D(iimage3D image, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint touchLayer(uimage2DArray image, ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touch2D(counters, ivec2(0, 1), 2);
                int b = touch3D(signedVolume, ivec3(1, 2, 3), -4);
                uint c = touchLayer(layerCounters, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32ui, binding = 0) uniform uimage2D counters;" in generated_code
    assert "layout(r32i, binding = 1) uniform iimage3D signedVolume;" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2DArray layerCounters;"
        in generated_code
    )
    assert "uint oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "imageStore(image, pixel, uvec4((oldValue + value)));" in generated_code
    assert "int oldValue = imageLoad(image, voxel).x;" in generated_code
    assert "imageStore(image, voxel, ivec4((oldValue + value)));" in generated_code
    assert "uint oldValue = imageLoad(image, pixelLayer).x;" in generated_code
    assert "imageStore(image, pixelLayer, uvec4((oldValue + value)));" in generated_code


def test_opengl_texture_query_functions():
    shader = """
    shader TextureQueries {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler2DMS msMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        ivec2 query2D(sampler2D tex, sampler s, vec2 uv) {
            ivec2 size = textureSize(tex, 1);
            int levels = textureQueryLevels(tex);
            vec2 lod = textureQueryLod(tex, s, uv);
            return size + ivec2(levels) + ivec2(lod);
        }

        ivec3 queryArray(sampler2DArray tex) {
            return textureSize(tex, 0);
        }

        ivec2 queryMs(sampler2DMS tex) {
            return textureSize(tex);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                ivec2 q = query2D(colorMap, linearSampler, input.uv);
                ivec3 qa = queryArray(layerMap);
                ivec2 qm = queryMs(msMap);
                return vec4(q.x + qa.z + qm.x);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msMap;" in generated_code
    assert "ivec2 query2D(sampler2D tex, vec2 uv)" in generated_code
    assert "ivec2 size = textureSize(tex, 1);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert "vec2 lod = textureQueryLod(tex, uv);" in generated_code
    assert "ivec3 queryArray(sampler2DArray tex)" in generated_code
    assert "return textureSize(tex, 0);" in generated_code
    assert "ivec2 queryMs(sampler2DMS tex)" in generated_code
    assert "return textureSize(tex);" in generated_code
    assert "linearSampler" not in generated_code
    assert "textureQueryLod(tex, s, uv)" not in generated_code


def test_opengl_texture_operation_variants_passthrough():
    shader = """
    shader TextureOps {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 lodColor = textureLod(colorMap, input.uv, 1.0);
                vec4 gradColor = textureGrad(colorMap, input.uv, vec2(0.1), vec2(0.2));
                vec4 fetched = texelFetch(colorMap, input.pixel, 0);
                vec4 gathered = textureGather(colorMap, input.uv);
                return lodColor + gradColor + fetched + gathered;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "textureLod(colorMap, input.uv, 1.0)" in generated_code
    assert "textureGrad(colorMap, input.uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "texelFetch(colorMap, input.pixel, 0)" in generated_code
    assert "textureGather(colorMap, input.uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code


def test_opengl_hlsl_texture_aliases_map_to_glsl_names():
    shader = """
    shader TextureAliases {
        sampler2D colorMap;

        void main() {
            vec2 uv;
            vec4 color = tex2Dlod(colorMap, uv, 1.0);
            vec4 grad = tex2Dgrad(colorMap, uv, vec2(0.1), vec2(0.2));
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "textureLod(colorMap, uv, 1.0)" in generated_code
    assert "textureGrad(colorMap, uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "tex2Dlod(" not in generated_code
    assert "tex2Dgrad(" not in generated_code


def test_opengl_explicit_sampler_argument_is_dropped():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler linearSampler;
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(colorMap, linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCube envMap;" in generated_code
    assert "linearSampler" not in generated_code
    assert "texture(colorMap, input.uv)" in generated_code


def test_opengl_sampler_parameter_is_dropped():
    shader = """
    shader SamplerParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler sampleState, vec2 uv) {
            return texture(colorMap, sampleState, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "vec4 sampleColor(vec2 uv)" in generated_code
    assert "texture(colorMap, uv)" in generated_code
    assert "sampleColor(input.uv)" in generated_code
    assert "linearSampler" not in generated_code
    assert "sampleState" not in generated_code


def test_opengl_texture_parameter_keeps_combined_sampler():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, sampler sampleState, vec2 uv) {
            return texture(tex, sampleState, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(colorMap, linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "vec4 sampleColor(sampler2D tex, vec2 uv)" in generated_code
    assert "texture(tex, uv)" in generated_code
    assert "sampleColor(colorMap, input.uv)" in generated_code
    assert "linearSampler" not in generated_code
    assert "sampleState" not in generated_code


def test_opengl_implicit_sampler_for_texture_parameter():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, vec2 uv) {
            return texture(tex, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(colorMap, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "vec4 sampleColor(sampler2D tex, vec2 uv)" in generated_code
    assert "texture(tex, uv)" in generated_code
    assert "sampleColor(colorMap, input.uv)" in generated_code


def test_opengl_shadow_texture_compare():
    shader = """
    shader ShadowTexture {
        sampler2DShadow shadowMap;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert "texture(shadowMap, vec3(input.uv, input.depth))" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_texture_array_compare():
    shader = """
    shader ShadowTextureArray {
        sampler2DShadow shadowMaps[4];

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMaps[input.layer], input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "texture(shadowMaps[input.layer], vec3(input.uv, input.depth))"
        in generated_code
    )
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_texture_array_compare_filters_indexed_sampler_array():
    shader = """
    shader ShadowSamplerArrayHelper {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[4], sampler shadowSamplers[4], int layer, vec2 uv, float depth) {
            return textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], int layer, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in generated_code
    assert (
        "shadowLayer(shadowMaps, input.layer, input.uv, input.depth)" in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[layer]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_keeps_declared_size_with_constant_indices():
    shader = """
    shader FixedShadowArrayConstantIndex {
        const int LAYER = 2;
        sampler2DShadow shadowMaps[6];
        sampler shadowSamplers[6];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[6], sampler shadowSamplers[6], vec2 uv, float depth) {
            return textureCompare(shadowMaps[LAYER], shadowSamplers[LAYER], uv, depth) + textureCompare(shadowMaps[1 + 2], shadowSamplers[1 + 2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 2;" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[6];" in generated_code
    )
    assert "layout(binding = 6) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[6], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[LAYER], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[(1 + 2)], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, input.uv, input.depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];"
        not in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "shadowSamplers[LAYER]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_resolves_constant_declared_size_for_bindings():
    shader = """
    shader ConstSizedShadowResourceArrays {
        const int BASE_COUNT = 2;
        const int SHADOW_COUNT = BASE_COUNT * 3;
        sampler2DShadow shadowMaps[SHADOW_COUNT];
        sampler shadowSamplers[SHADOW_COUNT];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[SHADOW_COUNT], sampler shadowSamplers[SHADOW_COUNT], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE_COUNT = 2;" in generated_code
    assert "const int SHADOW_COUNT = (BASE_COUNT * 3);" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[SHADOW_COUNT];"
        in generated_code
    )
    assert "layout(binding = 6) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[SHADOW_COUNT], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, input.uv, input.depth)" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_resolves_inline_declared_size_expression_for_bindings():
    shader = """
    shader ExprSizedShadowResourceArrays {
        sampler2DShadow shadowMaps[2 * 3];
        sampler shadowSamplers[2 * 3];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[2 * 3], sampler shadowSamplers[2 * 3], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[(2 * 3)];"
        in generated_code
    )
    assert "layout(binding = 6) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[(2 * 3)], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, input.uv, input.depth)" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
    shader = """
    shader ParenthesizedSizedShadowResourceArrays {
        sampler2DShadow shadowMaps[(2 + 1) * 2];
        sampler shadowSamplers[(2 + 1) * 2];
        sampler2DShadow unaryShadowMaps[+6];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[(2 + 1) * 2], sampler shadowSamplers[(2 + 1) * 2], sampler2DShadow unaryShadowMaps[+6], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth) + textureCompare(unaryShadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[((2 + 1) * 2)];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform sampler2DShadow unaryShadowMaps[(+6)];"
        in generated_code
    )
    assert "layout(binding = 12) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[((2 + 1) * 2)], sampler2DShadow unaryShadowMaps[(+6)], vec2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, unaryShadowMaps, input.uv, input.depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "texture(unaryShadowMaps[2], vec3(uv, depth))" in generated_code
    assert (
        "layout(binding = 6) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_infers_helper_size_and_filters_sampler_array():
    shader = """
    shader UnsizedShadowSamplerArrayHelper {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            float high = textureCompare(shadowMaps[3], shadowSamplers[3], uv, depth);
            float low = textureCompare(shadowMaps[1], shadowSamplers[1], uv, depth);
            return high + low;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert "layout(binding = 4) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[3], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[1], vec3(uv, depth))" in generated_code
    assert "texture(afterShadow, vec3(input.uv, input.depth))" in generated_code
    assert "shadowLayer(shadowMaps, input.uv, input.depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[3]" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_infers_transitive_helper_size():
    shader = """
    shader MultiHopUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowDeep(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            float high = textureCompare(shadowMaps[4], shadowSamplers[4], uv, depth);
            float low = textureCompare(shadowMaps[1], shadowSamplers[1], uv, depth);
            return high + low;
        }

        float shadowMid(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return shadowDeep(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert "layout(binding = 5) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowDeep(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert (
        "float shadowMid(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[4], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[1], vec3(uv, depth))" in generated_code
    assert "shadowDeep(shadowMaps, uv, depth)" in generated_code
    assert "shadowMid(shadowMaps, input.uv, input.depth)" in generated_code
    assert "texture(afterShadow, vec3(input.uv, input.depth))" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[4]" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_preserves_dynamic_indexing():
    shader = """
    shader MixedIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], int layer, vec2 uv, float depth) {
            float dynamicShadow = textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
            float fixedShadow = textureCompare(shadowMaps[3], shadowSamplers[3], uv, depth);
            return dynamicShadow + fixedShadow;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert "layout(binding = 4) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], int layer, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in generated_code
    assert "texture(shadowMaps[3], vec3(uv, depth))" in generated_code
    assert (
        "shadowLayer(shadowMaps, input.layer, input.uv, input.depth)" in generated_code
    )
    assert "texture(afterShadow, vec3(input.uv, input.depth))" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert (
        "layout(binding = 1) uniform sampler2DShadow afterShadow;" not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[layer]" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_ignores_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], int layer, vec2 uv, float depth) {
            return textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[-1], shadowSamplers[-1], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    dynamic_code = GLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = GLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "layout(binding = 0) uniform sampler2DShadow shadowMaps[];" in dynamic_code
    assert "layout(binding = 1) uniform sampler2DShadow afterShadow;" in dynamic_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[], int layer, vec2 uv, float depth)"
        in dynamic_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in dynamic_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[1];" not in dynamic_code
    )
    assert (
        "layout(binding = 2) uniform sampler2DShadow afterShadow;" not in dynamic_code
    )
    assert "sampler shadowSamplers" not in dynamic_code
    assert "shadowSamplers[layer]" not in dynamic_code
    assert "textureCompare(" not in dynamic_code

    assert "layout(binding = 0) uniform sampler2DShadow shadowMaps[];" in negative_code
    assert "layout(binding = 1) uniform sampler2DShadow afterShadow;" in negative_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[], vec2 uv, float depth)"
        in negative_code
    )
    assert "texture(shadowMaps[(-1)], vec3(uv, depth))" in negative_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[0];"
        not in negative_code
    )
    assert (
        "layout(binding = 0) uniform sampler2DShadow afterShadow;" not in negative_code
    )
    assert "sampler shadowSamplers" not in negative_code
    assert "textureCompare(" not in negative_code


def test_opengl_unsized_shadow_texture_array_infers_constant_expression_size():
    shader = """
    shader ExprIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2 * 2], shadowSamplers[2 * 2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert "layout(binding = 5) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[(2 * 2)], vec3(uv, depth))" in generated_code
    assert "texture(afterShadow, vec3(input.uv, input.depth))" in generated_code
    assert "shadowLayer(shadowMaps, input.uv, input.depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_unsized_shadow_texture_array_infers_named_constant_size():
    shader = """
    shader ConstIndexedUnsizedShadowResources {
        const int BASE = 2;
        const int LAYER = BASE * 2;
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[LAYER], shadowSamplers[LAYER], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int BASE = 2;" in generated_code
    assert "const int LAYER = (BASE * 2);" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert "layout(binding = 5) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[5], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[LAYER], vec3(uv, depth))" in generated_code
    assert "texture(afterShadow, vec3(input.uv, input.depth))" in generated_code
    assert "shadowLayer(shadowMaps, input.uv, input.depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_implicit_sampler_for_shadow_texture_parameter():
    shader = """
    shader ShadowParameter {
        sampler2DShadow shadowMap;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler2DShadow tex, vec2 uv, float depth) {
            return textureCompare(tex, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "float sampleShadow(sampler2DShadow tex, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(tex, vec3(uv, depth))" in generated_code
    assert "sampleShadow(shadowMap, input.uv, input.depth)" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_texture_parameter_keeps_combined_sampler():
    shader = """
    shader ShadowParameter {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler2DShadow tex, sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(tex, compareSampler, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowMap, shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "float sampleShadow(sampler2DShadow tex, vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(tex, vec3(uv, depth))" in generated_code
    assert "sampleShadow(shadowMap, input.uv, input.depth)" in generated_code
    assert "shadowSampler" not in generated_code
    assert "compareSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_shadow_compare_sampler_parameter_is_dropped():
    shader = """
    shader ShadowHelper {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(shadowMap, compareSampler, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert "float sampleShadow(vec2 uv, float depth)" in generated_code
    assert "texture(shadowMap, vec3(uv, depth))" in generated_code
    assert "sampleShadow(input.uv, input.depth)" in generated_code
    assert "shadowSampler" not in generated_code
    assert "compareSampler" not in generated_code
    assert "textureCompare(" not in generated_code


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
    code_gen = GLSLCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


if __name__ == "__main__":
    pytest.main()
