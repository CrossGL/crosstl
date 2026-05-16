import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
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


def test_ray_payload_semantics():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        ray_generation {
            void main(Payload payload @ payload) {
                payload.color = vec3(1.0, 0.0, 0.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "[[payload]]" in generated


def test_mesh_object_stage_codegen():
    code = """
    shader meshpipe {
        object {
            void main() { }
        }
        mesh {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "object void object_main" in generated
    assert "mesh void mesh_main" in generated


def test_anyhit_stage_codegen():
    code = """
    shader rt {
        anyhit {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "anyhit void anyhit_main" in generated


def test_metal_atomic_fetch_codegen():
    code = """
    shader main {
        compute {
            void main() {
                atomic_int counter;
                int old = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "atomic_fetch_add_explicit" in generated


def test_compute_builtin_semantics_roundtrip():
    code = """
    shader cs {
        compute {
            void main(uvec3 gid @ gl_GlobalInvocationID,
                      uvec3 lid @ gl_LocalInvocationID,
                      uvec3 group @ gl_WorkGroupID,
                      uint idx @ gl_LocalInvocationIndex) { }
        }
    }
    """
    ast = crosstl.translator.parse(code)
    generated = MetalCodeGen().generate(ast)
    for expected in [
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "thread_index_in_threadgroup",
    ]:
        assert expected in generated


def test_metal_raytrace_and_mesh_intrinsics():
    code = """
    shader main {
        ray_generation {
            void main() {
                TraceRay();
            }
        }
        mesh {
            void main() {
                SetMeshOutputCounts(64, 32);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "TraceRay" in generated
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
    code_gen = MetalCodeGen()
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
    code_gen = MetalCodeGen()
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
    code_gen = MetalCodeGen()
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
    code_gen = MetalCodeGen()
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

        # Verify sampling operations
        # Just check for texture operations, not specific syntax
        assert "albedoMap" in generated_code
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

        # Check that some form of texture access is happening
        # Metal uses either .sample() or other methods
        assert "colorMap" in generated_code and "texCoord" in generated_code
        assert "normalMap" in generated_code
        assert "envMap" in generated_code
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


def test_metal_array_handling(array_test_data):
    """Test the Metal code generator's handling of array types and array access."""
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
        for expected in array_test_data["metal"]["array_type_declarations"]:
            assert expected in generated_code

        for expected in array_test_data["metal"]["array_access"]:
            assert expected in generated_code

    except SyntaxError as e:
        pytest.fail(f"Metal array codegen failed: {e}")


def test_metal_array_member_semantics():
    code = """
    struct VertexData {
        float weights[4] @ TEXCOORD0;
        vec3 colors[] @ COLOR;
    };
    """

    ast = crosstl.translator.parse(code)
    generated_code = MetalCodeGen().generate(ast)

    assert "float weights[4] [[attribute(5)]];" in generated_code
    assert "array<float3> colors [[COLOR]];" in generated_code


def test_metal_local_array_declarations_use_c_style_order():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "float3 localColors[4];" in generated_code
    assert "float weights[8];" in generated_code
    assert "float3[4] localColors" not in generated_code
    assert "float[8] weights" not in generated_code


def test_metal_array_parameters_use_c_style_order():
    shader = """
    shader TestShader {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "float weights[4]" in generated_code
    assert "float3 colors[2]" in generated_code
    assert "float weights[4] [[stage_in]]" not in generated_code
    assert "float3 colors[2] [[stage_in]]" not in generated_code
    assert "float[4] weights" not in generated_code
    assert "float3[2] colors" not in generated_code


def test_metal_non_resource_arrays_preserve_expression_sizes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "float3 colors[(2 + 1) * 2];" in generated_code
    assert "float weights[+6];" in generated_code
    assert (
        "float accumulate(float values[(2 + 1) * 2], float3 normals[+6])"
        in generated_code
    )
    assert "float localWeights[(2 + 1) * 2];" in generated_code
    assert "float3 localNormals[+6];" in generated_code
    assert "float values[]" not in generated_code
    assert "float localWeights[]" not in generated_code
    assert "2 + 1 * 2" not in generated_code


def test_metal_sampler_cube_uses_cube_texture_type():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texturecube<float> envMap [[texture(1)]]" in generated_code
    assert "texture2d<float> envMap" not in generated_code
    assert "colorMap.sample" in generated_code
    assert "envMap.sample" in generated_code


def test_metal_texture_array_resources_and_indexed_sampling():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texturecube<float> envMap [[texture(4)]]" in generated_code
    assert (
        "textures[input.layer].sample(sampler(mag_filter::linear, min_filter::linear), input.uv)"
        in generated_code
    )
    assert "envMap.sample" in generated_code


def test_metal_fixed_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 2;" in generated_code
    assert "array<texture2d<float>, 6> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(6)]]" in generated_code
    assert "array<sampler, 6> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 6> textures, array<sampler, 6> samplers"
        in generated_code
    )
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in generated_code
    assert "textures[1 + 2].sample(samplers[1 + 2], uv)" in generated_code
    assert "array<texture2d<float>, 4> textures" not in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" not in generated_code


def test_metal_fixed_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE_COUNT = 2;" in generated_code
    assert "constant int TEXTURE_COUNT = BASE_COUNT * 3;" in generated_code
    assert (
        "array<texture2d<float>, TEXTURE_COUNT> textures [[texture(0)]]"
        in generated_code
    )
    assert "texture2d<float> afterTexture [[texture(6)]]" in generated_code
    assert "array<sampler, TEXTURE_COUNT> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, TEXTURE_COUNT> textures, array<sampler, TEXTURE_COUNT> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "texture2d<float> afterTexture [[texture(1)]]" not in generated_code


def test_metal_fixed_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<texture2d<float>, 2 * 3> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(6)]]" in generated_code
    assert "array<sampler, 2 * 3> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 2 * 3> textures, array<sampler, 2 * 3> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "texture2d<float> afterTexture [[texture(1)]]" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code


def test_metal_fixed_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<float>, (2 + 1) * 2> textures [[texture(0)]]" in generated_code
    )
    assert "array<texture2d<float>, +6> unaryTextures [[texture(6)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(12)]]" in generated_code
    assert "array<sampler, (2 + 1) * 2> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, (2 + 1) * 2> textures, array<sampler, (2 + 1) * 2> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "unaryTextures[2].sample" in generated_code
    assert "texture2d<float> afterTexture [[texture(6)]]" not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code


def test_metal_storage_image_load_store():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> outputImage [[texture(0)]], texture3d<float, access::read_write> volumeImage [[texture(1)]], texture2d_array<float, access::read_write> layerImage [[texture(2)]])"
        in generated_code
    )
    assert (
        "float4 touchImages(texture2d<float, access::read_write> outImg, texture3d<float, access::read_write> volume, texture2d_array<float, access::read_write> layers, int2 pixel, int3 voxel, int3 pixelLayer)"
        in generated_code
    )
    assert "float4 color = outImg.read(uint2(pixel));" in generated_code
    assert "float4 volumeColor = volume.read(uint3(voxel));" in generated_code
    assert (
        "float4 layerColor = layers.read(uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "outImg.write(color + layerColor, uint2(pixel));" in generated_code
    assert "volume.write(volumeColor, uint3(voxel));" in generated_code
    assert (
        "layers.write(color, uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "kernel void kernel_main(," not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_integer_image_atomic_add():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> counters [[texture(0)]], texture2d<int, access::read_write> signedCounters [[texture(1)]])"
        in generated_code
    )
    assert (
        "uint addCounter(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int addSignedCounter(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "uint previous = image.atomic_fetch_add(uint2(pixel), value).x;"
        in generated_code
    )
    assert (
        "int previous = image.atomic_fetch_add(uint2(pixel), value).x;"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code


def test_metal_integer_image_atomic_operations():
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
    generated_code = MetalCodeGen().generate(ast)

    for method in [
        "atomic_fetch_min",
        "atomic_fetch_max",
        "atomic_fetch_and",
        "atomic_fetch_or",
        "atomic_fetch_xor",
        "atomic_exchange",
    ]:
        assert f"image.{method}(uint2(pixel), value).x" in generated_code

    assert (
        "uint unsignedOps(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int signedOps(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert "imageAtomicMin(" not in generated_code
    assert "imageAtomicMax(" not in generated_code
    assert "imageAtomicExchange(" not in generated_code


def test_metal_integer_image_atomic_compare_swap():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "uint imageAtomicCompSwap_uimage2D(texture2d<uint, access::read_write> image, int2 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2D(texture2d<int, access::read_write> image, int2 coord, int compareValue, int value)"
        in generated_code
    )
    assert "uint4 original;" in generated_code
    assert "int4 original;" in generated_code
    assert "original.x = compareValue;" in generated_code
    assert (
        "image.atomic_compare_exchange_weak(uint2(coord), &original, value)"
        in generated_code
    )
    assert "return original.x;" in generated_code
    assert (
        "uint previous = imageAtomicCompSwap_uimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicCompSwap_iimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(image" not in generated_code


def test_metal_integer_image_dimension_atomics():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture3d<uint, access::read_write> volumeCounters [[texture(0)]], texture3d<int, access::read_write> signedVolumeCounters [[texture(1)]], texture2d_array<int, access::read_write> layerCounters [[texture(2)]], texture2d_array<uint, access::read_write> unsignedLayerCounters [[texture(3)]])"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(texture3d<uint, access::read_write> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2DArray(texture2d_array<int, access::read_write> image, int3 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint3(coord), &original, value)"
        in generated_code
    )
    assert (
        "image.atomic_compare_exchange_weak(uint2(coord.xy), uint(coord.z), &original, value)"
        in generated_code
    )
    assert (
        "uint oldValue = image.atomic_fetch_add(uint3(voxel), value).x;"
        in generated_code
    )
    assert (
        "int oldValue = image.atomic_fetch_min(uint2(pixelLayer.xy), uint(pixelLayer.z), value).x;"
        in generated_code
    )
    assert (
        "uint swapped = imageAtomicCompSwap_uimage3D(image, voxel, oldValue, value);"
        in generated_code
    )
    assert (
        "int swapped = imageAtomicCompSwap_iimage2DArray(image, pixelLayer, oldValue, value);"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicMin(" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code


def test_metal_integer_image_scalar_load_store():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> counters [[texture(0)]], texture3d<int, access::read_write> signedVolume [[texture(1)]], texture2d_array<uint, access::read_write> layerCounters [[texture(2)]])"
        in generated_code
    )
    assert "uint oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "image.write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "int oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert "image.write(int4(oldValue + value), uint3(voxel));" in generated_code
    assert (
        "uint oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_scalar_image_formats():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> scalarFloat [[texture(0)]], texture3d<int, access::read_write> signedVolume [[texture(1)]], texture2d_array<uint, access::read_write> unsignedLayers [[texture(2)]])"
        in generated_code
    )
    assert (
        "float touchFloat(texture2d<float, access::read_write> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "int touchSigned(texture3d<int, access::read_write> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint touchUnsigned(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "float oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "image.write(float4(oldValue + value), uint2(pixel));" in generated_code
    assert "int oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert "image.write(int4(oldValue + value), uint3(voxel));" in generated_code
    assert (
        "uint oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "[[r32" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_rg_image_formats():
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

        compute {
            void main() {
                vec2 a = touchFloat(rgFloat, ivec2(0, 1), vec2(0.5, 1.5));
                ivec2 b = touchSigned(rgSigned, ivec3(1, 2, 3), ivec2(-4, 5));
                uvec2 c = touchUnsigned(rgUnsigned, ivec3(4, 5, 6), uvec2(7, 8));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> rgFloat [[texture(0)]], texture3d<int, access::read_write> rgSigned [[texture(1)]], texture2d_array<uint, access::read_write> rgUnsigned [[texture(2)]])"
        in generated_code
    )
    assert (
        "float2 touchFloat(texture2d<float, access::read_write> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(texture3d<int, access::read_write> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert "int2 oldValue = image.read(uint3(voxel)).xy;" in generated_code
    assert (
        "uint2 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).xy;"
        in generated_code
    )
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "image.write(int4(oldValue + value, 0, 0), uint3(voxel));" in generated_code
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "texture2d<float4, access::read_write> rgFloat" not in generated_code
    assert ".read(uint2(pixel)).x;" not in generated_code
    assert "image.write(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_narrow_rg_image_formats():
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

        compute {
            void main() {
                vec2 a = touchFloat(rg8Float, ivec2(0, 1), vec2(0.5, 1.5));
                vec2 b = touchHalf(rg16Half, ivec3(1, 2, 3), vec2(2.5, 3.5));
                ivec2 c = touchSigned(rg16Signed, ivec3(2, 3, 4), ivec2(-4, 5));
                uvec2 d = touchUnsigned(rg8Unsigned, ivec2(4, 5), uvec2(7, 8));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d<float, access::read_write> rg8Float [[texture(0)]]" in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rg8Snorm [[texture(1)]]" in generated_code
    )
    assert (
        "texture3d<float, access::read_write> rg16Float [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rg16Snorm [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> rg16Half [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<int, access::read_write> rg8Signed [[texture(5)]]" in generated_code
    )
    assert (
        "texture3d<int, access::read_write> rg16Signed [[texture(6)]]" in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> rg8Unsigned [[texture(7)]]"
        in generated_code
    )
    assert (
        "texture2d_array<uint, access::read_write> rg16Unsigned [[texture(8)]]"
        in generated_code
    )
    assert (
        "float2 touchFloat(texture2d<float, access::read_write> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchHalf(texture2d_array<float, access::read_write> image, int3 pixelLayer, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(texture3d<int, access::read_write> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(texture2d<uint, access::read_write> image, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert (
        "float2 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).xy;"
        in generated_code
    )
    assert "int2 oldValue = image.read(uint3(voxel)).xy;" in generated_code
    assert "uint2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "image.write(int4(oldValue + value, 0, 0), uint3(voxel));" in generated_code
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u), uint2(pixel));" in generated_code
    )
    assert "texture2d<float4, access::read_write> rg8Float" not in generated_code
    assert ".read(uint2(pixel)).x;" not in generated_code
    assert "image.write(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_rgba_float_image_formats():
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

        compute {
            void main() {
                vec4 a = touchColor(rgba8Color, ivec2(0, 1), vec4(0.5));
                vec4 b = touchHalf(rgba16Half, ivec3(1, 2, 3), vec4(0.25));
                vec4 c = touchFloat(rgba32Float, ivec3(2, 3, 4), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "texture2d<float, access::read_write> rgba8Color [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rgba8Snorm [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> rgba16Color [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> rgba16Snorm [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> rgba16Half [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> rgba32Float [[texture(5)]]"
        in generated_code
    )
    assert (
        "float4 touchColor(texture2d<float, access::read_write> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchHalf(texture2d_array<float, access::read_write> image, int3 pixelLayer, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchFloat(texture3d<float, access::read_write> image, int3 voxel, float4 value)"
        in generated_code
    )
    assert (
        "float4 typedOverride(texture2d<float, access::read_write> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert "float4 oldValue = image.read(uint2(pixel));" in generated_code
    assert (
        "float4 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "float4 oldValue = image.read(uint3(voxel));" in generated_code
    assert "image.write(oldValue + value, uint2(pixel));" in generated_code
    assert (
        "image.write(oldValue + value, uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "image.write(oldValue + value, uint3(voxel));" in generated_code
    assert "texture2d<int, access::read_write> image" not in generated_code
    assert "image.read(uint2(pixel)).x" not in generated_code
    assert "image.read(uint2(pixel)).xy" not in generated_code
    assert "image.write(float4(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_preserve_format_metadata():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "array<texture2d<uint, access::read_write>, 2> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(2)]]"
        in generated_code
    )
    assert (
        "array<texture3d<float, access::read_write>, 2> rgbaVolumes [[texture(5)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert "texture2d<float> sampled [[texture(8)]]" in generated_code
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float4 touchVolumes(array<texture3d<float, access::read_write>, 2> images, int3 voxel, float4 value)"
        in generated_code
    )
    assert "uint oldValue = images[1].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float4 oldValue = images[1].read(uint3(voxel));" in generated_code
    assert "images[0].write(oldValue + value, uint3(voxel));" in generated_code
    assert (
        "texture2d<float, access::read_write> counters [[texture(0)]]"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 2> counters" not in generated_code
    )
    assert "images[1].read(uint2(pixel));" not in generated_code
    assert "images[2].read(uint2(pixel)).x;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_preserve_expression_sizes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<uint, access::read_write>, (1 + 1) * 2> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, +3> rgPairs [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert "texture2d<float> sampled [[texture(8)]]" in generated_code
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, (1 + 1) * 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, +3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[2].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(4)]]"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, (1 + 1) * 2> counters"
        not in generated_code
    )
    assert "2 + 1 * 2" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_unsized_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader UnsizedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
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

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> counters [[texture(0)]]"
        not in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters" not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgPairs" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> images, int2 pixel" not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> images, int2 pixel" not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_infer_named_constant_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int COUNT = 3;" in generated_code
    assert "constant int LAYER = COUNT - 1;" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, COUNT> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(6)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[LAYER].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 1> rgPairs" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> counters [[texture(0)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_ignore_shadowed_local_constant():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 3;" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 4> counters" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(4)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_infer_transitive_helper_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 3> rgPairs [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(7)]]"
        in generated_code
    )
    assert (
        "uint touchCountersDeep(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint touchCountersMid(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairsDeep(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchPairsMid(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "images[1].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "return touchCountersDeep(images, pixel, value);" in generated_code
    assert "return touchPairsDeep(images, pixel, value);" in generated_code
    assert "uint a = touchCountersMid(counters, int2(1, 2), 3);" in generated_code
    assert (
        "float2 b = touchPairsMid(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters" not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgPairs" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_scalar_float_image_formats():
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

        compute {
            void main() {
                float a = touchR8(normalized8, ivec2(0, 1), 0.125);
                float b = touchR16(normalized16, ivec3(1, 2, 3), 0.25);
                float c = touchR16f(halfLayers, ivec3(4, 5, 6), 0.5);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<float, access::read_write> normalized8 [[texture(0)]], texture3d<float, access::read_write> normalized16 [[texture(1)]], texture2d_array<float, access::read_write> halfLayers [[texture(2)]], texture2d<float, access::read_write> signedNormalized [[texture(3)]])"
        in generated_code
    )
    assert (
        "float touchR8(texture2d<float, access::read_write> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float touchR16(texture3d<float, access::read_write> image, int3 voxel, float value)"
        in generated_code
    )
    assert (
        "float touchR16f(texture2d_array<float, access::read_write> image, int3 pixelLayer, float value)"
        in generated_code
    )
    assert "float oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "float oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert (
        "float oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert "image.write(float4(oldValue + value), uint2(pixel));" in generated_code
    assert "image.write(float4(oldValue + value), uint3(voxel));" in generated_code
    assert (
        "image.write(float4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert ".read(uint2(pixel));" not in generated_code
    assert "image.write(oldValue + value" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_narrow_integer_image_formats():
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

        compute {
            void main() {
                int a = loadSigned8(signed8, ivec2(0, 1), -2);
                int b = loadSigned16(signed16, ivec3(1, 2, 3), -4);
                uint c = loadUnsigned8(unsigned8, ivec2(2, 3), 5);
                uint d = loadUnsigned16(unsigned16, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<int, access::read_write> signed8 [[texture(0)]], texture3d<int, access::read_write> signed16 [[texture(1)]], texture2d<uint, access::read_write> unsigned8 [[texture(2)]], texture2d_array<uint, access::read_write> unsigned16 [[texture(3)]])"
        in generated_code
    )
    assert (
        "int loadSigned8(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "int loadSigned16(texture3d<int, access::read_write> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned8(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned16(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "int oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "int oldValue = image.read(uint3(voxel)).x;" in generated_code
    assert "uint oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert (
        "uint oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z)).x;"
        in generated_code
    )
    assert "image.write(int4(oldValue + value), uint2(pixel));" in generated_code
    assert "image.write(int4(oldValue + value), uint3(voxel));" in generated_code
    assert "image.write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert (
        "image.write(uint4(oldValue + value), uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "texture2d<float, access::read_write> signed8" not in generated_code
    assert "texture3d<float, access::read_write> signed16" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_explicit_integer_image_formats_use_texture_atomics():
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
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> unsignedCounters [[texture(0)]], texture2d<int, access::read_write> signedCounters [[texture(1)]], texture3d<uint, access::read_write> unsignedVolume [[texture(2)]], texture2d_array<int, access::read_write> signedLayers [[texture(3)]])"
        in generated_code
    )
    assert (
        "uint addUnsigned(texture2d<uint, access::read_write> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int minSigned(texture2d<int, access::read_write> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "uint swapVolume(texture3d<uint, access::read_write> image, int3 voxel, uint expected, uint value)"
        in generated_code
    )
    assert (
        "int exchangeLayer(texture2d_array<int, access::read_write> image, int3 pixelLayer, int value)"
        in generated_code
    )
    assert "return image.atomic_fetch_add(uint2(pixel), value).x;" in generated_code
    assert "return image.atomic_fetch_min(uint2(pixel), value).x;" in generated_code
    assert (
        "return imageAtomicCompSwap_uimage3D(image, voxel, expected, value);"
        in generated_code
    )
    assert (
        "return image.atomic_exchange(uint2(pixelLayer.xy), uint(pixelLayer.z), value).x;"
        in generated_code
    )
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicMin(" not in generated_code
    assert "imageAtomicExchange(" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code


def test_metal_explicit_vector_integer_image_formats():
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

        compute {
            void main() {
                uvec4 a = touchUnsigned(unsignedColor, ivec2(0, 1), uvec4(1, 2, 3, 4));
                ivec4 b = touchSigned(signedVolume, ivec3(1, 2, 3), ivec4(-1, -2, -3, -4));
                uvec4 c = touchLayers(unsignedLayers, ivec3(4, 5, 6), uvec4(5, 6, 7, 8));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "kernel void kernel_main(texture2d<uint, access::read_write> unsignedColor [[texture(0)]], texture3d<int, access::read_write> signedVolume [[texture(1)]], texture2d_array<uint, access::read_write> unsignedLayers [[texture(2)]])"
        in generated_code
    )
    assert (
        "uint4 touchUnsigned(texture2d<uint, access::read_write> image, int2 pixel, uint4 value)"
        in generated_code
    )
    assert (
        "int4 touchSigned(texture3d<int, access::read_write> image, int3 voxel, int4 value)"
        in generated_code
    )
    assert (
        "uint4 touchLayers(texture2d_array<uint, access::read_write> image, int3 pixelLayer, uint4 value)"
        in generated_code
    )
    assert "uint4 oldValue = image.read(uint2(pixel));" in generated_code
    assert "int4 oldValue = image.read(uint3(voxel));" in generated_code
    assert (
        "uint4 oldValue = image.read(uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "image.write(oldValue + value, uint2(pixel));" in generated_code
    assert "image.write(oldValue + value, uint3(voxel));" in generated_code
    assert (
        "image.write(oldValue + value, uint2(pixelLayer.xy), uint(pixelLayer.z));"
        in generated_code
    )
    assert "texture2d<float, access::read_write> unsignedColor" not in generated_code
    assert "texture3d<float, access::read_write> signedVolume" not in generated_code
    assert ".read(uint2(pixel)).x" not in generated_code
    assert "uint4(oldValue + value)" not in generated_code
    assert "int4(oldValue + value)" not in generated_code


def test_metal_texture_array_helper_parameter_uses_resource_array():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "float4 sampleLayer(array<texture2d<float>, 4> textures" in generated_code
    assert (
        "textures[layer].sample(sampler(mag_filter::linear, min_filter::linear), uv)"
        in generated_code
    )
    assert "texture2d<float> textures[4] [[stage_in]]" not in generated_code


def test_metal_texture_array_helper_parameter_uses_indexed_sampler_array():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[layer].sample(samplers[layer], uv)" in generated_code
    assert "sampler samplers[4] [[stage_in]]" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_infer_helper_size():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 3> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(3)]]" in generated_code
    assert "array<sampler, 3> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 3> textures, array<sampler, 3> samplers"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "textures[1].sample(samplers[1], uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code
    assert "array<sampler, 1> samplers" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_infer_transitive_helper_size():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(5)]]" in generated_code
    assert "array<sampler, 5> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleDeep(array<texture2d<float>, 5> textures, array<sampler, 5> samplers"
        in generated_code
    )
    assert (
        "float4 sampleMid(array<texture2d<float>, 5> textures, array<sampler, 5> samplers"
        in generated_code
    )
    assert "textures[4].sample(samplers[4], uv)" in generated_code
    assert "textures[1].sample(samplers[1], uv)" in generated_code
    assert "sampleDeep(textures, samplers, uv)" in generated_code
    assert "sampleMid(textures, samplers, input.uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code
    assert "array<sampler, 1> samplers" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_preserve_dynamic_indexing():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[layer].sample(samplers[layer], uv)" in generated_code
    assert "textures[3].sample(samplers[3], uv)" in generated_code
    assert "sampleLayer(textures, samplers, input.layer, input.uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code
    assert "array<sampler, 1> samplers" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_ignore_unsupported_indices():
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

    dynamic_code = MetalCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = MetalCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "array<texture2d<float>, 1> textures [[texture(0)]]" in dynamic_code
    assert "texture2d<float> afterTexture [[texture(1)]]" in dynamic_code
    assert "array<sampler, 1> samplers [[sampler(0)]]" in dynamic_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 1> textures, array<sampler, 1> samplers"
        in dynamic_code
    )
    assert "textures[layer].sample(samplers[layer], uv)" in dynamic_code
    assert "array<texture2d<float>, 2> textures" not in dynamic_code
    assert "texture2d<float> afterTexture [[texture(2)]]" not in dynamic_code

    assert "array<texture2d<float>, 1> textures [[texture(0)]]" in negative_code
    assert "texture2d<float> afterTexture [[texture(1)]]" in negative_code
    assert "array<sampler, 1> samplers [[sampler(0)]]" in negative_code
    assert "textures[-1].sample(samplers[-1], uv)" in negative_code
    assert "array<texture2d<float>, 0> textures" not in negative_code
    assert "texture2d<float> afterTexture [[texture(0)]]" not in negative_code


def test_metal_unsized_texture_and_sampler_arrays_infer_constant_expression_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[1 + 2].sample(samplers[1 + 2], uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_infer_named_constant_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE = 1;" in generated_code
    assert "constant int LAYER = BASE + 2;" in generated_code
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, array<sampler, 4> samplers"
        in generated_code
    )
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in generated_code
    assert "array<texture2d<float>, 1> textures" not in generated_code


def test_metal_unsized_texture_and_sampler_arrays_ignore_shadowed_constant_name():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 3;" in generated_code
    assert "array<texture2d<float>, 1> textures [[texture(0)]]" in generated_code
    assert "texture2d<float> afterTexture [[texture(1)]]" in generated_code
    assert "array<sampler, 1> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleLayer(array<texture2d<float>, 1> textures, array<sampler, 1> samplers, int LAYER"
        in generated_code
    )
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in generated_code
    assert "array<texture2d<float>, 4> textures" not in generated_code
    assert "texture2d<float> afterTexture [[texture(4)]]" not in generated_code


def test_metal_texture_array_helper_operation_variants():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleOps(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, int layer, float2 uv, int2 pixel)"
        in generated_code
    )
    assert "textures[layer].sample(samplers[layer], uv, level(2.0))" in generated_code
    assert (
        "textures[layer].sample(samplers[layer], uv, gradient2d(float2(0.1), float2(0.2)))"
        in generated_code
    )
    assert "textures[layer].gather(samplers[layer], uv)" in generated_code
    assert "textures[layer].read(pixel, 0)" in generated_code
    assert (
        "sampleOps(textures, samplers, input.layer, input.uv, input.pixel)"
        in generated_code
    )
    assert "int layer [[stage_in]]" not in generated_code
    assert "float2 uv [[stage_in]]" not in generated_code
    assert "int2 pixel [[stage_in]]" not in generated_code


def test_metal_array_texture_types_split_coordinates_and_layers():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d_array<float> colorArray [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "depthcube<float> cubeShadow [[texture(2)]]" in generated_code
    assert "sampler arraySampler [[sampler(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(1)]]" in generated_code
    assert (
        "float4 sampleArray(texture2d_array<float> tex, sampler s, float3 uvLayer, int3 pixelLayer)"
        in generated_code
    )
    assert "tex.sample(s, uvLayer.xy, uint(uvLayer.z))" in generated_code
    assert "tex.sample(s, uvLayer.xy, uint(uvLayer.z), level(1.0))" in generated_code
    assert (
        "tex.sample(s, uvLayer.xy, uint(uvLayer.z), gradient2d(float2(0.1), float2(0.2)))"
        in generated_code
    )
    assert "tex.gather(s, uvLayer.xy, uint(uvLayer.z))" in generated_code
    assert "tex.read(pixelLayer.xy, uint(pixelLayer.z), 0)" in generated_code
    assert (
        "float sampleShadowArray(depth2d_array<float> tex, sampler s, float3 uvLayer, float depth)"
        in generated_code
    )
    assert "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth)" in generated_code
    assert (
        "float sampleCubeShadow(depthcube<float> tex, sampler s, float3 direction, float depth)"
        in generated_code
    )
    assert "tex.sample_compare(s, direction, depth)" in generated_code
    assert ".sample(s, uvLayer)" not in generated_code
    assert ".sample_compare(s, uvLayer, depth)" not in generated_code


def test_metal_cube_array_and_multisample_texture_types():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texturecube_array<float> cubeArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArrayShadow [[texture(1)]]" in generated_code
    assert "texture2d_ms<float> msTex [[texture(2)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(3)]]" in generated_code
    assert "sampler cubeSampler [[sampler(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(1)]]" in generated_code
    assert (
        "float4 sampleCubeArray(texturecube_array<float> tex, sampler s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.sample(s, cubeLayer.xyz, uint(cubeLayer.w))" in generated_code
    assert (
        "tex.sample(s, cubeLayer.xyz, uint(cubeLayer.w), level(2.0))" in generated_code
    )
    assert (
        "float sampleCubeArrayShadow(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in generated_code
    )
    assert (
        "float4 fetchMs(texture2d_ms<float> tex, texture2d_ms_array<float> texArray, int2 pixel, int3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "tex.read(pixel, uint(sampleIndex))" in generated_code
    assert (
        "texArray.read(pixelLayer.xy, uint(pixelLayer.z), uint(sampleIndex))"
        in generated_code
    )
    assert "tex.sample(s, cubeLayer)" not in generated_code
    assert "tex.sample_compare(s, cubeLayer, depth)" not in generated_code


def test_metal_texture_query_functions():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texture2d_ms<float> msMap [[texture(2)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "int2 size = int2(tex.get_width(uint(1)), tex.get_height(uint(1)));"
        in generated_code
    )
    assert "int levels = int(tex.get_num_mip_levels());" in generated_code
    assert (
        "float2 lod = float2(tex.calculate_unclamped_lod(s, uv), tex.calculate_clamped_lod(s, uv));"
        in generated_code
    )
    assert (
        "return int3(tex.get_width(uint(0)), tex.get_height(uint(0)), tex.get_array_size());"
        in generated_code
    )
    assert "return int2(tex.get_width(), tex.get_height());" in generated_code


def test_metal_texture_operation_variants():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "colorMap.sample(" in generated_code
    assert "level(1.0)" in generated_code
    assert "gradient2d(float2(0.1), float2(0.2))" in generated_code
    assert "colorMap.read(input.pixel, 0)" in generated_code
    assert "colorMap.gather(" in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "textureGather(" not in generated_code


def test_metal_texture_and_sampler_bindings_are_independent():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler colorMapSampler;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(colorMap, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler colorMapSampler [[sampler(0)]]" in generated_code
    assert "colorMap.sample(colorMapSampler, input.uv)" in generated_code


def test_metal_explicit_sampler_argument():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler linearSampler;

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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "colorMap.sample(linearSampler, input.uv)" in generated_code


def test_metal_sampler_parameter_texture_call():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "float4 sampleColor(sampler sampleState, float2 uv)" in generated_code
    assert "colorMap.sample(sampleState, uv)" in generated_code
    assert "sampleColor(linearSampler, input.uv)" in generated_code
    assert "sampleState [[stage_in]]" not in generated_code


def test_metal_texture_and_sampler_parameters():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleColor(texture2d<float> tex, sampler sampleState, float2 uv)"
        in generated_code
    )
    assert "tex.sample(sampleState, uv)" in generated_code
    assert "sampleColor(colorMap, linearSampler, input.uv)" in generated_code
    assert "tex [[stage_in]]" not in generated_code


def test_metal_implicit_sampler_for_texture_parameter():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "float4 sampleColor(texture2d<float> tex, float2 uv)" in generated_code
    assert (
        "tex.sample(sampler(mag_filter::linear, min_filter::linear), uv)"
        in generated_code
    )
    assert "sampleColor(colorMap, input.uv)" in generated_code


def test_metal_shadow_texture_compare():
    shader = """
    shader ShadowTexture {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMap, shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "shadowMap.sample_compare(shadowSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "textureCompare(" not in generated_code


def test_metal_shadow_texture_array_compare_with_indexed_sampler_array():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert "depth2d<float> shadowMaps[4] [[stage_in]]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int LAYER = 2;" in generated_code
    assert "array<depth2d<float>, 6> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(6)]]" in generated_code
    assert "array<sampler, 6> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 6> shadowMaps, array<sampler, 6> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].sample_compare(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert (
        "shadowMaps[1 + 2].sample_compare(shadowSamplers[1 + 2], uv, depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 4> shadowMaps" not in generated_code
    assert "depth2d<float> afterShadow [[texture(4)]]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE_COUNT = 2;" in generated_code
    assert "constant int SHADOW_COUNT = BASE_COUNT * 3;" in generated_code
    assert (
        "array<depth2d<float>, SHADOW_COUNT> shadowMaps [[texture(0)]]"
        in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(6)]]" in generated_code
    assert (
        "array<sampler, SHADOW_COUNT> shadowSamplers [[sampler(0)]]" in generated_code
    )
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, SHADOW_COUNT> shadowMaps, array<sampler, SHADOW_COUNT> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(1)]]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 2 * 3> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(6)]]" in generated_code
    assert "array<sampler, 2 * 3> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 2 * 3> shadowMaps, array<sampler, 2 * 3> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(1)]]" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_fixed_shadow_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<depth2d<float>, (2 + 1) * 2> shadowMaps [[texture(0)]]" in generated_code
    )
    assert "array<depth2d<float>, +6> unaryShadowMaps [[texture(6)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(12)]]" in generated_code
    assert "array<sampler, (2 + 1) * 2> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(6)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, (2 + 1) * 2> shadowMaps, array<sampler, (2 + 1) * 2> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert (
        "unaryShadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)"
        in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(6)]]" not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert "BinaryOpNode" not in generated_code
    assert "None" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_helper_size():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(4)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(4)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[3].sample_compare(shadowSamplers[3], uv, depth)" in generated_code
    )
    assert (
        "shadowMaps[1].sample_compare(shadowSamplers[1], uv, depth)" in generated_code
    )
    assert (
        "afterShadow.sample_compare(afterSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "array<sampler, 1> shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_transitive_helper_size():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(5)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowDeep(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "float shadowMid(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[4].sample_compare(shadowSamplers[4], uv, depth)" in generated_code
    )
    assert (
        "shadowMaps[1].sample_compare(shadowSamplers[1], uv, depth)" in generated_code
    )
    assert "shadowDeep(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth)" in generated_code
    )
    assert (
        "afterShadow.sample_compare(afterSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "array<sampler, 1> shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_preserve_dynamic_indexing():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(4)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(4)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert (
        "shadowMaps[3].sample_compare(shadowSamplers[3], uv, depth)" in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth)"
        in generated_code
    )
    assert (
        "afterShadow.sample_compare(afterSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "array<sampler, 1> shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_ignore_unsupported_indices():
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

    dynamic_code = MetalCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = MetalCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "array<depth2d<float>, 1> shadowMaps [[texture(0)]]" in dynamic_code
    assert "depth2d<float> afterShadow [[texture(1)]]" in dynamic_code
    assert "array<sampler, 1> shadowSamplers [[sampler(0)]]" in dynamic_code
    assert "sampler afterSampler [[sampler(1)]]" in dynamic_code
    assert (
        "float shadowLayer(array<depth2d<float>, 1> shadowMaps, array<sampler, 1> shadowSamplers"
        in dynamic_code
    )
    assert (
        "shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth)"
        in dynamic_code
    )
    assert "array<depth2d<float>, 2> shadowMaps" not in dynamic_code
    assert "depth2d<float> afterShadow [[texture(2)]]" not in dynamic_code

    assert "array<depth2d<float>, 1> shadowMaps [[texture(0)]]" in negative_code
    assert "depth2d<float> afterShadow [[texture(1)]]" in negative_code
    assert "array<sampler, 1> shadowSamplers [[sampler(0)]]" in negative_code
    assert "sampler afterSampler [[sampler(1)]]" in negative_code
    assert (
        "shadowMaps[-1].sample_compare(shadowSamplers[-1], uv, depth)" in negative_code
    )
    assert "array<depth2d<float>, 0> shadowMaps" not in negative_code
    assert "depth2d<float> afterShadow [[texture(0)]]" not in negative_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_constant_expression_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(5)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2 * 2].sample_compare(shadowSamplers[2 * 2], uv, depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_unsized_shadow_texture_and_sampler_arrays_infer_named_constant_size():
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

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int BASE = 2;" in generated_code
    assert "constant int LAYER = BASE * 2;" in generated_code
    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert "depth2d<float> afterShadow [[texture(5)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 5> shadowMaps, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].sample_compare(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_shadow_compare_sampler_parameter():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "float sampleShadow(sampler compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMap.sample_compare(compareSampler, uv, depth)" in generated_code
    assert "sampleShadow(shadowSampler, input.uv, input.depth)" in generated_code
    assert "compareSampler [[stage_in]]" not in generated_code


def test_metal_shadow_texture_and_sampler_parameters():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "float sampleShadow(depth2d<float> tex, sampler compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "tex.sample_compare(compareSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowMap, shadowSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "tex [[stage_in]]" not in generated_code


def test_metal_implicit_sampler_for_shadow_texture_parameter():
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
    generated_code = MetalCodeGen().generate(ast)

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert (
        "float sampleShadow(depth2d<float> tex, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "tex.sample_compare(sampler(mag_filter::linear, min_filter::linear), uv, depth)"
        in generated_code
    )
    assert "sampleShadow(shadowMap, input.uv, input.depth)" in generated_code


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
    code_gen = MetalCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


if __name__ == "__main__":
    pytest.main()
