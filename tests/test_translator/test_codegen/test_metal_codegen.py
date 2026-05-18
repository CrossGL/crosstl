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


def test_metal_unsigned_integer_literal_suffix_codegen():
    code = """
    shader UIntLiteralCodegen {
        compute {
            void main() {
                uint a = 7u;
                uint b = 0xFu;
                uint c = 0b101U;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "uint a = 7u;" in generated_code
    assert "uint b = 15u;" in generated_code
    assert "uint c = 5u;" in generated_code
    assert "7, u" not in generated_code


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


def test_for_statement_preserves_declaration_initializers():
    shader = """
    shader LoopDeclarationInitializers {
        float helper() {
            const float weights[2];
            int i = 0;
            float total = 0.0;
            for (int i = 0; i < 2; i++) {
                total = total + weights[0];
            }
            for (i = 0; i < 4; i++) {
                if (i == 0) {
                    continue;
                }
                break;
            }
            for (const int fixed = 0; fixed < 0;) {
                total = total + 1.0;
            }
            for (;;) {
                break;
            }
            return total;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "const float weights[2];" in generated_code
    assert "for (int i = 0; i < 2; ++i)" in generated_code
    assert "for (i = 0; i < 4; ++i)" in generated_code
    assert "for (const int fixed = 0; fixed < 0; )" in generated_code
    assert "for (; ; )" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "for (i; i < 2; ++i)" not in generated_code
    assert "for (fixed; fixed < 0; )" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code


def test_loop_statement_lowers_to_while_true():
    shader = """
    shader LoopNodeSmoke {
        int helper(int limit) {
            int i = 0;
            loop {
                i = i + 1;
                if (i >= limit) {
                    break;
                }
            }
            return i;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (true)" in generated_code
    assert "i = i + 1;" in generated_code
    assert "if (i >= limit)" in generated_code
    assert "break;" in generated_code
    assert "return i;" in generated_code
    assert "LoopNode(" not in generated_code


def test_for_in_statement_lowers_to_counted_loop():
    shader = """
    shader ForInNodeSmoke {
        int helper(int limit) {
            int total = 0;
            for i in limit {
                total = total + i;
            }
            return total;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 0; i < limit; ++i)" in generated_code
    assert "total = total + i;" in generated_code
    assert "return total;" in generated_code
    assert "ForInNode(" not in generated_code


def test_for_in_range_statement_lowers_to_counted_loop():
    shader = """
    shader ForInRangeNodeSmoke {
        int helper(int limit) {
            int total = 0;
            for i in 2..5 {
                total = total + i;
            }
            for j in 1..=limit {
                total = total + j;
            }
            return total;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 2; i < 5; ++i)" in generated_code
    assert "for (int j = 1; j <= limit; ++j)" in generated_code
    assert "total = total + i;" in generated_code
    assert "total = total + j;" in generated_code
    assert "return total;" in generated_code
    assert "RangeNode(" not in generated_code
    assert "ForInNode(" not in generated_code


def test_while_switch_and_void_return_emit_c_style_syntax():
    shader = """
    shader StatementLeakSmoke {
        void helper() {
            int i = 0;
            while (i < 4) {
                switch (i) {
                    case 0:
                        i = i + 1;
                        continue;
                    default:
                        break;
                }
            }
            return;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (i < 4)" in generated_code
    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "i = i + 1;" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "return;" in generated_code
    assert "WhileNode(" not in generated_code
    assert "SwitchNode(" not in generated_code
    assert "return ;" not in generated_code
    assert "return None;" not in generated_code


def test_switch_fallthrough_and_nested_switch_emit_c_style_syntax():
    shader = """
    shader SwitchEdgeSmoke {
        int helper(int mode, int submode) {
            int value = 0;
            switch (mode) {
                case 0:
                case 1:
                    value = value + 1;
                    break;
                case 2:
                    switch (submode) {
                        case 0:
                            value = value + 2;
                            break;
                        default:
                            value = value + 3;
                            break;
                    }
                    break;
                default:
                    value = value + 4;
                    break;
            }
            return value;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:\n        case 1:" in generated_code
    assert "case 2:\n            switch (submode)" in generated_code
    assert generated_code.count("default:") == 2
    assert "value = value + 1;" in generated_code
    assert "value = value + 2;" in generated_code
    assert "value = value + 3;" in generated_code
    assert "value = value + 4;" in generated_code
    assert "return value;" in generated_code
    assert "SwitchNode(" not in generated_code
    assert "CaseNode(" not in generated_code


def test_match_literal_and_wildcard_arms_lower_to_switch():
    shader = """
    shader MatchLeakSmoke {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                1 => { value = 2; }
                _ => { value = 3; }
            }
            return value;
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "case 1:" in generated_code
    assert "default:" in generated_code
    assert "value = 1;" in generated_code
    assert "value = 2;" in generated_code
    assert "value = 3;" in generated_code
    assert generated_code.count("break;") == 3
    assert "return value;" in generated_code
    assert "MatchNode(" not in generated_code
    assert "MatchArmNode(" not in generated_code


def test_match_return_arms_do_not_emit_extra_breaks():
    shader = """
    shader MatchReturnArms {
        int helper(int mode) {
            match mode {
                0 => { return 1; }
                _ => { return 2; }
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "return 1;" in generated_code
    assert "return 2;" in generated_code
    assert "break;" not in generated_code
    assert "MatchNode(" not in generated_code


def test_match_unsupported_binding_or_guarded_arm_raises():
    binding_shader = """
    shader MatchBindingPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                other => { value = other; }
            }
            return value;
        }
    }
    """
    guarded_shader = """
    shader MatchGuardPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 if mode > 0 => { value = 1; }
                _ => { value = 2; }
            }
            return value;
        }
    }
    """

    for shader in (binding_shader, guarded_shader):
        with pytest.raises(ValueError, match="Unsupported match arm"):
            MetalCodeGen().generate(crosstl.translator.parse(shader))


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


def test_generate_stage_filters_combined_vertex_fragment_units():
    code = """
    shader combined {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        float adjust(float value) {
            return value + 1.0;
        }

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generator = MetalCodeGen()

    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")

    assert "float adjust(float value)" in vertex_code
    assert "float adjust(float value)" in fragment_code
    assert "vertex VSOutput vertex_main" in vertex_code
    assert "fragment float4 fragment_main" not in vertex_code
    assert "fragment float4 fragment_main" in fragment_code
    assert "vertex VSOutput vertex_main" not in fragment_code


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


def test_compute_direct_builtin_references_inject_metal_parameters():
    code = """
    shader cs {
        compute {
            void main() {
                uint gx = gl_GlobalInvocationID.x;
                uint lx = gl_LocalInvocationID.x;
                uint group = gl_WorkGroupID.x;
                uint index = gl_LocalInvocationIndex;
                uint size = gl_WorkGroupSize.x;
                uint groups = gl_NumWorkGroups.x;
            }
        }
    }
    """
    ast = crosstl.translator.parse(code)
    generated = MetalCodeGen().generate_stage(ast, "compute")

    assert "uint3 gl_GlobalInvocationID [[thread_position_in_grid]]" in generated
    assert "uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]]" in generated
    assert "uint3 gl_WorkGroupID [[threadgroup_position_in_grid]]" in generated
    assert "uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]]" in generated
    assert "uint3 gl_WorkGroupSize [[threads_per_threadgroup]]" in generated
    assert "uint3 gl_NumWorkGroups [[threadgroups_per_grid]]" in generated
    assert "uint gx = gl_GlobalInvocationID.x;" in generated


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


def test_metal_direct_stage_image_load_store_and_atomics_use_input_members():
    shader = """
    shader DirectStageImageOps {
        image2D outImg;
        image3D volume;
        image2DArray layers;
        uimage2D counters;
        iimage2DArray signedLayers;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 voxel @ TEXCOORD1;
            ivec3 pixelLayer @ TEXCOORD2;
            uint amount;
            int signedAmount;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = imageLoad(outImg, input.pixel);
                vec4 volumeColor = imageLoad(volume, input.voxel);
                vec4 layerColor = imageLoad(layers, input.pixelLayer);
                imageStore(outImg, input.pixel, color + layerColor);
                imageStore(volume, input.voxel, volumeColor);
                imageStore(layers, input.pixelLayer, color);
                uint previous = imageAtomicAdd(counters, input.pixel, input.amount);
                int previousSigned = imageAtomicMin(signedLayers, input.pixelLayer, input.signedAmount);
                return color + volumeColor + layerColor + vec4(float(previous + uint(previousSigned)));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float, access::read_write> outImg [[texture(0)]], texture3d<float, access::read_write> volume [[texture(1)]], texture2d_array<float, access::read_write> layers [[texture(2)]], texture2d<uint, access::read_write> counters [[texture(3)]], texture2d_array<int, access::read_write> signedLayers [[texture(4)]])"
        in generated_code
    )
    assert "float4 color = outImg.read(uint2(input.pixel));" in generated_code
    assert "float4 volumeColor = volume.read(uint3(input.voxel));" in generated_code
    assert (
        "float4 layerColor = layers.read(uint2(input.pixelLayer.xy), uint(input.pixelLayer.z));"
        in generated_code
    )
    assert "outImg.write(color + layerColor, uint2(input.pixel));" in generated_code
    assert "volume.write(volumeColor, uint3(input.voxel));" in generated_code
    assert (
        "layers.write(color, uint2(input.pixelLayer.xy), uint(input.pixelLayer.z));"
        in generated_code
    )
    assert (
        "uint previous = counters.atomic_fetch_add(uint2(input.pixel), input.amount).x;"
        in generated_code
    )
    assert (
        "int previousSigned = signedLayers.atomic_fetch_min(uint2(input.pixelLayer.xy), uint(input.pixelLayer.z), input.signedAmount).x;"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "imageAtomicAdd(" not in generated_code
    assert "imageAtomicMin(" not in generated_code


def test_metal_direct_stage_explicit_image_formats_use_input_members():
    shader = """
    shader DirectStageImageFormats {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        uimage2D unsignedScalar @r32ui;
        uimage2DArray unsignedLayers @rg32ui;
        iimage3D signedVolume @r32i;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 voxel @ TEXCOORD1;
            ivec3 pixelLayer @ TEXCOORD2;
            float amount;
            uint unsignedAmount;
            int signedAmount;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float scalarValue = imageLoad(scalarFloat, input.pixel);
                vec2 rgValue = imageLoad(rgFloat, input.pixel);
                uint unsignedValue = imageLoad(unsignedScalar, input.pixel);
                uvec2 layerValue = imageLoad(unsignedLayers, input.pixelLayer);
                int signedValue = imageLoad(signedVolume, input.voxel);
                imageStore(scalarFloat, input.pixel, scalarValue + input.amount);
                imageStore(rgFloat, input.pixel, rgValue + vec2(input.amount));
                imageStore(unsignedScalar, input.pixel, unsignedValue + input.unsignedAmount);
                imageStore(unsignedLayers, input.pixelLayer, layerValue + uvec2(input.unsignedAmount));
                imageStore(signedVolume, input.voxel, signedValue + input.signedAmount);
                return vec4(scalarValue + rgValue.x + float(unsignedValue + layerValue.x) + float(signedValue));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float, access::read_write> scalarFloat [[texture(0)]], texture2d<float, access::read_write> rgFloat [[texture(1)]], texture2d<uint, access::read_write> unsignedScalar [[texture(2)]], texture2d_array<uint, access::read_write> unsignedLayers [[texture(3)]], texture3d<int, access::read_write> signedVolume [[texture(4)]])"
        in generated_code
    )
    assert (
        "float scalarValue = scalarFloat.read(uint2(input.pixel)).x;" in generated_code
    )
    assert "float2 rgValue = rgFloat.read(uint2(input.pixel)).xy;" in generated_code
    assert (
        "uint unsignedValue = unsignedScalar.read(uint2(input.pixel)).x;"
        in generated_code
    )
    assert (
        "uint2 layerValue = unsignedLayers.read(uint2(input.pixelLayer.xy), uint(input.pixelLayer.z)).xy;"
        in generated_code
    )
    assert (
        "int signedValue = signedVolume.read(uint3(input.voxel)).x;" in generated_code
    )
    assert (
        "scalarFloat.write(float4(scalarValue + input.amount), uint2(input.pixel));"
        in generated_code
    )
    assert (
        "rgFloat.write(float4(rgValue + float2(input.amount), 0.0, 0.0), uint2(input.pixel));"
        in generated_code
    )
    assert (
        "unsignedScalar.write(uint4(unsignedValue + input.unsignedAmount), uint2(input.pixel));"
        in generated_code
    )
    assert (
        "unsignedLayers.write(uint4(layerValue + uint2(input.unsignedAmount), 0u, 0u), uint2(input.pixelLayer.xy), uint(input.pixelLayer.z));"
        in generated_code
    )
    assert (
        "signedVolume.write(int4(signedValue + input.signedAmount), uint3(input.voxel));"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_direct_stage_image_compare_swap_use_input_members():
    shader = """
    shader DirectStageImageCompareSwap {
        uimage3D volumeCounters;
        iimage2DArray layerCounters;

        struct FSInput {
            ivec3 voxel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            uint expected;
            uint replacement;
            int signedExpected;
            int signedReplacement;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                uint volumeOld = imageAtomicCompSwap(volumeCounters, input.voxel, input.expected, input.replacement);
                int layerOld = imageAtomicCompSwap(layerCounters, input.pixelLayer, input.signedExpected, input.signedReplacement);
                return vec4(float(volumeOld), float(layerOld), 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture3d<uint, access::read_write> volumeCounters [[texture(0)]], texture2d_array<int, access::read_write> layerCounters [[texture(1)]])"
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
        "uint volumeOld = imageAtomicCompSwap_uimage3D(volumeCounters, input.voxel, input.expected, input.replacement);"
        in generated_code
    )
    assert (
        "int layerOld = imageAtomicCompSwap_iimage2DArray(layerCounters, input.pixelLayer, input.signedExpected, input.signedReplacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(volumeCounters" not in generated_code
    assert "imageAtomicCompSwap(layerCounters" not in generated_code


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


def test_metal_default_float_image_scalar_and_vector_load_store():
    shader = """
    shader DefaultFloatImageLoadStore {
        image2D storageImage;

        float touchScalar(image2D image, ivec2 pixel, float value) {
            float scalarOld = imageLoad(image, pixel);
            imageStore(image, pixel, scalarOld + value);
            return scalarOld;
        }

        vec4 touchVector(image2D image, ivec2 pixel, vec4 value) {
            vec4 vectorOld = imageLoad(image, pixel);
            imageStore(image, pixel, vectorOld + value);
            return vectorOld;
        }

        compute {
            void main() {
                float a = touchScalar(storageImage, ivec2(0, 1), 0.25);
                vec4 b = touchVector(storageImage, ivec2(2, 3), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = MetalCodeGen().generate(ast)

    assert "float scalarOld = image.read(uint2(pixel)).x;" in generated_code
    assert "image.write(float4(scalarOld + value), uint2(pixel));" in generated_code
    assert "float4 vectorOld = image.read(uint2(pixel));" in generated_code
    assert "image.write(vectorOld + value, uint2(pixel));" in generated_code
    assert "float4 vectorOld = image.read(uint2(pixel)).x;" not in generated_code
    assert "image.write(float4(vectorOld + value), uint2(pixel));" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_rg_image_scalar_and_vector_load_store():
    shader = """
    shader RGImageScalarVector {
        image2D rgFloat @rg32f;
        uimage2D rgUnsigned @rg32ui;

        float scalarFloat(image2D image @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D image @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D image @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = scalarFloat(rgFloat, ivec2(0, 1), 0.25);
                vec2 b = vectorFloat(rgFloat, ivec2(2, 3), vec2(1.0));
                uint c = scalarUnsigned(rgUnsigned, ivec2(4, 5), 7u);
                uvec2 d = vectorUnsigned(rgUnsigned, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "float oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = image.read(uint2(pixel)).x;" in generated_code
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert "uint2 oldValue = image.read(uint2(pixel)).xy;" in generated_code
    assert (
        "image.write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert (
        "image.write(uint4(oldValue + value, 0u, 0u), uint2(pixel));" in generated_code
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


def test_metal_rg_image_arrays_respect_scalar_and_vector_context():
    shader = """
    shader RGImageArrayContext {
        image2D rgFloatImages @rg32f[3];
        uimage2D rgUnsignedImages @rg32ui[2];

        float scalarFloat(image2D images[3] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 3> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 2> rgUnsignedImages [[texture(3)]]"
        in generated_code
    )
    assert (
        "float scalarFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(array<texture2d<uint, access::read_write>, 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(array<texture2d<uint, access::read_write>, 2> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[1].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[1].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "uint2 oldValue = images[1].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "float oldValue = images[1].read(uint2(pixel));" not in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).x;" not in generated_code
    assert "uint oldValue = images[1].read(uint2(pixel));" not in generated_code
    assert "uint2 oldValue = images[1].read(uint2(pixel)).x;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_inferred_rg_image_arrays_respect_scalar_context():
    shader = """
    shader RGImageArrayInferredScalarContext {
        const int COUNT = 3;
        const int LAYER = COUNT - 1;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[COUNT];
        image2D afterImages @rg32f;

        float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 3;" in generated_code
    assert "constant int LAYER = COUNT - 1;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 3> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, COUNT> rgUnsignedImages [[texture(3)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(6)]]"
        in generated_code
    )
    assert (
        "float scalarFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(array<texture2d<float, access::read_write>, 3> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[1].write(uint4(oldValue + value, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(3)]]"
        not in generated_code
    )
    assert "float oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).x;" not in generated_code
    assert "uint oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).x;" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_transitive_rg_image_arrays_share_call_site_size():
    shader = """
    shader TransitiveRGImageArrayScalarContext {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];
        image2D afterImages @rg32f;

        float scalarFloatDeep(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        float scalarFloatMid(image2D images[] @rg32f, ivec2 pixel, float value) {
            return scalarFloatDeep(images, pixel, value);
        }

        vec2 vectorFloatDeep(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatMid(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            return vectorFloatDeep(images, pixel, value);
        }

        uint scalarUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            return scalarUnsignedDeep(images, pixel, value);
        }

        uvec2 vectorUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
            return vectorUnsignedDeep(images, pixel, value);
        }

        compute {
            void main() {
                float sf = scalarFloatMid(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatMid(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedMid(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedMid(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(8)]]"
        in generated_code
    )
    assert (
        "float scalarFloatDeep(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float scalarFloatMid(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatDeep(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatMid(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedDeep(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedDeep(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[1].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[1].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert (
        "float2 vectorFloatDeep(array<texture2d<float, access::read_write>, 3> images"
        not in generated_code
    )
    assert (
        "uint2 vectorUnsignedDeep(array<texture2d<uint, access::read_write>, 3> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_fixed_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedParamRGImageArrayContext {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[4] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> rgUnsignedImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_const_sized_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedConstParamRGImageArrayContext {
        const int COUNT = 4;
        const int LAST = COUNT - 1;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[LAST], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAST], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert "constant int LAST = COUNT - 1;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, COUNT> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(array<texture2d<float, access::read_write>, COUNT> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(array<texture2d<uint, access::read_write>, COUNT> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAST].read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = images[LAST].read(uint2(pixel)).x;" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, 4> images"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_expr_sized_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedExprParamRGImageArrayContext {
        const int COUNT = 3;
        const int UINT_COUNT = 2;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[COUNT], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 3;" in generated_code
    assert "constant int UINT_COUNT = 2;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, COUNT + 1> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(array<texture2d<float, access::read_write>, COUNT + 1> images, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(array<texture2d<uint, access::read_write>, UINT_COUNT * 2> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(array<texture2d<uint, access::read_write>, UINT_COUNT * 2> images, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "float2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert "uint2 oldValue = images[2].read(uint2(pixel)).xy;" in generated_code
    assert (
        "float scalarFloatFixed(array<texture2d<float, access::read_write>, 4> images"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_conflicting_fixed_rg_image_array_sizes_raise():
    shader = """
    shader ConflictingFixedRGImageArrayContext {
        image2D rgFloatImages @rg32f[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchThree(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchFour(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 b = touchThree(rgFloatImages, ivec2(2, 3), vec2(1.0));
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_direct_rg_image_array_index_conflicts_with_fixed_parameter_size():
    shader = """
    shader DirectIndexFixedConflict {
        image2D rgFloatImages @rg32f[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[5], pixel);
                float helper = touchFour(rgFloatImages, pixel, direct);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_direct_rg_image_array_index_within_fixed_parameter_size():
    shader = """
    shader DirectIndexWithinFixedSize {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint touchUnsignedFour(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float directFloat = imageLoad(rgFloatImages[2], pixel);
                uint directUint = imageLoad(rgUnsignedImages[1], pixel);
                float helperFloat = touchFour(rgFloatImages, pixel, directFloat);
                uint helperUint = touchUnsignedFour(rgUnsignedImages, pixel, directUint);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 4> rgUnsignedImages [[texture(4)]]"
        in generated_code
    )
    assert (
        "float touchFour(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint touchUnsignedFour(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float directFloat = rgFloatImages[2].read(uint2(pixel)).x;" in generated_code
    )
    assert (
        "uint directUint = rgUnsignedImages[1].read(uint2(pixel)).x;" in generated_code
    )
    assert "float oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert "uint oldValue = images[3].read(uint2(pixel)).x;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 3> rgFloatImages"
        not in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_fixed_rg_image_array_global_conflicts_with_fixed_parameter_size():
    shader = """
    shader FixedGlobalToMismatchedFixedHelper {
        image2D rgFloatImages @rg32f[4];

        float touchThree(image2D images[3] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchThree(rgFloatImages, ivec2(0, 1), 0.25);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_global_widens_unsized_parameter_size():
    shader = """
    shader FixedGlobalToUnsizedHelper {
        image2D rgFloatImages @rg32f[4];

        float touchUnsized(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchUnsized(rgFloatImages, ivec2(0, 1), 0.25);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float touchUnsized(array<texture2d<float, access::read_write>, 4> images, int2 pixel, float value)"
        in generated_code
    )
    assert "float oldValue = images[2].read(uint2(pixel)).x;" in generated_code
    assert (
        "float touchUnsized(array<texture2d<float, access::read_write>, 3> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_fixed_rg_image_array_global_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedGlobalDirectIndexOutOfBounds {
        image2D rgFloatImages @rg32f[4];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = imageLoad(rgFloatImages[4], pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_parameter_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedParameterDirectIndexOutOfBounds {
        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[4], pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_global_const_index_out_of_bounds_raises():
    shader = """
    shader FixedGlobalConstIndexOutOfBounds {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = imageLoad(rgFloatImages[COUNT], pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_parameter_const_index_out_of_bounds_raises():
    shader = """
    shader FixedParameterConstIndexOutOfBounds {
        const int COUNT = 4;

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_fixed_rg_image_array_const_index_within_bounds_generates():
    shader = """
    shader FixedConstIndexWithinBounds {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT - 1], pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[COUNT - 1], pixel);
                float helper = touch(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float touch(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert "return images[COUNT - 1].read(uint2(pixel)).x;" in generated_code
    assert (
        "float direct = rgFloatImages[COUNT - 1].read(uint2(pixel)).x;"
        in generated_code
    )
    assert (
        "float touch(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_metal_fixed_rg_image_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader FixedShadowedConstIndex {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel);
        }

        compute {
            void main() {
                int COUNT = 0;
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[COUNT], pixel);
                float helper = touch(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float touch(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert "return images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert "float direct = rgFloatImages[COUNT].read(uint2(pixel)).x;" in generated_code
    assert (
        "float touch(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_metal_transitive_rg_image_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveShadowedConstIndex {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float leaf(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel);
        }

        float passThrough(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            float sampled = imageLoad(images[COUNT], pixel);
            return sampled + leaf(images, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = passThrough(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "float leaf(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert (
        "float passThrough(array<texture2d<float, access::read_write>, 4> images, int2 pixel)"
        in generated_code
    )
    assert "return images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert "float sampled = images[COUNT].read(uint2(pixel)).x;" in generated_code
    assert (
        "float leaf(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert (
        "float passThrough(array<texture2d<float, access::read_write>, 5> images"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_metal_transitive_rg_image_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveUnshadowedConstIndexConflict {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float leaf(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        float passThrough(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return leaf(images, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = passThrough(rgFloatImages, pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_metal_shadowed_rg_image_array_constant_keeps_scalar_context():
    shader = """
    shader ShadowedRGImageArrayScalarContext {
        const int LAYER = 3;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];
        image2D afterImages @rg32f;

        float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
            int LAYER = 0;
            float oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            int LAYER = 0;
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "constant int LAYER = 3;" in generated_code
    assert (
        "array<texture2d<float, access::read_write>, 1> rgFloatImages [[texture(0)]]"
        in generated_code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 1> rgUnsignedImages [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(2)]]"
        in generated_code
    )
    assert (
        "float scalarFloat(array<texture2d<float, access::read_write>, 1> images, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "float oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(float4(oldValue + value, 0.0, 0.0, 0.0), uint2(pixel));"
        in generated_code
    )
    assert "uint oldValue = images[LAYER].read(uint2(pixel)).x;" in generated_code
    assert (
        "images[0].write(uint4(oldValue + value, 0u, 0u, 0u), uint2(pixel));"
        in generated_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages"
        not in generated_code
    )
    assert (
        "texture2d<float, access::read_write> afterImages [[texture(4)]]"
        not in generated_code
    )
    assert "float oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
    assert "uint oldValue = images[LAYER].read(uint2(pixel));" not in generated_code
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


def test_metal_formatted_image_arrays_ignore_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, int layer, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[layer], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, 0, ivec2(1, 2), 3);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[-1], pixel);
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

    dynamic_code = MetalCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = MetalCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in dynamic_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in dynamic_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int layer, int2 pixel, uint value)"
        in dynamic_code
    )
    assert "uint oldValue = images[layer].read(uint2(pixel)).x;" in dynamic_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in dynamic_code
    assert "uint a = touchCounters(counters, 0, int2(1, 2), 3);" in dynamic_code
    assert "array<texture2d<uint, access::read_write>, 2> counters" not in dynamic_code
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in dynamic_code
    )
    assert "imageLoad(" not in dynamic_code
    assert "imageStore(" not in dynamic_code

    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in negative_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in negative_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in negative_code
    )
    assert "uint oldValue = images[-1].read(uint2(pixel)).x;" in negative_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in negative_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in negative_code
    assert "array<texture2d<uint, access::read_write>, 0> counters" not in negative_code
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(0)]]"
        not in negative_code
    )
    assert "imageLoad(" not in negative_code
    assert "imageStore(" not in negative_code


def test_metal_formatted_image_arrays_ignore_function_call_indices():
    shader = """
    shader FunctionIndexedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        int getLayer() {
            return 0;
        }

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[getLayer()], pixel);
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

    assert (
        "array<texture2d<uint, access::read_write>, 1> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        in generated_code
    )
    assert "int getLayer()" in generated_code
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 1> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = images[getLayer()].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 2> counters" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(2)]]"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_metal_formatted_image_arrays_infer_local_constant_alias_size():
    shader = """
    shader LocalConstAliasFormattedImageArrays {
        const int GLOBAL = 2;
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            const int LOCAL = GLOBAL + 1;
            uint oldValue = imageLoad(images[LOCAL], pixel);
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

    assert "constant int GLOBAL = 2;" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(4)]]"
        in generated_code
    )
    assert (
        "uint touchCounters(array<texture2d<uint, access::read_write>, 4> images, int2 pixel, uint value)"
        in generated_code
    )
    assert "const int LOCAL = GLOBAL + 1;" in generated_code
    assert "uint oldValue = images[LOCAL].read(uint2(pixel)).x;" in generated_code
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "array<texture2d<uint, access::read_write>, 1> counters" not in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> afterCounters [[texture(1)]]"
        not in generated_code
    )
    assert "    int LOCAL = GLOBAL + 1;" not in generated_code
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


def test_metal_fixed_texture_array_global_conflicts_with_fixed_parameter_size():
    shader = """
    shader FixedSampledGlobalMismatch {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleThree(sampler2D textures[3], sampler samplers[3], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleThree(textures, samplers, input.uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_global_widens_unsized_parameter_size():
    shader = """
    shader FixedSampledGlobalToUnsizedHelper {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleUnsized(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleUnsized(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleUnsized(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert "textures[2].sample(samplers[2], uv)" in generated_code
    assert "sampleUnsized(textures, samplers, input.uv)" in generated_code
    assert (
        "float4 sampleUnsized(array<texture2d<float>, 3> textures" not in generated_code
    )


def test_metal_fixed_texture_array_global_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledGlobalDirectOutOfBounds {
        sampler2D textures[4];
        sampler samplers[4];

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(textures[4], samplers[4], uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_parameter_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledParamDirectOutOfBounds {
        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[4], samplers[4], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_global_const_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledGlobalConstIndexOutOfBounds {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(textures[COUNT], samplers[COUNT], uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_parameter_const_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledParamConstIndexOutOfBounds {
        const int COUNT = 4;

        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_metal_fixed_texture_array_const_index_and_shadowing_generate():
    shader = """
    shader FixedSampledConstIndexWithinBounds {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 sampleConst(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[COUNT - 1], samplers[COUNT - 1], uv);
        }

        vec4 sampleShadowed(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            int COUNT = 0;
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                int COUNT = 0;
                vec4 direct = texture(textures[COUNT], samplers[COUNT], input.uv);
                return direct + sampleConst(textures, samplers, input.uv) + sampleShadowed(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 sampleConst(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleShadowed(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert "textures[COUNT - 1].sample(samplers[COUNT - 1], uv)" in generated_code
    assert "textures[COUNT].sample(samplers[COUNT], uv)" in generated_code
    assert "sampleConst(textures, samplers, input.uv)" in generated_code
    assert "sampleShadowed(textures, samplers, input.uv)" in generated_code
    assert "array<texture2d<float>, 5> textures" not in generated_code
    assert "array<sampler, 5> samplers" not in generated_code


def test_metal_transitive_texture_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveSampledShadowedConstIndex {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            vec4 sampled = texture(textures[COUNT], samplers[COUNT], uv);
            return sampled + leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return passThrough(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "constant int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in generated_code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in generated_code
    assert (
        "float4 leaf(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert (
        "float4 passThrough(array<texture2d<float>, 4> textures, array<sampler, 4> samplers, float2 uv)"
        in generated_code
    )
    assert "return textures[COUNT].sample(samplers[COUNT], uv);" in generated_code
    assert (
        "float4 sampled = textures[COUNT].sample(samplers[COUNT], uv);"
        in generated_code
    )
    assert "return sampled + leaf(textures, samplers, uv);" in generated_code
    assert "return passThrough(textures, samplers, input.uv);" in generated_code
    assert "array<texture2d<float>, 5> textures" not in generated_code
    assert "array<sampler, 5> samplers" not in generated_code


def test_metal_transitive_texture_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveSampledUnshadowedConstIndexConflict {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            return leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return passThrough(textures, samplers, input.uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        MetalCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


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


def test_metal_cube_array_texture_grad_gather_uses_cube_gradients():
    shader = """
    shader CubeArrayGradGather {
        samplerCubeArray cubeArray;
        samplerCubeArray cubeArrays[4];
        sampler cubeSampler;
        sampler cubeSamplers[4];

        struct FSInput {
            vec4 cubeLayer @ TEXCOORD0;
            vec3 ddx @ TEXCOORD1;
            vec3 ddy @ TEXCOORD2;
        };

        vec4 sampleCubeArrayOps(samplerCubeArray tex, sampler s, vec4 cubeLayer, vec3 ddx, vec3 ddy) {
            vec4 gradColor = textureGrad(tex, s, cubeLayer, ddx, ddy);
            vec4 gathered = textureGather(tex, s, cubeLayer);
            return gradColor + gathered;
        }

        vec4 sampleCubeArrayElements(samplerCubeArray cubeArrays[], sampler cubeSamplers[], vec4 cubeLayer, vec3 ddx, vec3 ddy) {
            vec4 gradColor = textureGrad(cubeArrays[2], cubeSamplers[2], cubeLayer, ddx, ddy);
            vec4 gathered = textureGather(cubeArrays[3], cubeSamplers[3], cubeLayer);
            return gradColor + gathered;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)
                    + sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube_array<float> cubeArray [[texture(0)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(1)]]" in generated_code
    )
    assert "sampler cubeSampler [[sampler(0)]]" in generated_code
    assert "array<sampler, 4> cubeSamplers [[sampler(1)]]" in generated_code
    assert (
        "float4 sampleCubeArrayOps(texturecube_array<float> tex, sampler s, float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "tex.sample(s, cubeLayer.xyz, uint(cubeLayer.w), gradientcube(ddx, ddy))"
        in generated_code
    )
    assert "tex.gather(s, cubeLayer.xyz, uint(cubeLayer.w))" in generated_code
    assert (
        "float4 sampleCubeArrayElements(array<texturecube_array<float>, 4> cubeArrays, array<sampler, 4> cubeSamplers, float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "cubeArrays[2].sample(cubeSamplers[2], cubeLayer.xyz, uint(cubeLayer.w), gradientcube(ddx, ddy))"
        in generated_code
    )
    assert (
        "cubeArrays[3].gather(cubeSamplers[3], cubeLayer.xyz, uint(cubeLayer.w))"
        in generated_code
    )
    assert (
        "sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert "gradient2d(ddx, ddy)" not in generated_code
    assert ".sample(s, cubeLayer, gradientcube" not in generated_code


def test_metal_texture_grad_uses_dimension_specific_gradient_options():
    shader = """
    shader MetalGradientFamily {
        sampler2D colorMap;
        samplerCube cubeMap;
        sampler3D volumeMap;
        sampler colorSampler;
        sampler cubeSampler;
        sampler volumeSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 direction @ TEXCOORD1;
            vec3 volumeUv @ TEXCOORD2;
            vec2 ddx2 @ TEXCOORD3;
            vec2 ddy2 @ TEXCOORD4;
            vec3 ddx3 @ TEXCOORD5;
            vec3 ddy3 @ TEXCOORD6;
        };

        vec4 sample2DGrad(sampler2D tex, sampler s, vec2 uv, vec2 ddx, vec2 ddy) {
            return textureGrad(tex, s, uv, ddx, ddy);
        }

        vec4 sampleCubeGrad(samplerCube tex, sampler s, vec3 direction, vec3 ddx, vec3 ddy) {
            return textureGrad(tex, s, direction, ddx, ddy);
        }

        vec4 sampleVolumeGrad(sampler3D tex, sampler s, vec3 volumeUv, vec3 ddx, vec3 ddy) {
            return textureGrad(tex, s, volumeUv, ddx, ddy);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sample2DGrad(colorMap, colorSampler, input.uv, input.ddx2, input.ddy2)
                    + sampleCubeGrad(cubeMap, cubeSampler, input.direction, input.ddx3, input.ddy3)
                    + sampleVolumeGrad(volumeMap, volumeSampler, input.volumeUv, input.ddx3, input.ddy3);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(1)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(2)]]" in generated_code
    assert "sampler colorSampler [[sampler(0)]]" in generated_code
    assert "sampler cubeSampler [[sampler(1)]]" in generated_code
    assert "sampler volumeSampler [[sampler(2)]]" in generated_code
    assert "tex.sample(s, uv, gradient2d(ddx, ddy))" in generated_code
    assert "tex.sample(s, direction, gradientcube(ddx, ddy))" in generated_code
    assert "tex.sample(s, volumeUv, gradient3d(ddx, ddy))" in generated_code
    assert "gradient2d(ddx3, ddy3)" not in generated_code


def test_metal_texture_gather_offset_variants_use_metal_overloads():
    shader = """
    shader GatherOffsetVariants {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;
        sampler cubeSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 direction @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
            ivec2 offset0 @ TEXCOORD5;
            ivec2 offset1 @ TEXCOORD6;
            ivec2 offset2 @ TEXCOORD7;
            ivec2 offset3 @ TEXCOORD8;
            int component @ TEXCOORD9;
        };

        vec4 gatherOps(
            sampler2D tex,
            sampler2DArray layers,
            sampler s,
            vec2 uv,
            vec3 uvLayer,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 green = textureGather(tex, s, uv, 1);
            vec4 dynamic = textureGather(tex, s, uv, component);
            vec4 offsetGather = textureGatherOffset(tex, s, uv, offset, 3);
            vec4 dynamicOffset = textureGatherOffset(tex, s, uv, offset, component);
            vec4 offsetsGather = textureGatherOffsets(
                layers,
                s,
                uvLayer,
                offset0,
                offset1,
                offset2,
                offset3,
                component
            );
            return green + dynamic + offsetGather + dynamicOffset + offsetsGather;
        }

        vec4 gatherArrayOffsets(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 offsets[4]
        ) {
            return textureGatherOffsets(layers, s, uvLayer, offsets, 2);
        }

        vec4 gatherCubeOps(
            samplerCube cubeMap,
            samplerCubeArray cubeArray,
            sampler s,
            vec3 direction,
            vec4 cubeLayer
        ) {
            return textureGather(cubeMap, s, direction, 2)
                + textureGather(cubeArray, s, cubeLayer, 3);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherOps(
                    colorMap,
                    layerMap,
                    linearSampler,
                    input.uv,
                    input.uvLayer,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + gatherCubeOps(
                    cubeMap,
                    cubeArray,
                    cubeSampler,
                    input.direction,
                    input.cubeLayer
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(3)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "sampler cubeSampler [[sampler(1)]]" in generated_code
    assert "float4 green = tex.gather(s, uv, int2(0), component::y);" in generated_code
    assert (
        "component == 0 ? tex.gather(s, uv, int2(0), component::x) : "
        "component == 1 ? tex.gather(s, uv, int2(0), component::y) : "
        "component == 2 ? tex.gather(s, uv, int2(0), component::z) : "
        "tex.gather(s, uv, int2(0), component::w)" in generated_code
    )
    assert (
        "float4 offsetGather = tex.gather(s, uv, offset, component::w);"
        in generated_code
    )
    assert (
        "component == 0 ? tex.gather(s, uv, offset, component::x) : "
        "component == 1 ? tex.gather(s, uv, offset, component::y) : "
        "component == 2 ? tex.gather(s, uv, offset, component::z) : "
        "tex.gather(s, uv, offset, component::w)" in generated_code
    )
    assert (
        "float4(layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset0, component::x).x, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset1, component::x).y, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset2, component::x).z, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offset3, component::x).w)"
        in generated_code
    )
    assert (
        "float4 gatherArrayOffsets(texture2d_array<float> layers, sampler s, float3 uvLayer, int2 offsets[4])"
        in generated_code
    )
    assert (
        "return float4(layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[0], component::z).x, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[1], component::z).y, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[2], component::z).z, "
        "layers.gather(s, uvLayer.xy, uint(uvLayer.z), offsets[3], component::z).w);"
        in generated_code
    )
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "cubeMap.gather(s, direction, component::z)" in generated_code
    assert (
        "cubeArray.gather(s, cubeLayer.xyz, uint(cubeLayer.w), component::w)"
        in generated_code
    )
    assert "cubeMap.gather(s, direction, int2(0)" not in generated_code
    assert (
        "cubeArray.gather(s, cubeLayer.xyz, uint(cubeLayer.w), int2(0)"
        not in generated_code
    )


def test_metal_cube_texture_gather_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedCubeGatherOffsets {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler cubeSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        vec4 gatherCube(
            samplerCube tex,
            sampler s,
            vec3 direction,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 offsetGather = textureGatherOffset(tex, s, direction, offset, component);
            vec4 offsetsGather = textureGatherOffsets(tex, s, direction, offset0, offset1, offset2, offset3, component);
            return offsetGather + offsetsGather;
        }

        vec4 gatherCubeArray(
            samplerCubeArray tex,
            sampler s,
            vec4 cubeLayer,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 offsetGather = textureGatherOffset(tex, s, cubeLayer, offset, component);
            vec4 offsetsGather = textureGatherOffsets(tex, s, cubeLayer, offset0, offset1, offset2, offset3, component);
            return offsetGather + offsetsGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCube(
                    cubeMap,
                    cubeSampler,
                    input.direction,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + gatherCubeArray(
                    cubeArray,
                    cubeSampler,
                    input.cubeLayer,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + textureGatherOffset(cubeMap, input.direction, input.offset, input.component)
                    + textureGatherOffset(cubeArray, input.cubeLayer, input.offset, input.component)
                    + textureGatherOffsets(cubeMap, input.direction, input.offset0, input.offset1, input.offset2, input.offset3, input.component)
                    + textureGatherOffsets(cubeArray, input.cubeLayer, input.offset0, input.offset1, input.offset2, input.offset3, input.component);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    offset_diagnostic = (
        "/* unsupported Metal texture gather: textureGatherOffset "
        "offsets require 2D or 2D-array textures */ float4(0.0)"
    )
    offsets_diagnostic = (
        "/* unsupported Metal texture gather: textureGatherOffsets "
        "offsets require 2D or 2D-array textures */ float4(0.0)"
    )
    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler cubeSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherCube(texturecube<float> tex, sampler s, float3 direction, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert (
        "float4 gatherCubeArray(texturecube_array<float> tex, sampler s, float4 cubeLayer, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert generated_code.count(offset_diagnostic) == 4
    assert generated_code.count(offsets_diagnostic) == 4
    assert ".gather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_metal_unsupported_dimension_texture_gather_emits_diagnostics():
    shader = """
    shader UnsupportedDimensionGather {
        sampler1D lineMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec3 volumeUv @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherLine(sampler1D tex, sampler s, float u, int component) {
            vec4 fixedGather = textureGather(tex, s, u, 2);
            vec4 dynamicGather = textureGather(tex, s, u, component);
            return fixedGather + dynamicGather;
        }

        vec4 gatherVolume(sampler3D tex, sampler s, vec3 volumeUv, int component) {
            vec4 fixedGather = textureGather(tex, s, volumeUv, 1);
            vec4 dynamicGather = textureGather(tex, s, volumeUv, component);
            return fixedGather + dynamicGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLine(lineMap, linearSampler, input.u, input.component)
                    + gatherVolume(volumeMap, linearSampler, input.volumeUv, input.component)
                    + textureGather(lineMap, input.u, input.component)
                    + textureGather(volumeMap, input.volumeUv, input.component);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ float4(0.0)"
    )
    assert "texture1d<float> lineMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherLine(texture1d<float> tex, sampler s, float u, int component)"
        in generated_code
    )
    assert (
        "float4 gatherVolume(texture3d<float> tex, sampler s, float3 volumeUv, int component)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 6
    assert ".gather(" not in generated_code
    assert "textureGather(" not in generated_code


def test_metal_texture_gather_offsets_mix_literal_and_dynamic_offsets():
    shader = """
    shader GatherOffsetMixed {
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            ivec2 dynamic0 @ TEXCOORD1;
            ivec2 dynamic1 @ TEXCOORD2;
            int component @ TEXCOORD3;
        };

        vec4 gatherOps(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 dynamic0,
            ivec2 dynamic1,
            int component
        ) {
            return textureGatherOffsets(
                layers,
                s,
                uvLayer,
                ivec2(-1, 0),
                dynamic0,
                ivec2(1, -1),
                dynamic1,
                component
            );
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherOps(
                    layerMap,
                    linearSampler,
                    input.uvLayer,
                    input.dynamic0,
                    input.dynamic1,
                    input.component
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherOps(texture2d_array<float> layers, sampler s, float3 uvLayer, int2 dynamic0, int2 dynamic1, int component)"
        in generated_code
    )
    assert "textureGatherOffsets(" not in generated_code
    components = {
        0: "component::x",
        1: "component::y",
        2: "component::z",
        3: "component::w",
    }
    for component, component_option in components.items():
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), int2(-1, 0), {component_option}).x"
            in generated_code
        )
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), dynamic0, {component_option}).y"
            in generated_code
        )
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), int2(1, -1), {component_option}).z"
            in generated_code
        )
        assert (
            f"layers.gather(s, uvLayer.xy, uint(uvLayer.z), dynamic1, {component_option}).w"
            in generated_code
        )
        if component < 3:
            assert f"component == {component} ? float4(" in generated_code
    assert (
        "gatherOps(layerMap, linearSampler, input.uvLayer, input.dynamic0, input.dynamic1, input.component)"
        in generated_code
    )


def test_metal_direct_stage_gather_offsets_use_input_members():
    shader = """
    shader DirectStageGatherOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 dynamic = textureGather(colorMap, linearSampler, input.uv, input.component);
                vec4 dynamicOffset = textureGatherOffset(colorMap, linearSampler, input.uv, input.offset, input.component);
                vec4 offsetsGather = textureGatherOffsets(
                    layerMap,
                    linearSampler,
                    input.uvLayer,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                );
                return dynamic + dynamicOffset + offsetsGather;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float> colorMap [[texture(0)]], texture2d_array<float> layerMap [[texture(1)]], sampler linearSampler [[sampler(0)]])"
        in generated_code
    )
    assert (
        "float4 dynamic = (input.component == 0 ? colorMap.gather(linearSampler, input.uv, int2(0), component::x) : "
        "input.component == 1 ? colorMap.gather(linearSampler, input.uv, int2(0), component::y) : "
        "input.component == 2 ? colorMap.gather(linearSampler, input.uv, int2(0), component::z) : "
        "colorMap.gather(linearSampler, input.uv, int2(0), component::w));"
        in generated_code
    )
    assert (
        "input.component == 0 ? colorMap.gather(linearSampler, input.uv, input.offset, component::x) : "
        "input.component == 1 ? colorMap.gather(linearSampler, input.uv, input.offset, component::y) : "
        "input.component == 2 ? colorMap.gather(linearSampler, input.uv, input.offset, component::z) : "
        "colorMap.gather(linearSampler, input.uv, input.offset, component::w)"
        in generated_code
    )
    for component_option in [
        "component::x",
        "component::y",
        "component::z",
        "component::w",
    ]:
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset0, {component_option}).x"
            in generated_code
        )
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset1, {component_option}).y"
            in generated_code
        )
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset2, {component_option}).z"
            in generated_code
        )
        assert (
            f"layerMap.gather(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset3, {component_option}).w"
            in generated_code
        )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_metal_texture_sample_offset_variants_use_sample_offsets():
    shader = """
    shader TextureSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 direction @ TEXCOORD2;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 offsetOps(
            sampler2D tex,
            sampler2DArray layers,
            sampler s,
            vec2 uv,
            vec3 uvLayer,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, uv, offset);
            vec4 lodSample = textureLodOffset(tex, s, uv, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, uv, ddx, ddy, offset);
            vec4 arrayPlain = textureOffset(layers, s, uvLayer, offset);
            vec4 arrayLod = textureLodOffset(layers, s, uvLayer, lod, offset);
            vec4 arrayGrad = textureGradOffset(layers, s, uvLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample + arrayPlain + arrayLod + arrayGrad;
        }

        vec4 unsupportedCubeOffset(
            samplerCube tex,
            sampler s,
            vec3 direction,
            ivec2 offset
        ) {
            return textureOffset(tex, s, direction, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return offsetOps(
                    colorMap,
                    layerMap,
                    linearSampler,
                    input.uv,
                    input.uvLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + unsupportedCubeOffset(
                    cubeMap,
                    linearSampler,
                    input.direction,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 offsetOps(texture2d<float> tex, texture2d_array<float> layers, sampler s, float2 uv, float3 uvLayer, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 plain = tex.sample(s, uv, offset);" in generated_code
    assert "float4 lodSample = tex.sample(s, uv, level(lod), offset);" in generated_code
    assert (
        "float4 gradSample = tex.sample(s, uv, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float4 arrayPlain = layers.sample(s, uvLayer.xy, uint(uvLayer.z), offset);"
        in generated_code
    )
    assert (
        "float4 arrayLod = layers.sample(s, uvLayer.xy, uint(uvLayer.z), level(lod), offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layers.sample(s, uvLayer.xy, uint(uvLayer.z), gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "return /* unsupported Metal texture offset: textureOffset offsets require 2D or 2D-array textures */ float4(0.0);"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_metal_cube_texture_sample_offsets_emit_diagnostics():
    shader = """
    shader CubeTextureSampleOffsetDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        samplerCubeShadow shadowMap;
        samplerCubeArrayShadow shadowArray;
        sampler linearSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 cubeOffsets(
            samplerCube tex,
            sampler s,
            vec3 direction,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, direction, offset);
            vec4 lodSample = textureLodOffset(tex, s, direction, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, direction, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        vec4 cubeArrayOffsets(
            samplerCubeArray tex,
            sampler s,
            vec4 cubeLayer,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, cubeLayer, offset);
            vec4 lodSample = textureLodOffset(tex, s, cubeLayer, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, cubeLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 sampled = cubeOffsets(
                    cubeMap,
                    linearSampler,
                    input.direction,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayOffsets(
                    cubeArray,
                    linearSampler,
                    input.cubeLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                )
                    + textureOffset(cubeMap, input.direction, input.offset)
                    + textureLodOffset(cubeMap, input.direction, input.lod, input.offset)
                    + textureGradOffset(cubeMap, input.direction, input.ddx, input.ddy, input.offset)
                    + textureOffset(cubeArray, input.cubeLayer, input.offset)
                    + textureLodOffset(cubeArray, input.cubeLayer, input.lod, input.offset)
                    + textureGradOffset(cubeArray, input.cubeLayer, input.ddx, input.ddy, input.offset);
                float compared = textureCompareOffset(shadowMap, input.direction, input.depth, input.offset)
                    + textureCompareLodOffset(shadowMap, input.direction, input.depth, input.lod, input.offset)
                    + textureCompareGradOffset(shadowMap, input.direction, input.depth, input.ddx, input.ddy, input.offset)
                    + textureCompareOffset(shadowArray, input.cubeLayer, input.depth, input.offset)
                    + textureCompareLodOffset(shadowArray, input.cubeLayer, input.depth, input.lod, input.offset)
                    + textureCompareGradOffset(shadowArray, input.cubeLayer, input.depth, input.ddx, input.ddy, input.offset);
                return sampled + vec4(compared);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "depthcube<float> shadowMap [[texture(2)]]" in generated_code
    assert "depthcube_array<float> shadowArray [[texture(3)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        generated_code.count(
            "/* unsupported Metal texture offset: textureOffset offsets require 2D or 2D-array textures */ float4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture offset: textureLodOffset offsets require 2D or 2D-array textures */ float4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture offset: textureGradOffset offsets require 2D or 2D-array textures */ float4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareLodOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareGradOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert ".sample(" not in generated_code
    assert ".sample_compare(" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_direct_stage_sample_offsets_and_texel_fetch_offset_use_input_members():
    shader = """
    shader DirectStageSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 pixel @ TEXCOORD2;
            ivec3 pixelLayer @ TEXCOORD3;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 plain = textureOffset(colorMap, linearSampler, input.uv, input.offset);
                vec4 lodSample = textureLodOffset(colorMap, linearSampler, input.uv, input.lod, input.offset);
                vec4 gradSample = textureGradOffset(colorMap, linearSampler, input.uv, input.ddx, input.ddy, input.offset);
                vec4 arrayPlain = textureOffset(layerMap, linearSampler, input.uvLayer, input.offset);
                vec4 arrayLod = textureLodOffset(layerMap, linearSampler, input.uvLayer, input.lod, input.offset);
                vec4 arrayGrad = textureGradOffset(layerMap, linearSampler, input.uvLayer, input.ddx, input.ddy, input.offset);
                vec4 fetched = texelFetchOffset(colorMap, input.pixel, int(input.lod), input.offset);
                vec4 fetchedLayer = texelFetchOffset(layerMap, input.pixelLayer, int(input.lod), input.offset);
                return plain + lodSample + gradSample + arrayPlain + arrayLod + arrayGrad + fetched + fetchedLayer;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "fragment float4 fragment_main(FSInput input [[stage_in]], texture2d<float> colorMap [[texture(0)]], texture2d_array<float> layerMap [[texture(1)]], sampler linearSampler [[sampler(0)]])"
        in generated_code
    )
    assert (
        "float4 plain = colorMap.sample(linearSampler, input.uv, input.offset);"
        in generated_code
    )
    assert (
        "float4 lodSample = colorMap.sample(linearSampler, input.uv, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        "float4 gradSample = colorMap.sample(linearSampler, input.uv, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayPlain = layerMap.sample(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayLod = layerMap.sample(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layerMap.sample(linearSampler, input.uvLayer.xy, uint(input.uvLayer.z), gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert (
        "float4 fetched = colorMap.read((input.pixel + input.offset), int(input.lod));"
        in generated_code
    )
    assert (
        "float4 fetchedLayer = layerMap.read((input.pixelLayer.xy + input.offset), uint(input.pixelLayer.z), int(input.lod));"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_metal_projected_texture_variants_use_sample_projection():
    shader = """
    shader TextureProjectionVariants {
        sampler2D colorMap;
        sampler3D volumeMap;
        samplerCube cubeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec3 direction @ TEXCOORD3;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        vec4 projectedOps(
            sampler2D tex,
            sampler3D volume,
            sampler s,
            vec3 uvq,
            vec4 uvqw,
            vec4 xyzq,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 projected = textureProj(tex, s, uvq);
            vec4 projectedBias = textureProj(tex, s, uvqw, 0.25);
            vec4 volumeProjected = textureProj(volume, s, xyzq);
            vec4 projectedOffset = textureProjOffset(tex, s, uvq, offset);
            vec4 projectedOffsetBias = textureProjOffset(tex, s, uvq, offset, 0.5);
            vec4 projectedLod = textureProjLod(tex, s, uvq, 2.0);
            vec4 projectedLodOffset = textureProjLodOffset(tex, s, uvq, 3.0, offset);
            vec4 projectedGrad = textureProjGrad(tex, s, uvq, ddx, ddy);
            vec4 projectedGradOffset = textureProjGradOffset(tex, s, uvq, ddx, ddy, offset);
            return projected
                + projectedBias
                + volumeProjected
                + projectedOffset
                + projectedOffsetBias
                + projectedLod
                + projectedLodOffset
                + projectedGrad
                + projectedGradOffset;
        }

        vec4 unsupportedCubeProjection(samplerCube tex, sampler s, vec3 direction) {
            return textureProj(tex, s, direction);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedOps(
                    colorMap,
                    volumeMap,
                    linearSampler,
                    input.uvq,
                    input.uvqw,
                    input.xyzq,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + unsupportedCubeProjection(
                    cubeMap,
                    linearSampler,
                    input.direction
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 projectedOps(texture2d<float> tex, texture3d<float> volume, sampler s, float3 uvq, float4 uvqw, float4 xyzq, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 projected = tex.sample(s, uvq.xy / uvq.z);" in generated_code
    assert (
        "float4 projectedBias = tex.sample(s, uvqw.xy / uvqw.w, bias(0.25));"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volume.sample(s, xyzq.xyz / xyzq.w);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = tex.sample(s, uvq.xy / uvq.z, offset);"
        in generated_code
    )
    assert (
        "float4 projectedOffsetBias = tex.sample(s, uvq.xy / uvq.z, bias(0.5), offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = tex.sample(s, uvq.xy / uvq.z, level(2.0));"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = tex.sample(s, uvq.xy / uvq.z, level(3.0), offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = tex.sample(s, uvq.xy / uvq.z, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = tex.sample(s, uvq.xy / uvq.z, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "return /* unsupported Metal projected texture: textureProj requires 1D, 2D, or 3D projection coordinates */ float4(0.0);"
        in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_metal_direct_projected_texture_stage_input_members():
    shader = """
    shader DirectProjectedTexture {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(colorMap, linearSampler, input.uvq);
                vec4 projectedBias = textureProj(colorMap, linearSampler, input.uvqw, 0.25);
                vec4 volumeProjected = textureProj(volumeMap, linearSampler, input.xyzq);
                vec4 projectedOffset = textureProjOffset(colorMap, linearSampler, input.uvq, input.offset);
                vec4 projectedLod = textureProjLod(colorMap, linearSampler, input.uvq, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(colorMap, linearSampler, input.uvq, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(colorMap, linearSampler, input.uvq, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(colorMap, linearSampler, input.uvq, input.ddx, input.ddy, input.offset);
                return projected
                    + projectedBias
                    + volumeProjected
                    + projectedOffset
                    + projectedLod
                    + projectedLodOffset
                    + projectedGrad
                    + projectedGradOffset;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 projected = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z);"
        in generated_code
    )
    assert (
        "float4 projectedBias = colorMap.sample(linearSampler, input.uvqw.xy / input.uvqw.w, bias(0.25));"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volumeMap.sample(linearSampler, input.xyzq.xyz / input.xyzq.w);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, input.offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, level(input.lod));"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = colorMap.sample(linearSampler, input.uvq.xy / input.uvq.z, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_metal_direct_projected_array_texture_stage_input_members():
    shader = """
    shader DirectProjectedArrayTexture {
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float lod;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(layerMap, linearSampler, input.uvLayerQ);
                vec4 projectedOffset = textureProjOffset(layerMap, linearSampler, input.uvLayerQ, input.offset);
                vec4 projectedLod = textureProjLod(layerMap, linearSampler, input.uvLayerQ, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(layerMap, linearSampler, input.uvLayerQ, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(layerMap, linearSampler, input.uvLayerQ, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(layerMap, linearSampler, input.uvLayerQ, input.ddx, input.ddy, input.offset);
                return projected + projectedOffset + projectedLod + projectedLodOffset + projectedGrad + projectedGradOffset;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_args = "input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z)"
    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        f"float4 projected = layerMap.sample(linearSampler, {projected_args});"
        in generated_code
    )
    assert (
        f"float4 projectedOffset = layerMap.sample(linearSampler, {projected_args}, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedLod = layerMap.sample(linearSampler, {projected_args}, level(input.lod));"
        in generated_code
    )
    assert (
        f"float4 projectedLodOffset = layerMap.sample(linearSampler, {projected_args}, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedGrad = layerMap.sample(linearSampler, {projected_args}, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        f"float4 projectedGradOffset = layerMap.sample(linearSampler, {projected_args}, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_metal_implicit_projected_array_texture_stage_input_members_use_default_sampler():
    shader = """
    shader DirectImplicitProjectedArrayTexture {
        sampler2DArray layerMap;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float lod;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(layerMap, input.uvLayerQ);
                vec4 projectedOffset = textureProjOffset(layerMap, input.uvLayerQ, input.offset);
                vec4 projectedLod = textureProjLod(layerMap, input.uvLayerQ, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(layerMap, input.uvLayerQ, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(layerMap, input.uvLayerQ, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(layerMap, input.uvLayerQ, input.ddx, input.ddy, input.offset);
                return projected + projectedOffset + projectedLod + projectedLodOffset + projectedGrad + projectedGradOffset;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    projected_args = "input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z)"
    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert (
        f"float4 projected = layerMap.sample({default_sampler}, {projected_args});"
        in generated_code
    )
    assert (
        f"float4 projectedOffset = layerMap.sample({default_sampler}, {projected_args}, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedLod = layerMap.sample({default_sampler}, {projected_args}, level(input.lod));"
        in generated_code
    )
    assert (
        f"float4 projectedLodOffset = layerMap.sample({default_sampler}, {projected_args}, level(input.lod), input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedGrad = layerMap.sample({default_sampler}, {projected_args}, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert (
        f"float4 projectedGradOffset = layerMap.sample({default_sampler}, {projected_args}, gradient2d(input.ddx, input.ddy), input.offset);"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code


def test_metal_projected_array_texture_resource_arrays_forward_samplers():
    shader = """
    shader ProjectedArrayTextureResourceArrays {
        sampler2DArray layerMaps[4];
        sampler linearSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 projectedLeaf(
            sampler2DArray maps[],
            sampler samplers[],
            int layer,
            vec4 uvLayerQ,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 fixedProj = textureProj(maps[2], samplers[2], uvLayerQ);
            vec4 dynamicOffset = textureProjOffset(maps[layer], samplers[layer], uvLayerQ, offset);
            vec4 fixedLod = textureProjLod(maps[1], samplers[1], uvLayerQ, lod);
            vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], samplers[layer], uvLayerQ, lod, offset);
            vec4 fixedGrad = textureProjGrad(maps[3], samplers[3], uvLayerQ, ddx, ddy);
            vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], samplers[layer], uvLayerQ, ddx, ddy, offset);
            return fixedProj + dynamicOffset + fixedLod + dynamicLodOffset + fixedGrad + dynamicGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedLeaf(
                    layerMaps,
                    linearSamplers,
                    input.layer,
                    input.uvLayerQ,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_args = "uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z)"
    assert "array<texture2d_array<float>, 4> layerMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> linearSamplers [[sampler(0)]]" in generated_code
    assert (
        "float4 projectedLeaf(array<texture2d_array<float>, 4> maps, array<sampler, 4> samplers, int layer, float4 uvLayerQ, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float4 fixedProj = maps[2].sample(samplers[2], {projected_args});"
        in generated_code
    )
    assert (
        f"float4 dynamicOffset = maps[layer].sample(samplers[layer], {projected_args}, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedLod = maps[1].sample(samplers[1], {projected_args}, level(lod));"
        in generated_code
    )
    assert (
        f"float4 dynamicLodOffset = maps[layer].sample(samplers[layer], {projected_args}, level(lod), offset);"
        in generated_code
    )
    assert (
        f"float4 fixedGrad = maps[3].sample(samplers[3], {projected_args}, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        f"float4 dynamicGradOffset = maps[layer].sample(samplers[layer], {projected_args}, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, linearSamplers, input.layer, input.uvLayerQ, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code


def test_metal_implicit_projected_array_texture_resource_arrays_use_default_sampler():
    shader = """
    shader ImplicitProjectedArrayTextureResourceArrays {
        sampler2DArray layerMaps[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 projectedLeaf(
            sampler2DArray maps[],
            int layer,
            vec4 uvLayerQ,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 fixedProj = textureProj(maps[2], uvLayerQ);
            vec4 dynamicOffset = textureProjOffset(maps[layer], uvLayerQ, offset);
            vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);
            vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], uvLayerQ, lod, offset);
            vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);
            vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], uvLayerQ, ddx, ddy, offset);
            return fixedProj + dynamicOffset + fixedLod + dynamicLodOffset + fixedGrad + dynamicGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedLeaf(
                    layerMaps,
                    input.layer,
                    input.uvLayerQ,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    projected_args = "uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z)"
    assert "array<texture2d_array<float>, 4> layerMaps [[texture(0)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert (
        "float4 projectedLeaf(array<texture2d_array<float>, 4> maps, int layer, float4 uvLayerQ, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float4 fixedProj = maps[2].sample({default_sampler}, {projected_args});"
        in generated_code
    )
    assert (
        f"float4 dynamicOffset = maps[layer].sample({default_sampler}, {projected_args}, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedLod = maps[1].sample({default_sampler}, {projected_args}, level(lod));"
        in generated_code
    )
    assert (
        f"float4 dynamicLodOffset = maps[layer].sample({default_sampler}, {projected_args}, level(lod), offset);"
        in generated_code
    )
    assert (
        f"float4 fixedGrad = maps[3].sample({default_sampler}, {projected_args}, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        f"float4 dynamicGradOffset = maps[layer].sample({default_sampler}, {projected_args}, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, input.layer, input.uvLayerQ, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "textureProj(" not in generated_code


def test_metal_implicit_projected_stage_input_members_use_default_sampler():
    shader = """
    shader DirectImplicitProjection {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec4 uvLayerQ @ TEXCOORD3;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = textureProj(colorMap, input.uvq);
                vec4 colorLod = textureProjLod(colorMap, input.uvqw, input.lod);
                vec4 volume = textureProj(volumeMap, input.xyzq);
                float shadow = textureCompareProj(shadowMap, input.uvq, input.depth);
                float shadowGrad = textureCompareProjGrad(shadowArray, input.uvLayerQ, input.depth, input.ddx, input.ddy);
                return color + colorLod + volume + vec4(shadow + shadowGrad);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture3d<float> volumeMap [[texture(1)]]" in generated_code
    assert "depth2d<float> shadowMap [[texture(2)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(3)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert (
        f"float4 color = colorMap.sample({default_sampler}, input.uvq.xy / input.uvq.z);"
        in generated_code
    )
    assert (
        f"float4 colorLod = colorMap.sample({default_sampler}, input.uvqw.xy / input.uvqw.w, level(input.lod));"
        in generated_code
    )
    assert (
        f"float4 volume = volumeMap.sample({default_sampler}, input.xyzq.xyz / input.xyzq.w);"
        in generated_code
    )
    assert (
        f"float shadow = shadowMap.sample_compare({default_sampler}, input.uvq.xy / input.uvq.z, input.depth);"
        in generated_code
    )
    assert (
        f"float shadowGrad = shadowArray.sample_compare({default_sampler}, input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z), input.depth, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert "unsupported Metal projected texture" not in generated_code
    assert "unsupported Metal texture compare" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_projected_shadow_compare_variants_use_sample_compare_projection():
    shader = """
    shader ProjectedShadowCompareVariants {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedShadow(
            sampler2DShadow tex,
            sampler s,
            vec3 uvq,
            vec4 uvqw,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvq, depth);
            float projectedW = textureCompareProj(tex, s, uvqw, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvq, depth, offset);
            float lodProjected = textureCompareProjLod(tex, s, uvq, depth, lod);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvq, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvq, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvq, depth, ddx, ddy, offset);
            return projected + projectedW + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        float projectedArrayShadow(
            sampler2DArrayShadow tex,
            sampler s,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvLayerQ, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvLayerQ, depth, offset);
            float lodProjected = textureCompareProjLod(tex, s, uvLayerQ, depth, lod);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvLayerQ, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvLayerQ, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvLayerQ, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(projectedShadow(
                    shadowMap,
                    compareSampler,
                    input.uvq,
                    input.uvqw,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    vec4(input.uvLayer, input.uvqw.w),
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float projectedShadow(depth2d<float> tex, sampler s, float3 uvq, float4 uvqw, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.sample_compare(s, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float projectedW = tex.sample_compare(s, uvqw.xy / uvqw.w, depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, level(lod));"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, level(lod), offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.sample_compare(s, uvq.xy / uvq.z, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float projectedArrayShadow(depth2d_array<float> tex, sampler s, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, level(lod));"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, level(lod), offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.sample_compare(s, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert "textureCompareProj(" not in generated_code


def test_metal_direct_projected_shadow_compare_stage_input_members():
    shader = """
    shader DirectProjectedShadowCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float planar = textureCompareProj(shadowMap, compareSampler, input.uvq, input.depth);
                float planarOffset = textureCompareProjOffset(shadowMap, compareSampler, input.uvqw, input.depth, input.offset);
                float arrayGrad = textureCompareProjGrad(shadowArray, compareSampler, input.uvLayerQ, input.depth, input.ddx, input.ddy);
                return vec4(planar + planarOffset + arrayGrad);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float planar = shadowMap.sample_compare(compareSampler, input.uvq.xy / input.uvq.z, input.depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMap.sample_compare(compareSampler, input.uvqw.xy / input.uvqw.w, input.depth, input.offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArray.sample_compare(compareSampler, input.uvLayerQ.xy / input.uvLayerQ.w, uint(input.uvLayerQ.z), input.depth, gradient2d(input.ddx, input.ddy));"
        in generated_code
    )
    assert "unsupported Metal texture compare" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_projected_shadow_compare_resource_arrays_forward_samplers():
    shader = """
    shader ProjectedShadowResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedLeaf(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planar = textureCompareProj(shadowMaps[layer], shadowSamplers[layer], uvq, depth);
            float planarOffset = textureCompareProjOffset(shadowMaps[1], shadowSamplers[1], uvq, depth, offset);
            float planarLod = textureCompareProjLod(shadowMaps[2], shadowSamplers[2], uvq, depth, lod);
            float planarGradOffset = textureCompareProjGradOffset(shadowMaps[layer], shadowSamplers[layer], uvq, depth, ddx, ddy, offset);
            float arrayProjected = textureCompareProj(shadowArrays[2], shadowSamplers[2], uvLayerQ, depth);
            float arrayOffset = textureCompareProjOffset(shadowArrays[layer], shadowSamplers[layer], uvLayerQ, depth, offset);
            float arrayGrad = textureCompareProjGrad(shadowArrays[1], shadowSamplers[1], uvLayerQ, depth, ddx, ddy);
            return planar + planarOffset + planarLod + planarGradOffset + arrayProjected + arrayOffset + arrayGrad;
        }

        float projectedWrapper(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return projectedLeaf(shadowMaps, shadowArrays, shadowSamplers, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return projectedWrapper(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.layer,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float projectedLeaf(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float planar = shadowMaps[layer].sample_compare(shadowSamplers[layer], uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMaps[1].sample_compare(shadowSamplers[1], uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[2].sample_compare(shadowSamplers[2], uvq.xy / uvq.z, depth, level(lod));"
        in generated_code
    )
    assert (
        "float planarGradOffset = shadowMaps[layer].sample_compare(shadowSamplers[layer], uvq.xy / uvq.z, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float arrayProjected = shadowArrays[2].sample_compare(shadowSamplers[2], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float arrayOffset = shadowArrays[layer].sample_compare(shadowSamplers[layer], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[1].sample_compare(shadowSamplers[1], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float projectedWrapper(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, shadowSamplers, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, shadowSamplers, input.layer, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj(" not in generated_code


def test_metal_implicit_projected_shadow_compare_resource_arrays_use_default_sampler():
    shader = """
    shader ImplicitProjectedShadowResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedLeaf(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planar = textureCompareProj(shadowMaps[layer], uvq, depth);
            float planarOffset = textureCompareProjOffset(shadowMaps[1], uvq, depth, offset);
            float planarLod = textureCompareProjLod(shadowMaps[2], uvq, depth, lod);
            float planarGradOffset = textureCompareProjGradOffset(shadowMaps[layer], uvq, depth, ddx, ddy, offset);
            float arrayProjected = textureCompareProj(shadowArrays[2], uvLayerQ, depth);
            float arrayOffset = textureCompareProjOffset(shadowArrays[layer], uvLayerQ, depth, offset);
            float arrayGrad = textureCompareProjGrad(shadowArrays[1], uvLayerQ, depth, ddx, ddy);
            return planar + planarOffset + planarLod + planarGradOffset + arrayProjected + arrayOffset + arrayGrad;
        }

        float projectedWrapper(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return projectedWrapper(
                    shadowMaps,
                    shadowArrays,
                    input.layer,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler" not in generated_code
    assert (
        "float projectedLeaf(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays"
        in generated_code
    )
    assert (
        f"float planar = shadowMaps[layer].sample_compare({default_sampler}, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        f"float planarOffset = shadowMaps[1].sample_compare({default_sampler}, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        f"float planarLod = shadowMaps[2].sample_compare({default_sampler}, uvq.xy / uvq.z, depth, level(lod));"
        in generated_code
    )
    assert (
        f"float planarGradOffset = shadowMaps[layer].sample_compare({default_sampler}, uvq.xy / uvq.z, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        f"float arrayProjected = shadowArrays[2].sample_compare({default_sampler}, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        f"float arrayOffset = shadowArrays[layer].sample_compare({default_sampler}, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        f"float arrayGrad = shadowArrays[1].sample_compare({default_sampler}, uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float projectedWrapper(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, input.layer, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj(" not in generated_code


def test_metal_unsized_projected_shadow_compare_arrays_infer_transitive_constant_size():
    shader = """
    shader UnsizedProjectedShadowResources {
        const int LAYER = 4;
        sampler2DShadow shadowMaps[];
        sampler2DArrayShadow shadowArrays[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float shadowDeep(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planarHigh = textureCompareProj(shadowMaps[LAYER], shadowSamplers[LAYER], uvq, depth);
            float planarLow = textureCompareProjOffset(shadowMaps[1], shadowSamplers[1], uvq, depth, offset);
            float arrayHigh = textureCompareProjGrad(shadowArrays[2 * 2], shadowSamplers[2 * 2], uvLayerQ, depth, ddx, ddy);
            float arrayLow = textureCompareProjOffset(shadowArrays[3], shadowSamplers[3], uvLayerQ, depth, offset);
            return planarHigh + planarLow + arrayHigh + arrayLow;
        }

        float shadowMid(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return shadowDeep(shadowMaps, shadowArrays, shadowSamplers, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float arrayShadow = shadowMid(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uvq.xy, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 5> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 5> shadowArrays [[texture(5)]]" in generated_code
    )
    assert "depth2d<float> afterShadow [[texture(10)]]" in generated_code
    assert "array<sampler, 5> shadowSamplers [[sampler(0)]]" in generated_code
    assert "sampler afterSampler [[sampler(5)]]" in generated_code
    assert (
        "float shadowDeep(array<depth2d<float>, 5> shadowMaps, array<depth2d_array<float>, 5> shadowArrays, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "float planarHigh = shadowMaps[LAYER].sample_compare(shadowSamplers[LAYER], uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarLow = shadowMaps[1].sample_compare(shadowSamplers[1], uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float arrayHigh = shadowArrays[2 * 2].sample_compare(shadowSamplers[2 * 2], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float arrayLow = shadowArrays[3].sample_compare(shadowSamplers[3], uvLayerQ.xy / uvLayerQ.w, uint(uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float shadowMid(array<depth2d<float>, 5> shadowMaps, array<depth2d_array<float>, 5> shadowArrays, array<sampler, 5> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowDeep(shadowMaps, shadowArrays, shadowSamplers, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "shadowMid(shadowMaps, shadowArrays, shadowSamplers, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "float singleShadow = afterShadow.sample_compare(afterSampler, input.uvq.xy, input.depth);"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompareProj(" not in generated_code


def test_metal_projected_cube_shadow_compare_reports_unsupported():
    shader = """
    shader ProjectedCubeShadowDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;
        sampler compareSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeLayerProj @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float cubeProjected(
            samplerCubeShadow tex,
            sampler s,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, cubeProj, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, cubeProj, depth, offset);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, cubeProj, depth, lod, offset);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, cubeProj, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodOffsetProjected + gradOffsetProjected;
        }

        float cubeArrayProjected(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayerProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, cubeLayerProj, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, cubeLayerProj, depth, offset);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, cubeLayerProj, depth, lod, offset);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, cubeLayerProj, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodOffsetProjected + gradOffsetProjected;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeProjected(
                    cubeMap,
                    compareSampler,
                    input.cubeProj,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayProjected(
                    cubeArray,
                    compareSampler,
                    input.cubeLayerProj,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depthcube<float> cubeMap [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float cubeProjected(depthcube<float> tex, sampler s, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayProjected(depthcube_array<float> tex, sampler s, float4 cubeLayerProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    reasons = {
        "textureCompareProj",
        "textureCompareProjOffset",
        "textureCompareProjLodOffset",
        "textureCompareProjGradOffset",
    }
    for func_name in reasons:
        assert (
            generated_code.count(
                f"/* unsupported Metal texture compare: {func_name} requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates */ 0.0"
            )
            == 2
        )
    assert "textureCompareProj(" not in generated_code
    assert "textureCompareProjOffset(" not in generated_code
    assert "textureCompareProjLodOffset(" not in generated_code
    assert "textureCompareProjGradOffset(" not in generated_code
    assert ".sample_compare(s, cubeProj" not in generated_code
    assert ".sample_compare(s, cubeLayerProj" not in generated_code


def test_metal_direct_projected_cube_texture_reports_unsupported():
    shader = """
    shader DirectProjectedCubeDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeArrayProj @ TEXCOORD1;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 cubeProjected = textureProj(cubeMap, linearSampler, input.cubeProj);
                vec4 cubeLodOffset = textureProjLodOffset(cubeMap, linearSampler, input.cubeProj, input.lod, input.offset);
                vec4 cubeGradOffset = textureProjGradOffset(cubeMap, linearSampler, input.cubeProj, input.ddx, input.ddy, input.offset);
                vec4 cubeArrayProjected = textureProj(cubeArray, linearSampler, input.cubeArrayProj);
                vec4 cubeArrayLodOffset = textureProjLodOffset(cubeArray, linearSampler, input.cubeArrayProj, input.lod, input.offset);
                vec4 cubeArrayGradOffset = textureProjGradOffset(cubeArray, linearSampler, input.cubeArrayProj, input.ddx, input.ddy, input.offset);
                vec4 implicitCube = textureProj(cubeMap, input.cubeProj);
                vec4 implicitCubeArray = textureProjLod(cubeArray, input.cubeArrayProj, input.lod);
                return cubeProjected + cubeLodOffset + cubeGradOffset + cubeArrayProjected + cubeArrayLodOffset + cubeArrayGradOffset + implicitCube + implicitCubeArray;
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    expected_counts = {
        "textureProj": 3,
        "textureProjLodOffset": 2,
        "textureProjGradOffset": 2,
        "textureProjLod": 1,
    }
    for func_name, count in expected_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported Metal projected texture: {func_name} requires 1D, 2D, or 3D projection coordinates */ float4(0.0)"
            )
            == count
        )
    assert "textureProj(cubeMap" not in generated_code
    assert "textureProj(cubeArray" not in generated_code
    assert ".sample(linearSampler, input.cubeProj" not in generated_code
    assert ".sample(linearSampler, input.cubeArrayProj" not in generated_code


def test_metal_projected_cube_texture_resource_arrays_report_unsupported_without_samplers():
    shader = """
    shader ProjectedCubeArrayDiagnostics {
        samplerCube cubeMaps[4];
        samplerCubeArray cubeArrays[4];
        samplerCubeShadow shadowCubes[4];
        samplerCubeArrayShadow shadowCubeArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 cubeProj @ TEXCOORD1;
            vec4 cubeArrayProj @ TEXCOORD2;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 projectedCube(
            samplerCube maps[],
            int layer,
            vec4 cubeProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 fixedProjected = textureProj(maps[2], cubeProj);
            vec4 dynamicLodProjected = textureProjLod(maps[layer], cubeProj, lod);
            vec4 dynamicGradOffsetProjected = textureProjGradOffset(maps[layer], cubeProj, ddx, ddy, offset);
            return fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected;
        }

        vec4 projectedCubeArray(
            samplerCubeArray maps[],
            int layer,
            vec4 cubeArrayProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 fixedProjected = textureProj(maps[2], cubeArrayProj);
            vec4 dynamicLodProjected = textureProjLod(maps[layer], cubeArrayProj, lod);
            vec4 dynamicGradOffsetProjected = textureProjGradOffset(maps[layer], cubeArrayProj, ddx, ddy, offset);
            return fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected;
        }

        vec4 projectedShadow(
            samplerCubeShadow maps[],
            int layer,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float fixedProjected = textureCompareProj(maps[2], cubeProj, depth);
            float dynamicLodProjected = textureCompareProjLod(maps[layer], cubeProj, depth, lod);
            float dynamicGradOffsetProjected = textureCompareProjGradOffset(maps[layer], cubeProj, depth, ddx, ddy, offset);
            return vec4(fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected);
        }

        vec4 projectedShadowArray(
            samplerCubeArrayShadow maps[],
            int layer,
            vec4 cubeArrayProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float fixedProjected = textureCompareProj(maps[2], cubeArrayProj, depth);
            float dynamicLodProjected = textureCompareProjLod(maps[layer], cubeArrayProj, depth, lod);
            float dynamicGradOffsetProjected = textureCompareProjGradOffset(maps[layer], cubeArrayProj, depth, ddx, ddy, offset);
            return vec4(fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 globalProjected = textureProj(cubeMaps[input.layer], input.cubeProj);
                vec4 globalArrayProjected = textureProjLod(cubeArrays[2], input.cubeArrayProj, input.lod);
                float globalShadow = textureCompareProj(shadowCubes[input.layer], input.cubeProj, input.depth);
                float globalArrayShadow = textureCompareProjLod(shadowCubeArrays[2], input.cubeArrayProj, input.depth, input.lod);
                return globalProjected + globalArrayProjected + vec4(globalShadow + globalArrayShadow)
                    + projectedCube(
                        cubeMaps,
                        input.layer,
                        input.cubeProj,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + projectedCubeArray(
                        cubeArrays,
                        input.layer,
                        input.cubeArrayProj,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + projectedShadow(
                        shadowCubes,
                        input.layer,
                        input.cubeProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + projectedShadowArray(
                        shadowCubeArrays,
                        input.layer,
                        input.cubeArrayProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "array<texturecube<float>, 4> cubeMaps [[texture(0)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(4)]]" in generated_code
    )
    assert "array<depthcube<float>, 4> shadowCubes [[texture(8)]]" in generated_code
    assert (
        "array<depthcube_array<float>, 4> shadowCubeArrays [[texture(12)]]"
        in generated_code
    )
    assert (
        "float4 projectedCube(array<texturecube<float>, 4> maps, int layer, float4 cubeProj, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 projectedCubeArray(array<texturecube_array<float>, 4> maps, int layer, float4 cubeArrayProj, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 projectedShadow(array<depthcube<float>, 4> maps, int layer, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 projectedShadowArray(array<depthcube_array<float>, 4> maps, int layer, float4 cubeArrayProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    projected_counts = {
        "textureProj": 3,
        "textureProjLod": 3,
        "textureProjGradOffset": 2,
    }
    for func_name, count in projected_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported Metal projected texture: {func_name} requires 1D, 2D, or 3D projection coordinates */ float4(0.0)"
            )
            == count
        )
    compare_counts = {
        "textureCompareProj": 3,
        "textureCompareProjLod": 3,
        "textureCompareProjGradOffset": 2,
    }
    for func_name, count in compare_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported Metal texture compare: {func_name} requires depth2d vec3/vec4 or depth2d_array vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert (
        "projectedCube(cubeMaps, input.layer, input.cubeProj, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedCubeArray(cubeArrays, input.layer, input.cubeArrayProj, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedShadow(shadowCubes, input.layer, input.cubeProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedShadowArray(shadowCubeArrays, input.layer, input.cubeArrayProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "[[sampler(" not in generated_code
    assert "sampler(" not in generated_code
    assert ".sample(" not in generated_code
    assert ".sample_compare(" not in generated_code


def test_metal_shadow_gather_compare_offsets_use_depth_overloads():
    shader = """
    shader ShadowGatherCompareOffsets {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        vec4 gatherShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, s, uv, depth);
            vec4 offsetGathered = textureGatherCompareOffset(tex, s, uv, depth, offset);
            float offsetCompared = textureCompareOffset(tex, s, uv, depth, offset);
            return gathered + offsetGathered + vec4(offsetCompared);
        }

        vec4 gatherShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, s, uvLayer, depth);
            vec4 offsetGathered = textureGatherCompareOffset(tex, s, uvLayer, depth, offset);
            float offsetCompared = textureCompareOffset(tex, s, uvLayer, depth, offset);
            return gathered + offsetGathered + vec4(offsetCompared);
        }

        vec4 gatherCubeShadowArray(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, s, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)
                    + gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset)
                    + gatherCubeShadowArray(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherShadow(depth2d<float> tex, sampler s, float2 uv, float depth, int2 offset)"
        in generated_code
    )
    assert "float4 gathered = tex.gather_compare(s, uv, depth);" in generated_code
    assert (
        "float4 offsetGathered = tex.gather_compare(s, uv, depth, offset);"
        in generated_code
    )
    assert (
        "float offsetCompared = tex.sample_compare(s, uv, depth, offset);"
        in generated_code
    )
    assert (
        "float4 gatherShadowArray(depth2d_array<float> tex, sampler s, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert "tex.gather_compare(s, uvLayer.xy, uint(uvLayer.z), depth)" in generated_code
    assert (
        "tex.gather_compare(s, uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "float4 gatherCubeShadowArray(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "return tex.gather_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_metal_cube_shadow_gather_compare_supports_cube_and_cube_array():
    shader = """
    shader CubeShadowGatherCompare {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
        };

        vec4 gatherCubeShadow(samplerCubeShadow tex, sampler s, vec3 direction, float depth) {
            return textureGatherCompare(tex, s, direction, depth);
        }

        vec4 gatherCubeArrayShadow(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, s, cubeLayer, depth);
        }

        vec4 implicitCubeShadow(samplerCubeShadow tex, vec3 direction, float depth) {
            return textureGatherCompare(tex, direction, depth);
        }

        vec4 implicitCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCubeShadow(cubeShadow, compareSampler, input.direction, input.depth)
                    + gatherCubeArrayShadow(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)
                    + implicitCubeShadow(cubeShadow, input.direction, input.depth)
                    + implicitCubeArrayShadow(cubeShadowArray, input.cubeLayer, input.depth)
                    + textureGatherCompare(cubeShadow, compareSampler, input.direction, input.depth)
                    + textureGatherCompare(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)
                    + textureGatherCompare(cubeShadow, input.direction, input.depth)
                    + textureGatherCompare(cubeShadowArray, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "depthcube<float> cubeShadow [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float4 gatherCubeShadow(depthcube<float> tex, sampler s, float3 direction, float depth)"
        in generated_code
    )
    assert "return tex.gather_compare(s, direction, depth);" in generated_code
    assert (
        "float4 gatherCubeArrayShadow(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "return tex.gather_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "float4 implicitCubeShadow(depthcube<float> tex, float3 direction, float depth)"
        in generated_code
    )
    assert (
        f"return tex.gather_compare({default_sampler}, direction, depth);"
        in generated_code
    )
    assert (
        "float4 implicitCubeArrayShadow(depthcube_array<float> tex, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        f"return tex.gather_compare({default_sampler}, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "gatherCubeShadow(cubeShadow, compareSampler, input.direction, input.depth)"
        in generated_code
    )
    assert (
        "cubeShadowArray.gather_compare(compareSampler, input.cubeLayer.xyz, uint(input.cubeLayer.w), input.depth)"
        in generated_code
    )
    assert (
        f"cubeShadow.gather_compare({default_sampler}, input.direction, input.depth)"
        in generated_code
    )
    assert "unsupported Metal texture gather compare" not in generated_code
    assert "textureGatherCompare(" not in generated_code


def test_metal_implicit_shadow_gather_compare_offsets_cover_arrays_and_cube_arrays():
    shader = """
    shader ImplicitShadowGatherCompare {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 implicitArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(tex, uvLayer, depth)
                + textureGatherCompareOffset(tex, uvLayer, depth, offset)
                + vec4(textureCompareOffset(tex, uvLayer, depth, offset));
        }

        vec4 implicitCubeArray(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitArray(shadowArray, input.uvLayer, input.depth, input.offset)
                    + implicitCubeArray(cubeShadowArray, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "float4 implicitArray(depth2d_array<float> tex, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert (
        "tex.gather_compare(sampler(mag_filter::linear, min_filter::linear), uvLayer.xy, uint(uvLayer.z), depth)"
        in generated_code
    )
    assert (
        "tex.gather_compare(sampler(mag_filter::linear, min_filter::linear), uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(sampler(mag_filter::linear, min_filter::linear), uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in generated_code
    )
    assert (
        "float4 implicitCubeArray(depthcube_array<float> tex, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "return tex.gather_compare(sampler(mag_filter::linear, min_filter::linear), cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "implicitArray(shadowArray, input.uvLayer, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "implicitCubeArray(cubeShadowArray, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_metal_direct_shadow_gather_compare_stage_input_members():
    explicit_shader = """
    shader DirectShadowGatherCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 planar = textureGatherCompare(shadowMap, compareSampler, input.uv, input.depth);
                vec4 planarOffset = textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset);
                vec4 arrayGather = textureGatherCompare(shadowArray, compareSampler, input.uvLayer, input.depth);
                vec4 arrayOffset = textureGatherCompareOffset(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset);
                vec4 cubeArrayGather = textureGatherCompare(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
                return planar + planarOffset + arrayGather + arrayOffset + cubeArrayGather;
            }
        }
    }
    """
    implicit_shader = """
    shader DirectImplicitShadowGatherCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 planar = textureGatherCompare(shadowMap, input.uv, input.depth);
                vec4 planarOffset = textureGatherCompareOffset(shadowMap, input.uv, input.depth, input.offset);
                vec4 arrayGather = textureGatherCompare(shadowArray, input.uvLayer, input.depth);
                vec4 arrayOffset = textureGatherCompareOffset(shadowArray, input.uvLayer, input.depth, input.offset);
                vec4 cubeArrayGather = textureGatherCompare(cubeShadowArray, input.cubeLayer, input.depth);
                return planar + planarOffset + arrayGather + arrayOffset + cubeArrayGather;
            }
        }
    }
    """

    explicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "depth2d<float> shadowMap [[texture(0)]]" in explicit_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in explicit_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in explicit_code
    assert "sampler compareSampler [[sampler(0)]]" in explicit_code
    assert (
        "float4 planar = shadowMap.gather_compare(compareSampler, input.uv, input.depth);"
        in explicit_code
    )
    assert (
        "float4 planarOffset = shadowMap.gather_compare(compareSampler, input.uv, input.depth, input.offset);"
        in explicit_code
    )
    assert (
        "float4 arrayGather = shadowArray.gather_compare(compareSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.depth);"
        in explicit_code
    )
    assert (
        "float4 arrayOffset = shadowArray.gather_compare(compareSampler, input.uvLayer.xy, uint(input.uvLayer.z), input.depth, input.offset);"
        in explicit_code
    )
    assert (
        "float4 cubeArrayGather = cubeShadowArray.gather_compare(compareSampler, input.cubeLayer.xyz, uint(input.cubeLayer.w), input.depth);"
        in explicit_code
    )
    assert "unsupported Metal texture gather compare" not in explicit_code
    assert "textureGatherCompare(" not in explicit_code

    assert "depth2d<float> shadowMap [[texture(0)]]" in implicit_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in implicit_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in implicit_code
    assert "[[sampler(" not in implicit_code
    assert (
        f"float4 planar = shadowMap.gather_compare({default_sampler}, input.uv, input.depth);"
        in implicit_code
    )
    assert (
        f"float4 planarOffset = shadowMap.gather_compare({default_sampler}, input.uv, input.depth, input.offset);"
        in implicit_code
    )
    assert (
        f"float4 arrayGather = shadowArray.gather_compare({default_sampler}, input.uvLayer.xy, uint(input.uvLayer.z), input.depth);"
        in implicit_code
    )
    assert (
        f"float4 arrayOffset = shadowArray.gather_compare({default_sampler}, input.uvLayer.xy, uint(input.uvLayer.z), input.depth, input.offset);"
        in implicit_code
    )
    assert (
        f"float4 cubeArrayGather = cubeShadowArray.gather_compare({default_sampler}, input.cubeLayer.xyz, uint(input.cubeLayer.w), input.depth);"
        in implicit_code
    )
    assert "unsupported Metal texture gather compare" not in implicit_code
    assert "textureGatherCompare(" not in implicit_code


def test_metal_shadow_gather_compare_resource_arrays_forward_samplers():
    explicit_shader = """
    shader ShadowGatherCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeArrays[4];
        sampler compareSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 gatherLayer(
            sampler2DShadow maps[],
            sampler2DArrayShadow arrays[],
            samplerCubeArrayShadow cubes[],
            sampler samplers[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            vec4 fixedPlanar = textureGatherCompare(maps[2], samplers[2], uv, depth);
            vec4 dynamicPlanarOffset = textureGatherCompareOffset(maps[layer], samplers[layer], uv, depth, offset);
            vec4 fixedArray = textureGatherCompare(arrays[1], samplers[1], uvLayer, depth);
            vec4 dynamicArrayOffset = textureGatherCompareOffset(arrays[layer], samplers[layer], uvLayer, depth, offset);
            vec4 fixedCubeArray = textureGatherCompare(cubes[3], samplers[3], cubeLayer, depth);
            vec4 dynamicCubeArray = textureGatherCompare(cubes[layer], samplers[layer], cubeLayer, depth);
            return fixedPlanar + dynamicPlanarOffset + fixedArray + dynamicArrayOffset + fixedCubeArray + dynamicCubeArray;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLayer(
                    shadowMaps,
                    shadowArrays,
                    cubeArrays,
                    compareSamplers,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
            }
        }
    }
    """
    implicit_shader = """
    shader ImplicitShadowGatherCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 gatherLayer(
            sampler2DShadow maps[],
            sampler2DArrayShadow arrays[],
            samplerCubeArrayShadow cubes[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            vec4 fixedPlanar = textureGatherCompare(maps[2], uv, depth);
            vec4 dynamicPlanarOffset = textureGatherCompareOffset(maps[layer], uv, depth, offset);
            vec4 fixedArray = textureGatherCompare(arrays[1], uvLayer, depth);
            vec4 dynamicArrayOffset = textureGatherCompareOffset(arrays[layer], uvLayer, depth, offset);
            vec4 fixedCubeArray = textureGatherCompare(cubes[3], cubeLayer, depth);
            vec4 dynamicCubeArray = textureGatherCompare(cubes[layer], cubeLayer, depth);
            return fixedPlanar + dynamicPlanarOffset + fixedArray + dynamicArrayOffset + fixedCubeArray + dynamicCubeArray;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLayer(
                    shadowMaps,
                    shadowArrays,
                    cubeArrays,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
            }
        }
    }
    """

    explicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in explicit_code
    assert "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in explicit_code
    assert "array<depthcube_array<float>, 4> cubeArrays [[texture(8)]]" in explicit_code
    assert "array<sampler, 4> compareSamplers [[sampler(0)]]" in explicit_code
    assert (
        "float4 gatherLayer(array<depth2d<float>, 4> maps, array<depth2d_array<float>, 4> arrays, array<depthcube_array<float>, 4> cubes, array<sampler, 4> samplers, int layer, float2 uv, float3 uvLayer, float4 cubeLayer, float depth, int2 offset)"
        in explicit_code
    )
    assert "maps[2].gather_compare(samplers[2], uv, depth)" in explicit_code
    assert (
        "maps[layer].gather_compare(samplers[layer], uv, depth, offset)"
        in explicit_code
    )
    assert (
        "arrays[1].gather_compare(samplers[1], uvLayer.xy, uint(uvLayer.z), depth)"
        in explicit_code
    )
    assert (
        "arrays[layer].gather_compare(samplers[layer], uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in explicit_code
    )
    assert (
        "cubes[3].gather_compare(samplers[3], cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in explicit_code
    )
    assert (
        "cubes[layer].gather_compare(samplers[layer], cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in explicit_code
    )
    assert (
        "gatherLayer(shadowMaps, shadowArrays, cubeArrays, compareSamplers, input.layer, input.uv, input.uvLayer, input.cubeLayer, input.depth, input.offset)"
        in explicit_code
    )
    assert "textureGatherCompare" not in explicit_code

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in implicit_code
    assert "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in implicit_code
    assert "array<depthcube_array<float>, 4> cubeArrays [[texture(8)]]" in implicit_code
    assert "[[sampler(" not in implicit_code
    assert (
        "float4 gatherLayer(array<depth2d<float>, 4> maps, array<depth2d_array<float>, 4> arrays, array<depthcube_array<float>, 4> cubes, int layer, float2 uv, float3 uvLayer, float4 cubeLayer, float depth, int2 offset)"
        in implicit_code
    )
    assert f"maps[2].gather_compare({default_sampler}, uv, depth)" in implicit_code
    assert (
        f"maps[layer].gather_compare({default_sampler}, uv, depth, offset)"
        in implicit_code
    )
    assert (
        f"arrays[1].gather_compare({default_sampler}, uvLayer.xy, uint(uvLayer.z), depth)"
        in implicit_code
    )
    assert (
        f"arrays[layer].gather_compare({default_sampler}, uvLayer.xy, uint(uvLayer.z), depth, offset)"
        in implicit_code
    )
    assert (
        f"cubes[3].gather_compare({default_sampler}, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in implicit_code
    )
    assert (
        f"cubes[layer].gather_compare({default_sampler}, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in implicit_code
    )
    assert (
        "gatherLayer(shadowMaps, shadowArrays, cubeArrays, input.layer, input.uv, input.uvLayer, input.cubeLayer, input.depth, input.offset)"
        in implicit_code
    )
    assert "textureGatherCompare" not in implicit_code


def test_metal_unsupported_cube_shadow_gather_compare_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedCubeGatherCompareOffset {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 cubeOffset(
            samplerCubeShadow tex,
            vec3 direction,
            float depth,
            ivec2 offset
        ) {
            return textureGatherCompareOffset(tex, direction, depth, offset);
        }

        vec4 cubeArrayOffset(
            samplerCubeArrayShadow tex,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            return textureGatherCompareOffset(tex, cubeLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return cubeOffset(cubeMap, input.direction, input.depth, input.offset)
                    + cubeArrayOffset(cubeArray, input.cubeLayer, input.depth, input.offset)
                    + textureGatherCompareOffset(cubeArray, input.cubeLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array depth textures */ "
        "float4(0.0)"
    )
    assert "depthcube<float> cubeMap [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "[[sampler(" not in generated_code
    assert "sampler(" not in generated_code
    assert (
        "float4 cubeOffset(depthcube<float> tex, float3 direction, float depth, int2 offset)"
        in generated_code
    )
    assert (
        "float4 cubeArrayOffset(depthcube_array<float> tex, float4 cubeLayer, float depth, int2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 3
    assert (
        "cubeOffset(cubeMap, input.direction, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "cubeArrayOffset(cubeArray, input.cubeLayer, input.depth, input.offset)"
        in generated_code
    )
    assert ".gather_compare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_metal_shadow_compare_lod_grad_use_depth_overloads():
    shader = """
    shader ShadowCompareLodGrad {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            vec3 cubeDdx @ TEXCOORD5;
            vec3 cubeDdy @ TEXCOORD6;
            ivec2 offset @ TEXCOORD7;
        };

        float compareShadow(
            sampler2DShadow tex,
            sampler s,
            vec2 uv,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, s, uv, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, s, uv, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, uv, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uv, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, s, uvLayer, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, s, uvLayer, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareCubeShadowArray(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float lodValue = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float gradValue = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            return lodValue + gradValue;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = compareShadow(
                    shadowMap,
                    compareSampler,
                    input.uv,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayShadow = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float cubeArrayShadow = compareCubeShadowArray(
                    cubeShadowArray,
                    compareSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.cubeDdx,
                    input.cubeDdy
                );
                return vec4(shadow + arrayShadow + cubeArrayShadow);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(1)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(2)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float compareShadow(depth2d<float> tex, sampler s, float2 uv, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float lodValue = tex.sample_compare(s, uv, depth, level(lod));"
        in generated_code
    )
    assert (
        "float lodOffsetValue = tex.sample_compare(s, uv, depth, level(lod), offset);"
        in generated_code
    )
    assert (
        "float gradValue = tex.sample_compare(s, uv, depth, gradient2d(ddx, ddy));"
        in generated_code
    )
    assert (
        "float gradOffsetValue = tex.sample_compare(s, uv, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float compareShadowArray(depth2d_array<float> tex, sampler s, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, level(lod))"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, level(lod), offset)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, gradient2d(ddx, ddy))"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, uvLayer.xy, uint(uvLayer.z), depth, gradient2d(ddx, ddy), offset)"
        in generated_code
    )
    assert (
        "float compareCubeShadowArray(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, level(lod))"
        in generated_code
    )
    assert (
        "tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, gradientcube(ddx, ddy))"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_cube_shadow_compare_offsets_report_unsupported():
    shader = """
    shader CubeShadowCompareOffsetDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;
        sampler compareSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float cubeOffsets(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float compareOffset = textureCompareOffset(tex, s, direction, depth, offset);
            float lodOffset = textureCompareLodOffset(tex, s, direction, depth, lod, offset);
            float gradOffset = textureCompareGradOffset(tex, s, direction, depth, ddx, ddy, offset);
            return compareOffset + lodOffset + gradOffset;
        }

        float cubeArrayOffsets(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float compareOffset = textureCompareOffset(tex, s, cubeLayer, depth, offset);
            float lodOffset = textureCompareLodOffset(tex, s, cubeLayer, depth, lod, offset);
            float gradOffset = textureCompareGradOffset(tex, s, cubeLayer, depth, ddx, ddy, offset);
            return compareOffset + lodOffset + gradOffset;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeOffsets(
                    cubeMap,
                    compareSampler,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayOffsets(
                    cubeArray,
                    compareSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depthcube<float> cubeMap [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert (
        "float cubeOffsets(depthcube<float> tex, sampler s, float3 direction, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayOffsets(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareLodOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture compare: textureCompareGradOffset offsets require 2D or 2D-array depth textures */ 0.0"
        )
        == 2
    )
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert ".sample_compare(s, direction, depth, offset)" not in generated_code
    assert (
        ".sample_compare(s, direction, depth, level(lod), offset)" not in generated_code
    )
    assert (
        ".sample_compare(s, direction, depth, gradientcube(ddx, ddy), offset)"
        not in generated_code
    )


def test_metal_nested_implicit_shadow_compare_lod_grad_uses_depth_overloads():
    shader = """
    shader NestedShadowCompareQuery {
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        float shadowOps(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec2 lod = textureQueryLod(tex, uv);
            float cmp = textureCompareLod(tex, uv, depth, lod.x);
            float grad = textureCompareGradOffset(tex, uv, depth, ddx, ddy, offset);
            return cmp + grad + lod.y;
        }

        float wrappedShadow(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return shadowOps(tex, uv, depth, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(wrappedShadow(
                    shadowMap,
                    input.uv,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                ));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    assert "depth2d<float> shadowMap [[texture(0)]]" in generated_code
    assert (
        "float shadowOps(depth2d<float> tex, float2 uv, float depth, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float2 lod = float2(tex.calculate_unclamped_lod({default_sampler}, uv), tex.calculate_clamped_lod({default_sampler}, uv));"
        in generated_code
    )
    assert (
        f"float cmp = tex.sample_compare({default_sampler}, uv, depth, level(lod.x));"
        in generated_code
    )
    assert (
        f"float grad = tex.sample_compare({default_sampler}, uv, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float wrappedShadow(depth2d<float> tex, float2 uv, float depth, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "return shadowOps(tex, uv, depth, ddx, ddy, offset);" in generated_code
    assert (
        "wrappedShadow(shadowMap, input.uv, input.depth, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_array_shadow_texture_resource_arrays_keep_compare_coordinates():
    shader = """
    shader ArrayShadowTextureResourceArrays {
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
        };

        float sampleArrayLayer(sampler2DArrayShadow shadowArrays[], sampler shadowSamplers[], vec3 uvLayer, float depth) {
            return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
        }

        float sampleCubeLayer(samplerCubeArrayShadow cubeShadowArrays[], sampler shadowSamplers[], vec4 cubeLayer, float depth) {
            return textureCompare(cubeShadowArrays[3], shadowSamplers[3], cubeLayer, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth) + sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(0)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(4)]]"
        in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float sampleArrayLayer(array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowArrays[2].sample_compare(shadowSamplers[2], uvLayer.xy, uint(uvLayer.z), depth)"
        in generated_code
    )
    assert (
        "float sampleCubeLayer(array<depthcube_array<float>, 4> cubeShadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].sample_compare(shadowSamplers[3], cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in generated_code
    )
    assert (
        "sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth)"
        in generated_code
    )
    assert (
        "sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "array<depth2d_array<float>, 1> shadowArrays" not in generated_code
    assert "textureCompare(" not in generated_code
    assert ".sample_compare(shadowSamplers[2], uvLayer, depth)" not in generated_code
    assert ".sample_compare(shadowSamplers[3], cubeLayer, depth)" not in generated_code


def test_metal_array_shadow_compare_lod_grad_resource_arrays():
    shader = """
    shader ShadowCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float shadowLayer(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planarLod = textureCompareLod(shadowMaps[layer], shadowSamplers[layer], uv, depth, lod);
            float planarGrad = textureCompareGradOffset(shadowMaps[1], shadowSamplers[1], uv, depth, ddx, ddy, offset);
            float arrayLod = textureCompareLod(shadowArrays[2], shadowSamplers[2], uvLayer, depth, lod);
            float arrayGrad = textureCompareGradOffset(shadowArrays[layer], shadowSamplers[layer], uvLayer, depth, ddx, ddy, offset);
            return planarLod + planarGrad + arrayLod + arrayGrad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowLayer(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float shadowLayer(array<depth2d<float>, 4> shadowMaps, array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[layer].sample_compare(shadowSamplers[layer], uv, depth, level(lod));"
        in generated_code
    )
    assert (
        "float planarGrad = shadowMaps[1].sample_compare(shadowSamplers[1], uv, depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "float arrayLod = shadowArrays[2].sample_compare(shadowSamplers[2], uvLayer.xy, uint(uvLayer.z), depth, level(lod));"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[layer].sample_compare(shadowSamplers[layer], uvLayer.xy, uint(uvLayer.z), depth, gradient2d(ddx, ddy), offset);"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowArrays, shadowSamplers, input.layer, input.uv, input.uvLayer, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_metal_cube_shadow_compare_lod_grad_resource_arrays():
    shader = """
    shader CubeShadowCompareResourceArrays {
        samplerCubeShadow cubeMaps[4];
        samplerCubeArrayShadow cubeArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 direction @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
        };

        float cubeShadowLayer(
            samplerCubeShadow cubeMaps[],
            samplerCubeArrayShadow cubeArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 direction,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float cubeLod = textureCompareLod(cubeMaps[layer], shadowSamplers[layer], direction, depth, lod);
            float cubeGrad = textureCompareGrad(cubeMaps[1], shadowSamplers[1], direction, depth, ddx, ddy);
            float cubeArrayLod = textureCompareLod(cubeArrays[2], shadowSamplers[2], cubeLayer, depth, lod);
            float cubeArrayGrad = textureCompareGrad(cubeArrays[layer], shadowSamplers[layer], cubeLayer, depth, ddx, ddy);
            return cubeLod + cubeGrad + cubeArrayLod + cubeArrayGrad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeShadowLayer(
                    cubeMaps,
                    cubeArrays,
                    shadowSamplers,
                    input.layer,
                    input.direction,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depthcube<float>, 4> cubeMaps [[texture(0)]]" in generated_code
    assert (
        "array<depthcube_array<float>, 4> cubeArrays [[texture(4)]]" in generated_code
    )
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float cubeShadowLayer(array<depthcube<float>, 4> cubeMaps, array<depthcube_array<float>, 4> cubeArrays, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float cubeLod = cubeMaps[layer].sample_compare(shadowSamplers[layer], direction, depth, level(lod));"
        in generated_code
    )
    assert (
        "float cubeGrad = cubeMaps[1].sample_compare(shadowSamplers[1], direction, depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "float cubeArrayLod = cubeArrays[2].sample_compare(shadowSamplers[2], cubeLayer.xyz, uint(cubeLayer.w), depth, level(lod));"
        in generated_code
    )
    assert (
        "float cubeArrayGrad = cubeArrays[layer].sample_compare(shadowSamplers[layer], cubeLayer.xyz, uint(cubeLayer.w), depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "cubeShadowLayer(cubeMaps, cubeArrays, shadowSamplers, input.layer, input.direction, input.cubeLayer, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_metal_array_shadow_texture_resource_arrays_reject_mismatched_fixed_helper_size():
    shader = """
    shader ArrayShadowTextureResourceArrayMismatch {
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
        };

        float sampleArrayLayer(sampler2DArrayShadow shadowArrays[3], sampler shadowSamplers[3], vec3 uvLayer, float depth) {
            return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowArrays': 4 and 3",
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_storage_image_size_queries_use_texture_dimensions():
    shader = """
    shader StorageImageSizeQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        ivec2 queryImage2D(image2D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryImage3D(image3D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryImageArray(image2DArray image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec2 queryUintImage(uimage2D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryIntVolume(iimage3D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 a = textureSize(colorImage, 0) + imageSize(colorImage);
                ivec3 b = textureSize(volumeImage, 0) + imageSize(volumeImage);
                ivec3 c = textureSize(layerImage, 0) + imageSize(layerImage);
                ivec2 d = textureSize(uintImage, 0) + imageSize(uintImage);
                ivec3 e = textureSize(intVolume, 0) + imageSize(intVolume);
                return vec4(a.x + b.x + c.z + d.x + e.z);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> uintImage [[texture(3)]]" in generated_code
    )
    assert (
        "texture3d<int, access::read_write> intVolume [[texture(4)]]" in generated_code
    )
    assert (
        "int2 queryImage2D(texture2d<float, access::read_write> image)"
        in generated_code
    )
    assert (
        "int3 queryImageArray(texture2d_array<float, access::read_write> image)"
        in generated_code
    )
    assert (
        "return int2(image.get_width(), image.get_height()) + int2(image.get_width(), image.get_height());"
        in generated_code
    )
    assert (
        "return int3(image.get_width(), image.get_height(), image.get_depth()) + int3(image.get_width(), image.get_height(), image.get_depth());"
        in generated_code
    )
    assert (
        "return int3(image.get_width(), image.get_height(), image.get_array_size()) + int3(image.get_width(), image.get_height(), image.get_array_size());"
        in generated_code
    )
    assert (
        "int2 a = int2(colorImage.get_width(), colorImage.get_height()) + int2(colorImage.get_width(), colorImage.get_height());"
        in generated_code
    )
    assert (
        "int3 c = int3(layerImage.get_width(), layerImage.get_height(), layerImage.get_array_size()) + int3(layerImage.get_width(), layerImage.get_height(), layerImage.get_array_size());"
        in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "get_width(uint" not in generated_code


def test_metal_storage_image_levels_and_lod_queries_emit_diagnostics():
    shader = """
    shader StorageImageInvalidQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
        };

        int imageLevels(image2D image) {
            return textureQueryLevels(image);
        }

        vec2 imageLod(image2D image, vec2 uv) {
            return textureQueryLod(image, uv);
        }

        int volumeLevels(image3D image) {
            return textureQueryLevels(image);
        }

        vec2 volumeLod(image3D image, vec3 uvw) {
            return textureQueryLod(image, uvw);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                int levels = textureQueryLevels(colorImage)
                    + textureQueryLevels(volumeImage)
                    + textureQueryLevels(layerImage)
                    + textureQueryLevels(uintImage)
                    + textureQueryLevels(intVolume);
                vec2 lod = textureQueryLod(colorImage, input.uv)
                    + textureQueryLod(volumeImage, input.uvw)
                    + imageLod(colorImage, input.uv)
                    + volumeLod(volumeImage, input.uvw);
                return vec4(float(levels) + lod.x);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLevels on texture2d<float, access::read_write> */ 0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLevels on texture3d<float, access::read_write> */ 0"
        )
        == 2
    )
    assert (
        "/* unsupported Metal texture query: textureQueryLevels on texture2d_array<float, access::read_write> */ 0"
        in generated_code
    )
    assert (
        "/* unsupported Metal texture query: textureQueryLevels on texture2d<uint, access::read_write> */ 0"
        in generated_code
    )
    assert (
        "/* unsupported Metal texture query: textureQueryLevels on texture3d<int, access::read_write> */ 0"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLod on texture2d<float, access::read_write> */ float2(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported Metal texture query: textureQueryLod on texture3d<float, access::read_write> */ float2(0.0)"
        )
        == 2
    )
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "calculate_unclamped_lod" not in generated_code
    assert "calculate_clamped_lod" not in generated_code
    assert "sampler(" not in generated_code


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


def test_metal_multisample_texture_samples_queries_use_get_num_samples():
    shader = """
    shader MultisampleSamplesQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(sampler2DMS tex, sampler2DMSArray texArray) {
            return textureSamples(tex) + textureSamples(texArray);
        }

        int queryArraySamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return textureSamples(textures[2]) + textureSamples(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return float(querySamples(msTex, msArray)
                    + queryArraySamples(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "array<texture2d_ms<float>, 4> msTextures [[texture(2)]]" in generated_code
    assert (
        "array<texture2d_ms_array<float>, 4> msArrays [[texture(6)]]" in generated_code
    )
    assert (
        "int querySamples(texture2d_ms<float> tex, texture2d_ms_array<float> texArray)"
        in generated_code
    )
    assert (
        "return int(tex.get_num_samples()) + int(texArray.get_num_samples());"
        in generated_code
    )
    assert (
        "int queryArraySamples(array<texture2d_ms<float>, 4> textures, array<texture2d_ms_array<float>, 4> arrays, int layer)"
        in generated_code
    )
    assert (
        "return int(textures[2].get_num_samples()) + int(arrays[layer].get_num_samples());"
        in generated_code
    )
    assert (
        "querySamples(msTex, msArray) + queryArraySamples(msTextures, msArrays, input.layer)"
        in generated_code
    )
    assert "textureSamples(" not in generated_code


def test_metal_multisample_image_samples_queries_use_get_num_samples():
    shader = """
    shader MultisampleImageSamplesQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(sampler2DMS tex, sampler2DMSArray texArray) {
            return imageSamples(tex) + imageSamples(texArray);
        }

        int queryArraySamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return imageSamples(textures[2]) + imageSamples(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return float(imageSamples(msTex)
                    + imageSamples(msArray)
                    + querySamples(msTex, msArray)
                    + queryArraySamples(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "array<texture2d_ms<float>, 4> msTextures [[texture(2)]]" in generated_code
    assert (
        "array<texture2d_ms_array<float>, 4> msArrays [[texture(6)]]" in generated_code
    )
    assert (
        "int querySamples(texture2d_ms<float> tex, texture2d_ms_array<float> texArray)"
        in generated_code
    )
    assert (
        "return int(tex.get_num_samples()) + int(texArray.get_num_samples());"
        in generated_code
    )
    assert (
        "int queryArraySamples(array<texture2d_ms<float>, 4> textures, array<texture2d_ms_array<float>, 4> arrays, int layer)"
        in generated_code
    )
    assert (
        "return int(textures[2].get_num_samples()) + int(arrays[layer].get_num_samples());"
        in generated_code
    )
    assert (
        "int(msTex.get_num_samples()) + int(msArray.get_num_samples())"
        in generated_code
    )
    assert "queryArraySamples(msTextures, msArrays, input.layer)" in generated_code
    assert generated_code.count(".get_num_samples()") == 6
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "unsupported Metal texture samples query" not in generated_code


def test_metal_non_multisample_texture_samples_emit_diagnostics():
    shader = """
    shader NonMultisampleSamplesQuery {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler2D textures[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(
            sampler2D tex,
            sampler2DArray arrayTex,
            samplerCube cubeTex,
            samplerCubeArray cubeArrayTex,
            sampler2DShadow shadowTex,
            sampler2DArrayShadow shadowArrayTex
        ) {
            return textureSamples(tex)
                + textureSamples(arrayTex)
                + textureSamples(cubeTex)
                + textureSamples(cubeArrayTex)
                + textureSamples(shadowTex)
                + textureSamples(shadowArrayTex);
        }

        int queryResourceArray(sampler2D textures[], int layer) {
            return textureSamples(textures[2]) + textureSamples(textures[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int direct = textureSamples(colorMap) + textureSamples(shadowArray);
                return float(querySamples(
                    colorMap,
                    layerMap,
                    cubeMap,
                    cubeArray,
                    shadowMap,
                    shadowArray
                ) + queryResourceArray(textures, input.layer) + direct);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture samples query: "
        "requires multisample texture */ 0"
    )
    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube<float> cubeMap [[texture(2)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(3)]]" in generated_code
    assert "array<texture2d<float>, 4> textures" in generated_code
    assert generated_code.count(diagnostic) == 10
    assert "textureSamples(" not in generated_code
    assert "get_num_samples" not in generated_code
    assert " sampler " not in generated_code


def test_metal_storage_image_samples_queries_emit_diagnostics():
    shader = """
    shader StorageImageSamplesQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int samples2D(image2D image) {
            return imageSamples(image) + textureSamples(image);
        }

        int samples3D(image3D image) {
            return imageSamples(image) + textureSamples(image);
        }

        int samplesArray(image2DArray image) {
            return imageSamples(image) + textureSamples(image);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int samples = imageSamples(colorImage)
                    + imageSamples(volumeImage)
                    + imageSamples(layerImage)
                    + imageSamples(uintImage)
                    + imageSamples(intVolume)
                    + textureSamples(colorImage)
                    + textureSamples(volumeImage);
                return float(samples
                    + samples2D(colorImage)
                    + samples3D(volumeImage)
                    + samplesArray(layerImage));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texture samples query: "
        "requires multisample texture */ 0"
    )
    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert (
        "texture2d<uint, access::read_write> uintImage [[texture(3)]]" in generated_code
    )
    assert (
        "texture3d<int, access::read_write> intVolume [[texture(4)]]" in generated_code
    )
    assert generated_code.count(diagnostic) == 13
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "get_num_samples" not in generated_code
    assert " sampler " not in generated_code


def test_metal_storage_image_sampling_and_fetch_emit_diagnostics():
    shader = """
    shader StorageImageInvalidSampleFetch {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            ivec2 pixel @ TEXCOORD3;
            ivec3 voxel @ TEXCOORD4;
            ivec3 pixelLayer @ TEXCOORD5;
            ivec2 offset2 @ TEXCOORD6;
            ivec3 offset3 @ TEXCOORD7;
            int lod;
        };

        vec4 invalid2D(image2D image, vec2 uv, ivec2 pixel, int lod, ivec2 offset) {
            return texture(image, uv)
                + textureLod(image, uv, float(lod))
                + textureGather(image, uv)
                + texelFetch(image, pixel, lod)
                + texelFetchOffset(image, pixel, lod, offset);
        }

        vec4 invalid3D(image3D image, vec3 uvw, ivec3 voxel, int lod, ivec3 offset) {
            return texture(image, uvw)
                + textureLod(image, uvw, float(lod))
                + textureGather(image, uvw)
                + texelFetch(image, voxel, lod)
                + texelFetchOffset(image, voxel, lod, offset);
        }

        vec4 invalidArray(image2DArray image, vec3 uvLayer, ivec3 pixelLayer, int lod, ivec2 offset) {
            return texture(image, uvLayer)
                + textureLod(image, uvLayer, float(lod))
                + textureGather(image, uvLayer)
                + texelFetch(image, pixelLayer, lod)
                + texelFetchOffset(image, pixelLayer, lod, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return invalid2D(colorImage, input.uv, input.pixel, input.lod, input.offset2)
                    + invalid3D(volumeImage, input.uvw, input.voxel, input.lod, input.offset3)
                    + invalidArray(layerImage, input.uvLayer, input.pixelLayer, input.lod, input.offset2)
                    + texture(colorImage, input.uv)
                    + textureLod(volumeImage, input.uvw, float(input.lod))
                    + textureGather(layerImage, input.uvLayer)
                    + texelFetch(colorImage, input.pixel, input.lod)
                    + texelFetchOffset(layerImage, input.pixelLayer, input.lod, input.offset2);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = "unsupported Metal storage image texture operation"
    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 20
    for operation in {
        "texture",
        "textureLod",
        "textureGather",
        "texelFetch",
        "texelFetchOffset",
    }:
        assert f"{operation} on texture2d<float, access::read_write>" in generated_code
    assert ".sample(" not in generated_code
    assert ".read(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code
    assert "sampler(mag_filter" not in generated_code
    assert " sampler " not in generated_code


def test_metal_storage_image_compare_calls_emit_diagnostics():
    shader = """
    shader StorageImageInvalidCompare {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
            float depth;
            float lod;
        };

        float compareImplicit(image2D image, vec2 uv, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(image, uv, depth)
                + textureCompareOffset(image, uv, depth, offset)
                + textureCompareLod(image, uv, depth, lod)
                + textureCompareLodOffset(image, uv, depth, lod, offset)
                + textureCompareGrad(image, uv, depth, ddx, ddy)
                + textureCompareGradOffset(image, uv, depth, ddx, ddy, offset);
        }

        float compareExplicit(image3D image, sampler s, vec3 uvw, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(image, s, uvw, depth)
                + textureCompareOffset(image, s, uvw, depth, offset)
                + textureCompareLod(image, s, uvw, depth, lod)
                + textureCompareLodOffset(image, s, uvw, depth, lod, offset)
                + textureCompareGrad(image, s, uvw, depth, ddx, ddy)
                + textureCompareGradOffset(image, s, uvw, depth, ddx, ddy, offset);
        }

        vec4 gatherExplicit(image2DArray image, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(image, s, uvLayer, depth)
                + textureGatherCompareOffset(image, s, uvLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float cmp = compareImplicit(colorImage, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + compareExplicit(volumeImage, compareSampler, input.uvw, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + textureCompare(colorImage, input.uv, input.depth)
                    + textureCompare(volumeImage, compareSampler, input.uvw, input.depth);
                return vec4(cmp)
                    + gatherExplicit(layerImage, compareSampler, input.uvLayer, input.depth, input.offset)
                    + textureGatherCompare(layerImage, compareSampler, input.uvLayer, input.depth)
                    + textureGatherCompareOffset(layerImage, compareSampler, input.uvLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    comparison_diagnostic = "unsupported Metal storage image texture comparison"
    gather_diagnostic = "unsupported Metal storage image texture operation"
    assert (
        "texture2d<float, access::read_write> colorImage [[texture(0)]]"
        in generated_code
    )
    assert (
        "texture3d<float, access::read_write> volumeImage [[texture(1)]]"
        in generated_code
    )
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]"
        in generated_code
    )
    assert "sampler compareSampler [[sampler(0)]]" in generated_code
    assert generated_code.count(comparison_diagnostic) == 14
    assert generated_code.count(gather_diagnostic) == 4
    assert "sample_compare" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_metal_multisample_texture_query_levels_use_single_level():
    shader = """
    shader MultisampleLevelQueries {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int levels2D(sampler2DMS tex) {
            return textureQueryLevels(tex);
        }

        int levelsArray(sampler2DMSArray tex) {
            return textureQueryLevels(tex);
        }

        int levelsResourceArrays(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return textureQueryLevels(textures[2]) + textureQueryLevels(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int c = textureQueryLevels(msTex);
                int d = textureQueryLevels(msArray);
                int e = textureQueryLevels(msTextures[2]);
                int f = textureQueryLevels(msArrays[input.layer]);
                return float(c + d + e + f + levels2D(msTex) + levelsArray(msArray)
                    + levelsResourceArrays(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "array<texture2d_ms<float>, 4> msTextures [[texture(2)]]" in generated_code
    assert (
        "array<texture2d_ms_array<float>, 4> msArrays [[texture(6)]]" in generated_code
    )
    assert "int levels2D(texture2d_ms<float> tex)" in generated_code
    assert "int levelsArray(texture2d_ms_array<float> tex)" in generated_code
    assert (
        "int levelsResourceArrays(array<texture2d_ms<float>, 4> textures, array<texture2d_ms_array<float>, 4> arrays, int layer)"
        in generated_code
    )
    assert generated_code.count("return 1;") == 2
    assert "return 1 + 1;" in generated_code
    assert "int c = 1;" in generated_code
    assert "int d = 1;" in generated_code
    assert "int e = 1;" in generated_code
    assert "int f = 1;" in generated_code
    assert "get_num_mip_levels" not in generated_code
    assert "textureQueryLevels(" not in generated_code


def test_metal_multisample_sampling_operations_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleSampling {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 invalid2D(sampler2DMS tex, vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset) {
            vec4 sampled = texture(tex, uv);
            vec4 lod = textureLod(tex, uv, 0.0);
            vec4 grad = textureGrad(tex, uv, ddx, ddy);
            vec4 gathered = textureGather(tex, uv);
            vec4 offsetGathered = textureGatherOffset(tex, uv, offset);
            return sampled + lod + grad + gathered + offsetGathered;
        }

        vec4 invalidArray(sampler2DMSArray tex, vec3 uvLayer, vec2 ddx, vec2 ddy, ivec2 offset) {
            vec4 sampled = texture(tex, uvLayer);
            vec4 lod = textureLod(tex, uvLayer, 0.0);
            vec4 grad = textureGrad(tex, uvLayer, ddx, ddy);
            vec4 gathered = textureGather(tex, uvLayer);
            vec4 offsetGathered = textureGatherOffset(tex, uvLayer, offset);
            return sampled + lod + grad + gathered + offsetGathered;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return invalid2D(msTex, input.uv, input.ddx, input.ddy, input.offset)
                    + invalidArray(msArray, input.uvLayer, input.ddx, input.ddy, input.offset);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert (
        "float4 invalid2D(texture2d_ms<float> tex, float2 uv, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 invalidArray(texture2d_ms_array<float> tex, float3 uvLayer, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )

    for func_name in {
        "texture",
        "textureLod",
        "textureGrad",
        "textureGather",
        "textureGatherOffset",
    }:
        assert (
            f"unsupported Metal multisample texture call: {func_name} on texture2d_ms<float>"
            in generated_code
        )
        assert (
            f"unsupported Metal multisample texture call: {func_name} on texture2d_ms_array<float>"
            in generated_code
        )

    assert ".sample(" not in generated_code
    assert ".gather(" not in generated_code


def test_metal_multisample_texture_query_lod_emits_diagnostics():
    shader = """
    shader UnsupportedMultisampleQueryLod {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
        };

        vec2 query2D(sampler2DMS tex, vec2 uv) {
            return textureQueryLod(tex, uv);
        }

        vec2 queryArray(sampler2DMSArray tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = textureQueryLod(msTex, input.uv);
                vec2 b = textureQueryLod(msArray, input.uvLayer);
                vec2 c = query2D(msTex, input.uv);
                vec2 d = queryArray(msArray, input.uvLayer);
                return vec4(a + b + c + d, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert "float2 query2D(texture2d_ms<float> tex, float2 uv)" in generated_code
    assert (
        "float2 queryArray(texture2d_ms_array<float> tex, float3 uvLayer)"
        in generated_code
    )
    assert "calculate_unclamped_lod" not in generated_code
    assert "calculate_clamped_lod" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert (
        generated_code.count(
            "unsupported Metal multisample texture query: textureQueryLod on texture2d_ms<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal multisample texture query: textureQueryLod on texture2d_ms_array<float>"
        )
        == 2
    )


def test_metal_multisample_texel_fetch_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleTexelFetchOffset {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            int sampleIndex;
        };

        vec4 offset2D(sampler2DMS tex, ivec2 pixel, int sampleIndex, ivec2 offset) {
            return texelFetchOffset(tex, pixel, sampleIndex, offset);
        }

        vec4 offsetArray(sampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex, ivec2 offset) {
            return texelFetchOffset(tex, pixelLayer, sampleIndex, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return offset2D(msTex, input.pixel, input.sampleIndex, input.offset)
                    + offsetArray(msArray, input.pixelLayer, input.sampleIndex, input.offset)
                    + texelFetchOffset(msTex, input.pixel, input.sampleIndex, input.offset)
                    + texelFetchOffset(msArray, input.pixelLayer, input.sampleIndex, input.offset);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported Metal texel fetch offset: "
        "multisample textures do not support offsets */ float4(0.0)"
    )
    assert "texture2d_ms<float> msTex [[texture(0)]]" in generated_code
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in generated_code
    assert (
        "float4 offset2D(texture2d_ms<float> tex, int2 pixel, int sampleIndex, int2 offset)"
        in generated_code
    )
    assert (
        "float4 offsetArray(texture2d_ms_array<float> tex, int3 pixelLayer, int sampleIndex, int2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 4
    assert "texelFetchOffset(" not in generated_code
    assert ".read(" not in generated_code


def test_metal_cube_texel_fetches_emit_diagnostics():
    shader = """
    shader UnsupportedCubeTexelFetch {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;

        struct FSInput {
            ivec3 cubeCoord @ TEXCOORD0;
            ivec4 cubeLayerCoord @ TEXCOORD1;
            ivec3 offset @ TEXCOORD2;
            int lod;
        };

        vec4 fetchCube(samplerCube tex, ivec3 cubeCoord, int lod, ivec3 offset) {
            vec4 plain = texelFetch(tex, cubeCoord, lod);
            vec4 offsetFetch = texelFetchOffset(tex, cubeCoord, lod, offset);
            return plain + offsetFetch;
        }

        vec4 fetchCubeArray(samplerCubeArray tex, ivec4 cubeLayerCoord, int lod, ivec3 offset) {
            vec4 plain = texelFetch(tex, cubeLayerCoord, lod);
            vec4 offsetFetch = texelFetchOffset(tex, cubeLayerCoord, lod, offset);
            return plain + offsetFetch;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return fetchCube(cubeMap, input.cubeCoord, input.lod, input.offset)
                    + fetchCubeArray(cubeArray, input.cubeLayerCoord, input.lod, input.offset)
                    + texelFetch(cubeMap, input.cubeCoord, input.lod)
                    + texelFetch(cubeArray, input.cubeLayerCoord, input.lod)
                    + texelFetchOffset(cubeMap, input.cubeCoord, input.lod, input.offset)
                    + texelFetchOffset(cubeArray, input.cubeLayerCoord, input.lod, input.offset);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "texturecube<float> cubeMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert (
        "float4 fetchCube(texturecube<float> tex, int3 cubeCoord, int lod, int3 offset)"
        in generated_code
    )
    assert (
        "float4 fetchCubeArray(texturecube_array<float> tex, int4 cubeLayerCoord, int lod, int3 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetch on texturecube<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetchOffset on texturecube<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetch on texturecube_array<float>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported Metal texel fetch: texelFetchOffset on texturecube_array<float>"
        )
        == 2
    )
    assert ".read(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_metal_array_shadow_texture_query_functions():
    shader = """
    shader ArrayShadowTextureQueries {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        ivec3 query2DArrayShadow(sampler2DArrayShadow tex) {
            ivec3 size = textureSize(tex, 1);
            int levels = textureQueryLevels(tex);
            return size + ivec3(levels);
        }

        ivec3 queryCubeArrayShadow(samplerCubeArrayShadow tex) {
            ivec3 size = textureSize(tex, 0);
            int levels = textureQueryLevels(tex);
            return size + ivec3(levels);
        }

        ivec3 queryArrayElements(sampler2DArrayShadow shadowArrays[], samplerCubeArrayShadow cubeShadowArrays[]) {
            ivec3 arraySize = textureSize(shadowArrays[2], 1);
            ivec3 cubeSize = textureSize(cubeShadowArrays[3], 0);
            int arrayLevels = textureQueryLevels(shadowArrays[2]);
            int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);
            return arraySize + cubeSize + ivec3(arrayLevels + cubeLevels);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                ivec3 a = query2DArrayShadow(shadowArray);
                ivec3 b = queryCubeArrayShadow(cubeShadowArray);
                ivec3 c = queryArrayElements(shadowArrays, cubeShadowArrays);
                return vec4(float(a.x + b.y + c.z));
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(2)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(6)]]"
        in generated_code
    )
    assert "int3 query2DArrayShadow(depth2d_array<float> tex)" in generated_code
    assert "int3 queryCubeArrayShadow(depthcube_array<float> tex)" in generated_code
    assert (
        "int3 queryArrayElements(array<depth2d_array<float>, 4> shadowArrays, array<depthcube_array<float>, 4> cubeShadowArrays)"
        in generated_code
    )
    assert (
        "int3 size = int3(tex.get_width(uint(1)), tex.get_height(uint(1)), tex.get_array_size());"
        in generated_code
    )
    assert (
        "int3 size = int3(tex.get_width(uint(0)), tex.get_height(uint(0)), tex.get_array_size());"
        in generated_code
    )
    assert "int levels = int(tex.get_num_mip_levels());" in generated_code
    assert (
        "int3 arraySize = int3(shadowArrays[2].get_width(uint(1)), shadowArrays[2].get_height(uint(1)), shadowArrays[2].get_array_size());"
        in generated_code
    )
    assert (
        "int3 cubeSize = int3(cubeShadowArrays[3].get_width(uint(0)), cubeShadowArrays[3].get_height(uint(0)), cubeShadowArrays[3].get_array_size());"
        in generated_code
    )
    assert (
        "int arrayLevels = int(shadowArrays[2].get_num_mip_levels());" in generated_code
    )
    assert (
        "int cubeLevels = int(cubeShadowArrays[3].get_num_mip_levels());"
        in generated_code
    )
    assert "sample_compare" not in generated_code


def test_metal_direct_stage_texture_queries_mix_size_levels_and_lod():
    shader = """
    shader DirectTextureQueries {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float lod;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 colorSize = textureSize(colorMap, 1);
                ivec3 layerSize = textureSize(layerMap, 2);
                ivec3 cubeSize = textureSize(cubeArray, 0);
                ivec2 shadowSize = textureSize(shadowMap, 0);
                ivec3 shadowArraySize = textureSize(shadowArray, 1);
                ivec3 cubeShadowSize = textureSize(cubeShadowArray, 0);
                int colorLevels = textureQueryLevels(colorMap);
                int shadowLevels = textureQueryLevels(shadowArray);
                vec2 colorLod = textureQueryLod(colorMap, linearSampler, input.uv);
                vec2 layerLod = textureQueryLod(layerMap, linearSampler, input.uvLayer);
                vec2 cubeLod = textureQueryLod(cubeArray, linearSampler, input.cubeLayer);
                vec2 implicitLayerLod = textureQueryLod(layerMap, input.uvLayer);
                vec2 implicitCubeShadowLod = textureQueryLod(cubeShadowArray, input.cubeLayer);
                float total = float(colorSize.x + layerSize.z + cubeSize.z + shadowSize.x + shadowArraySize.z + cubeShadowSize.z + colorLevels + shadowLevels);
                return vec4(total + colorLod.x + layerLod.y + cubeLod.x + implicitLayerLod.x + implicitCubeShadowLod.y);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"

    assert "texture2d<float> colorMap [[texture(0)]]" in generated_code
    assert "texture2d_array<float> layerMap [[texture(1)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(2)]]" in generated_code
    assert "depth2d<float> shadowMap [[texture(3)]]" in generated_code
    assert "depth2d_array<float> shadowArray [[texture(4)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(5)]]" in generated_code
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert (
        "int2 colorSize = int2(colorMap.get_width(uint(1)), colorMap.get_height(uint(1)));"
        in generated_code
    )
    assert (
        "int3 layerSize = int3(layerMap.get_width(uint(2)), layerMap.get_height(uint(2)), layerMap.get_array_size());"
        in generated_code
    )
    assert (
        "int3 cubeSize = int3(cubeArray.get_width(uint(0)), cubeArray.get_height(uint(0)), cubeArray.get_array_size());"
        in generated_code
    )
    assert (
        "int2 shadowSize = int2(shadowMap.get_width(uint(0)), shadowMap.get_height(uint(0)));"
        in generated_code
    )
    assert (
        "int3 shadowArraySize = int3(shadowArray.get_width(uint(1)), shadowArray.get_height(uint(1)), shadowArray.get_array_size());"
        in generated_code
    )
    assert (
        "int3 cubeShadowSize = int3(cubeShadowArray.get_width(uint(0)), cubeShadowArray.get_height(uint(0)), cubeShadowArray.get_array_size());"
        in generated_code
    )
    assert "int colorLevels = int(colorMap.get_num_mip_levels());" in generated_code
    assert "int shadowLevels = int(shadowArray.get_num_mip_levels());" in generated_code
    assert (
        "float2 colorLod = float2(colorMap.calculate_unclamped_lod(linearSampler, input.uv), colorMap.calculate_clamped_lod(linearSampler, input.uv));"
        in generated_code
    )
    assert (
        "float2 layerLod = float2(layerMap.calculate_unclamped_lod(linearSampler, input.uvLayer.xy), layerMap.calculate_clamped_lod(linearSampler, input.uvLayer.xy));"
        in generated_code
    )
    assert (
        "float2 cubeLod = float2(cubeArray.calculate_unclamped_lod(linearSampler, input.cubeLayer.xyz), cubeArray.calculate_clamped_lod(linearSampler, input.cubeLayer.xyz));"
        in generated_code
    )
    assert (
        f"float2 implicitLayerLod = float2(layerMap.calculate_unclamped_lod({default_sampler}, input.uvLayer.xy), layerMap.calculate_clamped_lod({default_sampler}, input.uvLayer.xy));"
        in generated_code
    )
    assert (
        f"float2 implicitCubeShadowLod = float2(cubeShadowArray.calculate_unclamped_lod({default_sampler}, input.cubeLayer.xyz), cubeShadowArray.calculate_clamped_lod({default_sampler}, input.cubeLayer.xyz));"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code


def test_metal_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ArrayTextureQueryLod {
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DArray layerMaps[4];
        samplerCubeArray cubeArrays[4];
        sampler linearSampler;
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArray tex, sampler s, vec3 uvLayer) {
            return textureQueryLod(tex, s, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
            return textureQueryLod(tex, s, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArray layerMaps[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], linearSamplers[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArray cubeArrays[], sampler linearSamplers[], vec4 cubeLayer) {
            return textureQueryLod(cubeArrays[3], linearSamplers[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(layerMap, linearSampler, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeArray, linearSampler, input.cubeLayer);
                vec2 c = queryArrayElementLod(layerMaps, linearSamplers, input.uvLayer);
                vec2 d = queryCubeArrayElementLod(cubeArrays, linearSamplers, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "texture2d_array<float> layerMap [[texture(0)]]" in generated_code
    assert "texturecube_array<float> cubeArray [[texture(1)]]" in generated_code
    assert "array<texture2d_array<float>, 4> layerMaps [[texture(2)]]" in generated_code
    assert (
        "array<texturecube_array<float>, 4> cubeArrays [[texture(6)]]" in generated_code
    )
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "array<sampler, 4> linearSamplers [[sampler(1)]]" in generated_code
    assert (
        "float2 queryArrayLod(texture2d_array<float> tex, sampler s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, uvLayer.xy)" in generated_code
    assert "tex.calculate_clamped_lod(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(texturecube_array<float> tex, sampler s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, cubeLayer.xyz)" in generated_code
    assert "tex.calculate_clamped_lod(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(array<texture2d_array<float>, 4> layerMaps, array<sampler, 4> linearSamplers, float3 uvLayer)"
        in generated_code
    )
    assert (
        "layerMaps[2].calculate_unclamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "layerMaps[2].calculate_clamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(array<texturecube_array<float>, 4> cubeArrays, array<sampler, 4> linearSamplers, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeArrays[3].calculate_unclamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeArrays[3].calculate_clamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "calculate_unclamped_lod(s, uvLayer)" not in generated_code
    assert "calculate_unclamped_lod(s, cubeLayer)" not in generated_code


def test_metal_shadow_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ShadowArrayTextureQueryLod {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler linearSampler;
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArrayShadow tex, sampler s, vec3 uvLayer) {
            return textureQueryLod(tex, s, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer) {
            return textureQueryLod(tex, s, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(shadowArrays[2], linearSamplers[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], sampler linearSamplers[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], linearSamplers[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(shadowArray, linearSampler, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeShadowArray, linearSampler, input.cubeLayer);
                vec2 c = queryArrayElementLod(shadowArrays, linearSamplers, input.uvLayer);
                vec2 d = queryCubeArrayElementLod(cubeShadowArrays, linearSamplers, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(2)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(6)]]"
        in generated_code
    )
    assert "sampler linearSampler [[sampler(0)]]" in generated_code
    assert "array<sampler, 4> linearSamplers [[sampler(1)]]" in generated_code
    assert (
        "float2 queryArrayLod(depth2d_array<float> tex, sampler s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, uvLayer.xy)" in generated_code
    assert "tex.calculate_clamped_lod(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(depthcube_array<float> tex, sampler s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.calculate_unclamped_lod(s, cubeLayer.xyz)" in generated_code
    assert "tex.calculate_clamped_lod(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(array<depth2d_array<float>, 4> shadowArrays, array<sampler, 4> linearSamplers, float3 uvLayer)"
        in generated_code
    )
    assert (
        "shadowArrays[2].calculate_unclamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "shadowArrays[2].calculate_clamped_lod(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(array<depthcube_array<float>, 4> cubeShadowArrays, array<sampler, 4> linearSamplers, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].calculate_unclamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].calculate_clamped_lod(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "calculate_unclamped_lod(s, uvLayer)" not in generated_code
    assert "calculate_unclamped_lod(s, cubeLayer)" not in generated_code


def test_metal_mixed_cube_shadow_query_lod_and_compare_keep_coordinate_shapes():
    shader = """
    shader MixedCubeShadowQueryLodCompare {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
        sampler shadowSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
        };

        float inspectCubeShadow(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            vec2 lodValue = textureQueryLod(tex, s, direction);
            float cmp = textureCompare(tex, s, direction, depth);
            float cmpLod = textureCompareLod(tex, s, direction, depth, lod);
            float grad = textureCompareGrad(tex, s, direction, depth, ddx, ddy);
            return lodValue.x + cmp + cmpLod + grad;
        }

        float inspectCubeArrayShadow(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            vec2 lodValue = textureQueryLod(tex, s, cubeLayer);
            float cmp = textureCompare(tex, s, cubeLayer, depth);
            float cmpLod = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float grad = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            return lodValue.y + cmp + cmpLod + grad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspectCubeShadow(
                    cubeShadow,
                    shadowSampler,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                ) + inspectCubeArrayShadow(
                    cubeShadowArray,
                    shadowSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                );
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depthcube<float> cubeShadow [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert "sampler shadowSampler [[sampler(0)]]" in generated_code
    assert (
        "float inspectCubeShadow(depthcube<float> tex, sampler s, float3 direction, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 lodValue = float2(tex.calculate_unclamped_lod(s, direction), tex.calculate_clamped_lod(s, direction));"
        in generated_code
    )
    assert "float cmp = tex.sample_compare(s, direction, depth);" in generated_code
    assert (
        "float cmpLod = tex.sample_compare(s, direction, depth, level(lod));"
        in generated_code
    )
    assert (
        "float grad = tex.sample_compare(s, direction, depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "float inspectCubeArrayShadow(depthcube_array<float> tex, sampler s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 lodValue = float2(tex.calculate_unclamped_lod(s, cubeLayer.xyz), tex.calculate_clamped_lod(s, cubeLayer.xyz));"
        in generated_code
    )
    assert (
        "float cmp = tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth);"
        in generated_code
    )
    assert (
        "float cmpLod = tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, level(lod));"
        in generated_code
    )
    assert (
        "float grad = tex.sample_compare(s, cubeLayer.xyz, uint(cubeLayer.w), depth, gradientcube(ddx, ddy));"
        in generated_code
    )
    assert (
        "inspectCubeShadow(cubeShadow, shadowSampler, input.direction, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "inspectCubeArrayShadow(cubeShadowArray, shadowSampler, input.cubeLayer, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_metal_implicit_shadow_array_texture_query_lod_uses_default_sampler():
    shader = """
    shader ImplicitShadowArrayTextureQueryLod {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArrayShadow tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], vec3 uvLayer) {
            return textureQueryLod(shadowArrays[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = textureQueryLod(shadowArray, input.uvLayer);
                vec2 b = textureQueryLod(cubeShadowArray, input.cubeLayer);
                vec2 c = queryArrayLod(shadowArray, input.uvLayer);
                vec2 d = queryCubeArrayLod(cubeShadowArray, input.cubeLayer);
                vec2 e = queryArrayElementLod(shadowArrays, input.uvLayer);
                vec2 f = queryCubeArrayElementLod(cubeShadowArrays, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, e.x + f.y, 1.0);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "depth2d_array<float> shadowArray [[texture(0)]]" in generated_code
    assert "depthcube_array<float> cubeShadowArray [[texture(1)]]" in generated_code
    assert (
        "array<depth2d_array<float>, 4> shadowArrays [[texture(2)]]" in generated_code
    )
    assert (
        "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(6)]]"
        in generated_code
    )
    assert "sampler linearSampler" not in generated_code
    assert "linearSamplers" not in generated_code
    default_sampler = "sampler(mag_filter::linear, min_filter::linear)"
    assert (
        f"shadowArray.calculate_unclamped_lod({default_sampler}, input.uvLayer.xy)"
        in generated_code
    )
    assert (
        f"cubeShadowArray.calculate_unclamped_lod({default_sampler}, input.cubeLayer.xyz)"
        in generated_code
    )
    assert (
        f"tex.calculate_unclamped_lod({default_sampler}, uvLayer.xy)" in generated_code
    )
    assert (
        f"tex.calculate_unclamped_lod({default_sampler}, cubeLayer.xyz)"
        in generated_code
    )
    assert (
        f"shadowArrays[2].calculate_unclamped_lod({default_sampler}, uvLayer.xy)"
        in generated_code
    )
    assert (
        f"cubeShadowArrays[3].calculate_unclamped_lod({default_sampler}, cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "calculate_unclamped_lod(sampler(mag_filter::linear, min_filter::linear), uvLayer)"
        not in generated_code
    )
    assert (
        "calculate_unclamped_lod(sampler(mag_filter::linear, min_filter::linear), cubeLayer)"
        not in generated_code
    )


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


def test_metal_fixed_shadow_texture_array_rejects_mismatched_fixed_helper_size():
    shader = """
    shader FixedShadowGlobalMismatch {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float shadowThree(sampler2DShadow shadowMaps[3], sampler shadowSamplers[3], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowThree(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowMaps': 4 and 3",
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


def test_metal_fixed_shadow_texture_array_widens_unsized_helper():
    shader = """
    shader FixedShadowGlobalToUnsizedHelper {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float shadowUnsized(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float shadowUnsized(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "shadowMaps[2].sample_compare(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert (
        "shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_transitive_shadow_texture_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveShadowSamplerShadowedConstIndex {
        const int COUNT = 4;
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        }

        float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            float sampled = textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
            return sampled + leaf(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = MetalCodeGen().generate(crosstl.translator.parse(shader))

    assert "constant int COUNT = 4;" in generated_code
    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in generated_code
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in generated_code
    assert (
        "float leaf(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert (
        "float passThrough(array<depth2d<float>, 4> shadowMaps, array<sampler, 4> shadowSamplers"
        in generated_code
    )
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "shadowMaps[COUNT].sample_compare(shadowSamplers[COUNT], uv, depth)"
        in generated_code
    )
    assert "leaf(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "passThrough(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "array<depth2d<float>, 1> shadowMaps" not in generated_code
    assert "textureCompare(" not in generated_code


def test_metal_transitive_shadow_texture_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveShadowSamplerUnshadowedConstIndexConflict {
        const int COUNT = 4;
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        }

        float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            return leaf(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowMaps': 4 and 5",
    ):
        MetalCodeGen().generate(crosstl.translator.parse(shader))


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
