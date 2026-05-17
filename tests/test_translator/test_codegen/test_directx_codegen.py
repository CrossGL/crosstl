import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
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
    codegen = HLSLCodeGen()
    return codegen.generate(ast_node)


def test_hlsl_unsigned_integer_literal_suffix_codegen():
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


def test_hlsl_default_float_image_scalar_and_vector_load_store():
    code = """
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float scalarOld = image[pixel].x;" in generated_code
    assert "image[pixel] = float4((scalarOld + value));" in generated_code
    assert "float4 vectorOld = image[pixel];" in generated_code
    assert "image[pixel] = (vectorOld + value);" in generated_code
    assert "float4 vectorOld = image[pixel].x;" not in generated_code
    assert "image[pixel] = float4((vectorOld + value));" not in generated_code


def test_hlsl_rg_image_scalar_and_vector_load_store():
    code = """
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float oldValue = image[pixel].x;" in generated_code
    assert "uint oldValue = image[pixel].x;" in generated_code
    assert "image[pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "image[pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "uint2 oldValue = image[pixel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code


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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const float weights[2];" in generated_code
    assert "for (int i = 0; (i < 2); ++i)" in generated_code
    assert "for (i = 0; (i < 4); ++i)" in generated_code
    assert "for (const int fixed = 0; (fixed < 0); )" in generated_code
    assert "for (; ; )" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "for (i; (i < 2); ++i)" not in generated_code
    assert "for (fixed; (fixed < 0); )" not in generated_code
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (true)" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "if ((i >= limit))" in generated_code
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 0; i < limit; ++i)" in generated_code
    assert "total = (total + i);" in generated_code
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 2; i < 5; ++i)" in generated_code
    assert "for (int j = 1; j <= limit; ++j)" in generated_code
    assert "total = (total + i);" in generated_code
    assert "total = (total + j);" in generated_code
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "while ((i < 4))" in generated_code
    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "i = (i + 1);" in generated_code
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:\n        case 1:" in generated_code
    assert "case 2:\n            switch (submode)" in generated_code
    assert generated_code.count("default:") == 2
    assert "value = (value + 1);" in generated_code
    assert "value = (value + 2);" in generated_code
    assert "value = (value + 3);" in generated_code
    assert "value = (value + 4);" in generated_code
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

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
            HLSLCodeGen().generate(crosstl.translator.parse(shader))


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
    assert "payload" in generated


def test_ray_and_mesh_shader_attributes():
    code = """
    shader rt {
        ray_generation {
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
    assert '[shader("raygeneration")]' in generated
    assert '[shader("mesh")]' in generated


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
    generator = HLSLCodeGen()

    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")

    assert "float adjust(float value)" in vertex_code
    assert "float adjust(float value)" in fragment_code
    assert "float3 position: POSITION;" in vertex_code
    assert "float4 position: SV_POSITION;" in vertex_code
    assert "float2 uv: TEXCOORD0;" in fragment_code
    assert "VSOutput VSMain" in vertex_code
    assert "float4 PSMain" not in vertex_code
    assert "float4 PSMain(VSOutput input): SV_TARGET" in fragment_code
    assert "VSOutput VSMain" not in fragment_code


def test_compute_stage_emits_default_numthreads_attribute():
    shader = """
    shader ComputeNumthreadsSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    compute_code = HLSLCodeGen().generate_stage(ast, "compute")

    assert "[numthreads(1, 1, 1)]" in compute_code
    assert compute_code.index("[numthreads(1, 1, 1)]") < compute_code.index("CSMain")


def test_compute_stage_uses_execution_config_numthreads():
    shader = """
    shader ComputeConfiguredNumthreadsSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    ast.stages[ShaderStage.COMPUTE].execution_config = {"numthreads": (8, 4, 2)}

    compute_code = HLSLCodeGen().generate_stage(ast, "compute")

    assert "[numthreads(8, 4, 2)]" in compute_code


def test_wave_and_rayquery_intrinsics_codegen():
    code = """
    shader main {
        compute {
            void main() {
                uint v;
                uint sum = WaveActiveSum(v);
                RayQuery rq;
                rq.Proceed();
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert "WaveActiveSum" in generated
    assert "rq.Proceed" in generated


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


def test_multiview_and_viewport_semantics_roundtrip():
    shader = """
    shader ViewShader {
        struct VSOut {
            vec4 position @ gl_Position;
            uint view @ gl_ViewID;
            uint layer @ gl_Layer;
            uint viewport @ gl_ViewportIndex;
        };

        vertex {
            VSOut main() {
                VSOut o;
                o.position = vec4(0.0, 0.0, 0.0, 1.0);
                o.view = 1;
                o.layer = 2;
                o.viewport = 3;
                return o;
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)
    for semantic in ["SV_ViewID", "SV_RenderTargetArrayIndex", "SV_ViewportArrayIndex"]:
        assert semantic in generated_code


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
            assert (
                expected in generated_code
                or expected.replace("[", "<").replace("]", ">") in generated_code
            )

        for expected in array_test_data["hlsl"]["array_access"]:
            assert expected in generated_code

    except SyntaxError as e:
        pytest.fail(f"DirectX array codegen failed: {e}")


def test_directx_local_array_declarations_use_hlsl_order():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "float3 localColors[4];" in generated_code
    assert "float weights[8];" in generated_code
    assert "float3[4] localColors" not in generated_code
    assert "float[8] weights" not in generated_code


def test_directx_array_parameters_use_hlsl_order():
    shader = """
    shader TestShader {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "float accumulate(float weights[4], float3 colors[2])" in generated_code
    assert "float[4] weights" not in generated_code
    assert "float3[2] colors" not in generated_code


def test_directx_non_resource_arrays_preserve_expression_sizes():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float3 colors[((2 + 1) * 2)];" in generated_code
    assert "float weights[+6];" in generated_code
    assert (
        "float accumulate(float values[((2 + 1) * 2)], float3 normals[+6])"
        in generated_code
    )
    assert "float localWeights[((2 + 1) * 2)];" in generated_code
    assert "float3 localNormals[+6];" in generated_code
    assert "float values[]" not in generated_code
    assert "float localWeights[]" not in generated_code


def test_directx_texture_resources_and_sampling():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "TextureCube envMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState envMapSampler : register(s1);" in generated_code
    assert "colorMap.Sample(colorMapSampler, input.uv)" in generated_code
    assert "envMap.Sample(envMapSampler, input.normal)" in generated_code
    assert "sampler2D colorMap" not in generated_code
    assert "VectorType(" not in generated_code
    assert generated_code.count("// Fragment Shader") == 1


def test_directx_storage_image_load_store():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float4> outputImage : register(u0);" in generated_code
    assert "RWTexture3D<float4> volumeImage : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layerImage : register(u2);" in generated_code
    assert (
        "float4 touchImages(RWTexture2D<float4> outImg, RWTexture3D<float4> volume, RWTexture2DArray<float4> layers, int2 pixel, int3 voxel, int3 pixelLayer)"
        in generated_code
    )
    assert "float4 color = outImg[pixel];" in generated_code
    assert "float4 volumeColor = volume[voxel];" in generated_code
    assert "float4 layerColor = layers[pixelLayer];" in generated_code
    assert "outImg[pixel] = (color + layerColor);" in generated_code
    assert "volume[voxel] = volumeColor;" in generated_code
    assert "layers[pixelLayer] = color;" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_integer_image_atomic_add():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters : register(u0);" in generated_code
    assert "RWTexture2D<int> signedCounters : register(u1);" in generated_code
    assert (
        "uint imageAtomicAdd_uimage2D(RWTexture2D<uint> image, int2 coord, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicAdd_iimage2D(RWTexture2D<int> image, int2 coord, int value)"
        in generated_code
    )
    assert "InterlockedAdd(image[coord], value, original);" in generated_code
    assert (
        "uint addCounter(RWTexture2D<uint> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int addSignedCounter(RWTexture2D<int> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "uint previous = imageAtomicAdd_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicAdd_iimage2D(image, pixel, value);" in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code


def test_directx_integer_image_atomic_operations():
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
    generated_code = HLSLCodeGen().generate(ast)

    for intrinsic in [
        "InterlockedMin",
        "InterlockedMax",
        "InterlockedAnd",
        "InterlockedOr",
        "InterlockedXor",
        "InterlockedExchange",
    ]:
        assert f"{intrinsic}(image[coord], value, original);" in generated_code

    for operation in [
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
    ]:
        assert f"{operation}_uimage2D(RWTexture2D<uint> image" in generated_code
        assert f"{operation}_uimage2D(image, pixel, value)" in generated_code
        assert f"{operation}(image" not in generated_code

    for operation in ["imageAtomicMin", "imageAtomicMax", "imageAtomicExchange"]:
        assert f"{operation}_iimage2D(RWTexture2D<int> image" in generated_code
        assert f"{operation}_iimage2D(image, pixel, value)" in generated_code


def test_directx_integer_image_atomic_compare_swap():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "uint imageAtomicCompSwap_uimage2D(RWTexture2D<uint> image, int2 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2D(RWTexture2D<int> image, int2 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "InterlockedCompareExchange(image[coord], compareValue, value, original);"
        in generated_code
    )
    assert (
        "uint previous = imageAtomicCompSwap_uimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicCompSwap_iimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(image" not in generated_code


def test_directx_integer_image_dimension_atomics():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture3D<uint> volumeCounters : register(u0);" in generated_code
    assert "RWTexture3D<int> signedVolumeCounters : register(u1);" in generated_code
    assert "RWTexture2DArray<int> layerCounters : register(u2);" in generated_code
    assert (
        "RWTexture2DArray<uint> unsignedLayerCounters : register(u3);" in generated_code
    )
    assert (
        "uint imageAtomicAdd_uimage3D(RWTexture3D<uint> image, int3 coord, uint value)"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(RWTexture3D<uint> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicMin_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "uint oldValue = imageAtomicAdd_uimage3D(image, voxel, value);"
        in generated_code
    )
    assert (
        "uint swapped = imageAtomicCompSwap_uimage3D(image, voxel, oldValue, value);"
        in generated_code
    )
    assert (
        "int oldValue = imageAtomicMin_iimage2DArray(image, pixelLayer, value);"
        in generated_code
    )
    assert (
        "int swapped = imageAtomicCompSwap_iimage2DArray(image, pixelLayer, oldValue, value);"
        in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicMin(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code


def test_directx_integer_image_scalar_load_store():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters : register(u0);" in generated_code
    assert "RWTexture3D<int> signedVolume : register(u1);" in generated_code
    assert "RWTexture2DArray<uint> layerCounters : register(u2);" in generated_code
    assert "uint oldValue = image[pixel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "int oldValue = image[voxel];" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "uint oldValue = image[pixelLayer];" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_scalar_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float> scalarFloat : register(u0);" in generated_code
    assert "RWTexture3D<int> signedVolume : register(u1);" in generated_code
    assert "RWTexture2DArray<uint> unsignedLayers : register(u2);" in generated_code
    assert (
        "float touchFloat(RWTexture2D<float> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "int touchSigned(RWTexture3D<int> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint touchUnsigned(RWTexture2DArray<uint> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "float oldValue = image[pixel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "int oldValue = image[voxel];" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "uint oldValue = image[pixelLayer];" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert ": r32" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_rg_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float2> rgFloat : register(u0);" in generated_code
    assert "RWTexture3D<int2> rgSigned : register(u1);" in generated_code
    assert "RWTexture2DArray<uint2> rgUnsigned : register(u2);" in generated_code
    assert (
        "float2 touchFloat(RWTexture2D<float2> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(RWTexture3D<int2> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(RWTexture2DArray<uint2> image, int3 pixelLayer, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "int2 oldValue = image[voxel];" in generated_code
    assert "uint2 oldValue = image[pixelLayer];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> rgFloat" not in generated_code
    assert "RWTexture3D<float4> rgSigned" not in generated_code
    assert "RWTexture2DArray<float4> rgUnsigned" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_narrow_rg_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float2> rg8Float : register(u0);" in generated_code
    assert "RWTexture2D<float2> rg8Snorm : register(u1);" in generated_code
    assert "RWTexture3D<float2> rg16Float : register(u2);" in generated_code
    assert "RWTexture2D<float2> rg16Snorm : register(u3);" in generated_code
    assert "RWTexture2DArray<float2> rg16Half : register(u4);" in generated_code
    assert "RWTexture2D<int2> rg8Signed : register(u5);" in generated_code
    assert "RWTexture3D<int2> rg16Signed : register(u6);" in generated_code
    assert "RWTexture2D<uint2> rg8Unsigned : register(u7);" in generated_code
    assert "RWTexture2DArray<uint2> rg16Unsigned : register(u8);" in generated_code
    assert (
        "float2 touchFloat(RWTexture2D<float2> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchHalf(RWTexture2DArray<float2> image, int3 pixelLayer, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(RWTexture3D<int2> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(RWTexture2D<uint2> image, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "float2 oldValue = image[pixelLayer];" in generated_code
    assert "int2 oldValue = image[voxel];" in generated_code
    assert "uint2 oldValue = image[pixel];" in generated_code
    assert "RWTexture2D<float4> rg8Float" not in generated_code
    assert "RWTexture3D<float4> rg16Float" not in generated_code
    assert "RWTexture2D<float4> rg8Signed" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_rgba_float_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float4> rgba8Color : register(u0);" in generated_code
    assert "RWTexture2D<float4> rgba8Snorm : register(u1);" in generated_code
    assert "RWTexture3D<float4> rgba16Color : register(u2);" in generated_code
    assert "RWTexture2D<float4> rgba16Snorm : register(u3);" in generated_code
    assert "RWTexture2DArray<float4> rgba16Half : register(u4);" in generated_code
    assert "RWTexture3D<float4> rgba32Float : register(u5);" in generated_code
    assert (
        "float4 touchColor(RWTexture2D<float4> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchHalf(RWTexture2DArray<float4> image, int3 pixelLayer, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchFloat(RWTexture3D<float4> image, int3 voxel, float4 value)"
        in generated_code
    )
    assert (
        "float4 typedOverride(RWTexture2D<float4> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert "float4 oldValue = image[pixel];" in generated_code
    assert "float4 oldValue = image[pixelLayer];" in generated_code
    assert "float4 oldValue = image[voxel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "RWTexture2D<int> image" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_preserve_format_metadata():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters[2] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[3] : register(u2);" in generated_code
    assert "RWTexture3D<float4> rgbaVolumes[2] : register(u5);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert "Texture2D sampled : register(t0);" in generated_code
    assert "SamplerState sampledSampler" not in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[2], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float4 touchVolumes(RWTexture3D<float4> images[2], int3 voxel, float4 value)"
        in generated_code
    )
    assert "uint oldValue = images[1][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float4 oldValue = images[1][voxel];" in generated_code
    assert "images[0][voxel] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "RWTexture3D<int" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_rg_image_arrays_respect_scalar_and_vector_context():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[3] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[2] : register(u3);" in generated_code
    assert (
        "float scalarFloat(RWTexture2D<float2> images[3], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(RWTexture2D<uint2> images[2], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(RWTexture2D<uint2> images[2], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[1][pixel].x;" in generated_code
    assert "images[0][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "uint oldValue = images[1][pixel].x;" in generated_code
    assert "images[0][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "uint2 oldValue = images[1][pixel];" in generated_code
    assert "float oldValue = images[1][pixel];" not in generated_code
    assert "uint oldValue = images[1][pixel];" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_inferred_rg_image_arrays_respect_scalar_context():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 3;" in generated_code
    assert "static const int LAYER = (COUNT - 1);" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[3] : register(u0);" in generated_code
    assert (
        "RWTexture2D<uint2> rgUnsignedImages[COUNT] : register(u3);" in generated_code
    )
    assert "RWTexture2D<float2> afterImages : register(u6);" in generated_code
    assert (
        "float scalarFloat(RWTexture2D<float2> images[3], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(RWTexture2D<uint2> images[COUNT], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(RWTexture2D<uint2> images[COUNT], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "uint oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> afterImages : register(u3);" not in generated_code
    assert "float oldValue = images[LAYER][pixel];" not in generated_code
    assert "uint oldValue = images[LAYER][pixel];" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_transitive_rg_image_arrays_share_call_site_size():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert "RWTexture2D<float2> afterImages : register(u8);" in generated_code
    assert (
        "float scalarFloatDeep(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float scalarFloatMid(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatDeep(RWTexture2D<float2> images[4], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatMid(RWTexture2D<float2> images[4], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedDeep(RWTexture2D<uint2> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedDeep(RWTexture2D<uint2> images[4], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3][pixel].x;" in generated_code
    assert "images[1][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "images[1][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "float2 vectorFloatDeep(RWTexture2D<float2> images[3]" not in generated_code
    assert "uint2 vectorUnsignedDeep(RWTexture2D<uint2> images[3]" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_fixed_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float scalarFloatFixed(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(RWTexture2D<float2> images[4], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(RWTexture2D<uint2> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(RWTexture2D<uint2> images[4], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3][pixel].x;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[] : register(u1);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_const_sized_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert "static const int LAST = (COUNT - 1);" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float scalarFloatFixed(RWTexture2D<float2> images[COUNT], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(RWTexture2D<float2> images[COUNT], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(RWTexture2D<uint2> images[COUNT], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(RWTexture2D<uint2> images[COUNT], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAST][pixel].x;" in generated_code
    assert "uint oldValue = images[LAST][pixel].x;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "float scalarFloatFixed(RWTexture2D<float2> images[4]" not in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_expr_sized_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 3;" in generated_code
    assert "static const int UINT_COUNT = 2;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float scalarFloatFixed(RWTexture2D<float2> images[(COUNT + 1)], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(RWTexture2D<float2> images[(COUNT + 1)], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(RWTexture2D<uint2> images[(UINT_COUNT * 2)], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(RWTexture2D<uint2> images[(UINT_COUNT * 2)], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[COUNT][pixel].x;" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "float scalarFloatFixed(RWTexture2D<float2> images[4]" not in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_conflicting_fixed_rg_image_array_sizes_raise():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_direct_rg_image_array_index_conflicts_with_fixed_parameter_size():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_direct_rg_image_array_index_within_fixed_parameter_size():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float touchFour(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint touchUnsignedFour(RWTexture2D<uint2> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert "float directFloat = rgFloatImages[2][pixel].x;" in generated_code
    assert "uint directUint = rgUnsignedImages[1][pixel].x;" in generated_code
    assert "float oldValue = images[3][pixel].x;" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[3] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_fixed_rg_image_array_global_conflicts_with_fixed_parameter_size():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_global_widens_unsized_parameter_size():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert (
        "float touchUnsized(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert "float oldValue = images[2][pixel].x;" in generated_code
    assert (
        "float touchUnsized(RWTexture2D<float2> images[3], int2 pixel, float value)"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_fixed_rg_image_array_global_direct_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_parameter_direct_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_global_const_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_parameter_const_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_const_index_within_bounds_generates():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "float touch(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    assert "return images[(COUNT - 1)][pixel].x;" in generated_code
    assert "float direct = rgFloatImages[(COUNT - 1)][pixel].x;" in generated_code
    assert (
        "float touch(RWTexture2D<float2> images[5], int2 pixel)" not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_directx_fixed_rg_image_array_shadowed_const_index_stays_dynamic():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "float touch(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    assert "return images[COUNT][pixel].x;" in generated_code
    assert "float direct = rgFloatImages[COUNT][pixel].x;" in generated_code
    assert (
        "float touch(RWTexture2D<float2> images[5], int2 pixel)" not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_directx_transitive_rg_image_array_shadowed_const_index_stays_dynamic():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "float leaf(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    assert (
        "float passThrough(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    )
    assert "return images[COUNT][pixel].x;" in generated_code
    assert "float sampled = images[COUNT][pixel].x;" in generated_code
    assert "float leaf(RWTexture2D<float2> images[5], int2 pixel)" not in generated_code
    assert (
        "float passThrough(RWTexture2D<float2> images[5], int2 pixel)"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_directx_transitive_rg_image_array_unshadowed_const_index_conflict_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_shadowed_rg_image_array_constant_keeps_scalar_context():
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
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 3;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[] : register(u1);" in generated_code
    assert "RWTexture2D<float2> afterImages : register(u2);" in generated_code
    assert (
        "float scalarFloat(RWTexture2D<float2> images[], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(RWTexture2D<uint2> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "float oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "uint oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> afterImages : register(u4);" not in generated_code
    assert "float oldValue = images[LAYER][pixel];" not in generated_code
    assert "uint oldValue = images[LAYER][pixel];" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_preserve_expression_sizes():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "RWTexture2D<uint> counters[((1 + 1) * 2)] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[+3] : register(u4);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert "Texture2D sampled : register(t0);" in generated_code
    assert "SamplerState sampledSampler" not in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[((1 + 1) * 2)], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[+3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u4);" not in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_unsized_formatted_image_arrays_preserve_format_metadata():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters[] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[] : register(u1);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[0][pixel];" in generated_code
    assert "float2 oldValue = images[0][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_infer_named_constant_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int COUNT = 3;" in generated_code
    assert "static const int LAYER = (COUNT - 1);" in generated_code
    assert "RWTexture2D<uint> counters[COUNT] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[3] : register(u3);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u6);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[COUNT], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[LAYER][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[LAYER][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert "RWTexture2D<float2> rgPairs[] : register(u3);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_ignore_shadowed_local_constant():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 3;" in generated_code
    assert "RWTexture2D<uint> counters[] : register(u0);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "uint oldValue = images[LAYER][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "RWTexture2D<uint> counters[4] : register(u0);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u4);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_infer_transitive_helper_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "RWTexture2D<uint> counters[4] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[3] : register(u4);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert (
        "uint touchCountersDeep(RWTexture2D<uint> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint touchCountersMid(RWTexture2D<uint> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairsDeep(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchPairsMid(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[3][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "return touchCountersDeep(images, pixel, value);" in generated_code
    assert "return touchPairsDeep(images, pixel, value);" in generated_code
    assert "uint a = touchCountersMid(counters, int2(1, 2), 3);" in generated_code
    assert (
        "float2 b = touchPairsMid(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    )
    assert "RWTexture2D<uint> counters[] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> rgPairs[] : register(u4);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_ignore_unsupported_indices():
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

    dynamic_code = HLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = HLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "RWTexture2D<uint> counters[] : register(u0);" in dynamic_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in dynamic_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int layer, int2 pixel, uint value)"
        in dynamic_code
    )
    assert "uint oldValue = images[layer][pixel];" in dynamic_code
    assert "images[0][pixel] = (oldValue + value);" in dynamic_code
    assert "uint a = touchCounters(counters, 0, int2(1, 2), 3);" in dynamic_code
    assert "RWTexture2D<uint> counters[2] : register(u0);" not in dynamic_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in dynamic_code
    assert "imageLoad(" not in dynamic_code
    assert "imageStore(" not in dynamic_code

    assert "RWTexture2D<uint> counters[] : register(u0);" in negative_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in negative_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in negative_code
    )
    assert "uint oldValue = images[-1][pixel];" in negative_code
    assert "images[0][pixel] = (oldValue + value);" in negative_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in negative_code
    assert "RWTexture2D<uint> counters[0] : register(u0);" not in negative_code
    assert "RWTexture2D<uint> afterCounters : register(u0);" not in negative_code
    assert "imageLoad(" not in negative_code
    assert "imageStore(" not in negative_code


def test_directx_formatted_image_arrays_ignore_function_call_indices():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "RWTexture2D<uint> counters[] : register(u0);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in generated_code
    assert "int getLayer()" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = images[getLayer()][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "RWTexture2D<uint> counters[1] : register(u0);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_infer_local_constant_alias_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int GLOBAL = 2;" in generated_code
    assert "RWTexture2D<uint> counters[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u4);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert "const int LOCAL = (GLOBAL + 1);" in generated_code
    assert "uint oldValue = images[LOCAL][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "RWTexture2D<uint> counters[] : register(u0);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" not in generated_code
    assert "    int LOCAL = (GLOBAL + 1);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_scalar_float_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float> normalized8 : register(u0);" in generated_code
    assert "RWTexture3D<float> normalized16 : register(u1);" in generated_code
    assert "RWTexture2DArray<float> halfLayers : register(u2);" in generated_code
    assert "RWTexture2D<float> signedNormalized : register(u3);" in generated_code
    assert (
        "float touchR8(RWTexture2D<float> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float touchR16(RWTexture3D<float> image, int3 voxel, float value)"
        in generated_code
    )
    assert (
        "float touchR16f(RWTexture2DArray<float> image, int3 pixelLayer, float value)"
        in generated_code
    )
    assert "float oldValue = image[pixel];" in generated_code
    assert "float oldValue = image[voxel];" in generated_code
    assert "float oldValue = image[pixelLayer];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> normalized8" not in generated_code
    assert "RWTexture3D<float4> normalized16" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_narrow_integer_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<int> signed8 : register(u0);" in generated_code
    assert "RWTexture3D<int> signed16 : register(u1);" in generated_code
    assert "RWTexture2D<uint> unsigned8 : register(u2);" in generated_code
    assert "RWTexture2DArray<uint> unsigned16 : register(u3);" in generated_code
    assert (
        "int loadSigned8(RWTexture2D<int> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "int loadSigned16(RWTexture3D<int> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned8(RWTexture2D<uint> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned16(RWTexture2DArray<uint> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "int oldValue = image[pixel];" in generated_code
    assert "int oldValue = image[voxel];" in generated_code
    assert "uint oldValue = image[pixel];" in generated_code
    assert "uint oldValue = image[pixelLayer];" in generated_code
    assert "RWTexture2D<float4> signed8" not in generated_code
    assert "RWTexture3D<float4> signed16" not in generated_code
    assert "RWTexture2D<float4> unsigned8" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_integer_image_formats_use_atomic_helpers():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> unsignedCounters : register(u0);" in generated_code
    assert "RWTexture2D<int> signedCounters : register(u1);" in generated_code
    assert "RWTexture3D<uint> unsignedVolume : register(u2);" in generated_code
    assert "RWTexture2DArray<int> signedLayers : register(u3);" in generated_code
    assert (
        "uint imageAtomicAdd_uimage2D(RWTexture2D<uint> image, int2 coord, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicMin_iimage2D(RWTexture2D<int> image, int2 coord, int value)"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(RWTexture3D<uint> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicExchange_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int value)"
        in generated_code
    )
    assert "return imageAtomicAdd_uimage2D(image, pixel, value);" in generated_code
    assert "return imageAtomicMin_iimage2D(image, pixel, value);" in generated_code
    assert (
        "return imageAtomicCompSwap_uimage3D(image, voxel, expected, value);"
        in generated_code
    )
    assert (
        "return imageAtomicExchange_iimage2DArray(image, pixelLayer, value);"
        in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicMin(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code
    assert "imageAtomicExchange(image" not in generated_code


def test_directx_explicit_vector_integer_image_formats():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint4> unsignedColor : register(u0);" in generated_code
    assert "RWTexture3D<int4> signedVolume : register(u1);" in generated_code
    assert "RWTexture2DArray<uint4> unsignedLayers : register(u2);" in generated_code
    assert (
        "uint4 touchUnsigned(RWTexture2D<uint4> image, int2 pixel, uint4 value)"
        in generated_code
    )
    assert (
        "int4 touchSigned(RWTexture3D<int4> image, int3 voxel, int4 value)"
        in generated_code
    )
    assert (
        "uint4 touchLayers(RWTexture2DArray<uint4> image, int3 pixelLayer, uint4 value)"
        in generated_code
    )
    assert "uint4 oldValue = image[pixel];" in generated_code
    assert "int4 oldValue = image[voxel];" in generated_code
    assert "uint4 oldValue = image[pixelLayer];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> unsignedColor" not in generated_code
    assert "RWTexture3D<float4> signedVolume" not in generated_code


def test_directx_texture_array_resources_and_indexed_sampling():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "TextureCube envMap : register(t4);" in generated_code
    assert "SamplerState texturesSampler : register(s0);" in generated_code
    assert "SamplerState envMapSampler : register(s1);" in generated_code
    assert "textures[input.layer].Sample(texturesSampler, input.uv)" in generated_code
    assert "textures[input.layer]Sampler" not in generated_code


def test_directx_fixed_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 2;" in generated_code
    assert "Texture2D textures[6] : register(t0);" in generated_code
    assert "SamplerState samplers[6] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t6);" in generated_code
    assert "SamplerState afterTextureSampler : register(s6);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[6], SamplerState samplers[6], float2 uv)"
        in generated_code
    )
    assert "textures[LAYER].Sample(samplers[LAYER], uv)" in generated_code
    assert "textures[(1 + 2)].Sample(samplers[(1 + 2)], uv)" in generated_code
    assert "Texture2D textures[4] : register(t0);" not in generated_code
    assert "Texture2D afterTexture : register(t4);" not in generated_code


def test_directx_fixed_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE_COUNT = 2;" in generated_code
    assert "static const int TEXTURE_COUNT = (BASE_COUNT * 3);" in generated_code
    assert "Texture2D textures[TEXTURE_COUNT] : register(t0);" in generated_code
    assert "SamplerState samplers[TEXTURE_COUNT] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t6);" in generated_code
    assert "SamplerState afterTextureSampler : register(s6);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[TEXTURE_COUNT], SamplerState samplers[TEXTURE_COUNT], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "Texture2D afterTexture : register(t1);" not in generated_code


def test_directx_fixed_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D textures[(2 * 3)] : register(t0);" in generated_code
    assert "SamplerState samplers[(2 * 3)] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t6);" in generated_code
    assert "SamplerState afterTextureSampler : register(s6);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[(2 * 3)], SamplerState samplers[(2 * 3)], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "Texture2D afterTexture : register(t1);" not in generated_code
    assert "[None]" not in generated_code


def test_directx_fixed_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D textures[((2 + 1) * 2)] : register(t0);" in generated_code
    assert "SamplerState samplers[((2 + 1) * 2)] : register(s0);" in generated_code
    assert "Texture2D unaryTextures[+6] : register(t6);" in generated_code
    assert "SamplerState unaryTexturesSampler : register(s6);" in generated_code
    assert "Texture2D afterTexture : register(t12);" in generated_code
    assert "SamplerState afterTextureSampler : register(s7);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[((2 + 1) * 2)], SamplerState samplers[((2 + 1) * 2)], Texture2D unaryTextures[+6], SamplerState unaryTexturesSampler, float2 uv)"
        in generated_code
    )
    assert (
        "sampleLayer(textures, samplers, unaryTextures, unaryTexturesSampler, input.uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "unaryTextures[2].Sample(unaryTexturesSampler, uv)" in generated_code
    assert "Texture2D afterTexture : register(t6);" not in generated_code
    assert "[None]" not in generated_code


def test_directx_texture_array_explicit_sampler():
    shader = """
    shader TextureArrayShader {
        sampler2D textures[4];
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(textures[0], linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "textures[0].Sample(linearSampler, input.uv)" in generated_code
    assert "texturesSampler" not in generated_code


def test_directx_texture_array_helper_parameter_uses_implicit_sampler():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState texturesSampler, int layer, float2 uv)"
        in generated_code
    )
    assert "textures[layer].Sample(texturesSampler, uv)" in generated_code
    assert (
        "sampleLayer(textures, texturesSampler, input.layer, input.uv)"
        in generated_code
    )
    assert "textures[layer]Sampler" not in generated_code


def test_directx_texture_array_helper_parameter_uses_indexed_sampler_array():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], int layer, float2 uv)"
        in generated_code
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in generated_code
    assert "sampleLayer(textures, samplers, input.layer, input.uv)" in generated_code
    assert "texturesSampler" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_infer_helper_size():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[3] : register(t0);" in generated_code
    assert "SamplerState samplers[3] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t3);" in generated_code
    assert "SamplerState afterTextureSampler : register(s3);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[3], SamplerState samplers[3], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "textures[1].Sample(samplers[1], uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code
    assert "SamplerState samplers[] : register(s0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_infer_transitive_helper_size():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[5] : register(t0);" in generated_code
    assert "SamplerState samplers[5] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t5);" in generated_code
    assert "SamplerState afterTextureSampler : register(s5);" in generated_code
    assert (
        "float4 sampleDeep(Texture2D textures[5], SamplerState samplers[5], float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleMid(Texture2D textures[5], SamplerState samplers[5], float2 uv)"
        in generated_code
    )
    assert "textures[4].Sample(samplers[4], uv)" in generated_code
    assert "textures[1].Sample(samplers[1], uv)" in generated_code
    assert "sampleDeep(textures, samplers, uv)" in generated_code
    assert "sampleMid(textures, samplers, input.uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code
    assert "SamplerState samplers[] : register(s0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_preserve_dynamic_indexing():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t4);" in generated_code
    assert "SamplerState afterTextureSampler : register(s4);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], int layer, float2 uv)"
        in generated_code
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in generated_code
    assert "textures[3].Sample(samplers[3], uv)" in generated_code
    assert "sampleLayer(textures, samplers, input.layer, input.uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code
    assert "SamplerState samplers[] : register(s0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_ignore_unsupported_indices():
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

    dynamic_code = HLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = HLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "Texture2D textures[] : register(t0);" in dynamic_code
    assert "SamplerState samplers[] : register(s0);" in dynamic_code
    assert "Texture2D afterTexture : register(t1);" in dynamic_code
    assert "SamplerState afterTextureSampler : register(s1);" in dynamic_code
    assert (
        "float4 sampleLayer(Texture2D textures[], SamplerState samplers[], int layer, float2 uv)"
        in dynamic_code
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in dynamic_code
    assert "Texture2D textures[1] : register(t0);" not in dynamic_code
    assert "Texture2D afterTexture : register(t2);" not in dynamic_code

    assert "Texture2D textures[] : register(t0);" in negative_code
    assert "SamplerState samplers[] : register(s0);" in negative_code
    assert "Texture2D afterTexture : register(t1);" in negative_code
    assert "SamplerState afterTextureSampler : register(s1);" in negative_code
    assert "textures[-1].Sample(samplers[-1], uv)" in negative_code
    assert "Texture2D textures[0] : register(t0);" not in negative_code
    assert "Texture2D afterTexture : register(t0);" not in negative_code


def test_directx_unsized_texture_and_sampler_arrays_infer_constant_expression_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t4);" in generated_code
    assert "SamplerState afterTextureSampler : register(s4);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[(1 + 2)].Sample(samplers[(1 + 2)], uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_infer_named_constant_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE = 1;" in generated_code
    assert "static const int LAYER = (BASE + 2);" in generated_code
    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t4);" in generated_code
    assert "SamplerState afterTextureSampler : register(s4);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[LAYER].Sample(samplers[LAYER], uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_ignore_shadowed_constant_name():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 3;" in generated_code
    assert "Texture2D textures[] : register(t0);" in generated_code
    assert "SamplerState samplers[] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t1);" in generated_code
    assert "SamplerState afterTextureSampler : register(s1);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[], SamplerState samplers[], int LAYER, float2 uv)"
        in generated_code
    )
    assert "textures[LAYER].Sample(samplers[LAYER], uv)" in generated_code
    assert "Texture2D textures[4] : register(t0);" not in generated_code
    assert "Texture2D afterTexture : register(t4);" not in generated_code


def test_directx_fixed_texture_array_global_conflicts_with_fixed_parameter_size():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_global_widens_unsized_parameter_size():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 sampleUnsized(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "sampleUnsized(textures, samplers, input.uv)" in generated_code
    assert (
        "float4 sampleUnsized(Texture2D textures[3], SamplerState samplers[3], float2 uv)"
        not in generated_code
    )


def test_directx_fixed_texture_array_global_direct_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_parameter_direct_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_global_const_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_parameter_const_index_out_of_bounds_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_const_index_and_shadowing_generate():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 sampleConst(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleShadowed(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[(COUNT - 1)].Sample(samplers[(COUNT - 1)], uv)" in generated_code
    assert "textures[COUNT].Sample(samplers[COUNT], uv)" in generated_code
    assert "sampleConst(textures, samplers, input.uv)" in generated_code
    assert "sampleShadowed(textures, samplers, input.uv)" in generated_code
    assert "Texture2D textures[5]" not in generated_code
    assert "SamplerState samplers[5]" not in generated_code


def test_directx_transitive_texture_array_shadowed_const_index_stays_dynamic():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 leaf(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert (
        "float4 passThrough(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "return textures[COUNT].Sample(samplers[COUNT], uv);" in generated_code
    assert (
        "float4 sampled = textures[COUNT].Sample(samplers[COUNT], uv);"
        in generated_code
    )
    assert "return (sampled + leaf(textures, samplers, uv));" in generated_code
    assert "return passThrough(textures, samplers, input.uv);" in generated_code
    assert "Texture2D textures[5]" not in generated_code
    assert "SamplerState samplers[5]" not in generated_code


def test_directx_transitive_texture_array_unshadowed_const_index_conflict_raises():
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
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_texture_array_helper_operation_variants():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "float4 sampleOps(Texture2D textures[4], SamplerState samplers[4], int layer, float2 uv, int2 pixel)"
        in generated_code
    )
    assert "textures[layer].SampleLevel(samplers[layer], uv, 2.0)" in generated_code
    assert (
        "textures[layer].SampleGrad(samplers[layer], uv, float2(0.1), float2(0.2))"
        in generated_code
    )
    assert "textures[layer].Gather(samplers[layer], uv)" in generated_code
    assert "textures[layer].Load(int3(pixel, 0))" in generated_code
    assert (
        "sampleOps(textures, samplers, input.layer, input.uv, input.pixel)"
        in generated_code
    )
    assert "texturesSampler" not in generated_code


def test_directx_array_texture_types_and_shadow_compares():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2DArray colorArray : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "TextureCube cubeShadow : register(t2);" in generated_code
    assert "SamplerState arraySampler : register(s0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s1);" in generated_code
    assert (
        "float4 sampleArray(Texture2DArray tex, SamplerState s, float3 uvLayer, int3 pixelLayer)"
        in generated_code
    )
    assert "tex.Sample(s, uvLayer)" in generated_code
    assert "tex.SampleLevel(s, uvLayer, 1.0)" in generated_code
    assert "tex.SampleGrad(s, uvLayer, float2(0.1), float2(0.2))" in generated_code
    assert "tex.Gather(s, uvLayer)" in generated_code
    assert "tex.Load(int4(pixelLayer, 0))" in generated_code
    assert (
        "float sampleShadowArray(Texture2DArray tex, SamplerComparisonState s, float3 uvLayer, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(s, uvLayer, depth)" in generated_code
    assert (
        "float sampleCubeShadow(TextureCube tex, SamplerComparisonState s, float3 direction, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(s, direction, depth)" in generated_code


def test_directx_cube_array_and_multisample_texture_types():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "TextureCubeArray cubeArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeArrayShadow : register(t1);" in generated_code
    assert "Texture2DMS<float4> msTex : register(t2);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t3);" in generated_code
    assert "SamplerState cubeSampler : register(s0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s1);" in generated_code
    assert "msTexSampler" not in generated_code
    assert "msArraySampler" not in generated_code
    assert (
        "float4 sampleCubeArray(TextureCubeArray tex, SamplerState s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.Sample(s, cubeLayer)" in generated_code
    assert "tex.SampleLevel(s, cubeLayer, 2.0)" in generated_code
    assert (
        "float sampleCubeArrayShadow(TextureCubeArray tex, SamplerComparisonState s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(s, cubeLayer, depth)" in generated_code
    assert (
        "float4 fetchMs(Texture2DMS<float4> tex, Texture2DMSArray<float4> texArray, int2 pixel, int3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "tex.Load(pixel, sampleIndex)" in generated_code
    assert "texArray.Load(pixelLayer, sampleIndex)" in generated_code


def test_directx_cube_array_texture_grad_gather_keep_sampler_arguments():
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

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCubeArray cubeArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t1);" in generated_code
    assert "SamplerState cubeSampler : register(s0);" in generated_code
    assert "SamplerState cubeSamplers[4] : register(s1);" in generated_code
    assert (
        "float4 sampleCubeArrayOps(TextureCubeArray tex, SamplerState s, float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert "tex.SampleGrad(s, cubeLayer, ddx, ddy)" in generated_code
    assert "tex.Gather(s, cubeLayer)" in generated_code
    assert (
        "float4 sampleCubeArrayElements(TextureCubeArray cubeArrays[4], SamplerState cubeSamplers[4], float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "cubeArrays[2].SampleGrad(cubeSamplers[2], cubeLayer, ddx, ddy)"
        in generated_code
    )
    assert "cubeArrays[3].Gather(cubeSamplers[3], cubeLayer)" in generated_code
    assert (
        "sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert "cubeArraySampler" not in generated_code
    assert "cubeArraysSampler" not in generated_code


def test_directx_texture_gather_offset_variants_keep_sampler_arguments():
    shader = """
    shader GatherOffsetVariants {
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

        vec4 implicitGatherOffset(sampler2D tex, vec2 uv, ivec2 offset) {
            return textureGatherOffset(tex, uv, offset);
        }

        vec4 gatherArrayOffsets(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 offsets[4]
        ) {
            return textureGatherOffsets(layers, s, uvLayer, offsets, 2);
        }

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

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitGatherOffset(colorMap, input.uv, input.offset)
                    + gatherOps(
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
                    );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert (
        "float4 implicitGatherOffset(Texture2D tex, SamplerState texSampler, float2 uv, int2 offset)"
        in generated_code
    )
    assert "return tex.Gather(texSampler, uv, offset);" in generated_code
    assert (
        "float4 gatherArrayOffsets(Texture2DArray layers, SamplerState s, float3 uvLayer, int2 offsets[4])"
        in generated_code
    )
    assert (
        "return layers.GatherBlue(s, uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]);"
        in generated_code
    )
    assert (
        "float4 gatherOps(Texture2D tex, Texture2DArray layers, SamplerState s, float2 uv, float3 uvLayer, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert "float4 green = tex.GatherGreen(s, uv);" in generated_code
    assert (
        "float4 dynamic = (component == 0 ? tex.GatherRed(s, uv) : "
        "component == 1 ? tex.GatherGreen(s, uv) : "
        "component == 2 ? tex.GatherBlue(s, uv) : tex.GatherAlpha(s, uv));"
        in generated_code
    )
    assert "float4 offsetGather = tex.GatherAlpha(s, uv, offset);" in generated_code
    assert (
        "component == 0 ? tex.GatherRed(s, uv, offset) : "
        "component == 1 ? tex.GatherGreen(s, uv, offset) : "
        "component == 2 ? tex.GatherBlue(s, uv, offset) : "
        "tex.GatherAlpha(s, uv, offset)" in generated_code
    )
    assert (
        "component == 0 ? layers.GatherRed(s, uvLayer, offset0, offset1, offset2, offset3) : "
        "component == 1 ? layers.GatherGreen(s, uvLayer, offset0, offset1, offset2, offset3) : "
        "component == 2 ? layers.GatherBlue(s, uvLayer, offset0, offset1, offset2, offset3) : "
        "layers.GatherAlpha(s, uvLayer, offset0, offset1, offset2, offset3)"
        in generated_code
    )
    assert (
        "implicitGatherOffset(colorMap, colorMapSampler, input.uv, input.offset)"
        in generated_code
    )
    assert (
        "gatherOps(colorMap, layerMap, linearSampler, input.uv, input.uvLayer, input.offset, input.offset0, input.offset1, input.offset2, input.offset3, input.component)"
        in generated_code
    )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_directx_texture_sample_offset_variants_use_sample_offsets():
    shader = """
    shader TextureSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 implicitOffsetOps(
            sampler2D tex,
            vec2 uv,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, uv, offset);
            vec4 lodSample = textureLodOffset(tex, uv, lod, offset);
            vec4 gradSample = textureGradOffset(tex, uv, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

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
            vec4 arrayLod = textureLodOffset(layers, s, uvLayer, lod, offset);
            vec4 arrayGrad = textureGradOffset(layers, s, uvLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample + arrayLod + arrayGrad;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitOffsetOps(
                    colorMap,
                    input.uv,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + offsetOps(
                    colorMap,
                    layerMap,
                    linearSampler,
                    input.uv,
                    input.uvLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert (
        "float4 implicitOffsetOps(Texture2D tex, SamplerState texSampler, float2 uv, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 plain = tex.Sample(texSampler, uv, offset);" in generated_code
    assert (
        "float4 lodSample = tex.SampleLevel(texSampler, uv, lod, offset);"
        in generated_code
    )
    assert (
        "float4 gradSample = tex.SampleGrad(texSampler, uv, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float4 offsetOps(Texture2D tex, Texture2DArray layers, SamplerState s, float2 uv, float3 uvLayer, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 plain = tex.Sample(s, uv, offset);" in generated_code
    assert "float4 lodSample = tex.SampleLevel(s, uv, lod, offset);" in generated_code
    assert (
        "float4 gradSample = tex.SampleGrad(s, uv, ddx, ddy, offset);" in generated_code
    )
    assert (
        "float4 arrayLod = layers.SampleLevel(s, uvLayer, lod, offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layers.SampleGrad(s, uvLayer, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "implicitOffsetOps(colorMap, colorMapSampler, input.uv, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "offsetOps(colorMap, layerMap, linearSampler, input.uv, input.uvLayer, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_directx_projected_texture_variants_use_sample_projection():
    shader = """
    shader TextureProjectionVariants {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 implicitProjectedOps(sampler2D tex, vec3 uvq, vec2 ddx, vec2 ddy) {
            vec4 projected = textureProj(tex, uvq);
            vec4 projectedGrad = textureProjGrad(tex, uvq, ddx, ddy);
            return projected + projectedGrad;
        }

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

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitProjectedOps(colorMap, input.uvq, input.ddx, input.ddy)
                    + projectedOps(
                        colorMap,
                        volumeMap,
                        linearSampler,
                        input.uvq,
                        input.uvqw,
                        input.xyzq,
                        input.ddx,
                        input.ddy,
                        input.offset
                    );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert (
        "float4 implicitProjectedOps(Texture2D tex, SamplerState texSampler, float3 uvq, float2 ddx, float2 ddy)"
        in generated_code
    )
    assert (
        "float4 projected = tex.Sample(texSampler, uvq.xy / uvq.z);" in generated_code
    )
    assert (
        "float4 projectedGrad = tex.SampleGrad(texSampler, uvq.xy / uvq.z, ddx, ddy);"
        in generated_code
    )
    assert (
        "float4 projectedOps(Texture2D tex, Texture3D volume, SamplerState s, float3 uvq, float4 uvqw, float4 xyzq, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 projected = tex.Sample(s, uvq.xy / uvq.z);" in generated_code
    assert (
        "float4 projectedBias = tex.SampleBias(s, uvqw.xy / uvqw.w, 0.25);"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volume.Sample(s, xyzq.xyz / xyzq.w);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = tex.Sample(s, uvq.xy / uvq.z, offset);"
        in generated_code
    )
    assert (
        "float4 projectedOffsetBias = tex.SampleBias(s, uvq.xy / uvq.z, 0.5, offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = tex.SampleLevel(s, uvq.xy / uvq.z, 2.0);"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = tex.SampleLevel(s, uvq.xy / uvq.z, 3.0, offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = tex.SampleGrad(s, uvq.xy / uvq.z, ddx, ddy);"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = tex.SampleGrad(s, uvq.xy / uvq.z, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "implicitProjectedOps(colorMap, colorMapSampler, input.uvq, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "projectedOps(colorMap, volumeMap, linearSampler, input.uvq, input.uvqw, input.xyzq, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_directx_projected_shadow_compare_variants_use_sample_cmp_projection():
    shader = """
    shader ProjectedShadowCompareVariants {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float implicitProjectedShadow(
            sampler2DShadow tex,
            vec3 uvq,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, uvq, depth);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, uvq, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, uvq, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, uvq, depth, ddx, ddy, offset);
            return projected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

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
                float implicitValue = implicitProjectedShadow(
                    shadowMap,
                    input.uvq,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float explicitValue = projectedShadow(
                    shadowMap,
                    compareSampler,
                    input.uvq,
                    input.uvqw,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayValue = projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(implicitValue + explicitValue + arrayValue);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s1);" in generated_code
    assert (
        "float implicitProjectedShadow(Texture2D tex, SamplerComparisonState texSampler, float3 uvq, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(texSampler, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.SampleCmpLevel(texSampler, uvq.xy / uvq.z, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.SampleCmpGrad(texSampler, uvq.xy / uvq.z, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.SampleCmpGrad(texSampler, uvq.xy / uvq.z, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float projectedShadow(Texture2D tex, SamplerComparisonState s, float3 uvq, float4 uvqw, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(s, uvq.xy / uvq.z, depth);" in generated_code
    )
    assert (
        "float projectedW = tex.SampleCmp(s, uvqw.xy / uvqw.w, depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.SampleCmp(s, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.SampleCmpLevel(s, uvq.xy / uvq.z, depth, lod);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.SampleCmpLevel(s, uvq.xy / uvq.z, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.SampleCmpGrad(s, uvq.xy / uvq.z, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.SampleCmpGrad(s, uvq.xy / uvq.z, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "implicitProjectedShadow(shadowMap, shadowMapSampler, input.uvq, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedShadow(shadowMap, compareSampler, input.uvq, input.uvqw, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "float projectedArrayShadow(Texture2DArray tex, SamplerComparisonState s, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.SampleCmp(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.SampleCmpLevel(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, lod);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.SampleCmpLevel(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.SampleCmpGrad(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.SampleCmpGrad(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedArrayShadow(shadowArray, compareSampler, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj" not in generated_code


def test_directx_shadow_gather_compare_offsets_use_comparison_samplers():
    shader = """
    shader ShadowGatherCompareOffsets {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 implicitShadow(sampler2DShadow tex, vec2 uv, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, uv, depth);
            float compared = textureCompareOffset(tex, uv, depth, offset);
            return gathered + vec4(compared);
        }

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

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitShadow(shadowMap, input.uv, input.depth, input.offset)
                    + gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)
                    + gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s1);" in generated_code
    assert (
        "float4 implicitShadow(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth, int2 offset)"
        in generated_code
    )
    assert "float4 gathered = tex.GatherCmp(texSampler, uv, depth);" in generated_code
    assert (
        "float compared = tex.SampleCmp(texSampler, uv, depth, offset);"
        in generated_code
    )
    assert (
        "float4 gatherShadow(Texture2D tex, SamplerComparisonState s, float2 uv, float depth, int2 offset)"
        in generated_code
    )
    assert "float4 gathered = tex.GatherCmp(s, uv, depth);" in generated_code
    assert (
        "float4 offsetGathered = tex.GatherCmp(s, uv, depth, offset);" in generated_code
    )
    assert (
        "float offsetCompared = tex.SampleCmp(s, uv, depth, offset);" in generated_code
    )
    assert (
        "float4 gatherShadowArray(Texture2DArray tex, SamplerComparisonState s, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert "tex.GatherCmp(s, uvLayer, depth, offset)" in generated_code
    assert "tex.SampleCmp(s, uvLayer, depth, offset)" in generated_code
    assert (
        "implicitShadow(shadowMap, shadowMapSampler, input.uv, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset)"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_directx_shadow_compare_lod_grad_use_comparison_samplers():
    shader = """
    shader ShadowCompareLodGrad {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float implicitShadow(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, uv, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, uv, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, uv, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, uv, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

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

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float implicitValue = implicitShadow(
                    shadowMap,
                    input.uv,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float explicitValue = compareShadow(
                    shadowMap,
                    compareSampler,
                    input.uv,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayValue = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(implicitValue + explicitValue + arrayValue);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s1);" in generated_code
    assert (
        "float implicitShadow(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float lodValue = tex.SampleCmpLevel(texSampler, uv, depth, lod);"
        in generated_code
    )
    assert (
        "float lodOffsetValue = tex.SampleCmpLevel(texSampler, uv, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradValue = tex.SampleCmpGrad(texSampler, uv, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetValue = tex.SampleCmpGrad(texSampler, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float compareShadow(Texture2D tex, SamplerComparisonState s, float2 uv, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float lodValue = tex.SampleCmpLevel(s, uv, depth, lod);" in generated_code
    assert (
        "float lodOffsetValue = tex.SampleCmpLevel(s, uv, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradValue = tex.SampleCmpGrad(s, uv, depth, ddx, ddy);" in generated_code
    )
    assert (
        "float gradOffsetValue = tex.SampleCmpGrad(s, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float compareShadowArray(Texture2DArray tex, SamplerComparisonState s, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "tex.SampleCmpLevel(s, uvLayer, depth, lod)" in generated_code
    assert "tex.SampleCmpLevel(s, uvLayer, depth, lod, offset)" in generated_code
    assert "tex.SampleCmpGrad(s, uvLayer, depth, ddx, ddy)" in generated_code
    assert "tex.SampleCmpGrad(s, uvLayer, depth, ddx, ddy, offset)" in generated_code
    assert (
        "implicitShadow(shadowMap, shadowMapSampler, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "compareShadow(shadowMap, compareSampler, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "compareShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_directx_array_shadow_texture_resource_arrays_keep_compare_coordinates():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray shadowArrays[4] : register(t0);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t4);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float sampleArrayLayer(Texture2DArray shadowArrays[4], SamplerComparisonState shadowSamplers[4], float3 uvLayer, float depth)"
        in generated_code
    )
    assert (
        "shadowArrays[2].SampleCmp(shadowSamplers[2], uvLayer, depth)" in generated_code
    )
    assert (
        "float sampleCubeLayer(TextureCubeArray cubeShadowArrays[4], SamplerComparisonState shadowSamplers[4], float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].SampleCmp(shadowSamplers[3], cubeLayer, depth)"
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
    assert "Texture2DArray shadowArrays[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_array_shadow_texture_resource_arrays_reject_mismatched_fixed_helper_size():
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
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_texture_query_functions():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "int textureQueryLevels(Texture2D tex)" in generated_code
    assert "tex.GetDimensions(0, width, height, levels);" in generated_code
    assert "int2 textureSize(Texture2D tex, int lod)" in generated_code
    assert "tex.GetDimensions(lod, width, height, levels);" in generated_code
    assert "int3 textureSize(Texture2DArray tex, int lod)" in generated_code
    assert "tex.GetDimensions(lod, width, height, elements, levels);" in generated_code
    assert "int2 textureSize(Texture2DMS<float4> tex)" in generated_code
    assert "tex.GetDimensions(width, height, samples);" in generated_code
    assert "int2 size = textureSize(tex, 1);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert (
        "float2 lod = float2(tex.CalculateLevelOfDetailUnclamped(s, uv), tex.CalculateLevelOfDetail(s, uv));"
        in generated_code
    )
    assert "return textureSize(tex, 0);" in generated_code
    assert "return textureSize(tex);" in generated_code


def test_directx_array_shadow_texture_query_functions_prune_implicit_samplers():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray shadowArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t2);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t6);" in generated_code
    assert "SamplerComparisonState shadowArraySampler" not in generated_code
    assert "SamplerComparisonState cubeShadowArraySampler" not in generated_code
    assert "SamplerComparisonState shadowArraysSampler" not in generated_code
    assert "SamplerComparisonState cubeShadowArraysSampler" not in generated_code
    assert "int3 textureSize(Texture2DArray tex, int lod)" in generated_code
    assert "int3 textureSize(TextureCubeArray tex, int lod)" in generated_code
    assert "int textureQueryLevels(Texture2DArray tex)" in generated_code
    assert "int textureQueryLevels(TextureCubeArray tex)" in generated_code
    assert "tex.GetDimensions(lod, width, height, elements, levels);" in generated_code
    assert "tex.GetDimensions(0, width, height, elements, levels);" in generated_code
    assert "int3 query2DArrayShadow(Texture2DArray tex)" in generated_code
    assert "int3 queryCubeArrayShadow(TextureCubeArray tex)" in generated_code
    assert (
        "int3 queryArrayElements(Texture2DArray shadowArrays[4], TextureCubeArray cubeShadowArrays[4])"
        in generated_code
    )
    assert "int3 arraySize = textureSize(shadowArrays[2], 1);" in generated_code
    assert "int3 cubeSize = textureSize(cubeShadowArrays[3], 0);" in generated_code
    assert "int arrayLevels = textureQueryLevels(shadowArrays[2]);" in generated_code
    assert "int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);" in generated_code
    assert "SampleCmp" not in generated_code


def test_directx_mixed_explicit_and_implicit_texture_sampling_keeps_synthetic_sampler():
    shader = """
    shader MixedExplicitImplicitSampling {
        sampler2D colorMap;
        sampler linearSampler;

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(colorMap, linearSampler, uv) + texture(colorMap, uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert "colorMap.Sample(linearSampler, uv)" in generated_code
    assert "colorMap.Sample(colorMapSampler, uv)" in generated_code


def test_directx_array_texture_query_lod_uses_non_layer_coordinates():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray layerMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "Texture2DArray layerMaps[4] : register(t2);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t6);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState linearSamplers[4] : register(s1);" in generated_code
    assert (
        "float2 queryArrayLod(Texture2DArray tex, SamplerState s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, uvLayer.xy)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(TextureCubeArray tex, SamplerState s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, cubeLayer.xyz)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(Texture2DArray layerMaps[4], SamplerState linearSamplers[4], float3 uvLayer)"
        in generated_code
    )
    assert (
        "layerMaps[2].CalculateLevelOfDetailUnclamped(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "layerMaps[2].CalculateLevelOfDetail(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(TextureCubeArray cubeArrays[4], SamplerState linearSamplers[4], float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeArrays[3].CalculateLevelOfDetailUnclamped(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeArrays[3].CalculateLevelOfDetail(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "CalculateLevelOfDetailUnclamped(s, uvLayer)" not in generated_code
    assert "CalculateLevelOfDetailUnclamped(s, cubeLayer)" not in generated_code


def test_directx_shadow_array_texture_query_lod_uses_non_layer_coordinates():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray shadowArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t2);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t6);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState linearSamplers[4] : register(s1);" in generated_code
    assert "SamplerComparisonState linearSampler" not in generated_code
    assert "SamplerComparisonState linearSamplers" not in generated_code
    assert (
        "float2 queryArrayLod(Texture2DArray tex, SamplerState s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, uvLayer.xy)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(TextureCubeArray tex, SamplerState s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, cubeLayer.xyz)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(Texture2DArray shadowArrays[4], SamplerState linearSamplers[4], float3 uvLayer)"
        in generated_code
    )
    assert (
        "shadowArrays[2].CalculateLevelOfDetailUnclamped(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "shadowArrays[2].CalculateLevelOfDetail(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(TextureCubeArray cubeShadowArrays[4], SamplerState linearSamplers[4], float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].CalculateLevelOfDetailUnclamped(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].CalculateLevelOfDetail(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "CalculateLevelOfDetailUnclamped(s, uvLayer)" not in generated_code
    assert "CalculateLevelOfDetailUnclamped(s, cubeLayer)" not in generated_code


def test_directx_texture_operation_variants():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "colorMap.SampleLevel(colorMapSampler, input.uv, 1.0)" in generated_code
    assert (
        "colorMap.SampleGrad(colorMapSampler, input.uv, float2(0.1), float2(0.2))"
        in generated_code
    )
    assert "colorMap.Load(int3(input.pixel, 0))" in generated_code
    assert "colorMap.Gather(colorMapSampler, input.uv)" in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "textureGather(" not in generated_code


def test_directx_explicit_sampler_argument():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "colorMap.Sample(linearSampler, input.uv)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_sampler_parameter_texture_call():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "float4 sampleColor(SamplerState sampleState, float2 uv)" in generated_code
    assert "colorMap.Sample(sampleState, uv)" in generated_code
    assert "sampleColor(linearSampler, input.uv)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_texture_and_sampler_parameters():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 sampleColor(Texture2D tex, SamplerState sampleState, float2 uv)"
        in generated_code
    )
    assert "tex.Sample(sampleState, uv)" in generated_code
    assert "sampleColor(colorMap, linearSampler, input.uv)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_texture_and_sampler_parameters_transitive():
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

        vec4 sampleInput(sampler2D tex, sampler sampleState, VSOutput input) {
            return sampleColor(tex, sampleState, input.uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleInput(colorMap, linearSampler, input);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 sampleColor(Texture2D tex, SamplerState sampleState, float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleInput(Texture2D tex, SamplerState sampleState, VSOutput input)"
        in generated_code
    )
    assert "sampleColor(tex, sampleState, input.uv)" in generated_code
    assert "sampleInput(colorMap, linearSampler, input)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_implicit_sampler_for_texture_parameter():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert (
        "float4 sampleColor(Texture2D tex, SamplerState texSampler, float2 uv)"
        in generated_code
    )
    assert "tex.Sample(texSampler, uv)" in generated_code
    assert "sampleColor(colorMap, colorMapSampler, input.uv)" in generated_code


def test_directx_implicit_sampler_for_texture_parameter_transitive():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, vec2 uv) {
            return texture(tex, uv);
        }

        vec4 sampleInput(sampler2D tex, VSOutput input) {
            return sampleColor(tex, input.uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleInput(colorMap, input);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "float4 sampleColor(Texture2D tex, SamplerState texSampler, float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleInput(Texture2D tex, SamplerState texSampler, VSOutput input)"
        in generated_code
    )
    assert "sampleColor(tex, texSampler, input.uv)" in generated_code
    assert "sampleInput(colorMap, colorMapSampler, input)" in generated_code


def test_directx_shadow_texture_compare():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert (
        "shadowMap.SampleCmp(shadowMapSampler, input.uv, input.depth)" in generated_code
    )
    assert "textureCompare(" not in generated_code


def test_directx_shadow_texture_array_compare():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert (
        "shadowMaps[input.layer].SampleCmp(shadowMapsSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMaps[input.layer]Sampler" not in generated_code


def test_directx_shadow_texture_array_compare_with_indexed_sampler_array():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], int layer, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[layer].SampleCmp(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMapsSampler" not in generated_code


def test_directx_fixed_shadow_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 2;" in generated_code
    assert "Texture2D shadowMaps[6] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[6] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t6);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[6], SamplerComparisonState shadowSamplers[6], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].SampleCmp(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert (
        "shadowMaps[(1 + 2)].SampleCmp(shadowSamplers[(1 + 2)], uv, depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[4] : register(t0);" not in generated_code
    assert "Texture2D afterShadow : register(t4);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE_COUNT = 2;" in generated_code
    assert "static const int SHADOW_COUNT = (BASE_COUNT * 3);" in generated_code
    assert "Texture2D shadowMaps[SHADOW_COUNT] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowSamplers[SHADOW_COUNT] : register(s0);"
        in generated_code
    )
    assert "Texture2D afterShadow : register(t6);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[SHADOW_COUNT], SamplerComparisonState shadowSamplers[SHADOW_COUNT], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert "Texture2D afterShadow : register(t1);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[(2 * 3)] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowSamplers[(2 * 3)] : register(s0);"
        in generated_code
    )
    assert "Texture2D afterShadow : register(t6);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[(2 * 3)], SamplerComparisonState shadowSamplers[(2 * 3)], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert "Texture2D afterShadow : register(t1);" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[((2 + 1) * 2)] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowSamplers[((2 + 1) * 2)] : register(s0);"
        in generated_code
    )
    assert "Texture2D unaryShadowMaps[+6] : register(t6);" in generated_code
    assert "Texture2D afterShadow : register(t12);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[((2 + 1) * 2)], SamplerComparisonState shadowSamplers[((2 + 1) * 2)], Texture2D unaryShadowMaps[+6], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert (
        "unaryShadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert "Texture2D afterShadow : register(t6);" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_helper_size():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t4);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s4);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[3].SampleCmp(shadowSamplers[3], uv, depth)" in generated_code
    assert "shadowMaps[1].SampleCmp(shadowSamplers[1], uv, depth)" in generated_code
    assert (
        "afterShadow.SampleCmp(afterSampler, input.uv, input.depth)" in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "shadowMapsSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_transitive_helper_size():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t5);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowDeep(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float shadowMid(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[4].SampleCmp(shadowSamplers[4], uv, depth)" in generated_code
    assert "shadowMaps[1].SampleCmp(shadowSamplers[1], uv, depth)" in generated_code
    assert "shadowDeep(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth)" in generated_code
    )
    assert (
        "afterShadow.SampleCmp(afterSampler, input.uv, input.depth)" in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "shadowMapsSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_preserve_dynamic_indexing():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t4);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s4);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], int layer, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[layer].SampleCmp(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert "shadowMaps[3].SampleCmp(shadowSamplers[3], uv, depth)" in generated_code
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth)"
        in generated_code
    )
    assert (
        "afterShadow.SampleCmp(afterSampler, input.uv, input.depth)" in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "shadowMapsSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_ignore_unsupported_indices():
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

    dynamic_code = HLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = HLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "Texture2D shadowMaps[] : register(t0);" in dynamic_code
    assert "SamplerComparisonState shadowSamplers[] : register(s0);" in dynamic_code
    assert "Texture2D afterShadow : register(t1);" in dynamic_code
    assert "SamplerComparisonState afterSampler : register(s1);" in dynamic_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[], SamplerComparisonState shadowSamplers[], int layer, float2 uv, float depth)"
        in dynamic_code
    )
    assert (
        "shadowMaps[layer].SampleCmp(shadowSamplers[layer], uv, depth)" in dynamic_code
    )
    assert "Texture2D shadowMaps[1] : register(t0);" not in dynamic_code
    assert "Texture2D afterShadow : register(t2);" not in dynamic_code

    assert "Texture2D shadowMaps[] : register(t0);" in negative_code
    assert "SamplerComparisonState shadowSamplers[] : register(s0);" in negative_code
    assert "Texture2D afterShadow : register(t1);" in negative_code
    assert "SamplerComparisonState afterSampler : register(s1);" in negative_code
    assert "shadowMaps[-1].SampleCmp(shadowSamplers[-1], uv, depth)" in negative_code
    assert "Texture2D shadowMaps[0] : register(t0);" not in negative_code
    assert "Texture2D afterShadow : register(t0);" not in negative_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_constant_expression_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t5);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[(2 * 2)].SampleCmp(shadowSamplers[(2 * 2)], uv, depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_named_constant_size():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE = 2;" in generated_code
    assert "static const int LAYER = (BASE * 2);" in generated_code
    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t5);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].SampleCmp(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_array_rejects_mismatched_fixed_helper_size():
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
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_fixed_shadow_texture_array_widens_unsized_helper():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float shadowUnsized(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert (
        "shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_transitive_shadow_texture_array_shadowed_const_index_stays_dynamic():
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

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int COUNT = 4;" in generated_code
    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float leaf(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float passThrough(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "shadowMaps[COUNT].SampleCmp(shadowSamplers[COUNT], uv, depth)"
        in generated_code
    )
    assert "leaf(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "passThrough(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_transitive_shadow_texture_array_unshadowed_const_index_conflict_raises():
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
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_shadow_compare_sampler_parameter():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s0);" in generated_code
    assert (
        "float sampleShadow(SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMap.SampleCmp(compareSampler, uv, depth)" in generated_code
    assert "sampleShadow(shadowSampler, input.uv, input.depth)" in generated_code
    assert "shadowMapSampler" not in generated_code


def test_directx_shadow_texture_and_sampler_parameters():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s0);" in generated_code
    assert (
        "float sampleShadow(Texture2D tex, SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(compareSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowMap, shadowSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMapSampler" not in generated_code


def test_directx_implicit_comparison_sampler_for_shadow_texture_parameter():
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
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert (
        "float sampleShadow(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(texSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowMap, shadowMapSampler, input.uv, input.depth)"
        in generated_code
    )


def test_directx_shadow_compare_sampler_parameter_transitive():
    shader = """
    shader ShadowHelper {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float compareShadow(sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(shadowMap, compareSampler, uv, depth);
        }

        float sampleShadow(sampler compareSampler, VSOutput input) {
            return compareShadow(compareSampler, input.uv, input.depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowSampler, input);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "SamplerComparisonState shadowSampler : register(s0);" in generated_code
    assert (
        "float compareShadow(SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float sampleShadow(SamplerComparisonState compareSampler, VSOutput input)"
        in generated_code
    )
    assert "compareShadow(compareSampler, input.uv, input.depth)" in generated_code
    assert "sampleShadow(shadowSampler, input)" in generated_code


if __name__ == "__main__":
    pytest.main()
