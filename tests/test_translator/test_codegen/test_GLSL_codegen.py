import pytest
import crosstl.translator
from crosstl.translator.parser import Parser
from crosstl.translator.lexer import Lexer
from crosstl.translator.ast import (
    PrimitiveType,
    ShaderStage,
    StructMemberNode,
    StructNode,
)
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


def test_glsl_unsigned_integer_literal_suffix_codegen():
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


def test_glsl_default_float_image_scalar_and_vector_load_store():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float scalarOld = imageLoad(image, pixel).x;" in generated_code
    assert "imageStore(image, pixel, vec4((scalarOld + value)));" in generated_code
    assert "vec4 vectorOld = imageLoad(image, pixel);" in generated_code
    assert "imageStore(image, pixel, (vectorOld + value));" in generated_code
    assert "vec4 vectorOld = imageLoad(image, pixel).x;" not in generated_code
    assert "imageStore(image, pixel, vec4((vectorOld + value)));" not in generated_code


def test_glsl_rg_image_scalar_and_vector_load_store():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float oldValue = imageLoad(image, pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(image, pixel).x;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(image, pixel).xy;" in generated_code
    assert (
        "imageStore(image, pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(image, pixel, uvec4((oldValue + value), 0u, 0u));" in generated_code
    )


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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const float weights[2];" in generated_code
    assert "for (int i = 0; (i < 2); (++i))" in generated_code
    assert "for (i = 0; (i < 4); (++i))" in generated_code
    assert "for (const int fixed = 0; (fixed < 0); )" in generated_code
    assert "for (; ; )" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "for (int i = 0;;" not in generated_code
    assert "for (const int fixed = 0;;" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code
    assert "None" not in generated_code


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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 2; i < 5; ++i)" in generated_code
    assert "for (int j = 1; j <= limit; ++j)" in generated_code
    assert "total = (total + i);" in generated_code
    assert "total = (total + j);" in generated_code
    assert "return total;" in generated_code
    assert "RangeNode(" not in generated_code
    assert "ForInNode(" not in generated_code


def test_fragment_return_semantic_writes_output_variable():
    shader = """
    shader FragmentReturnSemanticSmoke {
        fragment {
            vec4 main() @ gl_FragColor {
                int total = 0;
                for i in 1..=3 {
                    total = total + i;
                }
                return vec4(float(total), 0.0, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 0) out vec4 fragColor;" in generated_code
    assert "void main()" in generated_code
    assert "for (int i = 1; i <= 3; ++i)" in generated_code
    assert "fragColor = vec4(float(total), 0.0, 0.0, 1.0);" in generated_code
    assert "return;" in generated_code
    assert "return vec4" not in generated_code
    assert "RangeNode(" not in generated_code


def test_vertex_struct_io_lowers_to_flattened_stage_variables():
    shader = """
    shader VertexStructIOSmoke {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 0) in vec3 position;" in generated_code
    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "layout(location = 5) out vec2 out_uv;" in generated_code
    assert "gl_Position = vec4(position, 1.0);" in generated_code
    assert "out_uv = uv;" in generated_code
    assert "return;" in generated_code
    assert "VSOutput output;" not in generated_code
    assert "input." not in generated_code
    assert "output." not in generated_code
    assert "return output;" not in generated_code
    assert "out vec4 position;" not in generated_code


def test_fragment_struct_input_lowers_to_flattened_stage_variables():
    shader = """
    shader FragmentStructInputSmoke {
        struct VSOutput {
            vec2 uv @ TEXCOORD0;
            vec3 normal @ NORMAL;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, input.normal.x, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(location = 5) in vec2 uv;" in generated_code
    assert "layout(location = 1) in vec3 normal;" in generated_code
    assert "layout(location = 0) out vec4 fragColor;" in generated_code
    assert "fragColor = vec4(uv, normal.x, 1.0);" in generated_code
    assert "return;" in generated_code
    assert "input." not in generated_code
    assert "out vec2 uv;" not in generated_code
    assert "return vec4" not in generated_code


def test_generate_stage_splits_combined_vertex_fragment_units():
    shader = """
    shader CombinedStageIOSmoke {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

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

    ast = crosstl.translator.parse(shader)
    generator = GLSLCodeGen()
    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")

    assert "layout(location = 0) in vec3 position;" in vertex_code
    assert "layout(location = 5) in vec2 uv;" in vertex_code
    assert "layout(location = 5) out vec2 out_uv;" in vertex_code
    assert "gl_Position = vec4(position, 1.0);" in vertex_code
    assert "out_uv = uv;" in vertex_code
    assert "fragColor" not in vertex_code
    assert "layout(location = 5) in vec2 out_uv;" not in vertex_code

    assert "layout(location = 5) in vec2 out_uv;" in fragment_code
    assert "layout(location = 0) out vec4 fragColor;" in fragment_code
    assert "fragColor = vec4(out_uv, 0.0, 1.0);" in fragment_code
    assert "layout(location = 0) in vec3 position;" not in fragment_code
    assert "layout(location = 5) out vec2 out_uv;" not in fragment_code
    assert "gl_Position" not in fragment_code
    assert "input." not in fragment_code


def test_generate_compute_stage_suppresses_graphics_io_declarations():
    shader = """
    shader MixedComputeStageIOSmoke {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct WorkItem {
            int id;
            vec4 value;
        };

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

        compute {
            void main() {
                WorkItem item;
                item.id = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    compute_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert "// Compute Shader" in compute_code
    assert "struct WorkItem" in compute_code
    assert "struct VSOutput" in compute_code
    assert "out vec4 position;" not in compute_code
    assert "out vec2 uv;" not in compute_code
    assert "layout(location = 0) in vec3 position;" not in compute_code
    assert "layout(location = 5) in vec2 out_uv;" not in compute_code
    assert "fragColor" not in compute_code
    assert "// Vertex Shader" not in compute_code
    assert "// Fragment Shader" not in compute_code


def test_compute_stage_emits_default_local_size_layout():
    shader = """
    shader ComputeLayoutSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    compute_code = GLSLCodeGen().generate_stage(ast, "compute")

    layout = "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;"
    assert layout in compute_code
    assert compute_code.index(layout) < compute_code.index("void main()")


def test_compute_stage_uses_execution_config_local_size():
    shader = """
    shader ComputeConfiguredLayoutSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    ast.stages[ShaderStage.COMPUTE].execution_config = {"local_size": (8, 4, 2)}

    compute_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;"
        in compute_code
    )


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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

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
            GLSLCodeGen().generate(crosstl.translator.parse(shader))


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


def test_opengl_cbuffer_members_are_not_global_variables():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float weights[8];
            vec3 colors[2];
        };

        compute {
            void main() {
                float value = weights[2];
                vec3 color = colors[1];
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = GLSLCodeGen().generate_stage(ast, "compute")

    assert [buffer.name for buffer in getattr(ast, "cbuffers", [])] == ["Constants"]
    assert [var.name for var in ast.global_variables] == []
    assert "layout(std140, binding = 0) uniform Constants" in generated_code
    assert "float weights[8];" in generated_code
    assert "vec3 colors[2];" in generated_code
    assert "float value = weights[2];" in generated_code
    assert "vec3 color = colors[1];" in generated_code
    assert "layout(std140, binding = 0) float weights[8];" not in generated_code


def test_opengl_duplicate_cbuffer_members_error():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float value;
        };

        compute {
            void main() {
                float x = value;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers.append(
        StructNode(
            name="Lighting",
            members=[
                StructMemberNode(name="value", member_type=PrimitiveType("float"))
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Ambiguous cbuffer member name\\(s\\) in OpenGL output: value",
    ):
        GLSLCodeGen().generate_stage(ast, "compute")


def test_opengl_duplicate_cbuffer_names_error():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float exposure;
        };

        compute {
            void main() {
                float x = exposure;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers.append(
        StructNode(
            name="Camera",
            members=[
                StructMemberNode(name="gamma", member_type=PrimitiveType("float"))
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate cbuffer name\\(s\\) in OpenGL output: Camera",
    ):
        GLSLCodeGen().generate_stage(ast, "compute")


def test_opengl_cbuffer_name_conflicts_with_struct_error():
    code = """
    shader CBufferScope {
        struct Camera {
            float exposure;
        };

        cbuffer Lighting {
            float gamma;
        };

        compute {
            void main() {
                float x = gamma;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers[0].name = "Camera"

    with pytest.raises(
        ValueError,
        match="Cbuffer name\\(s\\) conflict with existing OpenGL declaration\\(s\\): Camera",
    ):
        GLSLCodeGen().generate_stage(ast, "compute")


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
                vec4 color = texture(colorMap, uv);
                vec4 env = texture(envMap, normal);
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
    assert "in vec2 uv;" in generated_code
    assert "in vec3 normal;" in generated_code
    assert "VectorType(" not in generated_code


def test_opengl_rejects_non_resource_shadow_of_global_resource():
    shader = """
    shader ResourceShadow {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv;
        };

        vec4 shade(float colorMap, FSInput input) {
            float linearSampler = 1.0;
            return vec4(colorMap + linearSampler);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return shade(1.0, input);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Non-resource local declaration\\(s\\) shadow OpenGL global "
            "resource\\(s\\): colorMap, linearSampler"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_texture_call_with_non_resource_argument():
    shader = """
    shader InvalidTextureArgument {
        struct FSInput {
            vec2 uv;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float value = 1.0;
                return texture(value, input.uv);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture operation 'texture' requires a declared texture "
            "or image resource argument: value"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            "texture(colorMap)",
            "OpenGL texture operation 'texture' requires at least 2 "
            "argument\\(s\\), got 1",
        ),
        (
            "textureLod(colorMap, input.uv)",
            "OpenGL texture operation 'textureLod' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
        (
            "textureGrad(colorMap, input.uv, input.uv)",
            "OpenGL texture operation 'textureGrad' requires at least 4 "
            "argument\\(s\\), got 3",
        ),
        (
            "texture(colorMap, linearSampler)",
            "OpenGL texture operation 'texture' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
    ],
)
def test_opengl_rejects_texture_call_with_too_few_arguments(call, match):
    shader = f"""
    shader InvalidTextureArity {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expression", "operation", "max_count", "arg_count"),
    [
        (
            "vec4(float(textureSamples(msTex, input.layer)))",
            "textureSamples",
            1,
            2,
        ),
        (
            "vec4(float(imageSamples(msTex, input.layer)))",
            "imageSamples",
            1,
            2,
        ),
        (
            "vec4(float(textureQueryLevels(colorMap, input.layer)))",
            "textureQueryLevels",
            1,
            2,
        ),
        (
            "vec4(vec2(imageSize(colorImage, input.layer)), 0.0, 1.0)",
            "imageSize",
            1,
            2,
        ),
        (
            "vec4(textureSize(colorMap, input.layer, input.layer), 0.0, 1.0)",
            "textureSize",
            2,
            3,
        ),
    ],
)
def test_opengl_rejects_resource_query_call_with_too_many_arguments(
    return_expression, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidTextureQueryArity {{
        sampler2D colorMap;
        sampler2DMS msTex;
        image2D colorImage;

        struct FSInput {{
            ivec2 pixel;
            int layer;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expression};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("statement", "operation", "max_count", "arg_count"),
    [
        (
            "vec4 fetched = texelFetch(colorMap, input.pixel, input.layer, input.layer);",
            "texelFetch",
            3,
            4,
        ),
        (
            "vec4 fetched = texelFetchOffset(colorMap, input.pixel, input.layer, input.offset, input.layer);",
            "texelFetchOffset",
            4,
            5,
        ),
        (
            "vec4 color = imageLoad(colorImage, input.pixel, input.layer);",
            "imageLoad",
            2,
            3,
        ),
        (
            "imageStore(colorImage, input.pixel, vec4(1.0), input.layer);",
            "imageStore",
            3,
            4,
        ),
        (
            "uint oldValue = imageAtomicAdd(counterImage, input.pixel, input.amount, input.amount);",
            "imageAtomicAdd",
            3,
            4,
        ),
        (
            "uint oldValue = imageAtomicCompSwap(counterImage, input.pixel, input.amount, input.amount, input.amount);",
            "imageAtomicCompSwap",
            4,
            5,
        ),
    ],
)
def test_opengl_rejects_fixed_resource_call_with_too_many_arguments(
    statement, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidFixedResourceArity {{
        sampler2D colorMap;
        image2D colorImage;
        uimage2D counterImage;

        struct FSInput {{
            ivec2 pixel;
            ivec2 offset;
            int layer;
            uint amount;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                {statement}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expression", "operation", "max_count", "arg_count"),
    [
        (
            "texture(colorMap, input.uv, input.bias, input.bias)",
            "texture",
            3,
            4,
        ),
        (
            "textureLod(colorMap, input.uv, input.bias, input.bias)",
            "textureLod",
            3,
            4,
        ),
        (
            "textureGrad(colorMap, input.uv, input.ddx, input.ddy, input.bias)",
            "textureGrad",
            4,
            5,
        ),
        (
            "textureOffset(colorMap, input.uv, input.offset, input.bias, input.bias)",
            "textureOffset",
            4,
            5,
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depth, input.bias))",
            "textureCompare",
            4,
            5,
        ),
        (
            "textureGather(colorMap, input.uv, input.component, input.component)",
            "textureGather",
            3,
            4,
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset, input.component)",
            "textureGatherCompareOffset",
            5,
            6,
        ),
    ],
)
def test_opengl_rejects_texture_sampling_call_with_too_many_arguments(
    return_expression, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidTextureSamplingArity {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec2 ddx;
            vec2 ddy;
            ivec2 offset;
            float bias;
            float depth;
            int component;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expression};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_texture_gather_offsets_with_ambiguous_argument_count():
    shader = """
    shader InvalidTextureGatherOffsetsArity {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            ivec2 offset;
            int component;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureGatherOffsets(
                    colorMap,
                    input.uv,
                    input.offset,
                    input.offset,
                    input.component
                );
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture operation 'textureGatherOffsets' accepts "
            "3, 4, 6, 7 argument\\(s\\), got 5"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("statement", "operation"),
    [
        ("vec4 color = imageLoad(colorMap, input.pixel);", "imageLoad"),
        ("imageStore(colorMap, input.pixel, vec4(1.0));", "imageStore"),
        ("ivec2 size = imageSize(colorMap);", "imageSize"),
        (
            "uint oldValue = imageAtomicAdd(colorMap, input.pixel, input.amount);",
            "imageAtomicAdd",
        ),
    ],
)
def test_opengl_rejects_image_call_with_sampled_texture_argument(statement, operation):
    shader = f"""
    shader InvalidImageResource {{
        sampler2D colorMap;

        struct FSInput {{
            ivec2 pixel;
            uint amount;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                {statement}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL image operation '{operation}' requires a storage "
            "image resource argument: colorMap"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("helper", "helper_call", "match"),
    [
        (
            "vec4 misuse(sampler sampleState, vec2 uv) { return texture(sampleState, uv); }",
            "misuse(linearSampler, input.uv)",
            "OpenGL texture operation 'texture' requires a declared texture "
            "or image resource argument: sampleState",
        ),
        (
            "vec4 misuse(sampler sampleState, ivec2 pixel) { return imageLoad(sampleState, pixel); }",
            "misuse(linearSampler, input.pixel)",
            "OpenGL image operation 'imageLoad' requires a storage image "
            "resource argument: sampleState",
        ),
    ],
)
def test_opengl_rejects_sampler_parameter_as_texture_or_image_operand(
    helper, helper_call, match
):
    shader = f"""
    shader InvalidSamplerOperand {{
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            ivec2 pixel;
        }};

        {helper}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {helper_call};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("texelFetch(colorMap, input.uv, 0)", "texelFetch"),
        ("imageLoad(colorImage, input.uv)", "imageLoad"),
    ],
)
def test_opengl_rejects_float_coordinate_for_integer_resource_operation(
    call, operation
):
    shader = f"""
    shader InvalidResourceCoordinate {{
        sampler2D colorMap;
        image2D colorImage;

        struct FSInput {{
            vec2 uv;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires an integer "
            "coordinate argument: input.uv has type vec2"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("texelFetch(arrayMap, input.pixel2, 0)", "texelFetch", 3),
        ("imageLoad(volumeImage, input.pixel2)", "imageLoad", 3),
        ("imageLoad(colorImage, input.pixel3)", "imageLoad", 2),
    ],
)
def test_opengl_rejects_wrong_coordinate_dimension_for_resource_operation(
    call, operation, dimension
):
    shader = f"""
    shader InvalidResourceCoordinateDimension {{
        sampler2DArray arrayMap;
        image2D colorImage;
        image3D volumeImage;

        struct FSInput {{
            ivec2 pixel2;
            ivec3 pixel3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a "
            f"{dimension}D integer coordinate"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("textureOffset(arrayMap, input.uvLayer, input.offset3)", "textureOffset"),
        (
            "texelFetchOffset(arrayMap, input.pixelLayer, 0, input.offset3)",
            "texelFetchOffset",
        ),
    ],
)
def test_opengl_rejects_wrong_offset_dimension_for_resource_operation(call, operation):
    shader = f"""
    shader InvalidResourceOffsetDimension {{
        sampler2DArray arrayMap;

        struct FSInput {{
            vec3 uvLayer;
            ivec3 pixelLayer;
            ivec3 offset3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a " "2D integer offset"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            "textureProjOffset(arrayMap, input.projLayer, input.offset3)",
            "textureProjOffset",
        ),
        (
            "textureGatherOffset(arrayMap, input.uvLayer, input.offset3)",
            "textureGatherOffset",
        ),
        (
            "textureGatherOffsets(arrayMap, input.uvLayer, input.offset3, input.offset3, input.offset3, input.offset3)",
            "textureGatherOffsets",
        ),
        (
            "vec4(textureCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset3))",
            "textureCompareOffset",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.ddx, input.ddy, input.offset3))",
            "textureCompareProjGradOffset",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset3)",
            "textureGatherCompareOffset",
        ),
    ],
)
def test_opengl_rejects_wrong_offset_dimension_for_extended_resource_operation(
    call, operation
):
    shader = f"""
    shader InvalidExtendedResourceOffsetDimension {{
        sampler2DArray arrayMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 uvLayer;
            vec3 projCoord;
            vec4 projLayer;
            vec2 ddx;
            vec2 ddy;
            float depth;
            ivec3 offset3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a " "2D integer offset"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_float_offset_for_resource_operation():
    shader = """
    shader InvalidResourceOffsetType {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            vec2 offset;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureOffset(colorMap, input.uv, input.offset);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL resource operation 'textureOffset' requires an integer "
            "offset argument"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            "textureGatherOffset(colorMap, input.uv, input.offset)",
            "textureGatherOffset",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset)",
            "textureGatherCompareOffset",
        ),
    ],
)
def test_opengl_rejects_float_offset_for_extended_resource_operation(call, operation):
    shader = f"""
    shader InvalidExtendedResourceOffsetType {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec2 offset;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires an integer "
            "offset argument"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("textureGrad(colorMap, input.uv, input.ddx3, input.ddy3)", "textureGrad", 2),
        (
            "textureGrad(cubeMap, input.direction, input.ddx2, input.ddy2)",
            "textureGrad",
            3,
        ),
        (
            "textureGradOffset(colorMap, input.uv, input.ddx3, input.ddy3, input.offset2)",
            "textureGradOffset",
            2,
        ),
        (
            "textureProjGrad(colorMap, input.projCoord, input.ddx3, input.ddy3)",
            "textureProjGrad",
            2,
        ),
        (
            "vec4(textureCompareGrad(shadowMap, compareSampler, input.uv, input.depth, input.ddx3, input.ddy3))",
            "textureCompareGrad",
            2,
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.ddx3, input.ddy3, input.offset2))",
            "textureCompareProjGradOffset",
            2,
        ),
    ],
)
def test_opengl_rejects_wrong_gradient_dimension_for_resource_operation(
    call, operation, dimension
):
    shader = f"""
    shader InvalidResourceGradientDimension {{
        sampler2D colorMap;
        samplerCube cubeMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 direction;
            vec3 projCoord;
            vec2 ddx2;
            vec2 ddy2;
            vec3 ddx3;
            vec3 ddy3;
            ivec2 offset2;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a "
            f"{dimension}D floating gradient"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_integer_gradient_for_resource_operation():
    shader = """
    shader InvalidResourceGradientType {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureGrad(colorMap, input.uv, input.pixel, input.pixel);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL resource operation 'textureGrad' requires a floating "
            "gradient argument"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "textureLod(colorMap, linearSampler, input.uv, input.lodVec)",
            "textureLod",
            "vec2",
        ),
        (
            "textureLodOffset(colorMap, linearSampler, input.uv, input.lodVec, input.offset)",
            "textureLodOffset",
            "vec2",
        ),
        (
            "textureProjLod(colorMap, linearSampler, input.projCoord, input.lodVec)",
            "textureProjLod",
            "vec2",
        ),
        (
            "vec4(textureCompareLod(shadowMap, compareSampler, input.uv, input.depth, input.lodVec))",
            "textureCompareLod",
            "vec2",
        ),
        (
            "vec4(textureCompareProjLodOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.lodVec, input.offset))",
            "textureCompareProjLodOffset",
            "vec2",
        ),
        (
            "textureLod(colorMap, linearSampler, input.uv, colorMap)",
            "textureLod",
            "sampler2D",
        ),
    ],
)
def test_opengl_rejects_non_scalar_numeric_lod_argument(call, operation, type_name):
    shader = f"""
    shader InvalidTextureLodArgument {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler linearSampler;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 lodVec;
            ivec2 offset;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture LOD operation '{operation}' requires a scalar "
            f"numeric lod argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texelFetch(colorMap, input.pixel, input.floatLevel)",
            "texelFetch",
            "float",
        ),
        (
            "texelFetchOffset(colorMap, input.pixel, input.levelVec, input.offset)",
            "texelFetchOffset",
            "ivec2",
        ),
        (
            "vec4(textureSize(colorMap, input.floatLevel), 0.0, 1.0)",
            "textureSize",
            "float",
        ),
        (
            "texelFetch(colorMap, input.pixel, colorMap)",
            "texelFetch",
            "sampler2D",
        ),
    ],
)
def test_opengl_rejects_non_scalar_integer_mip_level_argument(
    return_expr, operation, type_name
):
    shader = f"""
    shader InvalidTextureMipLevelArgument {{
        sampler2D colorMap;

        struct FSInput {{
            ivec2 pixel;
            ivec2 offset;
            ivec2 levelVec;
            float floatLevel;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL resource operation '{operation}' requires a scalar integer "
            f"mip/sample level argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "type_name"),
    [
        ("texelFetch(msTex, input.pixel, input.floatSample)", "float"),
        ("texelFetch(msArray, input.pixelLayer, input.sampleVec)", "ivec2"),
        ("texelFetch(msTex, input.pixel, msTex)", "sampler2DMS"),
    ],
)
def test_opengl_rejects_non_scalar_integer_multisample_sample_index(
    return_expr, type_name
):
    shader = f"""
    shader InvalidMultisampleTexelFetchSampleIndex {{
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {{
            ivec2 pixel;
            ivec3 pixelLayer;
            ivec2 sampleVec;
            float floatSample;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL multisample texel fetch operation 'texelFetch' requires a "
            f"scalar integer sample index argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "helper", "operation", "type_name"),
    [
        (
            "textureGather(colorMap, linearSampler, input.uv, input.floatComponent)",
            "",
            "textureGather",
            "float",
        ),
        (
            "textureGatherOffset(colorMap, linearSampler, input.uv, input.offset, input.componentVec)",
            "",
            "textureGatherOffset",
            "ivec2",
        ),
        (
            "textureGatherOffsets(colorMap, linearSampler, input.uv, input.offset, input.offset, input.offset, input.offset, colorMap)",
            "",
            "textureGatherOffsets",
            "sampler2D",
        ),
        (
            "vec4(1.0)",
            "vec4 invalidArrayGather(sampler2D tex, sampler s, vec2 uv, ivec2 offsets[4], float component) { return textureGatherOffsets(tex, s, uv, offsets, component); }",
            "textureGatherOffsets",
            "float",
        ),
    ],
)
def test_opengl_rejects_non_scalar_integer_gather_component_argument(
    return_expr, helper, operation, type_name
):
    shader = f"""
    shader InvalidTextureGatherComponentArgument {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            ivec2 offset;
            ivec2 componentVec;
            float floatComponent;
        }};

        {helper}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture gather operation '{operation}' requires a scalar "
            f"integer component argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texture(colorMap, linearSampler, input.uv, input.biasVec)",
            "texture",
            "vec2",
        ),
        (
            "textureOffset(colorMap, linearSampler, input.uv, input.offset, input.biasVec)",
            "textureOffset",
            "vec2",
        ),
        (
            "textureProj(colorMap, linearSampler, input.projCoord, input.biasVec)",
            "textureProj",
            "vec2",
        ),
        (
            "textureProjOffset(colorMap, linearSampler, input.projCoord, input.offset, colorMap)",
            "textureProjOffset",
            "sampler2D",
        ),
    ],
)
def test_opengl_rejects_non_scalar_numeric_bias_argument(
    return_expr, operation, type_name
):
    shader = f"""
    shader InvalidTextureBiasArgument {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 biasVec;
            ivec2 offset;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture bias operation '{operation}' requires a scalar "
            f"numeric bias argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_rejects_non_floating_query_lod_coordinate():
    shader = """
    shader InvalidTextureQueryLodCoordinate {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            ivec2 pixel;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(textureQueryLod(colorMap, linearSampler, input.pixel), 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture query operation 'textureQueryLod' requires a "
            "floating coordinate argument: .* has type ivec2"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("query_call", "dimension", "resource_type", "type_name"),
    [
        (
            "textureQueryLod(layerMap, linearSampler, input.uv)",
            3,
            "sampler2DArray",
            "vec2",
        ),
        (
            "textureQueryLod(cubeMap, linearSampler, input.uv)",
            3,
            "samplerCube",
            "vec2",
        ),
        (
            "textureQueryLod(cubeArray, linearSampler, input.direction)",
            4,
            "samplerCubeArray",
            "vec3",
        ),
    ],
)
def test_opengl_rejects_wrong_query_lod_coordinate_dimension(
    query_call, dimension, resource_type, type_name
):
    shader = f"""
    shader InvalidTextureQueryLodCoordinateDimension {{
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            vec3 direction;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return vec4({query_call}, 0.0, 1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "OpenGL texture query operation 'textureQueryLod' requires a "
            f"{dimension}D floating coordinate for {resource_type}: "
            f".* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depthVec))",
            "textureCompare",
            "vec2",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depthVec, input.ddx, input.ddy, input.offset))",
            "textureCompareProjGradOffset",
            "vec2",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depthVec, input.offset)",
            "textureGatherCompareOffset",
            "vec2",
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.layer))",
            "textureCompare",
            "int",
        ),
    ],
)
def test_opengl_rejects_non_scalar_float_compare_argument(call, operation, type_name):
    shader = f"""
    shader InvalidTextureCompareArgument {{
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 depthVec;
            vec2 ddx;
            vec2 ddy;
            ivec2 offset;
            int layer;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"OpenGL texture compare operation '{operation}' requires a scalar "
            f"floating compare argument: .* has type {type_name}"
        ),
    ):
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


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
                vec4 color = texture(textures[layer], uv);
                vec4 env = texture(envMap, normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform samplerCube envMap;" in generated_code
    assert "texture(textures[layer], uv)" in generated_code
    assert "texture(envMap, normal)" in generated_code


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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, uv)" in generated_code
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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, uv)" in generated_code
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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[(2 * 3)];" in generated_code
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[(2 * 3)], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
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
                return sampleLayer(textures, samplers, unaryTextures, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, unaryTextures, uv)" in generated_code
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
                return sampleLayer(textures, layer, uv);
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
    assert "sampleLayer(textures, layer, uv)" in generated_code


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
                return sampleLayer(textures, samplers, layer, uv);
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
    assert "sampleLayer(textures, layer, uv)" in generated_code
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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, uv)" in generated_code
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
                return sampleMid(textures, samplers, uv) + texture(afterTexture, uv);
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
    assert "sampleMid(textures, uv)" in generated_code
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
                return sampleLayer(textures, samplers, layer, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, layer, uv)" in generated_code
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
                return sampleLayer(textures, samplers, layer, uv) + texture(afterTexture, uv);
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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in generated_code
    assert "vec4 sampleLayer(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[(1 + 2)], uv)" in generated_code
    assert "sampleLayer(textures, uv)" in generated_code
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
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, uv)" in generated_code
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
                return sampleLayer(textures, samplers, layer, uv) + texture(afterTexture, uv);
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
    assert "sampleLayer(textures, layer, uv)" in generated_code
    assert "layout(binding = 0) uniform sampler2D textures[4];" not in generated_code
    assert "layout(binding = 4) uniform sampler2D afterTexture;" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[LAYER]" not in generated_code


def test_opengl_fixed_texture_array_global_conflicts_with_fixed_parameter_size():
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
                return sampleThree(textures, samplers, uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_global_widens_unsized_parameter_size():
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
                return sampleUnsized(textures, samplers, uv);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "vec4 sampleUnsized(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[2], uv)" in generated_code
    assert "sampleUnsized(textures, uv)" in generated_code
    assert "vec4 sampleUnsized(sampler2D textures[3], vec2 uv)" not in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[2]" not in generated_code


def test_opengl_fixed_texture_array_global_direct_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_parameter_direct_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_global_const_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_parameter_const_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_opengl_fixed_texture_array_const_index_and_shadowing_generate():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "vec4 sampleConst(sampler2D textures[4], vec2 uv)" in generated_code
    assert "vec4 sampleShadowed(sampler2D textures[4], vec2 uv)" in generated_code
    assert "texture(textures[(COUNT - 1)], uv)" in generated_code
    assert "texture(textures[COUNT], uv)" in generated_code
    assert "sampleConst(textures, uv)" in generated_code
    assert "sampleShadowed(textures, uv)" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[COUNT]" not in generated_code
    assert "samplers[(COUNT - 1)]" not in generated_code
    assert "sampler2D textures[5]" not in generated_code


def test_opengl_transitive_texture_array_shadowed_const_index_stays_dynamic():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated_code
    assert "vec4 leaf(sampler2D textures[4], vec2 uv)" in generated_code
    assert "vec4 passThrough(sampler2D textures[4], vec2 uv)" in generated_code
    assert "return texture(textures[COUNT], uv);" in generated_code
    assert "vec4 sampled = texture(textures[COUNT], uv);" in generated_code
    assert "return (sampled + leaf(textures, uv));" in generated_code
    assert "fragColor = passThrough(textures, uv);" in generated_code
    assert "sampler samplers" not in generated_code
    assert "samplers[COUNT]" not in generated_code
    assert "sampler2D textures[5]" not in generated_code


def test_opengl_transitive_texture_array_unshadowed_const_index_conflict_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


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
                return sampleOps(textures, samplers, layer, uv, pixel);
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
    assert "sampleOps(textures, layer, uv, pixel)" in generated_code
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
                float shadow = sampleShadowArray(shadowArray, shadowSampler, uvLayer, depth);
                float cube = sampleCubeShadow(cubeShadow, shadowSampler, direction, depth);
                return sampleArray(colorArray, arraySampler, uvLayer, pixelLayer) * shadow * cube;
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
                float shadow = sampleCubeArrayShadow(cubeArrayShadow, shadowSampler, cubeLayer, depth);
                return sampleCubeArray(cubeArray, cubeSampler, cubeLayer) * shadow + fetchMs(msTex, msArray, pixel, pixelLayer, sampleIndex);
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


def test_opengl_cube_array_texture_grad_gather_filter_sampler_arguments():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert (
        "vec4 sampleCubeArrayOps(samplerCubeArray tex, vec4 cubeLayer, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "textureGrad(tex, cubeLayer, ddx, ddy)" in generated_code
    assert "textureGather(tex, cubeLayer)" in generated_code
    assert (
        "vec4 sampleCubeArrayElements(samplerCubeArray cubeArrays[4], vec4 cubeLayer, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "textureGrad(cubeArrays[2], cubeLayer, ddx, ddy)" in generated_code
    assert "textureGather(cubeArrays[3], cubeLayer)" in generated_code
    assert "sampleCubeArrayOps(cubeArray, cubeLayer, ddx, ddy)" in generated_code
    assert "sampleCubeArrayElements(cubeArrays, cubeLayer, ddx, ddy)" in generated_code
    assert "cubeSampler" not in generated_code
    assert "cubeSamplers" not in generated_code


def test_opengl_texture_gather_offset_variants_filter_sampler_arguments():
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
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert (
        "vec4 gatherOps(sampler2D tex, sampler2DArray layers, vec2 uv, vec3 uvLayer, ivec2 offset, ivec2 offset0, ivec2 offset1, ivec2 offset2, ivec2 offset3, int component)"
        in generated_code
    )
    assert "textureGather(tex, uv, 1)" in generated_code
    assert (
        "component == 0 ? textureGather(tex, uv, 0) : "
        "component == 1 ? textureGather(tex, uv, 1) : "
        "component == 2 ? textureGather(tex, uv, 2) : textureGather(tex, uv, 3)"
        in generated_code
    )
    assert "textureGatherOffset(tex, uv, offset, 3)" in generated_code
    assert (
        "component == 0 ? textureGatherOffset(tex, uv, offset, 0) : "
        "component == 1 ? textureGatherOffset(tex, uv, offset, 1) : "
        "component == 2 ? textureGatherOffset(tex, uv, offset, 2) : "
        "textureGatherOffset(tex, uv, offset, 3)" in generated_code
    )
    assert (
        "component == 0 ? vec4(textureGatherOffset(layers, uvLayer, offset0, 0).x, "
        "textureGatherOffset(layers, uvLayer, offset1, 0).y, "
        "textureGatherOffset(layers, uvLayer, offset2, 0).z, "
        "textureGatherOffset(layers, uvLayer, offset3, 0).w)" in generated_code
    )
    assert "textureGather(tex, uv, component)" not in generated_code
    assert "textureGatherOffset(tex, uv, offset, component)" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert (
        "gatherOps(colorMap, layerMap, uv, uvLayer, offset, offset0, offset1, offset2, offset3, component)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", s, uv" not in generated_code
    assert ", s, uvLayer" not in generated_code


def test_opengl_texture_gather_offsets_mix_literal_and_dynamic_offsets():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert (
        "vec4 gatherOps(sampler2DArray layers, vec3 uvLayer, ivec2 dynamic0, ivec2 dynamic1, int component)"
        in generated_code
    )
    assert "textureGatherOffsets(" not in generated_code
    assert "linearSampler" not in generated_code
    assert ", s, uvLayer" not in generated_code
    assert generated_code.count("component == 0 ? vec4(") == 1
    assert generated_code.count("component == 1 ? vec4(") == 1
    assert generated_code.count("component == 2 ? vec4(") == 1
    for component in range(4):
        assert (
            f"textureGatherOffset(layers, uvLayer, ivec2((-1), 0), {component}).x"
            in generated_code
        )
        assert (
            f"textureGatherOffset(layers, uvLayer, dynamic0, {component}).y"
            in generated_code
        )
        assert (
            f"textureGatherOffset(layers, uvLayer, ivec2(1, (-1)), {component}).z"
            in generated_code
        )
        assert (
            f"textureGatherOffset(layers, uvLayer, dynamic1, {component}).w"
            in generated_code
        )
    assert (
        "gatherOps(layerMap, uvLayer, dynamic0, dynamic1, component)" in generated_code
    )


def test_opengl_direct_stage_gather_offsets_use_input_members():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert (
        "vec4 dynamic = (component == 0 ? textureGather(colorMap, uv, 0) : "
        "component == 1 ? textureGather(colorMap, uv, 1) : "
        "component == 2 ? textureGather(colorMap, uv, 2) : "
        "textureGather(colorMap, uv, 3));" in generated_code
    )
    assert (
        "component == 0 ? textureGatherOffset(colorMap, uv, offset, 0) : "
        "component == 1 ? textureGatherOffset(colorMap, uv, offset, 1) : "
        "component == 2 ? textureGatherOffset(colorMap, uv, offset, 2) : "
        "textureGatherOffset(colorMap, uv, offset, 3)" in generated_code
    )
    for component in range(4):
        assert (
            f"textureGatherOffset(layerMap, uvLayer, offset0, {component}).x"
            in generated_code
        )
        assert (
            f"textureGatherOffset(layerMap, uvLayer, offset1, {component}).y"
            in generated_code
        )
        assert (
            f"textureGatherOffset(layerMap, uvLayer, offset2, {component}).z"
            in generated_code
        )
        assert (
            f"textureGatherOffset(layerMap, uvLayer, offset3, {component}).w"
            in generated_code
        )
    assert "textureGather(colorMap, uv, component)" not in generated_code
    assert "textureGatherOffset(colorMap, uv, offset, component)" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "input.uv" not in generated_code
    assert "input.offset" not in generated_code


def test_opengl_cube_texture_gather_dynamic_components_use_constant_branches():
    shader = """
    shader CubeGatherDynamicComponent {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler cubeSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherCubeOps(
            samplerCube cubeMap,
            samplerCubeArray cubeArray,
            sampler s,
            vec3 direction,
            vec4 cubeLayer,
            int component
        ) {
            vec4 cubeGather = textureGather(cubeMap, s, direction, component);
            vec4 cubeArrayGather = textureGather(cubeArray, s, cubeLayer, component);
            return cubeGather + cubeArrayGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCubeOps(
                    cubeMap,
                    cubeArray,
                    cubeSampler,
                    input.direction,
                    input.cubeLayer,
                    input.component
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 gatherCubeOps(samplerCube cubeMap, samplerCubeArray cubeArray, vec3 direction, vec4 cubeLayer, int component)"
        in generated_code
    )
    assert (
        "vec4 cubeGather = (component == 0 ? textureGather(cubeMap, direction, 0) : "
        "component == 1 ? textureGather(cubeMap, direction, 1) : "
        "component == 2 ? textureGather(cubeMap, direction, 2) : "
        "textureGather(cubeMap, direction, 3));" in generated_code
    )
    assert (
        "vec4 cubeArrayGather = (component == 0 ? textureGather(cubeArray, cubeLayer, 0) : "
        "component == 1 ? textureGather(cubeArray, cubeLayer, 1) : "
        "component == 2 ? textureGather(cubeArray, cubeLayer, 2) : "
        "textureGather(cubeArray, cubeLayer, 3));" in generated_code
    )
    assert (
        "gatherCubeOps(cubeMap, cubeArray, direction, cubeLayer, component)"
        in generated_code
    )
    assert "cubeSampler" not in generated_code
    assert "textureGather(cubeMap, direction, component)" not in generated_code
    assert "textureGather(cubeArray, cubeLayer, component)" not in generated_code


def test_opengl_cube_texture_gather_offsets_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    offset_diagnostic = (
        "/* unsupported GLSL texture gather: textureGatherOffset "
        "offsets require 2D or 2D-array textures */ vec4(0.0)"
    )
    offsets_diagnostic = (
        "/* unsupported GLSL texture gather: textureGatherOffsets "
        "offsets require 2D or 2D-array textures */ vec4(0.0)"
    )
    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 gatherCube(samplerCube tex, vec3 direction, ivec2 offset, ivec2 offset0, ivec2 offset1, ivec2 offset2, ivec2 offset3, int component)"
        in generated_code
    )
    assert (
        "vec4 gatherCubeArray(samplerCubeArray tex, vec4 cubeLayer, ivec2 offset, ivec2 offset0, ivec2 offset1, ivec2 offset2, ivec2 offset3, int component)"
        in generated_code
    )
    assert generated_code.count(offset_diagnostic) == 4
    assert generated_code.count(offsets_diagnostic) == 4
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "cubeSampler" not in generated_code


def test_opengl_unsupported_dimension_texture_gather_emits_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ vec4(0.0)"
    )
    assert "layout(binding = 0) uniform sampler1D lineMap;" in generated_code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in generated_code
    assert "vec4 gatherLine(sampler1D tex, float u, int component)" in generated_code
    assert (
        "vec4 gatherVolume(sampler3D tex, vec3 volumeUv, int component)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 6
    assert "textureGather(" not in generated_code
    assert "linearSampler" not in generated_code


def test_opengl_texture_lod_grad_offsets_filter_sampler_arguments():
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
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "vec4 offsetOps(sampler2D tex, sampler2DArray layers, vec2 uv, vec3 uvLayer, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "textureOffset(tex, uv, offset)" in generated_code
    assert "textureLodOffset(tex, uv, lod, offset)" in generated_code
    assert "textureGradOffset(tex, uv, ddx, ddy, offset)" in generated_code
    assert "textureLodOffset(layers, uvLayer, lod, offset)" in generated_code
    assert "textureGradOffset(layers, uvLayer, ddx, ddy, offset)" in generated_code
    assert (
        "offsetOps(colorMap, layerMap, uv, uvLayer, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", s, uv" not in generated_code
    assert ", s, uvLayer" not in generated_code


def test_opengl_cube_texture_sample_offsets_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 2) uniform samplerCubeShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 3) uniform samplerCubeArrayShadow shadowArray;"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert (
        "vec4 cubeOffsets(samplerCube tex, vec3 direction, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 cubeArrayOffsets(samplerCubeArray tex, vec4 cubeLayer, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture offset: textureOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture offset: textureLodOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture offset: textureGradOffset offsets require 1D, 2D, 2D-array, 3D, or planar shadow samplers */ vec4(0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareOffset offsets require 2D or 2D-array shadow samplers */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareLodOffset explicit LOD offsets require 2D shadow samplers */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareGradOffset explicit gradient offsets require 2D or 2D-array shadow samplers */ 0.0"
        )
        == 2
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_opengl_direct_stage_sample_offsets_and_texel_fetch_offset_use_input_members():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "vec4 plain = textureOffset(colorMap, uv, offset);" in generated_code
    assert (
        "vec4 lodSample = textureLodOffset(colorMap, uv, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 gradSample = textureGradOffset(colorMap, uv, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "vec4 arrayPlain = textureOffset(layerMap, uvLayer, offset);" in generated_code
    )
    assert (
        "vec4 arrayLod = textureLodOffset(layerMap, uvLayer, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 arrayGrad = textureGradOffset(layerMap, uvLayer, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "vec4 fetched = texelFetchOffset(colorMap, pixel, int(lod), offset);"
        in generated_code
    )
    assert (
        "vec4 fetchedLayer = texelFetchOffset(layerMap, pixelLayer, int(lod), offset);"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert "input.uv" not in generated_code
    assert "input.pixel" not in generated_code


def test_opengl_projected_texture_variants_filter_sampler_arguments():
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
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "vec4 projectedOps(sampler2D tex, sampler3D volume, vec3 uvq, vec4 uvqw, vec4 xyzq, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "textureProj(tex, uvq)" in generated_code
    assert "textureProj(tex, uvqw, 0.25)" in generated_code
    assert "textureProj(volume, xyzq)" in generated_code
    assert "textureProjOffset(tex, uvq, offset)" in generated_code
    assert "textureProjOffset(tex, uvq, offset, 0.5)" in generated_code
    assert "textureProjLod(tex, uvq, 2.0)" in generated_code
    assert "textureProjLodOffset(tex, uvq, 3.0, offset)" in generated_code
    assert "textureProjGrad(tex, uvq, ddx, ddy)" in generated_code
    assert "textureProjGradOffset(tex, uvq, ddx, ddy, offset)" in generated_code
    assert (
        "projectedOps(colorMap, volumeMap, uvq, uvqw, xyzq, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", s, uvq" not in generated_code
    assert ", s, uvqw" not in generated_code
    assert ", s, xyzq" not in generated_code


def test_opengl_direct_projected_texture_stage_input_members():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in generated_code
    assert "vec4 projected = textureProj(colorMap, uvq);" in generated_code
    assert "vec4 projectedBias = textureProj(colorMap, uvqw, 0.25);" in generated_code
    assert "vec4 volumeProjected = textureProj(volumeMap, xyzq);" in generated_code
    assert (
        "vec4 projectedOffset = textureProjOffset(colorMap, uvq, offset);"
        in generated_code
    )
    assert "vec4 projectedLod = textureProjLod(colorMap, uvq, lod);" in generated_code
    assert (
        "vec4 projectedLodOffset = textureProjLodOffset(colorMap, uvq, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 projectedGrad = textureProjGrad(colorMap, uvq, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 projectedGradOffset = textureProjGradOffset(colorMap, uvq, ddx, ddy, offset);"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", input." not in generated_code


def test_opengl_direct_projected_array_texture_stage_input_members():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "vec4 projected = textureProj(layerMap, uvLayerQ);" in generated_code
    assert (
        "vec4 projectedOffset = textureProjOffset(layerMap, uvLayerQ, offset);"
        in generated_code
    )
    assert (
        "vec4 projectedLod = textureProjLod(layerMap, uvLayerQ, lod);" in generated_code
    )
    assert (
        "vec4 projectedLodOffset = textureProjLodOffset(layerMap, uvLayerQ, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 projectedGrad = textureProjGrad(layerMap, uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 projectedGradOffset = textureProjGradOffset(layerMap, uvLayerQ, ddx, ddy, offset);"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert ", input." not in generated_code


def test_opengl_implicit_projected_array_texture_stage_input_members_strip_samplers():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "vec4 projected = textureProj(layerMap, uvLayerQ);" in generated_code
    assert (
        "vec4 projectedOffset = textureProjOffset(layerMap, uvLayerQ, offset);"
        in generated_code
    )
    assert (
        "vec4 projectedLod = textureProjLod(layerMap, uvLayerQ, lod);" in generated_code
    )
    assert (
        "vec4 projectedLodOffset = textureProjLodOffset(layerMap, uvLayerQ, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 projectedGrad = textureProjGrad(layerMap, uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 projectedGradOffset = textureProjGradOffset(layerMap, uvLayerQ, ddx, ddy, offset);"
        in generated_code
    )
    assert "input." not in generated_code


def test_opengl_projected_array_texture_resource_arrays_strip_samplers():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "vec4 projectedLeaf(sampler2DArray maps[4], int layer, vec4 uvLayerQ, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "vec4 fixedProj = textureProj(maps[2], uvLayerQ);" in generated_code
    assert (
        "vec4 dynamicOffset = textureProjOffset(maps[layer], uvLayerQ, offset);"
        in generated_code
    )
    assert "vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);" in generated_code
    assert (
        "vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], uvLayerQ, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], uvLayerQ, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, layer, uvLayerQ, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "linearSamplers" not in generated_code
    assert "samplers" not in generated_code
    assert "input." not in generated_code


def test_opengl_implicit_projected_array_texture_resource_arrays_strip_samplers():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "vec4 projectedLeaf(sampler2DArray maps[4], int layer, vec4 uvLayerQ, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "vec4 fixedProj = textureProj(maps[2], uvLayerQ);" in generated_code
    assert (
        "vec4 dynamicOffset = textureProjOffset(maps[layer], uvLayerQ, offset);"
        in generated_code
    )
    assert "vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);" in generated_code
    assert (
        "vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], uvLayerQ, lod, offset);"
        in generated_code
    )
    assert (
        "vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);"
        in generated_code
    )
    assert (
        "vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], uvLayerQ, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, layer, uvLayerQ, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "input." not in generated_code


def test_opengl_implicit_projected_stage_input_members_strip_samplers():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in generated_code
    assert "layout(binding = 2) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 3) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert "vec4 color = textureProj(colorMap, uvq);" in generated_code
    assert "vec4 colorLod = textureProjLod(colorMap, uvqw, lod);" in generated_code
    assert "vec4 volume = textureProj(volumeMap, xyzq);" in generated_code
    assert (
        "float shadow = texture(shadowMap, vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float shadowGrad = textureGrad(shadowArray, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert "unsupported GLSL texture compare" not in generated_code
    assert "textureCompareProj" not in generated_code
    assert "input." not in generated_code


def test_opengl_projected_shadow_compare_variants_filter_sampler_arguments():
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
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvLayerQ, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvLayerQ, depth, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvLayerQ, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvLayerQ, depth, ddx, ddy, offset);
            return projected + offsetProjected + gradProjected + gradOffsetProjected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = projectedShadow(
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
                float arrayShadow = projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    input.uvLayerQ,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(shadow + arrayShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "float projectedShadow(sampler2DShadow tex, vec3 uvq, vec4 uvqw, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float projected = texture(tex, vec3(uvq.xy / uvq.z, depth));" in generated_code
    )
    assert (
        "float projectedW = texture(tex, vec3(uvqw.xy / uvqw.w, depth));"
        in generated_code
    )
    assert (
        "float offsetProjected = textureOffset(tex, vec3(uvq.xy / uvq.z, depth), offset);"
        in generated_code
    )
    assert (
        "float lodProjected = textureLod(tex, vec3(uvq.xy / uvq.z, depth), lod);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = textureLodOffset(tex, vec3(uvq.xy / uvq.z, depth), lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = textureGrad(tex, vec3(uvq.xy / uvq.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = textureGradOffset(tex, vec3(uvq.xy / uvq.z, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedShadow(shadowMap, uvq, uvqw, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "float projectedArrayShadow(sampler2DArrayShadow tex, vec4 uvLayerQ, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float projected = texture(tex, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth));"
        in generated_code
    )
    assert (
        "float offsetProjected = textureOffset(tex, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), offset);"
        in generated_code
    )
    assert (
        "float gradProjected = textureGrad(tex, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = textureGradOffset(tex, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedArrayShadow(shadowArray, uvLayerQ, depth, ddx, ddy, offset)"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_opengl_direct_projected_shadow_compare_stage_input_members():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "float planar = texture(shadowMap, vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarOffset = textureOffset(shadowMap, vec3(uvqw.xy / uvqw.w, depth), offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = textureGrad(shadowArray, vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "unsupported GLSL texture compare" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_opengl_projected_shadow_compare_resource_arrays_strip_sampler_arguments():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "float projectedLeaf(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float planar = texture(shadowMaps[layer], vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarOffset = textureOffset(shadowMaps[1], vec3(uvq.xy / uvq.z, depth), offset);"
        in generated_code
    )
    assert (
        "float planarLod = textureLod(shadowMaps[2], vec3(uvq.xy / uvq.z, depth), lod);"
        in generated_code
    )
    assert (
        "float planarGradOffset = textureGradOffset(shadowMaps[layer], vec3(uvq.xy / uvq.z, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float arrayProjected = texture(shadowArrays[2], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth));"
        in generated_code
    )
    assert (
        "float arrayOffset = textureOffset(shadowArrays[layer], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = textureGrad(shadowArrays[1], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float projectedWrapper(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_opengl_implicit_projected_shadow_compare_resource_arrays_strip_samplers():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "float projectedLeaf(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float planar = texture(shadowMaps[layer], vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarOffset = textureOffset(shadowMaps[1], vec3(uvq.xy / uvq.z, depth), offset);"
        in generated_code
    )
    assert (
        "float planarLod = textureLod(shadowMaps[2], vec3(uvq.xy / uvq.z, depth), lod);"
        in generated_code
    )
    assert (
        "float planarGradOffset = textureGradOffset(shadowMaps[layer], vec3(uvq.xy / uvq.z, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float arrayProjected = texture(shadowArrays[2], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth));"
        in generated_code
    )
    assert (
        "float arrayOffset = textureOffset(shadowArrays[layer], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = textureGrad(shadowArrays[1], vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float projectedWrapper(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "sampler shadowMapsSampler" not in generated_code
    assert "sampler shadowArraysSampler" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_opengl_unsized_projected_shadow_compare_arrays_infer_transitive_constant_size():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[5];" in generated_code
    )
    assert (
        "layout(binding = 5) uniform sampler2DArrayShadow shadowArrays[5];"
        in generated_code
    )
    assert "layout(binding = 10) uniform sampler2DShadow afterShadow;" in generated_code
    assert (
        "float shadowDeep(sampler2DShadow shadowMaps[5], sampler2DArrayShadow shadowArrays[5], "
        "vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
    ) in generated_code
    assert (
        "float shadowMid(sampler2DShadow shadowMaps[5], sampler2DArrayShadow shadowArrays[5], "
        "vec3 uvq, vec4 uvLayerQ, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
    ) in generated_code
    assert (
        "float planarHigh = texture(shadowMaps[LAYER], vec3(uvq.xy / uvq.z, depth));"
        in generated_code
    )
    assert (
        "float planarLow = textureOffset(shadowMaps[1], vec3(uvq.xy / uvq.z, depth), offset);"
        in generated_code
    )
    assert (
        "float arrayHigh = textureGrad(shadowArrays[(2 * 2)], "
        "vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), ddx, ddy);"
    ) in generated_code
    assert (
        "float arrayLow = textureOffset(shadowArrays[3], "
        "vec4(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z, depth), offset);"
    ) in generated_code
    assert (
        "shadowDeep(shadowMaps, shadowArrays, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "shadowMid(shadowMaps, shadowArrays, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "float singleShadow = texture(afterShadow, vec3(uvq.xy, depth));"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers" not in generated_code
    assert "afterSampler" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_opengl_projected_array_shadow_lod_offset_reports_unsupported():
    shader = """
    shader ProjectedArrayShadowLodOffsetUnsupported {
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float depth;
            float lod;
            ivec2 offset @ TEXCOORD1;
        };

        float projectedArrayShadow(
            sampler2DArrayShadow tex,
            sampler s,
            vec4 uvLayerQ,
            float depth,
            float lod,
            ivec2 offset
        ) {
            float rejected = textureCompareProjLodOffset(tex, s, uvLayerQ, depth, lod, offset);
            return rejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.offset
                );
                return vec4(shadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float rejected = /* unsupported GLSL texture compare: textureCompareProjLodOffset projected explicit LOD is not supported for sampler2DArrayShadow */ 0.0;"
        in generated_code
    )
    assert "textureLodOffset(" not in generated_code


def test_opengl_projected_cube_shadow_compare_reports_unsupported():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform samplerCubeShadow cubeMap;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeArray;"
        in generated_code
    )
    assert (
        "float cubeProjected(samplerCubeShadow tex, vec4 cubeProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayProjected(samplerCubeArrayShadow tex, vec4 cubeLayerProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
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
                f"/* unsupported GLSL texture compare: {func_name} requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow vec4 projection coordinates */ 0.0"
            )
            == 2
        )
    assert "compareSampler" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureCompareProjOffset(" not in generated_code
    assert "textureCompareProjLodOffset(" not in generated_code
    assert "textureCompareProjGradOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_opengl_direct_projected_cube_texture_reports_unsupported():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    expected_counts = {
        "textureProj": 3,
        "textureProjLodOffset": 2,
        "textureProjGradOffset": 2,
        "textureProjLod": 1,
    }
    for func_name, count in expected_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported GLSL projected texture: {func_name} requires 1D, 2D, 2D-array, or 3D projection coordinates */ vec4(0.0)"
            )
            == count
        )
    assert "linearSampler" not in generated_code
    assert "textureProj(cubeMap" not in generated_code
    assert "textureProj(cubeArray" not in generated_code


def test_opengl_projected_cube_texture_resource_arrays_report_unsupported():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMaps[4];" in generated_code
    assert (
        "layout(binding = 4) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert (
        "layout(binding = 8) uniform samplerCubeShadow shadowCubes[4];"
        in generated_code
    )
    assert (
        "layout(binding = 12) uniform samplerCubeArrayShadow shadowCubeArrays[4];"
        in generated_code
    )
    assert (
        "vec4 projectedCube(samplerCube maps[4], int layer, vec4 cubeProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 projectedCubeArray(samplerCubeArray maps[4], int layer, vec4 cubeArrayProj, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 projectedShadow(samplerCubeShadow maps[4], int layer, vec4 cubeProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 projectedShadowArray(samplerCubeArrayShadow maps[4], int layer, vec4 cubeArrayProj, float depth, float lod, vec3 ddx, vec3 ddy, ivec2 offset)"
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
                f"/* unsupported GLSL projected texture: {func_name} requires 1D, 2D, 2D-array, or 3D projection coordinates */ vec4(0.0)"
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
                f"/* unsupported GLSL texture compare: {func_name} requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert (
        "projectedCube(cubeMaps, layer, cubeProj, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedCubeArray(cubeArrays, layer, cubeArrayProj, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedShadow(shadowCubes, layer, cubeProj, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedShadowArray(shadowCubeArrays, layer, cubeArrayProj, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureCompareProj(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code


def test_opengl_shadow_gather_compare_offsets_filter_sampler_arguments():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "vec4 gatherShadow(sampler2DShadow tex, vec2 uv, float depth, ivec2 offset)"
        in generated_code
    )
    assert "textureGather(tex, uv, depth)" in generated_code
    assert "textureGatherOffset(tex, uv, depth, offset)" in generated_code
    assert "textureOffset(tex, vec3(uv, depth), offset)" in generated_code
    assert (
        "vec4 gatherShadowArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, ivec2 offset)"
        in generated_code
    )
    assert "textureGather(tex, uvLayer, depth)" in generated_code
    assert "textureGatherOffset(tex, uvLayer, depth, offset)" in generated_code
    assert "textureOffset(tex, vec4(uvLayer, depth), offset)" in generated_code
    assert (
        "vec4 gatherCubeShadowArray(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return textureGather(tex, cubeLayer, depth);" in generated_code
    assert "gatherShadow(shadowMap, uv, depth, offset)" in generated_code
    assert "gatherShadowArray(shadowArray, uvLayer, depth, offset)" in generated_code
    assert "gatherCubeShadowArray(cubeShadowArray, cubeLayer, depth)" in generated_code
    assert "compareSampler" not in generated_code
    assert "textureGatherCompare" not in generated_code
    assert "textureGatherCompareOffset" not in generated_code
    assert "textureCompareOffset" not in generated_code


def test_opengl_cube_shadow_gather_compare_supports_cube_and_cube_array():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCubeShadow cubeShadow;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "vec4 gatherCubeShadow(samplerCubeShadow tex, vec3 direction, float depth)"
        in generated_code
    )
    assert "return textureGather(tex, direction, depth);" in generated_code
    assert (
        "vec4 gatherCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return textureGather(tex, cubeLayer, depth);" in generated_code
    assert (
        "vec4 implicitCubeShadow(samplerCubeShadow tex, vec3 direction, float depth)"
        in generated_code
    )
    assert (
        "vec4 implicitCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "gatherCubeShadow(cubeShadow, direction, depth)" in generated_code
    assert "gatherCubeArrayShadow(cubeShadowArray, cubeLayer, depth)" in generated_code
    assert "textureGather(cubeShadow, direction, depth)" in generated_code
    assert "textureGather(cubeShadowArray, cubeLayer, depth)" in generated_code
    assert "compareSampler" not in generated_code
    assert "unsupported GLSL texture gather compare" not in generated_code
    assert "textureGatherCompare" not in generated_code
    assert "input." not in generated_code


def test_opengl_direct_shadow_gather_compare_stage_input_members():
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

    explicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )

    for generated_code in (explicit_code, implicit_code):
        assert (
            "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
        )
        assert (
            "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
            in generated_code
        )
        assert (
            "layout(binding = 2) uniform samplerCubeArrayShadow cubeShadowArray;"
            in generated_code
        )
        assert "vec4 planar = textureGather(shadowMap, uv, depth);" in generated_code
        assert (
            "vec4 planarOffset = textureGatherOffset(shadowMap, uv, depth, offset);"
            in generated_code
        )
        assert (
            "vec4 arrayGather = textureGather(shadowArray, uvLayer, depth);"
            in generated_code
        )
        assert (
            "vec4 arrayOffset = textureGatherOffset(shadowArray, uvLayer, depth, offset);"
            in generated_code
        )
        assert (
            "vec4 cubeArrayGather = textureGather(cubeShadowArray, cubeLayer, depth);"
            in generated_code
        )
        assert "unsupported GLSL texture gather compare" not in generated_code
        assert "textureGatherCompare" not in generated_code
        assert "compareSampler" not in generated_code
        assert "input." not in generated_code


def test_opengl_shadow_gather_compare_resource_arrays_strip_samplers():
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

    explicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )

    for generated_code in (explicit_code, implicit_code):
        assert (
            "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];"
            in generated_code
        )
        assert (
            "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
            in generated_code
        )
        assert (
            "layout(binding = 8) uniform samplerCubeArrayShadow cubeArrays[4];"
            in generated_code
        )
        assert (
            "vec4 gatherLayer(sampler2DShadow maps[4], sampler2DArrayShadow arrays[4], samplerCubeArrayShadow cubes[4], int layer, vec2 uv, vec3 uvLayer, vec4 cubeLayer, float depth, ivec2 offset)"
            in generated_code
        )
        assert "vec4 fixedPlanar = textureGather(maps[2], uv, depth);" in generated_code
        assert (
            "vec4 dynamicPlanarOffset = textureGatherOffset(maps[layer], uv, depth, offset);"
            in generated_code
        )
        assert (
            "vec4 fixedArray = textureGather(arrays[1], uvLayer, depth);"
            in generated_code
        )
        assert (
            "vec4 dynamicArrayOffset = textureGatherOffset(arrays[layer], uvLayer, depth, offset);"
            in generated_code
        )
        assert (
            "vec4 fixedCubeArray = textureGather(cubes[3], cubeLayer, depth);"
            in generated_code
        )
        assert (
            "vec4 dynamicCubeArray = textureGather(cubes[layer], cubeLayer, depth);"
            in generated_code
        )
        assert (
            "gatherLayer(shadowMaps, shadowArrays, cubeArrays, layer, uv, uvLayer, cubeLayer, depth, offset)"
            in generated_code
        )
        assert "compareSamplers" not in generated_code
        assert "samplers" not in generated_code
        assert "textureGatherCompare" not in generated_code
        assert "input." not in generated_code


def test_opengl_unsupported_cube_shadow_gather_compare_offsets_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array shadow samplers */ "
        "vec4(0.0)"
    )
    assert "layout(binding = 0) uniform samplerCubeShadow cubeMap;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeArray;"
        in generated_code
    )
    assert (
        "vec4 cubeOffset(samplerCubeShadow tex, vec3 direction, float depth, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 cubeArrayOffset(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth, ivec2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 3
    assert "cubeOffset(cubeMap, direction, depth, offset)" in generated_code
    assert "cubeArrayOffset(cubeArray, cubeLayer, depth, offset)" in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "input." not in generated_code


def test_opengl_shadow_compare_lod_grad_filter_sampler_arguments():
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
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float gradValue = textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, offset);
            return gradValue + gradOffsetValue;
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
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(shadow + arrayShadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 1) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "float compareShadow(sampler2DShadow tex, vec2 uv, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "float lodValue = textureLod(tex, vec3(uv, depth), lod);" in generated_code
    assert (
        "float lodOffsetValue = textureLodOffset(tex, vec3(uv, depth), lod, offset);"
        in generated_code
    )
    assert (
        "float gradValue = textureGrad(tex, vec3(uv, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetValue = textureGradOffset(tex, vec3(uv, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float compareShadowArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float gradValue = textureGrad(tex, vec4(uvLayer, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetValue = textureGradOffset(tex, vec4(uvLayer, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "compareShadow(shadowMap, uv, depth, lod, ddx, ddy, offset)" in generated_code
    )
    assert (
        "compareShadowArray(shadowArray, uvLayer, depth, ddx, ddy, offset)"
        in generated_code
    )
    assert "compareSampler" not in generated_code
    assert "textureCompareLod" not in generated_code
    assert "textureCompareGrad" not in generated_code


def test_opengl_nested_implicit_shadow_compare_lod_grad_strips_samplers():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "float shadowOps(sampler2DShadow tex, vec2 uv, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "vec2 lod = textureQueryLod(tex, uv);" in generated_code
    assert "float cmp = textureLod(tex, vec3(uv, depth), lod.x);" in generated_code
    assert (
        "float grad = textureGradOffset(tex, vec3(uv, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float wrappedShadow(sampler2DShadow tex, vec2 uv, float depth, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert "return shadowOps(tex, uv, depth, ddx, ddy, offset);" in generated_code
    assert "wrappedShadow(shadowMap, uv, depth, ddx, ddy, offset)" in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_opengl_array_shadow_compare_lod_reports_unsupported():
    shader = """
    shader ArrayShadowCompareLodUnsupported {
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
            float lod;
        };

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod
        ) {
            float rejected = textureCompareLod(tex, s, uvLayer, depth, lod);
            return rejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod
                );
                return vec4(shadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float rejected = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert "textureLod(" not in generated_code


def test_opengl_array_shadow_compare_lod_offset_reports_unsupported():
    shader = """
    shader ArrayShadowCompareLodOffsetUnsupported {
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
            float lod;
            ivec2 offset @ TEXCOORD1;
        };

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod,
            ivec2 offset
        ) {
            float rejected = textureCompareLodOffset(tex, s, uvLayer, depth, lod, offset);
            return rejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float shadow = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.offset
                );
                return vec4(shadow);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float rejected = /* unsupported GLSL texture compare: textureCompareLodOffset explicit LOD offsets require 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert "textureLodOffset(" not in generated_code


def test_opengl_cube_shadow_compare_lod_offset_diagnostics():
    shader = """
    shader CubeShadowCompareLodOffsetDiagnostics {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
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

        float compareCubeShadow(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float lodRejected = textureCompareLod(tex, s, direction, depth, lod);
            float lodOffsetRejected = textureCompareLodOffset(tex, s, direction, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, direction, depth, ddx, ddy);
            float gradOffsetRejected = textureCompareGradOffset(tex, s, direction, depth, ddx, ddy, offset);
            return lodRejected + lodOffsetRejected + gradValue + gradOffsetRejected;
        }

        float compareCubeArrayShadow(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float lodRejected = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float lodOffsetRejected = textureCompareLodOffset(tex, s, cubeLayer, depth, lod, offset);
            float gradRejected = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            float gradOffsetRejected = textureCompareGradOffset(tex, s, cubeLayer, depth, ddx, ddy, offset);
            return lodRejected + lodOffsetRejected + gradRejected + gradOffsetRejected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(
                    compareCubeShadow(
                        cubeShadow,
                        compareSampler,
                        input.direction,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + compareCubeArrayShadow(
                        cubeShadowArray,
                        compareSampler,
                        input.cubeLayer,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                );
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        "float lodRejected = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float lodOffsetRejected = /* unsupported GLSL texture compare: textureCompareLodOffset explicit LOD offsets require 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float gradValue = textureGrad(tex, vec4(direction, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetRejected = /* unsupported GLSL texture compare: textureCompareGradOffset explicit gradient offsets require 2D or 2D-array shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float gradRejected = /* unsupported GLSL texture compare: textureCompareGrad explicit gradients require 2D, 2D-array, or cube shadow samplers */ 0.0;"
        in generated_code
    )
    assert "textureLod(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert "compareSampler" not in generated_code


def test_opengl_array_shadow_texture_resource_arrays_keep_compare_coordinates():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "layout(binding = 4) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert (
        "float sampleArrayLayer(sampler2DArrayShadow shadowArrays[4], vec3 uvLayer, float depth)"
        in generated_code
    )
    assert "texture(shadowArrays[2], vec4(uvLayer, depth))" in generated_code
    assert (
        "float sampleCubeLayer(samplerCubeArrayShadow cubeShadowArrays[4], vec4 cubeLayer, float depth)"
        in generated_code
    )
    assert "texture(cubeShadowArrays[3], cubeLayer, depth)" in generated_code
    assert "sampleArrayLayer(shadowArrays, uvLayer, depth)" in generated_code
    assert "sampleCubeLayer(cubeShadowArrays, cubeLayer, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[2]" not in generated_code
    assert "shadowSamplers[3]" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "vec3(uvLayer, depth)" not in generated_code
    assert "vec4(cubeLayer, depth)" not in generated_code


def test_opengl_array_shadow_compare_lod_grad_resource_arrays():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "float shadowLayer(sampler2DShadow shadowMaps[4], sampler2DArrayShadow shadowArrays[4], int layer, vec2 uv, vec3 uvLayer, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "float planarLod = textureLod(shadowMaps[layer], vec3(uv, depth), lod);"
        in generated_code
    )
    assert (
        "float planarGrad = textureGradOffset(shadowMaps[1], vec3(uv, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float arrayLod = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float arrayGrad = textureGradOffset(shadowArrays[layer], vec4(uvLayer, depth), ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowArrays, layer, uv, uvLayer, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_opengl_cube_shadow_compare_lod_grad_resource_arrays():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform samplerCubeShadow cubeMaps[4];" in generated_code
    )
    assert (
        "layout(binding = 4) uniform samplerCubeArrayShadow cubeArrays[4];"
        in generated_code
    )
    assert (
        "float cubeShadowLayer(samplerCubeShadow cubeMaps[4], samplerCubeArrayShadow cubeArrays[4], int layer, vec3 direction, vec4 cubeLayer, float depth, float lod, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0"
        )
        == 2
    )
    assert (
        "float cubeGrad = textureGrad(cubeMaps[1], vec4(direction, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float cubeArrayGrad = /* unsupported GLSL texture compare: textureCompareGrad explicit gradients require 2D, 2D-array, or cube shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "cubeShadowLayer(cubeMaps, cubeArrays, layer, direction, cubeLayer, depth, lod, ddx, ddy)"
        in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "shadowSamplers[" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_opengl_array_shadow_texture_resource_arrays_reject_mismatched_fixed_helper_size():
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
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


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


def test_opengl_direct_stage_image_load_store_and_atomics_use_input_members():
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
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(rgba32f, binding = 0) uniform image2D outImg;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volume;" in generated_code
    assert "layout(rgba32f, binding = 2) uniform image2DArray layers;" in generated_code
    assert "layout(r32ui, binding = 3) uniform uimage2D counters;" in generated_code
    assert (
        "layout(r32i, binding = 4) uniform iimage2DArray signedLayers;"
        in generated_code
    )
    assert "vec4 color = imageLoad(outImg, pixel);" in generated_code
    assert "vec4 volumeColor = imageLoad(volume, voxel);" in generated_code
    assert "vec4 layerColor = imageLoad(layers, pixelLayer);" in generated_code
    assert "imageStore(outImg, pixel, (color + layerColor));" in generated_code
    assert "imageStore(volume, voxel, volumeColor);" in generated_code
    assert "imageStore(layers, pixelLayer, color);" in generated_code
    assert "uint previous = imageAtomicAdd(counters, pixel, amount);" in generated_code
    assert (
        "int previousSigned = imageAtomicMin(signedLayers, pixelLayer, signedAmount);"
        in generated_code
    )
    assert "input.pixel" not in generated_code
    assert "input.voxel" not in generated_code
    assert "input.pixelLayer" not in generated_code


def test_opengl_direct_stage_explicit_image_formats_use_input_members():
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
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(r32f, binding = 0) uniform image2D scalarFloat;" in generated_code
    assert "layout(rg32f, binding = 1) uniform image2D rgFloat;" in generated_code
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D unsignedScalar;" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 3) uniform uimage2DArray unsignedLayers;"
        in generated_code
    )
    assert "layout(r32i, binding = 4) uniform iimage3D signedVolume;" in generated_code
    assert "float scalarValue = imageLoad(scalarFloat, pixel).x;" in generated_code
    assert "vec2 rgValue = imageLoad(rgFloat, pixel).xy;" in generated_code
    assert "uint unsignedValue = imageLoad(unsignedScalar, pixel).x;" in generated_code
    assert (
        "uvec2 layerValue = imageLoad(unsignedLayers, pixelLayer).xy;" in generated_code
    )
    assert "int signedValue = imageLoad(signedVolume, voxel).x;" in generated_code
    assert (
        "imageStore(scalarFloat, pixel, vec4((scalarValue + amount)));"
        in generated_code
    )
    assert (
        "imageStore(rgFloat, pixel, vec4((rgValue + vec2(amount)), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(unsignedScalar, pixel, uvec4((unsignedValue + unsignedAmount)));"
        in generated_code
    )
    assert (
        "imageStore(unsignedLayers, pixelLayer, uvec4((layerValue + uvec2(unsignedAmount)), 0u, 0u));"
        in generated_code
    )
    assert (
        "imageStore(signedVolume, voxel, ivec4((signedValue + signedAmount)));"
        in generated_code
    )
    assert "input.pixel" not in generated_code
    assert "input.voxel" not in generated_code
    assert "input.pixelLayer" not in generated_code


def test_opengl_direct_stage_image_compare_swap_use_input_members():
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
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(r32ui, binding = 0) uniform uimage3D volumeCounters;" in generated_code
    )
    assert (
        "layout(r32i, binding = 1) uniform iimage2DArray layerCounters;"
        in generated_code
    )
    assert (
        "uint volumeOld = imageAtomicCompSwap(volumeCounters, voxel, expected, replacement);"
        in generated_code
    )
    assert (
        "int layerOld = imageAtomicCompSwap(layerCounters, pixelLayer, signedExpected, signedReplacement);"
        in generated_code
    )
    assert "input.voxel" not in generated_code
    assert "input.pixelLayer" not in generated_code
    assert "input.expected" not in generated_code
    assert "input.signedExpected" not in generated_code


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


def test_opengl_rg_image_arrays_respect_scalar_and_vector_context():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[3];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 3) uniform uimage2D rgUnsignedImages[2];"
        in generated_code
    )
    assert (
        "float scalarFloat(image2D images[3], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloat(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert (
        "uint scalarUnsigned(uimage2D images[2], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsigned(uimage2D images[2], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[1], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[1], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert "uvec2 oldValue = imageLoad(images[1], pixel).xy;" in generated_code
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[1], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code
    assert "uint oldValue = imageLoad(images[1], pixel);" not in generated_code
    assert "uvec2 oldValue = imageLoad(images[1], pixel).x;" not in generated_code


def test_opengl_inferred_rg_image_arrays_respect_scalar_context():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 3;" in generated_code
    assert "const int LAYER = (COUNT - 1);" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[3];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 3) uniform uimage2D rgUnsignedImages[COUNT];"
        in generated_code
    )
    assert "layout(rg32f, binding = 6) uniform image2D afterImages;" in generated_code
    assert (
        "float scalarFloat(image2D images[3], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloat(image2D images[3], ivec2 pixel, vec2 value)" in generated_code
    )
    assert (
        "uint scalarUnsigned(uimage2D images[COUNT], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsigned(uimage2D images[COUNT], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )
    assert (
        "layout(rg32f, binding = 3) uniform image2D afterImages;" not in generated_code
    )
    assert "float oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).x;" not in generated_code


def test_opengl_transitive_rg_image_arrays_share_call_site_size():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert "layout(rg32f, binding = 8) uniform image2D afterImages;" in generated_code
    assert (
        "float scalarFloatDeep(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "float scalarFloatMid(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatDeep(image2D images[4], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatMid(image2D images[4], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedDeep(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedDeep(uimage2D images[4], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "imageStore(images[1], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(images[1], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u));"
        in generated_code
    )
    assert "vec2 vectorFloatDeep(image2D images[3]" not in generated_code
    assert "uvec2 vectorUnsignedDeep(uimage2D images[3]" not in generated_code


def test_opengl_fixed_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatFixed(image2D images[4], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedFixed(uimage2D images[4], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )
    assert (
        "layout(rg32ui, binding = 1) uniform uimage2D rgUnsignedImages[];"
        not in generated_code
    )


def test_opengl_const_sized_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert "const int LAST = (COUNT - 1);" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(image2D images[COUNT], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatFixed(image2D images[COUNT], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(uimage2D images[COUNT], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedFixed(uimage2D images[COUNT], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[LAST], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[LAST], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "float scalarFloatFixed(image2D images[4]" not in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )


def test_opengl_expr_sized_rg_image_array_parameters_size_unsized_globals():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 3;" in generated_code
    assert "const int UINT_COUNT = 2;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float scalarFloatFixed(image2D images[(COUNT + 1)], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "vec2 vectorFloatFixed(image2D images[(COUNT + 1)], ivec2 pixel, vec2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(uimage2D images[(UINT_COUNT * 2)], ivec2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uvec2 vectorUnsignedFixed(uimage2D images[(UINT_COUNT * 2)], ivec2 pixel, uvec2 value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "vec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "uvec2 oldValue = imageLoad(images[2], pixel).xy;" in generated_code
    assert "float scalarFloatFixed(image2D images[4]" not in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )


def test_opengl_conflicting_fixed_rg_image_array_sizes_raise():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_direct_rg_image_array_index_conflicts_with_fixed_parameter_size():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_direct_rg_image_array_index_within_fixed_parameter_size():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 4) uniform uimage2D rgUnsignedImages[4];"
        in generated_code
    )
    assert (
        "float touchFour(image2D images[4], ivec2 pixel, float value)" in generated_code
    )
    assert (
        "uint touchUnsignedFour(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "float directFloat = imageLoad(rgFloatImages[2], pixel).x;" in generated_code
    assert (
        "uint directUint = imageLoad(rgUnsignedImages[1], pixel).x;" in generated_code
    )
    assert "float oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[3], pixel).x;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[3];"
        not in generated_code
    )
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];"
        not in generated_code
    )


def test_opengl_fixed_rg_image_array_global_conflicts_with_fixed_parameter_size():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_global_widens_unsized_parameter_size():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert (
        "float touchUnsized(image2D images[4], ivec2 pixel, float value)"
        in generated_code
    )
    assert "float oldValue = imageLoad(images[2], pixel).x;" in generated_code
    assert (
        "float touchUnsized(image2D images[3], ivec2 pixel, float value)"
        not in generated_code
    )


def test_opengl_fixed_rg_image_array_global_direct_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_parameter_direct_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_global_const_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_parameter_const_index_out_of_bounds_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_fixed_rg_image_array_const_index_within_bounds_generates():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert "float touch(image2D images[4], ivec2 pixel)" in generated_code
    assert "return imageLoad(images[(COUNT - 1)], pixel).x;" in generated_code
    assert (
        "float direct = imageLoad(rgFloatImages[(COUNT - 1)], pixel).x;"
        in generated_code
    )
    assert "float touch(image2D images[5], ivec2 pixel)" not in generated_code


def test_opengl_fixed_rg_image_array_shadowed_const_index_stays_dynamic():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert "float touch(image2D images[4], ivec2 pixel)" in generated_code
    assert "return imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "float direct = imageLoad(rgFloatImages[COUNT], pixel).x;" in generated_code
    assert "float touch(image2D images[5], ivec2 pixel)" not in generated_code


def test_opengl_transitive_rg_image_array_shadowed_const_index_stays_dynamic():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in generated_code
    )
    assert "float leaf(image2D images[4], ivec2 pixel)" in generated_code
    assert "float passThrough(image2D images[4], ivec2 pixel)" in generated_code
    assert "return imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "float sampled = imageLoad(images[COUNT], pixel).x;" in generated_code
    assert "float leaf(image2D images[5], ivec2 pixel)" not in generated_code
    assert "float passThrough(image2D images[5], ivec2 pixel)" not in generated_code


def test_opengl_transitive_rg_image_array_unshadowed_const_index_conflict_raises():
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
        GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_opengl_shadowed_rg_image_array_constant_keeps_scalar_context():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int LAYER = 3;" in generated_code
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[];" in generated_code
    )
    assert (
        "layout(rg32ui, binding = 1) uniform uimage2D rgUnsignedImages[];"
        in generated_code
    )
    assert "layout(rg32f, binding = 2) uniform image2D afterImages;" in generated_code
    assert (
        "float scalarFloat(image2D images[], ivec2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(uimage2D images[], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "float oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel).x;" in generated_code
    assert (
        "imageStore(images[0], pixel, vec4((oldValue + value), 0.0, 0.0, 0.0));"
        in generated_code
    )
    assert (
        "imageStore(images[0], pixel, uvec4((oldValue + value), 0u, 0u, 0u));"
        in generated_code
    )
    assert (
        "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];"
        not in generated_code
    )
    assert (
        "layout(rg32f, binding = 4) uniform image2D afterImages;" not in generated_code
    )
    assert "float oldValue = imageLoad(images[LAYER], pixel);" not in generated_code
    assert "uint oldValue = imageLoad(images[LAYER], pixel);" not in generated_code


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


def test_opengl_formatted_image_arrays_ignore_unsupported_indices():
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

    dynamic_code = GLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = GLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[];" in dynamic_code
    assert "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in dynamic_code
    assert (
        "uint touchCounters(uimage2D images[], int layer, ivec2 pixel, uint value)"
        in dynamic_code
    )
    assert "uint oldValue = imageLoad(images[layer], pixel).x;" in dynamic_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in dynamic_code
    assert "uint a = touchCounters(counters, 0, ivec2(1, 2), 3);" in dynamic_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[2];" not in dynamic_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;" not in dynamic_code
    )
    assert "uint oldValue = imageLoad(images[layer], pixel);" not in dynamic_code

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[];" in negative_code
    assert "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in negative_code
    assert (
        "uint touchCounters(uimage2D images[], ivec2 pixel, uint value)"
        in negative_code
    )
    assert "uint oldValue = imageLoad(images[(-1)], pixel).x;" in negative_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in negative_code
    assert "uint a = touchCounters(counters, ivec2(1, 2), 3);" in negative_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[0];" not in negative_code
    )
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D afterCounters;"
        not in negative_code
    )
    assert "uint oldValue = imageLoad(images[(-1)], pixel);" not in negative_code


def test_opengl_formatted_image_arrays_ignore_function_call_indices():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(r32ui, binding = 0) uniform uimage2D counters[];" in generated_code
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D afterCounters;" in generated_code
    )
    assert "int getLayer()" in generated_code
    assert (
        "uint touchCounters(uimage2D images[], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = imageLoad(images[getLayer()], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "uint a = touchCounters(counters, ivec2(1, 2), 3);" in generated_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[1];" not in generated_code
    )
    assert (
        "layout(r32ui, binding = 2) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "uint oldValue = imageLoad(images[getLayer()], pixel);" not in generated_code


def test_opengl_formatted_image_arrays_infer_local_constant_alias_size():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int GLOBAL = 2;" in generated_code
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[4];" in generated_code
    assert (
        "layout(r32ui, binding = 4) uniform uimage2D afterCounters;" in generated_code
    )
    assert (
        "uint touchCounters(uimage2D images[4], ivec2 pixel, uint value)"
        in generated_code
    )
    assert "const int LOCAL = (GLOBAL + 1);" in generated_code
    assert "uint oldValue = imageLoad(images[LOCAL], pixel).x;" in generated_code
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in generated_code
    assert "uint a = touchCounters(counters, ivec2(1, 2), 3);" in generated_code
    assert (
        "layout(r32ui, binding = 0) uniform uimage2D counters[];" not in generated_code
    )
    assert (
        "layout(r32ui, binding = 1) uniform uimage2D afterCounters;"
        not in generated_code
    )
    assert "    int LOCAL = (GLOBAL + 1);" not in generated_code
    assert "uint oldValue = imageLoad(images[LOCAL], pixel);" not in generated_code


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


def test_opengl_storage_image_size_queries_use_image_size():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert "layout(r32ui, binding = 3) uniform uimage2D uintImage;" in generated_code
    assert "layout(r32i, binding = 4) uniform iimage3D intVolume;" in generated_code
    assert "ivec2 queryImage2D(image2D image)" in generated_code
    assert "ivec3 queryImageArray(image2DArray image)" in generated_code
    assert "return (imageSize(image) + imageSize(image));" in generated_code
    assert (
        "ivec2 a = (imageSize(colorImage) + imageSize(colorImage));" in generated_code
    )
    assert (
        "ivec3 b = (imageSize(volumeImage) + imageSize(volumeImage));" in generated_code
    )
    assert (
        "ivec3 c = (imageSize(layerImage) + imageSize(layerImage));" in generated_code
    )
    assert "ivec2 d = (imageSize(uintImage) + imageSize(uintImage));" in generated_code
    assert "ivec3 e = (imageSize(intVolume) + imageSize(intVolume));" in generated_code
    assert "textureSize(" not in generated_code


def test_opengl_storage_image_levels_and_lod_queries_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLevels on image2D */ 0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLevels on image3D */ 0"
        )
        == 2
    )
    assert (
        "/* unsupported GLSL texture query: textureQueryLevels on image2DArray */ 0"
        in generated_code
    )
    assert (
        "/* unsupported GLSL texture query: textureQueryLevels on uimage2D */ 0"
        in generated_code
    )
    assert (
        "/* unsupported GLSL texture query: textureQueryLevels on iimage3D */ 0"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLod on image2D */ vec2(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported GLSL texture query: textureQueryLod on image3D */ vec2(0.0)"
        )
        == 2
    )
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code


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
                ivec2 q = query2D(colorMap, linearSampler, uv);
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


def test_opengl_multisample_texture_samples_queries_keep_builtin():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msTextures[4];" in generated_code
    assert "layout(binding = 6) uniform sampler2DMSArray msArrays[4];" in generated_code
    assert (
        "int querySamples(sampler2DMS tex, sampler2DMSArray texArray)" in generated_code
    )
    assert "return (textureSamples(tex) + textureSamples(texArray));" in generated_code
    assert (
        "int queryArraySamples(sampler2DMS textures[4], sampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureSamples(textures[2]) + textureSamples(arrays[layer]));"
        in generated_code
    )
    assert (
        "querySamples(msTex, msArray) + queryArraySamples(msTextures, msArrays, layer)"
        in generated_code
    )
    assert "input." not in generated_code


def test_opengl_multisample_image_samples_queries_use_texture_samples_builtin():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msTextures[4];" in generated_code
    assert "layout(binding = 6) uniform sampler2DMSArray msArrays[4];" in generated_code
    assert (
        "int querySamples(sampler2DMS tex, sampler2DMSArray texArray)" in generated_code
    )
    assert "return (textureSamples(tex) + textureSamples(texArray));" in generated_code
    assert (
        "int queryArraySamples(sampler2DMS textures[4], sampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureSamples(textures[2]) + textureSamples(arrays[layer]));"
        in generated_code
    )
    assert "textureSamples(msTex) + textureSamples(msArray)" in generated_code
    assert "queryArraySamples(msTextures, msArrays, layer)" in generated_code
    assert "imageSamples(" not in generated_code
    assert "unsupported GLSL texture samples query" not in generated_code
    assert "input." not in generated_code


def test_opengl_non_multisample_texture_samples_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture samples query: "
        "requires multisample sampler */ 0"
    )
    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 2) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 3) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 6) uniform sampler2D textures[4];" in generated_code
    assert generated_code.count(diagnostic) == 10
    assert "textureSamples(" not in generated_code
    assert "input." not in generated_code


def test_opengl_storage_image_samples_queries_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported GLSL texture samples query: "
        "requires multisample sampler */ 0"
    )
    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert "layout(r32ui, binding = 3) uniform uimage2D uintImage;" in generated_code
    assert "layout(r32i, binding = 4) uniform iimage3D intVolume;" in generated_code
    assert generated_code.count(diagnostic) == 13
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "input." not in generated_code


def test_opengl_storage_image_sampling_and_fetch_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = "unsupported GLSL storage image texture operation"
    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
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
        assert f"{operation} on image2D" in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code
    assert "input." not in generated_code


def test_opengl_storage_image_compare_calls_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    comparison_diagnostic = "unsupported GLSL storage image texture comparison"
    gather_diagnostic = "unsupported GLSL storage image texture operation"
    assert "layout(rgba32f, binding = 0) uniform image2D colorImage;" in generated_code
    assert "layout(rgba32f, binding = 1) uniform image3D volumeImage;" in generated_code
    assert (
        "layout(rgba32f, binding = 2) uniform image2DArray layerImage;"
        in generated_code
    )
    assert generated_code.count(comparison_diagnostic) == 14
    assert generated_code.count(gather_diagnostic) == 4
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "input." not in generated_code


def test_opengl_multisample_texture_query_levels_lower_to_single_level():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DMS msTextures[4];" in generated_code
    assert "layout(binding = 6) uniform sampler2DMSArray msArrays[4];" in generated_code
    assert "int levels2D(sampler2DMS tex)" in generated_code
    assert "int levelsArray(sampler2DMSArray tex)" in generated_code
    assert (
        "int levelsResourceArrays(sampler2DMS textures[4], sampler2DMSArray arrays[4], int layer)"
        in generated_code
    )
    assert generated_code.count("return 1;") == 2
    assert "return (1 + 1);" in generated_code
    assert "int c = 1;" in generated_code
    assert "int d = 1;" in generated_code
    assert "int e = 1;" in generated_code
    assert "int f = 1;" in generated_code
    assert "textureQueryLevels(" not in generated_code


def test_opengl_multisample_sampling_operations_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert (
        "vec4 invalid2D(sampler2DMS tex, vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 invalidArray(sampler2DMSArray tex, vec3 uvLayer, vec2 ddx, vec2 ddy, ivec2 offset)"
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
            f"unsupported GLSL multisample texture call: {func_name} on sampler2DMS"
            in generated_code
        )
        assert (
            f"unsupported GLSL multisample texture call: {func_name} on sampler2DMSArray"
            in generated_code
        )

    for invalid_call in {
        "texture(tex,",
        "textureLod(tex,",
        "textureGrad(tex,",
        "textureGather(tex,",
        "textureGatherOffset(tex,",
    }:
        assert invalid_call not in generated_code


def test_opengl_multisample_texture_query_lod_emits_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert "vec2 query2D(sampler2DMS tex, vec2 uv)" in generated_code
    assert "vec2 queryArray(sampler2DMSArray tex, vec3 uvLayer)" in generated_code
    assert "textureQueryLod(" not in generated_code
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture query: textureQueryLod on sampler2DMS */ vec2(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL multisample texture query: textureQueryLod on sampler2DMSArray"
        )
        == 2
    )


def test_opengl_multisample_texel_fetch_offsets_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in generated_code
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in generated_code
    assert (
        "vec4 offset2D(sampler2DMS tex, ivec2 pixel, int sampleIndex, ivec2 offset)"
        in generated_code
    )
    assert (
        "vec4 offsetArray(sampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex, ivec2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch offset: multisample texture sampler2DMS does not support offsets"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch offset: multisample texture sampler2DMSArray does not support offsets"
        )
        == 2
    )
    assert "texelFetchOffset(" not in generated_code


def test_opengl_cube_texel_fetches_emit_diagnostics():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform samplerCube cubeMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "vec4 fetchCube(samplerCube tex, ivec3 cubeCoord, int lod, ivec3 offset)"
        in generated_code
    )
    assert (
        "vec4 fetchCubeArray(samplerCubeArray tex, ivec4 cubeLayerCoord, int lod, ivec3 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetch on samplerCube */ vec4(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetchOffset on samplerCube */ vec4(0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetch on samplerCubeArray"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported GLSL texel fetch: texelFetchOffset on samplerCubeArray"
        )
        == 2
    )
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_opengl_array_shadow_texture_query_functions():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert "ivec3 query2DArrayShadow(sampler2DArrayShadow tex)" in generated_code
    assert "ivec3 queryCubeArrayShadow(samplerCubeArrayShadow tex)" in generated_code
    assert (
        "ivec3 queryArrayElements(sampler2DArrayShadow shadowArrays[4], samplerCubeArrayShadow cubeShadowArrays[4])"
        in generated_code
    )
    assert "ivec3 size = textureSize(tex, 1);" in generated_code
    assert "ivec3 size = textureSize(tex, 0);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert "ivec3 arraySize = textureSize(shadowArrays[2], 1);" in generated_code
    assert "ivec3 cubeSize = textureSize(cubeShadowArrays[3], 0);" in generated_code
    assert "int arrayLevels = textureQueryLevels(shadowArrays[2]);" in generated_code
    assert "int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);" in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_direct_stage_texture_queries_mix_size_levels_and_lod():
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

    generated_code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 2) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 3) uniform sampler2DShadow shadowMap;" in generated_code
    assert (
        "layout(binding = 4) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 5) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert "linearSampler" not in generated_code
    assert "ivec2 colorSize = textureSize(colorMap, 1);" in generated_code
    assert "ivec3 layerSize = textureSize(layerMap, 2);" in generated_code
    assert "ivec3 cubeSize = textureSize(cubeArray, 0);" in generated_code
    assert "ivec2 shadowSize = textureSize(shadowMap, 0);" in generated_code
    assert "ivec3 shadowArraySize = textureSize(shadowArray, 1);" in generated_code
    assert "ivec3 cubeShadowSize = textureSize(cubeShadowArray, 0);" in generated_code
    assert "int colorLevels = textureQueryLevels(colorMap);" in generated_code
    assert "int shadowLevels = textureQueryLevels(shadowArray);" in generated_code
    assert "vec2 colorLod = textureQueryLod(colorMap, uv);" in generated_code
    assert "vec2 layerLod = textureQueryLod(layerMap, uvLayer.xy);" in generated_code
    assert "vec2 cubeLod = textureQueryLod(cubeArray, cubeLayer.xyz);" in generated_code
    assert (
        "vec2 implicitLayerLod = textureQueryLod(layerMap, uvLayer.xy);"
        in generated_code
    )
    assert (
        "vec2 implicitCubeShadowLod = textureQueryLod(cubeShadowArray, cubeLayer.xyz);"
        in generated_code
    )
    assert "textureQueryLod(layerMap, uvLayer);" not in generated_code
    assert "textureQueryLod(cubeArray, cubeLayer);" not in generated_code
    assert "input." not in generated_code


def test_opengl_array_texture_query_lod_uses_non_layer_coordinates():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert "layout(binding = 2) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "layout(binding = 6) uniform samplerCubeArray cubeArrays[4];" in generated_code
    )
    assert "vec2 queryArrayLod(sampler2DArray tex, vec3 uvLayer)" in generated_code
    assert "return textureQueryLod(tex, uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayLod(samplerCubeArray tex, vec4 cubeLayer)" in generated_code
    )
    assert "return textureQueryLod(tex, cubeLayer.xyz);" in generated_code
    assert (
        "vec2 queryArrayElementLod(sampler2DArray layerMaps[4], vec3 uvLayer)"
        in generated_code
    )
    assert "return textureQueryLod(layerMaps[2], uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayElementLod(samplerCubeArray cubeArrays[4], vec4 cubeLayer)"
        in generated_code
    )
    assert "return textureQueryLod(cubeArrays[3], cubeLayer.xyz);" in generated_code
    assert "queryArrayLod(layerMap, uvLayer)" in generated_code
    assert "queryCubeArrayLod(cubeArray, cubeLayer)" in generated_code
    assert "queryArrayElementLod(layerMaps, uvLayer)" in generated_code
    assert "queryCubeArrayElementLod(cubeArrays, cubeLayer)" in generated_code
    assert "linearSampler" not in generated_code
    assert "linearSamplers" not in generated_code
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert "textureQueryLod(tex, cubeLayer);" not in generated_code


def test_opengl_shadow_array_texture_query_lod_uses_non_layer_coordinates():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 2) uniform sampler2DArrayShadow shadowArrays[4];"
        in generated_code
    )
    assert (
        "layout(binding = 6) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert (
        "vec2 queryArrayLod(sampler2DArrayShadow tex, vec3 uvLayer)" in generated_code
    )
    assert "return textureQueryLod(tex, uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer)"
        in generated_code
    )
    assert "return textureQueryLod(tex, cubeLayer.xyz);" in generated_code
    assert (
        "vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[4], vec3 uvLayer)"
        in generated_code
    )
    assert "return textureQueryLod(shadowArrays[2], uvLayer.xy);" in generated_code
    assert (
        "vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[4], vec4 cubeLayer)"
        in generated_code
    )
    assert (
        "return textureQueryLod(cubeShadowArrays[3], cubeLayer.xyz);" in generated_code
    )
    assert "queryArrayLod(shadowArray, uvLayer)" in generated_code
    assert "queryCubeArrayLod(cubeShadowArray, cubeLayer)" in generated_code
    assert "queryArrayElementLod(shadowArrays, uvLayer)" in generated_code
    assert "queryCubeArrayElementLod(cubeShadowArrays, cubeLayer)" in generated_code
    assert "linearSampler" not in generated_code
    assert "linearSamplers" not in generated_code
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert "textureQueryLod(tex, cubeLayer);" not in generated_code


def test_opengl_mixed_cube_shadow_query_lod_and_compare_keep_coordinate_shapes():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform samplerCubeShadow cubeShadow;" in generated_code
    assert (
        "layout(binding = 1) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert "shadowSampler" not in generated_code
    assert (
        "float inspectCubeShadow(samplerCubeShadow tex, vec3 direction, float depth, float lod, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "vec2 lodValue = textureQueryLod(tex, direction);" in generated_code
    assert "float cmp = texture(tex, vec4(direction, depth));" in generated_code
    assert (
        "float cmpLod = /* unsupported GLSL texture compare: textureCompareLod explicit LOD requires 2D shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "float grad = textureGrad(tex, vec4(direction, depth), ddx, ddy);"
        in generated_code
    )
    assert (
        "float inspectCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth, float lod, vec3 ddx, vec3 ddy)"
        in generated_code
    )
    assert "vec2 lodValue = textureQueryLod(tex, cubeLayer.xyz);" in generated_code
    assert "float cmp = texture(tex, cubeLayer, depth);" in generated_code
    assert (
        "float grad = /* unsupported GLSL texture compare: textureCompareGrad explicit gradients require 2D, 2D-array, or cube shadow samplers */ 0.0;"
        in generated_code
    )
    assert (
        "inspectCubeShadow(cubeShadow, direction, depth, lod, ddx, ddy)"
        in generated_code
    )
    assert (
        "inspectCubeArrayShadow(cubeShadowArray, cubeLayer, depth, lod, ddx, ddy)"
        in generated_code
    )
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_opengl_implicit_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ImplicitArrayTextureQueryLod {
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArray layerMaps[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArray tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArray tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        vec2 queryShadowArrayLod(sampler2DArrayShadow tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        vec2 queryShadowCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArray layerMaps[], vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], uvLayer);
        }

        vec2 queryShadowCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(layerMap, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeArray, input.cubeLayer);
                vec2 c = queryShadowArrayLod(shadowArray, input.uvLayer);
                vec2 d = queryShadowCubeArrayLod(cubeShadowArray, input.cubeLayer);
                vec2 e = queryArrayElementLod(layerMaps, input.uvLayer);
                vec2 f = queryShadowCubeArrayElementLod(cubeShadowArrays, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, e.x + f.y, 1.0);
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2DArray layerMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCubeArray cubeArray;" in generated_code
    assert (
        "layout(binding = 2) uniform sampler2DArrayShadow shadowArray;"
        in generated_code
    )
    assert (
        "layout(binding = 3) uniform samplerCubeArrayShadow cubeShadowArray;"
        in generated_code
    )
    assert "layout(binding = 4) uniform sampler2DArray layerMaps[4];" in generated_code
    assert (
        "layout(binding = 8) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in generated_code
    )
    assert "vec2 queryArrayLod(sampler2DArray tex, vec3 uvLayer)" in generated_code
    assert (
        "vec2 queryCubeArrayLod(samplerCubeArray tex, vec4 cubeLayer)" in generated_code
    )
    assert (
        "vec2 queryShadowArrayLod(sampler2DArrayShadow tex, vec3 uvLayer)"
        in generated_code
    )
    assert (
        "vec2 queryShadowCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer)"
        in generated_code
    )
    assert (
        "vec2 queryArrayElementLod(sampler2DArray layerMaps[4], vec3 uvLayer)"
        in generated_code
    )
    assert (
        "vec2 queryShadowCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[4], vec4 cubeLayer)"
        in generated_code
    )
    assert generated_code.count("return textureQueryLod(tex, uvLayer.xy);") == 2
    assert generated_code.count("return textureQueryLod(tex, cubeLayer.xyz);") == 2
    assert "return textureQueryLod(layerMaps[2], uvLayer.xy);" in generated_code
    assert (
        "return textureQueryLod(cubeShadowArrays[3], cubeLayer.xyz);" in generated_code
    )
    assert "textureQueryLod(tex, uvLayer);" not in generated_code
    assert "textureQueryLod(tex, cubeLayer);" not in generated_code
    assert "textureQueryLod(layerMaps[2], uvLayer);" not in generated_code
    assert "textureQueryLod(cubeShadowArrays[3], cubeLayer);" not in generated_code


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
                vec4 lodColor = textureLod(colorMap, uv, 1.0);
                vec4 gradColor = textureGrad(colorMap, uv, vec2(0.1), vec2(0.2));
                vec4 fetched = texelFetch(colorMap, pixel, 0);
                vec4 gathered = textureGather(colorMap, uv);
                return lodColor + gradColor + fetched + gathered;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "textureLod(colorMap, uv, 1.0)" in generated_code
    assert "textureGrad(colorMap, uv, vec2(0.1), vec2(0.2))" in generated_code
    assert "texelFetch(colorMap, pixel, 0)" in generated_code
    assert "textureGather(colorMap, uv)" in generated_code
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
                return texture(colorMap, linearSampler, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "layout(binding = 1) uniform samplerCube envMap;" in generated_code
    assert "linearSampler" not in generated_code
    assert "texture(colorMap, uv)" in generated_code


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
                return sampleColor(linearSampler, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "vec4 sampleColor(vec2 uv)" in generated_code
    assert "texture(colorMap, uv)" in generated_code
    assert "sampleColor(uv)" in generated_code
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
                return sampleColor(colorMap, linearSampler, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "vec4 sampleColor(sampler2D tex, vec2 uv)" in generated_code
    assert "texture(tex, uv)" in generated_code
    assert "sampleColor(colorMap, uv)" in generated_code
    assert "linearSampler" not in generated_code
    assert "sampleState" not in generated_code


def test_opengl_struct_member_sampler_expression_is_dropped():
    shader = """
    shader StructSamplerExpression {
        sampler2D colorMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        vec4 samplePacked(sampler2D tex, SamplerPack pack, int index, vec2 uv) {
            return texture(tex, pack.samplers[index], uv);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                SamplerPack pack;
                return samplePacked(colorMap, pack, 0, vec2(0.5));
            }
        }
    }
    """

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct SamplerPack" in generated_code
    assert "sampler samplers[2];" in generated_code
    assert (
        "vec4 samplePacked(sampler2D tex, SamplerPack pack, int index, vec2 uv)"
        in generated_code
    )
    assert "return texture(tex, uv);" in generated_code
    assert "texture(tex, pack.samplers[index], uv)" not in generated_code


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
                return sampleColor(colorMap, uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2D colorMap;" in generated_code
    assert "vec4 sampleColor(sampler2D tex, vec2 uv)" in generated_code
    assert "texture(tex, uv)" in generated_code
    assert "sampleColor(colorMap, uv)" in generated_code


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
                return textureCompare(shadowMap, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert "texture(shadowMap, vec3(uv, depth))" in generated_code
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
                return textureCompare(shadowMaps[layer], uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert "texture(shadowMaps[layer], vec3(uv, depth))" in generated_code
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
                return shadowLayer(shadowMaps, shadowSamplers, layer, uv, depth);
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
    assert "shadowLayer(shadowMaps, layer, uv, depth)" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "shadowLayer(shadowMaps, unaryShadowMaps, uv, depth)" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
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
                float arrayShadow = shadowMid(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "shadowMid(shadowMaps, uv, depth)" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, layer, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "shadowLayer(shadowMaps, layer, uv, depth)" in generated_code
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, layer, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
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
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, uv, depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, uv, depth);
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
    assert "texture(afterShadow, vec3(uv, depth))" in generated_code
    assert "shadowLayer(shadowMaps, uv, depth)" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[];"
        not in generated_code
    )
    assert "sampler shadowSamplers" not in generated_code
    assert "sampler afterSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_fixed_shadow_texture_array_rejects_mismatched_fixed_helper_size():
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
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_opengl_fixed_shadow_texture_array_widens_unsized_helper():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "float shadowUnsized(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert "texture(shadowMaps[2], vec3(uv, depth))" in generated_code
    assert "shadowUnsized(shadowMaps, uv, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_transitive_shadow_texture_array_shadowed_const_index_stays_dynamic():
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

    generated_code = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int COUNT = 4;" in generated_code
    assert (
        "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in generated_code
    )
    assert (
        "float leaf(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert (
        "float passThrough(sampler2DShadow shadowMaps[4], vec2 uv, float depth)"
        in generated_code
    )
    assert generated_code.count("int COUNT = 0;") == 2
    assert "texture(shadowMaps[COUNT], vec3(uv, depth))" in generated_code
    assert "leaf(shadowMaps, uv, depth)" in generated_code
    assert "passThrough(shadowMaps, uv, depth)" in generated_code
    assert "sampler shadowSamplers" not in generated_code
    assert "textureCompare(" not in generated_code


def test_opengl_transitive_shadow_texture_array_unshadowed_const_index_conflict_raises():
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
        GLSLCodeGen().generate(crosstl.translator.parse(shader))


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
                return sampleShadow(shadowMap, uv, depth);
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
    assert "sampleShadow(shadowMap, uv, depth)" in generated_code
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
                return sampleShadow(shadowMap, shadowSampler, uv, depth);
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
    assert "sampleShadow(shadowMap, uv, depth)" in generated_code
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
                return sampleShadow(shadowSampler, uv, depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = GLSLCodeGen().generate(ast)

    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in generated_code
    assert "float sampleShadow(vec2 uv, float depth)" in generated_code
    assert "texture(shadowMap, vec3(uv, depth))" in generated_code
    assert "sampleShadow(uv, depth)" in generated_code
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
