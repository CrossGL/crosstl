from crosstl.backend.SPIRV import VulkanCrossGLCodeGen
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser
import pytest
from typing import List


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = VulkanCrossGLCodeGen.VulkanToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = VulkanLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = VulkanParser(tokens)
    return parser.parse()


FRAGMENT_SHADER = """
void main() {
    vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
    gl_FragColor = color;
}
"""

LAYOUT_SHADER = """
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj;
    vec4 tint;
} camera;
layout(set = 0, binding = 1) buffer Particles {
    vec4 position;
    float mass;
} particles;
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

ARRAY_LAYOUT_SHADER = """
layout(set = 0, binding = 0) uniform Bones {
    mat4 transforms[64];
    vec4 weights[4];
} bones;
layout(set = 0, binding = 1) buffer Particles {
    vec4 positions[128];
    float masses[128];
} particles;
void main() {
    gl_Position = transforms[0] * vec4(1.0, 0.0, 0.0, 1.0);
}
"""

RESOURCE_UNIFORM_SHADER = """
layout(set = 0, binding = 1) uniform sampler2D albedoTex;
uniform samplerCube skybox;
void main() {
    vec4 color = texture(albedoTex, vec2(0.5, 0.5));
    gl_FragColor = color;
}
"""


def test_vulkan_to_crossgl_emits_fragment_main():
    tokens = tokenize_code(FRAGMENT_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(ast.functions) == 1
    assert ast.functions[0].return_type == "void"
    assert ast.functions[0].name == "main"
    assert "fragment {" in generated_code
    assert "void main()" in generated_code
    assert "float4 color = float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "gl_FragColor = color;" in generated_code


def test_standalone_function_call_codegen():
    code = """
    void helper(int amount, int current) {
    }

    void main() {
        int value = 1;
        helper(1, value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "helper(1, value);" in generated_code
    assert "int value = 1;" in generated_code
    assert " helper;" not in generated_code
    assert "Unhandled statement type" not in generated_code


def test_member_call_statement_codegen():
    code = """
    void main() {
        image.store(1, value);
        int result = object.method();
        objects[0].store(value);
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "image.store(1, value);" in generated_code
    assert "int result = object.method();" in generated_code
    assert "objects[0].store(value);" in generated_code
    assert " image.store;" not in generated_code
    assert "int value = 1;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_function_parameter_qualifiers_codegen_smoke():
    code = """
    void accumulate(in vec3 normal, inout float weight, out vec4 color) {
        color = vec4(normal, weight);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void accumulate(float3 normal, float weight, float4 color)" in generated_code
    )
    assert "color = float4(normal, weight);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_member_access_assignment_codegen():
    code = """
    void main() {
        color.r = 1.0;
        color.g += 0.5;
        objects[0].field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "color.r = 1.0;" in generated_code
    assert "color.g += 0.5;" in generated_code
    assert "objects[0].field = value;" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_typed_local_array_declaration_codegen():
    code = """
    void main() {
        float weights[4];
        weights[0] = 1.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float weights[4];" in generated_code
    assert "weights[0] = 1.0;" in generated_code
    assert "float weights;" not in generated_code


def test_translate_api_accepts_spirv_source(tmp_path):
    import crosstl

    shader_path = tmp_path / "fragment.spirv"
    shader_path.write_text(FRAGMENT_SHADER, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="rust", format_output=False
    )

    assert "#[fragment_shader]" in generated_code
    assert "pub fn main()" in generated_code
    assert "let mut color: float4 = float4(1.0, 0.0, 0.0, 1.0);" in generated_code


def test_vulkan_layout_blocks_emit_crossgl_resources():
    tokens = tokenize_code(LAYOUT_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(ast.global_variables) == 3
    assert ast.global_variables[0].block_name == "Camera"
    assert ast.global_variables[1].layout_type == "BUFFER"
    assert ast.global_variables[2].data_type == "vec3"
    assert "cbuffer Camera" in generated_code
    assert "float4x4 viewProj;" in generated_code
    assert "struct Particles" in generated_code
    assert "RWStructuredBuffer<Particles> particles;" in generated_code
    assert "float3 position;" in generated_code
    assert "vertex {" in generated_code
    assert "gl_Position = float4(position, 1.0);" in generated_code


def test_vulkan_layout_interpolation_qualifier_codegen():
    code = """
    layout(location = 0) flat in int faceID;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].declaration_qualifiers == ["flat"]
    assert "int faceID;" in generated_code


def test_vulkan_readonly_buffer_layout_codegen():
    code = """
    layout(set = 0, binding = 0) readonly buffer Particles {
        vec4 pos[];
    } particles;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].declaration_qualifiers == ["readonly"]
    assert "struct Particles" in generated_code
    assert "float4 pos[];" in generated_code
    assert "StructuredBuffer<Particles> particles;" in generated_code
    assert "RWStructuredBuffer<Particles> particles;" not in generated_code


def test_vulkan_compute_local_size_layout_codegen():
    code = """
    layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "compute {" in generated_code
    assert (
        "layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;"
        in generated_code
    )
    assert "void main()" in generated_code
    assert "fragment {" not in generated_code


def test_vulkan_uniform_block_instance_member_access_flattens_to_cbuffer_member():
    code = """
    layout(set = 0, binding = 0) uniform Camera {
        mat4 viewProj;
    } camera;
    layout(location = 0) in vec3 position;
    void main() {
        gl_Position = camera.viewProj * vec4(position, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "cbuffer Camera" in generated_code
    assert "float4x4 viewProj;" in generated_code
    assert "gl_Position = (viewProj * float4(position, 1.0));" in generated_code
    assert "camera.viewProj" not in generated_code


def test_vulkan_layout_blocks_preserve_array_members():
    tokens = tokenize_code(ARRAY_LAYOUT_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].struct_fields == [
        ("mat4", "transforms[64]"),
        ("vec4", "weights[4]"),
    ]
    assert ast.global_variables[1].struct_fields == [
        ("vec4", "positions[128]"),
        ("float", "masses[128]"),
    ]
    assert "float4x4 transforms[64];" in generated_code
    assert "float4 weights[4];" in generated_code
    assert "float4 positions[128];" in generated_code
    assert "float masses[128];" in generated_code
    assert "transforms[0] * float4(1.0, 0.0, 0.0, 1.0)" in generated_code


def test_standalone_struct_array_members_codegen():
    code = """
    struct LightBlock {
        vec3 positions[4];
        float weights[4];
    };
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct LightBlock" in generated_code
    assert "float3 positions[4];" in generated_code
    assert "float weights[4];" in generated_code


def test_vulkan_resource_uniforms_emit_crossgl_resources():
    tokens = tokenize_code(RESOURCE_UNIFORM_SHADER)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "sampler2D"
    assert ast.global_variables[0].variable_name == "albedoTex"
    assert ast.global_variables[1].vtype == "samplerCube"
    assert ast.global_variables[1].name == "skybox"
    assert "Texture2D albedoTex;" in generated_code
    assert "TextureCube skybox;" in generated_code


def test_vulkan_one_dimensional_sampler_uniforms_emit_crossgl_resources():
    code = """
    layout(set = 0, binding = 0) uniform sampler1D ramp;
    uniform sampler1DArray ramps;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert ast.global_variables[0].data_type == "sampler1D"
    assert ast.global_variables[0].variable_name == "ramp"
    assert ast.global_variables[1].vtype == "sampler1DArray"
    assert ast.global_variables[1].name == "ramps"
    assert "Texture1D ramp;" in generated_code
    assert "Texture1DArray ramps;" in generated_code


def test_vulkan_precision_declarations_do_not_emit_crossgl_resources():
    code = """
    precision highp float;
    precision mediump sampler2D;
    layout(set = 0, binding = 0) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert len(ast.global_variables) == 1
    assert ast.global_variables[0].data_type == "sampler2D"
    assert "Texture2D albedoTex;" in generated_code
    assert "precision" not in generated_code


def test_vulkan_layout_precision_after_in_codegen():
    code = """
    #version 310 es
    precision highp float;
    layout(location = 0) in highp vec2 vUV;
    layout(location = 0) out mediump vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float2 vUV;" in generated_code
    assert "float4 fragColor;" in generated_code
    assert "highp" not in generated_code
    assert "mediump" not in generated_code


def test_translate_api_accepts_spirv_layout_source(tmp_path):
    import crosstl

    shader_path = tmp_path / "layout_vertex.spirv"
    shader_path.write_text(LAYOUT_SHADER, encoding="utf-8")

    generated_code = crosstl.translate(
        str(shader_path), backend="rust", format_output=False
    )

    assert "pub struct Particles" in generated_code
    assert "static viewProj: float4x4 = Default::default();" in generated_code
    assert (
        "static particles: RWStructuredBuffer<Particles> = Default::default();"
        in generated_code
    )
    assert "static position: float3 = Default::default();" in generated_code
    assert "#[vertex_shader]" in generated_code


def test_vulkan_bitwise_shift_precedence_codegen():
    code = """
    void main() {
        int a = 1;
        int b = 2;
        int c = 3;
        int value = a & b << c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int value = (a & (b << c));" in generated_code
    assert "int value = ((a & b) << c);" not in generated_code


def test_struct_codegen():
    code = """
    struct VertexInput {
        vec4 position;
        vec4 color;
    };

    struct VertexOutput {
        vec4 position;
        vec4 color;
    };

    void main() {
        VertexOutput output;
        output.position = vec4(1.0, 0.0, 0.0, 1.0);
        output.color = vec4(0.0, 1.0, 0.0, 1.0);
        gl_Position = output.position;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "VertexOutput output;" in generated_code
    except SyntaxError:
        pytest.fail("Struct parsing or code generation not implemented.")


def test_if_codegen():
    code = """
    void main() {
        vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
        if (color.r > 0.5) {
            color = vec4(0.0, 1.0, 0.0, 1.0);
        }
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("If statement parsing or code generation not implemented.")


def test_for_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        for (int i = 0; i < 10; i++) {
            color.r += 0.1;
        }
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")


def test_for_structured_update_codegen():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; items[i]++) {
        }
        for (int i = 0; i < 4; object.field--) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); items[i]++) {" in generated_code
    assert "for (int i = 0; (i < 4); object.field--) {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_for_compound_update_codegen():
    code = """
    void main() {
        for (int i = 0; i < 4; i += 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); i += 1) {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_for_structured_assignment_update_codegen():
    code = """
    void main() {
        int value = 1;
        int items[4];
        for (int i = 0; i < 4; items[i] += value) {
        }
        for (int i = 0; i < 4; object.field = value) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); items[i] += value) {" in generated_code
    assert "for (int i = 0; (i < 4); object.field = value) {" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_while_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        int i = 0;
        while (i < 10) {
            color.r += 0.1;
            i++;
        }
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_continue_codegen():
    code = """
    void main() {
        for (int i = 0; i < 4; i++) {
            if (i == 2) {
                continue;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "continue;" in generated_code
        assert "Unhandled statement type" not in generated_code
    except SyntaxError:
        pytest.fail("Continue statement parsing or code generation not implemented.")


def test_do_while_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        int i = 0;
        do {
            color.r += 0.1;
            i++;
            i--;
        } while (i < 10);
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "do {" in generated_code
        assert "i++;" in generated_code
        assert "i = i++;" not in generated_code
        assert "i--;" in generated_code
        assert "i = i--;" not in generated_code
        assert "} while ((i < 10));" in generated_code
        assert "Unhandled statement type" not in generated_code
    except SyntaxError:
        pytest.fail("Do-while loop parsing or code generation not implemented.")


def test_else_if_codegen():
    code = """
    void main() {
        float value = 0.7;
        vec4 color;
        
        if (value > 0.8) {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        } else if (value > 0.5) {
            color = vec4(0.0, 1.0, 0.0, 1.0);
        } else {
            color = vec4(0.0, 0.0, 1.0, 1.0);
        }
        
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Else-if statement parsing or code generation not implemented.")


def test_switch_case_codegen():
    code = """
    void main() {
        int value = 2;
        vec4 color;
        
        switch (value) {
            case 0:
                color = vec4(1.0, 0.0, 0.0, 1.0);
                break;
            case 1:
                color = vec4(0.0, 1.0, 0.0, 1.0);
                break;
            case 2:
                color = vec4(0.0, 0.0, 1.0, 1.0);
                break;
            default:
                color = vec4(0.5, 0.5, 0.5, 1.0);
                break;
        }
        
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)

        assert "switch (value) {" in generated_code
        assert "case 0:" in generated_code
        assert "case 1:" in generated_code
        assert "case 2:" in generated_code
        assert "default:" in generated_code
        assert generated_code.count("break;") == 4
        assert "break;\n                break;" not in generated_code
        assert "Unhandled statement type" not in generated_code
    except SyntaxError:
        pytest.fail("Switch-case parsing or code generation not implemented.")


def test_switch_fallthrough_codegen_does_not_insert_break():
    code = """
    void main() {
        int value = 0;
        vec4 color;

        switch (value) {
            case 0:
                color = vec4(1.0, 0.0, 0.0, 1.0);
            case 1:
                color = vec4(0.0, 1.0, 0.0, 1.0);
                break;
            default:
                color = vec4(0.5, 0.5, 0.5, 1.0);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("case 1:", 1)[0]
    default_block = generated_code.split("default:", 1)[1].split("}", 1)[0]
    assert "break;" not in case_zero_block
    assert "break;" not in default_block
    assert generated_code.count("break;") == 1
    assert "Unhandled statement type" not in generated_code


def test_switch_return_and_discard_codegen():
    code = """
    void main() {
        int value = 0;
        vec4 color;

        switch (value) {
            case 0:
                discard;
            case 1:
                return;
            default:
                color = vec4(0.5, 0.5, 0.5, 1.0);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "discard;" in generated_code
    assert "return;" in generated_code
    assert "default:" in generated_code
    assert "color = float4(0.5, 0.5, 0.5, 1.0);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_switch_loop_control_codegen_preserves_nesting_without_extra_breaks():
    code = """
    void main() {
        int hits = 0;
        for (int i = 0; i < 4; i++) {
            switch (i) {
                case 0:
                    hits = hits + 1;
                    continue;
                case 1:
                    while (hits < 3) {
                        hits = hits + 1;
                        break;
                    }
                    break;
                default:
                    hits = hits + 100;
                    break;
            }
            hits = hits + 1000;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("case 1:", 1)[0]
    case_one_block = generated_code.split("case 1:", 1)[1].split("default:", 1)[0]

    assert "continue;" in case_zero_block
    assert "break;" not in case_zero_block
    assert "while ((hits < 3)) {" in case_one_block
    assert case_one_block.count("break;") == 2
    assert "            }\n            hits = (hits + 1000);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_loop_inside_switch_case_codegen_keeps_following_case_statement():
    code = """
    void main() {
        int hits = 0;
        int value = 0;
        switch (value) {
            case 0:
                for (int j = 0; j < 2; j++) {
                    hits = hits + j;
                    break;
                }
                hits = hits + 10;
                break;
            default:
                break;
        }
        hits = hits + 1000;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    case_zero_block = generated_code.split("case 0:", 1)[1].split("default:", 1)[0]

    assert "for (int j = 0; (j < 2); j++) {" in case_zero_block
    assert "hits = (hits + j);" in case_zero_block
    assert "hits = (hits + 10);" in case_zero_block
    assert case_zero_block.count("break;") == 2
    assert "        }\n        hits = (hits + 1000);" in generated_code
    assert "Unhandled statement type" not in generated_code


def test_bitwise_and_ops_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint mask = 3u;
        uint result = value & mask;  // Bitwise AND
        if (result == 1u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        # Print tokens for debugging
        print("Tokens for bitwise AND test:")
        for i, token in enumerate(tokens):
            print(f"{i}: {token}")

        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise AND operator parsing or code generation not implemented.")


def test_bitwise_or_ops_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint mask = 2u;
        uint result = value | mask;  // Bitwise OR
        if (result == 7u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR operator parsing or code generation not implemented.")


def test_bitwise_xor_ops_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint mask = 3u;
        uint result = value ^ mask;  // Bitwise XOR
        if (result == 6u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise XOR operator parsing or code generation not implemented.")


def test_bitwise_not_codegen():
    code = """
    void main() {
        uint value = 5u;
        uint result = ~value;  // Bitwise NOT
        if (result != 5u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing or code generation not implemented.")


def test_bitwise_shift_ops_codegen():
    code = """
    void main() {
        uint value = 4u;
        uint result1 = value << 1;  // Left shift
        uint result2 = value >> 1;  // Right shift
        if (result1 == 8u && result2 == 2u) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail(
            "Bitwise shift operators parsing or code generation not implemented."
        )


def test_double_dtype_codegen():
    code = """
    void main() {
        double value1 = 3.14159265358979323846;
        double value2 = 2.71828182845904523536;
        double result = value1 + value2;
        if (result > 5.0) {
            gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Double data type parsing or code generation not implemented.")


if __name__ == "__main__":
    pytest.main()
