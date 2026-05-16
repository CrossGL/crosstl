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


def test_do_while_codegen():
    code = """
    void main() {
        vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
        int i = 0;
        do {
            color.r += 0.1;
            i++;
        } while (i < 10);
        gl_FragColor = color;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
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
        print(generated_code)
    except SyntaxError:
        pytest.fail("Switch-case parsing or code generation not implemented.")


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
