from crosstl.backend.Mojo import MojoCrossGLCodeGen
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Mojo.MojoParser import MojoParser
import pytest
from typing import List


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = MojoCrossGLCodeGen.MojoToCrossGLConverter()
    return codegen.generate(ast_node)


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = MojoLexer(code)
    return lexer.tokenize()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST."""
    parser = MojoParser(tokens)
    return parser.parse()


def test_struct_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float2

    struct VSOutput:
        var out_position: float2

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position.xy
        return output

    struct PSInput:
        var in_position: float2

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = float4(input.in_position, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "struct VSInput" in generated_code
        assert "struct VSOutput" in generated_code
    except SyntaxError:
        pytest.fail("Struct parsing or code generation not implemented.")


def test_struct_generic_member_codegen():
    code = """
    struct Resources:
        @position
        var transform: Matrix[DType.float32, 4, 4]
        SIMD[DType.float32, 4] tint @color
        samples: InlineArray[SIMD[DType.float32, 4], 2]
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "mat4 transform @ Position;" in generated_code
    assert "vec4 tint @ Color;" in generated_code
    assert "InlineArray[SIMD[DType.float32, 4], 2] samples;" in generated_code


def test_brace_struct_codegen_preserves_generic_members_and_attributes():
    code = """
    struct Resources {
        @position
        var transform: Matrix[DType.float32, 4, 4]
        SIMD[DType.float32, 4] tint @color;
        samples: InlineArray[SIMD[DType.float32, 4], 2];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Resources" in generated_code
    assert "mat4 transform @ Position;" in generated_code
    assert "vec4 tint @ Color;" in generated_code
    assert "InlineArray[SIMD[DType.float32, 4], 2] samples;" in generated_code


def test_if_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        if input.color.x > 0.5:
            output.out_position = input.color
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        if input.in_position.x > 0.5:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "if " in generated_code
    except SyntaxError:
        pytest.fail("If statement parsing or code generation not implemented.")


def test_bare_identifier_block_condition_codegen():
    code = """
    fn main():
        if ready:
            sink()
        while running:
            tick()
        switch state:
            case active:
                sink()
            default:
                tick()
        let result = flag ? yes : no
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (ready)" in generated_code
    assert "while (running)" in generated_code
    assert "switch (state)" in generated_code
    assert "case active:" in generated_code
    assert "(flag ? yes : no)" in generated_code


def test_for_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        for var i: Int = 0; i < 10; i = i + 1:
            output.out_position = input.color
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        for var i: Int = 0; i < 10; i = i + 1:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "for " in generated_code or "while " in generated_code
    except SyntaxError:
        pytest.fail("For loop parsing or code generation not implemented.")


def test_c_style_for_codegen_preserves_initializer():
    code = """
    fn main():
        for var i: Int = 0; i < 4; i = i + 1:
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); i = (i + 1))" in generated_code


def test_c_style_for_array_assignment_update_codegen():
    code = """
    fn main():
        var value = 1
        for var i: Int = 0; i < 4; items[i] += value:
            pass
        for var i: Int = 0; i < 4; object.field = value:
            pass
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; (i < 4); items[i] += value)" in generated_code
    assert "for (int i = 0; (i < 4); object.field = value)" in generated_code


def test_for_in_iterable_codegen_preserves_iterable_loop():
    code = """
    fn main():
        for value in values:
            sink(value)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for value in values {" in generated_code


def test_for_in_range_codegen_lowers_range_call():
    code = """
    fn main():
        for i in range(4):
            sink(i)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; i++)" in generated_code
    assert "range(4)" not in generated_code


def test_while_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        var i: Int = 0
        while i < 10:
            output.out_position = input.color
            i = i + 1
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        var i: Int = 0
        while i < 10:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
            i = i + 1
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "while " in generated_code
    except SyntaxError:
        pytest.fail("While loop parsing or code generation not implemented.")


def test_else_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        if input.color.x > 0.5:
            output.out_position = input.color
        else:
            output.out_position = float4(0.0, 0.0, 0.0, 1.0)
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        if input.in_position.x > 0.5:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        else:
            output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "} else {" in generated_code or "else" in generated_code
    except SyntaxError:
        pytest.fail("Else statement parsing or code generation not implemented.")


def test_function_call_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn add(a: float4, b: float4) -> float4:
        return a + b

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        let result = add(input.position, input.color)
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = input.in_position
        let result = add(input.in_position, float4(1.0, 1.0, 1.0, 1.0))
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "add(" in generated_code
    except SyntaxError:
        pytest.fail("Function call parsing or code generation not implemented.")


def test_multiline_function_call_codegen():
    code = """
    fn main():
        let result = add(
            1.0,
            2.0
        )
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let result = add(1.0, 2.0);" in generated_code


def test_generic_function_signature_codegen():
    code = """
    fn build(
        value: SIMD[DType.float32, 4],
        Matrix[DType.float32, 4, 4] transform @binding(0)
    ) -> Matrix[DType.float32, 4, 4]:
        return transform
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "mat4 build(vec4 value, mat4 transform @ binding(0))" in generated_code
    assert "return transform;" in generated_code


def test_method_call_chain_codegen_preserves_receiver():
    code = """
    fn main():
        let channel = texture.sample(coord).x
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let channel = texture.sample(coord).x;" in generated_code


def test_parenthesized_method_call_chain_codegen():
    code = """
    fn main():
        let channel = (texture.sample(coord)).x
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let channel = texture.sample(coord).x;" in generated_code


def test_def_function_codegen_preserves_parameter_name():
    code = """
    def helper(x: Float32) -> Float32:
        return x
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "float helper(float x)" in generated_code
        assert "return x;" in generated_code
    except SyntaxError:
        pytest.fail("Def function parsing or code generation not implemented.")


def test_else_if_codegen():
    code = """
    struct VSInput:
        var position: float4
        var color: float4

    struct VSOutput:
        var out_position: float4

    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position
        if input.color.x > 0.5:
            output.out_position = input.color
        else:
            output.out_position = float4(0.0, 0.0, 0.0, 1.0)
        return output

    struct PSInput:
        var in_position: float4

    struct PSOutput:
        var out_color: float4

    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.in_position.x > 0.5:
            output.out_color = input.in_position
        elif input.in_position.x == 0.5:
            output.out_color = float4(1.0, 1.0, 1.0, 1.0)
        else:
            output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "} else if (" in generated_code or "else if" in generated_code
    except SyntaxError:
        pytest.fail("Else-if statement parsing or code generation not implemented.")


def test_assignment_ops_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = float4(0.0, 0.0, 0.0, 1.0)

        if input.in_position.x > 0.5:
            output.out_color += input.in_position

        if input.in_position.x < 0.5:
            output.out_color -= float4(0.1, 0.1, 0.1, 0.1)

        if input.in_position.y > 0.5:
            output.out_color *= 2.0

        if input.in_position.z > 0.5:
            output.out_color /= 2.0

        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "+=" in generated_code or "-=" in generated_code
    except SyntaxError:
        pytest.fail("Assignment ops parsing or code generation not implemented.")


def test_bitwise_ops_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        let val: UInt32 = 0x01
        if val | 0x02:
            # Test case for bitwise OR
            pass
        let filterA: UInt32 = 0b0001  # First filter
        let filterB: UInt32 = 0b1000  # Second filter

        # Merge both filters
        let combinedFilter = filterA | filterB  # combinedFilter becomes 0b1001
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "|" in generated_code
    except SyntaxError:
        pytest.fail("Bitwise ops parsing or code generation not implemented.")


def test_logical_ops_codegen():
    code = """
    fn main():
        let val_0: Bool = True
        let val_1 = val_0 and False
        let val_2 = val_0 or False
        let val_3 = not val_0
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "bool val_0 = true;" in generated_code
        assert "let val_1 = (val_0 && false);" in generated_code
        assert "let val_2 = (val_0 || false);" in generated_code
        assert "let val_3 = (!val_0);" in generated_code
        assert "\nand;" not in generated_code
        assert "\nor;" not in generated_code
    except SyntaxError:
        pytest.fail("Logical ops parsing or code generation not implemented.")


def test_switch_case_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        switch input.in_position.x:
            case 0.0:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0)
                break
            case 0.5:
                output.out_color = float4(0.0, 1.0, 0.0, 1.0)
                break
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0)
                break
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "switch " in generated_code
        assert "case " in generated_code
    except SyntaxError:
        pytest.fail("Switch case parsing or code generation not implemented.")


def test_pass_statement_codegen_is_noop():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.value > 0:
            pass
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "if " in generated_code
        assert "pass;" not in generated_code
        assert "Unhandled statement type: PassNode" not in generated_code
    except SyntaxError:
        pytest.fail("Pass statement code generation not implemented.")


def test_if_dedent_return_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.value > 0:
            pass
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert (
            "if ((input.value > 0)) {\n            }\n            return output;"
            in generated_code
        )
    except SyntaxError:
        pytest.fail("If dedent code generation not implemented.")


def test_break_continue_statement_codegen():
    code = """
    fn main():
        var i: Int = 0
        while i < 10:
            i = i + 1
            if i == 3:
                continue
            if i == 7:
                break
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "continue;" in generated_code
        assert "break;" in generated_code
        assert "continue;\n            }\n            if ((i == 7))" in generated_code
        assert "Unhandled statement type: ContinueNode" not in generated_code
        assert "Unhandled statement type: BreakNode" not in generated_code
    except SyntaxError:
        pytest.fail("Break/continue code generation not implemented.")


def test_switch_explicit_break_not_duplicated_codegen():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        switch input.value:
            case 1:
                output.out_color = float4(1.0, 0.0, 0.0, 1.0)
                break
            case 2:
                output.out_color = float4(0.0, 1.0, 0.0, 1.0)
                break
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert generated_code.count("break;") == 2
        assert "Unhandled statement type: BreakNode" not in generated_code
    except SyntaxError:
        pytest.fail("Switch explicit break code generation not implemented.")


def test_typed_local_declaration_codegen_preserves_type():
    code = """
    fn main():
        var x: Float32
        let y: Int = 1
        var transform: Matrix[DType.float32, 4, 4]
        var values: InlineArray[Float32, 4]
        let inferred = y
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float x;" in generated_code
    assert "int y = 1;" in generated_code
    assert "mat4 transform;" in generated_code
    assert "InlineArray[Float32, 4] values;" in generated_code
    assert "let inferred = y;" in generated_code
    assert "var transform;" not in generated_code


def test_double_dtype_codegen():
    code = """
    struct VSInput:
        var position: Float64
        var color: Float64

    fn vertex_main(input: VSInput) -> VSInput:
        var output: VSInput
        output.position = input.position * 2.0
        output.color = input.color / 2.0
        return output
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "Float64" in generated_code or "double" in generated_code
    except SyntaxError:
        pytest.fail("Double data type parsing or code generation not implemented.")


def test_vector_constructor_codegen():
    code = """
    fn vertex_main() -> float4:
        let uv = float2(0.5, 0.5)
        let color = float4(uv.x, uv.y, 0.0, 1.0)
        return color
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "float2(" in generated_code or "vec2(" in generated_code
        assert "float4(" in generated_code or "vec4(" in generated_code
    except SyntaxError:
        pytest.fail("Vector constructor parsing or code generation not implemented.")


def test_generic_simd_constructor_codegen():
    code = """
    fn main():
        let color = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let color = vec4(1.0, 2.0, 3.0, 4.0);" in generated_code


def test_generic_inline_array_constructor_codegen():
    code = """
    fn main():
        let values = InlineArray[Float32, 4](1.0, 2.0, 3.0, 4.0)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let values = InlineArray[Float32, 4](1.0, 2.0, 3.0, 4.0);" in generated_code


def test_generic_matrix_constructor_codegen():
    code = """
    fn main():
        let transform = Matrix[DType.float32, 3, 4](
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0
        )
        let precise = Matrix[DType.float64, 2, 2](1.0, 0.0, 0.0, 1.0)
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let transform = mat3x4(" in generated_code
    assert "let precise = dmat2(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "Matrix[DType" not in generated_code


def test_callable_array_element_codegen_remains_array_call():
    code = """
    fn main():
        let value = callbacks[i]()
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let value = callbacks[i]();" in generated_code


def test_array_access_codegen():
    code = """
    fn vertex_main() -> float4:
        var values: StaticTuple[Float32, 4]
        values[0] = 1.0
        values[1] = 2.0
        values[2] = 3.0
        values[3] = 4.0
        return float4(values[0], values[1], values[2], values[3])
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "[0]" in generated_code
        assert "[1]" in generated_code
    except SyntaxError:
        pytest.fail("Array access parsing or code generation not implemented.")


def test_array_member_access_chain_codegen():
    code = """
    fn main():
        let channel = values[0].x
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let channel = values[0].x;" in generated_code


def test_parenthesized_call_indexing_codegen():
    code = """
    fn main():
        let value = (factory())[i]
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "let value = factory()[i];" in generated_code


def test_ternary_operator_codegen():
    code = """
    fn vertex_main() -> float4:
        let x: Float32 = 0.5
        let result = x > 0.0 ? 1.0 : 0.0
        return float4(result, 0.0, 0.0, 1.0)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "?" in generated_code or "if" in generated_code
    except SyntaxError:
        pytest.fail("Ternary operator parsing or code generation not implemented.")


def test_import_codegen():
    code = """
    import math
    import simd as s
    from tensor import Tensor

    fn vertex_main() -> float4:
        let result = math.sin(0.5)
        return float4(result, 0.0, 0.0, 1.0)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "// import math" in generated_code
        assert "// import simd as s" in generated_code
        assert "// from tensor import Tensor" in generated_code
        assert "let result = sin(0.5);" in generated_code
    except SyntaxError:
        pytest.fail("Import parsing or code generation not implemented.")


def test_global_variable_codegen_preserves_typed_globals():
    code = """
    var exposure: Float32
    let sample_count: Int = 4
    var transform: Matrix[DType.float32, 4, 4]
    let scale = 1.0
    @position
    var global_data: Float32
    var bound_texture: Texture2D @binding(0)
    @group(0)
    @binding(1)
    var sampler: SamplerState
    @group(2)
    var combined_texture: Texture2D @binding(3)

    fn main():
        return
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "// Global Variables" in generated_code
    assert "float exposure;" in generated_code
    assert "int sample_count = 4;" in generated_code
    assert "mat4 transform;" in generated_code
    assert "let scale = 1.0;" in generated_code
    assert "float global_data @ Position;" in generated_code
    assert "sampler2D bound_texture @ binding(0);" in generated_code
    assert "SamplerState sampler @ group(0) @ binding(1);" in generated_code
    assert "sampler2D combined_texture @ group(2) @ binding(3);" in generated_code
    assert generated_code.index("float exposure;") < generated_code.index("void main")


def test_explicit_none_return_codegen_maps_to_void():
    code = """
    fn helper() -> None:
        return

    fn main() -> None:
        helper()
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void helper()" in generated_code
    assert "void main()" in generated_code
    assert "helper();" in generated_code
    assert "None helper" not in generated_code
    assert "None main" not in generated_code


def test_at_attribute_codegen():
    code = """
    struct MyStruct:
        @position
        var data: Int32

    @compute_shader
    fn kernel():
        pass
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "int data @ Position;" in generated_code
        assert "// Compute Shader" in generated_code
        assert "compute {" in generated_code
        assert "void kernel() {" in generated_code
        assert "kernel()@ compute_shader" not in generated_code
    except SyntaxError:
        pytest.fail("Attribute parsing or code generation not implemented.")


def test_bracket_attribute_codegen_uses_shader_stage():
    code = """
    [[compute_shader]]
    fn run():
        pass
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "// Compute Shader" in generated_code
    assert "compute {" in generated_code
    assert "void run() {" in generated_code
    assert "run()@ compute_shader" not in generated_code


def test_class_codegen_uses_struct_shape_and_method_functions():
    code = """
    class Accumulator:
        @position
        var value: Int
        Matrix[DType.float32, 4, 4] transform @binding(0)
        SIMD[DType.float32, 4] tint @color
        fn reset():
            pass
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "struct Accumulator" in generated_code
        assert "int value @ Position;" in generated_code
        assert "mat4 transform @ binding(0);" in generated_code
        assert "vec4 tint @ Color;" in generated_code
        assert "void reset() {" in generated_code
    except SyntaxError:
        pytest.fail("Class parsing or code generation not implemented.")


def test_constant_buffer_codegen_from_colon_body():
    code = """
    @group(0)
    constant Params:
        @binding(0)
        var exposure: Float32 @offset(0)
        count: Int @binding(1)
        Matrix[DType.float32, 4, 4] transform @binding(2)
        SIMD[DType.float32, 4] tint @binding(3)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "cbuffer Params @ group(0)" in generated_code
        assert "float exposure @ binding(0) @ offset(0);" in generated_code
        assert "int count @ binding(1);" in generated_code
        assert "mat4 transform @ binding(2);" in generated_code
        assert "vec4 tint @ binding(3);" in generated_code
    except SyntaxError:
        pytest.fail("Constant buffer parsing or code generation not implemented.")


if __name__ == "__main__":
    pytest.main()
