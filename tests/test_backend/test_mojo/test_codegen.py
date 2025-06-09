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
        assert "else:" in generated_code
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
        assert "elif " in generated_code
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

    fn vertex_main() -> float4:
        let result = math.sin(0.5)
        return float4(result, 0.0, 0.0, 1.0)
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "import math" in generated_code or "// import math" in generated_code
    except SyntaxError:
        pytest.fail("Import parsing or code generation not implemented.")


if __name__ == "__main__":
    pytest.main()
