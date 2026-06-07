from typing import List

import pytest

from crosstl.backend.Mojo.MojoLexer import MojoLexer


def tokenize_code(code: str) -> List:
    lexer = MojoLexer(code)
    return lexer.tokenize()


def test_struct_tokenization():
    code = """
    struct VSInput:
        var position: float4
        var color: float2

    struct VSOutput:
        var out_position: float2
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Struct tokenization not implemented.")


def test_if_tokenization():
    code = """
    fn vertex_main(input: VSInput) -> VSOutput:
        var output: VSOutput
        output.out_position = input.position.xy
        if input.color.x > 0.5:
            output.out_position = input.color
        return output
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("If tokenization not implemented.")


def test_for_tokenization():
    code = """
    fn main():
        for var i: Int = 0; i < 10; i = i + 1:
            print(i)
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("For tokenization not implemented.")


def test_while_tokenization():
    code = """
    fn main():
        var i: Int = 0
        while i < 10:
            i = i + 1
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("While tokenization not implemented.")


def test_else_tokenization():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        if input.in_position.x > 0.5:
            output.out_color = input.in_position
        else:
            output.out_color = float4(0.0, 0.0, 0.0, 1.0)
        return output
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Else tokenization not implemented.")


def test_function_call_tokenization():
    code = """
    fn fragment_main(input: PSInput) -> PSOutput:
        var output: PSOutput
        output.out_color = normalize(input.in_position)
        return output
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_def_function_tokenization():
    code = """
    def helper(x: Float32) -> Float32:
        return x
    """
    try:
        tokens = tokenize_code(code)
        assert ("DEF", "def") in tokens
        assert ("IDENTIFIER", "helper") in tokens
    except SyntaxError:
        pytest.fail("Def function tokenization not implemented.")


def test_else_if_tokenization():
    code = """
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Else-if tokenization not implemented.")


def test_assignment_ops_tokenization():
    code = """
    fn fragment_main():
        var a: Int = 5
        a += 3
        a -= 2
        a *= 4
        a /= 2
        a //= 2
        a %= 3
        a ^= 1
        a |= 2
        a &= 7
        a <<= 1
        a >>= 2
    """
    try:
        tokens = tokenize_code(code)
        assert ("FLOOR_DIVIDE_EQUALS", "//=") in tokens
    except SyntaxError:
        pytest.fail("Assignment operators tokenization not implemented.")


def test_power_operator_tokenization():
    code = """
    fn my_pow(base: Int, exp: Int = 2) -> Int:
        return base ** exp
    """
    tokens = tokenize_code(code)

    assert ("POWER", "**") in tokens


def test_bitwise_ops_tokenization():
    code = """
    fn main():
        let val: UInt32 = 0x01
        let result = val | 0x02
        let result2 = val & 0x04
        let result3 = val ^ 0x08
        let result4 = ~val
        let result5 = val << 2
        let result6 = val >> 1
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Bitwise operators tokenization not implemented.")


def test_logical_ops_tokenization():
    code = """
    fn main():
        let val_0: Bool = True
        let val_1 = val_0 and False
        let val_2 = val_0 or False
        let val_3 = not val_0
    """
    try:
        tokens = tokenize_code(code)
        assert ("BOOL_LITERAL", "true") in tokens
        assert ("BOOL_LITERAL", "false") in tokens
        assert tokens.count(("AND", "&&")) == 1
        assert tokens.count(("OR", "||")) == 1
        assert tokens.count(("NOT", "!")) == 1
    except SyntaxError:
        pytest.fail("Logical operators tokenization not implemented.")


def test_switch_case_tokenization():
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
            default:
                output.out_color = float4(0.0, 0.0, 1.0, 1.0)
                break
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Switch-case tokenization not implemented.")


def test_import_tokenization():
    code = """
    import math
    import simd as s
    from tensor import Tensor
    """
    try:
        tokens = tokenize_code(code)
        assert ("FROM", "from") in tokens
        assert ("IMPORT", "import") in tokens
        assert ("IDENTIFIER", "Tensor") in tokens
    except SyntaxError:
        pytest.fail("Import tokenization not implemented.")


def test_data_types_tokenization():
    code = """
    fn main():
        var a: Int8 = 10
        var b: Int16 = 20
        var c: Int32 = 30
        var d: Int64 = 40
        var e: UInt8 = 50
        var f: UInt16 = 60
        var g: UInt32 = 70
        var h: UInt64 = 80
        var i: Float16 = 1.5
        var j: Float32 = 2.5
        var k: Float64 = 3.5
        var l: Bool = True
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Data types tokenization not implemented.")


def test_vector_types_tokenization():
    code = """
    fn main():
        var a: float2 = float2(1.0, 2.0)
        var b: float3 = float3(1.0, 2.0, 3.0)
        var c: float4 = float4(1.0, 2.0, 3.0, 4.0)
        var d: int2 = int2(1, 2)
        var e: int3 = int3(1, 2, 3)
        var f: int4 = int4(1, 2, 3, 4)
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Vector types tokenization not implemented.")


def test_attributes_tokenization():
    code = """
    @value
    struct MyStruct:
        var data: Int32

    @compute_shader
    fn compute_main():
        pass
    """
    try:
        tokens = tokenize_code(code)
        assert tokens.count(("AT", "@")) == 2
        assert ("IDENTIFIER", "value") in tokens
        assert ("IDENTIFIER", "compute_shader") in tokens
    except SyntaxError:
        pytest.fail("Attributes tokenization not implemented.")


def test_bracket_attributes_tokenize_as_single_attribute():
    code = """
    [[compute_shader]]
    fn compute_main():
        pass
    """

    tokens = tokenize_code(code)

    assert ("ATTRIBUTE", "[[compute_shader]]") in tokens
    assert tokens[:2] != [("LBRACKET", "["), ("LBRACKET", "[")]


def test_ternary_operator_tokenization():
    code = """
    fn main():
        let x: Float32 = 0.5
        let result = x > 0.0 ? 1.0 : 0.0
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Ternary operator tokenization not implemented.")


def test_array_access_tokenization():
    code = """
    fn main():
        var values: StaticTuple[Float32, 4]
        values[0] = 1.0
        values[1] = 2.0
        let result = values[0] + values[1]
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Array access tokenization not implemented.")


def test_comments_tokenization():
    code = """
    # Single line comment
    fn main():
        var a: Int = 5  # End of line comment
        \"\"\"
        Multi-line comment
        spanning multiple lines
        \"\"\"
        return a
    """
    try:
        tokens = tokenize_code(code)
        comment_found = any(
            token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"] for token in tokens
        )
        # Since we skip comments in tokenization, we just ensure no error occurs
    except SyntaxError:
        pytest.fail("Comments tokenization not implemented.")


def test_hash_prefixed_lines_are_comments_not_preprocessor_tokens():
    code = """
    #define ENABLED 1
    #if ENABLED
    import math
    #endif

    fn main():
        let value: Int = 1
    """
    tokens = tokenize_code(code)
    token_values = [value for _, value in tokens]

    assert "define" not in token_values
    assert "ENABLED" not in token_values
    assert "if" not in token_values
    assert ("IMPORT", "import") in tokens
    assert ("IDENTIFIER", "math") in tokens


def test_mod_tokenization():
    code = """
        let a: Int = 10 % 3  # Basic modulus
    """
    tokens = tokenize_code(code)

    has_mod = False
    for token in tokens:
        if token == ("MOD", "%"):
            has_mod = True
            break

    assert has_mod, "Modulus operator (%) not tokenized correctly"


def test_bitwise_not_tokenization():
    code = """
        let a: Int = ~5  # Bitwise NOT
    """
    tokens = tokenize_code(code)
    has_not = False
    for token in tokens:
        if token == ("BITWISE_NOT", "~"):
            has_not = True
            break
    assert has_not, "Bitwise NOT operator (~) not tokenized correctly"


def test_string_literals_tokenization():
    code = """
    fn main():
        let message: String = "Hello, Mojo!"
        let path: String = "path/to/file.txt"
        let status: String = 'done'
        let escaped: String = 'it\\'s ready'
    """
    try:
        tokens = tokenize_code(code)
        string_literals = [token[1] for token in tokens if token[0] == "STRING_LITERAL"]
        assert (
            '"Hello, Mojo!"' in string_literals
        ), "String literal not tokenized correctly"
        assert (
            '"path/to/file.txt"' in string_literals
        ), "String literal not tokenized correctly"
        assert (
            "'done'" in string_literals
        ), "Single-quoted string not tokenized correctly"
        assert (
            "'it\\'s ready'" in string_literals
        ), "Escaped single quote not tokenized correctly"
    except SyntaxError:
        pytest.fail("String literals tokenization not implemented.")


def test_triple_quoted_string_literal_tokenization_from_mojo_reference():
    # Reduced from https://mojolang.org/docs/reference/literals/
    code = '''
    fn main():
        let message = """Multi-line
string"""
    '''
    tokens = tokenize_code(code)

    assert ("STRING_LITERAL", '"Multi-line\\nstring"') in tokens


def test_backtick_metadata_identifiers_tokenization_from_modular_kernels():
    code = """
    @__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
    fn kernel():
        pass
    """

    tokens = tokenize_code(code)

    assert ("BACKTICK_IDENTIFIER", "`nvvm.cluster_dim`") in tokens


def test_backslash_line_continuation_tokenization_from_modular_kernels():
    code = """
    fn main():
        comptime has_static_NK = (b.static_shape[0] > -1 and b.static_shape[1] > -1) \\
                      and a.static_shape[1] > -1
    """

    tokens = tokenize_code(code)

    assert ("IDENTIFIER", "has_static_NK") in tokens
    assert ("AND", "&&") in tokens


def test_numeric_literals_tokenization():
    code = """
    fn main():
        let decimal: Int = 42
        let grouped: Int = 1_000_000
        let relaxed: Int = 1__000_
        let hex: Int = 0x2A
        let hex_grouped: Int = 0xFF_FF
        let binary: Int = 0b101010
        let binary_grouped: Int = 0b1010_
        let octal: Int = 0o52
        let octal_grouped: Int = 0o52_
        let float_val: Float32 = 3.14159
        let fraction_only: Float32 = .5
        let trailing_point: Float32 = 2.
        let grouped_float: Float32 = 1_000.000_5
        let scientific: Float32 = 1.23e-4
        let exponent_only: Float32 = 1E10
    """
    try:
        tokens = tokenize_code(code)
        numbers = [token[1] for token in tokens if token[0] == "NUMBER"]
        assert "42" in numbers, "Decimal literal not tokenized correctly"
        assert "1_000_000" in numbers, "Grouped decimal literal not tokenized correctly"
        assert "1__000_" in numbers, "Relaxed decimal literal not tokenized correctly"
        assert "0xFF_FF" in numbers, "Grouped hex literal not tokenized correctly"
        assert "0b1010_" in numbers, "Grouped binary literal not tokenized correctly"
        assert "0o52_" in numbers, "Grouped octal literal not tokenized correctly"
        assert "3.14159" in numbers, "Float literal not tokenized correctly"
        assert ".5" in numbers, "Fraction-only float literal not tokenized correctly"
        assert "2." in numbers, "Trailing-point float literal not tokenized correctly"
        assert "1_000.000_5" in numbers, "Grouped float literal not tokenized correctly"
        assert "1E10" in numbers, "Exponent float literal not tokenized correctly"
    except SyntaxError:
        pytest.fail("Numeric literals tokenization not implemented.")


if __name__ == "__main__":
    pytest.main()
