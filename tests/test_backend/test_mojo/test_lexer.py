import pytest
from typing import List
from crosstl.backend.Mojo.MojoLexer import MojoLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
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
        a %= 3
        a ^= 1
        a |= 2
        a &= 7
        a <<= 1
        a >>= 2
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Assignment operators tokenization not implemented.")


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
        tokenize_code(code)
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
        tokenize_code(code)
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
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Attributes tokenization not implemented.")


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
        # Comments should be tokenized but filtered out
        comment_found = any(token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"] for token in tokens)
        # Since we skip comments in tokenization, we just ensure no error occurs
    except SyntaxError:
        pytest.fail("Comments tokenization not implemented.")


def test_mod_tokenization():
    code = """
        let a: Int = 10 % 3  # Basic modulus
    """
    tokens = tokenize_code(code)

    # Find the modulus operator in tokens
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
    code = '''
    fn main():
        let message: String = "Hello, Mojo!"
        let path: String = "path/to/file.txt"
    '''
    try:
        tokens = tokenize_code(code)
        string_literals = [token[1] for token in tokens if token[0] == "STRING_LITERAL"]
        assert '"Hello, Mojo!"' in string_literals, "String literal not tokenized correctly"
        assert '"path/to/file.txt"' in string_literals, "String literal not tokenized correctly"
    except SyntaxError:
        pytest.fail("String literals tokenization not implemented.")


def test_numeric_literals_tokenization():
    code = """
    fn main():
        let decimal: Int = 42
        let hex: Int = 0x2A
        let binary: Int = 0b101010
        let octal: Int = 0o52
        let float_val: Float32 = 3.14159
        let scientific: Float32 = 1.23e-4
    """
    try:
        tokens = tokenize_code(code)
        numbers = [token[1] for token in tokens if token[0] == "NUMBER"]
        assert "42" in numbers, "Decimal literal not tokenized correctly"
        assert "3.14159" in numbers, "Float literal not tokenized correctly"
    except SyntaxError:
        pytest.fail("Numeric literals tokenization not implemented.")


if __name__ == "__main__":
    pytest.main()
