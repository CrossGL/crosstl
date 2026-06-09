from typing import List

import pytest

from crosstl.backend.slang.SlangLexer import SlangLexer
from crosstl.backend.slang.SlangParser import SlangParser


def tokenize_code(code: str) -> List:
    lexer = SlangLexer(code)
    return lexer.tokenize()


def test_slang_lexer_struct_tokenization():
    code = """
    struct AssembledVertex
    {
    float3	position : POSITION;
    };

    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("struct tokenization not implemented.")


def test_if_tokenization():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("if tokenization not implemented.")


def test_for_tokenization():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        for (int i = 0; i < 10; i++) {
            output.out_position += assembledVertex.position;
        }

        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("for tokenization not implemented.")


def test_else_tokenization():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        else {
            output.out_position = float3(0.0, 0.0, 0.0);
        }
        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("else tokenization not implemented.")


def test_function_call_tokenization():
    code = """
    float4 saturate(float4 color) {
        return color;
    }

    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        output.out_position = saturate(assembledVertex.color);
        return output;
    }
    """
    try:
        tokenize_code(code)
    except SyntaxError:
        pytest.fail("Function call tokenization not implemented.")


def test_mod_tokenization():
    code = """
        int a = 10 % 3;  // Basic modulus
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
        int a = ~5;  // Bitwise NOT
    """
    tokens = tokenize_code(code)
    has_not = False
    for token in tokens:
        if token == ("BITWISE_NOT", "~"):
            has_not = True
            break
    assert has_not, "Bitwise NOT operator (~) not tokenized correctly"


def test_binary_bitwise_and_shift_tokenization():
    tokens = tokenize_code("a & b | c ^ d << 1 >> 2 && e || f;")

    assert ("BITWISE_AND", "&") in tokens
    assert ("BITWISE_OR", "|") in tokens
    assert ("BITWISE_XOR", "^") in tokens
    assert ("BITWISE_SHIFT_LEFT", "<<") in tokens
    assert ("BITWISE_SHIFT_RIGHT", ">>") in tokens
    assert ("AND", "&&") in tokens
    assert ("OR", "||") in tokens


def test_compound_bitwise_and_shift_assignment_tokenization():
    tokens = tokenize_code("a %= 3; a &= b; a |= c; a ^= d; a <<= 1; a >>= 2;")

    assert ("ASSIGN_MOD", "%=") in tokens
    assert ("ASSIGN_AND", "&=") in tokens
    assert ("ASSIGN_OR", "|=") in tokens
    assert ("ASSIGN_XOR", "^=") in tokens
    assert ("ASSIGN_SHIFT_LEFT", "<<=") in tokens
    assert ("ASSIGN_SHIFT_RIGHT", ">>=") in tokens


def test_logical_not_tokenization():
    tokens = tokenize_code("bool enabled = !disabled; disabled != enabled;")

    assert ("NOT", "!") in tokens
    assert ("NOT_EQUAL", "!=") in tokens


def test_increment_decrement_tokenization():
    tokens = tokenize_code("i++; --j;")

    assert ("INCREMENT", "++") in tokens
    assert ("DECREMENT", "--") in tokens


def test_lambda_arrow_tokenization():
    tokens = tokenize_code("auto f = (int x) => x + 1;")

    assert ("FAT_ARROW", "=>") in tokens


def test_declaration_qualifier_tokenization():
    tokens = tokenize_code("static inline constexpr const float value;")

    assert ("STATIC", "static") in tokens
    assert ("INLINE", "inline") in tokens
    assert ("CONSTEXPR", "constexpr") in tokens
    assert ("CONST", "const") in tokens


def test_interface_extension_and_where_tokenization():
    tokens = tokenize_code(
        "interface IFoo { int foo(); } "
        "extension MyType : IFoo { int foo(); } "
        "int use<T>(T value) where T : IFoo { return value.foo(); }"
    )

    assert ("INTERFACE", "interface") in tokens
    assert ("EXTENSION", "extension") in tokens
    assert ("WHERE", "where") in tokens


def test_dunder_extension_tokenization_from_compute_fixture():
    # Reduced from shader-slang/slang tests/compute/extension-multi-interface.slang.
    tokens = tokenize_code("__extension Simple : ISub { int subf(); };")

    assert ("EXTENSION", "__extension") in tokens
    assert ("IDENTIFIER", "__extension") not in tokens


def test_typealias_and_associatedtype_tokenization():
    tokens = tokenize_code(
        "typealias Color = float4; "
        "interface IMaterial { associatedtype BRDF : IBRDF; }"
    )

    assert ("TYPEALIAS", "typealias") in tokens
    assert ("ASSOCIATEDTYPE", "associatedtype") in tokens


def test_numeric_literal_tokenization():
    tokens = tokenize_code("1e-3f 1.0f .5f 1. 0xffu 0b1010 0B1111 123u")

    assert tokens == [
        ("NUMBER", "1e-3f"),
        ("NUMBER", "1.0f"),
        ("NUMBER", ".5f"),
        ("NUMBER", "1."),
        ("NUMBER", "0xffu"),
        ("NUMBER", "0b1010"),
        ("NUMBER", "0B1111"),
        ("NUMBER", "123u"),
        ("EOF", ""),
    ]


def test_numeric_literal_underscore_tokenization_from_generated_conformance_sample():
    # Source: shader-slang/slang docs/generated/tests/conformance/
    # lexical-structure/integer-literal-underscore-ignored.slang at d25453d.
    tokens = tokenize_code("1_000_000 0x_FF_FF 0b_1010_0101 1_2.3_4e+5_6f")

    assert tokens == [
        ("NUMBER", "1000000"),
        ("NUMBER", "0xFFFF"),
        ("NUMBER", "0b10100101"),
        ("NUMBER", "12.34e+56f"),
        ("EOF", ""),
    ]


def test_from_file_decodes_utf16_bom_crlf_preprocessor_fixtures(tmp_path):
    # Source: shader-slang/slang tests/preprocessor/utf16_{le,be}_bom_crlf.slang
    # at 6b9f98ff90facc35306a0ba643dfecb59a870156.
    source = "void main()\r\n{\r\n}\r\n//TEST:SIMPLE\r\n"
    encodings = [
        ("le", b"\xff\xfe", "utf-16-le"),
        ("be", b"\xfe\xff", "utf-16-be"),
    ]

    for suffix, bom, encoding in encodings:
        source_path = tmp_path / f"utf16_{suffix}_bom_crlf.slang"
        source_path.write_bytes(bom + source.encode(encoding))

        tokens = SlangLexer.from_file(str(source_path)).tokenize()
        ast = SlangParser(tokens).parse()

        assert [function.name for function in ast.functions] == ["main"]


if __name__ == "__main__":
    pytest.main()
