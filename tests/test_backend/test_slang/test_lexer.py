from typing import List

import pytest

from crosstl.backend.slang.SlangLexer import SlangLexer


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
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


def test_numeric_literal_tokenization():
    tokens = tokenize_code("1e-3f 1.0f .5f 1. 0xffu 123u")

    assert tokens == [
        ("NUMBER", "1e-3f"),
        ("NUMBER", "1.0f"),
        ("NUMBER", ".5f"),
        ("NUMBER", "1."),
        ("NUMBER", "0xffu"),
        ("NUMBER", "123u"),
        ("EOF", ""),
    ]


if __name__ == "__main__":
    pytest.main()
