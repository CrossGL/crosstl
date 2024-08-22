import pytest
from typing import List
from crosstl.src.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.src.backend.DirectX.DirectxParser import HLSLParser


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = HLSLParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = HLSLLexer(code)
    return lexer.tokens


def test_struct_parsing():
    code = """
    struct VSInput {
        float4 position : POSITION;
        float4 color : TEXCOORD0;
    };

    struct VSOutput {
        float4 out_position : TEXCOORD0;
    };
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_if_parsing():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.out_position = input.position;
        if (input.color.r > 0.5) {
            output.out_position = input.color;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_for_parsing():
    code = """
    VSOutput VSMain(VSInput input) {
        VSOutput output;
        for (int i = 0; i < 10; i=i+1) {
            output.out_position = input.position;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_else_parsing():
    code = """
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        if (input.in_position.r > 0.5) {
            output.out_color = input.in_position;
        } else {
            output.out_color = float4(0.0, 0.0, 0.0, 1.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_function_call_parsing():
    code = """
    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.out_color = saturate(input.in_position);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


# Run all tests
if __name__ == "__main__":
    pytest.main()
