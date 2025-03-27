#!/usr/bin/env python
"""
Test for the Slang parser.
"""
import os
import sys
from typing import List
import pytest

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Now import the modules
from crosstl.backend.Slang.slanglexer import SlangLexer
from crosstl.backend.Slang.slangparser import SlangParser


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = SlangParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = SlangLexer(code)
    return lexer.tokenize()


def test_mod_parsing():
    code = """
    
    void main() {
        int a = 10 % 3;  // Basic modulus
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_bitwise_not_parsing():
    code = """
    void main() {
        int a = 5;
        int b = ~a;  // Bitwise NOT
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing not implemented")


def test_basic_parsing():
    """Test basic Slang parsing functionality."""
    code = """
    float4 main(float2 uv : TEXCOORD) : SV_TARGET {
        float x = 1.0;
        return float4(x, 0.0, 0.0, 1.0);
    }
    """
    lexer = SlangLexer(code)
    tokens = lexer.tokenize()
    parser = SlangParser(tokens)
    ast = parser.parse()

    # Verify we have an AST
    assert ast is not None

    # Verify the AST structure - check that it's a ShaderNode
    assert hasattr(ast, "functions")
    assert len(ast.functions) > 0

    # Find the main function
    main_func = None
    for func in ast.functions:
        if func.name == "main":
            main_func = func
            break

    assert main_func is not None, "Main function not found in AST"


if __name__ == "__main__":
    test_mod_parsing()
    test_bitwise_not_parsing()
    test_basic_parsing()
    print("All tests passed!")
