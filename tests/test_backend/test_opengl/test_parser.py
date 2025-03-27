#!/usr/bin/env python
"""
Test for the OpenGL parser.
"""
import os
import sys
import pytest

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Now import the modules
from crosstl.backend.OpenGL.opengllexer import GLSLLexer
from crosstl.backend.OpenGL.openglparser import GLSLParser


def test_basic_parsing():
    """Test basic GLSL parsing functionality."""
    code = """
    void main() {
        float x = 1.0;
        gl_Position = vec4(x, 0.0, 0.0, 1.0);
    }
    """
    lexer = GLSLLexer(code)
    tokens = lexer.tokenize()
    parser = GLSLParser(tokens)
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


def test_mod_parsing():
    code = """
    
    void main() {
        int a = 10 % 3;  // Basic modulus
    }
    """
    try:
        lexer = GLSLLexer(code)
        tokens = lexer.tokenize()
        parser = GLSLParser(tokens)
        parser.parse()
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
        lexer = GLSLLexer(code)
        tokens = lexer.tokenize()
        parser = GLSLParser(tokens)
        parser.parse()
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing not implemented")


if __name__ == "__main__":
    test_basic_parsing()
    test_mod_parsing()
    test_bitwise_not_parsing()
    print("All tests passed!")
