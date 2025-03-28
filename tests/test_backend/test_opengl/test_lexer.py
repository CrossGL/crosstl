#!/usr/bin/env python
"""
Test for the OpenGL lexer.
"""
import os
import sys
import pytest

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Now import the module
from crosstl.backend.OpenGL.opengllexer import GLSLLexer


def test_basic_lexing():
    """Test basic GLSL lexing functionality."""
    code = """
    void main() {
        float x = 1.0;
        gl_Position = vec4(x, 0.0, 0.0, 1.0);
    }
    """
    lexer = GLSLLexer(code)
    tokens = lexer.tokenize()

    # Verify we have tokens
    assert len(tokens) > 0

    # Check for specific tokens
    token_types = [t[0] for t in tokens]  # tokens are (type, value) tuples
    assert "VOID" in token_types
    assert "IDENTIFIER" in token_types  # 'main'
    assert "FLOAT" in token_types
    assert "EQUALS" in token_types
    assert "NUMBER" in token_types  # 1.0


if __name__ == "__main__":
    test_basic_lexing()
    print("All tests passed!")
