#!/usr/bin/env python
"""
Test for the Slang lexer.
"""
import os
import sys
import pytest

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Now import the module
from crosstl.backend.Slang.slanglexer import SlangLexer


def test_basic_lexing():
    """Test basic Slang lexing functionality."""
    code = """
    float4 main(float2 uv : TEXCOORD) : SV_TARGET {
        float x = 1.0;
        return float4(x, 0.0, 0.0, 1.0);
    }
    """
    lexer = SlangLexer(code)
    tokens = lexer.tokenize()

    # Verify we have tokens
    assert len(tokens) > 0

    # Check for specific tokens
    token_types = [t[0] for t in tokens]  # tokens are (type, value) tuples
    assert "IDENTIFIER" in token_types  # 'main', 'float4', etc.
    assert "FLOAT_KW" in token_types
    assert "EQUALS" in token_types
    assert "NUMBER" in token_types  # 1.0


if __name__ == "__main__":
    test_basic_lexing()
    print("All tests passed!")
