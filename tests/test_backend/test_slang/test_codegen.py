#!/usr/bin/env python
"""
Test for the Slang code generator.
"""
import os
import sys
import pytest

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

# Now import the modules
from crosstl.backend.Slang.slanglexer import SlangLexer
from crosstl.backend.Slang.slangparser import SlangParser
from crosstl.backend.Slang.slangcrossglcodegen import SlangToCrossGLConverter


def test_struct_definition():
    """Test that struct definitions are properly converted to CrossGL."""
    slang_code = """
    struct Material {
        float3 diffuse;
        float shininess;
    };
    
    cbuffer MaterialBuffer {
        Material material;
    };
    
    float4 main(float2 uv : TEXCOORD) : SV_TARGET {
        float3 color = material.diffuse;
        return float4(color, 1.0);
    }
    """
    lexer = SlangLexer(slang_code)
    tokens = lexer.tokenize()
    parser = SlangParser(tokens)
    ast = parser.parse()
    converter = SlangToCrossGLConverter()
    result = converter.generate(ast)

    assert "struct Material" in result
    assert "diffuse" in result
    assert "shininess" in result


if __name__ == "__main__":
    test_struct_definition()
    print("All tests passed!")
