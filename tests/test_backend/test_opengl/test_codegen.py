#!/usr/bin/env python
"""
Test for the OpenGL code generator.
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
from crosstl.backend.OpenGL.openglcrossglcodegen import GLSLToCrossGLConverter


def test_struct_definition():
    """Test that struct definitions are properly converted to CrossGL."""
    glsl_code = """
    struct Material {
        vec3 diffuse;
        float shininess;
    };
    
    uniform Material material;
    
    void main() {
        vec3 color = material.diffuse;
        gl_FragColor = vec4(color, 1.0);
    }
    """
    lexer = GLSLLexer(glsl_code)
    tokens = lexer.tokenize()
    parser = GLSLParser(tokens)
    ast = parser.parse()
    converter = GLSLToCrossGLConverter()
    result = converter.generate(ast)

    assert "struct Material" in result
    assert "diffuse" in result
    assert "shininess" in result


if __name__ == "__main__":
    test_struct_definition()
    print("All tests passed!")
