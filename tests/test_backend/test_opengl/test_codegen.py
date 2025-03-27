import pytest

from crosstl.backend.OpenGL.OpenglLexer import GLSLLexer
from crosstl.backend.OpenGL.OpenglParser import GLSLParser
from crosstl.backend.OpenGL.OpenglCrossGLCodeGen import GLSLToCrossGLConverter


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
    pytest.main()
