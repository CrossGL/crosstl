import pytest
from crosstl.backend.OpenGL.OpenglLexer import GLSLLexer


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
    pytest.main()
