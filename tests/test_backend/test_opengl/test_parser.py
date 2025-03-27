import pytest

# Try both import paths
try:
    from crosstl.backend.OpenGL.OpenglLexer import GLSLLexer
    from crosstl.backend.OpenGL.OpenglParser import GLSLParser
except ImportError:
    from crosstl.backend.OpenGL.OpenglLexer import GLSLLexer
    from crosstl.backend.OpenGL.OpenglParser import GLSLParser


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
    pytest.main()
