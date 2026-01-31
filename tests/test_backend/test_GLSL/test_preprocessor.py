import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer


def test_preprocessor_conditional_expansion():
    code = """
    #version 450 core
    #define ENABLE_FEATURE 1
    #if ENABLE_FEATURE
    int only_enabled = 1;
    #else
    int only_disabled = 1;
    #endif
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "only_enabled" in values
    assert "only_disabled" not in values


def test_preprocessor_include(tmp_path):
    include_file = tmp_path / "inc.glsl"
    include_file.write_text("int from_include = 1;")

    code = """
    #version 450 core
    #include <inc.glsl>
    void main() { }
    """
    tokens = GLSLLexer(code, include_paths=[str(tmp_path)]).tokenize()
    values = [tok[1] for tok in tokens]
    assert "from_include" in values


def test_preprocessor_version_must_be_first():
    code = """
    #define SOMETHING 1
    #version 450 core
    void main() { }
    """
    with pytest.raises(SyntaxError):
        GLSLLexer(code).tokenize()


def test_preprocessor_function_like_macro():
    code = """
    #version 450 core
    #define ADD(a, b) ((a) + (b))
    int value = ADD(1, 2);
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "value" in values
    assert "1" in values
    assert "2" in values


def test_preprocessor_variadic_macro_and_paste():
    code = """
    #version 450 core
    #define JOIN(a, b) a##b
    #define MAKE(name, ...) int JOIN(name, _value) = __VA_ARGS__;
    MAKE(test, 7)
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "test_value" in values
    assert "7" in values


def test_preprocessor_stringize():
    code = """
    #version 450 core
    #define STR(x) #x
    const char* name = STR(hello);
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert '"hello"' in values


def test_preprocessor_extension_directive():
    code = """
    #version 450 core
    #extension GL_ARB_separate_shader_objects : enable
    void main() { }
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "extension" in values
    assert "GL_ARB_separate_shader_objects" in values


def test_preprocessor_error_directive_raises():
    code = """
    #version 450 core
    #error stop
    void main() { }
    """
    with pytest.raises(SyntaxError):
        GLSLLexer(code).tokenize()


def test_preprocessor_line_directive():
    code = """
    #version 450 core
    #line 42
    int value = 1;
    """
    tokens = GLSLLexer(code).tokenize()
    values = [tok[1] for tok in tokens]
    assert "value" in values


if __name__ == "__main__":
    pytest.main()
