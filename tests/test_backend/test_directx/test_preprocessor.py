import pytest

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor


def test_object_like_macro_expansion():
    code = """
    #define PI 3.14
    float value = PI;
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "PI" not in output
    assert "3.14" in output


def test_function_like_macro_expansion():
    code = """
    #define MUL(a, b) ((a) * (b))
    float value = MUL(2, 3);
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "MUL" not in output
    assert "2" in output and "3" in output


def test_variadic_stringize_and_paste():
    code = """
    #define LOG(fmt, ...) printf(fmt, __VA_ARGS__)
    #define STR(x) #x
    #define CAT(a, b) a##b
    int CAT(foo, bar) = 0;
    const char* s = STR(hello world);
    LOG("%d", 42);
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "foobar" in output
    assert "\"hello world\"" in output
    assert "printf" in output and "42" in output


def test_include_with_search_path(tmp_path):
    header = tmp_path / "include.hlsl"
    header.write_text("float4 includedValue;\n", encoding="utf-8")
    code = '#include "include.hlsl"\nfloat4 main() { return includedValue; }'
    output = HLSLPreprocessor(include_paths=[str(tmp_path)]).preprocess(
        code, file_path=str(tmp_path / "main.hlsl")
    )
    assert "includedValue" in output


def test_conditional_compilation_and_undef():
    code = """
    #define ENABLED 1
    #if ENABLED
    float ok = 1.0;
    #else
    float bad = ;
    #endif
    #undef ENABLED
    #ifdef ENABLED
    float broken = ;
    #else
    float still_ok = 2.0;
    #endif
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "bad" not in output
    assert "still_ok" in output


def test_line_directive_is_consumed():
    code = """
    #line 200
    float value = 1.0;
    """
    output = HLSLPreprocessor().preprocess(code)
    assert "#line" not in output
    assert "float value" in output


def test_error_and_warning_directives():
    error_code = "#error fail"
    with pytest.raises(SyntaxError):
        HLSLPreprocessor().preprocess(error_code)

    warning_code = "#warning warn"
    HLSLPreprocessor().preprocess(warning_code)

    with pytest.raises(SyntaxError):
        HLSLPreprocessor(strict=True).preprocess(warning_code)
