import pytest

from crosstl.backend.slang import SlangLexer, SlangParser


def token_values(code, **lexer_options):
    return [value for _, value in SlangLexer(code, **lexer_options).tokenize()]


def test_preprocessor_conditional_expansion():
    values = token_values("""
        #define ENABLE_SHADER 1
        #if ENABLE_SHADER
        float enabledValue;
        #else
        float disabledValue;
        #endif
    """)

    assert "enabledValue" in values
    assert "disabledValue" not in values
    assert "ENABLE_SHADER" not in values


def test_preprocessor_include_with_search_path(tmp_path):
    include_file = tmp_path / "common.slang"
    include_file.write_text(
        "#define INCLUDED_VALUE 7\nfloat includedValue = INCLUDED_VALUE;\n",
        encoding="utf-8",
    )

    values = token_values(
        """
        #include "common.slang"
        float localValue;
        """,
        include_paths=[str(tmp_path)],
    )

    assert "includedValue" in values
    assert "7" in values
    assert "INCLUDED_VALUE" not in values


def test_preprocessor_function_like_macro_expansion():
    values = token_values("""
        #define SCALE(x) ((x) * 2.0)
        float scaled = SCALE(3.0);
    """)

    assert "SCALE" not in values
    assert "scaled" in values
    assert "3.0" in values
    assert "2.0" in values


def test_preprocessor_unknown_directive_is_filtered():
    values = token_values("""
        #pragma once
        float value;
    """)

    assert "pragma" not in values
    assert "value" in values


def test_preprocessor_error_directive_raises():
    with pytest.raises(SyntaxError, match="#error"):
        SlangLexer("""
            #error stop
            float value;
        """).tokenize()


def test_parser_uses_preprocessed_conditionals():
    tokens = SlangLexer("""
        #define USE_MATH 1
        #if USE_MATH
        import math;
        #else
        import rejected;
        #endif
    """).tokenize()

    ast = SlangParser(tokens).parse()
    imports = [import_node.module_name for import_node in ast.imports]
    assert imports == ["math"]
