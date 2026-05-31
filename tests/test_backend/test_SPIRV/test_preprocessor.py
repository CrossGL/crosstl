import pytest

from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser


def token_values(code, **lexer_options):
    return [value for _, value in VulkanLexer(code, **lexer_options).tokenize()]


def test_preprocessor_conditional_expansion():
    values = token_values("""
        #version 450
        #define ENABLE_BINDING 1
        #if ENABLE_BINDING
        layout(set = 0, binding = 0) uniform sampler2D enabledTex;
        #else
        layout(set = 0, binding = 1) uniform sampler2D disabledTex;
        #endif
        void main() { }
    """)

    assert "enabledTex" in values
    assert "disabledTex" not in values
    assert "ENABLE_BINDING" not in values


def test_preprocessor_include_with_search_path(tmp_path):
    include_file = tmp_path / "bindings.glsl"
    include_file.write_text(
        "layout(set = 0, binding = 3) uniform sampler2D includedTex;",
        encoding="utf-8",
    )

    values = token_values(
        """
        #version 450
        #include <bindings.glsl>
        void main() { }
        """,
        include_paths=[str(tmp_path)],
    )

    assert "includedTex" in values


def test_preprocessor_function_like_macro_expansion():
    values = token_values("""
        #version 450
        #define MAKE_BINDING(name, binding_index) \
            layout(set = 0, binding = binding_index) uniform sampler2D name;
        MAKE_BINDING(albedoTex, 2)
        void main() { }
    """)

    assert "albedoTex" in values
    assert "2" in values
    assert "MAKE_BINDING" not in values


def test_preprocessor_version_must_be_first():
    with pytest.raises(SyntaxError, match="#version must appear before"):
        VulkanLexer("""
            #define VALUE 1
            #version 450
            void main() { }
        """).tokenize()


def test_parser_uses_preprocessed_conditionals():
    tokens = VulkanLexer("""
        #version 450
        #define USE_SELECTED 1
        #if USE_SELECTED
        layout(set = 0, binding = 0) uniform sampler2D selectedTex;
        #else
        layout(set = 0, binding = 1) uniform sampler2D rejectedTex;
        #endif
        void main() { }
    """).tokenize()

    ast = VulkanParser(tokens).parse()
    variable_names = [
        getattr(node, "variable_name", None) for node in ast.global_variables
    ]
    assert "selectedTex" in variable_names
    assert "rejectedTex" not in variable_names
