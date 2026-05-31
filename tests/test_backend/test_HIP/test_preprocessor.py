import pytest

from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser


def token_values(code, **lexer_options):
    return [token.value for token in HipLexer(code, **lexer_options).tokenize()]


def test_preprocessor_conditional_expansion():
    values = token_values("""
        #define ENABLE_KERNEL 1
        #if ENABLE_KERNEL
        __global__ void enabled_kernel(float* out) { out[0] = 1.0f; }
        #else
        __global__ void disabled_kernel(float* out) { out[0] = 0.0f; }
        #endif
    """)

    assert "enabled_kernel" in values
    assert "disabled_kernel" not in values
    assert "ENABLE_KERNEL" not in values


def test_preprocessor_include_with_search_path(tmp_path):
    include_file = tmp_path / "constants.hip"
    include_file.write_text(
        "#define SCALE_VALUE 4\n"
        "__device__ float scale(float value) { return value * SCALE_VALUE; }\n",
        encoding="utf-8",
    )

    values = token_values(
        """
        #include "constants.hip"
        __global__ void kernel(float* out) { out[0] = scale(1.0f); }
        """,
        include_paths=[str(tmp_path)],
    )

    assert "scale" in values
    assert "4" in values
    assert "SCALE_VALUE" not in values


def test_preprocessor_preserves_unresolved_system_includes():
    tokens = HipLexer("""
        #include <hip/hip_runtime.h>
        __global__ void kernel() { }
    """).tokenize()

    assert any(token.type == "HASH" and token.value == "#" for token in tokens)
    assert "include" in [token.value for token in tokens]
    assert "hip_runtime" in [token.value for token in tokens]


def test_preprocessor_function_like_macro_expansion():
    values = token_values("""
        #define STORE(out, index, value) out[index] = value
        __global__ void kernel(float* data) {
            STORE(data, 0, 3.0f);
        }
    """)

    assert "STORE" not in values
    assert "data" in values
    assert "3.0f" in values


def test_preprocessor_error_directive_raises():
    with pytest.raises(SyntaxError, match="#error"):
        HipLexer("""
            #error stop
            __global__ void kernel() { }
        """).tokenize()


def test_parser_uses_preprocessed_conditionals():
    tokens = HipLexer("""
        #define USE_SELECTED 1
        #if USE_SELECTED
        __global__ void selected_kernel(float* out) { out[0] = 1.0f; }
        #else
        __global__ void rejected_kernel(float* out) { out[0] = 0.0f; }
        #endif
    """).tokenize()

    ast = HipParser(tokens).parse()
    kernel_names = [
        statement.name
        for statement in ast.statements
        if statement.__class__.__name__ == "KernelNode"
    ]
    assert kernel_names == ["selected_kernel"]
