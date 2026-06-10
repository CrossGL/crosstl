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


def test_preprocessor_ignores_stb_style_documented_self_include(tmp_path):
    include_file = tmp_path / "stb_image.h"
    include_file.write_text(
        """
        /* Reduced from ROCm/rocm-examples Applications/optical_flow/stb_image.h.
           Usage docs include:
              #define STB_IMAGE_IMPLEMENTATION
              #include "stb_image.h"
        */
        #ifndef STBI_INCLUDE_STB_IMAGE_H
        #define STBI_INCLUDE_STB_IMAGE_H
        #define STBIDEF extern
        STBIDEF int stbi_info(char const *filename, int *x, int *y, int *comp);
        #endif
        """,
        encoding="utf-8",
    )

    tokens = HipLexer(
        """
        #include "stb_image.h"
        __global__ void optical_flow_kernel(int* out) { out[0] = 1; }
        """,
        include_paths=[str(tmp_path)],
    ).tokenize()

    ast = HipParser(tokens).parse()
    names = [
        statement.name
        for statement in ast.statements
        if statement.__class__.__name__ in {"FunctionNode", "KernelNode"}
    ]
    assert names == ["stbi_info", "optical_flow_kernel"]


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


def test_preprocessor_defaults_to_rocm_amd_linux_platform_branches():
    values = token_values("""
        #if defined(__HIP_PLATFORM_AMD__)
        int selected_amd;
        #elif defined(__HIP_PLATFORM_NVIDIA__)
        int selected_nvidia;
        #else
        #error unsupported hip platform
        #endif

        #if defined(_WIN32)
        int selected_windows;
        #elif defined(__linux__)
        int selected_linux;
        #else
        #error unsupported host platform
        #endif
    """)

    assert "selected_amd" in values
    assert "selected_linux" in values
    assert "selected_nvidia" not in values
    assert "selected_windows" not in values


def test_preprocessor_defaults_to_cplusplus_17_for_rocm_headers():
    values = token_values("""
        #if __cplusplus < 201402L
        #error "rocPRIM requires at least C++14"
        #else
        __device__ int selected_cpp_standard() { return __cplusplus; }
        #endif
    """)

    assert "selected_cpp_standard" in values
    assert "201703L" in values


def test_preprocessor_cplusplus_define_override_disables_default():
    with pytest.raises(SyntaxError, match="rocPRIM requires at least C\\+\\+14"):
        HipLexer(
            """
            #if __cplusplus < 201402L
            #error "rocPRIM requires at least C++14"
            #else
            __device__ int selected_cpp_standard() { return __cplusplus; }
            #endif
            """,
            defines={"__cplusplus": "199711L"},
        ).tokenize()


def test_preprocessor_platform_define_override_disables_matching_defaults():
    values = token_values(
        """
        #if defined(__HIP_PLATFORM_AMD__)
        int selected_amd;
        #elif defined(__HIP_PLATFORM_NVIDIA__)
        int selected_nvidia;
        #else
        int selected_no_hip_platform;
        #endif

        #if defined(_WIN32)
        int selected_windows;
        #elif defined(__linux__)
        int selected_linux;
        #else
        int selected_no_host_platform;
        #endif
        """,
        defines={"__HIP_PLATFORM_NVIDIA__": "1", "_WIN32": "1"},
    )

    assert "selected_nvidia" in values
    assert "selected_windows" in values
    assert "selected_amd" not in values
    assert "selected_linux" not in values
    assert "selected_no_hip_platform" not in values
    assert "selected_no_host_platform" not in values


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
