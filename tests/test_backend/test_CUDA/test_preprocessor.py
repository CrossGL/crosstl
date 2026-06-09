import pytest

from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser


def token_values(code, **lexer_options):
    return [value for _, value in CudaLexer(code, **lexer_options).tokenize()]


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
    include_file = tmp_path / "constants.cuh"
    include_file.write_text(
        "#define SCALE_VALUE 4\n"
        "__device__ float scale(float value) { return value * SCALE_VALUE; }\n",
        encoding="utf-8",
    )

    values = token_values(
        """
        #include "constants.cuh"
        __global__ void kernel(float* out) { out[0] = scale(1.0f); }
        """,
        include_paths=[str(tmp_path)],
    )

    assert "scale" in values
    assert "4" in values
    assert "SCALE_VALUE" not in values


def test_preprocessor_preserves_unresolved_system_includes():
    tokens = CudaLexer("""
        #include <cuda_runtime.h>
        __global__ void kernel() { }
    """).tokenize()

    assert ("PREPROCESSOR", "#include <cuda_runtime.h>") in tokens


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
        CudaLexer("""
            #error stop
            __global__ void kernel() { }
        """).tokenize()


def test_preprocessor_defaults_to_linux_cuda_sample_platform_branch():
    values = token_values("""
        #if defined(__linux__)
        #define CPU_ATOMIC_ADD32(a, x) __sync_add_and_fetch(a, x)
        #elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        #define CPU_ATOMIC_ADD32(a, x) InterlockedAdd((volatile LONG *)a, x)
        #else
        #error Unsupported system
        #endif

        __device__ int add_one(int* value) {
            return CPU_ATOMIC_ADD32(value, 1);
        }
    """)

    assert "__sync_add_and_fetch" in values
    assert "InterlockedAdd" not in values


def test_preprocessor_platform_define_override_disables_default_linux_branch():
    values = token_values(
        """
        #if defined(__linux__)
        int selected_linux;
        #elif defined(_WIN32)
        int selected_windows;
        #else
        int selected_none;
        #endif
        """,
        defines={"_WIN32": "1"},
    )

    assert "selected_windows" in values
    assert "selected_linux" not in values
    assert "selected_none" not in values


def test_parser_uses_preprocessed_conditionals():
    tokens = CudaLexer("""
        #define USE_SELECTED 1
        #if USE_SELECTED
        __global__ void selected_kernel(float* out) { out[0] = 1.0f; }
        #else
        __global__ void rejected_kernel(float* out) { out[0] = 0.0f; }
        #endif
    """).tokenize()

    ast = CudaParser(tokens).parse()
    kernel_names = [kernel.name for kernel in ast.kernels]
    assert kernel_names == ["selected_kernel"]
