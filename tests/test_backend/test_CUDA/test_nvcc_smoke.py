"""Optional nvcc-backed smoke tests for native CUDA frontend samples."""

import shutil
import subprocess

import pytest

from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser


def compile_cuda_if_nvcc_available(cuda_code, tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")

    source_path = tmp_path / "native_smoke.cu"
    object_path = tmp_path / "native_smoke.o"
    source_path.write_text(cuda_code)

    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_80",
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ],
        capture_output=True,
        text=True,
    )
    compiler_output = result.stdout + result.stderr
    if result.returncode != 0 and "Unsupported gpu architecture" in compiler_output:
        pytest.skip(compiler_output)

    assert result.returncode == 0, compiler_output


def test_native_cuda_pipeline_primitives_parse_and_compile_if_available(tmp_path):
    """Smoke-test CUDA pipeline primitives, compiling only when nvcc exists."""
    cuda_code = """
    #include <cuda_pipeline.h>

    __global__ void pipeline_primitive_smoke(
        int* shared_out,
        const int* global_in,
        __mbarrier_t* barrier
    ) {
        __shared__ int shared[32];
        int lane = threadIdx.x;
        __pipeline_memcpy_async(&shared[lane], &global_in[lane], sizeof(int));
        __pipeline_memcpy_async(&shared[(lane + 16) & 31], &global_in[lane], 4, 0);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __pipeline_arrive_on(barrier);
        shared_out[lane] = shared[lane];
    }
    """
    lexer = CudaLexer(cuda_code)
    tokens = lexer.tokenize()
    parser = CudaParser(tokens)
    ast = parser.parse()

    crossgl = CudaToCrossGLConverter().generate(ast)
    assert "// Kernel: pipeline_primitive_smoke" in crossgl
    assert "// cuda pipeline.memcpy_async not directly supported in CrossGL" in crossgl
    assert "// cuda pipeline.commit not directly supported in CrossGL" in crossgl
    assert "// cuda pipeline.wait_prior not directly supported in CrossGL" in crossgl
    assert "// cuda pipeline.arrive_on not directly supported in CrossGL" in crossgl
    assert "__pipeline_memcpy_async" not in crossgl

    compile_cuda_if_nvcc_available(cuda_code, tmp_path)
