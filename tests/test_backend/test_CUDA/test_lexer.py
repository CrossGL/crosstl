"""Test CUDA Lexer"""

import pytest
from crosstl.backend.CUDA.CudaLexer import CudaLexer


class TestCudaLexer:
    def test_cuda_keywords(self):
        """Test CUDA-specific keyword tokenization"""
        code = "__global__ __device__ __shared__ __constant__"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("GLOBAL", "__global__"),
            ("DEVICE", "__device__"),
            ("SHARED", "__shared__"),
            ("CONSTANT", "__constant__"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_cuda_builtin_variables(self):
        """Test CUDA built-in variable tokenization"""
        code = "threadIdx blockIdx gridDim blockDim"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("THREADIDX", "threadIdx"),
            ("BLOCKIDX", "blockIdx"),
            ("GRIDDIM", "gridDim"),
            ("BLOCKDIM", "blockDim"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_kernel_launch_syntax(self):
        """Test kernel launch syntax tokenization"""
        code = "kernel<<<blocks, threads>>>"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("IDENTIFIER", "kernel"),
            ("KERNEL_LAUNCH_START", "<<<"),
            ("IDENTIFIER", "blocks"),
            ("COMMA", ","),
            ("IDENTIFIER", "threads"),
            ("KERNEL_LAUNCH_END", ">>>"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_cuda_vector_types(self):
        """Test CUDA vector type tokenization"""
        code = "float2 float3 float4 int2 int3 int4"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("FLOAT2", "float2"),
            ("FLOAT3", "float3"),
            ("FLOAT4", "float4"),
            ("INT2", "int2"),
            ("INT3", "int3"),
            ("INT4", "int4"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_atomic_operations(self):
        """Test atomic operation tokenization"""
        code = "atomicAdd atomicSub atomicMax atomicMin"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("ATOMICADD", "atomicAdd"),
            ("ATOMICSUB", "atomicSub"),
            ("ATOMICMAX", "atomicMax"),
            ("ATOMICMIN", "atomicMin"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_sync_functions(self):
        """Test synchronization function tokenization"""
        code = "__syncthreads __syncwarp"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("SYNCTHREADS", "__syncthreads"),
            ("SYNCWARP", "__syncwarp"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens
