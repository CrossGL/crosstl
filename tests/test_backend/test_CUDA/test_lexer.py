"""Test CUDA Lexer"""

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

    def test_device_lambda_capture_tokenization(self):
        """Test CUDA device lambda captures use existing bracket/operator tokens."""
        code = "[&] __device__ (int x) { return x; }"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        assert tokens[:8] == [
            ("LBRACKET", "["),
            ("BITWISE_AND", "&"),
            ("RBRACKET", "]"),
            ("DEVICE", "__device__"),
            ("LPAREN", "("),
            ("INT", "int"),
            ("IDENTIFIER", "x"),
            ("RPAREN", ")"),
        ]

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
        code = (
            "atomicAdd atomicSub atomicMax atomicMin atomicExch atomicCAS "
            "atomicAnd atomicOr atomicXor atomicInc atomicDec"
        )
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("ATOMICADD", "atomicAdd"),
            ("ATOMICSUB", "atomicSub"),
            ("ATOMICMAX", "atomicMax"),
            ("ATOMICMIN", "atomicMin"),
            ("ATOMICEXCH", "atomicExch"),
            ("ATOMICCAS", "atomicCAS"),
            ("ATOMICAND", "atomicAnd"),
            ("ATOMICOR", "atomicOr"),
            ("ATOMICXOR", "atomicXor"),
            ("ATOMICINC", "atomicInc"),
            ("ATOMICDEC", "atomicDec"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_sync_functions(self):
        """Test synchronization function tokenization"""
        code = (
            "__syncthreads __syncwarp __threadfence "
            "__threadfence_block __threadfence_system"
        )
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        expected_tokens = [
            ("SYNCTHREADS", "__syncthreads"),
            ("SYNCWARP", "__syncwarp"),
            ("IDENTIFIER", "__threadfence"),
            ("IDENTIFIER", "__threadfence_block"),
            ("IDENTIFIER", "__threadfence_system"),
            ("EOF", ""),
        ]

        assert tokens == expected_tokens

    def test_shift_assignment_tokenization(self):
        """Test shift assignment operators are not split"""
        code = "out[0] <<= 1; out[1] >>= 1;"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        token_types = [token_type for token_type, _ in tokens]

        assert "SHIFT_LEFT_EQUALS" in token_types
        assert "SHIFT_RIGHT_EQUALS" in token_types

    def test_numeric_literal_tokenization(self):
        """Test integer masks and float suffixes stay intact"""
        code = "0xffu 0XCAFEull 0b1010u 0777u 1e-3f .5f"
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()

        values = [value for token_type, value in tokens if token_type == "NUMBER"]

        assert values == ["0xffu", "0XCAFEull", "0b1010u", "0777u", "1e-3f", ".5f"]
