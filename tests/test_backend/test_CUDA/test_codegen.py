"""Test CUDA Code Generation"""

import pytest
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter


class TestCudaCodeGen:
    def test_basic_kernel_conversion(self):
        """Test basic kernel to CrossGL conversion"""
        code = """
        __global__ void simple_kernel(float* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result
        assert "// Kernel: simple_kernel" in result

    def test_device_function_conversion(self):
        """Test device function conversion"""
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result
        assert "// Function: add" in result

    def test_multiple_kernels_conversion(self):
        """Test multiple kernels conversion"""
        code = """
        __global__ void kernel1(float* data) {
            data[threadIdx.x] = 1.0f;
        }
        
        __global__ void kernel2(int* data) {
            data[threadIdx.x] = 2;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel: kernel1" in result
        assert "// Kernel: kernel2" in result

    def test_type_conversion(self):
        """Test CUDA type to CrossGL type conversion"""
        codegen = CudaToCrossGLConverter()

        # Test basic types
        assert codegen.convert_cuda_type_to_crossgl("int") == "i32"
        assert codegen.convert_cuda_type_to_crossgl("float") == "f32"
        assert codegen.convert_cuda_type_to_crossgl("double") == "f64"
        assert codegen.convert_cuda_type_to_crossgl("bool") == "bool"
        assert codegen.convert_cuda_type_to_crossgl("void") == "void"

    def test_empty_program(self):
        """Test empty program conversion"""
        code = ""
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result
