"""Test HIP Code Generation"""

import pytest
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter


class TestHipCodeGen:
    def test_basic_kernel_conversion(self):
        """Test basic kernel to CrossGL conversion"""
        code = """
        __global__ void simple_kernel() {
            int idx = 0;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result
        assert "// Kernel: simple_kernel" in result

    def test_device_function_conversion(self):
        """Test device function conversion"""
        code = """
        __device__ float add() {
            return 1.0;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result
        assert "// Function: add" in result

    def test_multiple_kernels_conversion(self):
        """Test multiple kernels conversion"""
        code = """
        __global__ void kernel1() {
            int x = 1;
        }
        
        __global__ void kernel2() {
            int x = 2;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel: kernel1" in result
        assert "// Kernel: kernel2" in result

    def test_type_conversion(self):
        """Test HIP type to CrossGL type conversion"""
        codegen = HipToCrossGLConverter()

        # Test basic types
        assert codegen.convert_hip_type_to_crossgl("int") == "i32"
        assert codegen.convert_hip_type_to_crossgl("float") == "f32"
        assert codegen.convert_hip_type_to_crossgl("double") == "f64"
        assert codegen.convert_hip_type_to_crossgl("bool") == "bool"
        assert codegen.convert_hip_type_to_crossgl("void") == "void"

    def test_function_conversion(self):
        """Test HIP function to CrossGL function conversion"""
        codegen = HipToCrossGLConverter()

        # Test math functions
        assert codegen.convert_hip_builtin_function("sqrtf") == "sqrt"
        assert codegen.convert_hip_builtin_function("sinf") == "sin"
        assert codegen.convert_hip_builtin_function("cosf") == "cos"
        assert (
            codegen.convert_hip_builtin_function("__syncthreads") == "workgroupBarrier"
        )

    def test_struct_conversion(self):
        """Test struct conversion"""
        code = """
        struct Point {
            float x;
            float y;
        };
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result
        assert "struct Point" in result

    def test_empty_program(self):
        """Test empty program conversion"""
        code = ""
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result

    def test_vector_type_conversion(self):
        """Test HIP vector type conversion"""
        codegen = HipToCrossGLConverter()

        # Test vector type mappings
        assert codegen.convert_hip_type_to_crossgl("float2") == "vec2<f32>"
        assert codegen.convert_hip_type_to_crossgl("float3") == "vec3<f32>"
        assert codegen.convert_hip_type_to_crossgl("float4") == "vec4<f32>"
        assert codegen.convert_hip_type_to_crossgl("int2") == "vec2<i32>"
        assert codegen.convert_hip_type_to_crossgl("int3") == "vec3<i32>"
        assert codegen.convert_hip_type_to_crossgl("int4") == "vec4<i32>"
