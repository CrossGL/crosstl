"""Test CUDA Code Generation from CrossGL"""

import pytest
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen


class TestCudaCodeGen:
    def test_simple_function_generation(self):
        """Test generating a simple CUDA function from CrossGL"""
        source_code = """
        shader TestShader {
            vertex {
                void main() {
                    int x;
                }
            }
        }
        """
        
        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        
        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)
        
        assert "#include <cuda_runtime.h>" in cuda_code
        assert "#include <device_launch_parameters.h>" in cuda_code
        assert "__device__ void main()" in cuda_code

    def test_compute_shader_to_kernel(self):
        """Test converting a compute shader to CUDA kernel"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int id;
                }
            }
        }
        """
        
        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        
        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)
        
        assert "__global__ void main()" in cuda_code

    def test_type_conversion(self):
        """Test CrossGL to CUDA type conversion"""
        codegen = CudaCodeGen()
        
        # Test basic types
        assert codegen.convert_crossgl_type_to_cuda("i32") == "int"
        assert codegen.convert_crossgl_type_to_cuda("f32") == "float"
        assert codegen.convert_crossgl_type_to_cuda("f64") == "double"
        assert codegen.convert_crossgl_type_to_cuda("bool") == "bool"
        assert codegen.convert_crossgl_type_to_cuda("void") == "void"
        
        # Test vector types
        assert codegen.convert_crossgl_type_to_cuda("vec2<f32>") == "float2"
        assert codegen.convert_crossgl_type_to_cuda("vec3<f32>") == "float3"
        assert codegen.convert_crossgl_type_to_cuda("vec4<f32>") == "float4"
        assert codegen.convert_crossgl_type_to_cuda("vec2<i32>") == "int2"

    def test_function_conversion(self):
        """Test CrossGL to CUDA function conversion"""
        codegen = CudaCodeGen()
        
        # Test math functions
        assert codegen.convert_builtin_function("sqrt") == "sqrtf"
        assert codegen.convert_builtin_function("pow") == "powf"
        assert codegen.convert_builtin_function("sin") == "sinf"
        assert codegen.convert_builtin_function("cos") == "cosf"
        
        # Test vector constructors
        assert codegen.convert_builtin_function("vec2<f32>") == "make_float2"
        assert codegen.convert_builtin_function("vec3<f32>") == "make_float3"
        assert codegen.convert_builtin_function("vec4<f32>") == "make_float4"
        
        # Test atomic operations
        assert codegen.convert_builtin_function("atomicAdd") == "atomicAdd"
        assert codegen.convert_builtin_function("atomicExchange") == "atomicExch"
        
        # Test synchronization
        assert codegen.convert_builtin_function("workgroupBarrier") == "__syncthreads"

    def test_struct_generation(self):
        """Test struct generation"""
        source_code = """
        shader TestShader {
            struct Vertex {
                vec3 position;
                vec3 normal;
            }
        }
        """
        
        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        
        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)
        
        assert "struct Vertex {" in cuda_code
        assert "float3 position;" in cuda_code
        assert "float3 normal;" in cuda_code

    def test_variable_with_qualifiers(self):
        """Test variable generation with memory qualifiers"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float shared_data;
                    float constants;
                }
            }
        }
        """
        
        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        
        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)
        
        # For now, just check that basic variables are generated
        assert "float shared_data;" in cuda_code
        assert "float constants;" in cuda_code

    def test_empty_shader(self):
        """Test empty shader generation"""
        source_code = ""
        
        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        
        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)
        
        assert "#include <cuda_runtime.h>" in cuda_code
        assert "#include <device_launch_parameters.h>" in cuda_code 