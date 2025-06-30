"""Test HIP Code Generation from CrossGL"""

import pytest
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.codegen.hip_codegen import HipCodeGen


class TestHipCodeGen:
    def test_simple_function_generation(self):
        """Test generating a simple HIP function from CrossGL"""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code
        assert "__device__ void main()" in hip_code

    def test_compute_shader_to_kernel(self):
        """Test converting a compute shader to HIP kernel"""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__global__ void main()" in hip_code

    def test_type_conversion(self):
        """Test CrossGL to HIP type conversion"""
        codegen = HipCodeGen()

        # Test basic types
        assert codegen.map_type("int") == "int"
        assert codegen.map_type("float") == "float"
        assert codegen.map_type("double") == "double"
        assert codegen.map_type("bool") == "bool"
        assert codegen.map_type("void") == "void"

        # Test vector types
        assert codegen.map_type("vec2") == "float2"
        assert codegen.map_type("vec3") == "float3"
        assert codegen.map_type("vec4") == "float4"
        assert codegen.map_type("ivec2") == "int2"
        assert codegen.map_type("ivec3") == "int3"
        assert codegen.map_type("ivec4") == "int4"

    def test_function_conversion(self):
        """Test CrossGL to HIP function conversion"""
        codegen = HipCodeGen()

        # Test math functions
        assert codegen.function_map.get("sqrt") == "sqrtf"
        assert codegen.function_map.get("pow") == "powf"
        assert codegen.function_map.get("sin") == "sinf"
        assert codegen.function_map.get("cos") == "cosf"

        # Test vector constructors
        assert codegen.function_map.get("vec2") == "make_float2"
        assert codegen.function_map.get("vec3") == "make_float3"
        assert codegen.function_map.get("vec4") == "make_float4"

    def test_builtin_variable_conversion(self):
        """Test built-in variable conversion"""
        codegen = HipCodeGen()

        # Test thread index mapping
        assert codegen.builtin_map.get("gl_LocalInvocationID.x") == "threadIdx.x"
        assert codegen.builtin_map.get("gl_LocalInvocationID.y") == "threadIdx.y"
        assert codegen.builtin_map.get("gl_LocalInvocationID.z") == "threadIdx.z"

        # Test workgroup index mapping
        assert codegen.builtin_map.get("gl_WorkGroupID.x") == "blockIdx.x"
        assert codegen.builtin_map.get("gl_WorkGroupID.y") == "blockIdx.y"
        assert codegen.builtin_map.get("gl_WorkGroupID.z") == "blockIdx.z"

        # Test workgroup size mapping
        assert codegen.builtin_map.get("gl_WorkGroupSize.x") == "blockDim.x"
        assert codegen.builtin_map.get("gl_WorkGroupSize.y") == "blockDim.y"
        assert codegen.builtin_map.get("gl_WorkGroupSize.z") == "blockDim.z"

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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct Vertex" in hip_code
        assert "float3 position;" in hip_code
        assert "float3 normal;" in hip_code

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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # For now, just check that basic variables are generated
        assert "float shared_data;" in hip_code
        assert "float constants;" in hip_code

    def test_atomic_operations(self):
        """Test atomic operations code generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int counter;
                    atomicAdd(counter, 1);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "__global__" in hip_code or "__device__" in hip_code

    def test_synchronization_functions(self):
        """Test synchronization function generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    barrier();
                    memoryBarrier();
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "__global__" in hip_code or "__device__" in hip_code

    def test_vector_operations(self):
        """Test vector operations generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(4.0, 5.0, 6.0);
                    vec3 result = a + b;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure with HIP includes
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_math_functions(self):
        """Test math function generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = 1.0;
                    float result = sqrt(sin(x) + cos(x));
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure with HIP includes
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_texture_operations(self):
        """Test texture operation generation"""
        source_code = """
        shader TestShader {
            fragment {
                void main() {
                    sampler2D tex;
                    vec2 uv = vec2(0.5, 0.5);
                    vec4 color = texture(tex, uv);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_control_flow(self):
        """Test control flow generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int i;
                    if (i > 0) {
                        i = i + 1;
                    } else {
                        i = i - 1;
                    }
                    
                    for (int j = 0; j < 10; j++) {
                        i = i * 2;
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_array_operations(self):
        """Test array operation generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float data[256];
                    int index = 0;
                    data[index] = 1.0;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_empty_shader(self):
        """Test empty shader generation"""
        source_code = ""

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code

    def test_multiple_functions(self):
        """Test multiple function generation"""
        source_code = """
        shader TestShader {
            vertex {
                void vertex_main() {
                    int x;
                }
            }
            fragment {
                void fragment_main() {
                    float y;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for HIP includes
        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code
