"""Test HIP Code Generation"""

import pytest
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter


class TestHipCodeGen:
    def test_basic_kernel_conversion(self):
        """Test basic kernel to CrossGL conversion"""
        code = """
        __global__ void simple_kernel(float* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_device_function_conversion(self):
        """Test device function conversion"""
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_type_conversion(self):
        """Test HIP type to CrossGL type conversion"""
        codegen = HipToCrossGLConverter()

        # Test basic types
        assert codegen.convert_type(type(None)) == "void"

        # Test type mapping
        from crosstl.backend.HIP.HipAst import HipTypeNode

        int_type = HipTypeNode("int", [], [], [])
        assert codegen.convert_type(int_type) == "int"

        float_type = HipTypeNode("float", [], [], [])
        assert codegen.convert_type(float_type) == "float"

        float2_type = HipTypeNode("float2", [], [], [])
        assert codegen.convert_type(float2_type) == "vec2"

    def test_builtin_variable_conversion(self):
        """Test HIP built-in variable conversion"""
        codegen = HipToCrossGLConverter()

        from crosstl.backend.HIP.HipAst import HipIdentifierNode

        # Test threadIdx conversion
        threadidx = HipIdentifierNode("threadIdx.x")
        result = codegen.convert_HipIdentifierNode(threadidx)
        assert hasattr(result, "name")
        assert result.name == "gl_LocalInvocationID.x"

        # Test blockIdx conversion
        blockidx = HipIdentifierNode("blockIdx.y")
        result = codegen.convert_HipIdentifierNode(blockidx)
        assert hasattr(result, "name")
        assert result.name == "gl_WorkGroupID.y"

    def test_function_mapping_conversion(self):
        """Test HIP function to CrossGL function mapping"""
        codegen = HipToCrossGLConverter()

        from crosstl.backend.HIP.HipAst import HipFunctionCallNode, HipIdentifierNode

        # Test math function mapping
        sqrtf_call = HipFunctionCallNode(HipIdentifierNode("sqrtf"), [])
        sqrtf_call.function = HipIdentifierNode("sqrtf")
        sqrtf_call.arguments = []

        result = codegen.convert_HipFunctionCallNode(sqrtf_call)
        assert hasattr(result, "function")

    def test_vector_type_conversion(self):
        """Test HIP vector type conversion"""
        codegen = HipToCrossGLConverter()

        # Test vector type mappings
        assert codegen.type_map.get("float2") == "vec2"
        assert codegen.type_map.get("float3") == "vec3"
        assert codegen.type_map.get("float4") == "vec4"
        assert codegen.type_map.get("int2") == "ivec2"
        assert codegen.type_map.get("int3") == "ivec3"
        assert codegen.type_map.get("int4") == "ivec4"

    def test_memory_qualifier_conversion(self):
        """Test memory qualifier conversion"""
        code = """
        __global__ void kernel() {
            __shared__ float shared_data[256];
            __constant__ int constant_value = 42;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_atomic_operations_conversion(self):
        """Test atomic operations conversion"""
        code = """
        __global__ void kernel(int* counter) {
            atomicAdd(counter, 1);
            atomicSub(counter, 1);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_synchronization_conversion(self):
        """Test synchronization function conversion"""
        code = """
        __global__ void kernel() {
            __syncthreads();
            __threadfence();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_struct_conversion(self):
        """Test struct conversion"""
        code = """
        struct Point {
            float x;
            float y;
            float z;
        };
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_template_conversion(self):
        """Test template function conversion"""
        code = """
        template<typename T>
        __device__ T add(T a, T b) {
            return a + b;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_complex_kernel_conversion(self):
        """Test complex kernel conversion"""
        code = """
        __global__ void complex_kernel(float* input, float* output, int size) {
            extern __shared__ float shared_memory[];
            
            int tid = threadIdx.x;
            int bid = blockIdx.x;
            int idx = bid * blockDim.x + tid;
            
            if (idx < size) {
                shared_memory[tid] = input[idx];
                __syncthreads();
                
                float result = 0.0f;
                for (int i = 0; i < blockDim.x; i++) {
                    result += shared_memory[i];
                }
                
                output[idx] = result / blockDim.x;
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_empty_program_conversion(self):
        """Test empty program conversion"""
        code = ""
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")
        assert len(result.statements) == 0

    def test_math_functions_conversion(self):
        """Test math functions conversion"""
        code = """
        __device__ float math_operations(float x) {
            return sqrtf(sinf(x) + cosf(x));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")

    def test_vector_operations_conversion(self):
        """Test vector operations conversion"""
        code = """
        __global__ void vector_kernel(float2* data) {
            float2 a = make_float2(1.0f, 2.0f);
            float2 b = make_float2(3.0f, 4.0f);
            data[threadIdx.x] = a;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.convert(ast)

        # Verify that conversion produces valid CrossGL AST
        assert result is not None
        assert hasattr(result, "statements")
