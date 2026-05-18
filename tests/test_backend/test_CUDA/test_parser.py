"""Test CUDA Parser"""

import pytest
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.CUDA.CudaAst import (
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    DoWhileNode,
    InitializerListNode,
    ShaderNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
)


class TestCudaParser:
    def test_simple_kernel_parsing(self):
        """Test parsing a simple CUDA kernel"""
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

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1
        assert ast.kernels[0].name == "simple_kernel"

    def test_device_function_parsing(self):
        """Test parsing a device function"""
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.functions) == 1
        assert ast.functions[0].name == "add"
        assert "__device__" in ast.functions[0].qualifiers

    def test_shared_memory_parsing(self):
        """Test parsing shared memory declaration"""
        code = """
        __global__ void kernel() {
            __shared__ float shared_data[256];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_builtin_variables_parsing(self):
        """Test parsing CUDA built-in variables"""
        code = """
        __global__ void kernel() {
            int x = threadIdx.x;
            int y = blockIdx.y;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_vector_types_parsing(self):
        """Test parsing CUDA vector types"""
        code = """
        __global__ void kernel(float2* data) {
            float4 vec = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_atomic_operations_parsing(self):
        """Test parsing atomic operations"""
        code = """
        __global__ void kernel(int* counter) {
            atomicAdd(counter, 1);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_sync_parsing(self):
        """Test parsing synchronization functions"""
        code = """
        __global__ void kernel() {
            __syncthreads();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_fixed_arrays_and_initializer_lists_parsing(self):
        """Test fixed arrays and brace initializer lists"""
        code = """
        float weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        struct Filter {
            float taps[3];
            float matrix[2][2];
        };

        __global__ void kernel(float input[4]) {
            float local[2] = {1.0f, 2.0f};
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.global_variables[0].vtype == "float[4]"
        assert isinstance(ast.global_variables[0].value, InitializerListNode)
        assert ast.structs[0].members[0].vtype == "float[3]"
        assert ast.structs[0].members[1].vtype == "float[2][2]"
        assert ast.kernels[0].params[0].vtype == "float[4]"
        assert ast.kernels[0].body[0].vtype == "float[2]"
        assert isinstance(ast.kernels[0].body[0].value, InitializerListNode)

    def test_qualified_declarations_parsing(self):
        """Test const, unsigned, and static declarations"""
        code = """
        static float cached = 1.0f;
        unsigned int mask = 3u;

        __global__ void kernel(unsigned int* out, const float scale) {
            const int local = 1;
            unsigned int idx = 2u;
            static float tmp = 0.0f;
            out[0] = idx;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.global_variables[0].vtype == "float"
        assert ast.global_variables[1].vtype == "unsigned int"
        assert ast.kernels[0].params[0].vtype == "unsigned int *"
        assert ast.kernels[0].params[1].vtype == "const float"
        assert ast.kernels[0].body[0].vtype == "const int"
        assert ast.kernels[0].body[1].vtype == "unsigned int"
        assert ast.kernels[0].body[2].vtype == "float"

    def test_identifier_multiply_expression_is_not_declaration(self):
        """Test expression statements using * are not parsed as declarations"""
        code = """
        __global__ void kernel() {
            a * b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast.kernels[0].body[0], BinaryOpNode)

    def test_local_pointer_declarations_and_unary_pointer_expressions(self):
        """Test local pointer declarations, address-of, and dereference"""
        code = """
        void helper(float* data, unsigned int* ids) {
            float* p = data;
            const float* cp = data;
            unsigned int* ip = ids;
            float x = 1.0f;
            float* q = &x;
            *p = *q;
            ip[0] = 1u;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].vtype == "float *"
        assert body[1].vtype == "const float *"
        assert body[2].vtype == "unsigned int *"
        assert isinstance(body[4].value, UnaryOpNode)
        assert body[4].value.op == "&"
        assert isinstance(body[5], AssignmentNode)
        assert isinstance(body[5].left, UnaryOpNode)
        assert body[5].left.op == "*"
        assert isinstance(body[5].right, UnaryOpNode)
        assert body[5].right.op == "*"

    def test_bitwise_logical_and_shift_expression_parsing(self):
        """Test bitwise, shift, logical, and compound shift expressions"""
        code = """
        unsigned int helper(unsigned int a, unsigned int b) {
            unsigned int x = (a & b) | (a ^ b);
            unsigned int y = x << 2;
            unsigned int z = y >> 1;
            if (a > 0 && b > 0 || a == b) {
                z <<= 1;
                z >>= 1;
            }
            return z;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert isinstance(body[0].value, BinaryOpNode)
        assert body[0].value.op == "|"
        assert body[0].value.left.op == "&"
        assert body[0].value.right.op == "^"
        assert body[1].value.op == "<<"
        assert body[2].value.op == ">>"
        assert body[3].condition.op == "||"
        assert body[3].condition.left.op == "&&"
        assert body[3].if_body[0].operator == "<<="
        assert body[3].if_body[1].operator == ">>="

    def test_numeric_literal_parsing(self):
        """Test integer mask and float suffix literals parse intact"""
        code = """
        unsigned int helper() {
            unsigned int mask = 0xffu;
            unsigned int bits = 0b1010u;
            unsigned int oct = 0777u;
            float x = 1e-3f;
            float y = .5f;
            return mask | bits | oct;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].value == "0xffu"
        assert body[1].value == "0b1010u"
        assert body[2].value == "0777u"
        assert body[3].value == "1e-3f"
        assert body[4].value == ".5f"

    def test_qualified_and_pointer_return_functions_parsing(self):
        """Test qualified scalar and pointer return types"""
        code = """
        unsigned int lane_mask() { return 3u; }
        float* get_data(float* data) { return data; }
        const float* get_const_data(const float* data) { return data; }
        static inline unsigned int helper(unsigned int x) { return x; }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.functions[0].return_type == "unsigned int"
        assert ast.functions[1].return_type == "float *"
        assert ast.functions[1].params[0].vtype == "float *"
        assert ast.functions[2].return_type == "const float *"
        assert ast.functions[2].params[0].vtype == "const float *"
        assert ast.functions[3].return_type == "unsigned int"
        assert "static" in ast.functions[3].qualifiers
        assert "inline" in ast.functions[3].qualifiers

    def test_control_flow_and_cast_expression_parsing(self):
        """Test CUDA control-flow nodes and cast expressions"""
        code = """
        int helper(float x, int n) {
            int i = 0;
            do {
                i += 1;
            } while (i < n);
            switch (i) {
                case 1:
                    i += 2;
                    break;
                default:
                    i += 3;
                    break;
            }
            return i > 0 ? (int)x : n;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert isinstance(body[1], DoWhileNode)
        assert isinstance(body[2], SwitchNode)
        assert len(body[2].cases) == 1
        assert len(body[2].default_case) == 2
        assert isinstance(body[3].value, TernaryOpNode)
        assert isinstance(body[3].value.true_expr, CastNode)
