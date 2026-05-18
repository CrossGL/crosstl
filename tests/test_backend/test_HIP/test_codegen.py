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

    def test_constructor_style_vector_declaration_conversion(self):
        """Test HIP constructor-style vector declarations convert"""
        code = """
        void launch() {
            dim3 grid(16, 8, 1);
            dim3 block(32);
            float3 v(1.0f, 2.0f, 3.0f);
            double2 d = make_double2(1.0, 2.0);
            uint4 ids = make_uint4(1u, 2u, 3u, 4u);
            uchar2 bytes(1, 2);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var grid: vec3<u32> = vec3<u32>(16, 8, 1);" in result
        assert "var block: vec3<u32> = vec3<u32>(32);" in result
        assert "var v: vec3<f32> = vec3<f32>(1.0f, 2.0f, 3.0f);" in result
        assert "var d: vec2<f64> = vec2<f64>(1.0, 2.0);" in result
        assert "var ids: vec4<u32> = vec4<u32>(1u, 2u, 3u, 4u);" in result
        assert "var bytes: vec2<u8> = vec2<u8>(1, 2);" in result

    def test_fixed_array_initializer_conversion(self):
        """Test fixed arrays and brace initializer conversion"""
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var weights: array<f32, 4> = {1.0f, 2.0f, 3.0f, 4.0f};" in result
        assert "array<f32, 3> taps;" in result
        assert "array<array<f32, 2>, 2> matrix;" in result
        assert "array<f32, 4> input" in result
        assert "var local: array<f32, 2> = {1.0f, 2.0f};" in result

    def test_kernel_pointer_parameters_lower_to_storage_arrays(self):
        """Test pointer kernel params lower to storage arrays of element type"""
        code = """
        __global__ void kernel(float* data, const int* indices, float value) {
            data[indices[0]] = value;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "data: array<f32>" in result
        assert "indices: array<i32>" in result
        assert "array<ptr" not in result

    def test_qualified_declaration_conversion(self):
        """Test const, unsigned, and static declaration conversion"""
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var cached: f32 = 1.0f;" in result
        assert "var mask: u32 = 3u;" in result
        assert "out: array<u32>" in result
        assert "f32 scale" in result
        assert "var local: i32 = 1;" in result
        assert "var idx: u32 = 2u;" in result
        assert "var tmp: f32 = 0.0f;" in result

    def test_qualified_and_pointer_return_function_conversion(self):
        """Test qualified scalar and pointer return conversion"""
        code = """
        unsigned int lane_mask() { return 3u; }
        float* get_data(float* data) { return data; }
        const float* get_const_data(const float* data) { return data; }
        static inline unsigned int helper(unsigned int x) { return x; }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "u32 lane_mask()" in result
        assert "ptr<f32> get_data(ptr<f32> data)" in result
        assert "ptr<f32> get_const_data(ptr<f32> data)" in result
        assert "u32 helper(u32 x)" in result

    def test_expression_statements_and_for_header_conversion(self):
        """Test expression statements and for headers are preserved"""
        code = """
        void helper(float* data, int n) {
            int i = 0;
            i += 1;
            sink(i);
            for (int j = 0; j < n; j++) {
                sink(j);
            }
            __syncthreads();
            return;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i += 1;" in result
        assert "sink(i);" in result
        assert "for (var j: i32 = 0; (j < n); (j++))" in result
        assert "sink(j);" in result
        assert "workgroupBarrier();" in result
        assert "None" not in result

    def test_local_pointer_declarations_and_unary_pointer_conversion(self):
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var p: ptr<f32> = data;" in result
        assert "var cp: ptr<f32> = data;" in result
        assert "var ip: ptr<u32> = ids;" in result
        assert "var q: ptr<f32> = (&x);" in result
        assert "(*p) = (*q);" in result
        assert "ip[0] = 1u;" in result
        assert "BinaryOpNode" not in result
        assert "UnaryOpNode" not in result

    def test_bitwise_logical_and_shift_expression_conversion(self):
        """Test bitwise, shift, logical, and compound shift conversion"""
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var x: u32 = ((a & b) | (a ^ b));" in result
        assert "var y: u32 = (x << 2);" in result
        assert "var z: u32 = (y >> 1);" in result
        assert "if ((((a > 0) && (b > 0)) || (a == b))) {" in result
        assert "z <<= 1;" in result
        assert "z >>= 1;" in result
        assert "BinaryOpNode" not in result

    def test_numeric_literal_conversion(self):
        """Test integer mask and float suffix literals are preserved"""
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var mask: u32 = 0xffu;" in result
        assert "var bits: u32 = 0b1010u;" in result
        assert "var oct: u32 = 0777u;" in result
        assert "var x: f32 = 1e-3f;" in result
        assert "var y: f32 = .5f;" in result
        assert "return ((mask | bits) | oct);" in result

    def test_boolean_null_and_character_literal_conversion(self):
        """Test bool, null, nullptr, and character literals are preserved"""
        code = r"""
        bool helper(int* ptr) {
            bool yes = true;
            bool no = false;
            char c = 'x';
            char escaped = '\n';
            int* p = nullptr;
            int* q = NULL;
            return yes && !no;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var yes: bool = true;" in result
        assert "var no: bool = false;" in result
        assert "var c: i8 = 'x';" in result
        assert "var escaped: i8 = '\\n';" in result
        assert "var p: ptr<i32> = nullptr;" in result
        assert "var q: ptr<i32> = NULL;" in result
        assert "return (yes && (!no));" in result

    def test_control_flow_and_cast_expression_conversion(self):
        """Test while, do-while, switch, ternary, and casts are emitted"""
        code = """
        int helper(float x, int n) {
            int i = 0;
            while (i < n) {
                i += 1;
            }
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "while ((i < n)) {" in result
        assert "do {" in result
        assert "} while ((i < n));" in result
        assert "switch (i) {" in result
        assert "case 1:" in result
        assert "default:" in result
        assert "return ((i > 0) ? i32(x) : n);" in result
        assert "WhileNode" not in result
        assert "DoWhileNode" not in result
        assert "SwitchNode" not in result
        assert "TernaryOpNode" not in result
        assert "CastNode" not in result
