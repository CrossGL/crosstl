from crosstl.backend.HIP.HipLexer import HipLexer


class TestHipLexer:

    def test_keywords_tokenization(self):
        code = """
        __global__ __device__ __host__ __shared__ __constant__
        __forceinline__ __noinline__ __launch_bounds__ template typename class
        struct union enum namespace using extern static const
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "__GLOBAL__" in token_types
        assert "__DEVICE__" in token_types
        assert "__HOST__" in token_types
        assert "__SHARED__" in token_types
        assert "__CONSTANT__" in token_types
        assert "__FORCEINLINE__" in token_types
        assert "__NOINLINE__" in token_types
        assert "__LAUNCH_BOUNDS__" in token_types
        assert "TEMPLATE" in token_types
        assert "TYPENAME" in token_types
        assert "CLASS" in token_types
        assert "STRUCT" in token_types

    def test_launch_bounds_attribute_tokenization(self):
        code = """
        __global__ void __launch_bounds__(BlockSize, MinWarps)
        kernel(float* out) {}
        """
        lexer = HipLexer(code)
        tokens = [token for token in lexer.tokenize() if token.type != "NEWLINE"]

        assert [(token.type, token.value) for token in tokens[:9]] == [
            ("__GLOBAL__", "__global__"),
            ("VOID", "void"),
            ("__LAUNCH_BOUNDS__", "__launch_bounds__"),
            ("LPAREN", "("),
            ("IDENTIFIER", "BlockSize"),
            ("COMMA", ","),
            ("IDENTIFIER", "MinWarps"),
            ("RPAREN", ")"),
            ("IDENTIFIER", "kernel"),
        ]

    def test_inline_assembly_tokenization(self):
        code = "asm __asm__ volatile __volatile__"
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        assert [(token.type, token.value) for token in tokens] == [
            ("ASM", "asm"),
            ("ASM", "__asm__"),
            ("VOLATILE", "volatile"),
            ("VOLATILE", "__volatile__"),
        ]

    def test_c_linkage_kernel_tokenization(self):
        code = """
        extern "C"
        __global__ void vector_add(float* output) {}
        """
        lexer = HipLexer(code)
        tokens = [token for token in lexer.tokenize() if token.type != "NEWLINE"]

        assert [(token.type, token.value) for token in tokens[:7]] == [
            ("EXTERN", "extern"),
            ("STRING", '"C"'),
            ("__GLOBAL__", "__global__"),
            ("VOID", "void"),
            ("IDENTIFIER", "vector_add"),
            ("LPAREN", "("),
            ("FLOAT", "float"),
        ]

    def test_device_lambda_capture_tokenization(self):
        code = "[&] __device__ (int x) { return x; }"
        lexer = HipLexer(code)
        tokens = [token for token in lexer.tokenize() if token.type != "NEWLINE"]

        assert [(token.type, token.value) for token in tokens[:8]] == [
            ("LBRACKET", "["),
            ("AMPERSAND", "&"),
            ("RBRACKET", "]"),
            ("__DEVICE__", "__device__"),
            ("LPAREN", "("),
            ("INT", "int"),
            ("IDENTIFIER", "x"),
            ("RPAREN", ")"),
        ]

    def test_hip_vector_types(self):
        code = """
        int2 int3 int4 float2 float3 float4
        double2 double3 double4 uint2 uint3 uint4
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_pairs = [
            (token.type, token.value) for token in tokens if token.type != "NEWLINE"
        ]

        expected_tokens = [
            ("INT2", "int2"),
            ("INT3", "int3"),
            ("INT4", "int4"),
            ("FLOAT2", "float2"),
            ("FLOAT3", "float3"),
            ("FLOAT4", "float4"),
            ("DOUBLE2", "double2"),
            ("DOUBLE3", "double3"),
            ("DOUBLE4", "double4"),
            ("UINT2", "uint2"),
            ("UINT3", "uint3"),
            ("UINT4", "uint4"),
        ]

        for expected_token in expected_tokens:
            assert (
                expected_token in token_pairs
            ), f"Expected token {expected_token} not found in {token_pairs}"

    def test_hip_builtin_variables(self):
        code = """
        threadIdx.x blockIdx.y blockDim.z gridDim.x
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_values = [token.value for token in tokens]

        assert "threadIdx" in token_values
        assert "blockIdx" in token_values
        assert "blockDim" in token_values
        assert "gridDim" in token_values

    def test_hip_memory_functions(self):
        code = """
        hipMalloc hipFree hipMemcpy hipMemset
        __syncthreads __threadfence __threadfence_block
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_pairs = [
            (token.type, token.value) for token in tokens if token.type != "NEWLINE"
        ]
        token_values = [token.value for token in tokens if token.type == "IDENTIFIER"]

        assert "hipMalloc" in token_values
        assert "__threadfence" in token_values
        assert "__threadfence_block" in token_values

        assert ("SYNCTHREADS", "__syncthreads") in token_pairs

    def test_operators_tokenization(self):
        code = """
        + - * / % == != < > <= >= && || ! & | ^ << >> ++ --
        += -= *= /= %= &= |= ^= <<= >>= = ? : . -> ::
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "PLUS" in token_types
        assert "MINUS" in token_types
        assert "STAR" in token_types
        assert "SLASH" in token_types
        assert "EQ" in token_types
        assert "NE" in token_types
        assert "AND" in token_types
        assert "OR" in token_types
        assert "LSHIFT_ASSIGN" in token_types
        assert "RSHIFT_ASSIGN" in token_types
        assert "ARROW" in token_types
        assert "SCOPE" in token_types

    def test_numeric_literals(self):
        code = """
        42 3.14f 2.71828 0xFFu 0XCAFEull 0b1010u 0777u 1e5 2.5e-3f .5f
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        integers = [token for token in tokens if token.type == "INTEGER"]
        floats = [token for token in tokens if token.type == "FLOAT"]

        integer_values = [token.value for token in integers]
        float_values = [token.value for token in floats]

        assert "0xFFu" in integer_values
        assert "0XCAFEull" in integer_values
        assert "0b1010u" in integer_values
        assert "0777u" in integer_values
        assert "2.5e-3f" in float_values
        assert ".5f" in float_values

    def test_string_and_character_literals(self):
        code = """
        "hello world" 'c' '\\n' '\\x7f' '\\377' u8'a' L'b'
        "escaped \\"string\\"" "path/to/file" char
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        strings = [token for token in tokens if token.type == "STRING"]
        chars = [token for token in tokens if token.type == "CHAR_LIT"]

        assert len(strings) >= 3
        assert [token.value for token in chars] == [
            "'c'",
            "'\\n'",
            "'\\x7f'",
            "'\\377'",
            "u8'a'",
            "L'b'",
        ]
        assert ("CHAR", "char") in [(token.type, token.value) for token in tokens]

    def test_raw_string_literal_tokenization(self):
        code = """
        constexpr const char* shader = R"(
        #version 330 core
        void main() {}
        )";
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        strings = [token.value for token in tokens if token.type == "STRING"]

        assert len(strings) == 1
        assert strings[0].startswith('R"(')
        assert "#version 330 core" in strings[0]

    def test_preprocessor_directives(self):
        code = """
        #include <hip/hip_runtime.h>
        #define MAX_SIZE 1024
        #ifdef DEBUG
        #endif
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "HASH" in token_types
        assert "IDENTIFIER" in token_types  # include, define, etc.

    def test_delimiters_tokenization(self):
        code = """
        ( ) [ ] { } , ; : ::
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "LPAREN" in token_types
        assert "RPAREN" in token_types
        assert "LBRACKET" in token_types
        assert "RBRACKET" in token_types
        assert "LBRACE" in token_types
        assert "RBRACE" in token_types
        assert "COMMA" in token_types
        assert "SEMICOLON" in token_types
        assert "COLON" in token_types
        assert "SCOPE" in token_types

    def test_template_syntax(self):
        code = """
        template<typename T> class Vector<T, int N>
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]
        [token.value for token in tokens]

        assert "TEMPLATE" in token_types
        assert "LT" in token_types
        assert "GT" in token_types
        assert "TYPENAME" in token_types

    def test_function_declaration(self):
        code = """
        __global__ void kernel(float* data, int size) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "__GLOBAL__" in token_types
        assert "VOID" in token_types
        assert "LPAREN" in token_types
        assert "RPAREN" in token_types
        assert "LBRACE" in token_types
        assert "RBRACE" in token_types

    def test_struct_declaration(self):
        code = """
        struct Point {
            float x, y, z;
        };
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "STRUCT" in token_types
        assert "LBRACE" in token_types
        assert "RBRACE" in token_types
        assert "SEMICOLON" in token_types

    def test_control_flow_keywords(self):
        code = """
        if else for while do switch case default break continue return
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "IF" in token_types
        assert "ELSE" in token_types
        assert "FOR" in token_types
        assert "WHILE" in token_types
        assert "SWITCH" in token_types
        assert "CASE" in token_types
        assert "DEFAULT" in token_types
        assert "BREAK" in token_types
        assert "CONTINUE" in token_types
        assert "RETURN" in token_types

    def test_access_specifiers(self):
        code = """
        public: private: protected:
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "PUBLIC" in token_types
        assert "PRIVATE" in token_types
        assert "PROTECTED" in token_types

    def test_memory_qualifiers(self):
        code = """
        __shared__ float shared_data[256];
        __constant__ int constant_value = 42;
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "__SHARED__" in token_types
        assert "__CONSTANT__" in token_types

    def test_math_functions(self):
        code = """
        sinf cosf tanf sqrtf powf fabsf floorf ceilf
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_values = [token.value for token in tokens if token.type == "IDENTIFIER"]

        assert "sinf" in token_values
        assert "sqrtf" in token_values
        assert "powf" in token_values

    def test_vector_constructors(self):
        code = """
        make_float2(1.0f, 2.0f) make_int3(1, 2, 3) make_float4(0.0f)
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_values = [token.value for token in tokens if token.type == "IDENTIFIER"]

        assert "make_float2" in token_values
        assert "make_int3" in token_values
        assert "make_float4" in token_values

    def test_comments(self):
        code = """
        // Single line comment
        /* Multi-line
           comment */
        int x = 5; // End of line comment
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_values = [token.value for token in tokens if token.type == "IDENTIFIER"]
        assert "x" in token_values

    def test_multiline_comments_advance_token_locations(self):
        code = "int a;\n/* line 2\n   line 3 */\nfloat b;"
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        float_token = next(token for token in tokens if token.value == "float")
        name_token = next(token for token in tokens if token.value == "b")

        assert (float_token.line, float_token.column) == (4, 1)
        assert (name_token.line, name_token.column) == (4, 7)

    def test_inline_multiline_comments_advance_token_columns(self):
        code = "int a; /* gap */ float b;"
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        float_token = next(token for token in tokens if token.value == "float")
        name_token = next(token for token in tokens if token.value == "b")

        assert (float_token.line, float_token.column) == (1, 18)
        assert (name_token.line, name_token.column) == (1, 24)

    def test_whitespace_handling(self):
        code = """

        int    x   =   5   ;

        float y = 3.14f;

        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens if token.type != "NEWLINE"]

        assert "INT" in token_types
        assert "IDENTIFIER" in token_types
        assert "ASSIGN" in token_types
        assert "INTEGER" in token_types

    def test_complex_expression(self):
        code = """
        result = (a.x * b.y) + sqrt(c[i] - d->member) / 2.0f;
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_types = [token.type for token in tokens]

        assert "IDENTIFIER" in token_types
        assert "ASSIGN" in token_types
        assert "LPAREN" in token_types
        assert "DOT" in token_types
        assert "STAR" in token_types
        assert "PLUS" in token_types
        assert "LBRACKET" in token_types
        assert "ARROW" in token_types
        assert "SLASH" in token_types

    def test_hip_api_calls(self):
        code = """
        hipError_t err = hipMalloc(&ptr, size);
        hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, 0, args);
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        token_pairs = [
            (token.type, token.value) for token in tokens if token.type != "NEWLINE"
        ]
        token_values = [token.value for token in tokens if token.type == "IDENTIFIER"]

        assert ("HIPERROR", "hipError_t") in token_pairs

        assert "hipMalloc" in token_values
        assert "hipLaunchKernelGGL" in token_values

    def test_empty_input(self):
        code = ""
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        assert len(tokens) == 0

    def test_only_whitespace(self):
        code = "   \n  \t  \n  "
        lexer = HipLexer(code)
        tokens = lexer.tokenize()

        non_newline_tokens = [token for token in tokens if token.type != "NEWLINE"]
        assert len(non_newline_tokens) == 0
