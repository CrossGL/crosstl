"""
Tests for HIP Lexer

This module contains tests for the HIP lexer functionality,
ensuring proper tokenization of HIP code constructs.
"""

import pytest
from crosstl.backend.HIP.HipLexer import HipLexer


class TestHipLexer:
    """Test cases for HIP lexer"""

    def test_keywords_tokenization(self):
        """Test HIP keyword tokenization"""
        code = """
        __global__ __device__ __host__ __shared__ __constant__
        __forceinline__ __noinline__ template typename class
        struct union enum namespace using extern static const
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        # Extract token types
        token_types = [token.type for token in tokens]
        
        # Check for HIP-specific keywords
        assert '__GLOBAL__' in token_types
        assert '__DEVICE__' in token_types
        assert '__HOST__' in token_types
        assert '__SHARED__' in token_types
        assert '__CONSTANT__' in token_types
        assert '__FORCEINLINE__' in token_types
        assert '__NOINLINE__' in token_types
        assert 'TEMPLATE' in token_types
        assert 'TYPENAME' in token_types
        assert 'CLASS' in token_types
        assert 'STRUCT' in token_types

    def test_hip_vector_types(self):
        """Test HIP vector type tokenization"""
        code = """
        int2 int3 int4 float2 float3 float4
        double2 double3 double4 uint2 uint3 uint4
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        # All vector types should be tokenized as identifiers
        token_values = [token.value for token in tokens if token.type == 'IDENTIFIER']
        
        assert 'int2' in token_values
        assert 'float3' in token_values
        assert 'double4' in token_values

    def test_hip_builtin_variables(self):
        """Test HIP built-in variable tokenization"""
        code = """
        threadIdx.x blockIdx.y blockDim.z gridDim.x
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_values = [token.value for token in tokens]
        
        assert 'threadIdx' in token_values
        assert 'blockIdx' in token_values
        assert 'blockDim' in token_values
        assert 'gridDim' in token_values

    def test_hip_memory_functions(self):
        """Test HIP memory function tokenization"""
        code = """
        hipMalloc hipFree hipMemcpy hipMemset
        __syncthreads __threadfence __threadfence_block
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_values = [token.value for token in tokens if token.type == 'IDENTIFIER']
        
        assert 'hipMalloc' in token_values
        assert '__syncthreads' in token_values
        assert '__threadfence' in token_values

    def test_operators_tokenization(self):
        """Test operator tokenization"""
        code = """
        + - * / % == != < > <= >= && || ! & | ^ << >> ++ --
        += -= *= /= %= &= |= ^= <<= >>= = ? : . -> ::
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'PLUS' in token_types
        assert 'MINUS' in token_types
        assert 'STAR' in token_types
        assert 'SLASH' in token_types
        assert 'EQ' in token_types
        assert 'NE' in token_types
        assert 'AND' in token_types
        assert 'OR' in token_types
        assert 'ARROW' in token_types
        assert 'SCOPE' in token_types

    def test_numeric_literals(self):
        """Test numeric literal tokenization"""
        code = """
        42 3.14f 2.71828 0xFF 0777 1e5 2.5e-3f
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        # Check for different numeric types
        integers = [token for token in tokens if token.type == 'INTEGER']
        floats = [token for token in tokens if token.type == 'FLOAT']
        
        assert len(integers) >= 2  # At least decimal and hex
        assert len(floats) >= 2   # At least regular float and scientific notation

    def test_string_literals(self):
        """Test string literal tokenization"""
        code = '''
        "hello world" 'c' "escaped \\"string\\"" "path/to/file"
        '''
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        strings = [token for token in tokens if token.type == 'STRING']
        chars = [token for token in tokens if token.type == 'CHAR']
        
        assert len(strings) >= 3
        assert len(chars) >= 1

    def test_preprocessor_directives(self):
        """Test preprocessor directive tokenization"""
        code = """
        #include <hip/hip_runtime.h>
        #define MAX_SIZE 1024
        #ifdef DEBUG
        #endif
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'HASH' in token_types
        assert 'IDENTIFIER' in token_types  # include, define, etc.

    def test_delimiters_tokenization(self):
        """Test delimiter tokenization"""
        code = """
        ( ) [ ] { } , ; : ::
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'LPAREN' in token_types
        assert 'RPAREN' in token_types
        assert 'LBRACKET' in token_types
        assert 'RBRACKET' in token_types
        assert 'LBRACE' in token_types
        assert 'RBRACE' in token_types
        assert 'COMMA' in token_types
        assert 'SEMICOLON' in token_types
        assert 'COLON' in token_types
        assert 'SCOPE' in token_types

    def test_template_syntax(self):
        """Test template syntax tokenization"""
        code = """
        template<typename T> class Vector<T, int N>
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        token_values = [token.value for token in tokens]
        
        assert 'TEMPLATE' in token_types
        assert 'LT' in token_types
        assert 'GT' in token_types
        assert 'TYPENAME' in token_types

    def test_function_declaration(self):
        """Test function declaration tokenization"""
        code = """
        __global__ void kernel(float* data, int size) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert '__GLOBAL__' in token_types
        assert 'VOID' in token_types
        assert 'LPAREN' in token_types
        assert 'RPAREN' in token_types
        assert 'LBRACE' in token_types
        assert 'RBRACE' in token_types

    def test_struct_declaration(self):
        """Test struct declaration tokenization"""
        code = """
        struct Point {
            float x, y, z;
        };
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'STRUCT' in token_types
        assert 'LBRACE' in token_types
        assert 'RBRACE' in token_types
        assert 'SEMICOLON' in token_types

    def test_control_flow_keywords(self):
        """Test control flow keyword tokenization"""
        code = """
        if else for while do switch case default break continue return
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'IF' in token_types
        assert 'ELSE' in token_types
        assert 'FOR' in token_types
        assert 'WHILE' in token_types
        assert 'SWITCH' in token_types
        assert 'CASE' in token_types
        assert 'DEFAULT' in token_types
        assert 'BREAK' in token_types
        assert 'CONTINUE' in token_types
        assert 'RETURN' in token_types

    def test_access_specifiers(self):
        """Test access specifier tokenization"""
        code = """
        public: private: protected:
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'PUBLIC' in token_types
        assert 'PRIVATE' in token_types
        assert 'PROTECTED' in token_types

    def test_memory_qualifiers(self):
        """Test memory qualifier tokenization"""
        code = """
        __shared__ float shared_data[256];
        __constant__ int constant_value = 42;
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert '__SHARED__' in token_types
        assert '__CONSTANT__' in token_types

    def test_math_functions(self):
        """Test math function tokenization"""
        code = """
        sinf cosf tanf sqrtf powf fabsf floorf ceilf
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_values = [token.value for token in tokens if token.type == 'IDENTIFIER']
        
        assert 'sinf' in token_values
        assert 'sqrtf' in token_values
        assert 'powf' in token_values

    def test_vector_constructors(self):
        """Test vector constructor tokenization"""
        code = """
        make_float2(1.0f, 2.0f) make_int3(1, 2, 3) make_float4(0.0f)
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_values = [token.value for token in tokens if token.type == 'IDENTIFIER']
        
        assert 'make_float2' in token_values
        assert 'make_int3' in token_values
        assert 'make_float4' in token_values

    def test_comments(self):
        """Test comment tokenization"""
        code = """
        // Single line comment
        /* Multi-line
           comment */
        int x = 5; // End of line comment
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        # Comments should be filtered out, but code should still tokenize
        token_values = [token.value for token in tokens if token.type == 'IDENTIFIER']
        assert 'x' in token_values

    def test_whitespace_handling(self):
        """Test whitespace and newline handling"""
        code = """
        
        int    x   =   5   ;
        
        float y = 3.14f;
        
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        # Should have proper tokens despite irregular whitespace
        token_types = [token.type for token in tokens if token.type != 'NEWLINE']
        
        assert 'INT' in token_types
        assert 'IDENTIFIER' in token_types
        assert 'ASSIGN' in token_types
        assert 'INTEGER' in token_types

    def test_complex_expression(self):
        """Test complex expression tokenization"""
        code = """
        result = (a.x * b.y) + sqrt(c[i] - d->member) / 2.0f;
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_types = [token.type for token in tokens]
        
        assert 'IDENTIFIER' in token_types
        assert 'ASSIGN' in token_types
        assert 'LPAREN' in token_types
        assert 'DOT' in token_types
        assert 'STAR' in token_types
        assert 'PLUS' in token_types
        assert 'LBRACKET' in token_types
        assert 'ARROW' in token_types
        assert 'SLASH' in token_types

    def test_hip_api_calls(self):
        """Test HIP API call tokenization"""
        code = """
        hipError_t err = hipMalloc(&ptr, size);
        hipLaunchKernelGGL(kernel, gridSize, blockSize, 0, 0, args);
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        token_values = [token.value for token in tokens if token.type == 'IDENTIFIER']
        
        assert 'hipError_t' in token_values
        assert 'hipMalloc' in token_values
        assert 'hipLaunchKernelGGL' in token_values

    def test_empty_input(self):
        """Test empty input handling"""
        code = ""
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        assert len(tokens) == 0

    def test_only_whitespace(self):
        """Test whitespace-only input"""
        code = "   \n  \t  \n  "
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        
        # Should only contain newlines
        non_newline_tokens = [token for token in tokens if token.type != 'NEWLINE']
        assert len(non_newline_tokens) == 0 