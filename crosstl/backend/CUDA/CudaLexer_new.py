"""
Modernized CUDA Lexer using base infrastructure.
"""

from ..base_lexer import (
    BaseLexer,
    TokenType,
    Token,
    LanguageSpecificMixin,
    build_standard_patterns,
)
from typing import List, Tuple, Dict


class CudaTokenType(TokenType):
    """CUDA-specific token types extending base types."""

    # CUDA execution configuration
    KERNEL_LAUNCH_START = "kernel_launch_start"
    KERNEL_LAUNCH_END = "kernel_launch_end"

    # CUDA qualifiers
    GLOBAL = "global"
    DEVICE = "device"
    HOST = "host"
    SHARED = "shared"
    CONSTANT = "constant"
    RESTRICT = "restrict"
    MANAGED = "managed"
    NOINLINE = "noinline"
    FORCEINLINE = "forceinline"

    # CUDA built-in variables
    THREADIDX = "threadidx"
    BLOCKIDX = "blockidx"
    GRIDDIM = "griddim"
    BLOCKDIM = "blockdim"
    WARPSIZE = "warpsize"

    # CUDA built-in functions
    SYNCTHREADS = "syncthreads"
    SYNCWARP = "syncwarp"
    ATOMICADD = "atomicadd"
    ATOMICSUB = "atomicsub"
    ATOMICMAX = "atomicmax"
    ATOMICMIN = "atomicmin"
    ATOMICEXCH = "atomicexch"
    ATOMICCAS = "atomiccas"

    # CUDA vector types
    FLOAT2 = "float2"
    FLOAT3 = "float3"
    FLOAT4 = "float4"
    DOUBLE2 = "double2"
    DOUBLE3 = "double3"
    DOUBLE4 = "double4"
    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    UINT2 = "uint2"
    UINT3 = "uint3"
    UINT4 = "uint4"


class CudaLexer(BaseLexer, LanguageSpecificMixin):
    """CUDA lexer implementation using base infrastructure."""

    def get_token_patterns(self) -> List[Tuple[TokenType, str]]:
        """Get CUDA-specific token patterns."""

        # CUDA-specific patterns
        cuda_patterns = [
            # CUDA execution configuration (must come before operators)
            (CudaTokenType.KERNEL_LAUNCH_START, r"<<<"),
            (CudaTokenType.KERNEL_LAUNCH_END, r">>>"),
            # CUDA qualifiers
            (CudaTokenType.GLOBAL, r"\b__global__\b"),
            (CudaTokenType.DEVICE, r"\b__device__\b"),
            (CudaTokenType.HOST, r"\b__host__\b"),
            (CudaTokenType.SHARED, r"\b__shared__\b"),
            (CudaTokenType.CONSTANT, r"\b__constant__\b"),
            (CudaTokenType.RESTRICT, r"\b__restrict__\b"),
            (CudaTokenType.MANAGED, r"\b__managed__\b"),
            (CudaTokenType.NOINLINE, r"\b__noinline__\b"),
            (CudaTokenType.FORCEINLINE, r"\b__forceinline__\b"),
            # CUDA built-in variables
            (CudaTokenType.THREADIDX, r"\bthreadIdx\b"),
            (CudaTokenType.BLOCKIDX, r"\bblockIdx\b"),
            (CudaTokenType.GRIDDIM, r"\bgridDim\b"),
            (CudaTokenType.BLOCKDIM, r"\bblockDim\b"),
            (CudaTokenType.WARPSIZE, r"\bwarpSize\b"),
            # CUDA built-in functions
            (CudaTokenType.SYNCTHREADS, r"\b__syncthreads\b"),
            (CudaTokenType.SYNCWARP, r"\b__syncwarp\b"),
            (CudaTokenType.ATOMICADD, r"\batomicAdd\b"),
            (CudaTokenType.ATOMICSUB, r"\batomicSub\b"),
            (CudaTokenType.ATOMICMAX, r"\batomicMax\b"),
            (CudaTokenType.ATOMICMIN, r"\batomicMin\b"),
            (CudaTokenType.ATOMICEXCH, r"\batomicExch\b"),
            (CudaTokenType.ATOMICCAS, r"\batomicCAS\b"),
            # CUDA vector types
            (CudaTokenType.FLOAT2, r"\bfloat2\b"),
            (CudaTokenType.FLOAT3, r"\bfloat3\b"),
            (CudaTokenType.FLOAT4, r"\bfloat4\b"),
            (CudaTokenType.DOUBLE2, r"\bdouble2\b"),
            (CudaTokenType.DOUBLE3, r"\bdouble3\b"),
            (CudaTokenType.DOUBLE4, r"\bdouble4\b"),
            (CudaTokenType.INT2, r"\bint2\b"),
            (CudaTokenType.INT3, r"\bint3\b"),
            (CudaTokenType.INT4, r"\bint4\b"),
            (CudaTokenType.UINT2, r"\buint2\b"),
            (CudaTokenType.UINT3, r"\buint3\b"),
            (CudaTokenType.UINT4, r"\buint4\b"),
            # Additional C++ keywords
            (TokenType.TYPEDEF, r"\btypedef\b"),
            (TokenType.ENUM, r"\benum\b"),
            (TokenType.CLASS, r"\bclass\b"),
            (TokenType.TYPEDEF, r"\btemplate\b"),
            (TokenType.TYPEDEF, r"\btypename\b"),
            # Preprocessor
            (TokenType.PREPROCESSOR, r"#[^\n]*"),
        ]

        # Combine with base patterns
        base_patterns = build_standard_patterns()
        return self.add_language_patterns(base_patterns, cuda_patterns)

    def get_keywords(self) -> Dict[str, TokenType]:
        """Get CUDA-specific keyword mappings."""

        base_keywords = {
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "for": TokenType.FOR,
            "while": TokenType.WHILE,
            "do": TokenType.DO,
            "switch": TokenType.SWITCH,
            "case": TokenType.CASE,
            "default": TokenType.DEFAULT,
            "break": TokenType.BREAK,
            "continue": TokenType.CONTINUE,
            "return": TokenType.RETURN,
            "struct": TokenType.STRUCT,
            "void": TokenType.VOID,
            "bool": TokenType.BOOL,
            "int": TokenType.INT,
            "float": TokenType.FLOAT_TYPE,
            "double": TokenType.DOUBLE,
            "char": TokenType.CHAR,
            "short": TokenType.SHORT,
            "long": TokenType.LONG,
            "const": TokenType.CONST,
            "static": TokenType.STATIC,
            "inline": TokenType.INLINE,
            "extern": TokenType.EXTERN,
            "typedef": TokenType.TYPEDEF,
            "enum": TokenType.ENUM,
            "true": TokenType.BOOLEAN,
            "false": TokenType.BOOLEAN,
            "signed": TokenType.IDENTIFIER,  # Treat as modifier
            "unsigned": TokenType.IDENTIFIER,  # Treat as modifier
        }

        cuda_keywords = {
            # CUDA qualifiers
            "__global__": CudaTokenType.GLOBAL,
            "__device__": CudaTokenType.DEVICE,
            "__host__": CudaTokenType.HOST,
            "__shared__": CudaTokenType.SHARED,
            "__constant__": CudaTokenType.CONSTANT,
            "__restrict__": CudaTokenType.RESTRICT,
            "__managed__": CudaTokenType.MANAGED,
            "__noinline__": CudaTokenType.NOINLINE,
            "__forceinline__": CudaTokenType.FORCEINLINE,
            # CUDA built-in variables
            "threadIdx": CudaTokenType.THREADIDX,
            "blockIdx": CudaTokenType.BLOCKIDX,
            "gridDim": CudaTokenType.GRIDDIM,
            "blockDim": CudaTokenType.BLOCKDIM,
            "warpSize": CudaTokenType.WARPSIZE,
            # CUDA built-in functions
            "__syncthreads": CudaTokenType.SYNCTHREADS,
            "__syncwarp": CudaTokenType.SYNCWARP,
            "atomicAdd": CudaTokenType.ATOMICADD,
            "atomicSub": CudaTokenType.ATOMICSUB,
            "atomicMax": CudaTokenType.ATOMICMAX,
            "atomicMin": CudaTokenType.ATOMICMIN,
            "atomicExch": CudaTokenType.ATOMICEXCH,
            "atomicCAS": CudaTokenType.ATOMICCAS,
            # CUDA vector types
            "float2": CudaTokenType.FLOAT2,
            "float3": CudaTokenType.FLOAT3,
            "float4": CudaTokenType.FLOAT4,
            "double2": CudaTokenType.DOUBLE2,
            "double3": CudaTokenType.DOUBLE3,
            "double4": CudaTokenType.DOUBLE4,
            "int2": CudaTokenType.INT2,
            "int3": CudaTokenType.INT3,
            "int4": CudaTokenType.INT4,
            "uint2": CudaTokenType.UINT2,
            "uint3": CudaTokenType.UINT3,
            "uint4": CudaTokenType.UINT4,
        }

        return self.merge_keywords(base_keywords, cuda_keywords)


# Backward compatibility wrapper
class Lexer:
    """Compatibility wrapper for existing CUDA lexer interface."""

    def __init__(self, input_str: str):
        self.lexer = CudaLexer(input_str)
        self.tokens = [
            (token.type.name, token.value) for token in self.lexer.tokenize()
        ]
        self.current_pos = 0

    def next(self):
        if self.current_pos < len(self.tokens):
            token = self.tokens[self.current_pos]
            self.current_pos += 1
            return token
        return ("EOF", "")

    def peek(self):
        if self.current_pos < len(self.tokens):
            return self.tokens[self.current_pos]
        return ("EOF", "")
