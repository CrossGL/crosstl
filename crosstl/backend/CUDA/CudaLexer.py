"""Lexer for importing CUDA source into CrossGL Translator."""

import re
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

from .preprocessor import CudaPreprocessor

SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

# CUDA token definitions - order matters for correct tokenization
TOKENS = tuple(
    [
        # Comments
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        # CUDA execution configuration (must come before operators)
        ("KERNEL_LAUNCH_START", r"<<<"),
        ("KERNEL_LAUNCH_END", r">>>"),
        # CUDA keywords and qualifiers
        ("GLOBAL", r"\b__global__\b"),
        ("TILE_GLOBAL", r"\b__tile_global__\b"),
        ("TILE", r"\b__tile__\b"),
        ("DEVICE", r"\b__device__\b"),
        ("HOST", r"\b__host__\b"),
        ("SHARED", r"\b__shared__\b"),
        ("CONSTANT", r"\b__constant__\b"),
        ("RESTRICT", r"\b__restrict(?:__)?\b"),
        ("MANAGED", r"\b__managed__\b"),
        ("NOINLINE", r"\b__noinline__\b"),
        ("FORCEINLINE", r"\b__forceinline__\b"),
        ("LAUNCH_BOUNDS", r"\b__launch_bounds__\b"),
        ("CLUSTER_DIMS", r"\b__cluster_dims__\b"),
        ("BLOCK_SIZE", r"\b__block_size__\b"),
        ("GRID_CONSTANT", r"\b__grid_constant__\b"),
        ("ALIGNAS", r"\b(?:alignas|__align__|__builtin_align__)\b"),
        ("ASM", r"\b(?:asm|__asm__)\b"),
        # CUDA built-in variables
        ("THREADIDX", r"\bthreadIdx\b"),
        ("BLOCKIDX", r"\bblockIdx\b"),
        ("GRIDDIM", r"\bgridDim\b"),
        ("BLOCKDIM", r"\bblockDim\b"),
        ("WARPSIZE", r"\bwarpSize\b"),
        # CUDA built-in functions
        ("SYNCTHREADS", r"\b__syncthreads\b"),
        ("SYNCWARP", r"\b__syncwarp\b"),
        ("ATOMICADD", r"\batomicAdd(?:_(?:block|system))?\b"),
        ("ATOMICSUB", r"\batomicSub(?:_(?:block|system))?\b"),
        ("ATOMICMAX", r"\batomicMax(?:_(?:block|system))?\b"),
        ("ATOMICMIN", r"\batomicMin(?:_(?:block|system))?\b"),
        ("ATOMICEXCH", r"\batomicExch(?:_(?:block|system))?\b"),
        ("ATOMICCAS", r"\batomicCAS(?:_(?:block|system))?\b"),
        ("ATOMICAND", r"\batomicAnd(?:_(?:block|system))?\b"),
        ("ATOMICOR", r"\batomicOr(?:_(?:block|system))?\b"),
        ("ATOMICXOR", r"\batomicXor(?:_(?:block|system))?\b"),
        ("ATOMICINC", r"\batomicInc(?:_(?:block|system))?\b"),
        ("ATOMICDEC", r"\batomicDec(?:_(?:block|system))?\b"),
        # Standard C/C++ keywords
        ("TYPEDEF", r"\btypedef\b"),
        ("STRUCT", r"\bstruct\b"),
        ("UNION", r"\bunion\b"),
        ("ENUM", r"\benum\b"),
        ("CLASS", r"\bclass\b"),
        ("NAMESPACE", r"\bnamespace\b"),
        ("TEMPLATE", r"\btemplate\b"),
        ("TYPENAME", r"\btypename\b"),
        ("EXTERN", r"\bextern\b"),
        ("STATIC", r"\bstatic\b"),
        ("CONSTEXPR", r"\bconstexpr\b"),
        ("INLINE", r"\b(?:inline|__inline__)\b"),
        ("CONST", r"\bconst\b"),
        ("VOLATILE", r"\b(?:volatile|__volatile__)\b"),
        ("REGISTER", r"\bregister\b"),
        ("MUTABLE", r"\bmutable\b"),
        ("VIRTUAL", r"\bvirtual\b"),
        ("PUBLIC", r"\bpublic\b"),
        ("PRIVATE", r"\bprivate\b"),
        ("PROTECTED", r"\bprotected\b"),
        # Control flow
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("DO", r"\bdo\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("RETURN", r"\breturn\b"),
        ("GOTO", r"\bgoto\b"),
        # CUDA and C++ types
        ("VOID", r"\bvoid\b"),
        ("BOOL", r"\bbool\b"),
        ("CHAR", r"\bchar\b"),
        ("SHORT", r"\bshort\b"),
        ("INT", r"\bint\b"),
        ("LONG", r"\blong\b"),
        ("FLOAT", r"\bfloat\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("SIGNED", r"\bsigned\b"),
        ("UNSIGNED", r"\bunsigned\b"),
        ("SIZE_T", r"\bsize_t\b"),
        # CUDA vector types
        ("FLOAT2", r"\bfloat2\b"),
        ("FLOAT3", r"\bfloat3\b"),
        ("FLOAT4", r"\bfloat4\b"),
        ("DOUBLE2", r"\bdouble2\b"),
        ("DOUBLE3", r"\bdouble3\b"),
        ("DOUBLE4", r"\bdouble4\b"),
        ("INT2", r"\bint2\b"),
        ("INT3", r"\bint3\b"),
        ("INT4", r"\bint4\b"),
        ("UINT2", r"\buint2\b"),
        ("UINT3", r"\buint3\b"),
        ("UINT4", r"\buint4\b"),
        ("CHAR2", r"\bchar2\b"),
        ("CHAR3", r"\bchar3\b"),
        ("CHAR4", r"\bchar4\b"),
        ("UCHAR2", r"\buchar2\b"),
        ("UCHAR3", r"\buchar3\b"),
        ("UCHAR4", r"\buchar4\b"),
        ("SHORT2", r"\bshort2\b"),
        ("SHORT3", r"\bshort3\b"),
        ("SHORT4", r"\bshort4\b"),
        ("USHORT2", r"\bushort2\b"),
        ("USHORT3", r"\bushort3\b"),
        ("USHORT4", r"\bushort4\b"),
        ("LONG2", r"\blong2\b"),
        ("LONG3", r"\blong3\b"),
        ("LONG4", r"\blong4\b"),
        ("ULONG2", r"\bulong2\b"),
        ("ULONG3", r"\bulong3\b"),
        ("ULONG4", r"\bulong4\b"),
        ("LONGLONG2", r"\blonglong2\b"),
        ("LONGLONG3", r"\blonglong3\b"),
        ("LONGLONG4", r"\blonglong4\b"),
        ("ULONGLONG2", r"\bulonglong2\b"),
        ("ULONGLONG3", r"\bulonglong3\b"),
        ("ULONGLONG4", r"\bulonglong4\b"),
        # CUDA texture types
        ("TEXTURE", r"\btexture\b"),
        ("SURFACE", r"\bsurface\b"),
        ("CUDAARRAY", r"\bcudaArray\b"),
        ("CUDAARRAYT", r"\bcudaArray_t\b"),
        # Boolean literals
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        ("NULL", r"\bNULL\b"),
        ("NULLPTR", r"\bnullptr\b"),
        # Identifiers and literals (must come after keywords)
        (
            "NUMBER",
            r"(?:0[xX][0-9a-fA-F](?:'?[0-9a-fA-F])*|0[bB][01](?:'?[01])*|\d(?:'?\d)*\.\d(?:'?\d)*|\d(?:'?\d)*\.|\.\d(?:'?\d)*|\d(?:'?\d)*)(?:[eE][+-]?\d(?:'?\d)*)?[fFdDlLuU]*",
        ),
        ("STRING", r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\([\s\S]*?\)\1"'),
        ("STRING", r'"([^"\\]|\\.)*"'),
        ("CHAR_LIT", r"(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)+'"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        # Preprocessor
        ("PREPROCESSOR", r"#[^\n]*"),
        # Operators (multi-character first)
        ("SCOPE", r"::"),
        ("SHIFT_LEFT_EQUALS", r"<<="),
        ("SHIFT_RIGHT_EQUALS", r">>="),
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("MODULO_EQUALS", r"%="),
        ("AND_EQUALS", r"&="),
        ("OR_EQUALS", r"\|="),
        ("XOR_EQUALS", r"\^="),
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("ARROW", r"->"),
        ("DOT", r"\."),
        # Single character operators
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MODULO", r"%"),
        ("ASSIGN", r"="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("LOGICAL_NOT", r"!"),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_NOT", r"~"),
        ("QUESTION", r"\?"),
        ("COLON", r":"),
        # Delimiters
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("COMMA", r","),
        # Whitespace (must be last)
        ("WHITESPACE", r"\s+"),
    ]
)

# Keywords mapping for reserved word detection
KEYWORDS = {
    "__global__": "GLOBAL",
    "__tile_global__": "TILE_GLOBAL",
    "__tile__": "TILE",
    "__device__": "DEVICE",
    "__host__": "HOST",
    "__shared__": "SHARED",
    "__constant__": "CONSTANT",
    "__restrict__": "RESTRICT",
    "__restrict": "RESTRICT",
    "__managed__": "MANAGED",
    "__noinline__": "NOINLINE",
    "__forceinline__": "FORCEINLINE",
    "__launch_bounds__": "LAUNCH_BOUNDS",
    "__cluster_dims__": "CLUSTER_DIMS",
    "__block_size__": "BLOCK_SIZE",
    "__grid_constant__": "GRID_CONSTANT",
    "alignas": "ALIGNAS",
    "__align__": "ALIGNAS",
    "__builtin_align__": "ALIGNAS",
    "asm": "ASM",
    "__asm__": "ASM",
    "__volatile__": "VOLATILE",
    "threadIdx": "THREADIDX",
    "blockIdx": "BLOCKIDX",
    "gridDim": "GRIDDIM",
    "blockDim": "BLOCKDIM",
    "warpSize": "WARPSIZE",
    "__syncthreads": "SYNCTHREADS",
    "__syncwarp": "SYNCWARP",
    "atomicAdd": "ATOMICADD",
    "atomicSub": "ATOMICSUB",
    "atomicMax": "ATOMICMAX",
    "atomicMin": "ATOMICMIN",
    "atomicExch": "ATOMICEXCH",
    "atomicCAS": "ATOMICCAS",
    "atomicAnd": "ATOMICAND",
    "atomicOr": "ATOMICOR",
    "atomicXor": "ATOMICXOR",
    "atomicInc": "ATOMICINC",
    "atomicDec": "ATOMICDEC",
    "typedef": "TYPEDEF",
    "struct": "STRUCT",
    "union": "UNION",
    "enum": "ENUM",
    "class": "CLASS",
    "namespace": "NAMESPACE",
    "template": "TEMPLATE",
    "typename": "TYPENAME",
    "extern": "EXTERN",
    "static": "STATIC",
    "constexpr": "CONSTEXPR",
    "inline": "INLINE",
    "__inline__": "INLINE",
    "const": "CONST",
    "volatile": "VOLATILE",
    "register": "REGISTER",
    "mutable": "MUTABLE",
    "virtual": "VIRTUAL",
    "public": "PUBLIC",
    "private": "PRIVATE",
    "protected": "PROTECTED",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "do": "DO",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
    "continue": "CONTINUE",
    "return": "RETURN",
    "goto": "GOTO",
    "void": "VOID",
    "bool": "BOOL",
    "char": "CHAR",
    "short": "SHORT",
    "int": "INT",
    "long": "LONG",
    "float": "FLOAT",
    "double": "DOUBLE",
    "signed": "SIGNED",
    "unsigned": "UNSIGNED",
    "size_t": "SIZE_T",
    "true": "TRUE",
    "false": "FALSE",
    "NULL": "NULL",
    "nullptr": "NULLPTR",
}


class TokenType(Enum):
    """Token names emitted by the CUDA lexer."""

    COMMENT_SINGLE = auto()
    COMMENT_MULTI = auto()
    KERNEL_LAUNCH_START = auto()
    KERNEL_LAUNCH_END = auto()
    GLOBAL = auto()
    TILE_GLOBAL = auto()
    TILE = auto()
    DEVICE = auto()
    HOST = auto()
    SHARED = auto()
    CONSTANT = auto()
    RESTRICT = auto()
    MANAGED = auto()
    NOINLINE = auto()
    FORCEINLINE = auto()
    LAUNCH_BOUNDS = auto()
    CLUSTER_DIMS = auto()
    BLOCK_SIZE = auto()
    GRID_CONSTANT = auto()
    ALIGNAS = auto()
    ASM = auto()
    THREADIDX = auto()
    BLOCKIDX = auto()
    GRIDDIM = auto()
    BLOCKDIM = auto()
    WARPSIZE = auto()
    SYNCTHREADS = auto()
    SYNCWARP = auto()
    ATOMICADD = auto()
    ATOMICSUB = auto()
    ATOMICMAX = auto()
    ATOMICMIN = auto()
    ATOMICEXCH = auto()
    ATOMICCAS = auto()
    TYPEDEF = auto()
    STRUCT = auto()
    UNION = auto()
    ENUM = auto()
    CLASS = auto()
    NAMESPACE = auto()
    TEMPLATE = auto()
    TYPENAME = auto()
    EXTERN = auto()
    STATIC = auto()
    CONSTEXPR = auto()
    INLINE = auto()
    CONST = auto()
    VOLATILE = auto()
    REGISTER = auto()
    MUTABLE = auto()
    VIRTUAL = auto()
    PUBLIC = auto()
    PRIVATE = auto()
    PROTECTED = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    GOTO = auto()
    VOID = auto()
    BOOL = auto()
    CHAR = auto()
    SHORT = auto()
    INT = auto()
    LONG = auto()
    FLOAT = auto()
    DOUBLE = auto()
    SIGNED = auto()
    UNSIGNED = auto()
    SIZE_T = auto()
    FLOAT2 = auto()
    FLOAT3 = auto()
    FLOAT4 = auto()
    DOUBLE2 = auto()
    DOUBLE3 = auto()
    DOUBLE4 = auto()
    INT2 = auto()
    INT3 = auto()
    INT4 = auto()
    UINT2 = auto()
    UINT3 = auto()
    UINT4 = auto()
    CHAR2 = auto()
    CHAR3 = auto()
    CHAR4 = auto()
    UCHAR2 = auto()
    UCHAR3 = auto()
    UCHAR4 = auto()
    SHORT2 = auto()
    SHORT3 = auto()
    SHORT4 = auto()
    USHORT2 = auto()
    USHORT3 = auto()
    USHORT4 = auto()
    LONG2 = auto()
    LONG3 = auto()
    LONG4 = auto()
    ULONG2 = auto()
    ULONG3 = auto()
    ULONG4 = auto()
    LONGLONG2 = auto()
    LONGLONG3 = auto()
    LONGLONG4 = auto()
    ULONGLONG2 = auto()
    ULONGLONG3 = auto()
    ULONGLONG4 = auto()
    TEXTURE = auto()
    SURFACE = auto()
    CUDAARRAY = auto()
    CUDAARRAYT = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    NULLPTR = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    CHAR_LIT = auto()
    PREPROCESSOR = auto()
    SCOPE = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    PLUS_EQUALS = auto()
    MINUS_EQUALS = auto()
    MULTIPLY_EQUALS = auto()
    DIVIDE_EQUALS = auto()
    MODULO_EQUALS = auto()
    AND_EQUALS = auto()
    OR_EQUALS = auto()
    XOR_EQUALS = auto()
    SHIFT_LEFT_EQUALS = auto()
    SHIFT_RIGHT_EQUALS = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    ARROW = auto()
    DOT = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    ASSIGN = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LOGICAL_NOT = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    BITWISE_NOT = auto()
    QUESTION = auto()
    COLON = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    WHITESPACE = auto()


class CudaLexer:
    """Tokenize CUDA source for the CUDA backend parser."""

    def __init__(
        self,
        code: str,
        preprocess: bool = True,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict_preprocessor: bool = False,
        max_expansion_depth: int = 64,
        file_path: Optional[str] = None,
    ):
        code = code.lstrip("\ufeff")
        if preprocess:
            preprocessor = CudaPreprocessor(
                include_paths=include_paths,
                defines=defines,
                strict=strict_preprocessor,
                max_expansion_depth=max_expansion_depth,
            )
            code = preprocessor.preprocess(code, file_path=file_path)
        else:
            code = self._join_line_continuations(code)
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.reserved_keywords = KEYWORDS

    @staticmethod
    def _join_line_continuations(code: str) -> str:
        return re.sub(r"\\(?:\r\n|\r|\n)", "", code)

    def tokenize(self) -> List[Tuple[str, str]]:
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        pos = 0
        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
            new_pos, token_type, text = token

            if token_type == "IDENTIFIER" and text in self.reserved_keywords:
                token_type = self.reserved_keywords[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Tuple[int, str, str]:
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(
        cls,
        filepath: str,
        preprocess: bool = True,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict_preprocessor: bool = False,
        max_expansion_depth: int = 64,
    ):
        with open(filepath, encoding="utf-8") as f:
            return cls(
                f.read(),
                preprocess=preprocess,
                include_paths=include_paths,
                defines=defines,
                strict_preprocessor=strict_preprocessor,
                max_expansion_depth=max_expansion_depth,
                file_path=filepath,
            )


# Compatibility wrapper
class Lexer:
    """Compatibility wrapper around CudaLexer"""

    def __init__(self, input_str):
        self.lexer = CudaLexer(input_str)
        self.tokens = self.lexer.tokenize()
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
