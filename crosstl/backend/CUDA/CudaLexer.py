import re
from typing import Iterator, Tuple, List
from enum import Enum, auto

# Skip tokens that don't need processing
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
        ("DEVICE", r"\b__device__\b"),
        ("HOST", r"\b__host__\b"),
        ("SHARED", r"\b__shared__\b"),
        ("CONSTANT", r"\b__constant__\b"),
        ("RESTRICT", r"\b__restrict__\b"),
        ("MANAGED", r"\b__managed__\b"),
        ("NOINLINE", r"\b__noinline__\b"),
        ("FORCEINLINE", r"\b__forceinline__\b"),
        # CUDA built-in variables
        ("THREADIDX", r"\bthreadIdx\b"),
        ("BLOCKIDX", r"\bblockIdx\b"),
        ("GRIDDIM", r"\bgridDim\b"),
        ("BLOCKDIM", r"\bblockDim\b"),
        ("WARPSIZE", r"\bwarpSize\b"),
        # CUDA built-in functions
        ("SYNCTHREADS", r"\b__syncthreads\b"),
        ("SYNCWARP", r"\b__syncwarp\b"),
        ("ATOMICADD", r"\batomicAdd\b"),
        ("ATOMICSUB", r"\batomicSub\b"),
        ("ATOMICMAX", r"\batomicMax\b"),
        ("ATOMICMIN", r"\batomicMin\b"),
        ("ATOMICEXCH", r"\batomicExch\b"),
        ("ATOMICCAS", r"\batomicCAS\b"),
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
        ("INLINE", r"\binline\b"),
        ("CONST", r"\bconst\b"),
        ("VOLATILE", r"\bvolatile\b"),
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
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d+)?([fFdDlLuU]*|[eE][+-]?\d+[fFdDlLuU]*)?"),
        ("STRING", r'"([^"\\]|\\.)*"'),
        ("CHAR_LIT", r"'([^'\\]|\\.)'"),
        # Preprocessor
        ("PREPROCESSOR", r"#[^\n]*"),
        # Operators (multi-character first)
        ("SCOPE", r"::"),
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
        ("SHIFT_LEFT_EQUALS", r"<<="),
        ("SHIFT_RIGHT_EQUALS", r">>="),
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
    "__device__": "DEVICE",
    "__host__": "HOST",
    "__shared__": "SHARED",
    "__constant__": "CONSTANT",
    "__restrict__": "RESTRICT",
    "__managed__": "MANAGED",
    "__noinline__": "NOINLINE",
    "__forceinline__": "FORCEINLINE",
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
    "inline": "INLINE",
    "const": "CONST",
    "volatile": "VOLATILE",
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
    COMMENT_SINGLE = auto()
    COMMENT_MULTI = auto()
    KERNEL_LAUNCH_START = auto()
    KERNEL_LAUNCH_END = auto()
    GLOBAL = auto()
    DEVICE = auto()
    HOST = auto()
    SHARED = auto()
    CONSTANT = auto()
    RESTRICT = auto()
    MANAGED = auto()
    NOINLINE = auto()
    FORCEINLINE = auto()
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
    INLINE = auto()
    CONST = auto()
    VOLATILE = auto()
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
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.reserved_keywords = KEYWORDS

    def tokenize(self) -> List[Tuple[str, str]]:
        """Tokenize the input code and return list of tokens"""
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        """Generator function that yields tokens one at a time"""
        pos = 0
        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
            new_pos, token_type, text = token

            # Check if identifier is a reserved keyword
            if token_type == "IDENTIFIER" and text in self.reserved_keywords:
                token_type = self.reserved_keywords[text]

            # Skip whitespace and comments unless needed for debugging
            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Tuple[int, str, str]:
        """Find the next token starting at the given position"""
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str) -> "CudaLexer":
        """Create a lexer instance from a file"""
        with open(filepath, "r") as f:
            return cls(f.read())


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
