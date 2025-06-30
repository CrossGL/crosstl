import re
from typing import List


# Token class for HIP lexer
class Token:
    def __init__(self, token_type: str, value: str, line: int = 1, column: int = 1):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"


# Skip tokens that don't need processing
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

# HIP token definitions - order matters for correct tokenization
TOKENS = tuple(
    [
        # Comments
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        # HIP execution configuration (must come before operators)
        ("KERNEL_LAUNCH_START", r"<<<"),
        ("KERNEL_LAUNCH_END", r">>>"),
        # HIP keywords and qualifiers
        ("__GLOBAL__", r"\b__global__\b"),
        ("__DEVICE__", r"\b__device__\b"),
        ("__HOST__", r"\b__host__\b"),
        ("__SHARED__", r"\b__shared__\b"),
        ("__CONSTANT__", r"\b__constant__\b"),
        ("__RESTRICT__", r"\b__restrict__\b"),
        ("__MANAGED__", r"\b__managed__\b"),
        ("__NOINLINE__", r"\b__noinline__\b"),
        ("__FORCEINLINE__", r"\b__forceinline__\b"),
        # HIP built-in variables
        ("THREADIDX", r"\bthreadIdx\b"),
        ("BLOCKIDX", r"\bblockIdx\b"),
        ("GRIDDIM", r"\bgridDim\b"),
        ("BLOCKDIM", r"\bblockDim\b"),
        ("WARPSIZE", r"\bwarpSize\b"),
        ("HIPTHREADIDX", r"\bhipThreadIdx_x\b|hipThreadIdx_y\b|hipThreadIdx_z\b"),
        ("HIPBLOCKIDX", r"\bhipBlockIdx_x\b|hipBlockIdx_y\b|hipBlockIdx_z\b"),
        ("HIPBLOCKDIM", r"\bhipBlockDim_x\b|hipBlockDim_y\b|hipBlockDim_z\b"),
        ("HIPGRIDDIM", r"\bhipGridDim_x\b|hipGridDim_y\b|hipGridDim_z\b"),
        # HIP built-in functions
        ("SYNCTHREADS", r"\b__syncthreads\b|hipDeviceSynchronize\b"),
        ("SYNCWARP", r"\b__syncwarp\b"),
        ("ATOMICADD", r"\batomicAdd\b|hipAtomicAdd\b"),
        ("ATOMICSUB", r"\batomicSub\b|hipAtomicSub\b"),
        ("ATOMICMAX", r"\batomicMax\b|hipAtomicMax\b"),
        ("ATOMICMIN", r"\batomicMin\b|hipAtomicMin\b"),
        ("ATOMICEXCH", r"\batomicExch\b|hipAtomicExch\b"),
        ("ATOMICCAS", r"\batomicCAS\b|hipAtomicCAS\b"),
        # HIP error handling
        ("HIPERROR", r"\bhipError_t\b"),
        ("HIPSUCCESS", r"\bhipSuccess\b"),
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
        # HIP and C++ types
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
        # HIP vector types
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
        # HIP texture types
        ("TEXTURE", r"\btexture\b"),
        ("SURFACE", r"\bsurface\b"),
        ("HIPARRAY", r"\bhipArray\b"),
        ("HIPARRAYT", r"\bhipArray_t\b"),
        # Boolean literals
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        ("NULL", r"\bNULL\b"),
        ("NULLPTR", r"\bnullptr\b"),
        # Identifiers and literals (must come after keywords)
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("INTEGER", r"\d+[lLuU]*"),
        (
            "FLOAT",
            r"\d*\.\d+([fFdDlL]*|[eE][+-]?\d+[fFdDlL]*)?|\d+[eE][+-]?\d+[fFdDlL]*|\d+[fFdDlL]",
        ),
        ("STRING", r'"([^"\\]|\\.)*"'),
        ("CHAR", r"'([^'\\]|\\.)'"),
        # Preprocessor
        ("HASH", r"#"),
        # Operators (multi-character first)
        ("SCOPE", r"::"),
        ("LSHIFT", r"<<"),
        ("RSHIFT", r">>"),
        ("PLUS_ASSIGN", r"\+="),
        ("MINUS_ASSIGN", r"-="),
        ("STAR_ASSIGN", r"\*="),
        ("SLASH_ASSIGN", r"/="),
        ("PERCENT_ASSIGN", r"%="),
        ("AND_ASSIGN", r"&="),
        ("OR_ASSIGN", r"\|="),
        ("XOR_ASSIGN", r"\^="),
        ("LSHIFT_ASSIGN", r"<<="),
        ("RSHIFT_ASSIGN", r">>="),
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("EQ", r"=="),
        ("NE", r"!="),
        ("LE", r"<="),
        ("GE", r">="),
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("ARROW", r"->"),
        ("DOT", r"\."),
        # Single character operators
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("STAR", r"\*"),
        ("SLASH", r"/"),
        ("PERCENT", r"%"),
        ("ASSIGN", r"="),
        ("LT", r"<"),
        ("GT", r">"),
        ("NOT", r"!"),
        ("AMPERSAND", r"&"),
        ("PIPE", r"\|"),
        ("XOR", r"\^"),
        ("TILDE", r"~"),
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
        # Whitespace and newlines (must be last)
        ("NEWLINE", r"\n"),
        ("WHITESPACE", r"[ \t\r]+"),
    ]
)

# Keywords mapping for reserved word detection
KEYWORDS = {
    "__global__": "__GLOBAL__",
    "__device__": "__DEVICE__",
    "__host__": "__HOST__",
    "__shared__": "__SHARED__",
    "__constant__": "__CONSTANT__",
    "__restrict__": "__RESTRICT__",
    "__managed__": "__MANAGED__",
    "__noinline__": "__NOINLINE__",
    "__forceinline__": "__FORCEINLINE__",
    "threadIdx": "THREADIDX",
    "blockIdx": "BLOCKIDX",
    "gridDim": "GRIDDIM",
    "blockDim": "BLOCKDIM",
    "warpSize": "WARPSIZE",
}


class HipLexer:
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.reserved_keywords = KEYWORDS
        self.line = 1
        self.column = 1

    def tokenize(self) -> List[Token]:
        """Tokenize the input code and return list of tokens"""
        tokens = []
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
                tokens.append(Token(token_type, text, self.line, self.column))

            # Update line and column tracking
            if token_type == "NEWLINE":
                self.line += 1
                self.column = 1
            else:
                self.column += len(text)

            pos = new_pos

        return tokens

    def _next_token(self, pos: int):
        """Find the next token starting at the given position"""
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                text = match.group(0)
                return match.end(), token_type, text
        return None


# For compatibility with existing test expectations
def parse_hip_code(code: str):
    """
    Parse HIP code and return the AST

    Args:
        code: HIP source code as string

    Returns:
        AST representing the parsed HIP code
    """
    from .HipParser import HipParser

    lexer = HipLexer(code)
    tokens = lexer.tokenize()
    parser = HipParser(tokens)
    return parser.parse()
