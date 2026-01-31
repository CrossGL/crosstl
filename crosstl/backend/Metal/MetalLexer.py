import re
from typing import Iterator, Tuple, List, Optional

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

# Token definitions - order matters! More specific patterns should come first
TOKENS = tuple(
    [
        # Comments (must come first to avoid partial matches)
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        # Preprocessor directives
        ("PREPROCESSOR", r"#[^\n]*"),
        # Metal attributes (must come before LBRACKET)
        ("ATTRIBUTE", r"\[\[.*?\]\]"),
        # Keywords - struct and type qualifiers
        ("STRUCT", r"\bstruct\b"),
        ("CONSTANT", r"\bconstant\b"),
        ("DEVICE", r"\bdevice\b"),
        ("THREADGROUP", r"\bthreadgroup\b"),
        ("THREADGROUP_IMAGEBLOCK", r"\bthreadgroup_imageblock\b"),
        ("THREAD", r"\bthread\b"),
        ("CONST", r"\bconst\b"),
        ("CONSTEXPR", r"\bconstexpr\b"),
        ("STATIC", r"\bstatic\b"),
        ("VOLATILE", r"\bvolatile\b"),
        ("RESTRICT", r"\brestrict\b"),
        ("INLINE", r"\binline\b"),
        ("TYPEDEF", r"\btypedef\b"),
        ("ENUM", r"\benum\b"),
        ("CLASS", r"\bclass\b"),
        # Texture and sampler types
        ("TEXTURE1D", r"\btexture1d\b"),
        ("TEXTURE1D_ARRAY", r"\btexture1d_array\b"),
        ("TEXTURE2D", r"\btexture2d\b"),
        ("TEXTURE2D_MS", r"\btexture2d_ms\b"),
        ("TEXTURE2D_MS_ARRAY", r"\btexture2d_ms_array\b"),
        ("TEXTURE3D", r"\btexture3d\b"),
        ("TEXTURECUBE", r"\btexturecube\b"),
        ("TEXTURECUBE_ARRAY", r"\btexturecube_array\b"),
        ("TEXTURE2D_ARRAY", r"\btexture2d_array\b"),
        ("TEXTUREBUFFER", r"\btexture_buffer\b"),
        ("DEPTH2D", r"\bdepth2d\b"),
        ("DEPTH2D_ARRAY", r"\bdepth2d_array\b"),
        ("DEPTHCUBE", r"\bdepthcube\b"),
        ("DEPTHCUBE_ARRAY", r"\bdepthcube_array\b"),
        ("DEPTH2D_MS", r"\bdepth2d_ms\b"),
        ("DEPTH2D_MS_ARRAY", r"\bdepth2d_ms_array\b"),
        ("ACCELERATION_STRUCTURE", r"\bacceleration_structure\b"),
        ("INTERSECTION_FUNCTION_TABLE", r"\bintersection_function_table\b"),
        ("VISIBLE_FUNCTION_TABLE", r"\bvisible_function_table\b"),
        ("INDIRECT_COMMAND_BUFFER", r"\bindirect_command_buffer\b"),
        ("BUFFER", r"\bbuffer\b"),
        ("SAMPLER", r"\bsampler\b"),
        # Matrix types (must come before vector types)
        ("MATRIX", r"\b(float|half|double)[2-4]x[2-4]\b"),
        ("SIMD_MATRIX", r"\bsimd_float[2-4]x[2-4]\b"),
        # Vector types
        ("VECTOR", r"\b(float|half|double|int|uint|short|ushort|char|uchar|bool)[2-4]\b"),
        ("PACKED_VECTOR", r"\bpacked_(float|half|int|uint)[2-4]\b"),
        ("SIMD_VECTOR", r"\bsimd_(float|int|uint)[2-4]\b"),
        # Scalar types
        ("FLOAT", r"\bfloat\b"),
        ("HALF", r"\bhalf\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("LONG", r"\blong\b"),
        ("ULONG", r"\bulong\b"),
        ("INT8_T", r"\bint8_t\b"),
        ("UINT8_T", r"\buint8_t\b"),
        ("INT16_T", r"\bint16_t\b"),
        ("UINT16_T", r"\buint16_t\b"),
        ("INT32_T", r"\bint32_t\b"),
        ("UINT32_T", r"\buint32_t\b"),
        ("INT", r"\bint\b"),
        ("UINT", r"\buint\b"),
        ("SHORT", r"\bshort\b"),
        ("USHORT", r"\bushort\b"),
        ("CHAR", r"\bchar\b"),
        ("UCHAR", r"\buchar\b"),
        ("BOOL", r"\bbool\b"),
        ("VOID", r"\bvoid\b"),
        ("SIZE_T", r"\bsize_t\b"),
        ("PTRDIFF_T", r"\bptrdiff_t\b"),
        ("INT64_T", r"\bint64_t\b"),
        ("UINT64_T", r"\buint64_t\b"),
        ("ATOMIC_INT", r"\batomic_int\b"),
        ("ATOMIC_UINT", r"\batomic_uint\b"),
        ("ATOMIC_BOOL", r"\batomic_bool\b"),
        # Control flow keywords
        ("RETURN", r"\breturn\b"),
        ("ELSE_IF", r"\belse\s+if\b"),
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
        ("DISCARD", r"\bdiscard\b"),
        # Function qualifiers
        ("KERNEL", r"\bkernel\b"),
        ("VERTEX", r"\bvertex\b"),
        ("FRAGMENT", r"\bfragment\b"),
        ("COMPUTE", r"\bcompute\b"),
        ("INTERSECTION", r"\bintersection\b"),
        ("ANYHIT", r"\banyhit\b"),
        ("CLOSESTHIT", r"\bclosesthit\b"),
        ("MISS", r"\bmiss\b"),
        ("CALLABLE", r"\bcallable\b"),
        ("MESH", r"\bmesh\b"),
        ("OBJECT", r"\bobject\b"),
        ("AMPLIFICATION", r"\bamplification\b"),
        ("SIZEOF", r"\bsizeof\b"),
        ("ALIGNOF", r"\balignof\b"),
        ("ALIGNAS", r"\balignas\b"),
        ("STATIC_ASSERT", r"\bstatic_assert\b"),
        ("READ", r"\bread\b"),
        ("WRITE", r"\bwrite\b"),
        ("READ_WRITE", r"\bread_write\b"),
        # Namespace
        ("USING", r"\busing\b"),
        ("NAMESPACE", r"\bnamespace\b"),
        ("METAL", r"\bmetal\b"),
        # Boolean literals
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        # Identifiers (must come after all keywords)
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        # Numeric literals (decimal/hex/binary with suffixes)
        (
            "NUMBER",
            r"0[xX][0-9a-fA-F]+[uUlL]*|0[bB][01]+[uUlL]*|\d+\.\d+([eE][+-]?\d+)?[fFhH]?|\d+[eE][+-]?\d+[fFhH]?|\d+[fFhHuUlL]*",
        ),
        # Brackets and braces
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        # Punctuation
        ("SEMICOLON", r";"),
        ("STRING", r'"[^"]*"'),
        ("COMMA", r","),
        ("SCOPE", r"::"),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
        # Shift and assignment operators (multi-char first)
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        # Comparison operators (multi-char first)
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        # Logical operators
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("NOT", r"!"),
        # Assignment operators (compound first)
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("ASSIGN_MOD", r"%="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("EQUALS", r"="),
        # Bitwise operators
        ("BITWISE_NOT", r"~"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_AND", r"&"),
        # Arithmetic operators
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MOD", r"%"),
        # Member access
        ("ARROW", r"->"),
        ("DOT", r"\."),
        # Whitespace (skipped)
        ("WHITESPACE", r"\s+"),
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "constant": "CONSTANT",
    "device": "DEVICE",
    "threadgroup": "THREADGROUP",
    "threadgroup_imageblock": "THREADGROUP_IMAGEBLOCK",
    "thread": "THREAD",
    "const": "CONST",
    "constexpr": "CONSTEXPR",
    "static": "STATIC",
    "volatile": "VOLATILE",
    "restrict": "RESTRICT",
    "inline": "INLINE",
    "typedef": "TYPEDEF",
    "enum": "ENUM",
    "class": "CLASS",
    "texture1d": "TEXTURE1D",
    "texture1d_array": "TEXTURE1D_ARRAY",
    "texture2d": "TEXTURE2D",
    "texture2d_ms": "TEXTURE2D_MS",
    "texture2d_ms_array": "TEXTURE2D_MS_ARRAY",
    "texture3d": "TEXTURE3D",
    "texturecube": "TEXTURECUBE",
    "texturecube_array": "TEXTURECUBE_ARRAY",
    "texture2d_array": "TEXTURE2D_ARRAY",
    "texture_buffer": "TEXTUREBUFFER",
    "depth2d": "DEPTH2D",
    "depth2d_array": "DEPTH2D_ARRAY",
    "depthcube": "DEPTHCUBE",
    "depthcube_array": "DEPTHCUBE_ARRAY",
    "depth2d_ms": "DEPTH2D_MS",
    "depth2d_ms_array": "DEPTH2D_MS_ARRAY",
    "acceleration_structure": "ACCELERATION_STRUCTURE",
    "intersection_function_table": "INTERSECTION_FUNCTION_TABLE",
    "visible_function_table": "VISIBLE_FUNCTION_TABLE",
    "indirect_command_buffer": "INDIRECT_COMMAND_BUFFER",
    "buffer": "BUFFER",
    "sampler": "SAMPLER",
    "float": "FLOAT",
    "half": "HALF",
    "double": "DOUBLE",
    "long": "LONG",
    "ulong": "ULONG",
    "int8_t": "INT8_T",
    "uint8_t": "UINT8_T",
    "int16_t": "INT16_T",
    "uint16_t": "UINT16_T",
    "int32_t": "INT32_T",
    "uint32_t": "UINT32_T",
    "int": "INT",
    "uint": "UINT",
    "short": "SHORT",
    "ushort": "USHORT",
    "char": "CHAR",
    "uchar": "UCHAR",
    "bool": "BOOL",
    "void": "VOID",
    "size_t": "SIZE_T",
    "ptrdiff_t": "PTRDIFF_T",
    "int64_t": "INT64_T",
    "uint64_t": "UINT64_T",
    "atomic_int": "ATOMIC_INT",
    "atomic_uint": "ATOMIC_UINT",
    "atomic_bool": "ATOMIC_BOOL",
    "return": "RETURN",
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
    "discard": "DISCARD",
    "kernel": "KERNEL",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "compute": "COMPUTE",
    "intersection": "INTERSECTION",
    "anyhit": "ANYHIT",
    "closesthit": "CLOSESTHIT",
    "miss": "MISS",
    "callable": "CALLABLE",
    "mesh": "MESH",
    "object": "OBJECT",
    "amplification": "AMPLIFICATION",
    "sizeof": "SIZEOF",
    "alignof": "ALIGNOF",
    "alignas": "ALIGNAS",
    "static_assert": "STATIC_ASSERT",
    "read": "READ",
    "write": "WRITE",
    "read_write": "READ_WRITE",
    "using": "USING",
    "namespace": "NAMESPACE",
    "metal": "METAL",
    "true": "TRUE",
    "false": "FALSE",
}


class MetalLexer:
    """Lexer for Metal Shading Language (MSL)"""

    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)

    def tokenize(self) -> List[Tuple[str, str]]:
        """Tokenize the input code and return list of tokens"""
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        """Generator function that yields tokens one at a time"""
        pos = 0
        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                # Provide more context in error message
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)
                context = self.code[max(0, pos - 20) : min(self._length, pos + 20)]
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at line {line_num}, column {col_num}\n"
                    f"Context: ...{context}..."
                )
            new_pos, token_type, text = token

            # Check if identifier is a keyword
            if token_type == "IDENTIFIER" and text in KEYWORDS:
                token_type = KEYWORDS[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Optional[Tuple[int, str, str]]:
        """Find the next token starting at the given position"""
        if self.code.startswith("/*", pos):
            end_pos = self.code.find("*/", pos + 2)
            if end_pos == -1:
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)
                raise SyntaxError(
                    f"Unterminated block comment at line {line_num}, column {col_num}"
                )
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str) -> "MetalLexer":
        """Create a lexer instance from a file"""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls(f.read())
