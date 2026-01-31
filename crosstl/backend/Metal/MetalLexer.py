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
        ("PREPROCESSOR", r"#\w+"),
        # Metal attributes (must come before LBRACKET)
        ("ATTRIBUTE", r"\[\[.*?\]\]"),
        # Keywords - struct and type qualifiers
        ("STRUCT", r"\bstruct\b"),
        ("CONSTANT", r"\bconstant\b"),
        ("DEVICE", r"\bdevice\b"),
        ("THREADGROUP", r"\bthreadgroup\b"),
        ("THREAD", r"\bthread\b"),
        ("CONST", r"\bconst\b"),
        ("STATIC", r"\bstatic\b"),
        ("INLINE", r"\binline\b"),
        # Texture and sampler types
        ("TEXTURE2D", r"\btexture2d\b"),
        ("TEXTURE3D", r"\btexture3d\b"),
        ("TEXTURECUBE", r"\btexturecube\b"),
        ("TEXTURE2D_ARRAY", r"\btexture2d_array\b"),
        ("DEPTH2D", r"\bdepth2d\b"),
        ("BUFFER", r"\bbuffer\b"),
        ("SAMPLER", r"\bsampler\b"),
        # Matrix types (must come before vector types)
        ("MATRIX", r"\b(float|half)[2-4]x[2-4]\b"),
        # Vector types
        ("VECTOR", r"\b(float|half|int|uint|short|ushort|char|uchar|bool)[2-4]\b"),
        # Scalar types
        ("FLOAT", r"\bfloat\b"),
        ("HALF", r"\bhalf\b"),
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
        # Namespace
        ("USING", r"\busing\b"),
        ("NAMESPACE", r"\bnamespace\b"),
        ("METAL", r"\bmetal\b"),
        # Boolean literals
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        # Identifiers (must come after all keywords)
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        # Numeric literals (with suffixes for float/half/uint)
        (
            "NUMBER",
            r"\d+\.\d+([eE][+-]?\d+)?[fFhH]?|\d+[fFhHuUlL]*|\d+[eE][+-]?\d+[fFhH]?",
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
        ("COLON", r":"),
        ("QUESTION", r"\?"),
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
        # Bitwise operators
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        ("BITWISE_NOT", r"~"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_AND", r"&"),
        # Assignment operators (compound first)
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("ASSIGN_MOD", r"%="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("EQUALS", r"="),
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
    "thread": "THREAD",
    "const": "CONST",
    "static": "STATIC",
    "inline": "INLINE",
    "texture2d": "TEXTURE2D",
    "texture3d": "TEXTURE3D",
    "texturecube": "TEXTURECUBE",
    "texture2d_array": "TEXTURE2D_ARRAY",
    "depth2d": "DEPTH2D",
    "buffer": "BUFFER",
    "sampler": "SAMPLER",
    "float": "FLOAT",
    "half": "HALF",
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
