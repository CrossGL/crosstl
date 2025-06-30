import re
from typing import Iterator, Tuple, List

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("BITWISE_NOT", r"~"),
        ("PREPROCESSOR", r"#\w+"),
        ("STRUCT", r"\bstruct\b"),
        ("CONSTANT", r"\bconstant\b"),
        ("TEXTURE2D", r"\btexture2d\b"),
        ("TEXTURECUBE", r"\btexturecube\b"),
        ("buffer", r"\bbuffer\b"),
        ("SAMPLER", r"\bsampler\b"),
        ("VECTOR", r"\b(float|half|int|uint)[2-4]\b"),
        ("FLOAT", r"\bfloat\b"),
        ("HALF", r"\bhalf\b"),
        ("INT", r"\bint\b"),
        ("UINT", r"\buint\b"),
        ("QUESTION", r"\?"),
        ("BOOL", r"\bbool\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("ELSE_IF", r"\belse\s+if\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("KERNEL", r"\bkernel\b"),
        ("VERTEX", r"\bvertex\b"),
        ("FRAGMENT", r"\bfragment\b"),
        ("USING", r"\busing\b"),
        ("NAMESPACE", r"\bnamespace\b"),
        ("METAL", r"\bmetal\b"),
        ("DEVICE", r"\bdevice\b"),
        ("THREADGROUP", r"\bthreadgroup\b"),
        ("THREAD", r"\bthread\b"),
        ("CONST", r"\bconst\b"),
        ("ATTRIBUTE", r"\[\[.*?\]\]"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d+)?([fFhHuU])?"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("STRING", r'"[^"]*"'),
        ("COMMA", r","),
        ("COLON", r":"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("DOT", r"\."),
        ("EQUALS", r"="),
        ("bitwise_and", r"&"),
        ("WHITESPACE", r"\s+"),
        ("MOD", r"%"),
        ("ASSIGN_MOD", r"%="),
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "constant": "CONSTANT",
    "texture2d": "TEXTURE2D",
    "texturecube": "TEXTURECUBE",
    "sampler": "SAMPLER",
    "float": "FLOAT",
    "half": "HALF",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
    "return": "RETURN",
    "else if": "ELSE_IF",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
    "kernel": "KERNEL",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "using": "USING",
    "namespace": "NAMESPACE",
    "metal": "METAL",
    "device": "DEVICE",
    "threadgroup": "THREADGROUP",
    "thread": "THREAD",
    "const": "CONST",
}


class MetalLexer:
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)

    def tokenize(self) -> List[Tuple[str, str]]:
        # tokenize the input code and return list of tokens
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        # function that yields tokens one at a time
        pos = 0
        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
            new_pos, token_type, text = token

            if token_type == "IDENTIFIER" and text in KEYWORDS:
                token_type = KEYWORDS[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Tuple[int, str, str]:
        # find the next token starting at the given position
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "MetalLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())
