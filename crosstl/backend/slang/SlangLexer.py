import re
from typing import Iterator, Tuple, List

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("STRUCT", r"\bstruct\b"),
        ("CBUFFER", r"\bcbuffer\b"),
        ("TYPE_SHADER", r'\[shader\("(vertex|fragment|compute)"\)\]'),
        ("SHADER", r"\bshader\b"),
        ("STRING", r'"(?:\\.|[^"\\])*"'),
        ("TEXTURE2D", r"\bTexture2D\b"),
        ("SAMPLER_STATE", r"\bSamplerState\b"),
        ("FVECTOR", r"\bfloat[2-4]\b"),
        ("FLOAT", r"\bfloat\b"),
        ("INT", r"\bint\b"),
        ("UINT", r"\buint\b"),
        ("BOOL", r"\bbool\b"),
        ("MATRIX", r"\bfloat[2-4]x[2-4]\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("IF", r"\bif\b"),
        ("ELSE_IF", r"\belse\s+if\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("DO", r"\bdo\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("REGISTER", r"\bregister\b"),
        ("STRING", r'"[^"]*"'),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d+)?"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("COMMA", r","),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
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
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("DOT", r"\."),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("EQUALS", r"="),
        ("WHITESPACE", r"\s+"),
        ("IMPORT", r"\bimport\b"),
        ("EXPORT", r"\bexport\b"),
        ("GENERIC", r"\b__generic\b"),
        ("EXTENSION", r"\bextension\b"),
        ("TYPEDEF", r"\btypedef\b"),
        ("CONSTEXPR", r"\bconstexpr\b"),
        ("STATIC", r"\bstatic\b"),
        ("INLINE", r"\binline\b"),
        ("MOD", r"%"),
    ]
)

# Keywords map for matching identifiers to token types
KEYWORDS = {
    "struct": "STRUCT",
    "cbuffer": "CBUFFER",
    "Texture2D": "TEXTURE2D",
    "SamplerState": "SAMPLER_STATE",
    "float": "FLOAT",
    "float2": "FVECTOR",
    "float3": "FVECTOR",
    "float4": "FVECTOR",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
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
    "register": "REGISTER",
    "import": "IMPORT",
    "export": "EXPORT",
    "__generic": "GENERIC",
    "extension": "EXTENSION",
    "typedef": "TYPEDEF",
    "constexpr": "CONSTEXPR",
    "static": "STATIC",
    "inline": "INLINE",
}


class SlangLexer:
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
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "SlangLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())
