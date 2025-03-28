import re
from typing import Iterator, Tuple, List

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("VERSION", r"#version"),
        ("PREPROCESSOR", r"#\w+"),
        ("CONSTANT", r"\bconst\b"),
        ("STRUCT", r"\bstruct\b"),
        ("UNIFORM", r"\buniform\b"),
        ("SAMPLER2D", r"\bsampler2D\b"),
        ("SAMPLERCUBE", r"\bsamplerCube\b"),
        ("BUFFER", r"\bbuffer\b"),
        ("VECTOR", r"\b(vec|ivec|uvec|bvec)[234]\b"),
        ("MATRIX", r"\bmat[234](x[234])?\b"),
        ("FLOAT", r"\bfloat\b"),
        ("INT", r"\bint\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("UINT", r"\buint\b"),
        ("BOOL", r"\bbool\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("DO", r"\bdo\b"),
        ("IN", r"\bin\b"),
        ("OUT", r"\bout\b"),
        ("INOUT", r"\binout\b"),
        ("LAYOUT", r"\blayout\b"),
        ("ATTRIBUTE", r"\battribute\b"),
        ("VARYING", r"\bvarying\b"),
        ("CONST", r"\bconst\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d+)?([eE][+-]?\d+)?"),
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
        ("QUESTION", r"\?"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("ASSIGN_MOD", r"%="),
        ("MOD", r"%"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("DOT", r"\."),
        ("EQUALS", r"="),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_NOT", r"~"),
        ("WHITESPACE", r"\s+"),
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "uniform": "UNIFORM",
    "sampler2D": "SAMPLER2D",
    "samplerCube": "SAMPLERCUBE",
    "float": "FLOAT",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
    "double": "DOUBLE",
    "return": "RETURN",
    "else if": "ELSE_IF",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "do": "DO",
    "in": "IN",
    "out": "OUT",
    "inout": "INOUT",
    "layout": "LAYOUT",
    "attribute": "ATTRIBUTE",
    "varying": "VARYING",
    "const": "CONST",
}


class GLSLLexer:
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
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "GLSLLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())
