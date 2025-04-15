import re
from typing import Iterator, Tuple, List
from enum import Enum, auto

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

# use tuple for immutable token types that won't change
TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("BITWISE_NOT", r"~"),
        ("INCLUDE", r"\#include\b"),
        ("STRUCT", r"\bstruct\b"),
        ("CBUFFER", r"\bcbuffer\b"),
        ("TEXTURE2D", r"\bTexture2D\b"),
        ("SAMPLER_STATE", r"\bSamplerState\b"),
        ("FVECTOR", r"\bfloat[2-4]\b"),
        ("FLOAT", r"\bfloat\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("INT", r"\bint\b"),
        ("UINT", r"\buint\b"),
        ("BOOL", r"\bbool\b"),
        ("MATRIX", r"\bfloat[2-4]x[2-4]\b"),
        ("VOID", r"\bvoid\b"),
        ("RETURN", r"\breturn\b"),
        ("IF", r"\bif\b"),
        ("ELSE_IF", r"\belse\sif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("DO", r"\bdo\b"),
        ("REGISTER", r"\bregister\b"),
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
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
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
        ("ASSIGN_XOR", r"\^="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_AND", r"\&="),
        ("BITWISE_XOR", r"\^"),
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("BITWISE_OR", r"\|"),
        ("DOT", r"\."),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("EQUALS", r"="),
        ("WHITESPACE", r"\s+"),
        ("STRING", r"\"[^\"]*\""),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("MOD", r"%"),
        ("HALF", r"\bhalf\b"),
        ("BITWISE_AND", r"&"),
        ("PRAGMA", r"#\s*\bpragma\b"),
        ("AMPERSAND", r"&"),  # Alias for BITWISE_AND
        ("PIPE", r"\|"),  # Alias for BITWISE_OR
        ("CARET", r"\^"),  # Alias for BITWISE_XOR
        ("TILDE", r"~"),  # Alias for BITWISE_NOT
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "cbuffer": "CBUFFER",
    "Texture2D": "TEXTURE2D",
    "SamplerState": "SAMPLER_STATE",
    "float": "FLOAT",
    "float2": "FVECTOR",
    "float3": "FVECTOR",
    "float4": "FVECTOR",
    "double": "DOUBLE",
    "half": "HALF",
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
    "register": "REGISTER",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
}


class TokenType(Enum):
    COMMENT_SINGLE = auto()
    COMMENT_MULTI = auto()
    BITWISE_NOT = auto()
    INCLUDE = auto()
    STRUCT = auto()
    CBUFFER = auto()
    TEXTURE2D = auto()
    SAMPLER_STATE = auto()
    FVECTOR = auto()
    FLOAT = auto()
    DOUBLE = auto()
    INT = auto()
    UINT = auto()
    BOOL = auto()
    MATRIX = auto()
    VOID = auto()
    RETURN = auto()
    IF = auto()
    ELSE_IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    REGISTER = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    COLON = auto()
    QUESTION = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    PLUS_EQUALS = auto()
    MINUS_EQUALS = auto()
    MULTIPLY_EQUALS = auto()
    DIVIDE_EQUALS = auto()
    ASSIGN_XOR = auto()
    ASSIGN_OR = auto()
    ASSIGN_AND = auto()
    BITWISE_XOR = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    BITWISE_OR = auto()
    DOT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    PLUS = auto()
    MINUS = auto()
    EQUALS = auto()
    WHITESPACE = auto()
    STRING = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    BREAK = auto()
    MOD = auto()
    BITWISE_AND = auto()
    PRAGMA = auto()
    HALF = auto()
    # Aliases for bitwise operations
    AMPERSAND = auto()
    PIPE = auto()
    CARET = auto()
    TILDE = auto()


class Token:
    def __init__(self, token_type: TokenType, text: str):
        self.token_type = token_type
        self.text = text


class HLSLLexer:
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.reserved_keywords = {
            **KEYWORDS,
            "switch": TokenType.SWITCH,
            "case": TokenType.CASE,
            "default": TokenType.DEFAULT,
            "half": TokenType.FLOAT,
            "double": TokenType.DOUBLE,
        }

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

            # Skip comments and whitespace
            if token_type == "IDENTIFIER" and text in self.reserved_keywords:
                token_type = self.reserved_keywords[text]

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
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "HLSLLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())


class Lexer:
    """Compatibility wrapper around HLSLLexer"""

    def __init__(self, input_str):
        self.lexer = HLSLLexer(input_str)
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
