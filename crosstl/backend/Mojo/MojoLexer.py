import re
from typing import Iterator, Tuple, List

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"#.*"),
        ("COMMENT_MULTI", r'"""[\s\S]*?"""'),
        ("BITWISE_NOT", r"~"),
        ("STRUCT", r"\bstruct\b"),
        ("LET", r"\blet\b"),
        ("VAR", r"\bvar\b"),
        ("FN", r"\bfn\b"),
        ("RETURN", r"\breturn\b"),
        ("IF", r"\bif\b"),
        ("ELIF", r"\belif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("IMPORT", r"\bimport\b"),
        ("AS", r"\bas\b"),
        ("IN", r"\bin\b"),
        ("PASS", r"\bpass\b"),
        ("DEF", r"\bdef\b"),
        ("CLASS", r"\bclass\b"),
        ("CONSTANT", r"\bconstant\b"),
        ("INT", r"\bInt\b"),
        ("FLOAT", r"\bFloat\b"),
        ("BOOL", r"\bBool\b"),
        ("STRING", r"\bString\b"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        (
            "NUMBER",
            r"0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+(\.\d+)?([eE][+-]?\d+)?",
        ),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        ("SEMICOLON", r";"),
        ("STRING_LITERAL", r'"[^"]*"'),
        ("COMMA", r","),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
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
        ("ASSIGN_MOD", r"%="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("DOT", r"\."),
        ("EQUALS", r"="),
        ("WHITESPACE", r"\s+"),
        ("MOD", r"%"),
        ("AT", r"@"),
        ("ATTRIBUTE", r"\[\[[^\]]*\]\]"),
    ]
)


# Define keywords specific to mojo
KEYWORDS = {
    "struct": "STRUCT",
    "let": "LET",
    "var": "VAR",
    "fn": "FN",
    "return": "RETURN",
    "if": "IF",
    "elif": "ELIF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
    "continue": "CONTINUE",
    "import": "IMPORT",
    "as": "AS",
    "in": "IN",
    "pass": "PASS",
    "def": "DEF",
    "class": "CLASS",
    "constant": "CONSTANT",
    "Int": "INT",
    "Float": "FLOAT",
    "Bool": "BOOL",
    "String": "STRING",
}


class MojoLexer:
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
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "MojoLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())
