import re

# Define the tokens for Mojo syntax
TOKENS = [
    ("COMMENT_SINGLE", r"#.*"),
    ("COMMENT_MULTI", r'"""[\s\S]*?"""'),
    ("STRUCT", r"\bstruct\b"),
    ("LET", r"\blet\b"),
    ("VAR", r"\bvar\b"),
    ("FN", r"\bfn\b"),
    ("RETURN", r"\breturn\b"),
    ("IF", r"\bif\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("WHILE", r"\bwhile\b"),
    ("IMPORT", r"\bimport\b"),
    ("DEF", r"\bdef\b"),
    ("INT", r"\bInt\b"),
    ("FLOAT", r"\bFloat\b"),
    ("BOOL", r"\bBool\b"),
    ("STRING", r"\bString\b"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NUMBER", r"\d+(\.\d+)?"),
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
    ("WHITESPACE", r"\s+"),
    ("MOD", r"%"),
]

# Define keywords specific to mojo
KEYWORDS = {
    "struct": "STRUCT",
    "let": "LET",
    "var": "VAR",
    "fn": "FN",
    "return": "RETURN",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "while": "WHILE",
    "import": "IMPORT",
    "def": "DEF",
    "Int": "INT",
    "Float": "FLOAT",
    "Bool": "BOOL",
    "String": "STRING",
}


class MojoLexer:
    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.tokenize()

    def tokenize(self):
        pos = 0
        while pos < len(self.code):
            match = None
            for token_type, pattern in TOKENS:
                regex = re.compile(pattern)
                match = regex.match(self.code, pos)
                if match:
                    text = match.group(0)
                    if token_type == "IDENTIFIER" and text in KEYWORDS:
                        token_type = KEYWORDS[text]
                    if token_type not in [
                        "WHITESPACE",
                        "COMMENT_SINGLE",
                        "COMMENT_MULTI",
                    ]:
                        token = (token_type, text)
                        self.tokens.append(token)
                    pos = match.end(0)
                    break
            if not match:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )

        self.tokens.append(("EOF", ""))
