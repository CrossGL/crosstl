import re

# Token patterns
TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("ELSE_IF", r"\belse\s+if\b"),
    ("VERSION", r"#version"),
    ("NUMBER", r"\d+(\.\d+)?"),
    ("CORE", r"\bcore\b"),
    ("SHADER", r"\bshader\b"),
    ("INPUT", r"\binput\b"),
    ("OUTPUT", r"\boutput\b"),
    ("VOID", r"\bvoid\b"),
    ("MAIN", r"\bmain\b"),
    ("UNIFORM", r"\buniform\b"),
    ("VECTOR", r"\bvec[2-4]\b"),
    ("MATRIX", r"\bmat[2-4]\b"),
    ("BOOL", r"\bbool\b"),
    ("FLOAT", r"\bfloat\b"),
    ("INT", r"\bint\b"),
    ("SAMPLER2D", r"\bsampler2D\b"),
    ("PRE_INCREMENT", r"\+\+(?=\w)"),
    ("PRE_DECREMENT", r"--(?=\w)"),
    ("POST_INCREMENT", r"(?<=\w)\+\+"),
    ("POST_DECREMENT", r"(?<=\w)--"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SEMICOLON", r";"),
    ("COMMA", r","),
    ("ASSIGN_ADD", r"\+="),
    ("ASSIGN_SUB", r"-="),
    ("ASSIGN_MUL", r"\*="),
    ("ASSIGN_DIV", r"/="),
    ("EQUAL", r"=="),
    ("NOT_EQUAL", r"!="),
    ("WHITESPACE", r"\s+"),
    ("IF", r"\bif\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("RETURN", r"\breturn\b"),
    ("LESS_EQUAL", r"<="),
    ("GREATER_EQUAL", r">="),
    ("LESS_THAN", r"<"),
    ("GREATER_THAN", r">"),
    ("NOT_EQUAL", r"!="),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("NOT", r"!"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("DOT", r"\."),
    ("EQUALS", r"="),
    ("QUESTION", r"\?"),
    ("COLON", r":"),
    ("LAYOUT", r"\blayout\b"),
    ("IN", r"\bin\b"),
    ("OUT", r"\bout\b"),
]

KEYWORDS = {
    "void": "VOID",
    "main": "MAIN",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "else if": "ELSE_IF",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "return": "RETURN",
    "layout": "LAYOUT",
    "in": "IN",
    "out": "OUT",
}


class GLSLLexer:
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
                    if token_type == "VERSION":
                        self.tokens.append((token_type, text))
                    elif token_type == "VERSION_NUMBER":
                        self.tokens.append((token_type, text))
                    elif token_type == "CORE":
                        self.tokens.append((token_type, text))
                    elif token_type != "WHITESPACE":  # Ignore whitespace tokens
                        token = (token_type, text)
                        self.tokens.append(token)
                    pos = match.end(0)
                    break
            if not match:
                unmatched_char = self.code[pos]
                highlighted_code = (
                    self.code[:pos] + "[" + self.code[pos] + "]" + self.code[pos + 1 :]
                )
                raise SyntaxError(
                    f"Illegal character '{unmatched_char}' at position {pos}\n{highlighted_code}"
                )

        self.tokens.append(("EOF", None))
