import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("SHADER", r"shader"),
    ("INPUT", r"input"),
    ("OUTPUT", r"output"),
    ("VOID", r"void"),
    ("MAIN", r"main"),
    ("UNIFORM", r"uniform"),
    ("VECTOR", r"vec[2-4]"),
    ("MATRIX", r"mat[2-4]"),
    ("BOOL", r"bool"),
    ("VERTEX", r"vertex"),
    ("FRAGMENT", r"fragment"),
    # ("INCREMENT_DECREMENT", r"\b[a-zA-Z_][a-zA-Z_0-9]*\s*(\+\+|--)\b"),
    ("FLOAT", r"float"),
    ("INT", r"int"),
    ("SAMPLER2D", r"sampler2D"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
    ("NUMBER", r"\d+(\.\d+)?"),
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
    ("EQUALS", r"="),
    ("WHITESPACE", r"\s+"),
    ("IF", r"if"),
    ("ELSE", r"else"),
    ("FOR", r"for"),
    ("RETURN", r"return"),
    ("LESS_THAN", r"<"),
    ("GREATER_THAN", r">"),
    ("INCREMENT", r"\+\+"),
    ("DECREMENT", r"--"),
    ("LESS_EQUAL", r"<="),
    ("GREATER_EQUAL", r">="),
    ("EQUAL", r"=="),
    ("NOT_EQUAL", r"!="),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("NOT", r"!"),
    ("INCREMENT", r"\+\+"),
    ("DECREMENT", r"\-\-"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("DOT", r"\."),
    ("QUESTION", r"\?"),
    ("COLON", r":"),
]

KEYWORDS = {
    "shader": "SHADER",
    "input": "INPUT",
    "output": "OUTPUT",
    "void": "VOID",
    "main": "MAIN",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "return": "RETURN",
}


class Lexer:
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
                    if token_type != "WHITESPACE":  # Ignore whitespace tokens

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

        self.tokens.append(("EOF", None))  # End of file token
