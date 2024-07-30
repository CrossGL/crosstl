import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("PREPROCESSOR", r"#\w+"),
    ("STRUCT", r"struct"),
    ("CONSTANT", r"constant"),
    ("TEXTURE2D", r"texture2d"),
    ("SAMPLER", r"sampler"),
    ("VECTOR", r"(float|half|int|uint)[2-4]"),
    ("FLOAT", r"float"),
    ("HALF", r"half"),
    ("INT", r"int"),
    ("UINT", r"uint"),
    ("BOOL", r"bool"),
    ("VOID", r"void"),
    ("RETURN", r"return"),
    ("IF", r"if"),
    ("ELSE", r"else"),
    ("FOR", r"for"),
    ("KERNEL", r"kernel"),
    ("VERTEX", r"vertex"),
    ("FRAGMENT", r"fragment"),
    ("USING", r"using"),
    ("NAMESPACE", r"namespace"),
    ("METAL", r"metal"),
    ("DEVICE", r"device"),
    ("THREADGROUP", r"threadgroup"),
    ("THREAD", r"thread"),
    ("ATTRIBUTE", r"\[\[.*?\]\]"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("NUMBER", r"\d+(\.\d+)?"),
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
    ("EQUALS", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("LESS_THAN", r"<"),
    ("GREATER_THAN", r">"),
    ("LESS_EQUAL", r"<="),
    ("GREATER_EQUAL", r">="),
    ("EQUAL", r"=="),
    ("NOT_EQUAL", r"!="),
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("DOT", r"\."),
    ("WHITESPACE", r"\s+"),
]

KEYWORDS = {
    "struct": "STRUCT",
    "constant": "CONSTANT",
    "texture2d": "TEXTURE2D",
    "sampler": "SAMPLER",
    "float": "FLOAT",
    "half": "HALF",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
    "return": "RETURN",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "kernel": "KERNEL",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "using": "USING",
    "namespace": "NAMESPACE",
    "metal": "METAL",
    "device": "DEVICE",
    "threadgroup": "THREADGROUP",
    "thread": "THREAD",
}


class MetalLexer:
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
