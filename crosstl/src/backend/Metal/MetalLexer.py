import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("PREPROCESSOR", r"#\w+"),
    ("STRUCT", r"\bstruct\b"),
    ("CONSTANT", r"\bconstant\b"),
    ("TEXTURE2D", r"\btexture2d\b"),
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
    ("IF", r"\bif\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("KERNEL", r"\bkernel\b"),
    ("VERTEX", r"\bvertex\b"),
    ("FRAGMENT", r"\bfragment\b"),
    ("USING", r"\busing\b"),
    ("NAMESPACE", r"\bnamespace\b"),
    ("METAL", r"\bmetal\b"),
    ("DEVICE", r"\bdevice\b"),
    ("THREADGROUP", r"\bthreadgroup\b"),
    ("THREAD", r"\bthread\b"),
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
