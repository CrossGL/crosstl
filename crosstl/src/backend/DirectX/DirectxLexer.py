import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("STRUCT", r"struct"),
    ("CBUFFER", r"cbuffer"),
    ("TEXTURE2D", r"Texture2D"),
    ("SAMPLER_STATE", r"SamplerState"),
    ("FVECTOR", r"float[2-4]"),
    ("FLOAT", r"float"),
    ("INT", r"int"),
    ("UINT", r"uint"),
    ("BOOL", r"bool"),
    ("MATRIX", r"float[2-4]x[2-4]"),
    ("VOID", r"void"),
    ("RETURN", r"return"),
    ("IF", r"if"),
    ("ELSE", r"else"),
    ("FOR", r"for"),
    ("REGISTER", r"register"),
    ("SEMANTIC", r": [A-Z_][A-Z0-9_]*"),
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
    "register": "REGISTER",
}


class HLSLLexer:
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
