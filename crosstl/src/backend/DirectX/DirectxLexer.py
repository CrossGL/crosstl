import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("STRUCT", r"\bstruct\b"),
    ("CBUFFER", r"\bcbuffer\b"),
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
    ("ELSE_IF", r"\belse\sif\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("WHILE", r"\b\while\b"),
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
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("DOT", r"\."),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("EQUALS", r"="),
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
    "while": "WHILE",
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
