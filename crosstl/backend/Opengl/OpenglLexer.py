import re

TOKENS = [
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