import re

# Token definitions
TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("STRUCT", r"\bstruct\b"),
    ("CBUFFER", r"\bcbuffer\b"),
    ("SHADER", r"\bshader\b"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
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
    ("ELSE_IF", r"\belse\s+if\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("WHILE", r"\bwhile\b"),
    ("DO", r"\bdo\b"),
    ("SWITCH", r"\bswitch\b"),
    ("CASE", r"\bcase\b"),
    ("DEFAULT", r"\bdefault\b"),
    ("BREAK", r"\bbreak\b"),
    ("CONTINUE", r"\bcontinue\b"),
    ("REGISTER", r"\bregister\b"),
    (
        "SEMANTIC",
        r":\s*[A-Za-z_][A-Za-z0-9_]*",
    ),  # Correctly capturing the entire semantic token
    ("STRING", r'"[^"]*"'),
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
    ("COLON", r":"),  # Separate token for single colon, if needed separately
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
    ("AND", r"&&"),
    ("OR", r"\|\|"),
    ("DOT", r"\."),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("EQUALS", r"="),
    ("WHITESPACE", r"\s+"),
    # Slang-specific tokens
    ("IMPORT", r"\bimport\b"),
    ("EXPORT", r"\bexport\b"),
    ("GENERIC", r"\b__generic\b"),
    ("EXTENSION", r"\bextension\b"),
    ("TYPEDEF", r"\btypedef\b"),
    ("CONSTEXPR", r"\bconstexpr\b"),
    ("STATIC", r"\bstatic\b"),
    ("INLINE", r"\binline\b"),
]

# Keywords map for matching identifiers to token types
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
    "do": "DO",
    "switch": "SWITCH",
    "case": "CASE",
    "default": "DEFAULT",
    "break": "BREAK",
    "continue": "CONTINUE",
    "register": "REGISTER",
    "import": "IMPORT",
    "export": "EXPORT",
    "__generic": "GENERIC",
    "extension": "EXTENSION",
    "typedef": "TYPEDEF",
    "constexpr": "CONSTEXPR",
    "static": "STATIC",
    "inline": "INLINE",
}


class SlangLexer:
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
                        token = (token_type, text.strip())
                        self.tokens.append(token)
                    pos = match.end(0)
                    break
            if not match:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
        self.tokens.append(("EOF", ""))
