import re

TOKENS = [
    ("COMMENT_SINGLE", r"//.*"),
    ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
    ("SHADER", r"\bshader\b"),
    ("VOID", r"\bvoid\b"),
    ("STRUCT", r"\bstruct\b"),
    ("CBUFFER", r"\bcbuffer\b"),
    ("AT", r"\@"),
    ("SAMPLER2D", r"\bsampler2D\b"),
    ("SAMPLERCUBE", r"\bsamplerCube\b"),
    ("VECTOR", r"\bvec[2-4]\b"),
    ("MATRIX", r"\bmat[2-4]\b"),
    ("BOOL", r"\bbool\b"),
    ("VERTEX", r"\bvertex\b"),
    ("FRAGMENT", r"\bfragment\b"),
    ("FLOAT_NUMBER", r"\d*\.\d+|\d+\.\d*"),
    ("FLOAT", r"\bfloat\b"),
    ("INT", r"\bint\b"),
    ("UINT", r"\buint\b"),
    ("DOUBLE", r"\bdouble\b"),
    ("SAMPLER", r"\bsampler\b"),
    ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z_0-9]*"),
    ("ASSIGN_SHIFT_RIGHT", r">>="),
    ("ASSIGN_SHIFT_LEFT", r"<<="),
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
    ("WHITESPACE", r"\s+"),
    ("IF", r"\bif\b"),
    ("ELSE", r"\belse\b"),
    ("FOR", r"\bfor\b"),
    ("RETURN", r"\breturn\b"),
    ("BITWISE_SHIFT_LEFT", r"<<"),
    ("BITWISE_SHIFT_RIGHT", r">>"),
    ("LESS_EQUAL", r"<="),
    ("GREATER_EQUAL", r">="),
    ("GREATER_THAN", r">"),
    ("LESS_THAN", r"<"),
    ("INCREMENT", r"\+\+"),
    ("DECREMENT", r"--"),
    ("EQUAL", r"=="),
    ("NOT_EQUAL", r"!="),
    ("ASSIGN_AND", r"&="),
    ("ASSIGN_OR", r"\|="),
    ("ASSIGN_XOR", r"\^="),
    ("LOGICAL_AND", r"&&"),
    ("LOGICAL_OR", r"\|\|"),
    ("NOT", r"!"),
    ("ASSIGN_MOD", r"%="),
    ("MOD", r"%"),
    ("INCREMENT", r"\+\+"),
    ("DECREMENT", r"\-\-"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("DOT", r"\."),
    ("EQUALS", r"="),
    ("QUESTION", r"\?"),
    ("COLON", r":"),
    ("CONST", r"\bconst\b"),
    ("BITWISE_AND", r"&"),
    ("BITWISE_OR", r"\|"),
    ("BITWISE_XOR", r"\^"),
    ("BITWISE_NOT", r"~"),
    ("shift_left", r"<<"),
    ("shift_right", r">>"),
]

KEYWORDS = {
    "shader": "SHADER",
    "void": "VOID",
    "vertex": "VERTEX",
    "fragment": "FRAGMENT",
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "return": "RETURN",
    "const": "CONST",
}


class Lexer:
    """A simple lexer for the shader language

    This lexer tokenizes the input code into a list of tokens.

    Attributes:
        code (str): The input code to tokenize
        tokens (list): A list of tokens generated from the input code

    """

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
