import re
from collections import OrderedDict


TOKENS = OrderedDict(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("SHADER", r"\bshader\b"),
        ("BITWISE_NOT", r"~"),
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
        ("UNSIGNED_INT", r"\bunsigned int\b"),
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
        ("DOUBLE", r"\bdouble\b"),
    ]
)


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
    """A simple lexer for the shader language with optimizations

    This lexer tokenizes the input code into a list of tokens.
    Includes optimizations:
    - Token caching for frequently used tokens
    - Combined regex patterns for similar tokens
    - Precompiled regex patterns

    Attributes:
        code (str): The input code to tokenize
        tokens (list): A list of tokens generated from the input code
        token_cache (dict): Cache for frequently used tokens
    """

    def __init__(self, code):
        self.code = code
        self.tokens = []
        self.token_cache = {}
        self.regex_cache = self._compile_patterns()
        self.tokenize()

    def _compile_patterns(self):
        """Compile a single regex with named groups for all tokens."""
        combined_pattern = "|".join(
            f"(?P<{name}>{pattern})" for name, pattern in TOKENS.items()
        )
        return re.compile(combined_pattern)

    def _get_cached_token(self, text, token_type):
        """Return a cached token tuple if possible."""
        cache_key = (text, token_type)
        if cache_key not in self.token_cache:
            self.token_cache[cache_key] = (token_type, text)
        return self.token_cache[cache_key]

    def tokenize(self):
        pos = 0
        length = len(self.code)
        while pos < length:
            match = self.regex_cache.match(self.code, pos)
            if match:
                token_type = match.lastgroup
                text = match.group(token_type)
                if token_type == "IDENTIFIER" and text in KEYWORDS:
                    token_type = KEYWORDS[text]
                if token_type != "WHITESPACE":
                    token = self._get_cached_token(text, token_type)
                    self.tokens.append(token)
                pos = match.end(0)
            else:
                bad_char = self.code[pos]
                highlighted = (
                    self.code[:pos] + "[" + bad_char + "]" + self.code[pos + 1 :]
                )
                raise SyntaxError(
                    f"Illegal character '{bad_char}' at position {pos}\n{highlighted}"
                )
        self.tokens.append(self._get_cached_token(None, "EOF"))
