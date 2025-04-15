import re
from typing import Iterator, Tuple, List

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("BITWISE_NOT", r"~"),
        ("WHITESPACE", r"\s+"),
        ("SEMANTIC", r":\w+"),
        ("PRE_INCREMENT", r"\+\+(?=\w)"),
        ("PRE_DECREMENT", r"--(?=\w)"),
        ("POST_INCREMENT", r"(?<=\w)\+\+"),
        ("POST_DECREMENT", r"(?<=\w)--"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d*)?u?|\.\d+u?"),
        ("SEMICOLON", r";"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COMMA", r","),
        ("DOT", r"\."),
        ("EQUAL", r"=="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("ASSIGN_MOD", r"%="),
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("BITWISE_SHIFT_LEFT", r"<<"),
        ("BITWISE_SHIFT_RIGHT", r">>"),
        ("EQUALS", r"="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("NOT_EQUAL", r"!="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("BINARY_AND", r"&"),
        ("BINARY_OR", r"\|"),
        ("BINARY_XOR", r"\^"),
        ("BINARY_NOT", r"~"),
        ("QUESTION", r"\?"),
        ("COLON", r":"),
        ("MOD", r"%"),
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "layout": "LAYOUT",
    "uniform": "UNIFORM",
    "sampler2D": "SAMPLER2D",
    "samplerCube": "SAMPLERCUBE",
    "vec2": "VEC2",
    "vec3": "VEC3",
    "vec4": "VEC4",
    "ivec2": "IVEC2",
    "ivec3": "IVEC3",
    "ivec4": "IVEC4",
    "uvec2": "UVEC2",
    "uvec3": "UVEC3",
    "uvec4": "UVEC4",
    "bvec2": "BVEC2",
    "bvec3": "BVEC3",
    "bvec4": "BVEC4",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "float": "FLOAT",
    "double": "DOUBLE",
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
    "discard": "DISCARD",
    "in": "IN",
    "out": "OUT",
    "inout": "INOUT",
    "attribute": "ATTRIBUTE",
    "varying": "VARYING",
    "const": "CONST",
    "precision": "PRECISION",
    "highp": "HIGHP",
    "mediump": "MEDIUMP",
    "lowp": "LOWP",
    "subpassInput": "SUBPASSINPUT",
    "subpassInputMS": "SUBPASSINPUTMS",
    "sampler2DArray": "SAMPLER2DARRAY",
    "sampler2DMS": "SAMPLER2DMS",
    "sampler2DMSArray": "SAMPLER2DMSARRAY",
    "sampler3D": "SAMPLER3D",
    "samplerCubeArray": "SAMPLERCUBEARRAY",
    "image2D": "IMAGE2D",
    "image3D": "IMAGE3D",
    "imageCube": "IMAGECUBE",
    "imageBuffer": "IMAGEBUFFER",
    "image2DArray": "IMAGE2DARRAY",
    "imageCubeArray": "IMAGECUBEARRAY",
    "image1D": "IMAGE1D",
    "image1DArray": "IMAGE1DARRAY",
    "image2DMS": "IMAGE2DMS",
    "image2DMSArray": "IMAGE2DMSARRAY",
    "atomic_uint": "ATOMICUINT",
    "mat2": "MAT2",
    "mat3": "MAT3",
    "mat4": "MAT4",
}

VALID_DATA_TYPES = [
    "int",
    "float",
    "double",
    "vec2",
    "vec3",
    "vec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "uvec2",
    "uvec3",
    "uvec4",
    "bvec2",
    "bvec3",
    "bvec4",
    "mat2",
    "mat3",
    "mat4",
    "uint",
    "bool",
    "void",
]


class VulkanLexer:
    def __init__(self, code: str):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)

    def tokenize(self) -> List[Tuple[str, str]]:
        # tokenize the input code and return list of tokens
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        # function that yields tokens one at a time
        pos = 0
        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
            new_pos, token_type, text = token

            if token_type == "IDENTIFIER" and text in KEYWORDS:
                token_type = KEYWORDS[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Tuple[int, str, str]:
        # find the next token starting at the given position
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "VulkanLexer":
        # create a lexer instance from a file, reading in chunks
        with open(filepath, "r") as f:
            return cls(f.read())
