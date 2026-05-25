"""Lexer for importing Vulkan SPIR-V source into CrossGL Translator."""

import re
from typing import Iterator, Tuple, List

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI", "PREPROCESSOR"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("PREPROCESSOR", r"#[^\r\n]*"),
        ("BITWISE_NOT", r"~"),
        ("WHITESPACE", r"\s+"),
        ("SEMANTIC", r":\w+"),
        ("PRE_INCREMENT", r"\+\+(?=\w)"),
        ("PRE_DECREMENT", r"--(?=\w)"),
        ("POST_INCREMENT", r"(?<=[\w\]])\+\+"),
        ("POST_DECREMENT", r"(?<=[\w\]])--"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NUMBER", r"\d+(\.\d*)?u?|\.\d+u?"),
        ("SEMICOLON", r";"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
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
        ("NOT", r"!"),
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
    "buffer": "BUFFER",
    "push_constant": "PUSH_CONSTANT",
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
    "sampler": "SAMPLER",
    "sampler1D": "SAMPLER1D",
    "sampler1DArray": "SAMPLER1DARRAY",
    "sampler1DArrayShadow": "SAMPLER1DARRAYSHADOW",
    "sampler1DShadow": "SAMPLER1DSHADOW",
    "isampler1D": "ISAMPLER1D",
    "isampler1DArray": "ISAMPLER1DARRAY",
    "isampler2D": "ISAMPLER2D",
    "isampler2DArray": "ISAMPLER2DARRAY",
    "isampler2DMS": "ISAMPLER2DMS",
    "isampler2DMSArray": "ISAMPLER2DMSARRAY",
    "isampler3D": "ISAMPLER3D",
    "isamplerBuffer": "ISAMPLERBUFFER",
    "isamplerCube": "ISAMPLERCUBE",
    "isamplerCubeArray": "ISAMPLERCUBEARRAY",
    "sampler2DArray": "SAMPLER2DARRAY",
    "sampler2DArrayShadow": "SAMPLER2DARRAYSHADOW",
    "sampler2DShadow": "SAMPLER2DSHADOW",
    "sampler2DMS": "SAMPLER2DMS",
    "sampler2DMSArray": "SAMPLER2DMSARRAY",
    "samplerBuffer": "SAMPLERBUFFER",
    "sampler3D": "SAMPLER3D",
    "samplerCubeArray": "SAMPLERCUBEARRAY",
    "samplerCubeArrayShadow": "SAMPLERCUBEARRAYSHADOW",
    "samplerCubeShadow": "SAMPLERCUBESHADOW",
    "usampler1D": "USAMPLER1D",
    "usampler1DArray": "USAMPLER1DARRAY",
    "usampler2D": "USAMPLER2D",
    "usampler2DArray": "USAMPLER2DARRAY",
    "usampler2DMS": "USAMPLER2DMS",
    "usampler2DMSArray": "USAMPLER2DMSARRAY",
    "usampler3D": "USAMPLER3D",
    "usamplerBuffer": "USAMPLERBUFFER",
    "usamplerCube": "USAMPLERCUBE",
    "usamplerCubeArray": "USAMPLERCUBEARRAY",
    "image2D": "IMAGE2D",
    "image3D": "IMAGE3D",
    "imageCube": "IMAGECUBE",
    "imageBuffer": "IMAGEBUFFER",
    "image2DArray": "IMAGE2DARRAY",
    "imageCubeArray": "IMAGECUBEARRAY",
    "image1D": "IMAGE1D",
    "image1DArray": "IMAGE1DARRAY",
    "iimage1D": "IIMAGE1D",
    "iimage1DArray": "IIMAGE1DARRAY",
    "iimage2D": "IIMAGE2D",
    "iimage2DArray": "IIMAGE2DARRAY",
    "iimage2DMS": "IIMAGE2DMS",
    "iimage2DMSArray": "IIMAGE2DMSARRAY",
    "iimage3D": "IIMAGE3D",
    "iimageBuffer": "IIMAGEBUFFER",
    "iimageCube": "IIMAGECUBE",
    "iimageCubeArray": "IIMAGECUBEARRAY",
    "uimage1D": "UIMAGE1D",
    "uimage1DArray": "UIMAGE1DARRAY",
    "uimage2D": "UIMAGE2D",
    "uimage2DArray": "UIMAGE2DARRAY",
    "uimage2DMS": "UIMAGE2DMS",
    "uimage2DMSArray": "UIMAGE2DMSARRAY",
    "uimage3D": "UIMAGE3D",
    "uimageBuffer": "UIMAGEBUFFER",
    "uimageCube": "UIMAGECUBE",
    "uimageCubeArray": "UIMAGECUBEARRAY",
    "image2DMS": "IMAGE2DMS",
    "image2DMSArray": "IMAGE2DMSARRAY",
    "atomic_uint": "ATOMICUINT",
    "mat2": "MAT2",
    "mat3": "MAT3",
    "mat4": "MAT4",
}

RESOURCE_DATA_TYPES = [
    "subpassInput",
    "subpassInputMS",
    "sampler",
    "sampler1D",
    "sampler1DArray",
    "sampler1DArrayShadow",
    "sampler1DShadow",
    "isampler1D",
    "isampler1DArray",
    "isampler2D",
    "isampler2DArray",
    "isampler2DMS",
    "isampler2DMSArray",
    "isampler3D",
    "isamplerBuffer",
    "isamplerCube",
    "isamplerCubeArray",
    "sampler2D",
    "samplerCube",
    "sampler2DArray",
    "sampler2DArrayShadow",
    "sampler2DShadow",
    "sampler2DMS",
    "sampler2DMSArray",
    "samplerBuffer",
    "sampler3D",
    "samplerCubeArray",
    "samplerCubeArrayShadow",
    "samplerCubeShadow",
    "usampler1D",
    "usampler1DArray",
    "usampler2D",
    "usampler2DArray",
    "usampler2DMS",
    "usampler2DMSArray",
    "usampler3D",
    "usamplerBuffer",
    "usamplerCube",
    "usamplerCubeArray",
    "image1D",
    "image1DArray",
    "image2D",
    "image2DArray",
    "image2DMS",
    "image2DMSArray",
    "iimage1D",
    "iimage1DArray",
    "iimage2D",
    "iimage2DArray",
    "iimage2DMS",
    "iimage2DMSArray",
    "iimage3D",
    "iimageBuffer",
    "iimageCube",
    "iimageCubeArray",
    "uimage1D",
    "uimage1DArray",
    "uimage2D",
    "uimage2DArray",
    "uimage2DMS",
    "uimage2DMSArray",
    "uimage3D",
    "uimageBuffer",
    "uimageCube",
    "uimageCubeArray",
    "image3D",
    "imageBuffer",
    "imageCube",
    "imageCubeArray",
    "atomic_uint",
]

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
    *RESOURCE_DATA_TYPES,
]


class VulkanLexer:
    """Tokenize Vulkan/SPIR-V style source for the Vulkan backend parser."""

    def __init__(self, code: str):
        """Initialize the lexer with raw Vulkan/SPIR-V style source text."""
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)

    def tokenize(self) -> List[Tuple[str, str]]:
        """Return the full token stream as ``(token_type, text)`` tuples."""
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        """Yield Vulkan/SPIR-V tokens while skipping whitespace and comments."""
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
        """Match the next token at ``pos`` and return its end offset."""
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str, chunk_size: int = 8192) -> "VulkanLexer":
        """Create a lexer instance from a Vulkan/SPIR-V source file."""
        with open(filepath, "r") as f:
            return cls(f.read())
