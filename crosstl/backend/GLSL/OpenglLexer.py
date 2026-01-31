import re
from typing import Iterator, Tuple, List, Optional, Dict

from .preprocessor import GLSLPreprocessor

# Tokens to skip entirely
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

HEX_NUMBER = r"0[xX][0-9a-fA-F]+"
DECIMAL_FLOAT = r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?"
DECIMAL_EXP = r"\d+[eE][+-]?\d+"
DECIMAL_INT = r"\d+"
NUMBER_PATTERN = rf"(?:{HEX_NUMBER}|{DECIMAL_FLOAT}|{DECIMAL_EXP}|{DECIMAL_INT})(?:[uU])?"

# Order matters: longer tokens first
TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//[^\n]*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("NEWLINE", r"\n+"),
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("MOD_EQUALS", r"%="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("LOGICAL_NOT", r"!"),
        ("BITWISE_NOT", r"~"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_AND", r"&"),
        ("EQUALS", r"="),
        ("HASH", r"#"),
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
        ("NUMBER", NUMBER_PATTERN),
        ("DOT", r"\."),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MOD", r"%"),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("STRING", r'"([^"\\]|\\.)*"'),
        ("CHAR_LITERAL", r"'(?:[^'\\]|\\.)'"),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("WHITESPACE", r"[ \t\r\f\v]+"),
    ]
)

KEYWORDS = {
    "struct": "STRUCT",
    "uniform": "UNIFORM",
    "const": "CONST",
    "flat": "FLAT",
    "smooth": "SMOOTH",
    "noperspective": "NOPERSPECTIVE",
    "centroid": "CENTROID",
    "sample": "SAMPLE",
    "patch": "PATCH",
    "invariant": "INVARIANT",
    "precise": "PRECISE",
    "in": "IN",
    "out": "OUT",
    "inout": "INOUT",
    "layout": "LAYOUT",
    "attribute": "ATTRIBUTE",
    "varying": "VARYING",
    "buffer": "BUFFER",
    "shared": "SHARED",
    "readonly": "READONLY",
    "writeonly": "WRITEONLY",
    "coherent": "COHERENT",
    "volatile": "VOLATILE",
    "restrict": "RESTRICT",
    "precision": "PRECISION",
    "lowp": "LOWP",
    "mediump": "MEDIUMP",
    "highp": "HIGHP",
    "void": "VOID",
    "bool": "BOOL",
    "int": "INT",
    "uint": "UINT",
    "float": "FLOAT",
    "double": "DOUBLE",
    "vec2": "VECTOR",
    "vec3": "VECTOR",
    "vec4": "VECTOR",
    "ivec2": "VECTOR",
    "ivec3": "VECTOR",
    "ivec4": "VECTOR",
    "uvec2": "VECTOR",
    "uvec3": "VECTOR",
    "uvec4": "VECTOR",
    "bvec2": "VECTOR",
    "bvec3": "VECTOR",
    "bvec4": "VECTOR",
    "dvec2": "VECTOR",
    "dvec3": "VECTOR",
    "dvec4": "VECTOR",
    "mat2": "MATRIX",
    "mat3": "MATRIX",
    "mat4": "MATRIX",
    "mat2x2": "MATRIX",
    "mat2x3": "MATRIX",
    "mat2x4": "MATRIX",
    "mat3x2": "MATRIX",
    "mat3x3": "MATRIX",
    "mat3x4": "MATRIX",
    "mat4x2": "MATRIX",
    "mat4x3": "MATRIX",
    "mat4x4": "MATRIX",
    "sampler2D": "SAMPLER2D",
    "sampler3D": "SAMPLER3D",
    "samplerCube": "SAMPLERCUBE",
    "sampler1D": "SAMPLER1D",
    "sampler1DArray": "SAMPLER1DARRAY",
    "sampler1DShadow": "SAMPLER1DSHADOW",
    "sampler1DArrayShadow": "SAMPLER1DARRAYSHADOW",
    "sampler2DArray": "SAMPLER2DARRAY",
    "sampler2DArrayShadow": "SAMPLER2DARRAYSHADOW",
    "samplerCubeArray": "SAMPLERCUBEARRAY",
    "samplerCubeArrayShadow": "SAMPLERCUBEARRAYSHADOW",
    "sampler2DShadow": "SAMPLER2DSHADOW",
    "sampler2DRect": "SAMPLER2DRECT",
    "sampler2DRectShadow": "SAMPLER2DRECTSHADOW",
    "samplerBuffer": "SAMPLERBUFFER",
    "samplerCubeShadow": "SAMPLERCUBESHADOW",
    "sampler2DMS": "SAMPLER2DMS",
    "sampler2DMSArray": "SAMPLER2DMSARRAY",
    "isampler1D": "ISAMPLER1D",
    "isampler2D": "ISAMPLER2D",
    "isampler3D": "ISAMPLER3D",
    "isamplerCube": "ISAMPLERCUBE",
    "isampler1DArray": "ISAMPLER1DARRAY",
    "isampler2DArray": "ISAMPLER2DARRAY",
    "isamplerCubeArray": "ISAMPLERCUBEARRAY",
    "isampler2DRect": "ISAMPLER2DRECT",
    "isamplerBuffer": "ISAMPLERBUFFER",
    "isampler2DMS": "ISAMPLER2DMS",
    "isampler2DMSArray": "ISAMPLER2DMSARRAY",
    "usampler1D": "USAMPLER1D",
    "usampler2D": "USAMPLER2D",
    "usampler3D": "USAMPLER3D",
    "usamplerCube": "USAMPLERCUBE",
    "usampler1DArray": "USAMPLER1DARRAY",
    "usampler2DArray": "USAMPLER2DARRAY",
    "usamplerCubeArray": "USAMPLERCUBEARRAY",
    "usampler2DRect": "USAMPLER2DRECT",
    "usamplerBuffer": "USAMPLERBUFFER",
    "usampler2DMS": "USAMPLER2DMS",
    "usampler2DMSArray": "USAMPLER2DMSARRAY",
    "image1D": "IMAGE1D",
    "image2D": "IMAGE2D",
    "image3D": "IMAGE3D",
    "imageCube": "IMAGECUBE",
    "image1DArray": "IMAGE1DARRAY",
    "image2DArray": "IMAGE2DARRAY",
    "imageCubeArray": "IMAGECUBEARRAY",
    "image2DRect": "IMAGE2DRECT",
    "imageBuffer": "IMAGEBUFFER",
    "image2DMS": "IMAGE2DMS",
    "image2DMSArray": "IMAGE2DMSARRAY",
    "iimage1D": "IIMAGE1D",
    "iimage2D": "IIMAGE2D",
    "iimage3D": "IIMAGE3D",
    "iimageCube": "IIMAGECUBE",
    "iimage1DArray": "IIMAGE1DARRAY",
    "iimage2DArray": "IIMAGE2DARRAY",
    "iimageCubeArray": "IIMAGECUBEARRAY",
    "iimage2DRect": "IIMAGE2DRECT",
    "iimageBuffer": "IIMAGEBUFFER",
    "iimage2DMS": "IIMAGE2DMS",
    "iimage2DMSArray": "IIMAGE2DMSARRAY",
    "uimage1D": "UIMAGE1D",
    "uimage2D": "UIMAGE2D",
    "uimage3D": "UIMAGE3D",
    "uimageCube": "UIMAGECUBE",
    "uimage1DArray": "UIMAGE1DARRAY",
    "uimage2DArray": "UIMAGE2DARRAY",
    "uimageCubeArray": "UIMAGECUBEARRAY",
    "uimage2DRect": "UIMAGE2DRECT",
    "uimageBuffer": "UIMAGEBUFFER",
    "uimage2DMS": "UIMAGE2DMS",
    "uimage2DMSArray": "UIMAGE2DMSARRAY",
    "atomic_uint": "ATOMIC_UINT",
    "subroutine": "SUBROUTINE",
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
    "return": "RETURN",
    "discard": "DISCARD",
    "true": "TRUE",
    "false": "FALSE",
}


class GLSLLexer:
    def __init__(
        self,
        code: str,
        preprocess: bool = True,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict_preprocessor: bool = True,
        max_expansion_depth: int = 64,
        file_path: Optional[str] = None,
    ):
        if preprocess:
            preprocessor = GLSLPreprocessor(
                include_paths=include_paths,
                defines=defines,
                strict=strict_preprocessor,
                max_expansion_depth=max_expansion_depth,
            )
            code = preprocessor.preprocess(code, file_path=file_path)
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)

    def tokenize(self) -> List[Tuple[str, str]]:
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        pos = 0
        while pos < self._length:
            if self.code.startswith("/*", pos):
                if self.code.find("*/", pos + 2) == -1:
                    line_num = self.code[:pos].count("\n") + 1
                    col_num = pos - self.code.rfind("\n", 0, pos)
                    raise SyntaxError(
                        f"Unterminated block comment at line {line_num}, column {col_num}"
                    )
            token = self._next_token(pos)
            if token is None:
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)
                context = self.code[max(0, pos - 20) : min(self._length, pos + 20)]
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at line {line_num}, column {col_num}\n"
                    f"Context: ...{context}..."
                )

            new_pos, token_type, text = token

            if token_type == "IDENTIFIER" and text in KEYWORDS:
                token_type = KEYWORDS[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Optional[Tuple[int, str, str]]:
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(
        cls,
        filepath: str,
        preprocess: bool = True,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict_preprocessor: bool = True,
        max_expansion_depth: int = 64,
    ) -> "GLSLLexer":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls(
                f.read(),
                preprocess=preprocess,
                include_paths=include_paths,
                defines=defines,
                strict_preprocessor=strict_preprocessor,
                max_expansion_depth=max_expansion_depth,
                file_path=filepath,
            )
