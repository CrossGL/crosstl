"""
DirectX/HLSL Lexer Module.

This module provides lexical analysis (tokenization) for Microsoft HLSL
(High-Level Shading Language) used in DirectX. It converts HLSL source code
into a stream of tokens that can be processed by the HLSL parser.

The lexer supports:
    - All HLSL keywords and types
    - Texture and sampler types
    - Buffer and resource types
    - Shader model-specific features
    - Preprocessor directives

Example:
    >>> from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
    >>> lexer = HLSLLexer(hlsl_code)
    >>> tokens = lexer.tokens
"""

import os
import re
from typing import Iterator, Tuple, List, Optional
from enum import Enum, auto

from .preprocessor import HLSLPreprocessor

#: Set of token types to skip during parsing.
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI"}

#: Token definitions - order matters! More specific patterns should come first.
TOKENS = tuple(
    [
        # Comments (must come first to avoid partial matches)
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        # Preprocessor directives (capture entire line)
        ("PREPROCESSOR", r"#.*"),
        # Keywords - struct and buffer types
        ("ENUM", r"\benum\b"),
        ("TYPEDEF", r"\btypedef\b"),
        ("STRUCT", r"\bstruct\b"),
        ("CBUFFER", r"\bcbuffer\b"),
        ("TBUFFER", r"\btbuffer\b"),
        ("GROUPSHARED", r"\bgroupshared\b"),
        ("STATIC", r"\bstatic\b"),
        ("CONST", r"\bconst\b"),
        ("INLINE", r"\binline\b"),
        ("EXTERN", r"\bextern\b"),
        ("VOLATILE", r"\bvolatile\b"),
        ("PRECISE", r"\bprecise\b"),
        ("ROW_MAJOR", r"\brow_major\b"),
        ("COLUMN_MAJOR", r"\bcolumn_major\b"),
        # Interpolation modifiers
        ("NOINTERPOLATION", r"\bnointerpolation\b"),
        ("LINEAR", r"\blinear\b"),
        ("CENTROID", r"\bcentroid\b"),
        ("SAMPLE", r"\bsample\b"),
        # Texture and sampler types
        ("TEXTURE1D", r"\bTexture1D\b"),
        ("TEXTURE1DARRAY", r"\bTexture1DArray\b"),
        ("TEXTURE2D", r"\bTexture2D\b"),
        ("TEXTURE2DARRAY", r"\bTexture2DArray\b"),
        ("TEXTURE2DMS", r"\bTexture2DMS\b"),
        ("TEXTURE2DMSARRAY", r"\bTexture2DMSArray\b"),
        ("TEXTURE3D", r"\bTexture3D\b"),
        ("TEXTURECUBE", r"\bTextureCube\b"),
        ("TEXTURECUBEARRAY", r"\bTextureCubeArray\b"),
        ("FEEDBACKTEXTURE2D", r"\bFeedbackTexture2D\b"),
        ("FEEDBACKTEXTURE2DARRAY", r"\bFeedbackTexture2DArray\b"),
        ("RWTEXTURE1D", r"\bRWTexture1D\b"),
        ("RWTEXTURE1DARRAY", r"\bRWTexture1DArray\b"),
        ("RWTEXTURE2D", r"\bRWTexture2D\b"),
        ("RWTEXTURE2DARRAY", r"\bRWTexture2DArray\b"),
        ("RWTEXTURE2DMS", r"\bRWTexture2DMS\b"),
        ("RWTEXTURE2DMSARRAY", r"\bRWTexture2DMSArray\b"),
        ("RWTEXTURE3D", r"\bRWTexture3D\b"),
        ("RWTEXTURECUBE", r"\bRWTextureCube\b"),
        ("RWTEXTURECUBEARRAY", r"\bRWTextureCubeArray\b"),
        ("STRUCTUREDBUFFER", r"\bStructuredBuffer\b"),
        ("RWSTRUCTUREDBUFFER", r"\bRWStructuredBuffer\b"),
        ("APPENDSTRUCTUREDBUFFER", r"\bAppendStructuredBuffer\b"),
        ("CONSUMESTRUCTUREDBUFFER", r"\bConsumeStructuredBuffer\b"),
        ("BYTEADDRESSBUFFER", r"\bByteAddressBuffer\b"),
        ("RWBYTEADDRESSBUFFER", r"\bRWByteAddressBuffer\b"),
        ("RAYTRACING_ACCELERATION_STRUCTURE", r"\bRaytracingAccelerationStructure\b"),
        ("RAYQUERY", r"\bRayQuery\b"),
        ("BUFFER", r"\bBuffer\b"),
        ("RWBUFFER", r"\bRWBuffer\b"),
        ("SAMPLER_STATE", r"\bSamplerState\b"),
        ("SAMPLER_COMPARISON_STATE", r"\bSamplerComparisonState\b"),
        ("INPUTPATCH", r"\bInputPatch\b"),
        ("OUTPUTPATCH", r"\bOutputPatch\b"),
        ("POINTSTREAM", r"\bPointStream\b"),
        ("LINESTREAM", r"\bLineStream\b"),
        ("TRIANGLESTREAM", r"\bTriangleStream\b"),
        # Matrix types (must come before vector and scalar types)
        ("MATRIX", r"\b(float|half|double|int|uint|bool)[2-4]x[2-4]\b"),
        # Vector types (must come before scalar types)
        ("FVECTOR", r"\b(float|half|double)[2-4]\b"),
        ("IVECTOR", r"\bint[2-4]\b"),
        ("UVECTOR", r"\buint[2-4]\b"),
        ("BVECTOR", r"\bbool[2-4]\b"),
        # Scalar types
        ("FLOAT", r"\bfloat\b"),
        ("HALF", r"\bhalf\b"),
        ("DOUBLE", r"\bdouble\b"),
        ("INT", r"\bint\b"),
        ("UINT", r"\buint\b"),
        ("BOOL", r"\bbool\b"),
        ("VOID", r"\bvoid\b"),
        ("DWORD", r"\bdword\b"),
        ("MIN16FLOAT", r"\bmin16float\b"),
        ("MIN10FLOAT", r"\bmin10float\b"),
        ("MIN16INT", r"\bmin16int\b"),
        ("MIN12INT", r"\bmin12int\b"),
        ("MIN16UINT", r"\bmin16uint\b"),
        ("INT64_T", r"\bint64_t\b"),
        ("UINT64_T", r"\buint64_t\b"),
        # Control flow keywords
        ("RETURN", r"\breturn\b"),
        ("ELSE_IF", r"\belse\s+if\b"),
        ("IF", r"\bif\b"),
        ("ELSE", r"\belse\b"),
        ("FOR", r"\bfor\b"),
        ("WHILE", r"\bwhile\b"),
        ("DO", r"\bdo\b"),
        ("SWITCH", r"\bswitch\b"),
        ("CASE", r"\bcase\b"),
        ("DEFAULT", r"\bdefault\b"),
        ("BREAK", r"\bbreak\b"),
        ("CONTINUE", r"\bcontinue\b"),
        ("DISCARD", r"\bdiscard\b"),
        ("CLIP", r"\bclip\b"),
        # Register and semantic keywords
        ("REGISTER", r"\bregister\b"),
        ("PACKOFFSET", r"\bpackoffset\b"),
        # Input/output modifiers
        ("IN", r"\bin\b"),
        ("OUT", r"\bout\b"),
        ("INOUT", r"\binout\b"),
        ("UNIFORM", r"\buniform\b"),
        # Boolean literals
        ("TRUE", r"\btrue\b"),
        ("FALSE", r"\bfalse\b"),
        # Identifiers (must come after all keywords)
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        # Numeric literals (hex, binary, octal, float, int with suffixes)
        (
            "HEX_NUMBER",
            r"0[xX][0-9a-fA-F]+(?:[uU][lL]?|[lL][uU]?|[uUlL]{0,2})?",
        ),
        (
            "BINARY_NUMBER",
            r"0[bB][01]+(?:[uU][lL]?|[lL][uU]?|[uUlL]{0,2})?",
        ),
        (
            "OCT_NUMBER",
            r"0[0-7]+(?:[uU][lL]?|[lL][uU]?|[uUlL]{0,2})?",
        ),
        (
            "NUMBER",
            r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fFhH]?|\d+[eE][+-]?\d+[fFhH]?|\d+(?:[uU][lL]?|[lL][uU]?|[fFhH]|[uUlL]{0,2})?",
        ),
        # Brackets and braces
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACKET", r"\["),
        ("RBRACKET", r"\]"),
        # Punctuation
        ("SEMICOLON", r";"),
        ("STRING", r'"([^"\\]|\\.)*"'),
        ("CHAR_LITERAL", r"'(?:[^'\\]|\\.)'"),
        ("COMMA", r","),
        ("COLON", r":"),
        ("QUESTION", r"\?"),
        # Assignment operators (compound first)
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("MOD_EQUALS", r"%="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        # Shift operators (must come before comparison operators)
        ("SHIFT_LEFT", r"<<"),
        ("SHIFT_RIGHT", r">>"),
        # Comparison operators (multi-char first)
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        # Logical operators
        ("LOGICAL_AND", r"&&"),
        ("LOGICAL_OR", r"\|\|"),
        ("LOGICAL_NOT", r"!"),
        # Assignment operator (single)
        ("EQUALS", r"="),
        # Bitwise operators
        ("BITWISE_NOT", r"~"),
        ("BITWISE_XOR", r"\^"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_AND", r"&"),
        # Arithmetic operators
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("MOD", r"%"),
        # Member access
        ("DOT", r"\."),
        # Whitespace (skipped)
        ("WHITESPACE", r"\s+"),
    ]
)

KEYWORDS = {
    "enum": "ENUM",
    "typedef": "TYPEDEF",
    "struct": "STRUCT",
    "cbuffer": "CBUFFER",
    "tbuffer": "TBUFFER",
    "groupshared": "GROUPSHARED",
    "static": "STATIC",
    "const": "CONST",
    "inline": "INLINE",
    "extern": "EXTERN",
    "volatile": "VOLATILE",
    "precise": "PRECISE",
    "row_major": "ROW_MAJOR",
    "column_major": "COLUMN_MAJOR",
    "nointerpolation": "NOINTERPOLATION",
    "linear": "LINEAR",
    "centroid": "CENTROID",
    "sample": "SAMPLE",
    "Texture1D": "TEXTURE1D",
    "Texture1DArray": "TEXTURE1DARRAY",
    "Texture2D": "TEXTURE2D",
    "Texture2DArray": "TEXTURE2DARRAY",
    "Texture2DMS": "TEXTURE2DMS",
    "Texture2DMSArray": "TEXTURE2DMSARRAY",
    "Texture3D": "TEXTURE3D",
    "TextureCube": "TEXTURECUBE",
    "TextureCubeArray": "TEXTURECUBEARRAY",
    "FeedbackTexture2D": "FEEDBACKTEXTURE2D",
    "FeedbackTexture2DArray": "FEEDBACKTEXTURE2DARRAY",
    "RWTexture1D": "RWTEXTURE1D",
    "RWTexture1DArray": "RWTEXTURE1DARRAY",
    "RWTexture2D": "RWTEXTURE2D",
    "RWTexture2DArray": "RWTEXTURE2DARRAY",
    "RWTexture2DMS": "RWTEXTURE2DMS",
    "RWTexture2DMSArray": "RWTEXTURE2DMSARRAY",
    "RWTexture3D": "RWTEXTURE3D",
    "RWTextureCube": "RWTEXTURECUBE",
    "RWTextureCubeArray": "RWTEXTURECUBEARRAY",
    "StructuredBuffer": "STRUCTUREDBUFFER",
    "RWStructuredBuffer": "RWSTRUCTUREDBUFFER",
    "AppendStructuredBuffer": "APPENDSTRUCTUREDBUFFER",
    "ConsumeStructuredBuffer": "CONSUMESTRUCTUREDBUFFER",
    "ByteAddressBuffer": "BYTEADDRESSBUFFER",
    "RWByteAddressBuffer": "RWBYTEADDRESSBUFFER",
    "RaytracingAccelerationStructure": "RAYTRACING_ACCELERATION_STRUCTURE",
    "RayQuery": "RAYQUERY",
    "Buffer": "BUFFER",
    "RWBuffer": "RWBUFFER",
    "SamplerState": "SAMPLER_STATE",
    "SamplerComparisonState": "SAMPLER_COMPARISON_STATE",
    "InputPatch": "INPUTPATCH",
    "OutputPatch": "OUTPUTPATCH",
    "PointStream": "POINTSTREAM",
    "LineStream": "LINESTREAM",
    "TriangleStream": "TRIANGLESTREAM",
    "float": "FLOAT",
    "half": "HALF",
    "double": "DOUBLE",
    "int": "INT",
    "uint": "UINT",
    "bool": "BOOL",
    "void": "VOID",
    "dword": "DWORD",
    "min16float": "MIN16FLOAT",
    "min10float": "MIN10FLOAT",
    "min16int": "MIN16INT",
    "min12int": "MIN12INT",
    "min16uint": "MIN16UINT",
    "int64_t": "INT64_T",
    "uint64_t": "UINT64_T",
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
    "clip": "CLIP",
    "register": "REGISTER",
    "packoffset": "PACKOFFSET",
    "in": "IN",
    "out": "OUT",
    "inout": "INOUT",
    "uniform": "UNIFORM",
    "true": "TRUE",
    "false": "FALSE",
}


class TokenType(Enum):
    """Enumeration of all token types for type-safe token handling"""

    COMMENT_SINGLE = auto()
    COMMENT_MULTI = auto()
    PREPROCESSOR = auto()
    ENUM = auto()
    TYPEDEF = auto()
    STRUCT = auto()
    CBUFFER = auto()
    TBUFFER = auto()
    GROUPSHARED = auto()
    STATIC = auto()
    CONST = auto()
    INLINE = auto()
    EXTERN = auto()
    VOLATILE = auto()
    PRECISE = auto()
    ROW_MAJOR = auto()
    COLUMN_MAJOR = auto()
    NOINTERPOLATION = auto()
    LINEAR = auto()
    CENTROID = auto()
    SAMPLE = auto()
    TEXTURE1D = auto()
    TEXTURE1DARRAY = auto()
    TEXTURE2D = auto()
    TEXTURE2DARRAY = auto()
    TEXTURE2DMS = auto()
    TEXTURE2DMSARRAY = auto()
    TEXTURE3D = auto()
    TEXTURECUBE = auto()
    TEXTURECUBEARRAY = auto()
    FEEDBACKTEXTURE2D = auto()
    FEEDBACKTEXTURE2DARRAY = auto()
    RWTEXTURE1D = auto()
    RWTEXTURE1DARRAY = auto()
    RWTEXTURE2D = auto()
    RWTEXTURE2DARRAY = auto()
    RWTEXTURE2DMS = auto()
    RWTEXTURE2DMSARRAY = auto()
    RWTEXTURE3D = auto()
    RWTEXTURECUBE = auto()
    RWTEXTURECUBEARRAY = auto()
    STRUCTUREDBUFFER = auto()
    RWSTRUCTUREDBUFFER = auto()
    APPENDSTRUCTUREDBUFFER = auto()
    CONSUMESTRUCTUREDBUFFER = auto()
    BYTEADDRESSBUFFER = auto()
    RWBYTEADDRESSBUFFER = auto()
    RAYTRACING_ACCELERATION_STRUCTURE = auto()
    RAYQUERY = auto()
    BUFFER = auto()
    RWBUFFER = auto()
    SAMPLER_STATE = auto()
    SAMPLER_COMPARISON_STATE = auto()
    INPUTPATCH = auto()
    OUTPUTPATCH = auto()
    POINTSTREAM = auto()
    LINESTREAM = auto()
    TRIANGLESTREAM = auto()
    MATRIX = auto()
    FVECTOR = auto()
    IVECTOR = auto()
    UVECTOR = auto()
    BVECTOR = auto()
    FLOAT = auto()
    HALF = auto()
    DOUBLE = auto()
    INT = auto()
    UINT = auto()
    BOOL = auto()
    VOID = auto()
    DWORD = auto()
    MIN16FLOAT = auto()
    MIN10FLOAT = auto()
    MIN16INT = auto()
    MIN12INT = auto()
    MIN16UINT = auto()
    INT64_T = auto()
    UINT64_T = auto()
    RETURN = auto()
    ELSE_IF = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    BREAK = auto()
    CONTINUE = auto()
    DISCARD = auto()
    CLIP = auto()
    REGISTER = auto()
    PACKOFFSET = auto()
    IN = auto()
    OUT = auto()
    INOUT = auto()
    UNIFORM = auto()
    TRUE = auto()
    FALSE = auto()
    IDENTIFIER = auto()
    HEX_NUMBER = auto()
    BINARY_NUMBER = auto()
    OCT_NUMBER = auto()
    NUMBER = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    STRING = auto()
    CHAR_LITERAL = auto()
    COMMA = auto()
    COLON = auto()
    QUESTION = auto()
    ASSIGN_SHIFT_LEFT = auto()
    ASSIGN_SHIFT_RIGHT = auto()
    PLUS_EQUALS = auto()
    MINUS_EQUALS = auto()
    MULTIPLY_EQUALS = auto()
    DIVIDE_EQUALS = auto()
    MOD_EQUALS = auto()
    ASSIGN_AND = auto()
    ASSIGN_OR = auto()
    ASSIGN_XOR = auto()
    EQUALS = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()
    BITWISE_NOT = auto()
    BITWISE_XOR = auto()
    BITWISE_OR = auto()
    BITWISE_AND = auto()
    INCREMENT = auto()
    DECREMENT = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MOD = auto()
    DOT = auto()
    WHITESPACE = auto()
    EOF = auto()


class Token:
    """Represents a single token with type and text"""

    def __init__(self, token_type: TokenType, text: str):
        self.token_type = token_type
        self.text = text

    def __repr__(self):
        return f"Token({self.token_type}, '{self.text}')"


class HLSLLexer:
    """Lexer for High-Level Shading Language (HLSL)"""

    def __init__(
        self,
        code: str,
        preprocess: bool = True,
        file_path: Optional[str] = None,
        include_paths: Optional[List[str]] = None,
        defines: Optional[dict] = None,
        strict_preprocessor: bool = False,
    ):
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.file_path = file_path
        self.include_paths = include_paths or []
        if preprocess:
            preprocessor = HLSLPreprocessor(
                include_paths=self.include_paths,
                defines=defines,
                strict=strict_preprocessor,
            )
            code = preprocessor.preprocess(code, file_path=file_path)
        self.code = code
        self._length = len(code)

    def tokenize(self) -> List[Tuple[str, str]]:
        """Tokenize the input code and return list of tokens"""
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        """Generator function that yields tokens one at a time"""
        pos = 0
        while pos < self._length:
            if self.code.startswith("/*", pos) and "*/" not in self.code[pos + 2 :]:
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)
                raise SyntaxError(
                    f"Unterminated comment starting at line {line_num}, column {col_num}"
                )
            token = self._next_token(pos)
            if token is None:
                # Provide more context in error message
                line_num = self.code[:pos].count("\n") + 1
                col_num = pos - self.code.rfind("\n", 0, pos)
                context = self.code[max(0, pos - 20) : min(self._length, pos + 20)]
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at line {line_num}, column {col_num}\n"
                    f"Context: ...{context}..."
                )
            new_pos, token_type, text = token

            # Check if identifier is a keyword
            if token_type == "IDENTIFIER" and text in KEYWORDS:
                token_type = KEYWORDS[text]

            if token_type not in SKIP_TOKENS:
                yield (token_type, text)

            pos = new_pos

        yield ("EOF", "")

    def _next_token(self, pos: int) -> Optional[Tuple[int, str, str]]:
        """Find the next token starting at the given position"""
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                return match.end(0), token_type, match.group(0)
        return None

    @classmethod
    def from_file(cls, filepath: str) -> "HLSLLexer":
        """Create a lexer instance from a file"""
        with open(filepath, "r", encoding="utf-8") as f:
            base_dir = os.path.dirname(filepath)
            return cls(f.read(), file_path=filepath, include_paths=[base_dir])


class Lexer:
    """Compatibility wrapper around HLSLLexer for legacy code"""

    def __init__(self, input_str: str):
        self.lexer = HLSLLexer(input_str)
        self.tokens = self.lexer.tokenize()
        self.current_pos = 0

    def next(self) -> Tuple[str, str]:
        """Get the next token and advance position"""
        if self.current_pos < len(self.tokens):
            token = self.tokens[self.current_pos]
            self.current_pos += 1
            return token
        return ("EOF", "")

    def peek(self) -> Tuple[str, str]:
        """Look at the next token without advancing position"""
        if self.current_pos < len(self.tokens):
            return self.tokens[self.current_pos]
        return ("EOF", "")

    def reset(self):
        """Reset the lexer to the beginning"""
        self.current_pos = 0
