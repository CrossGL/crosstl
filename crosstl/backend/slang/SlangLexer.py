"""Lexer for importing Slang source into CrossGL Translator."""

import re
from typing import Dict, Iterator, List, Optional, Tuple

from .preprocessor import SlangPreprocessor

# using sets for faster lookup
SKIP_TOKENS = {"WHITESPACE", "COMMENT_SINGLE", "COMMENT_MULTI", "PREPROCESSOR"}

TOKENS = tuple(
    [
        ("COMMENT_SINGLE", r"//.*"),
        ("COMMENT_MULTI", r"/\*[\s\S]*?\*/"),
        ("PREPROCESSOR", r"#[^\r\n]*"),
        ("BITWISE_NOT", r"~"),
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
        ("DISCARD", r"\bdiscard\b"),
        ("REGISTER", r"\bregister\b"),
        ("STRING", r'"[^"]*"'),
        ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        (
            "NUMBER",
            r"0[xX][0-9a-fA-F]+[uUlL]*|"
            r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFhHuUlL]*",
        ),
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
        ("ASSIGN_SHIFT_LEFT", r"<<="),
        ("ASSIGN_SHIFT_RIGHT", r">>="),
        ("BITWISE_SHIFT_LEFT", r"<<"),
        ("BITWISE_SHIFT_RIGHT", r">>"),
        ("LESS_EQUAL", r"<="),
        ("GREATER_EQUAL", r">="),
        ("LESS_THAN", r"<"),
        ("GREATER_THAN", r">"),
        ("EQUAL", r"=="),
        ("NOT_EQUAL", r"!="),
        ("NOT", r"!"),
        ("PLUS_EQUALS", r"\+="),
        ("MINUS_EQUALS", r"-="),
        ("MULTIPLY_EQUALS", r"\*="),
        ("DIVIDE_EQUALS", r"/="),
        ("ASSIGN_MOD", r"%="),
        ("ASSIGN_AND", r"&="),
        ("ASSIGN_OR", r"\|="),
        ("ASSIGN_XOR", r"\^="),
        ("INCREMENT", r"\+\+"),
        ("DECREMENT", r"--"),
        ("FAT_ARROW", r"=>"),
        ("AND", r"&&"),
        ("OR", r"\|\|"),
        ("BITWISE_AND", r"&"),
        ("BITWISE_OR", r"\|"),
        ("BITWISE_XOR", r"\^"),
        ("DOT", r"\."),
        ("MULTIPLY", r"\*"),
        ("DIVIDE", r"/"),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("EQUALS", r"="),
        ("WHITESPACE", r"\s+"),
        ("IMPORT", r"\bimport\b"),
        ("EXPORT", r"\bexport\b"),
        ("GENERIC", r"\b__generic\b"),
        ("EXTENSION", r"\bextension\b"),
        ("TYPEDEF", r"\btypedef\b"),
        ("CONST", r"\bconst\b"),
        ("CONSTEXPR", r"\bconstexpr\b"),
        ("STATIC", r"\bstatic\b"),
        ("INLINE", r"\binline\b"),
        ("MOD", r"%"),
    ]
)

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
    "discard": "DISCARD",
    "register": "REGISTER",
    "import": "IMPORT",
    "export": "EXPORT",
    "module": "MODULE",
    "implementing": "IMPLEMENTING",
    "__include": "INCLUDE",
    "__generic": "GENERIC",
    "extension": "EXTENSION",
    "typedef": "TYPEDEF",
    "const": "CONST",
    "constexpr": "CONSTEXPR",
    "static": "STATIC",
    "inline": "INLINE",
}


class SlangLexer:
    """Tokenize Slang source for the Slang backend parser."""

    def __init__(
        self,
        code: str,
        preprocess: bool = True,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict_preprocessor: bool = False,
        max_expansion_depth: int = 64,
        file_path: Optional[str] = None,
    ):
        """Initialize the lexer and optionally preprocess Slang source text."""
        code = code.lstrip("\ufeff")
        if preprocess:
            preprocessor = SlangPreprocessor(
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
        """Return the full token stream as ``(token_type, text)`` tuples."""
        return list(self.token_generator())

    def token_generator(self) -> Iterator[Tuple[str, str]]:
        """Yield Slang tokens while skipping whitespace and comments."""
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
    def from_file(
        cls,
        filepath: str,
        chunk_size: int = 8192,
        preprocess: bool = True,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict_preprocessor: bool = False,
        max_expansion_depth: int = 64,
    ) -> "SlangLexer":
        """Create a lexer instance from a Slang source file."""
        del chunk_size
        with open(filepath, encoding="utf-8") as f:
            return cls(
                f.read(),
                preprocess=preprocess,
                include_paths=include_paths,
                defines=defines,
                strict_preprocessor=strict_preprocessor,
                max_expansion_depth=max_expansion_depth,
                file_path=filepath,
            )
