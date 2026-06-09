"""Lexer for importing OpenCL C source into CrossGL Translator."""

import re
from typing import Dict, List, Optional

from crosstl.backend.HIP.HipLexer import SKIP_TOKENS
from crosstl.backend.HIP.HipLexer import TOKENS as HIP_TOKENS
from crosstl.backend.HIP.HipLexer import Token

from .preprocessor import OpenCLPreprocessor

OPENCL_TOKENS = (
    ("__GLOBAL__", r"\b(?:__kernel|kernel)\b"),
    ("__DEVICE__", r"\b(?:__global|global)\b"),
    ("__SHARED__", r"\b(?:__local|local|LOCAL_PTR)\b"),
    ("__CONSTANT__", r"\b(?:__constant|constant)\b"),
    ("__MANAGED__", r"\b(?:__private|private)\b"),
    ("__GENERIC__", r"\b(?:__generic|generic)\b"),
    ("__RESTRICT__", r"\b(?:__restrict__|__restrict|restrict)\b"),
    ("CONST", r"\b(?:__const|__const__)\b"),
    ("READ_WRITE", r"\b(?:__read_write|read_write)\b"),
    ("READ_ONLY", r"\b(?:__read_only|read_only)\b"),
    ("WRITE_ONLY", r"\b(?:__write_only|write_only)\b"),
    ("SYNCTHREADS", r"\b(?:barrier|work_group_barrier)\b"),
    ("SYNCWARP", r"\bmem_fence\b"),
    (
        "ATOMICCAS",
        r"\batomic_(?:cmpxchg|compare_exchange(?:_strong|_weak)?(?:_explicit)?)\b",
    ),
    ("ATOMICADD", r"\batomic_(?:add|fetch_add(?:_explicit)?)\b"),
    ("ATOMICSUB", r"\batomic_(?:sub|fetch_sub(?:_explicit)?)\b"),
    ("ATOMICMAX", r"\batomic_(?:max|fetch_max(?:_explicit)?)\b"),
    ("ATOMICMIN", r"\batomic_(?:min|fetch_min(?:_explicit)?)\b"),
    ("ATOMICEXCH", r"\batomic_(?:xchg|exchange(?:_explicit)?)\b"),
    ("ATOMICAND", r"\batomic_(?:and|fetch_and(?:_explicit)?)\b"),
    ("ATOMICOR", r"\batomic_(?:or|fetch_or(?:_explicit)?)\b"),
    ("ATOMICXOR", r"\batomic_(?:xor|fetch_xor(?:_explicit)?)\b"),
    ("ATOMICINC", r"\batomic_inc\b"),
    ("ATOMICDEC", r"\batomic_dec\b"),
)

_HEX_DIGITS = r"[0-9a-fA-F](?:'?[0-9a-fA-F])*"
_DECIMAL_DIGITS = r"\d(?:'?\d)*"
OPENCL_HEX_FLOAT_LITERAL = (
    rf"0[xX](?:{_HEX_DIGITS}(?:\.(?:{_HEX_DIGITS})?)?|\.(?:{_HEX_DIGITS}))"
    rf"[pP][+-]?{_DECIMAL_DIGITS}(?:(?:[fF]16)|[fFdDlLhH])*"
)
OPENCL_HALF_FLOAT_LITERAL = (
    rf"(?:(?:{_DECIMAL_DIGITS}\.(?:{_DECIMAL_DIGITS})?|\.(?:{_DECIMAL_DIGITS}))"
    rf"(?:[eE][+-]?{_DECIMAL_DIGITS})?|{_DECIMAL_DIGITS}[eE][+-]?{_DECIMAL_DIGITS})"
    rf"(?:[hH]|[fF]16)"
)
OPENCL_LITERAL_TOKENS = (
    ("FLOAT", OPENCL_HEX_FLOAT_LITERAL),
    ("FLOAT", OPENCL_HALF_FLOAT_LITERAL),
)

NORMALIZED_VALUES = {
    "__DEVICE__": "__global__",
    "__SHARED__": "__shared__",
    "__CONSTANT__": "__constant__",
    "__MANAGED__": "__private__",
    "__GENERIC__": "__generic__",
    "__RESTRICT__": "__restrict__",
    "CONST": "const",
    "READ_ONLY": "read_only",
    "WRITE_ONLY": "write_only",
    "READ_WRITE": "read_write",
    "SYNCTHREADS": "barrier",
}


def _with_opencl_tokens():
    tokens = []
    inserted_opencl_tokens = False
    inserted_literal_tokens = False
    for token_type, pattern in HIP_TOKENS:
        if (
            not inserted_literal_tokens
            and token_type == "FLOAT"
            and pattern != r"\bfloat\b"
        ):
            tokens.extend(OPENCL_LITERAL_TOKENS)
            inserted_literal_tokens = True
        if not inserted_opencl_tokens and token_type == "IDENTIFIER":
            tokens.extend(OPENCL_TOKENS)
            inserted_opencl_tokens = True
        tokens.append((token_type, pattern))
    return tuple(tokens)


TOKENS = _with_opencl_tokens()

KEYWORDS = {
    "__kernel": "__GLOBAL__",
    "kernel": "__GLOBAL__",
    "__global": "__DEVICE__",
    "global": "__DEVICE__",
    "__local": "__SHARED__",
    "local": "__SHARED__",
    "__constant": "__CONSTANT__",
    "constant": "__CONSTANT__",
    "__private": "__MANAGED__",
    "private": "__MANAGED__",
    "__generic": "__GENERIC__",
    "generic": "__GENERIC__",
    "__restrict__": "__RESTRICT__",
    "__restrict": "__RESTRICT__",
    "restrict": "__RESTRICT__",
    "__read_only": "READ_ONLY",
    "read_only": "READ_ONLY",
    "__write_only": "WRITE_ONLY",
    "write_only": "WRITE_ONLY",
    "__read_write": "READ_WRITE",
    "read_write": "READ_WRITE",
    "barrier": "SYNCTHREADS",
    "work_group_barrier": "SYNCTHREADS",
    "mem_fence": "SYNCWARP",
    "__asm": "ASM",
}


class OpenCLLexer:
    """Tokenize OpenCL C source for the OpenCL backend parser."""

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
        """Initialize the lexer and optionally preprocess OpenCL source text."""
        code = code.lstrip("\ufeff")
        code = self._unwrap_cpp_raw_opencl_literal(code)
        code = self._mask_host_embedded_source_strings(code)
        if preprocess:
            preprocessor = OpenCLPreprocessor(
                include_paths=include_paths,
                defines=defines,
                strict=strict_preprocessor,
                max_expansion_depth=max_expansion_depth,
            )
            code = preprocessor.preprocess(code, file_path=file_path)
        self._token_patterns = [(name, re.compile(pattern)) for name, pattern in TOKENS]
        self.code = code
        self._length = len(code)
        self.reserved_keywords = KEYWORDS
        self.line = 1
        self.column = 1

    def _unwrap_cpp_raw_opencl_literal(self, code: str) -> str:
        match = re.search(r'R"([^\s()\\]{0,16})\(', code)
        if not match:
            return code

        delimiter = match.group(1)
        closing = ")" + delimiter + '"'
        content_start = match.end()
        content_end = code.find(closing, content_start)
        if content_end == -1:
            return code

        content = code[content_start:content_end]
        if not self._looks_like_embedded_opencl(content):
            return code

        return (
            code[: match.start()] + "\n" + content + code[content_end + len(closing) :]
        )

    def _mask_host_embedded_source_strings(self, code: str) -> str:
        """Keep host-side OpenCL templates inert during preprocessing.

        Some imported corpora store kernels in multi-line C string constants.
        Their bodies can contain preprocessor directives such as ``#error`` that
        belong to the eventual OpenCL build, not to the host source wrapper.
        """

        pattern = re.compile(
            r"(?P<prefix>\b(?:static\s+)?const\s+char\s*\*\s*"
            r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*)"
            r'"(?P<body>[\s\S]*?)"(?P<suffix>\s*;)',
            re.MULTILINE,
        )

        def replace(match):
            body = match.group("body")
            if "\n" not in body or not self._looks_like_embedded_opencl(body):
                return match.group(0)
            return f'{match.group("prefix")}"\n"{match.group("suffix")}'

        return pattern.sub(replace, code)

    def _looks_like_embedded_opencl(self, content: str) -> bool:
        return bool(
            re.search(
                r"\b(__kernel|kernel|__global|global|__local|local|INLINE_FUNC)\b"
                r"|#\s*(?:define|pragma|if|ifdef|ifndef|include)\b",
                content,
            )
        )

    def tokenize(self) -> List[Token]:
        """Return the full OpenCL token stream with source locations."""
        tokens = []
        pos = 0

        while pos < self._length:
            token = self._next_token(pos)
            if token is None:
                raise SyntaxError(
                    f"Illegal character '{self.code[pos]}' at position {pos}"
                )
            new_pos, token_type, text = token

            if token_type == "IDENTIFIER" and text in self.reserved_keywords:
                token_type = self.reserved_keywords[text]
            elif token_type == "PRIVATE" and text == "private":
                token_type = "__MANAGED__"

            if token_type not in SKIP_TOKENS:
                value = NORMALIZED_VALUES.get(token_type, text)
                tokens.append(Token(token_type, value, self.line, self.column))

            self._advance_position(text)
            pos = new_pos

        return tokens

    def _advance_position(self, text: str):
        newline_count = text.count("\n")
        if newline_count == 0:
            self.column += len(text)
            return

        self.line += newline_count
        self.column = len(text.rsplit("\n", 1)[1]) + 1

    def _next_token(self, pos: int):
        for token_type, pattern in self._token_patterns:
            match = pattern.match(self.code, pos)
            if match:
                text = match.group(0)
                return match.end(), token_type, text
        return None


def parse_opencl_code(code: str):
    """Parse OpenCL source text and return the backend AST."""
    from .OpenCLParser import OpenCLParser

    lexer = OpenCLLexer(code)
    tokens = lexer.tokenize()
    parser = OpenCLParser(tokens)
    return parser.parse()
