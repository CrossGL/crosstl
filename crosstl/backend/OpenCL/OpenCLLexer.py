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
    ("__SHARED__", r"\b(?:__local|local)\b"),
    ("__CONSTANT__", r"\b(?:__constant|constant)\b"),
    ("__MANAGED__", r"\b(?:__private|private)\b"),
    ("__RESTRICT__", r"\b(?:__restrict__|__restrict|restrict)\b"),
    ("READ_WRITE", r"\b(?:__read_write|read_write)\b"),
    ("READ_ONLY", r"\b(?:__read_only|read_only)\b"),
    ("WRITE_ONLY", r"\b(?:__write_only|write_only)\b"),
    ("SYNCTHREADS", r"\bbarrier\b"),
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

NORMALIZED_VALUES = {
    "__DEVICE__": "__global__",
    "__SHARED__": "__shared__",
    "__CONSTANT__": "__constant__",
    "__MANAGED__": "__private__",
    "__RESTRICT__": "__restrict__",
    "READ_ONLY": "read_only",
    "WRITE_ONLY": "write_only",
    "READ_WRITE": "read_write",
}


def _with_opencl_tokens():
    tokens = []
    inserted = False
    for token_type, pattern in HIP_TOKENS:
        if not inserted and token_type == "IDENTIFIER":
            tokens.extend(OPENCL_TOKENS)
            inserted = True
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
    "mem_fence": "SYNCWARP",
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
        if not re.search(r"\b(__kernel|kernel)\b", content):
            return code

        return (
            code[: match.start()] + "\n" + content + code[content_end + len(closing) :]
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
