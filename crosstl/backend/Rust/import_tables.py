"""Literal patterns, block node families and reserved CrossGL identifiers shared by the Rust importer."""

import re

from .RustAst import (
    AsyncBlockNode,
    BlockNode,
    ConstBlockNode,
    UnsafeBlockNode,
)

RUST_NUMERIC_LITERAL_RE = re.compile(
    r"^(?P<body>"
    r"0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*|"
    r"0[bB][01](?:_?[01])*|"
    r"0[oO][0-7](?:_?[0-7])*|"
    r"\d(?:_?\d)*(?:(?:\.\d(?:_?\d)*)|\.(?![._A-Za-z0-9]))?"
    r"(?:[eE][+-]?\d(?:_?\d)*)?"
    r")(?P<suffix>_?(?:(?:[iu](?:8|16|32|64|128|size))|f(?:32|64)))?$"
)
RUST_RAW_STRING_RE = re.compile(r'^r(?P<hashes>#*)"(.*?)"(?P=hashes)$', re.DOTALL)
RUST_BYTE_RAW_STRING_RE = re.compile(r'^br(?P<hashes>#*)"(.*?)"(?P=hashes)$', re.DOTALL)
RUST_C_RAW_STRING_RE = re.compile(r'^cr(?P<hashes>#*)"(.*?)"(?P=hashes)$', re.DOTALL)
RUST_STRING_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"$', re.DOTALL)
RUST_BYTE_STRING_RE = re.compile(r'^b"((?:[^"\\]|\\.)*)"$', re.DOTALL)
RUST_C_STRING_RE = re.compile(r'^c"((?:[^"\\]|\\.)*)"$', re.DOTALL)
RUST_BYTE_CHAR_RE = re.compile(r"^b'((?:[^'\\]|\\.)*)'$", re.DOTALL)
RUST_CHAR_RE = re.compile(
    r"^'(\\x[0-9a-fA-F]{2}|\\u\{[0-9a-fA-F_]{1,6}\}|[^'\\]|\\.)'$",
    re.DOTALL,
)
TRANSPARENT_BLOCK_NODE_TYPES = (AsyncBlockNode, UnsafeBlockNode, ConstBlockNode)
BLOCK_EXPRESSION_NODE_TYPES = (BlockNode,) + TRANSPARENT_BLOCK_NODE_TYPES
try:
    from crosstl.translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
except ImportError:
    CROSSGL_KEYWORDS = {}
CROSSGL_RESERVED_IDENTIFIERS = set(CROSSGL_KEYWORDS) | {"true", "false"}
