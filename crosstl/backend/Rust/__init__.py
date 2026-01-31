"""
Rust Backend Module.

This module provides parsing and reverse translation capabilities for Rust GPU
code. It includes:

- RustLexer: Tokenizes Rust source code
- RustParser: Parses tokens into an AST
- RustToCrossGLConverter: Converts Rust AST to CrossGL code

Example:
    >>> from crosstl.backend.Rust import RustLexer, RustParser
    >>> lexer = RustLexer(rust_code)
    >>> parser = RustParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .RustLexer import RustLexer
from .RustParser import RustParser
from .RustCrossGLCodeGen import RustToCrossGLConverter
