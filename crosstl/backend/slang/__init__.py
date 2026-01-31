"""
Slang Backend Module.

This module provides parsing and reverse translation capabilities for Slang
shading language code. Slang is a shading language designed for high-performance
real-time graphics. It includes:

- SlangLexer: Tokenizes Slang source code
- SlangParser: Parses tokens into an AST
- SlangToCrossGLConverter: Converts Slang AST to CrossGL code
- Slang AST node definitions

Example:
    >>> from crosstl.backend.slang import SlangLexer, SlangParser
    >>> lexer = SlangLexer(slang_code)
    >>> parser = SlangParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .SlangAst import *
from .SlangLexer import *
from .SlangCrossGLCodeGen import *
