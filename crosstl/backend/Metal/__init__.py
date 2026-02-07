"""
Metal Backend Module.

This module provides parsing and reverse translation capabilities for Apple
Metal Shading Language. It includes:

- MetalLexer: Tokenizes Metal source code
- MetalParser: Parses tokens into an AST
- MetalToCrossGLConverter: Converts Metal AST to CrossGL code

Example:
    >>> from crosstl.backend.Metal import MetalLexer, MetalParser
    >>> lexer = MetalLexer(metal_code)
    >>> parser = MetalParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .MetalParser import MetalParser
from .MetalLexer import MetalLexer
from .MetalCrossGLCodeGen import MetalToCrossGLConverter
