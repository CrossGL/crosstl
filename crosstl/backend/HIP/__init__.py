"""
HIP Backend Module.

This module provides parsing and reverse translation capabilities for AMD HIP
code. HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible
runtime API for GPU programming. It includes:

- HipLexer: Tokenizes HIP source code
- HipParser: Parses tokens into an AST
- HipToCrossGLConverter: Converts HIP AST to CrossGL code
- HIP AST node definitions

Example:
    >>> from crosstl.backend.HIP import HipLexer, HipParser
    >>> lexer = HipLexer(hip_code)
    >>> parser = HipParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .HipLexer import HipLexer
from .HipParser import HipParser, HipProgramNode
from .HipAst import *
from .HipCrossGLCodeGen import HipToCrossGLConverter
