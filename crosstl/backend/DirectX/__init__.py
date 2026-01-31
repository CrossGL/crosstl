"""
DirectX/HLSL Backend Module.

This module provides parsing and reverse translation capabilities for Microsoft
HLSL (High-Level Shading Language) used in DirectX. It includes:

- HLSLLexer: Tokenizes HLSL source code
- HLSLParser: Parses tokens into an AST
- HLSLToCrossGLConverter: Converts HLSL AST to CrossGL code

Example:
    >>> from crosstl.backend.DirectX import HLSLLexer, HLSLParser
    >>> lexer = HLSLLexer(hlsl_code)
    >>> parser = HLSLParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .DirectxLexer import HLSLLexer
from .DirectxParser import HLSLParser
from .DirectxCrossGLCodeGen import HLSLToCrossGLConverter
