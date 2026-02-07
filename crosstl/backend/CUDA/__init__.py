"""
CUDA Backend Module.

This module provides parsing and reverse translation capabilities for NVIDIA
CUDA code. It includes:

- CudaLexer: Tokenizes CUDA source code
- CudaParser: Parses tokens into an AST
- CudaToCrossGLConverter: Converts CUDA AST to CrossGL code

Example:
    >>> from crosstl.backend.CUDA import CudaLexer, CudaParser
    >>> lexer = CudaLexer(cuda_code)
    >>> parser = CudaParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .CudaLexer import CudaLexer
from .CudaParser import CudaParser
from .CudaCrossGLCodeGen import CudaToCrossGLConverter
