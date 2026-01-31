"""
OpenGL/GLSL Backend Module.

This module provides parsing and reverse translation capabilities for OpenGL
GLSL (OpenGL Shading Language). It includes:

- GLSLLexer: Tokenizes GLSL source code
- GLSLParser: Parses tokens into an AST
- GLSLToCrossGLConverter: Converts GLSL AST to CrossGL code
- AST node definitions for GLSL programs

Example:
    >>> from crosstl.backend.GLSL import GLSLLexer, GLSLParser
    >>> lexer = GLSLLexer(glsl_code)
    >>> parser = GLSLParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .OpenglAst import *
from .OpenglLexer import *
from .OpenglParser import *
from .openglCrossglCodegen import *
