"""
OpenGL backend implementation for CrossGL Translator
"""

# This file is intentionally empty to make the directory a proper Python package

from .OpenglLexer import GLSLLexer
from .OpenglParser import GLSLParser
from .OpenGLCrossGLCodeGen import GLSLToCrossGLConverter
from .OpenglAst import ASTNode
