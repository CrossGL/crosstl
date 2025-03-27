"""
OpenGL backend implementation for CrossGL Translator
"""

# This file is intentionally empty to make the directory a proper Python package

# Make sure the filenames match the case of the actual files on disk

# Import all the necessary components with explicit imports that match the filesystem case
from .OpenglLexer import GLSLLexer
from .OpenglParser import GLSLParser
from .OpenglCrossGLCodeGen import GLSLToCrossGLConverter
from .OpenglAst import ASTNode

# Define __all__ to explicitly control what gets imported with "from .backend.OpenGL import *"
__all__ = ["GLSLLexer", "GLSLParser", "GLSLToCrossGLConverter", "ASTNode"]
