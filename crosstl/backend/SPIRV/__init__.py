"""
Vulkan/SPIR-V Backend Module.

This module provides parsing and reverse translation capabilities for Vulkan
SPIR-V assembly. It includes:

- VulkanLexer: Tokenizes SPIR-V assembly source code
- VulkanParser: Parses tokens into an AST
- SPIR-V AST node definitions

Example:
    >>> from crosstl.backend.SPIRV import VulkanLexer, VulkanParser
    >>> lexer = VulkanLexer(spirv_code)
    >>> parser = VulkanParser(lexer.tokens)
    >>> ast = parser.parse()
"""

from .VulkanAst import *
from .VulkanLexer import *
from .VulkanParser import *
from .VulkanCrossGLCodeGen import *
