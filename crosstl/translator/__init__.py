"""
CrossGL Translator Module.

This module provides the core translation infrastructure for CrossGL shaders.
It includes:

- Lexer: Tokenizes CrossGL source code
- Parser: Parses tokens into an Abstract Syntax Tree (AST)
- Code generators: Translate AST to various target languages
- Registry systems: Manage source parsers and backend code generators

The translator supports both forward translation (CrossGL to target language)
and reverse translation (target language to CrossGL).

Example:
    >>> from crosstl.translator import parse, supported_backends
    >>> ast = parse(shader_code)
    >>> print(supported_backends())
    ['cuda', 'directx', 'hip', 'metal', 'mojo', 'opengl', 'rust', 'vulkan']
"""

from . import lexer
from . import parser
from . import codegen
from .lexer import Lexer
from .parser import Parser
from .codegen.registry import BackendSpec, register_backend, backend_names, get_backend
from .source_registry import SourceSpec, SOURCE_REGISTRY, register_default_sources
from .plugin_loader import discover_backend_plugins


def parse(shader_code):
    """
    Parse shader code and return the AST.

    This is a convenience function that creates a lexer and parser,
    tokenizes the input, and returns the parsed Abstract Syntax Tree.

    Args:
        shader_code (str): The CrossGL shader code to parse.

    Returns:
        The abstract syntax tree representing the parsed shader.

    Example:
        >>> ast = parse("shader main { vertex { fn main() {} } }")
    """
    lexer = Lexer(shader_code)
    tokens = lexer.tokens
    parser = Parser(tokens)
    return parser.parse()


def register_source(spec: SourceSpec, *, overwrite: bool = False) -> SourceSpec:
    """
    Register a new source language specification.

    Args:
        spec: The SourceSpec to register.
        overwrite: If True, overwrites existing registrations.

    Returns:
        The registered SourceSpec.
    """
    return SOURCE_REGISTRY.register(spec, overwrite=overwrite)


def register_backend_spec(spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
    """
    Register a new backend code generator specification.

    Args:
        spec: The BackendSpec to register.
        overwrite: If True, overwrites existing registrations.

    Returns:
        The registered BackendSpec.
    """
    return register_backend(spec, overwrite=overwrite)


def get_backend_spec(name: str):
    """
    Lookup a backend code generator specification by name.

    Args:
        name: The backend name or alias.

    Returns:
        The BackendSpec, or None if not found.
    """
    return get_backend(name)


def supported_backends():
    """
    Return the list of registered backend code generators.

    Returns:
        List of backend names (e.g., ['cuda', 'directx', 'metal', ...]).
    """
    discover_backend_plugins()
    return backend_names()


def supported_sources():
    """
    Return the list of registered source language parsers.

    Returns:
        List of source names (e.g., ['cgl', 'directx', 'metal', ...]).
    """
    register_default_sources()
    discover_backend_plugins()
    return SOURCE_REGISTRY.names()
