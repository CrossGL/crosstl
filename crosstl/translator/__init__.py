from . import lexer
from . import parser
from . import codegen
from .lexer import Lexer
from .parser import Parser
from .codegen.registry import BackendSpec, register_backend, backend_names, get_backend
from .source_registry import SourceSpec, SOURCE_REGISTRY, register_default_sources
from .plugin_loader import discover_backend_plugins


def parse(shader_code):
    """Parse shader code and return the AST.

    Args:
        shader_code (str): The shader code to parse

    Returns:
        The abstract syntax tree
    """
    lexer = Lexer(shader_code)
    tokens = lexer.tokens
    parser = Parser(tokens)
    return parser.parse()


def register_source(spec: SourceSpec, *, overwrite: bool = False) -> SourceSpec:
    """Register a new backend source (lexer/parser) spec."""
    return SOURCE_REGISTRY.register(spec, overwrite=overwrite)


def register_backend_spec(spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
    """Register a new backend codegen spec."""
    return register_backend(spec, overwrite=overwrite)


def get_backend_spec(name: str):
    """Lookup a backend codegen spec."""
    return get_backend(name)


def supported_backends():
    """Return registered backend codegens."""
    discover_backend_plugins()
    return backend_names()


def supported_sources():
    """Return registered source backends."""
    register_default_sources()
    discover_backend_plugins()
    return SOURCE_REGISTRY.names()
