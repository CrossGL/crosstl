"""
Source Registry Module.

This module provides a registry system for managing source language parsers in CrossTL.
It allows registering, discovering, and retrieving parsers for different shader languages
based on file extensions or language names.

The registry supports:
    - Registration of source language specifications
    - Lookup by name, alias, or file extension
    - Lazy loading of lexer/parser modules
    - Reverse translation code generators

Example:
    >>> from crosstl.translator.source_registry import SOURCE_REGISTRY
    >>> spec = SOURCE_REGISTRY.get_by_extension(".hlsl")
    >>> ast = spec.parse(hlsl_code)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Any
import os

from .lexer import Lexer as CglLexer
from .parser import Parser as CglParser


def _normalize_source_name(name: str) -> str:
    """
    Normalize a source language name for consistent lookup.

    Args:
        name: The source name to normalize.

    Returns:
        The normalized name (stripped and lowercased).

    Raises:
        TypeError: If name is not a string.
    """
    if not isinstance(name, str):
        raise TypeError(f"Source name must be a string, got {type(name)}")
    return name.strip().lower()


def _normalize_extension(ext: str) -> str:
    """
    Normalize a file extension for consistent lookup.

    Args:
        ext: The file extension to normalize.

    Returns:
        The normalized extension with a leading dot (e.g., ".hlsl").
    """
    ext = ext.strip().lower()
    if not ext:
        return ext
    return ext if ext.startswith(".") else f".{ext}"


def _extract_tokens(lexer) -> Any:
    """
    Extract tokens from a lexer using the appropriate interface.

    Args:
        lexer: A lexer instance with tokenization capabilities.

    Returns:
        The tokens from the lexer.

    Raises:
        ValueError: If the lexer doesn't support any known interface.
    """
    if hasattr(lexer, "tokenize"):
        return lexer.tokenize()
    if hasattr(lexer, "get_tokens"):
        return lexer.get_tokens()
    if hasattr(lexer, "tokens"):
        return lexer.tokens
    raise ValueError(f"Unsupported lexer interface: {type(lexer)}")


@dataclass(frozen=True)
class SourceSpec:
    """
    Specification for a source shader language.

    This dataclass holds all information needed to parse a source language,
    including lexer/parser factories and optional reverse translation support.

    Attributes:
        name: The canonical name of the source language (e.g., "directx", "metal").
        extensions: Tuple of file extensions for this language (e.g., (".hlsl",)).
        load_lexer_parser: Factory function that returns (Lexer, Parser) classes.
        reverse_codegen_factory: Optional factory for creating reverse code generators.
        aliases: Alternative names for this source (e.g., ("hlsl", "dx")).
    """

    name: str
    extensions: Sequence[str]
    load_lexer_parser: Callable[[], Tuple[type, type]]
    reverse_codegen_factory: Optional[Callable[[], Any]] = None
    aliases: Sequence[str] = ()

    def parse(self, code: str):
        """
        Parse source code into an AST.

        Args:
            code: The source code to parse.

        Returns:
            The parsed Abstract Syntax Tree.
        """
        lexer_cls, parser_cls = self.load_lexer_parser()
        lexer = lexer_cls(code)
        tokens = _extract_tokens(lexer)
        parser = parser_cls(tokens)
        return parser.parse()


class SourceRegistry:
    """
    Registry for source language specifications.

    Manages registration and lookup of source language parsers by name,
    alias, or file extension. Supports lazy loading of parser modules.

    Attributes:
        _by_name: Dictionary mapping canonical names to SourceSpec objects.
        _by_alias: Dictionary mapping aliases to canonical names.
        _by_extension: Dictionary mapping file extensions to canonical names.
    """

    def __init__(self) -> None:
        """Initialize an empty source registry."""
        self._by_name: Dict[str, SourceSpec] = {}
        self._by_alias: Dict[str, str] = {}
        self._by_extension: Dict[str, str] = {}

    def register(self, spec: SourceSpec, *, overwrite: bool = False) -> SourceSpec:
        """
        Register a source language specification.

        Args:
            spec: The SourceSpec to register.
            overwrite: If True, overwrites existing registrations.

        Returns:
            The registered SourceSpec.

        Raises:
            ValueError: If the source, alias, or extension is already registered
                and overwrite is False.
        """
        name = _normalize_source_name(spec.name)
        if name in self._by_name and not overwrite:
            existing = self._by_name[name]
            if existing.load_lexer_parser is spec.load_lexer_parser:
                return existing
            raise ValueError(f"Source '{name}' already registered")

        self._by_name[name] = spec

        for alias in spec.aliases:
            alias_key = _normalize_source_name(alias)
            if alias_key in self._by_alias and not overwrite:
                if self._by_alias[alias_key] == name:
                    continue
                raise ValueError(f"Source alias '{alias_key}' already registered")
            self._by_alias[alias_key] = name

        for ext in spec.extensions:
            ext_key = _normalize_extension(ext)
            if not ext_key:
                continue
            if ext_key in self._by_extension and not overwrite:
                if self._by_extension[ext_key] == name:
                    continue
                raise ValueError(f"Extension '{ext_key}' already registered")
            self._by_extension[ext_key] = name

        return spec

    def resolve_name(self, name: str) -> Optional[str]:
        """
        Resolve a source name or alias to its canonical name.

        Args:
            name: The source name or alias to resolve.

        Returns:
            The canonical name, or None if not found.
        """
        if not name:
            return None
        key = _normalize_source_name(name)
        if key in self._by_name:
            return key
        return self._by_alias.get(key)

    def get(self, name: str) -> Optional[SourceSpec]:
        """
        Get a source specification by name or alias.

        Args:
            name: The source name or alias.

        Returns:
            The SourceSpec, or None if not found.
        """
        resolved = self.resolve_name(name)
        if not resolved:
            return None
        return self._by_name.get(resolved)

    def get_by_extension(self, path_or_ext: str) -> Optional[SourceSpec]:
        """
        Get a source specification by file extension or file path.

        Args:
            path_or_ext: A file extension (e.g., ".hlsl") or file path.

        Returns:
            The SourceSpec for the given extension, or None if not found.
        """
        ext = path_or_ext
        if path_or_ext:
            looks_like_path = os.path.basename(path_or_ext) != path_or_ext
            looks_like_filename = not path_or_ext.startswith(".") and "." in path_or_ext
            if looks_like_path or looks_like_filename:
                _, ext = os.path.splitext(path_or_ext)
        ext_key = _normalize_extension(ext or "")
        name = self._by_extension.get(ext_key)
        if not name:
            return None
        return self._by_name.get(name)

    def names(self) -> Sequence[str]:
        """
        Get all registered source names.

        Returns:
            Sorted list of canonical source names.
        """
        return sorted(self._by_name.keys())

    def extensions(self) -> Sequence[str]:
        """
        Get all registered file extensions.

        Returns:
            Sorted list of registered file extensions.
        """
        return sorted(self._by_extension.keys())


#: Global source registry instance.
SOURCE_REGISTRY = SourceRegistry()


def _load_cgl():
    """Load CrossGL lexer and parser classes."""
    return CglLexer, CglParser


def _load_directx():
    """Load DirectX/HLSL lexer and parser classes."""
    from crosstl.backend.DirectX import HLSLLexer, HLSLParser

    return HLSLLexer, HLSLParser


def _load_metal():
    """Load Metal lexer and parser classes."""
    from crosstl.backend.Metal import MetalLexer, MetalParser

    return MetalLexer, MetalParser


def _load_glsl():
    """Load GLSL lexer and parser classes."""
    from crosstl.backend.GLSL import GLSLLexer, GLSLParser

    return GLSLLexer, GLSLParser


def _load_slang():
    """Load Slang lexer and parser classes."""
    from crosstl.backend.slang import SlangLexer, SlangParser

    return SlangLexer, SlangParser


def _load_spirv():
    """Load SPIR-V/Vulkan lexer and parser classes."""
    from crosstl.backend.SPIRV import VulkanLexer, VulkanParser

    return VulkanLexer, VulkanParser


def _load_mojo():
    """Load Mojo lexer and parser classes."""
    from crosstl.backend.Mojo import MojoLexer, MojoParser

    return MojoLexer, MojoParser


def _load_rust():
    """Load Rust lexer and parser classes."""
    from crosstl.backend.Rust import RustLexer, RustParser

    return RustLexer, RustParser


def _load_cuda():
    """Load CUDA lexer and parser classes."""
    from crosstl.backend.CUDA import CudaLexer, CudaParser

    return CudaLexer, CudaParser


def _load_hip():
    """Load HIP lexer and parser classes."""
    from crosstl.backend.HIP import HipLexer, HipParser

    return HipLexer, HipParser


def _reverse_directx():
    """Create a DirectX/HLSL to CrossGL converter instance."""
    from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter

    return HLSLToCrossGLConverter()


def _reverse_metal():
    """Create a Metal to CrossGL converter instance."""
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter

    return MetalToCrossGLConverter()


def _reverse_glsl():
    """Create a GLSL to CrossGL converter instance."""
    from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter

    return GLSLToCrossGLConverter()


def _reverse_slang():
    """Create a Slang to CrossGL converter instance."""
    from crosstl.backend.slang.SlangCrossGLCodeGen import SlangToCrossGLConverter

    return SlangToCrossGLConverter()


def _reverse_mojo():
    """Create a Mojo to CrossGL converter instance."""
    from crosstl.backend.Mojo.MojoCrossGLCodeGen import MojoToCrossGLConverter

    return MojoToCrossGLConverter()


def _reverse_rust():
    """Create a Rust to CrossGL converter instance."""
    from crosstl.backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter

    return RustToCrossGLConverter()


def _reverse_cuda():
    """Create a CUDA to CrossGL converter instance."""
    from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter

    return CudaToCrossGLConverter()


def _reverse_hip():
    """Create a HIP to CrossGL converter instance."""
    from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter

    return HipToCrossGLConverter()


def register_default_sources() -> None:
    """
    Register all default source language specifications.

    This function registers the built-in source languages:
    - CrossGL (cgl)
    - DirectX/HLSL
    - Metal
    - OpenGL/GLSL
    - Slang
    - Vulkan/SPIR-V
    - Mojo
    - Rust
    - CUDA
    - HIP

    Should be called during module initialization.
    """
    def _register(spec: SourceSpec) -> None:
        try:
            SOURCE_REGISTRY.register(spec)
        except ValueError:
            return

    _register(
        SourceSpec(
            name="cgl",
            extensions=(".cgl",),
            load_lexer_parser=_load_cgl,
            aliases=("crossgl",),
        )
    )
    _register(
        SourceSpec(
            name="directx",
            extensions=(".hlsl",),
            load_lexer_parser=_load_directx,
            reverse_codegen_factory=_reverse_directx,
            aliases=("hlsl", "dx"),
        )
    )
    _register(
        SourceSpec(
            name="metal",
            extensions=(".metal",),
            load_lexer_parser=_load_metal,
            reverse_codegen_factory=_reverse_metal,
            aliases=("metal",),
        )
    )
    _register(
        SourceSpec(
            name="opengl",
            extensions=(".glsl",),
            load_lexer_parser=_load_glsl,
            reverse_codegen_factory=_reverse_glsl,
            aliases=("glsl", "ogl"),
        )
    )
    _register(
        SourceSpec(
            name="slang",
            extensions=(".slang",),
            load_lexer_parser=_load_slang,
            reverse_codegen_factory=_reverse_slang,
            aliases=("slang",),
        )
    )
    _register(
        SourceSpec(
            name="vulkan",
            extensions=(".spv", ".spirv"),
            load_lexer_parser=_load_spirv,
            aliases=("spirv", "spv"),
        )
    )
    _register(
        SourceSpec(
            name="mojo",
            extensions=(".mojo",),
            load_lexer_parser=_load_mojo,
            reverse_codegen_factory=_reverse_mojo,
            aliases=("mojo",),
        )
    )
    _register(
        SourceSpec(
            name="rust",
            extensions=(".rs", ".rust"),
            load_lexer_parser=_load_rust,
            reverse_codegen_factory=_reverse_rust,
            aliases=("rust", "rs"),
        )
    )
    _register(
        SourceSpec(
            name="cuda",
            extensions=(".cu", ".cuh", ".cuda"),
            load_lexer_parser=_load_cuda,
            reverse_codegen_factory=_reverse_cuda,
            aliases=("cuda", "cu"),
        )
    )
    _register(
        SourceSpec(
            name="hip",
            extensions=(".hip",),
            load_lexer_parser=_load_hip,
            reverse_codegen_factory=_reverse_hip,
            aliases=("hip",),
        )
    )
