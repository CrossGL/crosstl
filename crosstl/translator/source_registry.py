"""Registry for source-language parsers and reverse code generators."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .lexer import Lexer as CglLexer
from .parser import Parser as CglParser


def _normalize_source_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Source name must be a string, got {type(name)}")
    return name.strip().lower()


def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        return ext
    return ext if ext.startswith(".") else f".{ext}"


def _extract_tokens(lexer) -> Any:
    if hasattr(lexer, "tokens") and lexer.tokens:
        return lexer.tokens
    if hasattr(lexer, "tokenize"):
        result = lexer.tokenize()
        if result is not None:
            return result
        if hasattr(lexer, "tokens") and lexer.tokens:
            return lexer.tokens
    if hasattr(lexer, "get_tokens"):
        result = lexer.get_tokens()
        if result is not None:
            return result
    if hasattr(lexer, "token_generator"):
        return list(lexer.token_generator())
    raise ValueError(f"Unsupported lexer interface: {type(lexer)}")


def _accepts_keyword(callable_obj, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


@dataclass(frozen=True)
class SourceSpec:
    """Descriptor for a source language frontend.

    A source spec connects file extensions and aliases to a lazily imported
    lexer/parser pair. Specs can also provide a reverse code generator factory
    when the source language can be converted back into CrossGL.
    """

    name: str
    extensions: Sequence[str]
    load_lexer_parser: Callable[[], tuple[type, type]]
    reverse_codegen_factory: Callable[[], Any] | None = None
    aliases: Sequence[str] = ()
    shader_type_from_path: Callable[[str], str | None] | None = None

    def parse(self, code: str, file_path: str | None = None):
        """Parse source code into that source backend's AST."""
        lexer_cls, parser_cls = self.load_lexer_parser()
        lexer_kwargs = {}
        if file_path is not None and _accepts_keyword(lexer_cls, "file_path"):
            lexer_kwargs["file_path"] = file_path
        lexer = lexer_cls(code, **lexer_kwargs)
        tokens = _extract_tokens(lexer)
        parser_kwargs = {}
        if (
            file_path is not None
            and self.shader_type_from_path is not None
            and _accepts_keyword(parser_cls, "shader_type")
        ):
            shader_type = self.shader_type_from_path(file_path)
            if shader_type:
                parser_kwargs["shader_type"] = shader_type
        parser = parser_cls(tokens, **parser_kwargs)
        return parser.parse()


class SourceRegistry:
    """Lookup table for source parsers by name, alias, and extension."""

    def __init__(self) -> None:
        self._by_name: dict[str, SourceSpec] = {}
        self._by_alias: dict[str, str] = {}
        self._by_extension: dict[str, str] = {}

    def register(self, spec: SourceSpec, *, overwrite: bool = False) -> SourceSpec:
        """Register a source spec and all of its aliases/extensions."""
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

    def resolve_name(self, name: str) -> str | None:
        """Resolve a source name or alias to its canonical registry name."""
        if not name:
            return None
        key = _normalize_source_name(name)
        if key in self._by_name:
            return key
        return self._by_alias.get(key)

    def get(self, name: str) -> SourceSpec | None:
        """Return the source spec registered for a name or alias."""
        resolved = self.resolve_name(name)
        if not resolved:
            return None
        return self._by_name.get(resolved)

    def get_by_extension(self, path_or_ext: str) -> SourceSpec | None:
        """Return the source spec registered for a file path or extension."""
        ext = path_or_ext
        if path_or_ext:
            looks_like_path = os.path.basename(path_or_ext) != path_or_ext
            looks_like_filename = not path_or_ext.startswith(".") and "." in path_or_ext
            if looks_like_path or looks_like_filename:
                _, ext = os.path.splitext(path_or_ext)
        ext_key = _normalize_extension(ext or "")
        name = self._by_extension.get(ext_key)
        if not name and ext_key.startswith("."):
            _, trailing_ext = os.path.splitext(ext_key)
            if trailing_ext and trailing_ext != ext_key:
                name = self._by_extension.get(_normalize_extension(trailing_ext))
        if not name:
            return None
        return self._by_name.get(name)

    def names(self) -> Sequence[str]:
        """Return registered canonical source names in sorted order."""
        return sorted(self._by_name.keys())

    def extensions(self) -> Sequence[str]:
        """Return registered source file extensions in sorted order."""
        return sorted(self._by_extension.keys())


SOURCE_REGISTRY = SourceRegistry()


def _load_cgl():
    return CglLexer, CglParser


def _load_directx():
    from crosstl.backend.DirectX import HLSLLexer, HLSLParser

    return HLSLLexer, HLSLParser


def _load_metal():
    from crosstl.backend.Metal import MetalLexer, MetalParser

    return MetalLexer, MetalParser


def _load_glsl():
    from crosstl.backend.GLSL import GLSLLexer, GLSLParser

    return GLSLLexer, GLSLParser


def _load_slang():
    from crosstl.backend.slang import SlangLexer, SlangParser

    return SlangLexer, SlangParser


def _load_spirv():
    from crosstl.backend.SPIRV import VulkanLexer, VulkanParser

    return VulkanLexer, VulkanParser


def _load_mojo():
    from crosstl.backend.Mojo import MojoLexer, MojoParser

    return MojoLexer, MojoParser


def _load_rust():
    from crosstl.backend.Rust import RustLexer, RustParser

    return RustLexer, RustParser


def _load_cuda():
    from crosstl.backend.CUDA import CudaLexer, CudaParser

    return CudaLexer, CudaParser


def _load_hip():
    from crosstl.backend.HIP import HipLexer, HipParser

    return HipLexer, HipParser


def _reverse_directx():
    from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter

    return HLSLToCrossGLConverter()


def _reverse_metal():
    from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter

    return MetalToCrossGLConverter()


def _reverse_glsl():
    from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter

    return GLSLToCrossGLConverter(shader_type=None)


_GLSL_EXTENSION_SHADER_TYPES = {
    ".glsl": "auto",
    ".vs": "vertex",
    ".vert": "vertex",
    ".fs": "fragment",
    ".frag": "fragment",
    ".comp": "compute",
    ".geom": "geometry",
    ".tesc": "tessellation_control",
    ".tese": "tessellation_evaluation",
    ".mesh": "mesh",
    ".task": "task",
    ".rgen": "ray_generation",
    ".rint": "ray_intersection",
    ".rahit": "ray_any_hit",
    ".rchit": "ray_closest_hit",
    ".rmiss": "ray_miss",
    ".rcall": "ray_callable",
}


def _glsl_shader_type_from_path(file_path: str) -> str | None:
    stem, ext = os.path.splitext(file_path)
    ext = _normalize_extension(ext)
    if ext == ".glsl":
        _, stage_ext = os.path.splitext(stem)
        shader_type = _GLSL_EXTENSION_SHADER_TYPES.get(_normalize_extension(stage_ext))
        if shader_type and shader_type != "auto":
            return shader_type
    return _GLSL_EXTENSION_SHADER_TYPES.get(ext)


def _reverse_slang():
    from crosstl.backend.slang.SlangCrossGLCodeGen import SlangToCrossGLConverter

    return SlangToCrossGLConverter()


def _reverse_spirv():
    from crosstl.backend.SPIRV.VulkanCrossGLCodeGen import VulkanToCrossGLConverter

    return VulkanToCrossGLConverter()


def _reverse_mojo():
    from crosstl.backend.Mojo.MojoCrossGLCodeGen import MojoToCrossGLConverter

    return MojoToCrossGLConverter()


def _reverse_rust():
    from crosstl.backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter

    return RustToCrossGLConverter()


def _reverse_cuda():
    from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter

    return CudaToCrossGLConverter()


def _reverse_hip():
    from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter

    return HipToCrossGLConverter()


def register_default_sources() -> None:
    """Register the built-in CrossGL and native source frontends."""

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
            extensions=(".hlsl", ".hlsli", ".fx", ".fxh"),
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
            aliases=("metal", "msl"),
        )
    )
    _register(
        SourceSpec(
            name="opengl",
            extensions=(
                ".glsl",
                ".vs",
                ".fs",
                ".vert",
                ".frag",
                ".comp",
                ".geom",
                ".tesc",
                ".tese",
                ".mesh",
                ".task",
                ".rgen",
                ".rint",
                ".rahit",
                ".rchit",
                ".rmiss",
                ".rcall",
            ),
            load_lexer_parser=_load_glsl,
            reverse_codegen_factory=_reverse_glsl,
            aliases=("glsl", "ogl"),
            shader_type_from_path=_glsl_shader_type_from_path,
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
            extensions=(".spvasm",),
            load_lexer_parser=_load_spirv,
            reverse_codegen_factory=_reverse_spirv,
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
