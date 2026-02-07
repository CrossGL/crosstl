"""
Backend Registry Module.

This module provides a registry system for managing code generation backends in CrossTL.
It allows registering, discovering, and retrieving code generators for different target
shader languages.

The registry supports:
    - Registration of backend specifications with codegen classes
    - Lookup by name or alias
    - File extension mapping
    - Backend enumeration

Example:
    >>> from crosstl.translator.codegen.registry import get_codegen
    >>> codegen = get_codegen("metal")
    >>> metal_code = codegen.generate(ast)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Type, Any


def _normalize_backend_name(name: str) -> str:
    """
    Normalize a backend name for consistent lookup.

    Args:
        name: The backend name to normalize.

    Returns:
        The normalized name (stripped and lowercased).

    Raises:
        TypeError: If name is not a string.
    """
    if not isinstance(name, str):
        raise TypeError(f"Backend name must be a string, got {type(name)}")
    return name.strip().lower()


@dataclass(frozen=True)
class BackendSpec:
    """
    Specification for a code generation backend.

    This dataclass holds all information needed to generate code for a target
    shader language, including the code generator class and metadata.

    Attributes:
        name: The canonical name of the backend (e.g., "metal", "directx").
        codegen_class: The code generator class for this backend.
        aliases: Alternative names for this backend (e.g., ("hlsl", "dx")).
        file_extensions: Output file extensions (e.g., (".metal",)).
        format_backend: Backend name for code formatting, if different from name.
    """

    name: str
    codegen_class: Type[Any]
    aliases: Sequence[str] = ()
    file_extensions: Sequence[str] = ()
    format_backend: Optional[str] = None


class BackendRegistry:
    """
    Registry for code generation backend specifications.

    Manages registration and lookup of code generator backends by name or alias.
    Provides methods to enumerate available backends.

    Attributes:
        _by_name: Dictionary mapping canonical names to BackendSpec objects.
        _by_alias: Dictionary mapping aliases to canonical names.
    """

    def __init__(self) -> None:
        """Initialize an empty backend registry."""
        self._by_name: Dict[str, BackendSpec] = {}
        self._by_alias: Dict[str, str] = {}

    def register(self, spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
        """
        Register a backend specification.

        Args:
            spec: The BackendSpec to register.
            overwrite: If True, overwrites existing registrations.

        Returns:
            The registered BackendSpec.

        Raises:
            ValueError: If the backend or alias is already registered
                and overwrite is False.
        """
        name = _normalize_backend_name(spec.name)
        if name in self._by_name and not overwrite:
            existing = self._by_name[name]
            if existing.codegen_class is spec.codegen_class:
                return existing
            raise ValueError(f"Backend '{name}' already registered")

        self._by_name[name] = spec

        for alias in spec.aliases:
            alias_key = _normalize_backend_name(alias)
            if alias_key in self._by_alias and not overwrite:
                if self._by_alias[alias_key] == name:
                    continue
                raise ValueError(f"Backend alias '{alias_key}' already registered")
            self._by_alias[alias_key] = name

        return spec

    def resolve_name(self, name: str) -> Optional[str]:
        """
        Resolve a backend name or alias to its canonical name.

        Args:
            name: The backend name or alias to resolve.

        Returns:
            The canonical name, or None if not found.
        """
        if not name:
            return None
        key = _normalize_backend_name(name)
        if key in self._by_name:
            return key
        return self._by_alias.get(key)

    def get(self, name: str) -> Optional[BackendSpec]:
        """
        Get a backend specification by name or alias.

        Args:
            name: The backend name or alias.

        Returns:
            The BackendSpec, or None if not found.
        """
        resolved = self.resolve_name(name)
        if not resolved:
            return None
        return self._by_name.get(resolved)

    def all(self) -> Iterable[BackendSpec]:
        """
        Get all registered backend specifications.

        Returns:
            List of all registered BackendSpec objects.
        """
        return list(self._by_name.values())

    def names(self) -> Sequence[str]:
        """
        Get all registered backend names.

        Returns:
            Sorted list of canonical backend names.
        """
        return sorted(self._by_name.keys())

    def aliases(self) -> Dict[str, str]:
        """
        Get all registered aliases and their canonical names.

        Returns:
            Dictionary mapping aliases to canonical names.
        """
        return dict(self._by_alias)


#: Global backend registry instance.
BACKEND_REGISTRY = BackendRegistry()


def register_backend(spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
    """
    Register a backend specification in the global registry.

    Args:
        spec: The BackendSpec to register.
        overwrite: If True, overwrites existing registrations.

    Returns:
        The registered BackendSpec.
    """
    return BACKEND_REGISTRY.register(spec, overwrite=overwrite)


def normalize_backend_name(name: str) -> Optional[str]:
    """
    Resolve a backend name or alias to its canonical name.

    Args:
        name: The backend name or alias.

    Returns:
        The canonical name, or None if not found.
    """
    return BACKEND_REGISTRY.resolve_name(name)


def get_backend(name: str) -> Optional[BackendSpec]:
    """
    Get a backend specification by name or alias.

    Args:
        name: The backend name or alias.

    Returns:
        The BackendSpec, or None if not found.
    """
    return BACKEND_REGISTRY.get(name)


def backend_names() -> Sequence[str]:
    """
    Get all registered backend names.

    Returns:
        Sorted list of canonical backend names.
    """
    return BACKEND_REGISTRY.names()


def get_backend_extension(name: str) -> Optional[str]:
    """
    Get the default file extension for a backend.

    Args:
        name: The backend name or alias.

    Returns:
        The file extension (e.g., ".metal"), or None if not found.
    """
    spec = BACKEND_REGISTRY.get(name)
    if not spec or not spec.file_extensions:
        return None
    return spec.file_extensions[0]


def get_codegen(name: str):
    """
    Get an instance of the code generator for a backend.

    Args:
        name: The backend name or alias.

    Returns:
        An instance of the code generator class for the specified backend.

    Raises:
        ValueError: If the backend is not supported.
    """
    spec = BACKEND_REGISTRY.get(name)
    if not spec:
        supported = ", ".join(backend_names())
        raise ValueError(
            f"Unsupported backend '{name}'. Supported backends: {supported}"
        )
    return spec.codegen_class()
