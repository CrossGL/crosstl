"""Backend registry and alias resolution for CrossGL code generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Type, Any


def _normalize_backend_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Backend name must be a string, got {type(name)}")
    return name.strip().lower()


@dataclass(frozen=True)
class BackendSpec:
    """Descriptor for a target code generator backend."""

    name: str
    codegen_class: Type[Any]
    aliases: Sequence[str] = ()
    file_extensions: Sequence[str] = ()
    format_backend: Optional[str] = None


class BackendRegistry:
    """Lookup table for target code generators by backend name and alias."""

    def __init__(self) -> None:
        self._by_name: Dict[str, BackendSpec] = {}
        self._by_alias: Dict[str, str] = {}

    def register(self, spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
        """Register a backend spec and all of its aliases."""
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
        """Resolve a backend name or alias to its canonical registry name."""
        if not name:
            return None
        key = _normalize_backend_name(name)
        if key in self._by_name:
            return key
        return self._by_alias.get(key)

    def get(self, name: str) -> Optional[BackendSpec]:
        """Return the backend spec registered for a name or alias."""
        resolved = self.resolve_name(name)
        if not resolved:
            return None
        return self._by_name.get(resolved)

    def all(self) -> Iterable[BackendSpec]:
        """Return all registered backend specs."""
        return list(self._by_name.values())

    def names(self) -> Sequence[str]:
        """Return registered canonical backend names in sorted order."""
        return sorted(self._by_name.keys())

    def aliases(self) -> Dict[str, str]:
        """Return a copy of the alias-to-backend mapping."""
        return dict(self._by_alias)


BACKEND_REGISTRY = BackendRegistry()


def register_backend(spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
    """Register a target backend in the global backend registry."""
    return BACKEND_REGISTRY.register(spec, overwrite=overwrite)


def normalize_backend_name(name: str) -> Optional[str]:
    """Resolve a backend name or alias through the global registry."""
    return BACKEND_REGISTRY.resolve_name(name)


def get_backend(name: str) -> Optional[BackendSpec]:
    """Return a backend spec from the global registry."""
    return BACKEND_REGISTRY.get(name)


def backend_names() -> Sequence[str]:
    """Return registered backend names from the global registry."""
    return BACKEND_REGISTRY.names()


def get_backend_extension(name: str) -> Optional[str]:
    """Return the preferred output extension for a backend, if known."""
    spec = BACKEND_REGISTRY.get(name)
    if not spec or not spec.file_extensions:
        return None
    return spec.file_extensions[0]


def get_codegen(name: str):
    """Instantiate the code generator class for a registered backend."""
    spec = BACKEND_REGISTRY.get(name)
    if not spec:
        supported = ", ".join(backend_names())
        raise ValueError(
            f"Unsupported backend '{name}'. Supported backends: {supported}"
        )
    return spec.codegen_class()
