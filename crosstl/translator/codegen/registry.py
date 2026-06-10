"""Backend registry and alias resolution for CrossGL code generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


def _normalize_backend_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Backend name must be a string, got {type(name)}")
    return name.strip().lower()


@dataclass(frozen=True)
class BackendSpec:
    """Descriptor for a target code generator backend."""

    name: str
    codegen_class: type[Any]
    aliases: Sequence[str] = ()
    file_extensions: Sequence[str] = ()
    format_backend: str | None = None
    source_name: str | None = None
    has_source_frontend: bool = True

    @property
    def source_registry_name(self) -> str | None:
        """Return the native source frontend name paired with this target."""
        if not self.has_source_frontend:
            return None
        return self.source_name or self.name


class BackendRegistry:
    """Lookup table for target code generators by backend name and alias."""

    def __init__(self) -> None:
        self._by_name: dict[str, BackendSpec] = {}
        self._by_alias: dict[str, str] = {}
        self._by_extension: dict[str, str] = {}

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

        for extension in spec.file_extensions:
            extension_key = _normalize_backend_name(extension)
            if extension_key in self._by_extension and not overwrite:
                if self._by_extension[extension_key] == name:
                    continue
                raise ValueError(
                    f"Backend extension '{extension_key}' already registered"
                )
            self._by_extension[extension_key] = name

        return spec

    def resolve_name(self, name: str) -> str | None:
        """Resolve a backend name or alias to its canonical registry name."""
        if not name:
            return None
        key = _normalize_backend_name(name)
        if key in self._by_name:
            return key
        if key in self._by_alias:
            return self._by_alias[key]
        if key in self._by_extension:
            return self._by_extension[key]

        suffixes = [suffix.lower() for suffix in Path(key).suffixes]
        for index in range(len(suffixes)):
            extension_key = "".join(suffixes[index:])
            if extension_key in self._by_extension:
                return self._by_extension[extension_key]
        return None

    def get(self, name: str) -> BackendSpec | None:
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

    def target_backend_names_with_source_frontends(self) -> Sequence[str]:
        """Return target backends that also have native source frontends."""
        return sorted(
            name
            for name, spec in self._by_name.items()
            if spec.source_registry_name is not None
        )

    def source_backend_names(self) -> Sequence[str]:
        """Return target backends that also have native source frontends."""
        return self.target_backend_names_with_source_frontends()

    def aliases(self) -> dict[str, str]:
        """Return a copy of the alias-to-backend mapping."""
        return dict(self._by_alias)

    def extensions(self) -> dict[str, str]:
        """Return a copy of the file-extension-to-backend mapping."""
        return dict(self._by_extension)


BACKEND_REGISTRY = BackendRegistry()


def register_backend(spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
    """Register a target backend in the global backend registry."""
    return BACKEND_REGISTRY.register(spec, overwrite=overwrite)


def normalize_backend_name(name: str) -> str | None:
    """Resolve a backend name or alias through the global registry."""
    return BACKEND_REGISTRY.resolve_name(name)


def get_backend(name: str) -> BackendSpec | None:
    """Return a backend spec from the global registry."""
    return BACKEND_REGISTRY.get(name)


def backend_names() -> Sequence[str]:
    """Return registered backend names from the global registry."""
    return BACKEND_REGISTRY.names()


def source_backend_names() -> Sequence[str]:
    """Return registered targets that also have native source frontends."""
    return BACKEND_REGISTRY.target_backend_names_with_source_frontends()


def target_backend_names_with_source_frontends() -> Sequence[str]:
    """Return registered targets that also have native source frontends."""
    return BACKEND_REGISTRY.target_backend_names_with_source_frontends()


def get_backend_extension(name: str) -> str | None:
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
