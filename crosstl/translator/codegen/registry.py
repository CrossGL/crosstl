from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Type, Any


def _normalize_backend_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Backend name must be a string, got {type(name)}")
    return name.strip().lower()


@dataclass(frozen=True)
class BackendSpec:
    name: str
    codegen_class: Type[Any]
    aliases: Sequence[str] = ()
    file_extensions: Sequence[str] = ()
    format_backend: Optional[str] = None


class BackendRegistry:
    def __init__(self) -> None:
        self._by_name: Dict[str, BackendSpec] = {}
        self._by_alias: Dict[str, str] = {}

    def register(self, spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
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
        if not name:
            return None
        key = _normalize_backend_name(name)
        if key in self._by_name:
            return key
        return self._by_alias.get(key)

    def get(self, name: str) -> Optional[BackendSpec]:
        resolved = self.resolve_name(name)
        if not resolved:
            return None
        return self._by_name.get(resolved)

    def all(self) -> Iterable[BackendSpec]:
        return list(self._by_name.values())

    def names(self) -> Sequence[str]:
        return sorted(self._by_name.keys())

    def aliases(self) -> Dict[str, str]:
        return dict(self._by_alias)


BACKEND_REGISTRY = BackendRegistry()


def register_backend(spec: BackendSpec, *, overwrite: bool = False) -> BackendSpec:
    return BACKEND_REGISTRY.register(spec, overwrite=overwrite)


def normalize_backend_name(name: str) -> Optional[str]:
    return BACKEND_REGISTRY.resolve_name(name)


def get_backend(name: str) -> Optional[BackendSpec]:
    return BACKEND_REGISTRY.get(name)


def backend_names() -> Sequence[str]:
    return BACKEND_REGISTRY.names()


def get_backend_extension(name: str) -> Optional[str]:
    spec = BACKEND_REGISTRY.get(name)
    if not spec or not spec.file_extensions:
        return None
    return spec.file_extensions[0]


def get_codegen(name: str):
    spec = BACKEND_REGISTRY.get(name)
    if not spec:
        supported = ", ".join(backend_names())
        raise ValueError(
            f"Unsupported backend '{name}'. Supported backends: {supported}"
        )
    return spec.codegen_class()
