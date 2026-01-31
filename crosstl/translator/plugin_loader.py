from __future__ import annotations

import importlib
import logging
import os
from typing import Iterable

from .codegen.registry import BackendSpec, register_backend
from .source_registry import SourceSpec, SOURCE_REGISTRY

logger = logging.getLogger(__name__)

_DISCOVERED = {"done": False}


def _backend_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))


def _backend_dirs() -> Iterable[str]:
    backend_root = _backend_root()
    if not os.path.isdir(backend_root):
        return []
    return [
        name
        for name in os.listdir(backend_root)
        if os.path.isdir(os.path.join(backend_root, name)) and not name.startswith(".")
    ]


def _register_backend_specs(module) -> None:
    if hasattr(module, "BACKEND_SPECS"):
        for spec in getattr(module, "BACKEND_SPECS"):
            if isinstance(spec, BackendSpec):
                register_backend(spec, overwrite=True)
    if hasattr(module, "BACKEND_SPEC"):
        spec = getattr(module, "BACKEND_SPEC")
        if isinstance(spec, BackendSpec):
            register_backend(spec, overwrite=True)


def _register_source_specs(module) -> None:
    if hasattr(module, "SOURCE_SPECS"):
        for spec in getattr(module, "SOURCE_SPECS"):
            if isinstance(spec, SourceSpec):
                SOURCE_REGISTRY.register(spec, overwrite=True)
    if hasattr(module, "SOURCE_SPEC"):
        spec = getattr(module, "SOURCE_SPEC")
        if isinstance(spec, SourceSpec):
            SOURCE_REGISTRY.register(spec, overwrite=True)


def discover_backend_plugins(force: bool = False) -> None:
    """Discover backend plugin specs under crosstl/backend."""
    if _DISCOVERED["done"] and not force:
        return

    for backend in _backend_dirs():
        for module_name in ["backend_spec", "source_spec"]:
            module_path = f"crosstl.backend.{backend}.{module_name}"
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                continue
            except Exception as exc:  # pragma: no cover - pylint: disable=broad-except
                logger.warning("Failed to import %s: %s", module_path, exc)
                continue

            try:
                _register_backend_specs(module)
                _register_source_specs(module)
            except Exception as exc:  # pragma: no cover - pylint: disable=broad-except
                logger.warning("Failed to register specs from %s: %s", module_path, exc)

            if hasattr(module, "register"):
                try:
                    module.register()
                except TypeError:
                    try:
                        module.register(register_backend, SOURCE_REGISTRY.register)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(
                            "Failed to call register() in %s: %s",
                            module_path,
                            exc,
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to call register() in %s: %s", module_path, exc
                    )

    _DISCOVERED["done"] = True
