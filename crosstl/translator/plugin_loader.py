"""
Plugin Loader Module.

This module provides dynamic discovery and loading of backend plugins for CrossTL.
It scans the backend directory for modules containing backend specifications and
registers them automatically.

The plugin system allows:
    - Automatic discovery of backend modules
    - Registration of BackendSpec and SourceSpec objects
    - Support for custom register() functions in plugin modules

Example:
    >>> from crosstl.translator.plugin_loader import discover_backend_plugins
    >>> discover_backend_plugins()  # Loads all backend plugins
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Iterable

from .codegen.registry import BackendSpec, register_backend
from .source_registry import SourceSpec, SOURCE_REGISTRY

logger = logging.getLogger(__name__)

#: Tracks whether plugin discovery has been performed.
_DISCOVERED = {"done": False}


def _backend_root() -> str:
    """
    Get the absolute path to the backend directory.

    Returns:
        Absolute path to the crosstl/backend directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))


def _backend_dirs() -> Iterable[str]:
    """
    Get the list of backend subdirectory names.

    Returns:
        Iterable of directory names under the backend directory,
        excluding hidden directories.
    """
    backend_root = _backend_root()
    if not os.path.isdir(backend_root):
        return []
    return [
        name
        for name in os.listdir(backend_root)
        if os.path.isdir(os.path.join(backend_root, name)) and not name.startswith(".")
    ]


def _register_backend_specs(module) -> None:
    """
    Register backend specifications from a module.

    Looks for BACKEND_SPECS (list) or BACKEND_SPEC (single) attributes
    in the module and registers them.

    Args:
        module: The module to extract backend specs from.
    """
    if hasattr(module, "BACKEND_SPECS"):
        for spec in getattr(module, "BACKEND_SPECS"):
            if isinstance(spec, BackendSpec):
                register_backend(spec, overwrite=True)
    if hasattr(module, "BACKEND_SPEC"):
        spec = getattr(module, "BACKEND_SPEC")
        if isinstance(spec, BackendSpec):
            register_backend(spec, overwrite=True)


def _register_source_specs(module) -> None:
    """
    Register source specifications from a module.

    Looks for SOURCE_SPECS (list) or SOURCE_SPEC (single) attributes
    in the module and registers them.

    Args:
        module: The module to extract source specs from.
    """
    if hasattr(module, "SOURCE_SPECS"):
        for spec in getattr(module, "SOURCE_SPECS"):
            if isinstance(spec, SourceSpec):
                SOURCE_REGISTRY.register(spec, overwrite=True)
    if hasattr(module, "SOURCE_SPEC"):
        spec = getattr(module, "SOURCE_SPEC")
        if isinstance(spec, SourceSpec):
            SOURCE_REGISTRY.register(spec, overwrite=True)


def discover_backend_plugins(force: bool = False) -> None:
    """
    Discover and register backend plugins from the crosstl/backend directory.

    Scans each backend subdirectory for 'backend_spec' and 'source_spec' modules,
    importing them and registering any BackendSpec or SourceSpec objects found.

    Args:
        force: If True, re-discovers plugins even if already discovered.
            Default is False.

    Note:
        This function is idempotent by default - calling it multiple times
        has no additional effect unless force=True is specified.
    """
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
