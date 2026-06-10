"""Public package interface for CrossGL Translator."""

from importlib import import_module

__all__ = ["translate", "supported_backends", "supported_sources", "project"]


def supported_backends():
    """Return registered target backend names."""
    from .translator import supported_backends as _supported_backends

    return _supported_backends()


def supported_sources():
    """Return registered source backend names."""
    from .translator import supported_sources as _supported_sources

    return _supported_sources()


def __getattr__(name):
    if name == "translate":
        from ._crosstl import translate

        return translate
    if name == "project":
        return import_module(f"{__name__}.project")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
