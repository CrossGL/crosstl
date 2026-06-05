"""Public package interface for CrossGL Translator."""

__all__ = ["translate"]


def __getattr__(name):
    if name == "translate":
        from ._crosstl import translate

        return translate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
