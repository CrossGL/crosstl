"""64-bit integer literals that initialize 32-bit integer objects in GLSL.

A Metal or C source spells ``2654435761`` with a 64-bit type because it does not fit ``int`` ([lex.icon]/2), and
CrossGL keeps that type (``2654435761l``). Initializing a ``uint`` from it converts the value, which fits
([conv.integral]/3). GLSL has no implicit conversion from ``int64_t`` to ``uint``, so ``uint[3](1, 2654435761l, 5)``
does not compile. When the value fits the 32-bit type it initializes, the literal is spelled with that type.
"""

from __future__ import annotations

from ..ast import LiteralNode

__all__ = ["narrowed_integer_literal"]

_WIDE_LITERAL_TYPES = frozenset({"int64_t", "uint64_t", "int64", "uint64"})
_RANGES = {"uint": (0, 0xFFFFFFFF, "u"), "int": (-(2**31), 2**31 - 1, "")}


def narrowed_integer_literal(generator, expr, expected_type):
    """``expr`` spelled as a ``uint``/``int`` literal when it is a 64-bit literal of a value that type holds."""
    if not isinstance(expr, LiteralNode) or expected_type is None:
        return None
    value = getattr(expr, "value", None)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
    if literal_type not in _WIDE_LITERAL_TYPES:
        return None
    target = _RANGES.get(generator.map_type(expected_type))
    if target is None:
        return None
    low, high, suffix = target
    if not low <= value <= high:
        return None
    return f"{value}{suffix}"
