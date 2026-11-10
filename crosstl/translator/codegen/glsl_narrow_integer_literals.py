"""Integer literals spelled with the GLSL integer type they initialize.

A Metal or C source spells ``2654435761`` with a 64-bit type because it does not fit ``int``, and CrossGL keeps
that type (``2654435761l``). Initializing a ``uint`` from it converts the value, which fits. GLSL has no implicit
conversion from ``int64_t`` to ``uint``, so ``uint[3](1, 2654435761l, 5)`` does not compile. When the value fits
the 32-bit type it initializes, the literal is spelled with that type.

The other direction needs the same spelling in aggregate constructors. ``GL_ARB_gpu_shader_int64`` adds implicit
conversions from ``int`` and ``uint`` to the 64-bit integer types, but drivers do not apply them to every operand:
Mesa accepts ``int64_t t = 0;`` and refuses ``int64_t[1](0)`` ("type error in array constructor"). Calls and
struct constructors already convert such arguments explicitly (``f(int64_t(3))``), so only the elements of an
aggregate initializer are spelled with the 64-bit type's suffix (``int64_t[1](0l)``, ``uint64_t[1](2ul)``).
"""

from __future__ import annotations

from ..ast import LiteralNode

__all__ = ["narrowed_integer_literal", "widened_integer_literal"]

_WIDE_LITERAL_TYPES = frozenset({"int64_t", "uint64_t", "int64", "uint64"})
_RANGES = {"uint": (0, 0xFFFFFFFF, "u"), "int": (-(2**31), 2**31 - 1, "")}
_WIDE_RANGES = {
    "uint64_t": (0, 2**64 - 1, "ul"),
    "int64_t": (-(2**63), 2**63 - 1, "l"),
}
_WIDE_LITERAL_SPELLING = {
    "uint64_t": {"uint64_t", "uint64"},
    "int64_t": {"int64_t", "int64"},
}


def _integer_literal(expr):
    """``(value, literal type name)`` of an integer literal, or ``None``."""
    if not isinstance(expr, LiteralNode):
        return None
    value = getattr(expr, "value", None)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    literal_type = getattr(expr, "literal_type", None)
    return value, getattr(literal_type, "name", literal_type)


def narrowed_integer_literal(generator, expr, expected_type):
    """``expr`` spelled as a ``uint``/``int`` literal when it is a 64-bit literal of a value that type holds."""
    literal = _integer_literal(expr) if expected_type is not None else None
    if literal is None:
        return None
    value, literal_type = literal
    if literal_type not in _WIDE_LITERAL_TYPES:
        return None
    target = _RANGES.get(generator.map_type(expected_type))
    if target is None:
        return None
    low, high, suffix = target
    if not low <= value <= high:
        return None
    return f"{value}{suffix}"


def widened_integer_literal(generator, expr, expected_type):
    """``expr`` spelled as an ``int64_t``/``uint64_t`` literal when it initializes that type and holds its value.

    Returns ``None`` when ``expr`` is not an integer literal, the expected type is not a 64-bit integer scalar,
    the literal already has that type, or the value is outside its range (the caller's conversion applies then).
    """
    literal = _integer_literal(expr) if expected_type is not None else None
    if literal is None:
        return None
    value, literal_type = literal
    mapped = generator.map_type(expected_type)
    target = _WIDE_RANGES.get(mapped)
    if target is None or literal_type in _WIDE_LITERAL_SPELLING[mapped]:
        return None
    low, high, suffix = target
    if not low <= value <= high:
        return None
    return f"{value}{suffix}"
