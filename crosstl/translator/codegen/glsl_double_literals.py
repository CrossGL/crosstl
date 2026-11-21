"""Floating literals that GLSL output writes where a ``double`` is required.

An unsuffixed CrossGL floating literal denotes its exact decimal value, rounded once to the type its context
requires; ``0.1f`` is the binary32 value nearest 0.1 (``ast.FLOAT_LITERAL_SUFFIX``). GLSL reads an unsuffixed
floating-point constant as a ``float`` and needs ``lf`` for a ``double`` (GLSL 4.60 section 4.1.4). GLSL output
wrote ``double(0.1)`` where a ``double`` was required: the conversion of a ``float`` constant, so ``x * 0.1`` with a
``double`` ``x`` multiplied by 0.100000001490116 on a compiler that follows the specification, and ``double(1e-300)``
was 0. glslang folds every constant in double precision, so it read ``double(0.1f)`` as 0.1, the other way round.

A floating literal, also under a unary ``-`` or ``+``, is now written as a ``double`` constant where its context
requires a ``double``: an assignment, an initializer, an operand, an argument or a return value of type ``double``;
an argument of a ``double``, ``dvecN`` or ``dmatN`` constructor; and the arms of a conditional whose arms are both
literals. An unsuffixed literal is written with its binary64 value (``0.1lf``); a suffixed one with the binary64
value of its binary32 value (``0.10000000149011612lf``), which every compiler reads alike. A literal whose value is
not finite keeps the old spelling.
"""

from __future__ import annotations

import math
import re
import struct

from ..ast import FLOAT_LITERAL_SUFFIX, LiteralNode, UnaryOpNode

__all__ = [
    "binary64_constructor_argument",
    "binary64_float_literal",
    "binary64_literal_arms",
    "is_float_literal",
]

_DOUBLE_CONSTRUCTOR = re.compile(r"double|dvec[234]|dmat[234](?:x[234])?")
_DOUBLE_AGGREGATE = re.compile(r"dvec[234]|dmat[234](?:x[234])?")
_SIGNS = frozenset({"-", "+"})


def _float_literal(expr):
    """``(sign, literal)`` of a floating literal, also under a unary ``-`` or ``+``, or ``None``."""
    sign = ""
    if isinstance(expr, UnaryOpNode) and str(getattr(expr, "operator", "")) in _SIGNS:
        sign = "-" if str(expr.operator) == "-" else ""
        expr = getattr(expr, "operand", None)
    if not isinstance(expr, LiteralNode):
        return None
    value = getattr(expr, "value", None)
    if not isinstance(value, float):
        return None
    literal_type = getattr(expr, "literal_type", None)
    if getattr(literal_type, "name", literal_type) != "float":
        return None
    return sign, expr


def is_float_literal(expr):
    """Whether ``expr`` is a floating literal, also under a unary ``-`` or ``+``."""
    return _float_literal(expr) is not None


def _double_constant(expr):
    """``expr``, a floating literal, as a GLSL ``double`` constant, or ``None`` when its value is not finite."""
    found = _float_literal(expr)
    if found is None:
        return None
    sign, literal = found
    value = literal.value
    if literal.get_annotation(FLOAT_LITERAL_SUFFIX) is not None:
        try:
            value = struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            return None
    if not math.isfinite(value):
        return None
    text = repr(value)
    if sign and text.startswith("-"):
        return f"{text[1:]}lf"
    return f"{sign}{text}lf"


def binary64_float_literal(generator, expr, expected_type):
    """``expr`` as a ``double`` constant when it is a floating literal where a ``double`` is required."""
    if expected_type is None:
        return None
    mapped = generator.map_type(expected_type)
    if mapped == "double":
        return _double_constant(expr)
    if isinstance(mapped, str) and _DOUBLE_AGGREGATE.fullmatch(mapped):
        constant = _double_constant(expr)
        return None if constant is None else f"{mapped}({constant})"
    return None


def binary64_constructor_argument(constructor, arg):
    """``arg`` as a ``double`` constant when it is a floating literal argument of a ``double`` constructor."""
    if not _DOUBLE_CONSTRUCTOR.fullmatch(str(constructor or "")):
        return None
    return _double_constant(arg)


def binary64_literal_arms(generator, expr):
    """Both arms of the conditional ``expr`` as ``double`` constants where a ``double`` is required, or ``None``."""
    expected = getattr(generator, "current_expression_expected_type", None)
    if expected is None or generator.map_type(expected) != "double":
        return None
    true_arm = _double_constant(getattr(expr, "true_expr", None))
    false_arm = _double_constant(getattr(expr, "false_expr", None))
    if true_arm is None or false_arm is None:
        return None
    return true_arm, false_arm
