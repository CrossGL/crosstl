"""Unsuffixed floating literals in SPIR-V output: the exact decimal value, rounded once to its context's type.

CrossGL writes a binary32 literal with an ``f`` suffix (``0.1f``). An unsuffixed floating literal denotes its exact
decimal value and takes the type its context requires (``ast.FLOAT_LITERAL_SUFFIX``), as a C floating literal, an
HLSL literal float and a WGSL AbstractFloat do. The generator types such a literal ``float`` and converts it where
the context needs ``double``: ``OpFConvert`` of the binary32 constant would make ``x * 0.1`` with a ``double`` ``x``
multiply by 0.100000001490116, not by the binary64 value nearest 0.1. So the value of an unsuffixed literal, also
under a unary ``-`` or ``+``, carries the literal's value, and its conversion to ``double`` is the ``double``
constant of that value (``OpConstant %double 0.1``) instead of an ``OpFConvert``.
"""

from __future__ import annotations


from ..ast import FLOAT_LITERAL_SUFFIX
from .spirv_values import SpirvId

__all__ = [
    "exact_float_literal_value",
    "float_literal_value",
    "signed_float_literal_value",
]

_SIGN_OPERATORS = frozenset({"-", "+"})


def _carrying(value: SpirvId, literal_value: float) -> SpirvId:
    # A new SpirvId, because register_constant shares one per value and type, and a suffixed 0.1f has the same
    # constant as an unsuffixed 0.1.
    carrying = SpirvId(value.id, value.type, value.name)
    carrying.exact_float_literal = literal_value
    return carrying


def float_literal_value(literal, constant: SpirvId, literal_value: float) -> SpirvId:
    """The value of a ``float``-typed literal: ``constant``, carrying ``literal_value`` when it is unsuffixed."""
    if literal.get_annotation(FLOAT_LITERAL_SUFFIX) is not None:
        return constant
    return _carrying(constant, literal_value)


def signed_float_literal_value(op: str, operand: SpirvId, result: SpirvId) -> SpirvId:
    """The result of unary ``op`` on ``operand``, carrying the signed literal value when ``operand`` carries one."""
    literal_value = exact_float_literal_value(operand)
    if literal_value is None or op not in _SIGN_OPERATORS or result is None:
        return result
    return _carrying(result, -literal_value if op == "-" else literal_value)


def exact_float_literal_value(value) -> float | None:
    """The exact value of an unsuffixed floating literal ``value`` was lowered from, or None."""
    return getattr(value, "exact_float_literal", None)
