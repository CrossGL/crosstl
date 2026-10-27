"""Unary minus on WGSL unsigned integers.

GLSL, HLSL, MSL, SPIR-V and CrossGL define ``-x`` for an unsigned ``x`` as the negation modulo 2^32. WGSL defines
unary ``-`` only for signed integers and floating-point values (WGSL, "Arithmetic Expressions"), so ``-a`` with a
``u32`` ``a`` is rejected. The two's-complement identity ``-x == ~x + 1`` gives the same wrapped value for ``u32``
scalars and vectors (a vector plus a scalar adds it to each component). An unsigned literal is folded instead, because
WGSL reports overflow in a constant expression.
"""

from __future__ import annotations

import re

from ..ast import LiteralNode
from .c_family_operands import prefix_unary_operand

__all__ = ["wgsl_unsigned_negation"]

_UNSIGNED_TYPE = re.compile(r"u32|vec[234]<u32>")


def wgsl_unsigned_negation(codegen, operand, rendered):
    """The WGSL spelling of ``-operand`` when ``operand`` is ``u32`` or ``vecN<u32>``, else ``None``."""
    operand_type = codegen.expression_type(operand)
    if operand_type is None:
        return None
    try:
        type_name = codegen.type_name_string(operand_type)
    except ValueError:
        return None
    if not isinstance(type_name, str) or not _UNSIGNED_TYPE.fullmatch(type_name):
        return None
    if (
        isinstance(operand, LiteralNode)
        and isinstance(operand.value, int)
        and not isinstance(operand.value, bool)
    ):
        return f"{(-operand.value) % (1 << 32)}u"
    return f"(~{prefix_unary_operand('~', rendered)} + 1u)"
