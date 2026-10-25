"""HLSL native 16-bit integer arithmetic under the CrossGL promotion rules.

CrossGL follows the MSL rules for 8- and 16-bit integers (MSL section 2.12, C++ [conv.prom] and [expr.shift]):

- a scalar operand is promoted to ``int`` before arithmetic, so ``-a`` and ``~a`` of a ``uint16_t`` are ``int``;
- a vector operand is not promoted, so ``n << k`` of a ``u16vec2`` keeps the ``u16vec2`` type and wraps at 16 bits,
  whatever the type of the shift count.

HLSL with ``-enable-16bit-types`` promotes nothing: ``-a`` stays ``uint16_t`` and wraps, and a shift keeps the
type of its left operand. The emitter therefore promotes a narrow scalar operand of ``-`` and ``~`` explicitly, and
computes a narrow vector shift natively instead of on ``int2`` operands, which never wrapped.
"""

from __future__ import annotations

__all__ = ["HLSLNarrowIntegerMixin"]

# Exact HLSL 16-bit integer types; min16int and min16uint may compute in 32 bits and are not narrow here.
NATIVE_NARROW_INTEGER_BASES = frozenset({"int16_t", "uint16_t"})
# Unary operators that promote a narrow scalar operand to int (C++ [expr.unary.op]).
PROMOTING_UNARY_OPERATORS = frozenset({"-", "~"})


class HLSLNarrowIntegerMixin:
    """Narrow integer promotion rules of ``HLSLCodeGen``."""

    def hlsl_native_narrow_integer_info(self, vtype):
        """Arithmetic type info of an exact 16-bit integer scalar or vector type, else ``None``."""
        info = self.hlsl_integer_arithmetic_type_info(vtype)
        if info is None or info["base_type"] not in NATIVE_NARROW_INTEGER_BASES:
            return None
        if not getattr(self, "requires_hlsl_native_16_bit_types", False):
            return None
        return info

    def hlsl_narrow_scalar_unary_operand(self, expr, operator):
        """The narrow scalar info of a promoting unary operation's operand, else ``None``."""
        if operator not in PROMOTING_UNARY_OPERATORS or getattr(
            expr, "is_postfix", False
        ):
            return None
        operand = getattr(expr, "operand", None)
        if operand is None:
            return None
        info = self.hlsl_native_narrow_integer_info(
            self.expression_result_type(operand)
        )
        if info is None or info["width"] != 1:
            return None
        return info

    def hlsl_narrow_scalar_unary_result_type(self, expr):
        """``int`` for ``-a`` and ``~a`` of a narrow integer scalar, else ``None``."""
        operator = self.map_operator(getattr(expr, "op", None))
        if self.hlsl_narrow_scalar_unary_operand(expr, operator) is None:
            return None
        return "int"

    def hlsl_narrow_scalar_unary_expression(self, expr, operand, operator):
        """``-int(a)`` and ``~int(a)`` for a narrow integer scalar ``a``, else ``None``."""
        if self.hlsl_narrow_scalar_unary_operand(expr, operator) is None:
            return None
        return f"{operator}int({operand})"

    @staticmethod
    def hlsl_narrow_vector_shift_bases(left_integer, right_integer):
        """The left and right operation bases of a shift of a narrow integer vector, else ``None``.

        The shift keeps the narrow vector type of its left operand and the count keeps its own type: HLSL, like
        MSL and GLSL, gives a shift the type of its left operand without converting the operands to a common type.
        """
        if left_integer is None or right_integer is None:
            return None
        if (
            left_integer["width"] == 1
            or left_integer["base_type"] not in NATIVE_NARROW_INTEGER_BASES
        ):
            return None
        return left_integer["base_type"], right_integer["base_type"]
