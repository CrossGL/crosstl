"""WGSL operands of integer shift and bitwise operators.

HLSL, GLSL, MSL and CrossGL accept a scalar shift amount for a vector, and a signed amount. A WGSL shift instead takes
an amount of type ``u32``, or ``vecN<u32>`` when the shifted value is a ``vecN`` (WGSL, "Bit Expressions"). Both
languages use only the low bits of the amount (HLSL and DXIL mask it to the operand width, WGSL takes it modulo the
width), so converting a signed amount to ``u32`` keeps those bits.

WGSL bitwise ``&``, ``|`` and ``^`` need both operands of one type, while the C-style languages replicate a scalar
across a vector operand. A scalar whose type already matches the vector's components, or an untyped literal, is
replicated; any other mix keeps its spelling, because converting it would change the result type that C-style
promotion picks.
"""

from __future__ import annotations

from ..ast import LiteralNode, UnaryOpNode

__all__ = [
    "BITWISE_OPERATORS",
    "SHIFT_OPERATORS",
    "wgsl_bitwise_operands",
    "wgsl_shift_amount",
]

SHIFT_OPERATORS = frozenset({"<<", ">>"})
BITWISE_OPERATORS = frozenset({"&", "|", "^"})


def _type_name(codegen, vtype):
    if vtype is None:
        return None
    try:
        return codegen.type_name_string(vtype)
    except ValueError:
        return None


def _is_integer_literal(expr):
    if isinstance(expr, UnaryOpNode) and expr.operator in {"-", "+"}:
        return _is_integer_literal(expr.operand)
    return isinstance(expr, LiteralNode) and isinstance(expr.value, int)


def _is_non_negative_integer_literal(expr):
    return (
        isinstance(expr, LiteralNode)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
        and expr.value >= 0
    )


def wgsl_shift_amount(codegen, amount, shifted_type, rendered=None):
    """The WGSL spelling of shift ``amount`` for a value of ``shifted_type``."""
    if rendered is None:
        rendered = codegen.generate_expression(amount)
    shifted_shape = codegen.vector_shape(shifted_type)
    amount_type = codegen.expression_type(amount)
    amount_shape = codegen.vector_shape(amount_type)
    if amount_shape is not None:
        if amount_shape[0] == "u32":
            return rendered
        return f"vec{amount_shape[1]}<u32>({rendered})"
    if _is_non_negative_integer_literal(amount) or _type_name(codegen, amount_type) in {
        "u32",
        None,
    }:
        scalar = rendered
    else:
        scalar = f"u32({rendered})"
    if shifted_shape is None:
        return scalar
    return f"vec{shifted_shape[1]}<u32>({scalar})"


def _replicated(codegen, scalar_expr, rendered, vector_type):
    shape = codegen.vector_shape(vector_type)
    if shape is None:
        return None
    scalar_type = _type_name(codegen, codegen.expression_type(scalar_expr))
    if scalar_type != shape[0] and not _is_integer_literal(scalar_expr):
        return None
    return f"{codegen.type_name_string(vector_type)}({rendered})"


def wgsl_bitwise_operands(codegen, left, right):
    """The WGSL spellings of the operands of a bitwise ``&``, ``|`` or ``^``."""
    left_rendered = codegen.generate_expression(left)
    right_rendered = codegen.generate_expression(right)
    left_type = codegen.expression_type(left)
    right_type = codegen.expression_type(right)
    left_vector = codegen.vector_shape(left_type) is not None
    right_vector = codegen.vector_shape(right_type) is not None
    if left_vector and not right_vector and right_type is not None:
        replicated = _replicated(codegen, right, right_rendered, left_type)
        if replicated is not None:
            right_rendered = replicated
    elif right_vector and not left_vector and left_type is not None:
        replicated = _replicated(codegen, left, left_rendered, right_type)
        if replicated is not None:
            left_rendered = replicated
    return left_rendered, right_rendered
