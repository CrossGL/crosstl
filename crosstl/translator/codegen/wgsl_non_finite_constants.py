"""WGSL output of a float whose bits are a constant NaN or infinity pattern.

CrossGL rules and imported sources spell NaN and infinity as bit casts of integer constants, such as
HLSL ``asfloat(0x7fc00000u)``. In WGSL, ``bitcast<f32>(2143289344u)`` is a const-expression, and a
const-expression whose value is not a finite ``f32`` is a shader-creation error (WGSL, "Floating Point
Evaluation" and "const-expressions"): Tint rejects it with ``value nan cannot be represented as 'f32'``.
A call of a user-declared function is never a const-expression, so the bit cast moves into a helper
function and the value is made at run time, with the same bits.
"""

from __future__ import annotations

from ..ast import LiteralNode

HELPER_NAME = "cgl_f32_from_bits"
HELPER = f"fn {HELPER_NAME}(bits: u32) -> f32 {{\n    return bitcast<f32>(bits);\n}}"
EXPONENT_MASK = 0x7F800000


def constant_integer_value(codegen, expr):
    """The value of an integer literal or integer constant ``expr``, else ``None``."""
    if isinstance(expr, LiteralNode) and isinstance(expr.value, str):
        text = expr.value.strip().rstrip("uUlL")
        try:
            return int(text, 0)
        except ValueError:
            return None
    try:
        return codegen.resolve_integer_expression(expr)
    except (TypeError, ValueError):
        return None


def is_non_finite_f32_bits(value):
    """Whether the 32-bit pattern ``value`` is a NaN or an infinity."""
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return (value & 0xFFFFFFFF) & EXPONENT_MASK == EXPONENT_MASK


def wgsl_runtime_f32_from_bits(codegen, rendered):
    """``rendered``, a ``u32`` or ``i32`` value, reinterpreted as ``f32`` through the run-time helper."""
    helpers = codegen._half_conversion_helpers
    if HELPER_NAME not in helpers:
        helpers[HELPER_NAME] = HELPER
    return f"{HELPER_NAME}(bitcast<u32>({rendered}))"
