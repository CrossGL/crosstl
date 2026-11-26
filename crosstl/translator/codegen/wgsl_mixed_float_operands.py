"""WGSL operands of arithmetic and comparisons that mix integers and floats.

C-style languages convert an integer operand to the floating-point type when
the other operand is a float (GLSL 4.60 section 5.9, "Implicit Conversions"):
``int * 0.5`` is a float multiplication, and ``uint + float`` is a float
addition. WGSL has no implicit conversions, and an abstract float literal
next to an ``i32`` or ``u32`` operand is one naga rejects outside a constant
expression ("Abstract types may only appear in constant expressions"), so the
generator converts the integer operand to ``f32`` itself. A scalar beside a
vector converts to ``f32`` only: ``vecN<f32>(i)`` is a splat constructor naga
rejects for an integer argument, so ``v * i`` becomes ``v * f32(i)``. The
mixed-integer pass does that scalar and vector conversion (#4771); this
module covers the case it cannot see, an integer scalar beside a float
matrix, which likewise becomes ``m * f32(i)``. An abstract integer literal
needs no conversion: it takes the float operand's type, as it does for the
shift amount and the bitwise operands.
"""

from __future__ import annotations

from ..ast import LiteralNode, UnaryOpNode
from .wgsl_integer_operands import is_abstract_integer_literal

__all__ = [
    "MIXED_FLOAT_ARITHMETIC_OPERATORS",
    "MIXED_FLOAT_OPERATORS",
    "is_abstract_float_literal",
    "mixed_float_common_type",
    "wgsl_mixed_float_operands",
]

MIXED_FLOAT_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
MIXED_FLOAT_COMPARISON_OPERATORS = frozenset({"<", "<=", ">", ">=", "==", "!="})
MIXED_FLOAT_OPERATORS = (
    MIXED_FLOAT_ARITHMETIC_OPERATORS | MIXED_FLOAT_COMPARISON_OPERATORS
)


def is_abstract_float_literal(expr):
    """True if ``expr`` is an untyped float literal, an abstract float in WGSL."""
    if isinstance(expr, UnaryOpNode) and expr.operator in {"-", "+"}:
        return is_abstract_float_literal(expr.operand)
    return isinstance(expr, LiteralNode) and isinstance(expr.value, float)


def _classify(codegen, expr):
    """``(element, size)`` for a float or integer ``expr``.

    ``size`` is None for a scalar, the lane count for a vector, and
    ``"matrix"`` for a matrix, whose element type is all the conversion
    needs: only the scalar beside it converts, to ``f32``.
    """
    vtype = codegen.expression_type(expr)
    shape = codegen.vector_shape(vtype)
    if shape is not None:
        return shape
    matrix = codegen.matrix_shape(vtype)
    if matrix is not None:
        return matrix[0], "matrix"
    scalar = codegen.scalar_type_name(vtype)
    if scalar is not None:
        return scalar, None
    # An untyped literal has no type yet; its value says which kind it is.
    literal = expr
    while isinstance(literal, UnaryOpNode) and literal.operator in {"-", "+"}:
        literal = literal.operand
    if isinstance(literal, LiteralNode):
        if isinstance(literal.value, float):
            return "f32", None
        if isinstance(literal.value, int) and not isinstance(literal.value, bool):
            return "i32", None
    return None, None


def _kind(element):
    if element == "f32":
        return "float"
    if element in {"i32", "u32"}:
        return "int"
    return None


def mixed_float_common_type(codegen, left, right):
    """The ``f32`` type a mixed integer/float ``left`` and ``right`` share, else None.

    Exactly one side must be floating point and the other integer; anything
    else (both float, both integer, unknown or boolean) is left to the
    caller's other rules. The result keeps a vector's size, so a float vector
    beside an integer operand shares that vector type — though the integer
    operand itself converts only to ``f32`` (see wgsl_mixed_float_operands).
    A float matrix beside an integer scalar shares ``f32``: the scalar is
    the only side that converts.
    """
    left_element, left_size = _classify(codegen, left)
    right_element, right_size = _classify(codegen, right)
    if _kind(left_element) is None or _kind(right_element) is None:
        return None
    if _kind(left_element) == _kind(right_element):
        return None
    if left_size == "matrix" or right_size == "matrix":
        matrix_element = left_element if left_size == "matrix" else right_element
        other_size = right_size if left_size == "matrix" else left_size
        if _kind(matrix_element) == "float" and other_size is None:
            return "f32"
        return None
    if left_size is not None and right_size is not None and left_size != right_size:
        return None
    size = left_size if left_size is not None else right_size
    return f"vec{size}<f32>" if size is not None else "f32"


def wgsl_mixed_float_operands(
    codegen, left, right, left_rendered=None, right_rendered=None
):
    """The WGSL spellings of mixed integer/float operands.

    The mixed-integer pass already converts scalar and vector integer
    operands to the float type (#4771), so this function covers the one case
    it cannot see: an integer scalar beside a float matrix, which becomes
    ``f32`` beside the untouched matrix (``m * f32(i)``). An abstract integer
    literal is left alone, since it takes the float operand's type.
    ``left_rendered`` and ``right_rendered`` let a caller that already
    rendered the operands (for the mixed signedness conversion) reuse them.
    """
    common = mixed_float_common_type(codegen, left, right)
    if left_rendered is None:
        left_rendered = codegen.generate_expression(left)
    if right_rendered is None:
        right_rendered = codegen.generate_expression(right)
    if common is None:
        return left_rendered, right_rendered
    left_element, left_size = _classify(codegen, left)
    right_element, right_size = _classify(codegen, right)
    if left_size != "matrix" and right_size != "matrix":
        # Scalar and vector operands are already converted by the
        # mixed-integer pass; converting again would double-wrap them
        # (``f32(f32(a))``).
        return left_rendered, right_rendered
    if _kind(left_element) == "int" and not is_abstract_integer_literal(left):
        target = "f32" if left_size is None else common
        left_rendered = f"{target}({left_rendered})"
    if _kind(right_element) == "int" and not is_abstract_integer_literal(right):
        target = "f32" if right_size is None else common
        right_rendered = f"{target}({right_rendered})"
    return left_rendered, right_rendered
