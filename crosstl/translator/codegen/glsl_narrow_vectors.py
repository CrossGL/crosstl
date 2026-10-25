"""GLSL arithmetic on 8- and 16-bit integer vectors.

The GLSL emitters keep a CrossGL ``u16vec2`` value in a 32-bit carrier (``uvec2``) and compute arithmetic on the
promoted 32-bit operands (``ivec2(n) + ivec2(m)``). CrossGL integer promotion applies to scalars only: vector
arithmetic keeps its narrow type and wraps at that width (MSL section 2.2 and OpenCL C 6.3 do not promote vectors,
and the explicit 16-bit types of GLSL, HLSL and SPIR-V do not promote at all; ``semantics/narrow.py`` legalizes the
same rule for SPIR-V and WGSL). A 32-bit carrier does not wrap by itself, so ``(n + m).x`` could exceed 65535.

An operation whose CrossGL result is a narrow vector, and whose operator can leave the operand range, is reduced
modulo 2^N with the emitter's narrow-integer contract (``bitfieldExtract`` or, for GLSL ES 3.00, masks and
shifts). The reduced value keeps the 32-bit type the emitter already reports for the expression, so conversions
around it are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ast import BinaryOpNode, LiteralNode, TernaryOpNode, UnaryOpNode

__all__ = [
    "GLSLNarrowVectorMixin",
    "NarrowShape",
    "narrow_vector_binary_result",
    "narrow_vector_unary_result",
]

# Operators whose result can leave the operand range (``semantics/narrow.py`` uses the same set).
RANGE_CHANGING_BINARY = frozenset({"+", "-", "*", "/", "%", "<<"})
RANGE_CHANGING_UNARY = frozenset({"-", "~"})
# Operators whose result keeps the type of a narrow vector operand.
_TYPED_BINARY = RANGE_CHANGING_BINARY | {"&", "|", "^", ">>"}
_SHIFTS = frozenset({"<<", ">>"})
_TYPED_UNARY = RANGE_CHANGING_UNARY | {"+"}


@dataclass(frozen=True)
class NarrowShape:
    """An 8- or 16-bit integer scalar or vector type."""

    signed: bool
    bits: int
    lanes: int

    @property
    def type_name(self) -> str:
        prefix = "i" if self.signed else "u"
        if self.lanes == 1:
            return f"{'' if self.signed else 'u'}int{self.bits}_t"
        return f"{prefix}{self.bits}vec{self.lanes}"


def narrow_vector_binary_result(operator, left, right, *, left_literal, right_literal):
    """The narrow vector type of ``left operator right`` under CrossGL rules, else ``None``.

    ``left`` and ``right`` are the operands' :class:`NarrowShape` or ``None`` when they are not narrow integers.
    Vectors keep their narrow type, scalars promote to ``int``, a shift has the type of its left operand, and only
    an integer literal adopts the narrow type of the other operand.
    """
    if operator not in _TYPED_BINARY:
        return None
    if operator in _SHIFTS:
        return left if left is not None and left.lanes > 1 else None
    if left is None and right is None:
        return None
    if left is None or right is None:
        known = left if right is None else right
        if not (right_literal if right is None else left_literal):
            return None
        return known if known.lanes > 1 else None
    if (left.signed, left.bits) != (right.signed, right.bits):
        return None
    if left.lanes != right.lanes and 1 not in (left.lanes, right.lanes):
        return None
    result = left if left.lanes >= right.lanes else right
    return result if result.lanes > 1 else None


def narrow_vector_unary_result(operator, operand):
    """The narrow vector type of ``operator operand`` under CrossGL rules, else ``None``."""
    if operator not in _TYPED_UNARY or operand is None or operand.lanes == 1:
        return None
    return operand


def _is_integer_literal(expr):
    if not isinstance(expr, LiteralNode):
        return False
    value = getattr(expr, "value", None)
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and value.rstrip("uU").isdigit()


class GLSLNarrowVectorMixin:
    """Narrow integer vector wrapping of ``GLSLCodeGen``."""

    def glsl_narrow_value_shape(self, expr):
        """The CrossGL narrow integer type of ``expr``, following narrow arithmetic results."""
        if isinstance(expr, BinaryOpNode):
            return self.glsl_narrow_vector_binary_shape(expr)
        if isinstance(expr, UnaryOpNode):
            return narrow_vector_unary_result(
                self.map_operator(expr.op), self.glsl_narrow_value_shape(expr.operand)
            )
        if isinstance(expr, TernaryOpNode):
            arms = {
                self.glsl_narrow_value_shape(expr.true_expr),
                self.glsl_narrow_value_shape(expr.false_expr),
            }
            return arms.pop() if len(arms) == 1 else None
        contract = self.glsl_narrow_integer_contract(
            self.glsl_source_expression_type(expr)
        )
        if contract is None:
            return None
        return NarrowShape(
            bool(contract["signed"]), int(contract["bits"]), int(contract["width"])
        )

    def glsl_narrow_vector_binary_shape(self, expr):
        return narrow_vector_binary_result(
            self.map_operator(expr.op),
            self.glsl_narrow_value_shape(expr.left),
            self.glsl_narrow_value_shape(expr.right),
            left_literal=_is_integer_literal(expr.left),
            right_literal=_is_integer_literal(expr.right),
        )

    def glsl_narrow_vector_wrap(self, expr, operator, rendered):
        """Reduce a narrow vector arithmetic result modulo 2^N, keeping its emitted 32-bit type."""
        if isinstance(expr, UnaryOpNode):
            if operator not in RANGE_CHANGING_UNARY:
                return rendered
        elif operator not in RANGE_CHANGING_BINARY:
            return rendered
        shape = self.glsl_narrow_value_shape(expr)
        if shape is None:
            return rendered
        carrier = self.map_type(shape.type_name)
        emitted = self.glsl_value_type_info(self.glsl_source_expression_type(expr))
        if (
            emitted is None
            or emitted["family"] not in {"int", "uint"}
            or emitted["width"] != shape.lanes
        ):
            raise self.GLSL_ARITHMETIC_CONVERSION_ERROR(
                f"{self.GLSL_TARGET_DISPLAY_NAME} cannot wrap the {shape.type_name} result of "
                f"operator '{operator}' at {shape.bits} bits: the emitted type is unknown",
                operator=operator,
                operand_types=(shape.type_name, shape.type_name),
                attempted_common_type=None,
                reason="narrow-vector-result-type-unresolved",
                source_location=getattr(expr, "source_location", None),
            )
        mapped = emitted["mapped"]
        value = rendered if mapped == carrier else f"{carrier}({rendered})"
        wrapped = self.glsl_apply_narrow_integer_contract(value, shape.type_name)
        result = wrapped if mapped == carrier else f"{mapped}({wrapped})"
        forms = self.__dict__.setdefault("_glsl_narrow_vector_carrier_forms", {})
        forms[result] = (shape, wrapped)
        return result

    def glsl_narrow_vector_carrier_form(self, generated, expected_type):
        """The carrier spelling of a result this mixin already wrapped at the width of ``expected_type``.

        A store into a narrow vector normalizes its value again; when the value is a wrapped result of the same
        type, the normalization is the identity, so the store reuses the wrapped carrier value instead.
        """
        entry = self.__dict__.get("_glsl_narrow_vector_carrier_forms", {}).get(
            generated
        )
        if entry is None:
            return None
        shape, wrapped = entry
        contract = self.glsl_narrow_integer_contract(expected_type)
        if contract is None or (
            bool(contract["signed"]),
            int(contract["bits"]),
            int(contract["width"]),
        ) != (shape.signed, shape.bits, shape.lanes):
            return None
        return wrapped
