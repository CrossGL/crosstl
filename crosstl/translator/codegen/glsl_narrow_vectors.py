"""GLSL arithmetic on 8- and 16-bit integers.

The GLSL emitters keep CrossGL 8- and 16-bit integer scalars and vectors in 32-bit carriers (``int``/``uint`` and
``ivecN``/``uvecN``) and compute arithmetic on promoted 32-bit operands. A 32-bit carrier does not wrap by itself,
so a narrow binary result consumed without first being stored into a narrow variable (such as ``float(v * v)`` on
``int8_t v = 13``, or ``(n + m).x`` on ``u16vec2``) would otherwise keep 32-bit precision instead of wrapping at
8 or 16 bits.

An operation whose result keeps a narrow integer type, and whose operator can leave the operand range, is reduced
modulo 2^N with the emitter's narrow-integer contract (``bitfieldExtract`` or, for GLSL ES 3.00, masks and
shifts). The reduced value keeps the 32-bit type the emitter already reports for the expression, so conversions
around it are unchanged.

Scalar operations keep their narrow type like the explicit arithmetic types of GLSL, HLSL and SPIR-V, except when
the translation's source promotes narrow scalar operands to ``int`` first (C++ [conv.prom], C99 6.3.1.1): there a
scalar result is computed exactly in 32 bits. Vector operations keep their narrow type for every source (MSL
section 2.2 and OpenCL C 6.3 do not promote vectors).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ast import BinaryOpNode, LiteralNode, TernaryOpNode, UnaryOpNode

__all__ = [
    "GLSLNarrowVectorMixin",
    "NarrowShape",
    "narrow_scalars_promote",
    "narrow_vector_binary_result",
    "narrow_vector_unary_result",
]

# Sources whose narrow scalar operands are promoted to int before arithmetic, as recorded by rule carrying.
SCALAR_PROMOTING_SOURCES = frozenset({"metal", "cuda", "hip", "opencl"})

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


def narrow_scalars_promote(program):
    """Whether the source of ``program`` promotes narrow scalar operands, from its rule-carrying annotation."""
    rules = (getattr(program, "annotations", None) or {}).get("rule_carrying") or {}
    return rules.get("source") in SCALAR_PROMOTING_SOURCES


def narrow_vector_binary_result(
    operator, left, right, *, left_literal, right_literal, scalars_promote=False
):
    """The narrow integer type of ``left operator right`` in the GLSL emitters, else ``None``.

    ``left`` and ``right`` are the operands' :class:`NarrowShape` or ``None`` when they are not narrow integers.
    Same-type narrow binary operations (scalar or vector) keep their narrow width, a vector shift keeps the type
    of its left operand, and an integer literal adopts the narrow type of a vector operand while promoting a
    narrow scalar to 32 bits. With ``scalars_promote`` no scalar result is narrow.
    """
    if operator not in _TYPED_BINARY:
        return None
    if operator in _SHIFTS:
        if left is None or (left.lanes == 1 and (scalars_promote or right != left)):
            return None
        return left
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
    return None if scalars_promote and result.lanes == 1 else result


def narrow_vector_unary_result(operator, operand, *, scalars_promote=False):
    """The narrow integer type of ``operator operand`` in the GLSL emitters, else ``None``."""
    if operator not in _TYPED_UNARY or operand is None:
        return None
    return None if scalars_promote and operand.lanes == 1 else operand


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
    """Narrow integer wrapping of ``GLSLCodeGen``."""

    # Set per program from :func:`narrow_scalars_promote`.
    glsl_narrow_scalars_promote = False

    def glsl_narrow_value_shape(self, expr):
        """The CrossGL narrow integer type of ``expr``, following narrow arithmetic results."""
        if isinstance(expr, BinaryOpNode):
            return self.glsl_narrow_vector_binary_shape(expr)
        if isinstance(expr, UnaryOpNode):
            return narrow_vector_unary_result(
                self.map_operator(expr.op),
                self.glsl_narrow_value_shape(expr.operand),
                scalars_promote=self.glsl_narrow_scalars_promote,
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
            scalars_promote=self.glsl_narrow_scalars_promote,
        )

    def glsl_narrow_vector_wrap(self, expr, operator, rendered):
        """Reduce a narrow integer arithmetic result modulo 2^N, keeping its emitted 32-bit type."""
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
