"""Fold element and member access of array and struct CrossGL constants.

Module-scope constants such as ``const float A[3] = {1.0, 2.0, 3.0}`` or
``const Light L = Light(1.0, vec3(0.5))`` are aggregates. Other constants may
read them (``const float X = A[1]``, ``const vec3 C = L.colour * 2.0``), and a
target such as SPIR-V needs the value of such an initializer during
translation.

:class:`AggregateConstant` holds the folded elements of an array, or the
members of a struct together with their names. Only element and member
access are folded: an operator, a conversion or a call that takes a whole
aggregate yields ``None``, and the caller must refuse the construct.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ast import CastNode
from .matrix_constant_folding import MatrixConstantFolder


@dataclass(frozen=True)
class AggregateConstant:
    """A folded array (``member_names`` is ``None``) or struct value."""

    elements: tuple
    member_names: tuple[str, ...] | None = None

    @property
    def is_struct(self) -> bool:
        return self.member_names is not None


def _has_aggregate(values) -> bool:
    return any(isinstance(value, AggregateConstant) for value in values)


class AggregateConstantFolder(MatrixConstantFolder):
    """Evaluate constant expressions that may index arrays or read struct members.

    *resolve_name* may return an :class:`AggregateConstant` as well as a
    scalar, vector or matrix.
    """

    def fold_name(self, name):
        if name and name not in {"true", "false"}:
            constant = self.resolve_name(name)
            if isinstance(constant, AggregateConstant):
                return constant
        return super().fold_name(name)

    def member_value(self, operand, member):
        if not isinstance(operand, AggregateConstant):
            return super().member_value(operand, member)
        if not operand.is_struct or member not in operand.member_names:
            return None
        return operand.elements[operand.member_names.index(member)]

    def index_value(self, operand, index: int):
        if not isinstance(operand, AggregateConstant):
            return super().index_value(operand, index)
        if operand.is_struct or not 0 <= index < len(operand.elements):
            return None
        return operand.elements[index]

    def unary_value(self, operator, operand):
        if isinstance(operand, AggregateConstant):
            return None
        return super().unary_value(operator, operand)

    def binary_value(self, operator, left, right):
        if _has_aggregate((left, right)):
            return None
        return super().binary_value(operator, left, right)

    def ternary_value(self, condition, true_value, false_value):
        if _has_aggregate((condition, true_value, false_value)):
            return None
        return super().ternary_value(condition, true_value, false_value)

    def call_value(self, name, arguments):
        if _has_aggregate(arguments):
            return None
        return super().call_value(name, arguments)

    def fold(self, expr):
        # A conversion of a whole aggregate has no folded value.
        if isinstance(expr, CastNode) and isinstance(
            self.fold(expr.expression), AggregateConstant
        ):
            return None
        return super().fold(expr)


__all__ = ("AggregateConstant", "AggregateConstantFolder")
