"""Fold vector CrossGL constant expressions component by component.

This extends :mod:`scalar_constant_folding` to the vector values a module-scope
constant can hold: constructors (``vec3(v.xy, 1.0)``, ``ivec2(3)``),
swizzles and constant indexes (``v.zyx``, ``v[1]``), component-wise operators
with scalar broadcast (``v * 2.0``, ``-v``), conversions and the component-wise
math built-ins, plus ``dot``, ``cross``, ``length``, ``distance``,
``normalize``, ``any``, ``all`` and ``not``.

Every component is evaluated with the scalar folder's arithmetic, so binary32
values are rounded after each operation. Vector comparisons and logical
operators are not folded, because the source languages disagree on whether
they yield one ``bool`` or a ``bool`` vector; the caller sees ``None`` and must
refuse the construct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

from ..ast import ArrayAccessNode, CastNode, MemberAccessNode, SwizzleNode
from .scalar_constant_folding import (
    FLOAT_KINDS,
    INTEGER_KINDS,
    KindResolver,
    ScalarConstant,
    ScalarConstantFolder,
    common_kind,
    convert_scalar_constant,
)


@dataclass(frozen=True)
class VectorConstant:
    """A folded vector: its component kind and its components, in order."""

    kind: str
    components: tuple[ScalarConstant, ...]

    @property
    def size(self) -> int:
        return len(self.components)


FoldedConstant = Union[ScalarConstant, VectorConstant]
VectorShapeResolver = Callable[[object], Optional[tuple[str, int]]]

_SWIZZLE_SETS = ("xyzw", "rgba", "stpq")

# Operators applied to each component; the others are not folded for vectors.
_COMPONENT_WISE_BINARY = frozenset({"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"})
_COMPONENT_WISE_UNARY = frozenset({"+", "-", "~", "!"})


def swizzle_indices(member: object, size: int) -> tuple[int, ...] | None:
    """Return the component indexes named by swizzle *member*, or ``None``."""
    if not isinstance(member, str) or not 1 <= len(member) <= 4:
        return None
    for letters in _SWIZZLE_SETS:
        if all(letter in letters for letter in member):
            indices = tuple(letters.index(letter) for letter in member)
            return indices if max(indices) < size else None
    return None


def make_vector(components) -> VectorConstant | None:
    """Return a vector of *components* (all of one kind), or ``None``."""
    components = tuple(components)
    if not 2 <= len(components) <= 4 or any(c is None for c in components):
        return None
    kind = components[0].kind
    if any(component.kind != kind for component in components):
        return None
    return VectorConstant(kind, components)


def convert_vector_constant(
    constant: VectorConstant, kind: str
) -> VectorConstant | None:
    """Convert each component of *constant* to *kind*."""
    if constant.kind == kind:
        return constant
    return make_vector(convert_scalar_constant(c, kind) for c in constant.components)


def _components(value: FoldedConstant, size: int) -> tuple[ScalarConstant, ...] | None:
    """Return *value* as *size* components, broadcasting a scalar."""
    if isinstance(value, ScalarConstant):
        return (value,) * size
    return value.components if value.size == size else None


def _common_size(values) -> int | None:
    sizes = {value.size for value in values if isinstance(value, VectorConstant)}
    if len(sizes) != 1:
        return None
    return sizes.pop()


class VectorConstantFolder(ScalarConstantFolder):
    """Evaluate scalar and vector constant expressions.

    *vector_shape* maps a type (a name or a type node) to ``(kind, size)`` for
    vector types and ``None`` otherwise. *resolve_name* may return either a
    :class:`ScalarConstant` or a :class:`VectorConstant`.
    """

    def __init__(
        self,
        resolve_name: Callable[[str], FoldedConstant | None],
        kind_of_type: KindResolver,
        vector_shape: VectorShapeResolver,
        *,
        fold_builtin=None,
    ):
        super().__init__(resolve_name, kind_of_type, fold_builtin=fold_builtin)
        self.vector_shape = vector_shape

    def fold(self, expr) -> FoldedConstant | None:
        if isinstance(expr, MemberAccessNode):
            return self.fold_swizzle(expr.object_expr, expr.member)
        if isinstance(expr, SwizzleNode):
            return self.fold_swizzle(expr.vector_expr, expr.components)
        if isinstance(expr, ArrayAccessNode):
            return self.fold_index(expr)
        if isinstance(expr, CastNode):
            operand = self.fold(expr.expression)
            if operand is None:
                return None
            shape = self.vector_shape(expr.target_type)
            if shape is not None:
                return self.construct(shape, [operand])
            kind = self.kind_of_type(expr.target_type)
            return None if kind is None else self.construct_scalar(kind, [operand])
        return super().fold(expr)

    def fold_name(self, name: str | None) -> FoldedConstant | None:
        if name and name not in {"true", "false"}:
            constant = self.resolve_name(name)
            if isinstance(constant, VectorConstant):
                return constant
        return super().fold_name(name)

    def fold_swizzle(self, operand_expr, member) -> FoldedConstant | None:
        operand = self.fold(operand_expr)
        return None if operand is None else self.member_value(operand, member)

    def member_value(self, operand, member) -> FoldedConstant | None:
        """Return the swizzle or member *member* of a folded operand."""
        if not isinstance(operand, VectorConstant):
            return None
        indices = swizzle_indices(member, operand.size)
        if indices is None:
            return None
        if len(indices) == 1:
            return operand.components[indices[0]]
        return make_vector(operand.components[index] for index in indices)

    def fold_index(self, node: ArrayAccessNode) -> FoldedConstant | None:
        operand = self.fold(node.array_expr)
        index = self.fold(node.index_expr)
        if operand is None or not isinstance(index, ScalarConstant):
            return None
        if index.kind not in INTEGER_KINDS:
            return None
        return self.index_value(operand, index.value)

    def index_value(self, operand, index: int) -> FoldedConstant | None:
        """Return element *index* of a folded operand, or ``None`` out of range."""
        if not isinstance(operand, VectorConstant) or not 0 <= index < operand.size:
            return None
        return operand.components[index]

    def unary_value(self, operator, operand):
        if not isinstance(operand, VectorConstant):
            return super().unary_value(operator, operand)
        if operator not in _COMPONENT_WISE_UNARY:
            return None
        return make_vector(
            super(VectorConstantFolder, self).unary_value(operator, component)
            for component in operand.components
        )

    def binary_value(self, operator, left, right):
        if isinstance(left, ScalarConstant) and isinstance(right, ScalarConstant):
            return super().binary_value(operator, left, right)
        if operator not in _COMPONENT_WISE_BINARY:
            return None
        size = _common_size((left, right))
        if size is None:
            return None
        left_components = _components(left, size)
        right_components = _components(right, size)
        if left_components is None or right_components is None:
            return None
        return make_vector(
            super(VectorConstantFolder, self).binary_value(operator, a, b)
            for a, b in zip(left_components, right_components)
        )

    def ternary_value(self, condition, true_value, false_value):
        if not isinstance(condition, ScalarConstant):
            return None
        if isinstance(true_value, ScalarConstant) and isinstance(
            false_value, ScalarConstant
        ):
            return super().ternary_value(condition, true_value, false_value)
        if not (
            isinstance(true_value, VectorConstant)
            and isinstance(false_value, VectorConstant)
            and true_value.size == false_value.size
        ):
            return None
        kind = common_kind(true_value.kind, false_value.kind, arithmetic=False)
        selected = convert_scalar_constant(condition, "bool").value
        return convert_vector_constant(true_value if selected else false_value, kind)

    def call_value(self, name, arguments):
        shape = self.vector_shape(name)
        if shape is not None:
            return self.construct(shape, arguments)
        kind = self.kind_of_type(name)
        if kind is not None:
            return self.construct_scalar(kind, arguments)
        if all(isinstance(argument, ScalarConstant) for argument in arguments):
            return super().call_value(name, arguments)
        geometric = self.geometric_value(name, arguments)
        if geometric is not None:
            return geometric
        return self.component_wise_call(name, arguments)

    def construct_scalar(self, kind: str, arguments) -> ScalarConstant | None:
        """Fold a scalar constructor; a vector argument yields its first component."""
        if len(arguments) != 1:
            return None
        (argument,) = arguments
        if isinstance(argument, VectorConstant):
            argument = argument.components[0]
        return convert_scalar_constant(argument, kind)

    def construct(self, shape: tuple[str, int], arguments) -> VectorConstant | None:
        """Fold a vector constructor of *shape* from *arguments*."""
        kind, size = shape
        if not arguments:
            return None
        if len(arguments) == 1:
            (argument,) = arguments
            if isinstance(argument, ScalarConstant):
                components = (argument,) * size
            elif argument.size >= size:
                components = argument.components[:size]
            else:
                return None
        else:
            components = []
            for argument in arguments:
                if isinstance(argument, ScalarConstant):
                    components.append(argument)
                else:
                    components.extend(argument.components)
            # Every argument must be consumed, and exactly fill the vector.
            if len(components) != size:
                return None
        return make_vector(convert_scalar_constant(c, kind) for c in components)

    def component_wise_call(self, name, arguments) -> VectorConstant | None:
        if self.fold_builtin is None:
            return None
        size = _common_size(arguments)
        if size is None:
            return None
        columns = [_components(argument, size) for argument in arguments]
        if any(column is None for column in columns):
            return None
        return make_vector(
            self.fold_builtin(name, [column[index] for column in columns])
            for index in range(size)
        )

    def geometric_value(self, name, arguments) -> FoldedConstant | None:
        """Fold the built-ins that combine components instead of mapping them."""
        if name in {"any", "all"} and len(arguments) == 1:
            (vector,) = arguments
            if vector.kind != "bool":
                return None
            values = [component.value for component in vector.components]
            return ScalarConstant("bool", any(values) if name == "any" else all(values))
        if name == "not" and len(arguments) == 1:
            (vector,) = arguments
            if vector.kind != "bool":
                return None
            return self.unary_value("!", vector)
        vectors = [a for a in arguments if isinstance(a, VectorConstant)]
        if len(vectors) != len(arguments) or any(
            v.kind not in FLOAT_KINDS for v in vectors
        ):
            return None
        if name == "dot" and len(vectors) == 2:
            return self.dot(*vectors)
        if name == "length" and len(vectors) == 1:
            return self.length(vectors[0])
        if name == "distance" and len(vectors) == 2:
            difference = self.binary_value("-", *vectors)
            return None if difference is None else self.length(difference)
        if name == "normalize" and len(vectors) == 1:
            length = self.length(vectors[0])
            if length is None or length.value == 0.0 or not math.isfinite(length.value):
                return None
            return self.binary_value("/", vectors[0], length)
        if name == "cross" and len(vectors) == 2:
            return self.cross(*vectors)
        return None

    def dot(self, left: VectorConstant, right: VectorConstant) -> ScalarConstant | None:
        products = self.binary_value("*", left, right)
        if products is None:
            return None
        total = products.components[0]
        for component in products.components[1:]:
            total = self.binary_value("+", total, component)
            if total is None:
                return None
        return total

    def length(self, vector: VectorConstant) -> ScalarConstant | None:
        squared = self.dot(vector, vector)
        if squared is None or self.fold_builtin is None:
            return None
        return self.fold_builtin("sqrt", [squared])

    def cross(
        self, left: VectorConstant, right: VectorConstant
    ) -> VectorConstant | None:
        if left.size != 3 or right.size != 3:
            return None
        a, b = left.components, right.components

        def term(i: int, j: int) -> ScalarConstant | None:
            first = self.binary_value("*", a[i], b[j])
            second = self.binary_value("*", a[j], b[i])
            if first is None or second is None:
                return None
            return self.binary_value("-", first, second)

        return make_vector((term(1, 2), term(2, 0), term(0, 1)))


__all__ = (
    "FoldedConstant",
    "VectorConstant",
    "VectorConstantFolder",
    "convert_vector_constant",
    "make_vector",
    "swizzle_indices",
)
