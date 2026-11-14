"""Fold matrix CrossGL constant expressions.

This extends :mod:`vector_constant_folding` to the matrices a module-scope
constant can hold. Matrices are column-major, as in GLSL and SPIR-V: ``m[i]`` is
column ``i``, and ``m * v`` treats ``v`` as a column vector, so a folded value
equals what ``OpMatrixTimesVector``, ``OpVectorTimesMatrix`` and
``OpMatrixTimesMatrix`` compute for the same operands.

Folded forms:

* constructors: a scalar fills the diagonal; scalars and vectors fill the
  matrix column by column; a single matrix copies the overlapping elements and
  takes the rest from the identity (GLSL 4.60 section 5.4.2);
* ``m[i]`` and, through the vector folder, ``m[i][j]`` and ``m[i].xy``;
* ``-m``, ``m + m``, ``m - m``, ``m / m`` and the same operators with a
  scalar, component by component;
* ``m * s``, ``m * v``, ``v * m`` and ``m * m`` with linear-algebra meaning;
* ``transpose``, ``matrixCompMult`` and ``outerProduct``.

Sums of products are evaluated in index order, rounding binary32 values after
every operation. ``determinant``, ``inverse`` and matrix comparisons are not
folded; the caller sees ``None`` and must refuse the construct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..ast import CastNode
from .scalar_constant_folding import (
    FLOAT_KINDS,
    ScalarConstant,
    convert_scalar_constant,
)
from .vector_constant_folding import (
    VectorConstant,
    VectorConstantFolder,
    convert_vector_constant,
    make_vector,
)


@dataclass(frozen=True)
class MatrixConstant:
    """A folded matrix: its component kind and its columns, in order."""

    kind: str
    columns: tuple[VectorConstant, ...]

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def row_count(self) -> int:
        return self.columns[0].size

    @property
    def shape(self) -> tuple[str, int, int]:
        return self.kind, self.column_count, self.row_count


MatrixShapeResolver = Callable[[object], Optional[tuple[str, int, int]]]


def make_matrix(columns) -> MatrixConstant | None:
    """Return a matrix of *columns* (vectors of one kind and size), or ``None``."""
    columns = tuple(columns)
    if not 2 <= len(columns) <= 4 or any(
        not isinstance(column, VectorConstant) for column in columns
    ):
        return None
    kind, size = columns[0].kind, columns[0].size
    if kind not in FLOAT_KINDS or any(
        column.kind != kind or column.size != size for column in columns
    ):
        return None
    return MatrixConstant(kind, columns)


def _has_matrix(values) -> bool:
    return any(isinstance(value, MatrixConstant) for value in values)


class MatrixConstantFolder(VectorConstantFolder):
    """Evaluate scalar, vector and matrix constant expressions.

    *matrix_shape* maps a type (a name or a type node) to
    ``(kind, columns, rows)`` for matrix types and ``None`` otherwise.
    """

    def __init__(
        self,
        resolve_name,
        kind_of_type,
        vector_shape,
        matrix_shape: MatrixShapeResolver,
        *,
        fold_builtin=None,
    ):
        super().__init__(
            resolve_name, kind_of_type, vector_shape, fold_builtin=fold_builtin
        )
        self.matrix_shape = matrix_shape

    def fold(self, expr):
        if isinstance(expr, CastNode):
            shape = self.matrix_shape(expr.target_type)
            if shape is not None:
                operand = self.fold(expr.expression)
                return (
                    None if operand is None else self.construct_matrix(shape, [operand])
                )
        return super().fold(expr)

    def fold_name(self, name):
        if name and name not in {"true", "false"}:
            constant = self.resolve_name(name)
            if isinstance(constant, MatrixConstant):
                return constant
        return super().fold_name(name)

    def index_value(self, operand, index: int):
        if isinstance(operand, MatrixConstant):
            if not 0 <= index < operand.column_count:
                return None
            return operand.columns[index]
        return super().index_value(operand, index)

    def unary_value(self, operator, operand):
        if not isinstance(operand, MatrixConstant):
            return super().unary_value(operator, operand)
        if operator not in {"+", "-"}:
            return None
        return make_matrix(
            super(MatrixConstantFolder, self).unary_value(operator, column)
            for column in operand.columns
        )

    def binary_value(self, operator, left, right):
        if not _has_matrix((left, right)):
            return super().binary_value(operator, left, right)
        if isinstance(left, VectorConstant) or isinstance(right, VectorConstant):
            if operator != "*":
                return None
            if isinstance(left, MatrixConstant):
                return self.matrix_times_vector(left, right)
            return self.vector_times_matrix(left, right)
        if (
            operator == "*"
            and isinstance(left, MatrixConstant)
            and isinstance(right, MatrixConstant)
        ):
            return self.matrix_times_matrix(left, right)
        if operator not in {"+", "-", "*", "/"}:
            return None
        return self.component_wise(operator, left, right)

    def component_wise(self, operator, left, right) -> MatrixConstant | None:
        """Apply *operator* to matching columns, broadcasting a scalar."""
        matrices = [
            value for value in (left, right) if isinstance(value, MatrixConstant)
        ]
        if len(matrices) == 2 and left.shape[1:] != right.shape[1:]:
            return None
        count = matrices[0].column_count

        def column(value, index):
            return value if isinstance(value, ScalarConstant) else value.columns[index]

        return make_matrix(
            super(MatrixConstantFolder, self).binary_value(
                operator, column(left, index), column(right, index)
            )
            for index in range(count)
        )

    def matrix_times_vector(
        self, matrix: MatrixConstant, vector: VectorConstant
    ) -> VectorConstant | None:
        if vector.size != matrix.column_count:
            return None
        total = self.binary_value("*", matrix.columns[0], vector.components[0])
        for column, component in zip(matrix.columns[1:], vector.components[1:]):
            product = self.binary_value("*", column, component)
            total = None if total is None else self.binary_value("+", total, product)
        return total

    def vector_times_matrix(
        self, vector: VectorConstant, matrix: MatrixConstant
    ) -> VectorConstant | None:
        if not isinstance(vector, VectorConstant) or vector.size != matrix.row_count:
            return None
        return make_vector(self.dot(vector, column) for column in matrix.columns)

    def matrix_times_matrix(
        self, left: MatrixConstant, right: MatrixConstant
    ) -> MatrixConstant | None:
        if left.column_count != right.row_count:
            return None
        return make_matrix(
            self.matrix_times_vector(left, column) for column in right.columns
        )

    def ternary_value(self, condition, true_value, false_value):
        if not _has_matrix((condition, true_value, false_value)):
            return super().ternary_value(condition, true_value, false_value)
        if not (
            isinstance(condition, ScalarConstant)
            and isinstance(true_value, MatrixConstant)
            and isinstance(false_value, MatrixConstant)
            and true_value.shape == false_value.shape
        ):
            return None
        selected = convert_scalar_constant(condition, "bool").value
        return true_value if selected else false_value

    def call_value(self, name, arguments):
        shape = self.matrix_shape(name)
        if shape is not None:
            return self.construct_matrix(shape, arguments)
        if name == "outerProduct" and len(arguments) == 2:
            return self.outer_product(*arguments)
        if not _has_matrix(arguments):
            return super().call_value(name, arguments)
        if name == "transpose" and len(arguments) == 1:
            return self.transpose(arguments[0])
        if name == "matrixCompMult" and len(arguments) == 2:
            left, right = arguments
            if not isinstance(left, MatrixConstant) or not isinstance(
                right, MatrixConstant
            ):
                return None
            return self.component_wise("*", left, right)
        return None

    def transpose(self, matrix) -> MatrixConstant | None:
        if not isinstance(matrix, MatrixConstant):
            return None
        return make_matrix(
            make_vector(column.components[row] for column in matrix.columns)
            for row in range(matrix.row_count)
        )

    def outer_product(self, column, row) -> MatrixConstant | None:
        if not (isinstance(column, VectorConstant) and isinstance(row, VectorConstant)):
            return None
        return make_matrix(
            self.binary_value("*", column, component) for component in row.components
        )

    def construct_matrix(
        self, shape: tuple[str, int, int], arguments
    ) -> MatrixConstant | None:
        """Fold a matrix constructor of *shape* from *arguments*."""
        kind, columns, rows = shape
        if not arguments:
            return None
        if len(arguments) == 1 and isinstance(arguments[0], ScalarConstant):
            diagonal = convert_scalar_constant(arguments[0], kind)
            zero = convert_scalar_constant(ScalarConstant("int", 0), kind)
            return self._matrix_of(
                kind, columns, rows, lambda c, r: diagonal if c == r else zero
            )
        if len(arguments) == 1 and isinstance(arguments[0], MatrixConstant):
            source = arguments[0]
            one = convert_scalar_constant(ScalarConstant("int", 1), kind)
            zero = convert_scalar_constant(ScalarConstant("int", 0), kind)

            def element(c, r):
                if c < source.column_count and r < source.row_count:
                    return convert_scalar_constant(
                        source.columns[c].components[r], kind
                    )
                return one if c == r else zero

            return self._matrix_of(kind, columns, rows, element)
        if _has_matrix(arguments):
            return None
        components = []
        for argument in arguments:
            if isinstance(argument, ScalarConstant):
                components.append(argument)
            else:
                components.extend(argument.components)
        if len(components) != columns * rows:
            return None
        return self._matrix_of(
            kind,
            columns,
            rows,
            lambda c, r: convert_scalar_constant(components[c * rows + r], kind),
        )

    @staticmethod
    def _matrix_of(kind, columns, rows, element) -> MatrixConstant | None:
        if kind not in FLOAT_KINDS:
            return None
        return make_matrix(
            make_vector(element(c, r) for r in range(rows)) for c in range(columns)
        )


def convert_matrix_constant(
    constant: MatrixConstant, kind: str
) -> MatrixConstant | None:
    """Convert each component of *constant* to *kind*."""
    if constant.kind == kind:
        return constant
    return make_matrix(convert_vector_constant(c, kind) for c in constant.columns)


__all__ = (
    "MatrixConstant",
    "MatrixConstantFolder",
    "convert_matrix_constant",
    "make_matrix",
)
