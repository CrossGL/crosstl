"""Half matrices in the binary16 legalization of :mod:`.half` (defect S7).

A half matrix register (``f16matC`` or ``f16matCxR``: ``C`` columns of ``R`` rows) becomes the ``float`` matrix of the
same shape, and keeps the invariant that every element holds a binary16 value:

- a conversion to a half matrix rounds every element (``cgl_half_roundCxR`` rounds per column), and a construction from
  columns or scalars rounds the parts that are not binary16 values already;
- arithmetic whose result is half is computed in binary32 and rounded once per element. ``+`` and ``-`` of half
  matrices and the product with a half scalar are then correctly rounded, as for scalars. The elements of a matrix
  product and of a matrix-vector product are dot products, which the pass computes in binary32 and rounds once (as it
  does ``dot``);
- ``m[i]`` is a half column and ``m[i][j]`` a half scalar; ``transpose`` is exact, and ``determinant`` and ``inverse``
  round their binary32 result.

An operation of a half matrix and a ``float`` value is a ``float`` operation, as for scalars and vectors. Refused with
:class:`.half.HalfFloatUnsupportedError`: division, remainder and increments of a half matrix, other built-ins on half
matrices, and half matrix spellings without a column-major shape (``half3x3`` has 3 columns in MSL and 3 rows in HLSL,
so the importers spell half matrices ``f16matCxR``). The pass also refuses half matrices left in device, constant or
buffer memory, whose binary16 layout the emitters cannot express.
"""

from __future__ import annotations

import re

from .. import ast as A
from .half_helpers import matrix_name
from .hub_types import HubType, call_name
from .rewrite import call, set_child

__all__ = [
    "MATRIX_MARK",
    "MatrixRules",
    "float_matrix",
    "mark_matrix",
    "matrix_shape",
    "widened_matrix",
]

HALF = HubType("f", 16)
MATRIX_MARK = "cgl_half_matrix"
_SHAPED = re.compile(r"f16mat([234])(?:x([234]))?")
_GENERIC = re.compile(r"mat([234])x([234])<(?:f16|half|float16|float16_t)>")
_FLOAT_MATRIX = re.compile(r"d?mat[234](?:x[234])?")
_HALF_ELEMENTS = frozenset({"half", "float16", "float16_t", "f16"})
_FLOAT_ELEMENTS = {
    "float": "float",
    "f32": "float",
    "double": "double",
    "f64": "double",
}


def _element_name(node) -> str:
    return str(getattr(node, "name", node) or "")


def _matrix_type_shape(node):
    if str(node.rows).isdigit() and str(node.cols).isdigit():
        return int(node.cols), int(node.rows)
    return None


def matrix_shape(node):
    """The ``(columns, rows)`` of a half matrix type node or spelling (through references), or ``None``."""
    if isinstance(node, str):
        name = node.replace(" ", "")
        match = _SHAPED.fullmatch(name)
        if match:
            columns = int(match.group(1))
            return columns, int(match.group(2) or columns)
        match = _GENERIC.fullmatch(name)
        return (int(match.group(1)), int(match.group(2))) if match else None
    if isinstance(node, (A.PrimitiveType, A.NamedType)):
        if getattr(node, "generic_args", None):
            return None
        return matrix_shape(str(node.name))
    if isinstance(node, A.MatrixType):
        if _element_name(node.element_type) in _HALF_ELEMENTS:
            return _matrix_type_shape(node)
        return None
    if isinstance(node, A.ReferenceType):
        return matrix_shape(node.referenced_type)
    return None


def float_matrix(node):
    """``(name, shape)`` of a ``float`` or ``double`` matrix type node (through references), or ``None``."""
    if isinstance(node, A.ReferenceType):
        return float_matrix(node.referenced_type)
    if isinstance(node, A.MatrixType):
        name = _FLOAT_ELEMENTS.get(_element_name(node.element_type))
        shape = _matrix_type_shape(node)
        if name is not None and shape is not None:
            return name, shape
    return None


def widened_matrix(node):
    """The ``float`` matrix type that holds the half matrix ``node``; ``None`` if ``node`` is not one."""
    shape = matrix_shape(node)
    if shape is None or isinstance(node, A.ReferenceType):
        return None
    return A.MatrixType(A.PrimitiveType("float"), shape[1], shape[0])


def mark_matrix(node, shape):
    node.annotations[MATRIX_MARK] = shape
    return node


def column_of(base, scope):
    """The half column type of ``base[i]`` when ``base`` is a half matrix (carried or declared), or ``None``."""
    shape = base.annotations.get(MATRIX_MARK) if isinstance(base, A.ASTNode) else None
    if shape is None and isinstance(
        base, (A.IdentifierNode, A.ArrayAccessNode, A.MemberAccessNode)
    ):
        shape = matrix_shape(scope.node_type(base))
    return HALF.with_lanes(shape[1]) if shape is not None else None


def _shape(kind):
    """An operand kind's shape: a ``(columns, rows)`` tuple for a matrix, else its lane count."""
    return (kind.columns, kind.lanes) if kind.columns else kind.lanes


def result_shape(op, left, right):
    """The shape of ``left op right`` with a matrix operand, or ``None`` when the shapes do not combine."""
    left_matrix, right_matrix = isinstance(left, tuple), isinstance(right, tuple)
    if op in ("+", "-"):
        if left_matrix and right_matrix:
            return left if left == right else None
        return (left if left_matrix else right) if 1 in (left, right) else None
    if op != "*":
        return None
    if left_matrix and right_matrix:
        return (right[0], left[1]) if left[0] == right[1] else None
    if left_matrix:
        return left if right == 1 else (left[1] if right == left[0] else None)
    return right if left == 1 else (right[0] if left == right[1] else None)


class MatrixRules:
    """The half matrix rewrites of :class:`.half.HalfLegalizer`, which provides ``scope``, ``functions``, ``kind``,
    ``coerce``, ``helper``, ``refuse`` and ``_half_operation``."""

    def matrix_value(self, expr):
        """The shape of a half-matrix-valued expression, or ``None``."""
        if not isinstance(expr, A.ASTNode):
            return None
        carried = expr.annotations.get(MATRIX_MARK)
        if carried is not None:
            return carried
        if isinstance(expr, (A.IdentifierNode, A.ArrayAccessNode, A.MemberAccessNode)):
            return matrix_shape(self.scope.node_type(expr))
        if isinstance(expr, A.UnaryOpNode) and expr.op == "*":
            pointer = self.scope.node_type(expr.operand)
            return matrix_shape(getattr(pointer, "pointee_type", None))
        if isinstance(expr, A.FunctionCallNode):
            callee = self.functions.get(call_name(expr))
            if callee and len(callee) == 1:
                return matrix_shape(callee[0].return_type)
        return None

    def float_matrix_kind(self, expr):
        """``(name, shape)`` of a named ``float`` or ``double`` matrix expression, or ``None``."""
        if isinstance(expr, (A.IdentifierNode, A.ArrayAccessNode, A.MemberAccessNode)):
            return float_matrix(self.scope.node_type(expr))
        return None

    def rewrite_matrix_binary(self, node, left, right):
        """``left`` and ``right`` are the operand kinds of ``node``, one of them a half matrix."""
        what = f"a half matrix '{node.op}'"
        if not self._half_operation((left, right), node, what):
            return node
        if node.op not in ("+", "-", "*"):
            self.refuse(f"'{node.op}' on a half matrix", "matrix-division", node)
        for side, kind in (("left", left), ("right", right)):
            if kind.name == "half":
                continue
            if kind.lanes != 1:
                self.refuse(
                    f"{what} with a non-half vector operand", "matrix-operand", node
                )
            set_child(node, side, None, self.coerce(getattr(node, side), HALF))
        shapes = [_shape(kind) if kind.name == "half" else 1 for kind in (left, right)]
        result = result_shape(node.op, *shapes)
        if result is None:
            self.refuse(f"{what} of operands that do not combine", "matrix-shape", node)
        return self.helper("cgl_half_round", result, [node], node)

    def convert_matrix(self, expr, shape, node):
        """``expr`` converted to the half matrix ``shape``: rounded per element unless it holds binary16 values."""
        current = self.matrix_value(expr)
        if current == shape:
            return expr
        if current is not None:
            return mark_matrix(call(matrix_name(shape), [expr]), shape)
        kind = self.kind(expr)
        if kind is None:
            self.refuse(
                "a conversion to a half matrix of an operand of unknown type",
                "unknown-operand",
                node,
            )
        if kind.columns:
            if kind.name != "float":
                self.refuse(
                    f"a {kind.name} matrix converted to a half matrix",
                    "double-rounding",
                    node,
                )
            same = (kind.columns, kind.lanes) == shape
            value = expr if same else call(matrix_name(shape), [expr])
            return self.helper("cgl_half_round", shape, [value], node)
        if kind.lanes != 1 or kind.name == "bool":
            self.refuse(
                "a half matrix converted from a vector or bool", "matrix-operand", node
            )
        diagonal = call(matrix_name(shape), [self.coerce(expr, HALF)])
        diagonal.source_location = getattr(node, "source_location", None)
        return mark_matrix(diagonal, shape)

    def coerce_matrix(self, expr, shape):
        return self.convert_matrix(expr, shape, expr)

    def construct_matrix(self, shape, arguments, node):
        """``f16matCxR(...)``: one value converts; columns, scalars or a mix fill the matrix in column order."""
        if len(arguments) == 1:
            return self.convert_matrix(arguments[0], shape, node)
        parts, total = [], 0
        for argument in arguments:
            kind = self.kind(argument)
            if kind is None:
                self.refuse(
                    "a half matrix constructor argument of unknown type",
                    "unknown-operand",
                    argument,
                )
            if kind.columns or kind.name == "bool":
                self.refuse(
                    "a half matrix constructor argument that is a matrix or bool",
                    "matrix-operand",
                    argument,
                )
            parts.append(self.coerce(argument, HALF.with_lanes(kind.lanes)))
            total += kind.lanes
        if total != shape[0] * shape[1]:
            self.refuse(
                "a half matrix constructor with the wrong number of components",
                "matrix-shape",
                node,
            )
        result = call(matrix_name(shape), parts)
        result.source_location = node.source_location
        return mark_matrix(result, shape)

    def rewrite_matrix_builtin(self, name, node):
        """A built-in with a half matrix argument; ``None`` when no argument is one."""
        arguments = list(node.arguments or [])
        shapes = [self.matrix_value(argument) for argument in arguments]
        if not any(shapes):
            return None
        if _FLOAT_MATRIX.fullmatch(name):
            # A float or double matrix holds every binary16 value exactly.
            return node
        if len(arguments) == 1:
            columns, rows = shapes[0]
            if name == "transpose":
                return mark_matrix(node, (rows, columns))
            if name == "determinant" and columns == rows:
                return self.helper("cgl_half_round", 1, [node], node)
            if name == "inverse" and columns == rows:
                return self.helper("cgl_half_round", shapes[0], [node], node)
        self.refuse(f"'{name}' on a half matrix", "matrix-builtin", node)
