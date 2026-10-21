"""Shape of CrossGL matrix types.

CrossGL matrices follow GLSL (GLSL 4.60 section 4.1.6): ``matCxR`` has C columns and R rows,
constructors fill it in column-major order, ``m[i]`` is column ``i``, ``m * v`` multiplies a
column vector and ``m.length()`` is the column count. ``MatrixType.cols`` and
``MatrixType.rows`` carry those two numbers.
"""

from __future__ import annotations

import re

_CROSSGL_MATRIX = re.compile(r"(d?)mat([234])(?:x([234]))?")


def crossgl_matrix_shape(spelling):
    """Return ``(scalar, columns, rows)`` for a CrossGL matrix spelling, or ``None``."""
    match = _CROSSGL_MATRIX.fullmatch(str(spelling or "").strip())
    if match is None:
        return None
    columns = int(match.group(2))
    rows = int(match.group(3) or match.group(2))
    return ("double" if match.group(1) else "float"), columns, rows


def crossgl_matrix_spelling(columns, rows, scalar="float", *, short_square=True):
    """The CrossGL spelling of a ``columns`` x ``rows`` matrix of ``scalar``."""
    prefix = "dmat" if scalar in {"double", "f64"} else "mat"
    if short_square and columns == rows:
        return f"{prefix}{columns}"
    return f"{prefix}{columns}x{rows}"


_CROSSGL_VECTOR = re.compile(r"(d?)vec([234])")


def _vector_shape(spelling):
    match = _CROSSGL_VECTOR.fullmatch(str(spelling or "").strip())
    if match is None:
        return None
    return ("double" if match.group(1) else "float"), int(match.group(2))


def _vector_spelling(scalar, width):
    return f"{'d' if scalar == 'double' else ''}vec{width}"


def crossgl_linear_algebra_product_type(left, right):
    """Result spelling of ``left * right`` when either operand is a CrossGL matrix.

    ``matCxR * vecC`` is a ``vecR``, ``vecR * matCxR`` a ``vecC`` and ``matCxR * matKxC`` a
    ``matKxR``. Returns ``None`` when neither operand is a matrix or the shapes do not
    conform; scalar operands keep the matrix type.
    """
    left_matrix = crossgl_matrix_shape(left)
    right_matrix = crossgl_matrix_shape(right)
    if left_matrix is None and right_matrix is None:
        return None
    if left_matrix is not None and right_matrix is not None:
        scalar, inner, rows = left_matrix
        _, columns, right_rows = right_matrix
        if right_rows != inner:
            return None
        return crossgl_matrix_spelling(columns, rows, scalar)
    if left_matrix is not None:
        scalar, columns, rows = left_matrix
        vector = _vector_shape(right)
        if vector is None:
            return (
                str(left).strip() if str(right).strip() in {"float", "double"} else None
            )
        return _vector_spelling(scalar, rows) if vector[1] == columns else None
    scalar, columns, rows = right_matrix
    vector = _vector_shape(left)
    if vector is None:
        return str(right).strip() if str(left).strip() in {"float", "double"} else None
    return _vector_spelling(scalar, columns) if vector[1] == rows else None


def matrix_type_spelling(type_node, scalar="float", *, short_square=True):
    """The CrossGL spelling of a ``MatrixType`` node.

    Other nodes that carry ``rows`` and ``cols`` (cooperative matrices) keep their own
    ``rows x cols`` order, so callers can use this on either.
    """
    if type(type_node).__name__ != "MatrixType":
        prefix = "dmat" if scalar in {"double", "f64"} else "mat"
        if short_square and type_node.rows == type_node.cols:
            return f"{prefix}{type_node.rows}"
        return f"{prefix}{type_node.rows}x{type_node.cols}"
    return crossgl_matrix_spelling(
        type_node.cols, type_node.rows, scalar, short_square=short_square
    )


def length_method_object(func_expr, args):
    """The object ``x`` of a ``x.length()`` call, or ``None`` for any other callee."""
    if args or type(func_expr).__name__ != "MemberAccessNode":
        return None
    if getattr(func_expr, "member", None) != "length":
        return None
    return getattr(func_expr, "object_expr", getattr(func_expr, "object", None))
