"""Result types of MSL matrix arithmetic, for the Metal front end's expression typing.

MSL (Metal Shading Language Specification, section 2.3.2 "Matrix data types" and the operators of section 6) defines
on a matrix ``TCxR`` (``C`` columns of ``R`` components):

* ``m * v`` for a vector of ``C`` components, giving a vector of ``R`` components;
* ``v * m`` for a vector of ``R`` components, giving a vector of ``C`` components;
* ``m1 * m2`` for ``m1`` of ``K`` columns and ``m2`` of ``K`` rows, giving ``m2``'s columns and ``m1``'s rows;
* ``m * s`` and ``s * m`` for a scalar, and ``m1 + m2`` and ``m1 - m2`` for matrices of one shape, giving the matrix.

The element types must agree. Without these rules the front end could not type ``normalize(m * v)`` once the call
sat inside another expression, and refused the program.
"""

from __future__ import annotations

__all__ = ["matrix_binary_result_type"]


def matrix_binary_result_type(operator, left, right):
    """The MSL result type of ``left operator right``, or ``None`` when no matrix rule applies.

    ``left`` and ``right`` describe the operands as ``("matrix", element, columns, rows)``,
    ``("vector", element, width)`` or ``("scalar", element)``; at least one must be a matrix.
    """
    if left[0] != "matrix" and right[0] != "matrix":
        return None
    if operator in ("+", "-"):
        if left[0] == right[0] == "matrix" and left[1:] == right[1:]:
            return _matrix(*left[1:])
        return None
    if operator != "*":
        return None
    if left[0] == "matrix" and right[0] == "scalar":
        return _matrix(*left[1:])
    if left[0] == "scalar" and right[0] == "matrix":
        return _matrix(*right[1:])
    if left[1] != right[1]:
        return None
    if left[0] == "matrix" and right[0] == "vector":
        _kind, element, columns, rows = left
        return f"{element}{rows}" if right[2] == columns else None
    if left[0] == "vector" and right[0] == "matrix":
        _kind, element, columns, rows = right
        return f"{element}{columns}" if left[2] == rows else None
    if left[0] == right[0] == "matrix":
        _kind, element, inner, rows = left
        _kind, _element, columns, right_rows = right
        return _matrix(element, columns, rows) if inner == right_rows else None
    return None


def _matrix(element, columns, rows):
    return f"{element}{columns}x{rows}"
