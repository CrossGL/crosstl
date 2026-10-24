"""Result shapes of MSL arithmetic on matrices (MSL 2.3 section 2.3.2, "Matrix Data Types").

A ``floatCxR`` matrix has ``C`` columns of ``R`` rows. MSL defines

* ``M * v`` for a ``CxR`` matrix and a vector of ``C`` lanes: a vector of ``R`` lanes;
* ``v * M`` for a vector of ``R`` lanes and a ``CxR`` matrix: a vector of ``C`` lanes;
* ``A * B`` for a ``CxR`` and a ``KxC`` matrix: a ``KxR`` matrix;
* ``M * s``, ``s * M`` and ``M / s`` with a scalar, and ``A + B`` and ``A - B`` of equal shape: the matrix's shape.

The importer's expression typing had no matrix rule, so ``-normalize(normalMatrix * n)`` could not type the
argument of ``normalize`` and refused the program. :func:`matrix_arithmetic_shape` gives the shape of the result,
``None`` for an operation MSL does not define, or :data:`NOT_MATRIX` when neither operand is a matrix.
"""

from __future__ import annotations

__all__ = ["NOT_MATRIX", "matrix_arithmetic_shape"]

NOT_MATRIX = object()


def matrix_arithmetic_shape(op, left_matrix, right_matrix, left_vector, right_vector):
    """``("matrix", columns, rows)``, ``("vector", lanes)``, ``None`` or :data:`NOT_MATRIX`.

    ``left_matrix``/``right_matrix`` are ``(columns, rows)`` or ``None``; ``left_vector``/``right_vector`` are the
    lane count of a vector operand, ``0`` for a scalar operand, or ``None`` for any other operand.
    """
    if left_matrix is None and right_matrix is None:
        return NOT_MATRIX
    if op in {"+", "-"}:
        if left_matrix is not None and left_matrix == right_matrix:
            return ("matrix", *left_matrix)
        return None
    if op == "/":
        if left_matrix is not None and right_vector == 0:
            return ("matrix", *left_matrix)
        return None
    if op != "*":
        return None
    if left_matrix is not None and right_matrix is not None:
        left_columns, left_rows = left_matrix
        right_columns, right_rows = right_matrix
        if left_columns != right_rows:
            return None
        return ("matrix", right_columns, left_rows)
    if left_matrix is not None:
        columns, rows = left_matrix
        if right_vector == 0:
            return ("matrix", columns, rows)
        return ("vector", rows) if right_vector == columns else None
    columns, rows = right_matrix
    if left_vector == 0:
        return ("matrix", columns, rows)
    return ("vector", columns) if left_vector == rows else None
