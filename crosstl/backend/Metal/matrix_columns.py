"""``m.columns[i]``: the column array of a Metal matrix.

The Metal standard library's ``metal::matrix<T, C, R>`` keeps its columns in the public array member
``vec<T, R> columns[C]``, like the ``<simd/simd.h>`` matrix structures that host code shares, so ``m.columns[i]``
is the column ``m[i]``. CrossGL indexes a matrix by column, and the importer reads ``m.columns[i]`` as ``m[i]``.
The array itself used as a value (``m.columns`` with no index) has no CrossGL spelling and is refused with
:class:`MetalMatrixColumnsError`.
"""

from __future__ import annotations

from .MetalAst import MemberAccessNode

__all__ = ["MetalMatrixColumnsError", "matrix_columns_object"]


class MetalMatrixColumnsError(ValueError):
    """Raised when the columns array of a Metal matrix is used as a value rather than indexed."""

    project_diagnostic_code = "project.translate.metal-matrix-columns-unsupported"
    missing_capabilities = ("metal.matrix-columns-array-value",)

    def __init__(self, source_location=None):
        self.source_location = source_location
        super().__init__(
            "Cannot import the columns array of a Metal matrix used as a value: CrossGL indexes a "
            "matrix by column (m[i]) and has no array of its columns"
        )


def matrix_columns_object(codegen, expr):
    """The matrix expression whose ``columns`` member ``expr`` reads, or ``None``."""
    if (
        not isinstance(expr, MemberAccessNode)
        or str(expr.member) != "columns"
        or getattr(expr, "is_pointer", False)
    ):
        return None
    object_type = codegen.expression_metal_type(expr.object)
    if object_type is None or codegen.metal_matrix_type_parts(object_type) is None:
        return None
    return expr.object
