"""Row-major matrix members in the SPIR-V importer.

SPIR-V gives every matrix member of an explicitly laid-out struct an orientation, ``RowMajor`` or ``ColMajor``
(SPIR-V 1.6 section 2.16.2): a ``RowMajor`` matrix is stored one row after another, ``MatrixStride`` bytes apart,
while the matrix value it loads is the same. CrossGL spells the row-major order as ``@row_major`` on the struct or
block member, as the GLSL import writes ``layout(row_major)``, and each generator stores that member row by row
(``semantics/row_major_members.py`` for the targets whose structs cannot carry an order). An importer that dropped the
decoration let every generator store the member column by column, so each read took the transposed matrix.
"""

from __future__ import annotations

__all__ = ["ROW_MAJOR_ATTRIBUTE", "member_is_row_major", "row_major_field_suffix"]

ROW_MAJOR_ATTRIBUTE = "row_major"


def member_is_row_major(member_decorations, struct_type_id, member_key) -> bool:
    """Whether member ``member_key`` of struct ``struct_type_id`` is decorated ``RowMajor``."""
    return any(
        member == str(member_key) and decoration == "RowMajor"
        for member, decoration, _operands in member_decorations.get(struct_type_id, [])
    )


def row_major_field_suffix(layout_node, field_index) -> str:
    """`` @row_major`` when field ``field_index`` of a uniform or buffer block layout is ``RowMajor``, else ``""``."""
    row_major_fields = getattr(layout_node, "row_major_fields", None) or ()
    return f" @{ROW_MAJOR_ATTRIBUTE}" if field_index in row_major_fields else ""
