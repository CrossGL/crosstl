"""SPIR-V array types for arrays of row-major non-square matrices in uniform blocks.

A row-major matrix in a std140 block is stored as an array of its rows (GLSL 4.60 section 7.6.2.2, rule 7), so a
``mat3x2`` takes two 16-byte rows when it is row-major and three 16-byte columns when it is column-major. An array of
such matrices therefore has a different element stride in each order: 32 bytes against 48 for ``mat3x2``.

SPIR-V puts ``ArrayStride`` on the array type itself (SPIR-V 1.6 section 3.20, Decoration), so one array type cannot
serve both orders. Aggregate types may be declared more than once with the same operands precisely so that they can
be decorated differently (SPIR-V 1.6 section 2.8, Types and Variables), and glslang declares a second
``OpTypeArray`` for the row-major member. This module does the same: the block member type of a row-major member is a
layout copy of its array type, registered under the layout ``<layout>_row_major`` and decorated with the row-major
stride, at every array level whose stride differs between the two orders. Arrays of square matrices keep the shared
type, whose stride is the same in both orders.
"""

from __future__ import annotations

__all__ = [
    "ROW_MAJOR_LAYOUT_SUFFIX",
    "row_major_array_has_own_type",
    "row_major_strides_differ",
    "uniform_block_member_layout_type",
]

ROW_MAJOR_LAYOUT_SUFFIX = "_row_major"


def row_major_strides_differ(codegen, type_id) -> bool:
    """Whether ``type_id`` is an array whose std140 element stride depends on the matrix order."""
    array_info = codegen.array_type_info_from_type(type_id)
    if array_info is None:
        return False
    element_type, _size = array_info
    return codegen.uniform_array_stride(
        element_type, True
    ) != codegen.uniform_array_stride(element_type, False)


def uniform_block_member_layout_type(codegen, member_type, layout: str, row_major):
    """The type of a uniform block member laid out with ``layout``.

    A row-major member gets its own array type at every array level whose stride depends on the order; any other
    member uses the shared layout copy of :meth:`storage_layout_type`.
    """
    if not row_major or not row_major_strides_differ(codegen, member_type):
        return codegen.storage_layout_type(member_type, layout)
    element_type, size = codegen.array_type_info_from_type(member_type)
    element_layout_type = uniform_block_member_layout_type(
        codegen, element_type, layout, row_major
    )
    return codegen.register_layout_array_type(
        element_layout_type, size, f"{layout}{ROW_MAJOR_LAYOUT_SUFFIX}"
    )


def _is_row_major_array_type(codegen, type_id) -> bool:
    return any(
        registered.id == type_id.id and layout.endswith(ROW_MAJOR_LAYOUT_SUFFIX)
        for (_element, _size, layout), registered in codegen.layout_array_types.items()
    )


def row_major_array_has_own_type(codegen, type_id) -> bool:
    """Whether every array level of ``type_id`` whose stride depends on the order is a row-major array type.

    Only such a member can be decorated with row-major strides without changing the stride of a column-major use of
    the same array type.
    """
    array_info = codegen.array_type_info_from_type(type_id)
    if array_info is None:
        return True
    if row_major_strides_differ(codegen, type_id) and not _is_row_major_array_type(
        codegen, type_id
    ):
        return False
    return row_major_array_has_own_type(codegen, array_info[0])
