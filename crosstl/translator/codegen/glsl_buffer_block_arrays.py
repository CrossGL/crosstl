"""Array members of GLSL buffer blocks lowered to byte offsets.

The HLSL, Slang and Metal emitters read and write std140 and std430 buffer blocks through byte offsets
(``glsl_buffer_layout``). An array member's descriptor holds the outermost extent (``array_count``) and stride,
and, for an array of arrays, ``inner_arrays``: one ``(count, stride)`` pair per further dimension, outermost
first. ``T a[2][3]`` holds two ``T[3]`` (GLSL 4.60 section 4.1.9), so ``a[i][j]`` lies ``i * stride + j *
inner stride`` bytes past ``a``.

Indexing an outer dimension yields a sub-array access, marked ``subarray``. Byte-offset loads and stores move one
scalar, vector, matrix or struct at a time, so an expression that still denotes an array, such as ``block.a`` or
``block.a[i]`` above, is refused where it would be read or written as a value.

A subscript of a matrix selects a column (GLSL 4.60 section 5.6), which a std140 or std430 matrix stores as one
vector every ``column_stride`` bytes. ``glsl_buffer_matrix_column_access`` describes ``block.m[c]`` as that vector,
so a write to the column, one of its components or one of its elements stores just those bytes.
"""

from __future__ import annotations

from itertools import product

from ..ast import ArrayAccessNode
from .glsl_buffer_layout import byte_offset_add, byte_offset_expression

__all__ = [
    "glsl_buffer_array_elements",
    "glsl_buffer_array_extents",
    "glsl_buffer_block_indexed_access",
    "glsl_buffer_indexed_value_type",
    "glsl_buffer_matrix_column_access",
    "glsl_buffer_whole_array_reason",
]


def glsl_buffer_block_indexed_access(codegen, expr):
    """The access ``expr``, a subscript of a buffer block array, makes, or ``None``.

    ``codegen`` provides ``glsl_buffer_block_member_access`` and ``generate_expression``. The result describes the
    element at the subscripted offset; when inner dimensions remain it is a ``subarray`` access whose
    ``array_count`` and ``stride`` are the next dimension's.
    """
    if not isinstance(expr, ArrayAccessNode):
        return None
    array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
    parent = codegen.glsl_buffer_block_member_access(array_expr)
    if parent is None:
        parent = glsl_buffer_block_indexed_access(codegen, array_expr)
        if parent is None or not parent.get("subarray"):
            return None
    elif not parent.get("is_array"):
        return None
    index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
    index = codegen.generate_expression(index_expr)
    offset = byte_offset_expression(parent["offset"], index, parent["stride"])
    access = {**parent, "offset": offset, "offset_expr": offset}
    inner = parent.get("inner_arrays") or ()
    if not inner:
        access.pop("subarray", None)
        return access
    (count, stride), rest = inner[0], inner[1:]
    sizes = tuple(parent.get("inner_array_sizes") or ())
    access.update(
        array_count=count,
        array_size=sizes[0] if sizes else count,
        stride=stride,
        runtime_array=False,
        subarray=True,
    )
    for key, remaining in (("inner_arrays", rest), ("inner_array_sizes", sizes[1:])):
        if remaining:
            access[key] = remaining
        else:
            access.pop(key, None)
    return access


_TARGET_TYPE_KEYS = ("hlsl_type", "metal_type", "slang_type")


def glsl_buffer_matrix_column_access(codegen, expr):
    """The vector access ``expr``, a subscript of a buffer block matrix, makes, or ``None``.

    ``codegen`` provides ``glsl_buffer_block_array_access``, ``glsl_buffer_block_member_access``,
    ``generate_expression`` and ``map_type``. The matrix is a member, a struct field or a fully subscripted array
    element; the column keeps its buffer, block flags and component type.
    """
    if not isinstance(expr, ArrayAccessNode):
        return None
    matrix_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
    matrix = codegen.glsl_buffer_block_array_access(matrix_expr)
    if matrix is None:
        matrix = codegen.glsl_buffer_block_member_access(matrix_expr)
        if matrix is not None and matrix.get("is_array"):
            return None
    if matrix is None or matrix.get("subarray") or not matrix.get("matrix_columns"):
        return None
    rows = matrix["matrix_rows"]
    index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
    index = codegen.generate_expression(index_expr)
    offset = byte_offset_expression(matrix["offset"], index, matrix["column_stride"])
    column_type = f"vec{rows}"
    column = {
        key: value
        for key, value in matrix.items()
        if key not in {
            "matrix_columns",
            "matrix_rows",
            "column_stride",
            "inner_arrays",
            "inner_array_sizes",
            "array_count",
            "array_size",
            "stride",
            "members",
        }
    }
    column.update(
        type=column_type,
        layout_type=column_type,
        size=rows * matrix.get("component_size", 4),
        components=rows,
        is_array=False,
        runtime_array=False,
        offset=offset,
        offset_expr=offset,
    )
    for key in _TARGET_TYPE_KEYS:
        if key in matrix:
            column[key] = codegen.map_type(column_type)
    return column


def glsl_buffer_array_extents(access):
    """The ``[n][m]...`` extents of the array ``access`` denotes, outermost first."""
    counts = [access.get("array_count")]
    counts.extend(count for count, _stride in access.get("inner_arrays") or ())
    return "".join(f"[{count}]" for count in counts)


def glsl_buffer_indexed_value_type(access):
    """The CrossGL type of the value a subscript access denotes: its element type, or a sub-array type."""
    if access.get("subarray"):
        return f"{access['type']}{glsl_buffer_array_extents(access)}"
    return access["type"]


def glsl_buffer_array_elements(member, base_offset):
    """``(subscripts, byte offset)`` for every innermost element of fixed array ``member``, in memory order.

    ``subscripts`` is the text ``[i]`` or ``[i][j]...`` that selects the element of a value of the member's type.
    """
    dimensions = [(member["array_count"], member["stride"])]
    dimensions.extend(member.get("inner_arrays") or ())
    for indices in product(*(range(count) for count, _stride in dimensions)):
        delta = sum(
            index * stride for index, (_count, stride) in zip(indices, dimensions)
        )
        subscripts = "".join(f"[{index}]" for index in indices)
        yield subscripts, byte_offset_add(base_offset, delta)


def glsl_buffer_whole_array_reason(access, *, indexed):
    """Why ``access`` cannot be read or written as one value, or ``None`` when it denotes one element.

    ``indexed`` says whether ``access`` came from a subscript (``glsl_buffer_block_indexed_access``) rather than a
    member selection, whose array members keep ``is_array`` on their elements.
    """
    if access is None:
        return None
    if indexed:
        is_array = bool(access.get("subarray"))
    else:
        is_array = bool(access.get("is_array")) and not access.get("runtime_array")
    if not is_array:
        return None
    return (
        f"{access.get('member')} is a whole {access.get('type')}"
        f"{glsl_buffer_array_extents(access)} array; "
        "byte-offset accesses read and write one element at a time, so subscript "
        "every array dimension"
    )
