"""Whole arrays read from or written to GLSL buffer blocks lowered to byte offsets (HLSL and Slang output).

A GLSL array is a value, so ``float c[4] = block.row;``, ``f(block.row)`` and ``block.grid[1] = r;`` read or write
every element (GLSL 4.60 sections 5.8 and 6.1.1). The HLSL and Slang emitters reach such a block through
``ByteAddressBuffer`` loads and stores of one element at a time (``glsl_buffer_block_arrays``).

HLSL has no array-typed expression other than an lvalue. A whole-array read is therefore built as the member of a
helper-constructed struct, as for by-value array arguments (``hlsl_array_value_arguments``):
``crossgl_array_value_float_4(e0, e1, e2, e3).v``, whose arguments are the element loads. A nested array builds its
inner arrays first. DXC and slangc accept that member wherever an array value goes: an initializer, the right side
of an assignment, or a by-value argument. A whole-array write copies the value into a temporary array and stores
each element.

Both repeat the access's byte offset once per element, so every subscript in the access must be free of side
effects; otherwise the access is refused.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, MemberAccessNode
from ..semantics.rewrite import is_pure
from .array_utils import format_c_style_array_declaration
from .glsl_buffer_block_arrays import (
    glsl_buffer_array_elements,
    glsl_buffer_array_extents,
)
from .glsl_buffer_layout import byte_offset_add

__all__ = [
    "glsl_buffer_array_store_lines",
    "glsl_buffer_array_value",
    "glsl_buffer_array_value_type",
    "glsl_buffer_subscripts_are_repeatable",
]


def glsl_buffer_array_value_type(access):
    """The CrossGL type of the whole array ``access`` denotes, such as ``float[2][3]``."""
    return f"{access['type']}{glsl_buffer_array_extents(access)}"


def glsl_buffer_subscripts_are_repeatable(expr):
    """Whether every subscript on the access path ``expr`` can be evaluated more than once."""
    while isinstance(expr, (ArrayAccessNode, MemberAccessNode)):
        if isinstance(expr, ArrayAccessNode):
            if not is_pure(getattr(expr, "index_expr", getattr(expr, "index", None))):
                return False
            expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        else:
            expr = getattr(expr, "object_expr", getattr(expr, "object", None))
    return True


def glsl_buffer_array_value(access, *, load_element, value_struct, map_type):
    """An expression holding the whole fixed-size array ``access`` denotes.

    ``load_element(element_access)`` renders one innermost element at ``element_access["offset"]``;
    ``value_struct((mapped_type, element_type, extent))`` returns the value struct's ``(name, constructor)``;
    ``map_type`` maps a CrossGL type to the target spelling.
    """
    inner = tuple(access.get("inner_arrays") or ())
    elements = []
    for index in range(access["array_count"]):
        offset = byte_offset_add(access["offset"], index * access["stride"])
        if inner:
            (count, stride), rest = inner[0], inner[1:]
            row = {
                **access,
                "offset": offset,
                "array_count": count,
                "stride": stride,
                "inner_arrays": rest,
            }
            elements.append(
                glsl_buffer_array_value(
                    row,
                    load_element=load_element,
                    value_struct=value_struct,
                    map_type=map_type,
                )
            )
        else:
            elements.append(load_element({**access, "offset": offset}))
    element_type = access["type"] + "".join(f"[{count}]" for count, _stride in inner)
    shape = (
        map_type(glsl_buffer_array_value_type(access)),
        element_type,
        access["array_count"],
    )
    _name, constructor = value_struct(shape)
    return f"{constructor}({', '.join(elements)}).v"


def glsl_buffer_array_store_lines(
    access, offset, value, temp_name, *, store_element, map_type
):
    """Statements that store the array ``value`` into the whole array ``access`` at byte ``offset``.

    ``store_element(element_access, offset, value)`` returns the statement lines of one innermost element.
    """
    declaration = format_c_style_array_declaration(
        map_type(glsl_buffer_array_value_type(access)), temp_name
    )
    lines = [f"{declaration} = {value}"]
    for subscripts, element_offset in glsl_buffer_array_elements(access, offset):
        element = {**access, "offset": element_offset}
        lines.extend(store_element(element, element_offset, f"{temp_name}{subscripts}"))
    return lines
