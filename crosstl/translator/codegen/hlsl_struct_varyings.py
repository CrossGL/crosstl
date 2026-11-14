"""HLSL semantics for stage varyings of structure type.

A stage structure member of structure type with ``@location(N)``, which the GLSL import produces for an instance
interface block (``out Data {...} data;``), occupies consecutive locations from N, one or more per field, as in
GLSL. HLSL expresses that with one semantic on the member: given ``Data data : TEXCOORD<N>``, DXC numbers the fields
TEXCOORD<N>, TEXCOORD<N+1>, ... in declaration order. The enclosing semantic overrides any semantic on a field, so a
structure whose fields carry their own locations cannot be expressed this way and is refused.
"""

from ..ast import ArrayType, MatrixType


def _location_span(type_node, structs_by_name):
    if isinstance(type_node, ArrayType):
        size = getattr(type_node, "size", None)
        size = getattr(size, "value", size)
        try:
            count = int(str(size))
        except (TypeError, ValueError):
            return None
        element = _location_span(type_node.element_type, structs_by_name)
        return None if element is None else count * element
    if isinstance(type_node, MatrixType):
        return int(type_node.cols)
    nested = structs_by_name.get(str(getattr(type_node, "name", type_node)))
    if nested is not None:
        return struct_location_span(nested, structs_by_name)
    return 1


def struct_location_span(struct_node, structs_by_name):
    """Locations a value of ``struct_node`` occupies as a varying, or None when an array size is not a literal."""
    total = 0
    for field in getattr(struct_node, "members", []) or []:
        span = _location_span(getattr(field, "member_type", None), structs_by_name)
        if span is None:
            return None
        total += span
    return total


def located_struct_varying(codegen, member):
    """The structure that a located stage-structure ``member`` has as its type (or element type), else None."""
    if codegen.hlsl_location_attribute_index(member) is None:
        return None
    type_node = getattr(member, "member_type", None)
    while isinstance(type_node, ArrayType):
        type_node = type_node.element_type
    name = getattr(type_node, "name", None)
    if name is None:
        return None
    return codegen.structs_by_name.get(str(name))


def assign_struct_varying_semantic(codegen, struct_node, member, used_ranges, defaults):
    """Give a located varying of structure type ``TEXCOORD<location>``.

    Returns the next free TEXCOORD index, or None when ``member`` is not such a varying.
    """
    nested = located_struct_varying(codegen, member)
    if nested is None:
        return None
    location = codegen.hlsl_location_attribute_index(member)
    located_fields = [
        field.name
        for field in getattr(nested, "members", []) or []
        if codegen.hlsl_location_attribute_index(field) is not None
    ]
    span = struct_location_span(nested, codegen.structs_by_name)
    if located_fields or span is None:
        detail = (
            f"its fields {', '.join(located_fields)} declare their own locations, which an enclosing HLSL "
            "semantic would override"
            if located_fields
            else "a field's array size is not a literal integer"
        )
        codegen.refuse_lowering(
            f"stage varying '{struct_node.name}.{member.name}' of structure type '{nested.name}' at "
            f"location {location}: {detail}"
        )
        return location + 1
    count = span * _array_count(getattr(member, "member_type", None))
    semantic = f"TEXCOORD{location}"
    codegen.hlsl_reserve_semantic_range(
        used_ranges, struct_node.name, member.name, semantic, max(count, 1)
    )
    defaults[member.name] = semantic
    return location + max(count, 1)


def _array_count(type_node):
    count = 1
    while isinstance(type_node, ArrayType):
        size = getattr(type_node.size, "value", type_node.size)
        try:
            count *= int(str(size))
        except (TypeError, ValueError):
            return count
        type_node = type_node.element_type
    return count
