"""SPIR-V clip and cull distances carried by stage structure members.

A CrossGL stage structure member with a ``gl_ClipDistance``/``gl_CullDistance`` semantic (HLSL
``SV_ClipDistanceN``/``SV_CullDistanceN``) is part of the stage's one ``BuiltIn ClipDistance`` or
``BuiltIn CullDistance`` float array, not a user location: Vulkan clips against that built-in only (Vulkan 1.3
section 27.4). :mod:`stage_distance_layout` places every member in the array as DXC does; this module declares the
array, sized by the members of the structure, and stores or loads each member element by element through access
chains. A vertex, tessellation evaluation or geometry stage writes it and a fragment stage reads it; any other use
is refused.
"""

from __future__ import annotations

from .stage_distance_layout import DistanceLayout, distance_layout

__all__ = [
    "entry_point_distance_layout",
    "entry_point_distance_variable",
    "load_entry_point_distance",
    "store_entry_point_distance",
]

_BUILTINS = {
    "gl_ClipDistance": ("ClipDistance", "ClipDistance"),
    "gl_CullDistance": ("CullDistance", "CullDistance"),
}
_STAGES = {
    "Output": {"Vertex", "TessellationEvaluation", "Geometry"},
    "Input": {"Fragment"},
}


def _member_type_text(codegen, member_type):
    """CrossGL spelling of a member's float, float vector or float array type for layout purposes."""
    array_info = codegen.array_type_info_from_type(member_type)
    if array_info is not None and array_info[0] is not None:
        element = codegen.normalize_primitive_name(array_info[0].type.base_type)
        return f"{element}[{array_info[1]}]"
    base_type = member_type.type.base_type
    vector = codegen.vector_component_type_and_count(base_type)
    if vector is not None:
        component, count = vector
        return f"vec{count}" if component == "float" else f"{component}x{count}"
    return codegen.normalize_primitive_name(base_type)


def entry_point_distance_layout(codegen, members, metadata) -> DistanceLayout:
    """The distance layout of a stage structure's ``(member_type, member_name)`` members."""
    entries = []
    for member_type, member_name in members:
        node = (metadata.get(member_name) or {}).get("node")
        semantic = codegen.semantic_from_node(node) if node is not None else None
        entries.append(
            (member_name, semantic, _member_type_text(codegen, member_type), None)
        )
    return distance_layout(entries)


def entry_point_distance_variable(codegen, builtin, size, storage_class):
    """The stage's ``float[size]`` distance built-in variable of ``storage_class``."""
    builtin_name, capability = _BUILTINS[builtin]
    execution_model = codegen.current_execution_model
    if execution_model not in _STAGES[storage_class]:
        raise ValueError(
            f"SPIR-V {builtin} is not a {storage_class.lower()} of the "
            f"{execution_model or 'current'} stage"
        )
    cache_key = f"__distance_builtin::{builtin}::{storage_class}"
    existing = codegen.global_variables.get(cache_key)
    if existing is not None:
        declared = codegen.array_type_info_from_type(
            codegen.variable_value_types.get(existing.id)
        )
        if declared is not None and declared[1] != size:
            raise ValueError(
                f"SPIR-V {builtin} is declared with {declared[1]} elements and with {size}"
            )
        codegen.mark_builtin_interface_variable(existing)
        return existing
    float_type = codegen.register_primitive_type("float")
    array_type = codegen.register_array_type(float_type, size)
    codegen.require_capability(capability)
    variable = codegen.register_builtin_variable(
        builtin, array_type, builtin_name, storage_class
    )
    codegen.global_variables[cache_key] = variable
    codegen.mark_builtin_interface_variable(variable)
    return variable


def _element_pointer(codegen, variable, element, storage_class):
    float_type = codegen.register_primitive_type("float")
    int_type = codegen.register_primitive_type("int")
    index = codegen.register_constant(element, int_type)
    pointer_type = codegen.register_pointer_type(float_type, storage_class)
    access = codegen.access_chain(variable, [index], pointer_type)
    codegen.variable_value_types[access.id] = float_type
    return access, float_type


def store_entry_point_distance(codegen, variable, slot, value, member_type):
    """Store a member ``value`` into elements ``slot.offset`` onwards of the distance output ``variable``."""
    if slot.shape == "scalar":
        access, float_type = _element_pointer(codegen, variable, slot.offset, "Output")
        codegen.store_to_variable(
            access, codegen.convert_value_to_type(value, float_type)
        )
        return
    float_type = codegen.register_primitive_type("float")
    for component in range(slot.count):
        element = codegen.composite_extract(value, float_type, component)
        access, _ = _element_pointer(
            codegen, variable, slot.offset + component, "Output"
        )
        codegen.store_to_variable(access, element)


def load_entry_point_distance(codegen, variable, slot, member_type):
    """The member value read from elements ``slot.offset`` onwards of the distance input ``variable``."""
    elements = []
    for component in range(slot.count):
        access, float_type = _element_pointer(
            codegen, variable, slot.offset + component, "Input"
        )
        elements.append(codegen.load_from_variable(access, float_type))
    if slot.shape == "scalar":
        return elements[0]
    return codegen.composite_construct(member_type, elements)
