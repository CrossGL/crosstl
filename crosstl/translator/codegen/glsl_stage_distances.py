"""GLSL names for clip and cull distance members of stage structures.

GLSL has one ``gl_ClipDistance`` and one ``gl_CullDistance`` float array per stage. A CrossGL stage structure may
instead carry several distance members, each a float, a float vector or a float array with its own semantic index
(``gl_ClipDistance``, ``gl_ClipDistance1``, ...; HLSL ``SV_ClipDistanceN``). :mod:`stage_distance_layout` places
every member in the array; this module names the member accordingly:

- a float member is the array element at its offset, ``gl_ClipDistance[2]``;
- a float array member at offset 0 is the array itself;
- a fragment input vector member is read as ``vec2(gl_ClipDistance[0], gl_ClipDistance[1])``;
- a vertex output vector member is a private global of its type, copied element by element into the array before
  every return of the entry point (:func:`distance_output_epilogue`), because no GLSL lvalue names part of an array
  as a vector.

A float array member that does not start at element 0 is refused.
"""

from __future__ import annotations

from .stage_distance_layout import DistanceLayout, distance_layout, distance_semantic

__all__ = [
    "distance_input_alias",
    "distance_output_epilogue",
    "distance_output_name",
    "distance_private_output_declaration",
    "struct_distance_layout",
]

_LAYOUTS = "_glsl_distance_layouts"


def _member_entry(codegen, member):
    member_type = getattr(member, "member_type", None) or getattr(member, "vtype", None)
    array_size = (
        getattr(member, "size", None) if hasattr(member, "element_type") else None
    )
    type_text = codegen.type_name_string(
        getattr(member, "element_type", None) if array_size is not None else member_type
    )
    semantic = codegen.map_semantic(codegen.stage_io_semantic_from_node(member))
    return member.name, semantic, type_text, array_size


def struct_distance_layout(codegen, struct) -> DistanceLayout:
    """The distance layout of a CrossGL structure's members, computed once per structure."""
    layouts = codegen.__dict__.setdefault(_LAYOUTS, {})
    key = id(struct)
    if key not in layouts:
        members = getattr(struct, "members", None) or []
        try:
            layout = distance_layout(
                _member_entry(codegen, member) for member in members
            )
        except ValueError as error:
            codegen.refuse_lowering(str(error), "")
            layout = DistanceLayout(())
        layouts[key] = (struct, layout)
    return layouts[key][1]


def _member_slot(codegen, member):
    for struct in codegen.structs_by_name.values():
        if any(
            candidate is member for candidate in getattr(struct, "members", None) or []
        ):
            return struct_distance_layout(codegen, struct).slot(member.name)
    return None


def _private_name(codegen, member):
    return codegen.glsl_generated_module_identifier(
        ("stage-io", "distance-output", member.name), f"crossgl_{member.name}"
    )


def _refuse_offset_array(codegen, member, slot):
    return codegen.refuse_lowering(
        f"{slot.builtin} array '{member.name}' starts at element {slot.offset}: GLSL names only the "
        "whole distance array, so an array value must be its first value",
        slot.builtin,
    )


def distance_output_name(codegen, member, builtin):
    """The GLSL name a vertex-output distance member's accesses use."""
    slot = _member_slot(codegen, member)
    if slot is None:
        parsed = distance_semantic(builtin)
        return codegen.refuse_lowering(
            f"{builtin} output '{member.name}' is not a member of a stage structure",
            parsed[0] if parsed else builtin,
        )
    if slot.shape == "scalar":
        return f"{slot.builtin}[{slot.offset}]"
    if slot.shape == "array":
        return (
            slot.builtin
            if slot.offset == 0
            else _refuse_offset_array(codegen, member, slot)
        )
    return _private_name(codegen, member)


def distance_private_output_declaration(codegen, member):
    """The private global declaration of a vertex-output distance vector member, or ``None``."""
    slot = _member_slot(codegen, member)
    if slot is None or slot.shape != "vector":
        return None
    return f"vec{slot.count} {_private_name(codegen, member)};\n"


def distance_output_epilogue(codegen, indent_str):
    """Statements copying the current stage structure's distance vectors into their arrays."""
    struct = codegen.structs_by_name.get(
        getattr(codegen, "current_stage_return_type", None)
    )
    if struct is None:
        return ""
    code = ""
    for slot in struct_distance_layout(codegen, struct).slots:
        if slot.shape != "vector":
            continue
        name = _private_name(codegen, _named_member(struct, slot.name))
        for component in range(slot.count):
            code += f"{indent_str}{slot.builtin}[{slot.offset + component}] = {name}[{component}];\n"
    return code


def _named_member(struct, name):
    for member in getattr(struct, "members", None) or []:
        if member.name == name:
            return member
    raise KeyError(name)


def distance_input_alias(codegen, member):
    """The GLSL expression a fragment-input distance member reads, or ``None`` for other members."""
    if (
        distance_semantic(
            codegen.map_semantic(codegen.stage_io_semantic_from_node(member))
        )
        is None
    ):
        return None
    slot = _member_slot(codegen, member)
    if slot is None:
        return None
    if slot.shape == "scalar":
        return f"{slot.builtin}[{slot.offset}]"
    if slot.shape == "array":
        return (
            slot.builtin
            if slot.offset == 0
            else _refuse_offset_array(codegen, member, slot)
        )
    elements = ", ".join(
        f"{slot.builtin}[{slot.offset + component}]" for component in range(slot.count)
    )
    return f"vec{slot.count}({elements})"
