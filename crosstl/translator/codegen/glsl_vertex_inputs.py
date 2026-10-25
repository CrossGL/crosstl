"""GLSL declarations of the members of a vertex entry point's input structure.

GLSL has no structure-typed stage inputs, so each member of a vertex input structure becomes one ``in`` variable. Two
kinds of member need more than the member's own layout:

* A member without a fixed location (another semantic than the classic ones, or none) used to be declared without
  one, so the linker chose its location and the host could not bind it at the location SPIR-V, WGSL and Metal output
  give it. It now takes that location (:mod:`.stage_io_semantic_locations`): the first free one, in declaration
  order, after the fixed locations of the structure are set aside.
* A member with a built-in semantic (``SV_VertexID``, ``SV_InstanceID``, ...) used to be declared as a user input of
  its own name, which read an unbound attribute instead of the built-in value. It is now not declared, and its reads
  read the built-in, converted to the member's type (``uint(gl_VertexID)``).
"""

from __future__ import annotations

import re

from ..ast import ArrayType, MatrixType
from .stage_io_semantic_locations import allocate_locations

__all__ = (
    "VertexInputLocationError",
    "allocated_stage_layouts",
    "vertex_input_builtin",
    "vertex_input_layouts",
    "vertex_parameter_layouts",
)

_LOCATION = re.compile(r"\blocation\s*=\s*(\d+)")


class VertexInputLocationError(ValueError):
    """A vertex input whose locations cannot be counted."""


def vertex_input_builtin(codegen, member) -> str | None:
    """The expression that reads a vertex input member with a built-in semantic, or ``None`` for a user input."""
    semantic = codegen.semantic_from_node(member)
    if semantic is None:
        return None
    builtin = str(codegen.map_stage_input_semantic(semantic, "vertex"))
    if not builtin.startswith("gl_"):
        return None
    mapped_type = str(codegen.member_type_name(member))
    return (
        codegen.stage_builtin_parameter_local_alias_expression(builtin, mapped_type)
        or builtin
    )


def vertex_input_layouts(codegen, members) -> dict:
    """``member name -> layout`` of the user inputs among ``members``: their own layout, or the allocated location."""
    users = [
        member for member in members if vertex_input_builtin(codegen, member) is None
    ]
    return {
        member.name: layout
        for member, layout in zip(
            users,
            _layouts(codegen, users, lambda node: getattr(node, "member_type", None)),
        )
    }


def vertex_parameter_layouts(codegen, parameters, is_input) -> dict:
    """``id(parameter) -> layout`` of a vertex entry point's loose user inputs, avoiding its structures' members.

    ``is_input(parameter)`` tells a loose user input from a structure, a resource or a built-in value.
    """
    occupied = set()
    loose = []
    for parameter in parameters:
        struct = codegen.structs_by_name.get(
            codegen.type_node_name(getattr(parameter, "param_type", None))
        )
        if struct is not None:
            members = getattr(struct, "members", []) or []
            member_layouts = vertex_input_layouts(codegen, members)
            for member in members:
                match = _LOCATION.search(member_layouts.get(member.name, ""))
                if match:
                    start = int(match.group(1))
                    span = _span(codegen, getattr(member, "member_type", None))
                    occupied.update(range(start, start + span))
            continue
        if is_input(parameter):
            loose.append(parameter)
    layouts = _layouts(
        codegen, loose, lambda node: getattr(node, "param_type", None), occupied
    )
    return {id(parameter): layout for parameter, layout in zip(loose, layouts)}


def _layouts(codegen, nodes, type_of, occupied=()):
    """The layout of each user input in ``nodes``: its own, with the allocated location when it has none."""
    layouts, entries = [], []
    for node in nodes:
        layout = codegen.stage_io_layout_for_node(node)
        if not layout.startswith("layout("):
            layout = ""
        match = _LOCATION.search(layout)
        layouts.append(layout)
        entries.append(
            (int(match.group(1)) if match else None, _span(codegen, type_of(node)))
        )
    result = []
    for layout, (fixed, _count), location in zip(
        layouts, entries, allocate_locations(entries, occupied)
    ):
        if fixed is None:
            inner = layout[len("layout(") : -1].strip() if layout else ""
            layout = f"layout(location = {location}{', ' + inner if inner else ''})"
        result.append(layout)
    return result


def allocated_stage_layouts(codegen, nodes, type_of, occupied=()):
    """The layout of each user stage variable in ``nodes``: its own, with the allocated location when it has none."""
    return _layouts(codegen, nodes, type_of, occupied)


def _span(codegen, type_node) -> int:
    """Locations a vertex input of ``type_node`` occupies.

    An array sized by a constant takes the constant's value (a specialization constant its default, as glslang
    assigns it); one whose size is not a constant is refused, because the inputs after it would overlap it.
    """
    if isinstance(type_node, ArrayType):
        size = getattr(type_node, "size", None)
        count = codegen.literal_int_value(
            size, codegen.glsl_specialization_default_int_constants
        )
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise VertexInputLocationError(
                f"A vertex input array needs a constant size to take locations, not '{size}'"
            )
        return count * _span(codegen, type_node.element_type)
    if isinstance(type_node, MatrixType):
        return int(type_node.cols)
    return 1
