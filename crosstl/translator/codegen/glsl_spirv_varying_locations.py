"""Locations of the GLSL varyings that OpenGL links by name, for a SPIR-V compilation of the same source.

OpenGL links a vertex output to the fragment input of the same name (GLSL 4.60 section 4.3.4), so GLSL output
leaves a varying whose semantic has no fixed location unlocated (:mod:`.stage_io_semantic_locations`). A SPIR-V
compilation of that source links by location only: glslang ``-V`` (Vulkan, which defines ``VULKAN``) and ``-G``
(GL_ARB_gl_spirv, which defines ``GL_SPIRV``) reject each such declaration with "SPIR-V requires location for user
input/output". Adding the location unconditionally would raise the ``#version`` of every program with varyings to
4.10 (GL_ARB_separate_shader_objects), and a fragment input with a location no longer links to an unlocated vertex
output of another OpenGL program.

Such a varying now takes its location from a function-like macro that only a SPIR-V compilation expands, so one
source serves both, as :mod:`.glsl_vulkan_descriptor_sets` does for descriptor sets:

.. code-block:: glsl

    #if defined(GL_SPIRV) || defined(VULKAN)
    #if !defined(GL_ES) && __VERSION__ < 410
    #extension GL_ARB_separate_shader_objects : require
    #endif
    #define CROSSGL_LOCATION(n) layout(location = n)
    #else
    #define CROSSGL_LOCATION(n)
    #endif
    CROSSGL_LOCATION(0) in vec2 uv;

The location is the one SPIR-V output gives the same value (:mod:`.spirv_interface_locations`): each structure is
one interface, whose members with an explicit location or a classic semantic keep it and whose other user members
take the first free locations in declaration order; loose parameters then take the first locations no structure
member holds. A SPIR-V compilation of the GLSL output therefore links with the SPIR-V, WGSL and Metal output of the
other stage. The OpenGL compilation reads exactly the declaration it read before.
"""

from __future__ import annotations

import re

from .glsl_interface_block_io import block_declared_stage_io
from .glsl_vertex_inputs import (
    VertexInputLocationError,
    allocated_stage_layouts,
    location_span,
)
from .stage_distance_layout import distance_semantic

__all__ = [
    "SPIRV_LOCATION_MACRO",
    "spirv_entry_input_locations",
    "spirv_entry_output_locations",
    "spirv_location_macro_definition",
    "spirv_location_prefixed",
    "spirv_loose_varying_locations",
    "spirv_struct_varying_locations",
    "strip_spirv_location_macro",
]

SPIRV_LOCATION_MACRO = "CROSSGL_LOCATION"

_LOCATION = re.compile(r"\blocation\s*=\s*(\d+)")
_MACRO_PREFIX = re.compile(rf"^{SPIRV_LOCATION_MACRO}\(\s*\d+\s*\)\s+")


def spirv_location_macro_definition() -> str:
    """The ``#if`` definitions of :data:`SPIRV_LOCATION_MACRO`, valid for every desktop and ES version."""
    return (
        "#if defined(GL_SPIRV) || defined(VULKAN)\n"
        "#if !defined(GL_ES) && __VERSION__ < 410\n"
        "#extension GL_ARB_separate_shader_objects : require\n"
        "#endif\n"
        f"#define {SPIRV_LOCATION_MACRO}(n) layout(location = n)\n"
        "#else\n"
        f"#define {SPIRV_LOCATION_MACRO}(n)\n"
        "#endif\n"
    )


def strip_spirv_location_macro(declaration: str) -> str:
    """``declaration`` without a leading :data:`SPIRV_LOCATION_MACRO` use, as an OpenGL compilation reads it."""
    return _MACRO_PREFIX.sub("", declaration)


def _member_type(node):
    return getattr(node, "member_type", None)


def _parameter_type(node):
    return getattr(node, "param_type", None)


def _allocated(codegen, nodes, type_of, occupied=()):
    """``(location or None, start, span)`` per node; the location is ``None`` when the node's own layout has one.

    An interface whose locations cannot be counted (an array whose size is not a constant) gets no SPIR-V
    locations, so its OpenGL declarations stay as they were.
    """
    try:
        allocated = allocated_stage_layouts(codegen, nodes, type_of, occupied)
        spans = [location_span(codegen, type_of(node)) for node in nodes]
    except VertexInputLocationError:
        return [(None, None, 0)] * len(nodes)
    result = []
    for node, placed, span in zip(nodes, allocated, spans):
        own = codegen.stage_io_layout_for_node(node)
        start = int(_LOCATION.search(placed).group(1))
        fixed = own.startswith("layout(") and _LOCATION.search(own)
        result.append((None if fixed else start, start, span))
    return result


def _emits_locations(codegen, stage, direction) -> bool:
    # GLSL ES 3.00 (WebGL 2) has no locations on varyings, and no SPIR-V compilation reads it.
    return codegen.should_emit_stage_io_layout(stage, direction)


def spirv_struct_varying_locations(codegen, members, stage, direction) -> dict:
    """``member name -> location`` of the user varyings among ``members`` that have no location of their own.

    ``members`` are the members of one stage structure that are user varyings, in declaration order, and
    ``direction`` is ``"in"`` or ``"out"``.
    """
    if not _emits_locations(codegen, stage, direction):
        return {}
    return {
        member.name: location
        for member, (location, _start, _span) in zip(
            members, _allocated(codegen, members, _member_type)
        )
        if location is not None
    }


def spirv_loose_varying_locations(
    codegen, struct_members, parameters, stage, direction
) -> dict:
    """``id(parameter) -> location`` of loose user varyings, after the locations ``struct_members`` take.

    ``struct_members`` lists, per structure-typed parameter, its user varying members; ``parameters`` are the
    loose user varyings, in declaration order.
    """
    if not parameters or not _emits_locations(codegen, stage, direction):
        return {}
    taken = set()
    for members in struct_members:
        for _location, start, span in _allocated(codegen, members, _member_type, taken):
            if start is not None:
                taken.update(range(start, start + max(span, 1)))
    return {
        id(parameter): location
        for parameter, (location, _start, _span) in zip(
            parameters, _allocated(codegen, parameters, _parameter_type, taken)
        )
        if location is not None
    }


def spirv_location_prefixed(codegen, declaration: str, location) -> str:
    """``declaration`` with the SPIR-V location ``location``, or unchanged when it has a layout or no location."""
    if location is None or declaration.startswith("layout("):
        return declaration
    codegen.glsl_spirv_location_macro_used = True
    return f"{SPIRV_LOCATION_MACRO}({location}) {declaration}"


def _struct_of(codegen, parameter):
    return codegen.structs_by_name.get(
        codegen.type_node_name(getattr(parameter, "param_type", None))
    )


def _vertex_output_members(codegen, struct):
    """The members of a vertex output structure that are user varyings declared on their own."""
    members = []
    for member in getattr(struct, "members", []) or []:
        mapped = codegen.map_semantic(codegen.stage_io_semantic_from_node(member))
        if distance_semantic(mapped) is not None:
            continue
        name = codegen.vertex_output_member_name(member)
        if codegen.is_vertex_stage_builtin_output(name):
            continue
        if block_declared_stage_io(codegen, "out", member, name) is None:
            members.append(member)
    return members


def _fragment_input_members(codegen, struct):
    """The members of a fragment input structure that are user varyings declared on their own."""
    members = []
    for member in getattr(struct, "members", []) or []:
        name = codegen.fragment_input_member_name(member, struct.name)
        if name is None:
            continue
        if block_declared_stage_io(codegen, "in", member, name) is None:
            members.append(member)
    return members


def spirv_entry_input_locations(codegen, parameters, stage, is_loose) -> dict:
    """``id(parameter) -> location`` of a fragment entry's loose user inputs, after its structures' members.

    ``is_loose(parameter)`` tells a loose user input from a structure, a resource or a built-in value.
    """
    if stage != "fragment":
        return {}
    structs = [
        _fragment_input_members(codegen, struct)
        for struct in (_struct_of(codegen, parameter) for parameter in parameters)
        if struct is not None
    ]
    loose = [
        parameter
        for parameter in parameters
        if _struct_of(codegen, parameter) is None and is_loose(parameter)
    ]
    return spirv_loose_varying_locations(codegen, structs, loose, stage, "in")


def spirv_entry_output_locations(codegen, function, parameters, stage) -> dict:
    """``id(parameter) -> location`` of a vertex entry's ``out`` parameters, after its returned structure's members.

    ``parameters`` are the entry's user output parameters, in declaration order.
    """
    if stage != "vertex":
        return {}
    struct = codegen.structs_by_name.get(
        codegen.type_node_name(getattr(function, "return_type", None))
    )
    structs = [] if struct is None else [_vertex_output_members(codegen, struct)]
    return spirv_loose_varying_locations(codegen, structs, parameters, stage, "out")
