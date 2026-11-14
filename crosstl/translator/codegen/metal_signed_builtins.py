"""GLSL's ``int`` built-in values on the ``uint`` attributes of Metal.

CrossGL spells its built-in values as GLSL does, and GLSL declares several of them ``int``: ``gl_VertexID``,
``gl_InstanceID``, ``gl_BaseVertex`` and ``gl_BaseInstance`` (GLSL 4.60 section 7.1.1), ``gl_PrimitiveID``,
``gl_Layer``, ``gl_ViewportIndex`` and ``gl_SampleID`` as fragment inputs (section 7.1.5), ``gl_ViewIndex``
(GL_EXT_multiview) and the stencil export ``gl_FragStencilRefARB`` (GL_ARB_shader_stencil_export). The Metal
attributes for them, ``[[vertex_id]]``, ``[[instance_id]]``, ``[[base_vertex]]``, ``[[base_instance]]``,
``[[primitive_id]]``, ``[[render_target_array_index]]``, ``[[viewport_array_index]]``, ``[[sample_id]]``,
``[[amplification_id]]`` and ``[[stencil]]``, take ``uint`` (Metal Shading Language Specification 3.2, sections
5.2.3.1 to 5.2.3.4; ``ushort`` is the only other form some of them allow). The Metal generator refused every ``int``
declaration of them with an untyped ``ValueError``, so a CrossGL program written with GLSL's types could not reach
Metal.

An entry parameter of such a built-in now receives the ``uint`` attribute under a fresh name, and a body-local of the
declared ``int`` type is initialised from it, so every use keeps GLSL's signedness. A stage input or output structure
member is declared ``uint``; each read of it is converted back to ``int`` and each write is converted to ``uint``.
The values these built-ins carry are below 2^31, and a stencil reference keeps its low bits, so the conversions are
exact.
"""

from __future__ import annotations

import re

__all__ = [
    "METAL_SIGNED_BUILTIN_PARAMETER_NAMES",
    "collect_metal_signed_builtin_members",
    "metal_signed_builtin_member_attribute",
    "metal_signed_builtin_parameter_name",
    "metal_signed_member_assignment",
]

# The attribute each stage reads, and the base of the name its ``uint`` entry parameter takes.
METAL_SIGNED_BUILTIN_PARAMETER_NAMES = {
    "vertex": {
        "vertex_id": "_crossglVertexID",
        "instance_id": "_crossglInstanceID",
        "base_vertex": "_crossglBaseVertex",
        "base_instance": "_crossglBaseInstance",
        "amplification_id": "_crossglViewIndex",
    },
    "fragment": {
        "primitive_id": "_crossglPrimitiveID",
        "render_target_array_index": "_crossglLayer",
        "viewport_array_index": "_crossglViewportIndex",
        "sample_id": "_crossglSampleID",
        "amplification_id": "_crossglViewIndex",
    },
}

# Structure members of these attributes may be declared ``int`` in CrossGL; inputs, plus the stencil output.
_SIGNED_MEMBER_ATTRIBUTES = frozenset(
    {
        "vertex_id",
        "instance_id",
        "base_vertex",
        "base_instance",
        "primitive_id",
        "render_target_array_index",
        "viewport_array_index",
        "sample_id",
        "amplification_id",
        "stencil",
    }
)

_ATTRIBUTE_NAME = re.compile(r"\[\[\s*([A-Za-z_]\w*)")

# Compound assignments whose ``uint`` result has the bits of the ``int`` one (two's-complement wrap-around).
_BITWISE_EQUAL_OPERATORS = frozenset({"=", "+=", "-=", "*=", "&=", "|=", "^=", "<<="})
_SIGNED_OPERATORS = frozenset({"/=", "%=", ">>="})


def metal_signed_builtin_parameter_name(stage_name, metal_semantic, mapped_type):
    """The base name of the ``uint`` parameter an ``int`` built-in parameter is read through, or ``None``."""
    if mapped_type != "int":
        return None
    return METAL_SIGNED_BUILTIN_PARAMETER_NAMES.get(stage_name, {}).get(metal_semantic)


def metal_signed_builtin_member_attribute(metal_semantic, mapped_type):
    """Whether a structure member of ``mapped_type`` on ``metal_semantic`` is GLSL's ``int`` on a ``uint`` attribute."""
    return mapped_type == "int" and metal_semantic in _SIGNED_MEMBER_ATTRIBUTES


def metal_signed_member_assignment(lhs, op, rhs, side_effect_free):
    """``lhs op rhs`` for a ``uint`` member that holds an ``int`` value, or ``None`` when it cannot be rewritten."""
    if op in _BITWISE_EQUAL_OPERATORS:
        return f"{lhs} {op} uint({rhs})"
    if op in _SIGNED_OPERATORS and side_effect_free:
        return f"{lhs} = uint(int({lhs}) {op[:-1]} ({rhs}))"
    return None


def collect_metal_signed_builtin_members(codegen, structs):
    """``{structure: {member, ...}}`` of the ``int`` scalar members declared on one of the ``uint`` attributes."""
    members = {}
    for struct in structs or []:
        struct_name = getattr(struct, "name", None)
        if not struct_name or struct_name in codegen.metal_union_layouts:
            continue
        defaults = codegen.metal_default_struct_member_semantics(struct)
        for member in getattr(struct, "members", []) or []:
            member_name = getattr(member, "name", None)
            member_type = getattr(member, "member_type", None)
            if not member_name or member_type is None:
                continue
            if codegen.is_array_type_node(member_type):
                continue
            mapped_type = codegen.map_type(codegen.struct_member_raw_type(member))
            semantic = codegen.metal_struct_member_effective_semantic(
                member, defaults, struct_name
            )
            attribute = codegen.metal_member_semantic_attribute(
                member, semantic, struct_name, defaults
            )
            match = _ATTRIBUTE_NAME.search(attribute or "")
            if match and metal_signed_builtin_member_attribute(
                match.group(1), mapped_type
            ):
                members.setdefault(struct_name, set()).add(member_name)
    return members
