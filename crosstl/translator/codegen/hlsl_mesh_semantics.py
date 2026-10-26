"""Default HLSL semantics for mesh-stage vertex and primitive output structs.

A CrossGL mesh entry writes its vertices and primitives through ``@vertices`` and ``@primitives`` arrays of
structs. Source languages that link stages by member name or order, such as Metal (whose mesh output structs carry
no semantics apart from built-ins like ``[[position]]``), leave the other members without a semantic, and the HLSL
generator refused every such member. HLSL links a mesh stage to the pixel stage by semantic, so the generator now
gives those members the ``TEXCOORD<n>`` semantics that vertex outputs and pixel inputs already default to, in
declaration order (``hlsl_default_location_member_semantics``).

The pixel stage reads per-vertex and per-primitive attributes from one input signature, so primitive members
continue after the last ``TEXCOORD`` the vertex struct uses: a pixel input struct that lists the vertex members
and then the primitive members, as Metal's ``struct { VertexOut v; PrimitiveOut p; }`` does, gets the same
indices.
"""

from __future__ import annotations

import re

__all__ = [
    "collect_hlsl_mesh_output_struct_offsets",
    "hlsl_struct_texcoord_end",
]

_TEXCOORD = re.compile(r"TEXCOORD(\d*)", re.IGNORECASE)


def _mesh_role_struct(codegen, func, role):
    for parameter in getattr(func, "parameters", getattr(func, "params", [])) or []:
        if role not in codegen.hlsl_mesh_parameter_roles(parameter):
            continue
        base_type, _suffix = codegen.hlsl_parameter_mapped_base_and_array_suffix(
            parameter
        )
        name = str(base_type or "").split("<", 1)[0].strip()
        return codegen.structs_by_name.get(name)
    return None


def hlsl_struct_texcoord_end(codegen, struct_node, defaults):
    """One past the highest ``TEXCOORD`` index the struct's declared or default semantics occupy."""
    end = 0
    for member in getattr(struct_node, "members", []) or []:
        name = getattr(member, "name", None)
        semantic = defaults.get(name) or codegen.hlsl_struct_member_declared_semantic(
            member
        )
        if semantic is None:
            continue
        match = _TEXCOORD.fullmatch(str(codegen.hlsl_semantic_name(semantic)))
        if match is None:
            continue
        index = int(match.group(1) or 0)
        count = codegen.hlsl_struct_member_semantic_slot_count(
            struct_node.name, member, semantic
        )
        end = max(end, index + count)
    return end


def collect_hlsl_mesh_output_struct_offsets(codegen, ast):
    """``{struct name: first TEXCOORD index}`` for the vertex and primitive structs of the mesh entries."""
    offsets = {}
    for func in codegen.hlsl_stage_entry_functions(ast, "mesh"):
        vertex_struct = _mesh_role_struct(codegen, func, "vertices")
        primitive_struct = _mesh_role_struct(codegen, func, "primitives")
        first_primitive = 0
        if vertex_struct is not None:
            offsets.setdefault(vertex_struct.name, 0)
            defaults = codegen.hlsl_default_location_member_semantics(
                vertex_struct, offsets[vertex_struct.name]
            )
            first_primitive = hlsl_struct_texcoord_end(codegen, vertex_struct, defaults)
        if primitive_struct is not None and primitive_struct is not vertex_struct:
            offsets.setdefault(primitive_struct.name, first_primitive)
    return offsets
