"""Locations of the user varyings of a GLSL mesh pipeline.

GL_EXT_mesh_shader exists only for SPIR-V generation, so a mesh shader and the fragment shader it feeds compile
under Vulkan rules, where every user input and output needs a ``location``. The generator left mesh outputs and
fragment inputs without a classic semantic unlocated, so glslang rejected the mesh shader. They now take locations
the way SPIR-V output places them (:mod:`.stage_io_semantic_locations`): a member with an explicit location or a
classic semantic keeps it, and the others take, in declaration order, the first free locations. A mesh entry's
vertex members come first and its primitive members after them, as SPIR-V and HLSL output order them, so a
fragment input structure that lists the vertex members and then the primitive members matches.
"""

from __future__ import annotations

from .glsl_vertex_inputs import allocated_stage_layouts

__all__ = ["mesh_fragment_input_layouts", "mesh_output_layouts"]


def _member_type(node):
    return getattr(node, "member_type", None)


def mesh_output_layouts(codegen, mesh_outputs) -> dict:
    """``(role, output name) -> layout`` of a mesh entry's user vertex and primitive outputs."""
    nodes, keys = [], []
    for role in ("vertices", "primitives"):
        for info in mesh_outputs.values():
            if info["role"] != role:
                continue
            for member_info in info["members"].values():
                semantic = member_info["semantic"]
                if role == "vertices" and codegen.is_vertex_builtin_output(semantic):
                    continue
                if role == "primitives" and codegen.is_mesh_primitive_builtin(semantic):
                    continue
                nodes.append(member_info["node"])
                keys.append((role, member_info["output_name"]))
    return dict(zip(keys, allocated_stage_layouts(codegen, nodes, _member_type)))


def mesh_fragment_input_layouts(codegen, members) -> dict:
    """``member name -> layout`` of a fragment input structure's user members in a mesh pipeline."""
    return {
        member.name: layout
        for member, layout in zip(
            members, allocated_stage_layouts(codegen, members, _member_type)
        )
    }
