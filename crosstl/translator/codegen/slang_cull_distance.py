"""Scalar ``gl_CullDistance`` struct members in Slang output.

A CrossGL stage struct may bind one cull distance as a scalar member: ``float cull @ gl_CullDistance``. Slang
accepts ``float`` or ``float[]`` for ``SV_CullDistance``, but slangc 2026.20 writes a scalar one as a scalar
``CullDistance`` variable, which the Vulkan specification forbids (VUID-CullDistance-CullDistance-04200: the
built-in must be an array of 32-bit floats), so spirv-val rejects the module. slangc wraps a scalar
``SV_ClipDistance`` in an array itself, so only the cull distance needs this.

Such a member is declared as the one-element array ``float cull[1] : SV_CullDistance``. A member access reads
and writes its element ``cull[0]``, and a positional or named constructor argument for it is the initializer list
``{value}``.
"""

from __future__ import annotations

__all__ = [
    "slang_cull_distance_argument",
    "slang_cull_distance_member_type",
    "slang_is_scalar_cull_distance_member",
]

_CULL_DISTANCE = "sv_culldistance"


def _member(codegen, struct_name, member_name):
    struct = getattr(codegen, "structs_by_name", {}).get(struct_name)
    for member in getattr(struct, "members", None) or []:
        if getattr(member, "name", None) == member_name:
            return member
    return None


def _is_scalar_cull_distance(codegen, member):
    semantic = codegen.semantic_from_node(member)
    if semantic is None:
        return False
    if not str(codegen.map_semantic(semantic)).lower().startswith(_CULL_DISTANCE):
        return False
    member_type = getattr(member, "member_type", None)
    if member_type is None:
        member_type = getattr(member, "vtype", None)
        rendered = member_type
    else:
        rendered = codegen.convert_type_node_to_string(member_type)
    return codegen.convert_type(rendered) == "float"


def slang_is_scalar_cull_distance_member(codegen, struct_name, member_name):
    """Whether ``struct_name.member_name`` is a scalar member bound to ``gl_CullDistance``."""
    member = _member(codegen, codegen.type_name_string(struct_name), member_name)
    return member is not None and _is_scalar_cull_distance(codegen, member)


def slang_cull_distance_member_type(codegen, member, member_type):
    """The Slang type a struct member is declared with: ``float[1]`` for a scalar cull distance."""
    if _is_scalar_cull_distance(codegen, member):
        return f"{member_type}[1]"
    return member_type


def slang_cull_distance_argument(codegen, struct_name, member_name, rendered):
    """``rendered`` as the constructor argument for ``struct_name.member_name``."""
    if slang_is_scalar_cull_distance_member(codegen, struct_name, member_name):
        return f"{{{rendered}}}"
    return rendered
