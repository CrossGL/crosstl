"""The WGSL ``clip_distances`` built-in output.

WGSL has user clip distances only as the vertex shader output ``@builtin(clip_distances)``, of type
``array<f32, N>`` with N at most 8 (WGSL section 13.3.1.1, "clip_distances"), in a module that starts with
``enable clip_distances;`` (WGSL section 4.1.1; WebGPU feature ``"clip-distances"``). CrossGL binds a ``float``
member (one distance, as an HLSL ``float`` ``SV_ClipDistance`` or an MSL ``float [[clip_distance]]``) or a
``float[N]`` member to ``gl_ClipDistance``.

A ``float[N]`` member keeps its type. A ``float`` member is declared ``array<f32, 1>``, and every access to it reads
or writes element 0, so the shader computes with the value it declared. The built-in as a stage input, in another
stage, with another type or with more than 8 distances is refused rather than handed to WGSL, which rejects it.
"""

from __future__ import annotations

from ..ast import ArrayType, LiteralNode, MemberAccessNode, PrimitiveType
from .wgsl_stage_io import WGSLStageIOUnsupportedError

__all__ = [
    "CLIP_DISTANCES_ENABLE_DIRECTIVE",
    "CLIP_DISTANCES_SEMANTICS",
    "binds_clip_distances",
    "clip_distances_member_type",
    "collect_clip_distance_members",
    "is_scalar_clip_distance_access",
    "refuse_misplaced_clip_distances",
    "scalar_clip_distance_value",
]

CLIP_DISTANCES_ENABLE_DIRECTIVE = "enable clip_distances;"
_BUILTIN = "@builtin(clip_distances)"
_MAX_DISTANCES = 8
# Normalized semantic keys (WGSLCodeGen.semantic_key) bound to the built-in. SV_ClipDistance1 names a second
# HLSL register of distances, which a single WGSL array does not follow, so it stays unmapped and refused.
CLIP_DISTANCES_SEMANTICS = {
    "clip_distances": "clip_distances",
    "clip_distance": "clip_distances",
    "gl_clipdistance": "clip_distances",
    "sv_clipdistance": "clip_distances",
    "sv_clipdistance0": "clip_distances",
}


def binds_clip_distances(attributes):
    """Return whether rendered WGSL attributes bind the clip distances."""
    return _BUILTIN in (attributes or "")


def collect_clip_distance_members(codegen, structs):
    """Record the ``(struct, member)`` pairs bound to ``clip_distances``, and which of them are ``float``.

    The members are known before any function is generated, so every access sees them. They are not split into one
    member per element as other stage-interface arrays are (:meth:`stage_interface_array_member_info`).
    """
    codegen._clip_distance_members = set()
    codegen._scalar_clip_distance_members = set()
    for struct in structs:
        struct_name = getattr(struct, "name", "")
        for member in getattr(struct, "members", []) or []:
            if not binds_clip_distances(codegen.wgsl_attributes(member.attributes)):
                continue
            key = (struct_name, member.name)
            codegen._clip_distance_members.add(key)
            if codegen.type_name_string(member.member_type) == "f32":
                codegen._scalar_clip_distance_members.add(key)


def clip_distances_member_type(codegen, struct_name, member):
    """The CrossGL type of a member bound to ``clip_distances``: ``float[N]`` for N from 1 to 8.

    A ``float`` member becomes ``float[1]``; its accesses use element 0 (:func:`is_scalar_clip_distance_access`).
    """
    type_name = codegen.type_name_string(member.member_type)
    codegen._uses_clip_distances = True
    if type_name == "f32":
        return ArrayType(PrimitiveType("float"), LiteralNode(1, PrimitiveType("int")))
    member_type = member.member_type
    if isinstance(member_type, ArrayType) and (
        codegen.type_name_string(member_type.element_type) == "f32"
    ):
        size = codegen.resolve_integer_expression(member_type.size)
        if size is not None and 1 <= size <= _MAX_DISTANCES:
            return member_type
    raise WGSLStageIOUnsupportedError(
        f"WGSL target cannot bind member {struct_name}.{member.name} of type {type_name} to "
        f"clip_distances, which is array<f32, N> with N from 1 to {_MAX_DISTANCES}; declare it "
        "float or float[N]"
    )


def is_scalar_clip_distance_access(codegen, expr):
    """Whether ``expr`` reads or writes a ``float`` member that WGSL declares ``array<f32, 1>``."""
    if not codegen._scalar_clip_distance_members or not isinstance(
        expr, MemberAccessNode
    ):
        return False
    object_type = codegen.array_element_type(codegen.expression_type(expr.object_expr))
    key = (codegen.struct_type_name(object_type), expr.member)
    return key in codegen._scalar_clip_distance_members


def scalar_clip_distance_value(codegen, struct_name, member_name, rendered):
    """``rendered`` as ``array<f32, 1>(rendered)`` when it constructs a ``float`` clip distance member."""
    if (struct_name, member_name) in codegen._scalar_clip_distance_members:
        return f"array<f32, 1>({rendered})"
    return rendered


def refuse_misplaced_clip_distances(codegen, entry_point, stage_name):
    """Refuse the clip distances anywhere but in the output structure of a vertex entry point."""
    structs = codegen._stage_interface_structs_by_name
    for parameter in getattr(entry_point, "parameters", []) or []:
        struct = structs.get(
            codegen.struct_type_name(getattr(parameter, "param_type", None))
        )
        if struct is not None:
            _refuse_struct(codegen, struct, stage_name, "input")
        elif binds_clip_distances(codegen.wgsl_attributes(parameter.attributes)):
            raise _stage_error(f"{stage_name} entry-point parameter '{parameter.name}'")
    struct = structs.get(
        codegen.struct_type_name(getattr(entry_point, "return_type", None))
    )
    if struct is not None and stage_name != "vertex":
        _refuse_struct(codegen, struct, stage_name, "output")


def _refuse_struct(codegen, struct, stage_name, direction):
    for member in getattr(struct, "members", []) or []:
        if binds_clip_distances(codegen.wgsl_attributes(member.attributes)):
            raise _stage_error(
                f"{stage_name} {direction} member {struct.name}.{member.name}"
            )


def _stage_error(context):
    return WGSLStageIOUnsupportedError(
        f"WGSL target cannot bind {context} to clip_distances; WGSL has the clip "
        "distances only as a vertex shader output"
    )
