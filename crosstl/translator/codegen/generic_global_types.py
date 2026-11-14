"""Types of global declarations that generic-call deduction needs before declarations are lowered.

``generic<T> T twice(T x)`` called as ``twice(scale)`` binds ``T`` from the type of ``scale``. Generators collect every
generic specialization before they lower the program's declarations, and only register global types while they
emit, so a uniform or storage buffer argument had no type during collection. The call was then refused as an
unspecialized generic, or resolved only while it was emitted, to a specialization that was never defined.

Each function here spells a declaration the way its generator spells it while emitting, so the specialization keys
collection creates are the keys emission looks up. Array declarations are left out: their extents are folded from
constants that are not lowered yet, so their calls keep resolving during emission.
"""

from __future__ import annotations

from ..ast import ArrayNode, ArrayType, PointerType, ReferenceType
from .stage_utils import collect_stage_local_variables

__all__ = [
    "collect_constant_and_cbuffer_member_types",
    "glsl_preliminary_global_variable_types",
    "metal_generic_global_types",
]


def _is_array_declaration(node, node_type):
    if isinstance(node, ArrayNode) or isinstance(node_type, ArrayType):
        return True
    return isinstance(node_type, str) and "[" in node_type


def glsl_preliminary_global_variable_types(codegen, ast, target_stage):
    """The OpenGL global type table for the non-array declarations visible to ``target_stage``.

    It has the entries the complete table, built once declarations are lowered, gives the same declarations.
    """
    declarations = [
        node
        for node in list(getattr(ast, "global_variables", []) or [])
        if not codegen.is_elided_glsl_type_alias_global(node)
    ]
    declarations += collect_stage_local_variables(
        ast,
        target_stage,
        lambda node: codegen.is_stage_local_resource_variable(node)
        or codegen.is_stage_local_interface_variable(node),
    )
    variable_types = {}
    for node in declarations:
        name = codegen.resource_node_name(node)
        if not name or name in variable_types:
            continue
        vtype, _array_size, array_suffix, _count = codegen.resource_declaration_shape(
            node
        )
        variable_types[name] = f"{codegen.type_name_string(vtype)}{array_suffix}"
    return variable_types


def metal_generic_global_types(codegen, global_vars):
    """``{name: type}`` for the Metal globals whose written type is a value, texture or buffer type.

    The Metal generator passes these globals to the functions that use them, and registers each parameter with
    the type it converts the declaration's written type to. Pointer and reference globals are left out, because
    their parameters take the type of the buffer they view.
    """
    variable_types = {}
    for node in global_vars or []:
        name = getattr(node, "name", None)
        node_type = getattr(node, "var_type", None)
        if (
            not name
            or name in variable_types
            or node_type is None
            or isinstance(node_type, (PointerType, ReferenceType))
        ):
            continue
        type_name = codegen.type_name_string(
            codegen.convert_type_node_to_string(node_type)
        )
        if type_name:
            variable_types[name] = type_name
    return variable_types


def collect_constant_and_cbuffer_member_types(
    codegen,
    ast,
    *,
    cbuffers=None,
    type_renderer=None,
):
    """Collect ``{name: type}`` for shader-scope constants and cbuffer members."""
    render = type_renderer or codegen.type_name_string
    types = {}
    for constant in getattr(ast, "constants", []) or []:
        name = getattr(constant, "name", None)
        if not name or name in types:
            continue
        const_type = getattr(
            constant,
            "const_type",
            getattr(constant, "var_type", getattr(constant, "vtype", None)),
        )
        type_name = render(const_type) if const_type is not None else None
        if type_name:
            types[name] = type_name

    if cbuffers is None:
        cbuffers = list(getattr(ast, "cbuffers", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            cbuffers.extend(getattr(stage, "local_cbuffers", []) or [])

    for cbuffer in cbuffers or []:
        for member in getattr(cbuffer, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in types:
                continue
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member,
                    "element_type",
                    getattr(member, "member_type", getattr(member, "vtype", None)),
                )
                size = getattr(member, "size", None)
                rendered = render(element_type) if element_type is not None else None
                type_name = f"{rendered}[{size or ''}]" if rendered else None
            else:
                member_type = getattr(
                    member,
                    "member_type",
                    getattr(member, "var_type", getattr(member, "vtype", None)),
                )
                type_name = render(member_type) if member_type is not None else None
            if type_name:
                types[member_name] = type_name
    return types
