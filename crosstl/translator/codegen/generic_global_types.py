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
