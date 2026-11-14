"""Whole-value uses of a flattened GLSL stage-input struct.

GLSL has no struct-typed stage inputs, so an entry point's struct parameter is
flattened into one ``in`` variable per member. ``input.UV`` then reads ``UV``.
A use of the whole parameter, such as ``sampleTexture(input)``, has no
flattened spelling. It used to be emitted as the bare parameter name, which
GLSL never declared (and ``input`` is even a reserved word).

Such an entry point now starts by rebuilding the struct from its flattened
members, and whole-value uses read that local.
"""

from __future__ import annotations

import re

from ..ast import IdentifierNode, MemberAccessNode, VariableNode

__all__ = ("stage_input_struct_value_declarations",)

_CONVERTIBLE = re.compile(r"(?:[biud]?vec[234]|float|double|int|uint|bool|f16vec[234])")


def _named(node) -> str | None:
    if isinstance(node, (IdentifierNode, VariableNode)):
        name = getattr(node, "name", None)
        return name if isinstance(name, str) else None
    return None


def _whole_value_names(codegen, body) -> set[str]:
    """Names in *body* used other than as the object of a member access."""
    member_objects = set()
    nodes = list(codegen.walk_ast(body))
    for node in nodes:
        if isinstance(node, MemberAccessNode):
            member_objects.add(id(getattr(node, "object", None)))
            member_objects.add(id(getattr(node, "object_expr", None)))
    return {
        name
        for node in nodes
        if id(node) not in member_objects
        and isinstance(node, IdentifierNode)
        and (name := _named(node)) is not None
    }


def stage_input_struct_value_declarations(codegen, func) -> list[str]:
    """Declarations that rebuild stage-input structs the entry point uses whole.

    Each rebuilt parameter is aliased to its local, so later whole-value uses
    read it. Member reads keep their flattened spelling.
    """
    stage_inputs = getattr(codegen, "current_stage_inputs", None) or {}
    if not stage_inputs:
        return []
    whole_values = _whole_value_names(codegen, getattr(func, "body", []))
    declarations = []
    for param in getattr(func, "params", None) or getattr(func, "parameters", []) or []:
        name = getattr(param, "name", None)
        member_map = stage_inputs.get(name)
        if not member_map or name not in whole_values:
            continue
        struct_type = codegen.type_name_string(
            getattr(param, "vtype", getattr(param, "param_type", None))
        )
        fields = codegen.struct_member_types.get(struct_type)
        if not fields or any(field not in member_map for field in fields):
            continue
        values = []
        for field, field_type in fields.items():
            mapped_type = codegen.map_type(field_type)
            value = member_map[field]
            # Struct constructors take exact types; flattened inputs may differ.
            values.append(
                f"{mapped_type}({value})"
                if _CONVERTIBLE.fullmatch(str(mapped_type))
                else value
            )
        local = codegen.glsl_local_identifier_name(f"{name}_value")
        codegen.current_identifier_aliases[name] = local
        declarations.append(
            f"{codegen.map_type(struct_type)} {local} = "
            f"{codegen.map_type(struct_type)}({', '.join(values)});"
        )
    return declarations
