"""Values stored into a stage output structure that GLSL declares as separate outputs.

GLSL has no output structure for vertex and fragment entry points: each member of the CrossGL entry's return
structure becomes its own ``out`` variable or built-in, and a local of that structure type is not declared at all,
its member accesses naming those outputs instead. A value given to the whole local, ``VSOutput o = VSOutput(...)``
or ``o = VSOutput(...)``, must therefore be stored member by member. Generators used to drop the declaration's
initializer, so outputs the source wrote through the constructor were left undefined.

:func:`render_flattened_stage_output_store` writes the constructor's arguments to the members' outputs in member
order. Every argument is generated before the first store only when no argument reads a flattened local, so a
value that reads the outputs it replaces is refused rather than reordered, as is any value that is not a
constructor of the output structure.
"""

from __future__ import annotations

from ..ast import AssignmentNode, IdentifierNode

__all__ = ["flattened_stage_output_assignment", "render_flattened_stage_output_store"]


def flattened_stage_output_assignment(codegen, expression):
    """The flattened local a plain ``=`` assignment replaces as a whole, or ``None``."""
    if not isinstance(expression, AssignmentNode):
        return None
    if getattr(expression, "operator", "=") != "=":
        return None
    target = getattr(expression, "target", None)
    if not isinstance(target, IdentifierNode):
        return None
    name = target.name
    return name if name in codegen.flattened_stage_variables else None


def render_flattened_stage_output_store(codegen, variable_name, value, indent):
    """GLSL statements storing ``value`` into the outputs of the flattened local ``variable_name``."""
    type_name = codegen.current_stage_return_type
    struct = codegen.structs_by_name.get(type_name)
    member_map = codegen.current_stage_outputs.get(variable_name) or {}
    members = list(getattr(struct, "members", None) or []) if struct else []
    field_values = (
        codegen.stage_struct_constructor_field_values(value, members)
        if members and member_map
        else None
    )
    if field_values is None:
        _refuse(
            codegen,
            variable_name,
            type_name,
            value,
            "flattened-stage-output-value",
            f"only a constructor of '{type_name}' can be stored member by member",
        )
    for _member, argument in field_values:
        if _reads_flattened_local(codegen, argument):
            _refuse(
                codegen,
                variable_name,
                type_name,
                value,
                "flattened-stage-output-self-reference",
                "a constructor argument reads a stage output the store replaces",
            )
    indent_str = "    " * indent
    code = ""
    for member, argument in field_values:
        output_name = member_map.get(member.name)
        if output_name is None:
            _refuse(
                codegen,
                variable_name,
                type_name,
                value,
                "flattened-stage-output-member",
                f"member '{member.name}' has no GLSL stage output",
            )
        rendered = codegen.generate_expression_with_expected(
            argument, codegen.member_type_name(member)
        )
        code += f"{indent_str}{output_name} = {rendered};\n"
    return code


def _reads_flattened_local(codegen, expression):
    names = codegen.flattened_stage_variables
    for node in codegen.walk_ast(expression):
        if isinstance(node, IdentifierNode) and node.name in names:
            return True
    return False


def _refuse(codegen, variable_name, type_name, value, reason, detail):
    from .GLSL_codegen import OpenGLStructConstructionError

    raise OpenGLStructConstructionError(
        f"OpenGL cannot store a value into stage output '{variable_name}' of type "
        f"'{type_name}': GLSL declares its members as separate outputs, and {detail}",
        destination_type=type_name,
        reason=reason,
        source_location=getattr(value, "source_location", None),
    )
