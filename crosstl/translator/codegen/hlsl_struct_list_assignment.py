"""HLSL assignment of a brace list to a structure.

A CrossGL assignment may store a brace list into a structure, ``s = {float[2](1.0, 2.0), 3.0};``. The stage-scope
pass writes one at the start of the entry point for a stage constant whose initializer reads a stage value
(``crosstl/translator/semantics/stage_private_variables.py``). HLSL accepts a brace list only as a declaration
initializer, so the generator used to pass the list's members to the structure's constructor helper,
``__crossgl_construct_W({1.0, 2.0}, 3.0)``, which DXC rejects.

The list instead initializes a temporary declared through the generator's own declaration path, which writes the
members in order and spells array members as HLSL can, and the temporary is then assigned whole::

    W __crossgl_struct_list_0;
    float __crossgl_array_list_1[2] = {1.0, 2.0};
    __crossgl_struct_list_0.m = __crossgl_array_list_1;
    __crossgl_struct_list_0.s = 3.0;
    s = __crossgl_struct_list_0;

Every member is evaluated before the destination is written, as the source requires, so a list that reads the
destination (``s = {s.m, s.s + 1.0}``) sees its old value.
"""

from __future__ import annotations

from ..ast import ArrayLiteralNode, VariableNode

__all__ = ["generate_hlsl_struct_list_assignment"]


def _structure_type(codegen, target_type):
    """The program structure ``target_type`` names, or ``None`` for any other type."""
    if target_type is None or codegen.hlsl_outer_array_type(target_type):
        return None
    type_name = codegen.type_name_string(target_type)
    if not type_name or not codegen.hlsl_struct_constructor_fields(type_name):
        return None
    return type_name


def generate_hlsl_struct_list_assignment(
    codegen, target, value, op, target_type, statement_context
):
    """``T t = {...}; target = t`` for a brace list assigned to a structure, else ``None``.

    An assignment that is not a statement of its own has nowhere to declare the temporary, so it is refused.
    """
    if not isinstance(value, ArrayLiteralNode):
        return None
    type_name = _structure_type(codegen, target_type)
    if type_name is None:
        return None
    if op != "=" or not statement_context:
        codegen.hlsl_aggregate_error(
            value,
            expected_type=type_name,
            reason="struct-list-assignment-unsupported",
            detail=(
                "a brace list is assigned to a structure inside an expression; HLSL builds a structure "
                "from a list only in a declaration"
            ),
        )
    temporary = codegen.next_hlsl_temp_variable("struct_list")
    declaration = VariableNode(temporary, type_name, initial_value=value)
    code = codegen.generate_statement(declaration, 0)
    destination = (
        target if isinstance(target, str) else codegen.generate_expression(target)
    )
    return f"{code.rstrip()}\n{destination} = {temporary}"
