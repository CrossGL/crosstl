"""HLSL assignment of a brace list to a fixed array.

HLSL accepts a brace list only as a declaration initializer, so an assignment ``a = {e0, e1};`` is spelled as
element stores ``a[0] = e0; a[1] = e1;``. Those stores interleave with the evaluation of the list: ``e1`` is
evaluated after ``a[0]`` has been overwritten. ``a = {a[1], a[0]}`` then leaves both elements equal to the old
``a[1]`` instead of swapping them, and a call in the list may observe the partly written array.

The list is evaluated first, as the source language requires. When no element reads the destination's root
variable, and either every element is free of effects or the root is a local variable or parameter of the function
being generated, the element stores are kept, because no store can be observed: HLSL has no references, so a callee
sees a local only through an argument, and an argument that names it reads it. Otherwise the list initializes a
temporary and the array is assigned whole, which HLSL permits for arrays of the same type::

    float __crossgl_array_list_0[2] = {a[1], a[0]};
    a = __crossgl_array_list_0;
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from ..semantics.hub_types import call_name
from ..semantics.rewrite import child_slots, get_child, is_pure
from .array_utils import format_c_style_array_declaration


def hlsl_root_name(expr):
    """Name of the variable an access path such as ``a.b[i].c`` starts from, or ``None``.

    A destination that is already rendered text has no root here: HLSL may have renamed its variable, so it
    cannot be compared with the source names in the list.
    """
    while True:
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return getattr(expr, "name", None)
        if isinstance(expr, ArrayAccessNode):
            expr = expr.array_expr
        elif isinstance(expr, MemberAccessNode):
            expr = expr.object_expr
        else:
            return None


def _is_pure_value(codegen, expr):
    """:func:`is_pure`, which also admits nested lists and constructors of the program's structures."""
    if isinstance(expr, ArrayLiteralNode):
        return all(_is_pure_value(codegen, element) for element in expr.elements)
    if isinstance(expr, FunctionCallNode) and call_name(expr) in (
        getattr(codegen, "structs_by_name", None) or {}
    ):
        return all(_is_pure_value(codegen, a) for a in expr.arguments or [])
    return is_pure(expr)


def _mentions(node, name):
    if isinstance(node, (IdentifierNode, VariableNode)):
        return getattr(node, "name", None) == name
    return any(
        _mentions(get_child(node, field, key), name) for field, key in child_slots(node)
    )


def _is_function_local(codegen, name):
    """True when ``name`` is a local or parameter of the function being generated and shadows no global.

    A name that is also a global stays observable, so the answer errs towards a temporary.
    """
    local_types = getattr(codegen, "local_variable_types", None) or {}
    global_types = getattr(codegen, "global_variable_types", None) or {}
    return name in local_types and name not in global_types


def hlsl_list_needs_temporary(codegen, target, value):
    """True when storing ``value``'s elements one by one into ``target`` could change the result."""
    root = hlsl_root_name(target)
    elements = list(getattr(value, "elements", []) or [])
    if root is None:
        return bool(elements)
    observable = not _is_function_local(codegen, root)
    return any(
        _mentions(element, root)
        or (observable and not _is_pure_value(codegen, element))
        for element in elements
    )


def generate_hlsl_array_list_assignment(codegen, target, value, target_type):
    """``T t[N] = {...}; target = t;`` when element stores would be observable, else ``None``.

    ``target`` is the assignment's destination node and ``value`` its :class:`ArrayLiteralNode`.
    """
    if not hlsl_list_needs_temporary(codegen, target, value):
        return None
    mapped_type = codegen.map_type(target_type)
    initializer = codegen.hlsl_fixed_array_aggregate_initializer(value, mapped_type)
    temporary = codegen.next_hlsl_temp_variable("array_list")
    declaration = format_c_style_array_declaration(mapped_type, temporary)
    destination = (
        target if isinstance(target, str) else codegen.generate_expression(target)
    )
    return f"{declaration} = {initializer};\n{destination} = {temporary}"
