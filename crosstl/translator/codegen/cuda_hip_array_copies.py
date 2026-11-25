"""Whole-array copies and initializers for CUDA and HIP.

CUDA and HIP write CrossGL arrays as C++ built-in arrays, which cannot be assigned or
initialized from another array. This module lowers those cases to element-by-element
loops (assignments) or brace initializer lists (declarations), the same approach Metal
uses in :mod:`metal_array_copies`. An assignment of a brace list stores a list of
constants element by element, and initializes a temporary array from any other list.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    UnaryOpNode,
    VariableNode,
)
from ..semantics.hub_types import call_name, is_pure_callee
from ..semantics.rewrite import is_pure

__all__ = [
    "generate_cpp_array_copy",
    "generate_cpp_array_initializer",
]

_ARRAY_TYPE = re.compile(
    r"^(?P<element>.*?)\s*\[\s*(?P<extent>\d+)\s*\](?P<rest>(?:\s*\[\s*\d+\s*\])*)\s*$"
)


def _array_parts(type_name):
    if not isinstance(type_name, str):
        return None
    match = _ARRAY_TYPE.match(type_name)
    if match is None:
        return None
    return match.group("element").strip() + match.group("rest"), int(
        match.group("extent")
    )


def _fresh_array_index(codegen):
    index = getattr(codegen, "_cpp_array_copy_index", 0)
    codegen._cpp_array_copy_index = index + 1
    return f"__crossgl_array_index_{index}"


def generate_cpp_array_copy(codegen, target, value):
    """Statement lines that copy array ``value`` into array ``target`` element by element, or ``None``.

    ``None`` means ``target`` is not a C array, or the value is a function call and
    should be handled by the array-constructor lowering instead.
    """
    target_type = codegen.type_name_string(codegen.expression_result_type(target))
    if _array_parts(target_type) is None:
        return None
    if isinstance(value, ArrayLiteralNode):
        return _array_literal_assignment(codegen, target, value, target_type)
    value_type = codegen.type_name_string(codegen.expression_result_type(value))
    if _array_parts(value_type) is None:
        return None

    if isinstance(value, FunctionCallNode):
        # Leave array-constructor and function-return assignments to the dedicated pass.
        return None

    if not (is_pure(target) and is_pure(value)):
        return codegen.refuse_lowering(
            f"array assignment: '{target_type}' values are copied element by element in "
            f"{codegen.__class__.__name__.replace('CodeGen', '')}, which evaluates both "
            "sides once per element, and this assignment's operands have effects"
        )

    base_indent = "    " * codegen.indent_level
    lines = _copy_lines(codegen, target, value, target_type, base_indent)
    return "\n".join(lines)


def _copy_lines(codegen, target_expr, value_expr, target_type, base_indent):
    element_type, extent = _array_parts(target_type)
    index = _fresh_array_index(codegen)
    target_access = ArrayAccessNode(target_expr, IdentifierNode(index))
    value_access = ArrayAccessNode(value_expr, IdentifierNode(index))

    lines = [f"{base_indent}for (int {index} = 0; {index} < {extent}; ++{index}) {{"]
    if _array_parts(element_type) is not None:
        lines.extend(
            _copy_lines(
                codegen,
                target_access,
                value_access,
                element_type,
                base_indent + "    ",
            )
        )
    else:
        lines.append(
            f"{base_indent}    {codegen.visit(target_access)} = {codegen.visit(value_access)};"
        )
    lines.append(f"{base_indent}}}")
    return lines


def _is_constant_element(expr):
    """Whether ``expr`` reads no storage: literals combined by operators, casts and value constructors."""
    if isinstance(expr, LiteralNode):
        return True
    if isinstance(expr, ArrayLiteralNode):
        return all(_is_constant_element(element) for element in expr.elements)
    if isinstance(expr, UnaryOpNode):
        return expr.op not in ("++", "--") and _is_constant_element(expr.operand)
    if isinstance(expr, BinaryOpNode):
        return _is_constant_element(expr.left) and _is_constant_element(expr.right)
    if isinstance(expr, CastNode):
        return _is_constant_element(expr.expression)
    if isinstance(expr, FunctionCallNode):
        # A value constructor such as vec2(0.5, 1.0) or a math built-in; any other call may read storage.
        return is_pure_callee(call_name(expr)) and all(
            _is_constant_element(argument) for argument in expr.arguments or []
        )
    return False


def _exact_literal(literal, target_type):
    """Whether ``literal`` gives every element of array ``target_type``, at every level of nesting."""
    parts = _array_parts(target_type)
    if parts is None:
        return not isinstance(literal, ArrayLiteralNode)
    element_type, extent = parts
    if not isinstance(literal, ArrayLiteralNode) or len(literal.elements) != extent:
        return False
    return all(_exact_literal(element, element_type) for element in literal.elements)


def _array_literal_assignment(codegen, target, literal, target_type):
    """Statement lines that assign brace list ``literal`` to array ``target``.

    A C++ built-in array cannot be assigned, so a list of constants that names every element
    is stored element by element. Any other list initializes a temporary array, as the
    declaration ``T a[n] = {...}`` would, which is then copied: the list's elements are
    evaluated once and in order, an element that reads ``target`` reads its old value, and
    elements the list leaves out are value-initialized as in C++ aggregate initialization.
    """
    if not is_pure(target):
        return codegen.refuse_lowering(
            f"array assignment: '{target_type}' values are assigned element by element in "
            f"{codegen.__class__.__name__.replace('CodeGen', '')}, which evaluates the "
            "assigned array once per element, and this assignment's target has effects"
        )
    base_indent = "    " * codegen.indent_level
    if _exact_literal(literal, target_type) and _is_constant_element(literal):
        return "\n".join(
            _literal_lines(codegen, target, literal, target_type, base_indent)
        )
    temporary = _fresh_array_value(codegen)
    declaration = codegen.format_variable_declaration(
        VariableNode(
            temporary,
            codegen.expression_result_type(target),
            initial_value=literal,
        )
    )
    lines = [f"{base_indent}{{", f"{base_indent}    {declaration};"]
    lines.extend(
        _copy_lines(
            codegen,
            target,
            IdentifierNode(temporary),
            target_type,
            base_indent + "    ",
        )
    )
    lines.append(f"{base_indent}}}")
    return "\n".join(lines)


def _fresh_array_value(codegen):
    index = getattr(codegen, "_cpp_array_value_index", 0)
    codegen._cpp_array_value_index = index + 1
    return f"__crossgl_array_value_{index}"


def _literal_lines(codegen, target_expr, literal, target_type, base_indent):
    element_type, _extent = _array_parts(target_type)
    lines = []
    for index, element in enumerate(literal.elements):
        access = ArrayAccessNode(target_expr, LiteralNode(index, PrimitiveType("int")))
        if _array_parts(element_type) is not None:
            lines.extend(
                _literal_lines(codegen, access, element, element_type, base_indent)
            )
        else:
            lines.append(
                f"{base_indent}{codegen.visit(access)} = {codegen.visit(element)};"
            )
    return lines


def generate_cpp_array_initializer(codegen, expr, field_type):
    """Brace list initializing array ``field_type`` from array value ``expr``, or ``None``.

    ``None`` means the field is not an array, ``expr`` is already an initializer list or
    function call, or its element type is not a C array.
    """
    field_type_name = codegen.type_name_string(field_type)
    parts = _array_parts(field_type_name)
    if (
        parts is None
        or expr is None
        or isinstance(expr, (ArrayLiteralNode, FunctionCallNode))
    ):
        return None
    if (
        _array_parts(codegen.type_name_string(codegen.expression_result_type(expr)))
        is None
    ):
        return None

    if not is_pure(expr):
        return codegen.refuse_lowering(
            f"array initializer: '{field_type_name}' members are initialized element by "
            f"element in {codegen.__class__.__name__.replace('CodeGen', '')}, which "
            "evaluates the array value once per element, and this value has effects"
        )

    return _initializer_elements(codegen, expr, parts)


def _initializer_elements(codegen, expr, parts):
    element_type, extent = parts
    elements = []
    for index in range(extent):
        element = ArrayAccessNode(expr, LiteralNode(index, PrimitiveType("int")))
        nested = _array_parts(element_type)
        if nested is not None:
            rendered = _initializer_elements(codegen, element, nested)
        else:
            rendered = codegen.visit(element)
        elements.append(rendered)
    return "{" + ", ".join(elements) + "}"
