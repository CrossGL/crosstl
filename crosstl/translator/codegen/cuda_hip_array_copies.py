"""Whole-array copies and initializers for CUDA and HIP.

CUDA and HIP write CrossGL arrays as C++ built-in arrays, which cannot be assigned or
initialized from another array. This module lowers those cases to element-by-element
loops (assignments) or brace initializer lists (declarations), the same approach Metal
uses in :mod:`metal_array_copies`.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
)
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
