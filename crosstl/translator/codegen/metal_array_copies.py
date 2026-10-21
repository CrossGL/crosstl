"""Whole-array copies for Metal, whose C arrays are not assignable.

GLSL, HLSL and SPIR-V treat a sized array as a value: ``a = b`` copies every element (GLSL 4.60 section 5.8,
"Assignments"; HLSL arrays assign by value; ``OpStore`` of an array type). The Metal emitter declares arrays with
C syntax, ``float3 a[2]``, and MSL's C++ built-in arrays cannot be assigned or initialized from another array (MSL
specification section 2.1 builds on C++14, [expr.ass]/[dcl.init.aggr]). Both forms used to reach the Metal
compiler and fail there.

:func:`generate_metal_array_copy` writes such an assignment as a loop over the leading extent that assigns one
element at a time. The element assignment goes back through the emitter, so nested arrays become nested loops and
stage-interface members lowered to one field per element are written through their own lowering. The loop
evaluates both sides once per element, so both must be free of effects; any other array copy is refused.

Initializers have the same restriction: neither ``float3 b[2] = a`` nor ``S{a, 1.0}`` can initialize an array
from the array ``a``. :func:`generate_metal_array_initializer` spells such an initializer as the brace list of its
elements, ``{a[0], a[1]}``, which also suits a ``const`` declaration. A stage-interface structure whose array or
matrix member the emitter splits into one field per element or column takes that member's elements as consecutive
initializers instead (:func:`render_metal_split_member_constructor_arguments`).
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
)
from ..semantics.rewrite import is_pure

__all__ = [
    "generate_metal_array_copy",
    "generate_metal_array_initializer",
    "metal_array_extent",
    "render_metal_split_member_constructor_arguments",
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


def _element(expr, index):
    return ArrayAccessNode(expr, LiteralNode(index, PrimitiveType("int")))


def metal_array_extent(type_name):
    """Leading element count of a C-array type spelling, ``2`` for ``float3[2]``, or ``None`` for other types."""
    if not isinstance(type_name, str):
        return None
    match = _ARRAY_TYPE.match(type_name)
    if match is None:
        return None
    return int(match.group("extent"))


def generate_metal_array_copy(codegen, target, value):
    """Statement lines that copy array ``value`` into array ``target`` element by element, or ``None``.

    ``None`` means ``target`` is not a C array, so ordinary assignment applies.
    """
    target_type = codegen.type_name_string(codegen.expression_result_type(target))
    extent = metal_array_extent(target_type)
    if extent is None:
        return None
    if not (is_pure(target) and is_pure(value)):
        return codegen.refuse_lowering(
            f"array assignment: '{target_type}' values are copied element by element in Metal, which evaluates "
            "both sides once per element, and this assignment's operands have effects"
        )
    index = codegen.next_metal_temp_variable("array_index")
    element = AssignmentNode(
        ArrayAccessNode(target, IdentifierNode(index)),
        ArrayAccessNode(value, IdentifierNode(index)),
        "=",
    )
    body = codegen.generate_assignment(element)
    return "\n".join(
        [f"for (int {index} = 0; {index} < {extent}; ++{index}) {{", str(body), "}"]
    )


def generate_metal_array_initializer(codegen, expr, field_type):
    """Brace list initializing array field ``field_type`` from array value ``expr``, or ``None``.

    ``None`` means the field is not an array or ``expr`` is already an initializer list.
    """
    parts = _array_parts(codegen.type_name_string(field_type))
    if parts is None or expr is None or isinstance(expr, ArrayLiteralNode):
        return None
    if (
        _array_parts(codegen.type_name_string(codegen.expression_result_type(expr)))
        is None
    ):
        return None
    if not is_pure(expr):
        return codegen.refuse_lowering(
            f"array member initializer: '{codegen.type_name_string(field_type)}' members are initialized element "
            "by element in Metal, which evaluates the array value once per element, and this value has effects"
        )
    return "{" + ", ".join(_array_elements(codegen, expr, parts)) + "}"


def _array_elements(codegen, expr, parts):
    element_type, extent = parts
    rendered = []
    for index in range(extent):
        element = _element(expr, index)
        nested = _array_parts(element_type)
        if nested is not None:
            rendered.append(
                "{" + ", ".join(_array_elements(codegen, element, nested)) + "}"
            )
        else:
            rendered.append(
                codegen.generate_expression_with_expected(element, element_type)
            )
    return rendered


def render_metal_split_member_constructor_arguments(codegen, expr, fields, lowerings):
    """Initializers for a structure whose members the emitter split into per-element fields, or ``None``.

    ``fields`` lists the structure's source members as ``(name, type)`` and ``lowerings`` maps a split member to
    its lowering (``kind`` ``"array"`` or ``"matrix"``, with one entry in ``fields`` per emitted field). Returns
    ``None`` when no argument initializes a split member or the call names its arguments, so the ordinary
    rendering applies.
    """
    arguments = list(getattr(expr, "arguments", None) or [])
    if getattr(expr, "named_arguments", None) or len(arguments) != len(fields):
        return None
    if not any(name in lowerings for name, _type in fields):
        return None
    rendered = []
    for (name, field_type), argument in zip(fields, arguments):
        lowering = lowerings.get(name)
        if lowering is None:
            custom = codegen.generate_struct_constructor_argument(argument, field_type)
            rendered.append(
                custom
                if custom is not None
                else codegen.generate_expression_with_expected(argument, field_type)
            )
            continue
        rendered.extend(_split_member_initializers(codegen, name, argument, lowering))
    return rendered


def _split_member_initializers(codegen, name, argument, lowering):
    split_fields = lowering.get("fields") or []
    if lowering.get("kind") == "array" and isinstance(argument, ArrayLiteralNode):
        elements = list(argument.elements or [])
        if len(elements) != len(split_fields):
            return [
                codegen.refuse_lowering(
                    f"stage interface member '{name}': {len(elements)} initializers for "
                    f"{len(split_fields)} elements"
                )
            ]
        return [
            codegen.generate_expression_with_expected(element, field_type)
            for element, (_field, field_type) in zip(elements, split_fields)
        ]
    if not is_pure(argument):
        return [
            codegen.refuse_lowering(
                f"stage interface member '{name}': Metal stores each element or column in its own field, so the "
                "initializer is evaluated once per field, and this initializer has effects"
            )
        ]
    return [
        codegen.generate_expression_with_expected(_element(argument, index), field_type)
        for index, (_field, field_type) in enumerate(split_fields)
    ]
