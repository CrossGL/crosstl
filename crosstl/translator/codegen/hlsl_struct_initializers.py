"""Brace initializers for HLSL struct constants whose array fields take literals.

HLSL has no array literal expression. A struct constructor is emitted as a call
to a generated ``__crossgl_construct_T`` helper, so a field initialized with an
array literal, as in ``static const P S1 = P(2.0, {6.0, 7.0})``, would pass
``{6.0, 7.0}`` as a call argument, which is not HLSL.

A declaration initializer can still use an initializer list instead:
``static const P S1 = {2.0, {6.0, 7.0}};``. This works for SM 5 and SM 6
compilers alike. Nested struct constructors become nested lists.
"""

from __future__ import annotations

from ..ast import ArrayLiteralNode, ConstructorNode, FunctionCallNode

__all__ = (
    "hlsl_array_literal_constructor_argument",
    "hlsl_brace_struct_initializer",
)


def _constructor_fields(codegen, expr):
    """Return ``(fields, arguments)`` for a positional struct constructor, or ``None``."""
    if isinstance(expr, ConstructorNode):
        type_name = codegen.type_name_string(getattr(expr, "constructor_type", None))
    elif isinstance(expr, FunctionCallNode):
        type_name = codegen.function_call_name(expr)
        if type_name in getattr(codegen, "function_return_types", {}):
            return None
    else:
        return None
    if getattr(expr, "named_arguments", None):
        return None
    fields = codegen.hlsl_struct_constructor_fields(type_name)
    arguments = list(getattr(expr, "arguments", None) or [])
    if fields is None or not fields or len(fields) != len(arguments):
        return None
    return fields, arguments


def hlsl_array_literal_constructor_argument(codegen, expr):
    """Whether a struct constructor (or one nested in it) takes an array literal field."""
    parts = _constructor_fields(codegen, expr)
    if parts is None:
        return False
    for (_name, field_type), argument in zip(*parts):
        if isinstance(argument, ArrayLiteralNode) and codegen.hlsl_outer_array_type(
            field_type
        ):
            return True
        if hlsl_array_literal_constructor_argument(codegen, argument):
            return True
    return False


def _render(codegen, expr, expected_type):
    parts = _constructor_fields(codegen, expr)
    if parts is None:
        if isinstance(expr, ArrayLiteralNode) and codegen.hlsl_outer_array_type(
            expected_type
        ):
            return codegen.hlsl_fixed_array_aggregate_initializer(expr, expected_type)
        return codegen.generate_constant_expression(expr, expected_type)
    fields, arguments = parts
    rendered = [
        _render(codegen, argument, field_type)
        for (_name, field_type), argument in zip(fields, arguments)
    ]
    return "{" + ", ".join(rendered) + "}"


def hlsl_brace_struct_initializer(codegen, expr, expected_type):
    """Initializer-list text for *expr* when its struct fields need array literals.

    Returns ``None`` when *expr* is not such a constructor, so the ordinary
    expression rendering applies.
    """
    if not hlsl_array_literal_constructor_argument(codegen, expr):
        return None
    return _render(codegen, expr, expected_type)
