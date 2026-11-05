"""Implicit conversions from a Metal aggregate to a scalar through its conversion operator.

Copy-initializing, assigning or returning a scalar from a class value calls the class's non-explicit conversion
function and then applies a standard conversion to the destination type ([dcl.init.general]/16.7,
[class.conv.fct]). A compound assignment ``x op= y`` behaves as ``x = x op y``, so a class operand of a built-in
``op`` is converted the same way ([expr.ass]/6, [over.built]), and ``p += y`` on a pointer needs an integral ``y``
([expr.add]). Struct-member lowering turns each conversion operator into a free helper named
``Struct__operator_<type>`` (``metal_lowered_readonly_scalar_conversion``). These functions find that helper for an
operand whose type is an aggregate and return the rendered call, so code generators receive a scalar instead of
the aggregate.
"""

from __future__ import annotations

from ...translator.structure_conversions import (
    registered_structure_conversion_for_identity,
)

_COMPOUND_SCALAR_OPERATORS = {
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "<<=",
    ">>=",
    "&=",
    "|=",
    "^=",
}
_POINTER_COMPOUND_OPERATORS = {"+=", "-="}


def _scalar_type(codegen, metal_type) -> bool:
    descriptor = codegen.metal_source_overload_type_descriptor(metal_type)
    return descriptor is not None and descriptor[0] == "scalar"


def _integral_scalar_type(codegen, metal_type) -> bool:
    normalized = codegen.normalized_metal_type(codegen.resolve_type_alias(metal_type))
    return normalized in {
        "bool",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "size_t",
    }


def aggregate_scalar_conversion(codegen, target_type, value, operator="="):
    """Return ``(helper, converted_type)`` when ``value`` converts to ``target_type``.

    ``operator`` is ``=`` for initialization, assignment and return, or a compound assignment operator. The value
    must have an aggregate type with exactly one readonly scalar conversion helper, and the destination must be a
    scalar, or a pointer for ``+=`` and ``-=`` with an integral converted type.
    """
    if not target_type or value is None:
        return None
    if operator != "=" and operator not in _COMPOUND_SCALAR_OPERATORS:
        return None
    target_value = codegen.metal_source_overload_value_type(target_type)
    if target_value is None:
        return None
    destination_is_scalar = _scalar_type(codegen, target_value)
    if not destination_is_scalar and (
        operator not in _POINTER_COMPOUND_OPERATORS
        or codegen.metal_pointer_pointee_type_once(target_type) is None
    ):
        return None
    try:
        value_type = codegen.metal_source_overload_value_type(
            codegen.expression_metal_type(value)
        )
    except ValueError:
        # An operand whose type cannot be resolved is diagnosed by the regular
        # expression path, which knows the construct being lowered.
        return None
    if value_type is None:
        return None
    resolved_value = codegen.normalized_metal_type(
        codegen.resolve_type_alias(value_type)
    )
    if resolved_value not in codegen.struct_member_types:
        return None
    destination_is_pointer = not destination_is_scalar
    conversion = codegen.metal_lowered_readonly_scalar_conversion(value_type)
    if conversion is None:
        return None
    helper, converted_type = conversion
    if destination_is_pointer and not _integral_scalar_type(codegen, converted_type):
        return None
    return helper, converted_type


def render_aggregate_scalar_conversion(
    codegen, target_type, value, operator, is_main
) -> str | None:
    """Render ``value`` converted for a scalar destination, or return ``None``."""
    conversion = aggregate_scalar_conversion(codegen, target_type, value, operator)
    if conversion is None:
        return None
    helper, converted_type = conversion
    name = codegen.sanitize_identifier(codegen.function_output_name(helper))
    rendered = f"{name}({codegen.generate_expression(value, is_main)})"
    if operator != "=":
        return rendered
    target_value = codegen.metal_source_overload_value_type(target_type)
    if codegen.normalized_metal_type(
        codegen.resolve_type_alias(target_value)
    ) != codegen.normalized_metal_type(codegen.resolve_type_alias(converted_type)):
        rendered = f"{codegen.map_type(target_value)}({rendered})"
    return rendered


_BUILTIN_BINARY_OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "<<",
    ">>",
    "&",
    "|",
    "^",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
}


def refuse_aggregate_builtin_operation(codegen, expression, source_location=None):
    """Raise when a built-in operator would be emitted for two convertible aggregates.

    No built-in operator accepts two class operands ([over.built]). When both classes convert to scalars, the
    expression means either a source operator template, which is an exact match and wins, or the built-in operator
    after both conversions ([over.match.oper], [over.ics.rank]). Reaching the built-in rendering path means no
    materialized operator was found, and the code generator cannot tell whether an unmaterialized template would
    have been selected, so the expression is refused instead of being emitted verbatim.
    """
    if expression.op not in _BUILTIN_BINARY_OPERATORS:
        return
    types = [
        codegen.metal_source_overload_value_type(codegen.expression_metal_type(operand))
        for operand in (expression.left, expression.right)
    ]
    if any(metal_type is None for metal_type in types):
        return
    if not all(
        codegen.normalized_metal_type(codegen.resolve_type_alias(metal_type))
        in codegen.struct_member_types
        and codegen.metal_lowered_readonly_scalar_conversion(metal_type) is not None
        for metal_type in types
    ):
        return
    from .MetalCrossGLCodeGen import MetalSourceOverloadResolutionError

    raise MetalSourceOverloadResolutionError(
        f"operator {expression.op}",
        types,
        [],
        "both aggregate operands convert to scalars, but no materialized source "
        "operator matches them, so the selected operator is unknown",
        source_location,
    )


def _registered_structure_value(codegen, value_type) -> bool:
    resolved = codegen.normalized_metal_type(codegen.resolve_type_alias(value_type))
    return any(
        registered_structure_conversion_for_identity(identity) is not None
        for identity in (resolved, codegen.map_type(resolved))
    )


def render_explicit_aggregate_scalar_conversion(
    codegen, target_type, arguments, is_main, source_location=None
) -> str | None:
    """Render an explicit conversion of an aggregate to a scalar through its conversion helper.

    ``T(value)``, ``T{value}`` and ``(T)value`` with a scalar ``T`` and a class-typed ``value`` direct-initialize
    ``T``, so the class's conversion functions are the candidates and the one whose result needs the best standard
    conversion to ``T`` is selected ([dcl.init.general]/16.7, [over.match.conv], [over.ics.rank]). A conversion
    reached only after member lowering, such as one in a function template first instantiated from a member of a
    struct template, still names the source aggregate here, while its operator now exists only as the free helper
    ``Struct__operator_<type>``. A helper returning exactly ``T`` is chosen; otherwise the single helper is called
    and its result converted to ``T``. Several inexact helpers, or none, raise instead of emitting a scalar
    constructor over a struct value, which no target accepts.

    A value whose type is the representation of a registered structure conversion contract
    (``crosstl.translator.structure_conversions``) and that has no lowered helper keeps the scalar constructor:
    every code generator lowers that constructor through the contract's value conversion, as it did before this
    lowering existed.
    """
    arguments = list(arguments or [])
    if len(arguments) != 1 or arguments[0] is None:
        return None
    target_value = codegen.metal_source_overload_value_type(target_type)
    if target_value is None or not _scalar_type(codegen, target_value):
        return None
    value = arguments[0]
    try:
        value_type = codegen.metal_source_overload_value_type(
            codegen.expression_metal_type(value)
        )
    except ValueError:
        # The regular expression path diagnoses an operand it cannot type.
        return None
    if value_type is None:
        return None
    if (
        codegen.normalized_metal_type(codegen.resolve_type_alias(value_type))
        not in codegen.struct_member_types
    ):
        return None
    candidates = codegen.metal_lowered_readonly_scalar_conversions(value_type)
    if not candidates and _registered_structure_value(codegen, value_type):
        return None
    target_identity = codegen.metal_source_overload_type_identity(target_value)
    exact = [
        candidate
        for candidate in candidates
        if codegen.metal_source_overload_type_identity(candidate[1]) == target_identity
    ]
    selected = exact or candidates
    if len(selected) != 1:
        from .MetalCrossGLCodeGen import MetalSourceOverloadResolutionError

        raise MetalSourceOverloadResolutionError(
            str(target_type),
            [str(value_type)],
            [str(getattr(function, "name", "")) for function, _result in candidates],
            (
                "the aggregate has several readonly scalar conversions and none "
                "returns the target type exactly"
                if selected
                else "the aggregate has no readonly scalar conversion helper"
            ),
            source_location,
        )
    helper, converted_type = selected[0]
    name = codegen.sanitize_identifier(codegen.function_output_name(helper))
    rendered = f"{name}({codegen.generate_expression(value, is_main)})"
    if codegen.metal_source_overload_type_identity(converted_type) != target_identity:
        rendered = f"{codegen.map_type(target_value)}({rendered})"
    return rendered
