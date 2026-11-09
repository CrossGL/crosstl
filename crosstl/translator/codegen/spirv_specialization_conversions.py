"""Type conversions of specialization constants inside SPIR-V module-scope constants.

A module-scope constant whose value follows a specialization constant is a constant instruction, so a conversion
inside it must be an ``OpSpecConstantOp``. Under the ``Shader`` capability that instruction allows ``OpSConvert``,
``OpUConvert`` (SPIR-V 1.4 and later), ``OpFConvert``, integer arithmetic, ``OpSelect`` and the integer comparisons,
but no conversion between integers and floating point: ``OpConvertSToF``, ``OpConvertUToF``, ``OpConvertFToS`` and
``OpConvertFToU`` are ``Kernel``-only (SPIR-V specification, ``OpSpecConstantOp``). glslang accordingly folds
``layout(constant_id = 0) const int n = 3; const uint u = uint(n);`` into ``OpSpecConstantOp %uint IAdd %n %uint_0``
and rejects ``const vec2 v = vec2(n, 1.0);`` with "global const initializers must be constant".

The generator used to place the integer constant itself into a ``float`` or ``uint`` composite, which ``spirv-val``
rejects because a constituent's type must match the element type. This module lowers the conversions that a
specialization-constant operation can express (integer signedness, integer narrowing, signed integer widening,
floating-point width, integer to Boolean and Boolean to integer) and refuses the rest with a typed diagnostic instead
of emitting an invalid module.
"""

from __future__ import annotations

from ..ast import CastNode, FunctionCallNode, IdentifierNode, PrimitiveType

INTEGER_WIDTHS = {"int": 32, "uint": 32, "i64": 64, "u64": 64}
FLOAT_WIDTHS = {"float": 32, "double": 64}
SCALAR_CONVERSION_TARGETS = (
    frozenset(INTEGER_WIDTHS) | frozenset(FLOAT_WIDTHS) | {"bool"}
)


def scalar_type_name(codegen, type_id) -> str | None:
    """Return the normalized scalar type name of a registered type, or None for composites."""
    if type_id is None:
        return None
    base_type = getattr(getattr(type_id, "type", None), "base_type", None)
    if not isinstance(base_type, str):
        return None
    name = codegen.normalize_primitive_name(base_type)
    return name if name in SCALAR_CONVERSION_TARGETS else None


KERNEL_ONLY = (
    "conversions between integer or Boolean and floating-point values are Kernel-only"
)
UNSIGNED_WIDENING = (
    "zero-extending an unsigned value needs OpUConvert, which OpSpecConstantOp allows "
    "under the Shader capability only from SPIR-V 1.4"
)


def convert_specialization_scalar(codegen, value, target_type):
    """Convert a scalar specialization-constant value to ``target_type`` with ``OpSpecConstantOp``.

    Returns the converted id (``value`` itself when the types already match), or a string naming why no
    specialization-constant operation under the ``Shader`` capability expresses the conversion.
    """
    source_type = codegen.registered_value_type(value)
    target_type = codegen.ensure_registered_type(target_type)
    if source_type is not None and source_type.id == target_type.id:
        return value
    source = scalar_type_name(codegen, source_type)
    target = scalar_type_name(codegen, target_type)
    if source is None or target is None:
        return "the operand or result is not a scalar"
    if source == target:
        return value
    if source in INTEGER_WIDTHS and target in INTEGER_WIDTHS:
        if INTEGER_WIDTHS[source] == INTEGER_WIDTHS[target]:
            zero = codegen.register_constant(0, target_type)
            return codegen.emit_spec_constant_op("IAdd", target_type, [value, zero])
        narrowing = INTEGER_WIDTHS[source] > INTEGER_WIDTHS[target]
        # Truncation keeps the low bits whatever the signedness, so OpSConvert narrows an unsigned value too.
        if narrowing or source in codegen.SIGNED_INTEGER_TYPES:
            return codegen.emit_spec_constant_op("SConvert", target_type, [value])
        return UNSIGNED_WIDENING
    if source in FLOAT_WIDTHS and target in FLOAT_WIDTHS:
        return codegen.emit_spec_constant_op("FConvert", target_type, [value])
    if source in INTEGER_WIDTHS and target == "bool":
        zero = codegen.register_constant(0, source_type)
        return codegen.emit_spec_constant_op("INotEqual", target_type, [value, zero])
    if source == "bool" and target in INTEGER_WIDTHS:
        one = codegen.register_constant(1, target_type)
        zero = codegen.register_constant(0, target_type)
        return codegen.emit_spec_constant_op("Select", target_type, [value, one, zero])
    return KERNEL_ONLY


def converted_or_refused(codegen, value, target_type):
    """Convert ``value`` or record a refusal; returns None when the conversion was refused."""
    converted = convert_specialization_scalar(codegen, value, target_type)
    if not isinstance(converted, str):
        return converted
    source = scalar_type_name(codegen, codegen.registered_value_type(value)) or "?"
    target = scalar_type_name(codegen, target_type) or "?"
    codegen.refuse_lowering(
        f"SPIR-V module-scope constant converts a specialization constant from {source} "
        f"to {target}, which OpSpecConstantOp cannot express: {converted}"
    )
    return None


def specialization_constituent(codegen, value, member_type):
    """Return ``value`` converted to ``member_type`` for a constant composite constituent.

    Only scalar specialization-constant values whose type differs from the member type are touched; anything
    else is returned unchanged. A conversion that cannot be expressed records a refusal and keeps the value so
    generation reaches the module's other refusals.
    """
    if value is None or member_type is None:
        return value
    if not codegen.is_specialization_constant_value(value):
        return value
    source_type = codegen.registered_value_type(value)
    member_type = codegen.ensure_registered_type(member_type)
    if source_type is None or source_type.id == member_type.id:
        return value
    if scalar_type_name(codegen, source_type) is None:
        return value
    if scalar_type_name(codegen, member_type) is None:
        return value
    converted = converted_or_refused(codegen, value, member_type)
    return value if converted is None else converted


def conversion_target_name(expr) -> str | None:
    """Return the scalar type spelled by ``T(x)`` or ``(T)x``, or None for anything else."""
    if isinstance(expr, CastNode):
        target = getattr(expr, "target_type", None)
        if isinstance(target, PrimitiveType):
            return target.name
        return target if isinstance(target, str) else None
    if isinstance(expr, FunctionCallNode):
        if len(getattr(expr, "args", None) or []) != 1:
            return None
        if getattr(expr, "generic_args", None):
            return None
        callee = getattr(expr, "function", getattr(expr, "name", None))
        if isinstance(callee, IdentifierNode):
            callee = callee.name
        return callee if isinstance(callee, str) else None
    return None


def specialization_conversion_expression(codegen, expr):
    """Lower a scalar conversion of a specialization constant in a module-scope constant.

    Returns None when ``expr`` is not a scalar conversion or its operand does not follow a specialization
    constant, so ordinary constant folding keeps handling it.
    """
    target_name = conversion_target_name(expr)
    if not isinstance(target_name, str):
        return None
    if codegen.normalize_primitive_name(target_name) not in SCALAR_CONVERSION_TARGETS:
        return None
    operand_expr = expr.expression if isinstance(expr, CastNode) else expr.args[0]
    operand = codegen.process_constant_expression(operand_expr)
    if operand is None or not codegen.is_specialization_constant_value(operand):
        return None
    if scalar_type_name(codegen, codegen.registered_value_type(operand)) is None:
        return None
    target_type = codegen.register_primitive_type(target_name)
    converted = converted_or_refused(codegen, operand, target_type)
    if converted is None:
        return codegen.default_value_for_type(target_type)
    return converted
