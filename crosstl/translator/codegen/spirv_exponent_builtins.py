"""SPIR-V lowering of ``frexp``, ``modf`` and ``ldexp`` (GLSL 4.60 sections 8.2 and 8.3).

``frexp(x, out e)`` and ``modf(x, out i)`` write their second operand. The lowering uses the struct-returning
GLSL.std.450 forms, ``FrexpStruct`` and ``ModfStruct`` (GLSL.std.450 sections 36 and 52), as glslang and DXC do:
the call evaluates ``x``, extracts the returned members, and stores the second member through an ordinary
assignment to the output operand. The assignment converts the value when the operand's type differs, which keeps
HLSL's spelling meaningful: HLSL declares the ``frexp`` exponent with the type of ``x`` (a float), and DXC stores
the integer exponent converted to float.

``ldexp(x, e)`` maps to GLSL.std.450 ``Ldexp`` when ``e`` is an integer. HLSL also accepts a floating-point ``e``
and defines the result as ``x * 2^e``; that form lowers to ``x * Exp2(e)``, which is what DXC emits. A scalar ``e``
next to a vector ``x`` is splatted, since ``Ldexp`` needs matching component counts.
"""

from __future__ import annotations

from ..ast import AssignmentNode, IdentifierNode

__all__ = [
    "OUTPUT_OPERAND_FUNCTIONS",
    "spirv_ldexp_call",
    "spirv_output_operand_call",
]

# CrossGL name: GLSL.std.450 struct-returning instruction.
OUTPUT_OPERAND_FUNCTIONS = {"frexp": "FrexpStruct", "modf": "ModfStruct"}

FLOAT_COMPONENTS = {"float", "double", "half", "float16"}
INTEGER_COMPONENTS = {"int", "uint"}

_TEMPORARY_PREFIX = "_crosstl_output_operand_"


def _component_and_count(codegen, value_type):
    vector = codegen.vector_component_type_and_count(value_type.type.base_type)
    if vector is not None:
        return codegen.normalize_primitive_name(vector[0]), vector[1]
    return codegen.normalize_primitive_name(value_type.type.base_type), 1


def _value_type(codegen, value):
    return codegen.registered_value_type(value) or codegen.ensure_registered_type(
        value.type
    )


def _int_type(codegen, count):
    scalar = codegen.register_primitive_type("int")
    return scalar if count == 1 else codegen.register_vector_type(scalar, count)


def _result_struct(codegen, function_name, value_type, second_type):
    name = f"_{function_name}_result_{value_type.type.base_type}"
    if name in codegen.struct_types:
        return codegen.struct_types[name]
    return codegen.register_struct_type(
        name, [(value_type, "value"), (second_type, "second")]
    )


def _store_through_assignment(codegen, target, value, value_type):
    """Store ``value`` to the CrossGL l-value ``target`` with the conversions an assignment applies."""
    codegen.output_operand_temporaries = (
        getattr(codegen, "output_operand_temporaries", 0) + 1
    )
    name = f"{_TEMPORARY_PREFIX}{codegen.output_operand_temporaries}"
    temporary = codegen.create_variable(value_type, "Function")
    codegen.store_to_variable(temporary, value)
    codegen.local_variables[name] = temporary
    try:
        codegen.process_assignment(AssignmentNode(target, IdentifierNode(name)))
    finally:
        del codegen.local_variables[name]


def spirv_output_operand_call(codegen, function_name, arguments):
    """Lower ``frexp`` or ``modf``; the arguments are CrossGL expressions, not yet evaluated."""
    if len(arguments) != 2:
        raise codegen.builtin_lowering_error(
            function_name,
            "it takes a floating-point value and an output operand",
            reason="arity",
        )
    value = codegen.process_expression(arguments[0])
    if value is None:
        return None
    value_type = _value_type(codegen, value)
    component, count = _component_and_count(codegen, value_type)
    if component not in FLOAT_COMPONENTS:
        raise codegen.builtin_lowering_error(
            function_name,
            "its first operand must be a floating-point scalar or vector",
        )
    if function_name == "frexp":
        second_type = _int_type(codegen, count)
    else:
        second_type = value_type
    struct_type = _result_struct(codegen, function_name, value_type, second_type)
    result = codegen.emit_glsl_std450_instruction(
        OUTPUT_OPERAND_FUNCTIONS[function_name], struct_type, [value]
    )
    first = codegen.composite_extract(result, value_type, 0)
    second = codegen.composite_extract(result, second_type, 1)
    _store_through_assignment(codegen, arguments[1], second, second_type)
    return first


def spirv_ldexp_call(codegen, args):
    """Lower ``ldexp`` on evaluated operands, or raise the typed refusal."""
    if len(args) != 2:
        raise codegen.builtin_lowering_error(
            "ldexp", "it takes a value and an exponent", reason="arity"
        )
    value, exponent = args
    value_type = _value_type(codegen, value)
    component, count = _component_and_count(codegen, value_type)
    if component not in FLOAT_COMPONENTS:
        raise codegen.builtin_lowering_error(
            "ldexp", "its first operand must be a floating-point scalar or vector"
        )
    exponent_type = _value_type(codegen, exponent)
    exponent_component, exponent_count = _component_and_count(codegen, exponent_type)
    if exponent_count not in {1, count}:
        raise codegen.builtin_lowering_error(
            "ldexp",
            "the exponent must be a scalar or match the value's component count",
        )
    if exponent_component in INTEGER_COMPONENTS:
        if exponent_count != count:
            target = codegen.register_vector_type(
                codegen.register_primitive_type(exponent_component), count
            )
            exponent = codegen.splat_scalar_to_vector(exponent, target)
        return codegen.emit_glsl_std450_instruction(
            "Ldexp", value_type, [value, exponent]
        )
    if exponent_component not in FLOAT_COMPONENTS:
        raise codegen.builtin_lowering_error(
            "ldexp",
            "the exponent must be an integer or floating-point scalar or vector",
        )
    if exponent_count != count:
        scalar_type = codegen.register_primitive_type(component)
        exponent = codegen.convert_value_to_type(exponent, scalar_type)
        exponent = codegen.splat_scalar_to_vector(exponent, value_type)
    else:
        exponent = codegen.convert_value_to_type(exponent, value_type)
    scale = codegen.emit_glsl_std450_instruction("Exp2", value_type, [exponent])
    return codegen.binary_operation("*", value_type, value, scale)
