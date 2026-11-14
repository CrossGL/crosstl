"""WGSL lowering of the floating-point classification functions ``isnan``, ``isinf``, ``isfinite`` and ``signbit``.

GLSL (GLSL 4.60 section 8.3), HLSL and Metal (MSL 3.2 section 6.5, "Relational Functions") classify a float or a
float vector component-wise, and CrossGL keeps their names. WGSL has none of these built-in functions (WGSL section
17, "Built-in Functions"), so emitting the call verbatim yields a module that fails validation. WGSL also lets an
implementation assume that no infinity or NaN occurs during execution (WGSL section 15.7, "Floating Point
Evaluation"), so a comparison such as ``x != x`` may fold to ``false``. The generator therefore tests the bits of the
IEEE-754 binary32 value, which ``bitcast`` exposes unchanged (WGSL section 17.2, "Bit Reinterpretation Built-in
Functions")::

    isnan(x)    ->  ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u)
    isinf(x)    ->  ((bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u)
    isfinite(x) ->  ((bitcast<u32>(x) & 0x7fffffffu) < 0x7f800000u)
    signbit(x)  ->  ((bitcast<u32>(x) & 0x80000000u) != 0u)

A vector operand uses ``vecN<u32>`` throughout, because WGSL's bitwise operators and comparisons take two operands of
one type (WGSL sections 8.8 and 8.9), and the comparison yields ``vecN<bool>`` as the source function does. Each
lowering reads its operand once, so an operand with side effects keeps them. WGSL output computes ``half`` values as
``f32``, so every float operand is binary32 here. An operand of another type, or one whose type is unknown, is
refused rather than reinterpreted, since the bit test is only correct for binary32.
"""

from __future__ import annotations

from ..ast import IdentifierNode

__all__ = [
    "WGSLFloatClassificationError",
    "is_float_classification_call",
    "wgsl_float_classification_call",
]

# Mask applied to the operand's bits, comparison operator and compared constant for each function.
_BIT_TESTS = {
    "isnan": ("0x7fffffffu", ">", "0x7f800000u"),
    "isinf": ("0x7fffffffu", "==", "0x7f800000u"),
    "isfinite": ("0x7fffffffu", "<", "0x7f800000u"),
    "signbit": ("0x80000000u", "!=", "0u"),
}


class WGSLFloatClassificationError(ValueError):
    """Raised when a float classification call has no exact WGSL bit test."""

    project_diagnostic_code = "project.translate.wgsl-float-classification-unsupported"
    missing_capabilities = ("wgsl.float-classification-lowering",)

    def __init__(self, message, *, operation, operand_type=None):
        super().__init__(message)
        self.operation = operation
        self.operand_type = operand_type


def is_float_classification_call(codegen, call):
    """Whether ``call`` is the built-in ``isnan``, ``isinf``, ``isfinite`` or ``signbit``."""
    function = call.function
    return (
        isinstance(function, IdentifierNode)
        and function.name in _BIT_TESTS
        and not codegen.is_user_function_name(function.name)
    )


def wgsl_float_classification_call(codegen, call):
    """The WGSL bit test that computes the classification ``call``."""
    name = call.function.name
    if len(call.arguments) != 1:
        raise WGSLFloatClassificationError(
            f"WGSL target supports {name}() with exactly 1 argument; got "
            f"{len(call.arguments)}",
            operation=name,
        )
    operand = call.arguments[0]
    operand_type = codegen.expression_type(operand)
    bits_type = _bits_type(codegen, operand_type)
    if bits_type is None:
        described = _describe_type(codegen, operand_type)
        raise WGSLFloatClassificationError(
            f"WGSL target cannot lower {name}() on {described}: WGSL has no {name} "
            "built-in, and the bit test it is lowered to needs an f32 or vecN<f32> "
            "operand",
            operation=name,
            operand_type=described,
        )
    mask, operator, constant = _BIT_TESTS[name]
    if bits_type != "u32":
        mask = f"{bits_type}({mask})"
        constant = f"{bits_type}({constant})"
    value = codegen.generate_expression(operand)
    return f"((bitcast<{bits_type}>({value}) & {mask}) {operator} {constant})"


def _bits_type(codegen, operand_type):
    """``u32`` or ``vecN<u32>`` for an ``f32`` or ``vecN<f32>`` operand type, else ``None``."""
    if operand_type is None:
        return None
    if codegen.scalar_type_name(operand_type) == "f32":
        return "u32"
    shape = codegen.vector_shape(operand_type)
    if shape is not None and shape[0] == "f32":
        return f"vec{shape[1]}<u32>"
    return None


def _describe_type(codegen, operand_type):
    if operand_type is None:
        return "an operand of unknown type"
    try:
        return f"a {codegen.type_name_string(operand_type)} operand"
    except ValueError:
        return f"a {operand_type} operand"
