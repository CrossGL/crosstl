"""WGSL lowering of the two-argument ``frexp`` and ``modf``.

CrossGL spells ``frexp(x, e)`` and ``modf(x, i)`` as GLSL does (GLSL 4.60 section 8.3): the call returns the
significand or the fractional part and writes the exponent or the whole part to its second, ``out`` argument. WGSL's
``frexp(x)`` and ``modf(x)`` take one argument and return a structure, with members ``fract`` and ``exp`` (``i32``, or
``vecN<i32>`` for a vector) or ``fract`` and ``whole`` (WGSL section 17.5, "Numeric Built-in Functions"). A call
therefore goes through a helper per built-in, value type and written type, which calls the one-argument built-in,
stores the second member through a pointer and returns ``fract``, so the call stays one expression wherever it
stands::

    fn crosstl_frexp_vec3_f32_vec3_i32(value: vec3<f32>, out: ptr<function, vec3<i32>>) -> vec3<f32> {
        let result = frexp(value);
        *out = result.exp;
        return result.fract;
    }

A written variable of another type, such as the ``float`` exponent that HLSL's ``frexp`` writes, receives the member
converted to its type. The written argument is passed as an ``out`` parameter is (``wgsl_parameter_lowering``): ``&v``
for a whole function-scope variable and the pointer itself for a pointer parameter. Core WGSL accepts no other
``ptr<function, T>`` argument (WGSL section 11.3.1) and no pointer to a vector component (WGSL section 8.14), so a
written argument such as ``e.x``, ``parts.e``, an array element, a storage buffer element or a module-scope variable
is refused. WGSL output computes ``half`` values as ``f32``, so the value is always ``f32`` or a vector of it.
"""

from __future__ import annotations

import re

from ..ast import IdentifierNode, ParameterNode

__all__ = [
    "is_two_argument_frexp_or_modf",
    "wgsl_frexp_modf_call",
    "wgsl_frexp_modf_helpers",
]

# The member of each built-in's result structure that the second argument receives.
WRITTEN_MEMBERS = {"frexp": "exp", "modf": "whole"}


def is_two_argument_frexp_or_modf(codegen, call):
    """Whether ``call`` is the built-in ``frexp`` or ``modf`` with a written second argument."""
    function = call.function
    return (
        isinstance(function, IdentifierNode)
        and function.name in WRITTEN_MEMBERS
        and len(call.arguments) == 2
        and not codegen.is_user_function_name(function.name)
    )


def wgsl_frexp_modf_call(codegen, call):
    """The helper call for a two-argument ``frexp`` or ``modf``, registering the helper on ``codegen``."""
    name = call.function.name
    value, written = call.arguments
    value_type = codegen.expression_type(value)
    written_type = codegen.expression_type(written)
    if value_type is None or written_type is None:
        raise ValueError(
            f"WGSL target cannot lower {name}() without known operand types"
        )
    value_type = codegen.type_name_string(value_type)
    written_type = codegen.type_name_string(written_type)
    value_shape = _shape(codegen, value_type)
    if value_shape is None or value_shape[0] != "f32":
        raise ValueError(
            f"WGSL target cannot lower {name}() on {value_type}; only float "
            "scalars and vectors are supported"
        )
    written_shape = _shape(codegen, written_type)
    if written_shape is None or written_shape[1] != value_shape[1]:
        raise ValueError(
            f"WGSL target cannot lower {name}() on {value_type} writing "
            f"{written_type}; the written argument must be a scalar or vector "
            "with the same number of components"
        )
    written_parameter = ParameterNode(
        WRITTEN_MEMBERS[name], written_type, qualifiers=["out"]
    )
    pointer = codegen.reference_argument_expression(
        written, written_parameter, call.function
    )
    helper = _helper_name(codegen, name, value_type, written_type)
    codegen._frexp_modf_helpers.setdefault(helper, (name, value_type, written_type))
    argument = codegen.generate_expression_for_target(
        value, codegen.expression_type(value)
    )
    return f"{helper}({argument}, {pointer})"


def wgsl_frexp_modf_helpers(codegen):
    """The helper functions that the generated calls use, in the order of their first call."""
    return "\n\n".join(
        _helper(helper, name, value_type, written_type)
        for helper, (name, value_type, written_type) in (
            codegen._frexp_modf_helpers.items()
        )
    )


def _shape(codegen, type_name):
    """``(element, components)`` of a WGSL scalar or vector type, or ``None``."""
    vector = codegen.vector_shape(type_name)
    if vector is not None:
        return vector
    scalar = codegen.scalar_type_name(type_name)
    return None if scalar is None else (scalar, 1)


def _helper_name(codegen, name, value_type, written_type):
    def flat(type_name):
        return re.sub(r"[<>, ]+", "_", type_name).strip("_")

    return codegen.safe_wgsl_identifier(
        f"crosstl_{name}_{flat(value_type)}_{flat(written_type)}"
    )


def _helper(helper, name, value_type, written_type):
    member = WRITTEN_MEMBERS[name]
    if name == "modf":
        member_type = value_type
    elif value_type.startswith("vec"):
        member_type = re.sub(r"<f32>$", "<i32>", value_type)
    else:
        member_type = "i32"
    stored = f"result.{member}"
    if member_type != written_type:
        stored = f"{written_type}({stored})"
    return (
        f"fn {helper}(value: {value_type}, out: ptr<function, {written_type}>) "
        f"-> {value_type} {{\n"
        f"    let result = {name}(value);\n"
        f"    *out = {stored};\n"
        "    return result.fract;\n"
        "}"
    )
