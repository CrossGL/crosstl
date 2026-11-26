"""Slang spellings of value calls whose CrossGL form Slang does not declare.

- An empty constructor of a scalar, vector or matrix (``vec4{}``, MSL's ``float4 v = {};``, C++'s ``float()``) is the
  zero value. Slang declares no constructor of those types without arguments (``float4()`` is "no overload for
  'float4' applicable to arguments of type ()", slangc 2026.20), so the zero value is written as the constructor
  from one scalar: ``float4(0)``, ``bool2(false)``.
- MSL's ``sincos(x, cosval)`` returns the sine and writes the cosine (MSL specification 4.0, section 6.5), while
  Slang's ``sincos(x, out s, out c)`` is HLSL's and returns nothing. The two-operand form calls a helper that makes
  the HLSL call; the three-operand form is Slang's own.
- GLSL's ``packDouble2x32(uvec2)`` and ``unpackDouble2x32(double)`` (GLSL 4.60 section 8.4), which the binary64 math
  helpers read and write words with, are HLSL's ``asdouble(lowbits, highbits)`` and ``asuint(value, out lowbits,
  out highbits)`` in Slang, written through helpers so that the operand is evaluated once.
"""

from __future__ import annotations

import re

__all__ = [
    "generate_slang_double_word_call",
    "generate_slang_sincos_call",
    "slang_empty_value_constructor",
]

# CrossGL name -> (helper base name, helper source with {name}).
_DOUBLE_WORD_HELPERS = {
    "packDouble2x32": (
        "cgl_pack_double2x32",
        "double {name}(uint2 words)\n{{\n    return asdouble(words.x, words.y);\n}}",
    ),
    "unpackDouble2x32": (
        "cgl_unpack_double2x32",
        "uint2 {name}(double value)\n{{\n    uint low;\n    uint high;\n"
        "    asuint(value, low, high);\n    return uint2(low, high);\n}}",
    ),
}

_SCALARS = (
    "bool",
    "half",
    "float",
    "double",
    "int",
    "uint",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int64_t",
    "uint64_t",
    "float16_t",
    "float32_t",
    "float64_t",
)
_SCALAR = "|".join(_SCALARS)
_VALUE_TYPE = re.compile(
    rf"(?P<named>{_SCALAR})(?:[234](?:x[234])?)?"
    rf"|(?:vector|matrix)<\s*(?P<generic>{_SCALAR})\s*(?:,\s*\d+\s*)+>"
)


def slang_empty_value_constructor(slang_type):
    """The zero value of the Slang scalar, vector or matrix ``slang_type``, or ``None`` for any other type."""
    if not isinstance(slang_type, str):
        return None
    match = _VALUE_TYPE.fullmatch(slang_type.strip())
    if match is None:
        return None
    scalar = match.group("named") or match.group("generic")
    return f"{slang_type.strip()}({'false' if scalar == 'bool' else '0'})"


def generate_slang_sincos_call(codegen, arguments):
    """Slang source of a two-operand ``sincos(x, cosval)``, or ``None`` for the other forms."""
    if len(arguments) != 2:
        return None
    value_type = codegen.expression_result_type(arguments[0])
    slang_type = codegen.convert_type(value_type) if value_type is not None else None
    if not isinstance(slang_type, str) or not slang_type.strip():
        return codegen.refuse_lowering(
            "Slang sincos: the operand type of sincos(x, cosval) is unknown",
            "sincos()",
        )
    slang_type = slang_type.strip()
    name = codegen.helper_function_name(
        "cgl_sincos_" + re.sub(r"\W+", "_", slang_type).strip("_")
    )
    codegen.register_helper_function(
        name,
        f"{slang_type} {name}({slang_type} x, out {slang_type} cosval)\n"
        "{\n"
        f"    {slang_type} sinval;\n"
        "    sincos(x, sinval, cosval);\n"
        "    return sinval;\n"
        "}",
    )
    rendered = [codegen.generate_expression_without_expected(arg) for arg in arguments]
    return f"{name}({', '.join(rendered)})"


def generate_slang_double_word_call(codegen, callee, arguments):
    """Slang source of a ``packDouble2x32`` or ``unpackDouble2x32`` call, or ``None`` for other callees."""
    if callee not in _DOUBLE_WORD_HELPERS:
        return None
    if len(arguments) != 1:
        return codegen.refuse_lowering(
            f"Slang {callee}: requires 1 argument, got {len(arguments)}",
            f"{callee}()",
        )
    base_name, source = _DOUBLE_WORD_HELPERS[callee]
    name = codegen.helper_function_name(base_name)
    codegen.register_helper_function(name, source.format(name=name))
    expected = "uint2" if callee == "packDouble2x32" else "double"
    return (
        f"{name}({codegen.generate_expression_with_expected(arguments[0], expected)})"
    )
