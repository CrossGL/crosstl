"""Built-in calls of other languages that the Slang generator writes as functions of Slang's core module.

Slang's core module (``hlsl.meta.slang`` and ``core.meta.slang``, Slang v2026.20) declares, besides the HLSL
intrinsics, functions that C, MSL, CUDA or GLSL built-ins share their name and meaning with:

- C and MSL math on floating-point operands: ``copysign``, ``divide`` (``x / y``), ``fabs``, ``fdim``
  (``max(0, x - y)``), ``fmax`` and ``fmin`` (``NMax`` and ``NMin`` on SPIR-V, the operand that is not NaN),
  ``nextafter`` and ``rint`` (ties to even);
- MSL's three-operand ``fmax3``, ``fmin3``, ``fmedian3``, ``max3``, ``min3`` and ``median3``. Slang declares them for
  floating-point operands only; MSL's integer ``max3``, ``min3`` and ``median3`` are written with ``max`` and
  ``min``, which give the same value for integers;
- GLSL's ``bitfieldExtract`` and ``bitfieldInsert``, whose offset and bit count Slang reads as ``uint``, and the
  float packing functions ``pack{Unorm,Snorm}4x8``, ``pack{Unorm,Snorm}2x16`` and ``packHalf2x16``.

Slang spells GLSL's unpacking functions with the result type in the name (``unpackUnorm4x8ToFloat``) and GLSL's
``interpolateAtCentroid`` and ``interpolateAtSample`` as HLSL's ``EvaluateAttributeAtCentroid`` and
``EvaluateAttributeAtSample``, whose sample index is a ``uint``. :data:`SLANG_CORE_FUNCTION_RENAMES` maps those.
GLSL's ``interpolateAtOffset`` takes a ``vec2`` offset in pixels, which ``EvaluateAttributeSnapped``'s ``int2``
offset in sixteenths of a pixel does not equal, so it stays refused.
"""

from __future__ import annotations

import re

__all__ = [
    "SLANG_CORE_FUNCTION_NAMES",
    "SLANG_CORE_FUNCTION_RENAMES",
    "generate_slang_core_function_call",
]

SLANG_CORE_FUNCTION_NAMES = frozenset(
    {
        # The inverse hyperbolic functions, which HLSL lacks.
        "acosh",
        "asinh",
        "atanh",
        # C and MSL floating-point math.
        "copysign",
        "divide",
        "fabs",
        "fdim",
        "fmax",
        "fmin",
        "nextafter",
        "rint",
        # MSL's three-operand minimum, maximum and median.
        "fmax3",
        "fmedian3",
        "fmin3",
        "max3",
        "median3",
        "min3",
        # GLSL's bit-field and float packing functions.
        "bitfieldExtract",
        "bitfieldInsert",
        "packHalf2x16",
        "packSnorm2x16",
        "packSnorm4x8",
        "packUnorm2x16",
        "packUnorm4x8",
    }
)

SLANG_CORE_FUNCTION_RENAMES = {
    "unpackHalf2x16": "unpackHalf2x16ToFloat",
    "unpackSnorm2x16": "unpackSnorm2x16ToFloat",
    "unpackSnorm4x8": "unpackSnorm4x8ToFloat",
    "unpackUnorm2x16": "unpackUnorm2x16ToFloat",
    "unpackUnorm4x8": "unpackUnorm4x8ToFloat",
    "interpolateAtCentroid": "EvaluateAttributeAtCentroid",
    "interpolateAtSample": "EvaluateAttributeAtSample",
}

# The operands Slang reads as ``uint``, by function: the bit-field offset and count, and the sample index.
_UINT_OPERANDS = {
    "bitfieldExtract": (1, 2),
    "bitfieldInsert": (2, 3),
    "interpolateAtSample": (1,),
}
_ARITIES = {
    "bitfieldExtract": 3,
    "bitfieldInsert": 4,
    "interpolateAtSample": 2,
    "max3": 3,
    "min3": 3,
    "median3": 3,
}
_INTEGER_SCALARS = (
    "int",
    "uint",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int64_t",
    "uint64_t",
)
_INTEGER_TYPE = re.compile(
    rf"(?:{'|'.join(_INTEGER_SCALARS)})[234]?"
    rf"|vector<\s*(?:{'|'.join(_INTEGER_SCALARS)})\s*,\s*[234]\s*>"
)


def _slang_type(codegen, argument):
    """The Slang type of ``argument``, or ``None`` when it is unknown."""
    argument_type = codegen.expression_result_type(argument)
    if argument_type is None:
        return None
    mapped = codegen.convert_type(argument_type)
    return mapped.strip() if isinstance(mapped, str) else None


def _integer_slang_type(codegen, argument):
    """The Slang integer scalar or vector type of ``argument``, or ``None`` when it is not one."""
    mapped = _slang_type(codegen, argument)
    if mapped is not None and _INTEGER_TYPE.fullmatch(mapped):
        return mapped
    return None


def _integer_median3_helper(codegen, slang_type):
    name = codegen.helper_function_name(
        "cgl_median3_" + re.sub(r"\W+", "_", slang_type).strip("_")
    )
    codegen.register_helper_function(
        name,
        f"{slang_type} {name}({slang_type} x, {slang_type} y, {slang_type} z)\n"
        "{\n"
        "    return max(min(x, y), min(max(x, y), z));\n"
        "}",
    )
    return name


def generate_slang_core_function_call(codegen, callee, arguments):
    """Slang source of a call to a core-module function that needs more than its name, or ``None``.

    ``callee`` is a built-in the program does not declare. Refuses a call with the wrong number of operands and a
    three-operand minimum, maximum or median whose operand type is unknown.
    """
    if callee not in _ARITIES:
        return None
    if len(arguments) != _ARITIES[callee]:
        return codegen.refuse_lowering(
            f"Slang {callee}: requires {_ARITIES[callee]} arguments, got {len(arguments)}",
            f"{callee}()",
        )
    rendered = [codegen.generate_expression_without_expected(arg) for arg in arguments]
    if callee in _UINT_OPERANDS:
        for index in _UINT_OPERANDS[callee]:
            if _slang_type(codegen, arguments[index]) != "uint":
                rendered[index] = f"uint({rendered[index]})"
        name = SLANG_CORE_FUNCTION_RENAMES.get(callee, callee)
        return f"{name}({', '.join(rendered)})"
    integer_type = _integer_slang_type(codegen, arguments[0])
    if integer_type is None:
        # Slang declares the floating-point overloads; the operands' own type selects one.
        return f"{callee}({', '.join(rendered)})"
    x, y, z = rendered
    if callee == "max3":
        return f"max({x}, max({y}, {z}))"
    if callee == "min3":
        return f"min({x}, min({y}, {z}))"
    return f"{_integer_median3_helper(codegen, integer_type)}({x}, {y}, {z})"
