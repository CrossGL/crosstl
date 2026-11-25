"""HLSL ``f32tof16`` and ``f16tof32`` in GLSL.

HLSL (Shader Model 5) converts each component between ``float`` and the IEEE binary16 bits held in the low 16 bits
of a ``uint``: ``f32tof16(float x)`` returns those bits with the high 16 bits zero, and ``f16tof32(uint x)`` reads
the low 16 bits and ignores the others. Both take and return vectors componentwise. GLSL has no such functions, and
GLSL output refused the calls as undeclared functions. GLSL 4.20 and GLSL ES 3.00 (section 8.4) convert two halves
at once: ``packHalf2x16(vec2(x, 0.0))`` puts ``x`` in the low 16 bits and +0.0, whose bits are 0, in the high 16
bits, and ``unpackHalf2x16(x).x`` reads the low 16 bits. DXC's SPIR-V backend spells the calls the same way, with
``PackHalf2x16`` and ``UnpackHalf2x16``.

HLSL converts a ``float`` operand of another type, an ``int`` or a ``double``, to ``float`` first, and the ``uint``
operand of ``f16tof32`` likewise; GLSL converts neither, so the operands are converted explicitly. A vector converts
through one helper per width and direction, so its operand is evaluated once. :func:`glsl_half_conversion_call`
writes the call and records the helper, and :func:`emit_glsl_half_conversion_helpers` writes the helpers the program
used.
"""

from __future__ import annotations

__all__ = [
    "HALF_CONVERSION_FUNCTIONS",
    "emit_glsl_half_conversion_helpers",
    "glsl_half_conversion_call",
]

HALF_CONVERSION_FUNCTIONS = frozenset({"f32tof16", "f16tof32"})

_COMPONENTS = "xyzw"


def glsl_half_conversion_call(codegen, func_name, arguments):
    """The GLSL for a call to HLSL ``f32tof16``/``f16tof32``, or None when ``func_name`` is neither of them.

    A call whose operand is not a numeric scalar or vector of 2 to 4 components is refused, as HLSL refuses it.
    """
    if func_name not in HALF_CONVERSION_FUNCTIONS:
        return None
    if len(arguments) != 1:
        codegen.refuse_lowering(
            f"call to '{func_name}' with {len(arguments)} arguments: HLSL {func_name} takes one"
        )
    argument = arguments[0]
    info = codegen.glsl_value_type_info(codegen.glsl_source_expression_type(argument))
    width = info["width"] if info is not None else None
    if info is None or info["family"] == "bool" or width not in (1, 2, 3, 4):
        codegen.refuse_lowering(
            f"call to '{func_name}' whose operand is not a numeric scalar or vector: HLSL {func_name} converts "
            "the components of a float or uint scalar or vector"
        )
    to_half = func_name == "f32tof16"
    operand_family = "float" if to_half else "uint"
    operand_type = operand_family if width == 1 else _vector(operand_family, width)
    if info["family"] == operand_family and info.get("bits", 32) == 32:
        operand = codegen.generate_expression(argument)
    else:
        operand = codegen.glsl_converted_operand(argument, operand_type)
    if width == 1:
        return _scalar_conversion(to_half, operand)
    name = f"crossgl_{func_name}_{operand_type}"
    helpers = codegen.__dict__.setdefault("glsl_half_conversion_helpers", {})
    if name not in helpers:
        helpers[name] = _helper_definition(name, to_half, width)
    return f"{name}({operand})"


def emit_glsl_half_conversion_helpers(codegen) -> str:
    """The ``crossgl_f32tof16_vecN`` and ``crossgl_f16tof32_uvecN`` helpers the program calls."""
    return "".join(codegen.__dict__.get("glsl_half_conversion_helpers", {}).values())


def _vector(family, width):
    return {"float": "vec", "uint": "uvec"}[family] + str(width)


def _scalar_conversion(to_half, operand):
    if to_half:
        return f"packHalf2x16(vec2({operand}, 0.0))"
    return f"unpackHalf2x16({operand}).x"


def _helper_definition(name, to_half, width):
    components = ", ".join(
        _scalar_conversion(to_half, f"v.{c}") for c in _COMPONENTS[:width]
    )
    result, parameter = (
        (_vector("uint", width), _vector("float", width))
        if to_half
        else (_vector("float", width), _vector("uint", width))
    )
    return (
        f"{result} {name}({parameter} v) {{\n"
        f"    return {result}({components});\n"
        "}\n\n"
    )
