"""``frexp``, ``modf`` and ``ldexp`` (GLSL 4.60 section 8.3) for GLSL and Metal when operand types differ.

CrossGL takes these functions from GLSL, but programs imported from HLSL spell them with HLSL's types: the
``frexp`` exponent has the type of the value (a float), ``modf`` may write its whole part to an integer, and the
``ldexp`` exponent may be a float or a scalar next to a vector value. GLSL accepts only ``frexp(genFType, out
genIType)``, ``modf(genFType, out genFType)`` and ``ldexp(genFType, genIType)`` with matching component counts, and
Metal has the same shapes (MSL 3.2 section 6.5). Metal binds the output only to a ``thread`` reference besides.

- ``frexp`` and ``modf`` whose output differs from the GLSL type call a helper that receives the GLSL-typed part in a
  local and converts it into the output, which is what an HLSL ``out`` argument of another type does. On Metal they
  always call a helper, overloaded for the ``thread``, ``device`` and ``threadgroup`` address spaces, because the
  output may live in any of them.
- ``ldexp`` with an integer exponent of another signedness or width converts or splats it. A floating-point exponent
  means ``x * 2^e`` in HLSL, which DXC lowers to ``x * exp2(e)``; that is the lowering here too, so a fractional
  exponent keeps its meaning.

A component-count mismatch that neither language defines, a non-floating-point value, and a double value with a
floating-point exponent (``exp2`` has no double overload) are refused.
"""

from __future__ import annotations

import re

__all__ = [
    "EXPONENT_FUNCTIONS",
    "emit_glsl_exponent_helpers",
    "emit_metal_exponent_helpers",
    "glsl_exponent_call",
    "metal_exponent_call",
]

EXPONENT_FUNCTIONS = frozenset({"frexp", "modf", "ldexp"})

_GLSL_PREFIX = {"float": "", "double": "d", "int": "i", "uint": "u", "bool": "b"}
_GLSL_VECTOR = re.compile(r"([diub]?)vec([234])")
_SIZED = re.compile(r"(float|double|half|int|uint|bool)([234]?)")
_PREFIX_COMPONENT = {"": "float", "d": "double", "i": "int", "u": "uint", "b": "bool"}
_FLOATS = {"float", "double", "half"}
_INTEGERS = {"int", "uint"}
_METAL_ADDRESS_SPACES = ("thread", "device", "threadgroup")


def _shape(type_name):
    """``(component, count)`` of a scalar or vector type in CrossGL, GLSL or Metal spelling, else ``None``."""
    text = str(type_name or "").strip()
    match = _GLSL_VECTOR.fullmatch(text)
    if match:
        return _PREFIX_COMPONENT[match.group(1)], int(match.group(2))
    match = _SIZED.fullmatch(text)
    if match:
        return match.group(1), int(match.group(2) or 1)
    return None


def _glsl_type(component, count):
    if count == 1:
        return component
    return f"{_GLSL_PREFIX[component]}vec{count}"


def _metal_type(component, count):
    return component if count == 1 else f"{component}{count}"


def _operand_shapes(codegen, args):
    return [_shape(codegen.expression_result_type(arg)) for arg in args]


def _refuse(codegen, name, detail):
    return codegen.refuse_lowering(f"{name}: {detail}", "0")


def _helper(codegen, attribute, name, text):
    helpers = codegen.__dict__.setdefault(attribute, {})
    helpers.setdefault(name, text)
    return name


def _ldexp(codegen, args, value, exponent, spell):
    """The lowered ``ldexp`` call, or ``None`` when the native call already has the language's types."""
    component, count = value
    exponent_component, exponent_count = exponent
    if exponent_count not in {1, count}:
        return _refuse(
            codegen,
            "ldexp",
            f"a {exponent_count}-component exponent for a {count}-component value",
        )
    if exponent_component in _INTEGERS:
        if exponent == ("int", count):
            return None
        x, e = (codegen.generate_expression(arg) for arg in args)
        return f"ldexp({x}, {spell('int', count)}({e}))"
    if exponent_component not in _FLOATS:
        return _refuse(
            codegen, "ldexp", "the exponent is neither an integer nor a float"
        )
    if component == "double":
        return _refuse(
            codegen,
            "ldexp",
            "a double value with a floating-point exponent needs exp2 on doubles",
        )
    x, e = (codegen.generate_expression(arg) for arg in args)
    return f"({x} * exp2({spell(component, count)}({e})))"


def _output_part(name, value):
    """The type of the part ``frexp`` or ``modf`` writes for ``value``."""
    component, count = value
    return ("int", count) if name == "frexp" else (component, count)


def _checked_shapes(codegen, name, args):
    """``(value, output)`` shapes, or a refusal string."""
    if len(args or ()) != 2:
        return _refuse(codegen, name, "it takes a value and a second operand")
    value, second = _operand_shapes(codegen, args)
    if value is None or second is None:
        return None
    if value[0] not in _FLOATS:
        return _refuse(
            codegen, name, "the value is not a floating-point scalar or vector"
        )
    if name != "ldexp" and (
        second[1] != value[1] or second[0] not in _FLOATS | _INTEGERS
    ):
        return _refuse(
            codegen,
            name,
            f"the output operand has no conversion from {_glsl_type(*_output_part(name, value))}",
        )
    return value, second


def _helper_body(name, value_type, part_type, output_type):
    if name == "frexp":
        return (
            f"    {part_type} exponent;\n"
            f"    {value_type} significand = frexp(x, exponent);\n"
            f"    part = {output_type}(exponent);\n"
            "    return significand;\n"
        )
    return (
        f"    {part_type} whole;\n"
        f"    {value_type} fraction = modf(x, whole);\n"
        f"    part = {output_type}(whole);\n"
        "    return fraction;\n"
    )


def glsl_exponent_call(codegen, name, args):
    """The GLSL spelling of a ``frexp``, ``modf`` or ``ldexp`` call, or ``None`` to emit the call natively."""
    if name not in EXPONENT_FUNCTIONS or name in codegen.function_return_types:
        return None
    shapes = _checked_shapes(codegen, name, args)
    if not isinstance(shapes, tuple):
        return shapes
    value, second = shapes
    if value[0] == "half":
        return None
    if name == "ldexp":
        return _ldexp(codegen, args, value, second, _glsl_type)
    part = _output_part(name, value)
    if second == part:
        return None
    value_type, part_type, output_type = (
        _glsl_type(*value),
        _glsl_type(*part),
        _glsl_type(*second),
    )
    helper = f"crosstl_{name}_{value_type}_{output_type}"
    _helper(
        codegen,
        "glsl_exponent_helpers",
        helper,
        f"{value_type} {helper}({value_type} x, out {output_type} part) {{\n"
        + _helper_body(name, value_type, part_type, output_type)
        + "}\n",
    )
    x, part = (codegen.generate_expression(arg) for arg in args)
    return f"{helper}({x}, {part})"


def metal_exponent_call(codegen, name, args):
    """The Metal spelling of a ``frexp``, ``modf`` or ``ldexp`` call, or ``None`` to emit the call natively."""
    if name not in EXPONENT_FUNCTIONS or name in codegen.user_function_names:
        return None
    shapes = _checked_shapes(codegen, name, args)
    if not isinstance(shapes, tuple):
        return shapes
    value, second = shapes
    if value[0] == "double":
        return None
    if name == "ldexp":
        return _ldexp(codegen, args, value, second, _metal_type)
    value_type, part_type = _metal_type(*value), _metal_type(*_output_part(name, value))
    helper = f"crosstl_{name}_{value_type}"
    body = _helper_body(name, value_type, part_type, "Output")
    _helper(
        codegen,
        "metal_exponent_helpers",
        helper,
        "".join(
            "template <typename Output>\n"
            f"static inline {value_type} {helper}({value_type} x, {space} Output& part) {{\n"
            + body
            + "}\n"
            for space in _METAL_ADDRESS_SPACES
        ),
    )
    x, part = (codegen.generate_expression(arg) for arg in args)
    return f"{helper}({x}, {part})"


def emit_glsl_exponent_helpers(codegen):
    """The GLSL helpers the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("glsl_exponent_helpers", {}).values())


def emit_metal_exponent_helpers(codegen):
    """The Metal helpers the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("metal_exponent_helpers", {}).values())
