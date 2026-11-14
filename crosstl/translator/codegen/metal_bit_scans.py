"""CrossGL ``findMSB`` and ``findLSB`` (GLSL 4.60 section 8.8) on Metal.

MSL has no bit-index scans, only the bit counts ``clz`` and ``ctz``, which return the component width for a zero
component (MSL 3.2 section 6.13). The scans return an ``int`` per component:

* ``findLSB(x)`` is the index of the lowest set bit, or -1 when ``x`` is zero: ``select(R(ctz(x)), R(-1), x == 0)``.
* ``findMSB(x)`` of an unsigned value is the index of the highest set bit, or -1 for zero: ``31 - clz(x)``. Of a
  signed value it is the highest bit that differs from the sign bit, so a negative value scans its complement:
  ``31 - clz(x < 0 ? ~x : x)``, which is -1 for both 0 and -1.

Each call goes through a helper per operand type, so the operand is evaluated once. Only 32-bit ``int`` and
``uint`` components are scanned, as in GLSL; any other operand is refused rather than emitted as an undeclared call.
"""

from __future__ import annotations

import re

__all__ = ["BIT_SCAN_FUNCTIONS", "emit_metal_bit_scan_helpers", "metal_bit_scan_call"]

BIT_SCAN_FUNCTIONS = frozenset({"findMSB", "findLSB"})

_METAL_INTEGER = re.compile(r"(int|uint)([234]?)")
_GLSL_INTEGER = re.compile(r"([iu])vec([234])")


def _metal_operand_type(type_name):
    """The Metal spelling of a 32-bit ``int``/``uint`` scalar or vector type, else ``None``."""
    text = str(type_name or "").strip()
    if _METAL_INTEGER.fullmatch(text):
        return text
    match = _GLSL_INTEGER.fullmatch(text)
    if match:
        return ("int" if match.group(1) == "i" else "uint") + match.group(2)
    return None


def _helper_body(name, operand, result):
    if name == "findLSB":
        return (
            f"    return select({result}(ctz(x)), {result}(-1), x == {operand}(0));\n"
        )
    if operand.startswith("uint"):
        return f"    return {result}(31) - {result}(clz(x));\n"
    return (
        f"    return {result}(31) - {result}(clz(select(x, ~x, x < {operand}(0))));\n"
    )


def metal_bit_scan_call(codegen, name, args):
    """The Metal spelling of a ``findMSB``/``findLSB`` call, or ``None`` for any other call."""
    if name not in BIT_SCAN_FUNCTIONS or name in codegen.user_function_names:
        return None
    if len(args or ()) != 1:
        return codegen.refuse_lowering(f"{name}: it takes one integer operand", "0")
    argument_type = codegen.expression_result_type(args[0])
    operand = _metal_operand_type(
        codegen.map_type(argument_type)
    ) or _metal_operand_type(argument_type)
    if operand is None:
        return codegen.refuse_lowering(
            f"{name}: the operand ({argument_type}) is not a 32-bit int or uint scalar or vector",
            "0",
        )
    result = "int" + operand.removeprefix("uint").removeprefix("int")
    helper = f"crosstl_{name}_{operand}"
    helpers = codegen.__dict__.setdefault("metal_bit_scan_helpers", {})
    helpers.setdefault(
        helper,
        f"static inline {result} {helper}({operand} x) {{\n"
        + _helper_body(name, operand, result)
        + "}\n",
    )
    return f"{helper}({codegen.generate_expression(args[0])})"


def emit_metal_bit_scan_helpers(codegen):
    """The Metal bit-scan helpers the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("metal_bit_scan_helpers", {}).values())
