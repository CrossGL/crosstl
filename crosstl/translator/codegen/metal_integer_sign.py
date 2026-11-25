"""CrossGL ``sign`` of an integer operand on Metal.

CrossGL's ``sign`` takes integer operands as well as floating-point ones, as GLSL's ``sign(genIType)`` (GLSL 4.60
section 8.3) and HLSL's ``sign`` do, and keeps the operand's type. MSL declares its common function ``sign`` for
``half`` and ``float`` types only, so ``sign(i)`` on an ``int`` is ambiguous and ``sign(v)`` on an ``int2`` has no
match. The call goes through a helper per operand type, ``T(x > T(0)) - T(x < T(0))``, or ``T(x != T(0))`` for an
unsigned operand, so the operand is evaluated once.
"""

from __future__ import annotations

import re

__all__ = ["emit_metal_integer_sign_helpers", "metal_integer_sign_call"]

_METAL_INTEGER = re.compile(r"(u?(?:char|short|int|long))([234]?)")


def _metal_integer_operand(codegen, type_name):
    """The Metal spelling of an integer scalar or vector type, else ``None``."""
    for candidate in (codegen.map_type(type_name), type_name):
        text = str(candidate or "").strip()
        if _METAL_INTEGER.fullmatch(text):
            return text
    return None


def metal_integer_sign_call(codegen, name, args):
    """The Metal spelling of ``sign`` of an integer operand, or ``None`` for any other call."""
    if name != "sign" or name in codegen.user_function_names or len(args or ()) != 1:
        return None
    argument_type = codegen.expression_result_type(args[0])
    if argument_type is None:
        return None
    operand = _metal_integer_operand(codegen, argument_type)
    if operand is None:
        return None
    helper = f"crosstl_sign_{operand}"
    helpers = codegen.__dict__.setdefault("metal_integer_sign_helpers", {})
    if operand.startswith("u"):
        body = f"{operand}(x != {operand}(0))"
    else:
        body = f"{operand}({operand}(x > {operand}(0)) - {operand}(x < {operand}(0)))"
    helpers.setdefault(
        helper,
        f"static inline {operand} {helper}({operand} x) {{\n    return {body};\n}}\n",
    )
    return f"{helper}({codegen.generate_expression(args[0])})"


def emit_metal_integer_sign_helpers(codegen):
    """The integer ``sign`` helpers the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("metal_integer_sign_helpers", {}).values())
