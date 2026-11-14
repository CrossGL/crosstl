"""``mix(x, y, a)`` with a boolean selector (GLSL 4.60 section 8.3).

With a ``bool`` or ``bvecN`` selector, ``mix`` selects: each component of the result is ``y`` where ``a`` is true
and ``x`` where it is false. It is not ``x * (1 - a) + y * a`` with ``a`` converted to a number: that form turns an
infinite or NaN operand that is not selected into NaN. Both value operands are evaluated, as for any call.

GLSL has the selecting overload natively. Metal spells it ``select(x, y, a)`` (MSL section 6.4); HLSL 2021 spells
it ``select(a, y, x)``. A scalar selector with vector values is splatted to the value width first.
"""

from __future__ import annotations

import re

__all__ = ["boolean_mix_selector", "hlsl_boolean_mix", "metal_boolean_mix"]

_BOOLEAN = re.compile(r"bool([234]?)|bvec([234])")
_VECTOR_WIDTH = re.compile(r"[a-z]*?([234])")


def _width(type_name):
    match = _BOOLEAN.fullmatch(type_name or "")
    if match:
        return int(match.group(1) or match.group(2) or 1)
    match = _VECTOR_WIDTH.fullmatch(type_name or "")
    return int(match.group(1)) if match else 1


def boolean_mix_selector(codegen, func_name, args, user_functions=()):
    """``(selector width, value width)`` for a ``mix`` call with a boolean selector, else ``None``."""
    if func_name != "mix" or len(args or ()) != 3 or func_name in user_functions:
        return None
    selector_type = codegen.expression_result_type(args[2])
    if not selector_type or not _BOOLEAN.fullmatch(str(selector_type)):
        return None
    value_types = [codegen.expression_result_type(arg) for arg in args[:2]]
    return _width(str(selector_type)), max(_width(str(t)) for t in value_types)


def _selector(codegen, rendered, widths, spell):
    selector_width, value_width = widths
    if selector_width == value_width:
        return rendered
    if selector_width == 1:
        return f"{spell(value_width)}({rendered})"
    return codegen.refuse_lowering(
        f"mix with a {selector_width}-component selector and {value_width}-component values",
        rendered,
    )


def metal_boolean_mix(codegen, args, widths):
    """``select(x, y, a)`` for Metal."""
    x, y, selector = (codegen.generate_expression(arg) for arg in args)
    selector = _selector(codegen, selector, widths, lambda n: f"bool{n}")
    return f"select({x}, {y}, {selector})"


def hlsl_boolean_mix(codegen, args, widths):
    """``select(a, y, x)`` for HLSL 2021."""
    x, y, selector = (codegen.generate_expression(arg) for arg in args)
    selector = _selector(codegen, selector, widths, lambda n: f"(bool{n})")
    return f"select({selector}, {y}, {x})"
