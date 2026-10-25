"""Comparisons of a signed with an unsigned 32-bit integer, spelled with the conversion GLSL and HLSL apply.

GLSL converts an ``int`` operand of a comparison with a ``uint`` to ``uint`` (GLSL 4.60 section 4.1.10, "Implicit
Conversions"), and HLSL does the same. MSL's usual arithmetic conversions compute the same result, but the Metal
compiler warns ("comparison of integers of different signs", ``-Wsign-compare``) and the warning fails a
warnings-as-errors build. :func:`metal_sign_matched_comparison_operands` writes the conversion explicitly:
``id.x < count`` with a ``uint`` ``id.x`` and an ``int`` ``count`` becomes ``id.x < uint(count)``.
"""

from __future__ import annotations

import re

__all__ = ["metal_sign_matched_comparison_operands"]

_INTEGER = re.compile(r"(u?)int([234]?)")
_NON_NEGATIVE_LITERAL = re.compile(r"\(?(?:0[xX][0-9a-fA-F]+|\d+)\)?")


def _integer_shape(type_name):
    match = _INTEGER.fullmatch(str(type_name or "").strip())
    if match is None:
        return None
    return match.group(1) == "u", match.group(2)


def metal_sign_matched_comparison_operands(left_type, right_type, left, right):
    """``(left, right)`` with the signed operand converted to the unsigned operand's type when the signs differ.

    Only 32-bit ``int``/``uint`` scalars and vectors are matched: a vector operand gives the converted type its
    width, and two vectors must have the same width.
    """
    left_shape, right_shape = _integer_shape(left_type), _integer_shape(right_type)
    if left_shape is None or right_shape is None or left_shape[0] == right_shape[0]:
        return left, right
    left_width, right_width = left_shape[1], right_shape[1]
    if left_width and right_width and left_width != right_width:
        return left, right
    signed = right if left_shape[0] else left
    if _NON_NEGATIVE_LITERAL.fullmatch(str(signed).strip()):
        # A non-negative literal has the same value as either type, and the compiler does not warn.
        return left, right
    if left_shape[0]:
        return left, f"uint{left_width or right_width}({right})"
    return f"uint{left_width or right_width}({left})", right
