"""Type a vector swizzle applied to any typed postfix expression.

``MetalPreprocessor._infer_argument_type`` typed ``v.xy`` only when ``v`` is a typed local. A component of an
element, a call result or a member (``partials[i].x``, ``load(p).yz``, ``s.v.w``) stayed untyped, so a
constrained call taking one was refused. Here the base is typed by the normal inference, and when that type
is a built-in vector the swizzle has the element type (one component) or the vector of the selected width
([MSL specification, section 2.2.1, "Accessing Vector Components"]).

The base must be a postfix expression (a name followed by subscripts, calls and member accesses) or fully
parenthesized, so ``a + b.x`` is never read as ``(a + b).x``. Structs with an ``x`` member, pointers,
unknown bases, mixed ``xyzw``/``rgba`` sets and out-of-range components stay untyped.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

__all__ = ["infer_swizzle_type"]

Infer = Callable[[str], Optional[str]]

_IDENTIFIER = re.compile(r"(?:::)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*")
# Canonical spellings of `_scalar_and_width` bases that have Metal names.
_ELEMENT_SPELLINGS = {"bfloat16_t": "bfloat", "float16_t": "half"}
_MEMBER = re.compile(r"\s*(?:\.|->)\s*[A-Za-z_]\w*")


def _is_postfix_expression(preprocessor, text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text[0] == "(":
        close = preprocessor._find_matching_delimiter(text, 0, "(", ")")
        position = close + 1 if close is not None else None
    else:
        match = _IDENTIFIER.match(text)
        position = match.end() if match is not None else None
    if position is None:
        return False
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        opener = text[position]
        if opener in "[(":
            closer = "]" if opener == "[" else ")"
            close = preprocessor._find_matching_delimiter(
                text, position, opener, closer
            )
            if close is None:
                return False
            position = close + 1
            continue
        member = _MEMBER.match(text, position)
        if member is None:
            return False
        position = member.end()
    return True


def _split_member(text: str):
    """``(base, member)`` at the last top-level ``.``, or None."""
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        character = text[index]
        if character in ")]":
            depth += 1
        elif character in "([":
            depth -= 1
        elif character == "." and depth == 0:
            if index > 0 and text[index - 1].isdigit():
                return None  # a floating literal such as 1.5
            return text[:index], text[index + 1 :]
    return None


def infer_swizzle_type(preprocessor, expr: str, infer: Infer) -> str | None:
    """The type of ``base.swizzle`` when ``base`` has a built-in vector type."""
    split = _split_member(expr)
    if split is None:
        return None
    base, swizzle = split[0].strip(), split[1].strip()
    if not re.fullmatch(r"[xyzwrgba]{1,4}", swizzle):
        return None
    if not _is_postfix_expression(preprocessor, base):
        return None
    base_type = infer(base)
    if base_type is None or "*" in base_type or "&" in base_type:
        return None
    decomposed = preprocessor._scalar_and_width(base_type)
    if decomposed is None:
        return None
    element, width = decomposed
    if width not in (2, 3, 4):
        return None
    components = "xyzw" if swizzle[0] in "xyzw" else "rgba"
    if any(component not in components[:width] for component in swizzle):
        return None
    if element in _ELEMENT_SPELLINGS:
        element = _ELEMENT_SPELLINGS[element]
    if len(swizzle) == 1:
        return preprocessor._normalize_inferred_expression_type(element)
    return preprocessor._normalize_inferred_expression_type(f"{element}{len(swizzle)}")
