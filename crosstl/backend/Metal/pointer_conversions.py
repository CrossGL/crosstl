"""Types of explicit Metal pointer conversions.

``reinterpret_cast<P>(e)`` has type ``P`` ([expr.reinterpret.cast]/1), and so
does the C-style cast ``(P)e`` ([expr.cast]/1). Template argument deduction
needs that type for a call argument such as ``put(reinterpret_cast<device
Box<float>*>(out), ...)`` and for an ``auto`` local initialized from one, which
takes the initializer's type ([dcl.type.auto.deduct]).

Only a single-level pointer naming exactly one Metal address space is typed,
and only when the cast spans the whole expression. Anything else, including a
cast followed by a binary tail, stays un-inferable so deduction fails closed.
"""

import re

_REINTERPRET_RE = re.compile(r"reinterpret_cast\s*<")


def pointer_conversion_type(preprocessor, expression):
    """Target pointer type of a whole-expression pointer conversion, or None."""
    text = expression.strip()
    match = _REINTERPRET_RE.match(text)
    if match is None:
        return preprocessor._c_style_pointer_cast_type(text)
    angle_start = match.end() - 1
    angle_end = preprocessor._find_matching_angle(text, angle_start)
    if angle_end is None:
        return None
    paren_start = angle_end + 1
    while paren_start < len(text) and text[paren_start].isspace():
        paren_start += 1
    if paren_start >= len(text) or text[paren_start] != "(":
        return None
    paren_end = preprocessor._find_matching_delimiter(text, paren_start, "(", ")")
    if paren_end != len(text) - 1:
        return None
    if not text[paren_start + 1 : paren_end].strip():
        return None
    return preprocessor._normalize_known_address_space_pointer_type(
        text[angle_start + 1 : angle_end]
    )
