"""Struct member declarations that declare several members.

``struct P { uint32_t m, n, k; long stride; };`` declares three ``uint32_t`` members: each member-declarator is
analyzed as if it were in a declaration by itself ([dcl.decl]/3, [class.mem]). The struct-body scanner recorded such
a declaration as one member named after its last declarator, typed ``uint32_t m, n,``, so the members before it were
missing and the struct had no computable layout: ``sizeof(P)`` in a static assertion stayed unresolved and the
kernel was refused.

:func:`split_member_declarator_list` gives the single-member declarations a plain declarator list stands for.
"Plain" means value members without initializers, pointers, references, bit-fields, attributes or storage-class
specifiers. Any other list returns ``None`` and keeps the scanner's previous, layout-refusing behaviour.
"""

from __future__ import annotations

import re

__all__ = ["split_member_declarator_list"]

_EXCLUDED_CHARACTERS = frozenset("=*&{}();")
# A single colon is a bit-field width; ``::`` qualifies a type name.
_BIT_FIELD = re.compile(r"(?<!:):(?!:)")
_EXCLUDED_WORDS = re.compile(
    r"\b(?:static|typedef|using|friend|template|operator|mutable|thread_local|extern)\b"
)
_FIRST_DECLARATOR = re.compile(
    r"(?P<type>.*?\S)\s+(?P<declarator>[A-Za-z_]\w*\s*(?:\[[^\[\]]*\]\s*)*)", re.DOTALL
)
_NEXT_DECLARATOR = re.compile(r"[A-Za-z_]\w*\s*(?:\[[^\[\]]*\]\s*)*")


def _top_level_parts(text: str) -> list[str] | None:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character in "<[":
            depth += 1
        elif character in ">]":
            depth -= 1
            if depth < 0:
                return None
        elif character == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    if depth != 0:
        return None
    parts.append(text[start:])
    return parts


def split_member_declarator_list(declaration: str) -> list[str] | None:
    """``["T a", "T b[2]"]`` for the member declaration ``T a, b[2]``, or ``None`` when it is not a plain list."""
    text = " ".join(declaration.split())
    if "," not in text or _EXCLUDED_CHARACTERS & set(text) or "[[" in text:
        return None
    if _BIT_FIELD.search(text):
        return None
    if _EXCLUDED_WORDS.search(text):
        return None
    parts = _top_level_parts(text)
    if parts is None or len(parts) < 2:
        return None
    first = _FIRST_DECLARATOR.fullmatch(parts[0].strip())
    if first is None:
        return None
    declarators = [first.group("declarator").strip()]
    for part in parts[1:]:
        declarator = part.strip()
        if _NEXT_DECLARATOR.fullmatch(declarator) is None:
            return None
        declarators.append(declarator)
    type_text = first.group("type")
    return [f"{type_text} {declarator}" for declarator in declarators]
