"""Find and replace unqualified identifiers in a Metal expression.

Pointer-member promotion rewrites constructor initializers that read a
promoted member. Only a name that stands alone refers to that member: in
``other.src``, ``other->src`` or ``Owner::src`` the name belongs to another
object or scope and is left alone.
"""

import re
from typing import List, Mapping

__all__ = [
    "qualified_identifiers",
    "replace_unqualified_identifiers",
    "unqualified_identifiers",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")


def _is_qualified(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    return prefix.endswith((".", "->", "::"))


def unqualified_identifiers(text: str) -> List[str]:
    """Identifiers in ``text`` that no member access or scope qualifies."""
    return [
        match.group(0)
        for match in _IDENTIFIER_RE.finditer(text)
        if not _is_qualified(text, match.start())
    ]


def qualified_identifiers(text: str) -> List[str]:
    """Identifiers in ``text`` that follow ``.``, ``->`` or ``::``."""
    return [
        match.group(0)
        for match in _IDENTIFIER_RE.finditer(text)
        if _is_qualified(text, match.start())
    ]


def replace_unqualified_identifiers(text: str, replacements: Mapping[str, str]) -> str:
    """Replace each unqualified identifier that ``replacements`` names."""
    if not replacements:
        return text
    pieces = []
    position = 0
    for match in _IDENTIFIER_RE.finditer(text):
        name = match.group(0)
        if name not in replacements or _is_qualified(text, match.start()):
            continue
        pieces.append(text[position : match.start()])
        pieces.append(replacements[name])
        position = match.end()
    pieces.append(text[position:])
    return "".join(pieces)
