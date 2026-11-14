"""Resolve ``decltype(<local>)`` in template arguments to the local's declared type.

For an unparenthesized id-expression that names a variable, ``decltype`` denotes the type the variable was declared
with ([dcl.type.decltype]/1.3). Template materialization substitutes that type before it looks for function-local
names it cannot resolve, so ``Limits<typename decltype(tile)::elem_type>`` materializes exactly like
``Limits<typename Tile<float, 2>::elem_type>``.

Only declarations whose type text is the complete declared type qualify: no array suffix, pointer, reference,
cv or address-space qualifier, ``auto`` or storage class. A parenthesized operand (``decltype((tile))``) denotes an
lvalue reference ([dcl.type.decltype]/1.5) and is never rewritten. Anything not rewritten stays in the argument, so
the caller's unresolved-local diagnostic still refuses it.
"""

import re
from typing import Iterable, Mapping, Optional

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_LOCAL_DECLTYPE_RE = re.compile(r"(?<![\w:.])decltype\s*\(\s*([A-Za-z_]\w*)\s*\)")
_DISQUALIFYING_TYPE_TOKENS = frozenset(
    {
        "auto",
        "const",
        "constant",
        "constexpr",
        "device",
        "extern",
        "register",
        "static",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
        "ray_data",
        "object_data",
        "volatile",
    }
)


def complete_declared_type(
    type_text: str, array_suffix: Optional[str]
) -> Optional[str]:
    """The declared type ``decltype`` names for a local with this declaration, or ``None`` if it cannot be spelled."""
    text = " ".join(str(type_text or "").split())
    if not text or array_suffix or "*" in text or "&" in text or "[" in text:
        return None
    if any(
        token in _DISQUALIFYING_TYPE_TOKENS for token in _IDENTIFIER_RE.findall(text)
    ):
        return None
    return text


def substitute_local_decltypes(argument: str, declared_types: Mapping[str, str]) -> str:
    """Replace each ``decltype(name)`` whose ``name`` has a known declared type."""
    if not declared_types or "decltype" not in argument:
        return argument

    def replace(match: "re.Match[str]") -> str:
        declared = declared_types.get(match.group(1))
        return match.group(0) if declared is None else declared

    return _LOCAL_DECLTYPE_RE.sub(replace, argument)


def substitute_local_decltypes_in(
    arguments: Iterable[str], declared_types: Mapping[str, str]
) -> list:
    """:func:`substitute_local_decltypes` over every template argument, in order."""
    return [
        substitute_local_decltypes(argument, declared_types) for argument in arguments
    ]
