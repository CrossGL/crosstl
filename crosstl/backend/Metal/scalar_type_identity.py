"""Type identity of Metal's built-in scalar aliases in template arguments.

MSL names its integer scalars two ways (MSL section 2.1, table 2.1): ``uint`` and ``uint32_t``, ``ushort`` and
``uint16_t``, ``ulong``, ``uint64_t`` and ``size_t``, and so on. Each pair is one type, so ``max_value<uint>()``
selects the explicit specialization ``max_value<uint32_t>``, but the template machinery compared the spellings and
left the call naming a specialization nobody declared. ``char`` is not ``int8_t``: C++ keeps ``char`` and
``signed char`` distinct, and MSL's ``int8_t`` is ``signed char``.

:func:`canonical_scalar_type_spelling` rewrites every built-in alias in a type spelling to one spelling per type, for
comparing template arguments. The Metal importer names a specialization with the first spelling it meets, so this
never changes the text of a source that spells each specialization one way.
"""

from __future__ import annotations

import re

__all__ = ["canonical_scalar_type_spelling"]

# Built-in spelling -> canonical spelling of the same MSL type (checked with ``is_same`` in the Metal compiler).
_SCALAR_ALIASES = {
    "unsigned int": "uint",
    "unsigned": "uint",
    "uint32_t": "uint",
    "signed int": "int",
    "int32_t": "int",
    "unsigned short int": "ushort",
    "unsigned short": "ushort",
    "uint16_t": "ushort",
    "signed short int": "short",
    "signed short": "short",
    "short int": "short",
    "int16_t": "short",
    "unsigned char": "uchar",
    "uint8_t": "uchar",
    "int8_t": "signed char",
    "unsigned long int": "ulong",
    "unsigned long": "ulong",
    "uint64_t": "ulong",
    "size_t": "ulong",
    "signed long int": "long",
    "signed long": "long",
    "long int": "long",
    "int64_t": "long",
    "ptrdiff_t": "long",
}
# Longest spellings first, so ``unsigned short int`` is not read as ``unsigned`` followed by ``short int``.
_ALIAS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_:])(?:metal::)?("
    + "|".join(
        re.escape(spelling).replace(r"\ ", r"\s+")
        for spelling in sorted(_SCALAR_ALIASES, key=len, reverse=True)
    )
    + r")(?![A-Za-z0-9_])"
)


def canonical_scalar_type_spelling(type_text: str) -> str:
    """``type_text`` with each built-in scalar alias spelled canonically (``device uint32_t*`` -> ``device uint*``)."""
    if not isinstance(type_text, str) or not type_text:
        return type_text
    return _ALIAS_PATTERN.sub(
        lambda match: _SCALAR_ALIASES[" ".join(match.group(1).split())], type_text
    )
