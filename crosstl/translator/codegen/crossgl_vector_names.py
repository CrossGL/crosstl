"""Named spellings of CrossGL's generic vector types.

CrossGL writes a vector type two ways: generically, ``vec2<u16>`` (the spelling the CUDA and HIP importers use for
``ushort2`` and its relatives, which the parser stores as ``VectorType(u16, 2)``), and by name, ``u16vec2``. The
GLSL and HLSL generators lower the named spellings, including the wrap-around emulation that 8-bit and 16-bit
integers need, but a generic vector over a 64-bit, 16-bit or 8-bit element fell through as ``i642`` or
``vec2<u16>``, which names no type on either target, and a generic 8-bit vector lost its wrap-around.

Generators that map vectors by name rewrite the generic spelling to the named one first.
"""

from __future__ import annotations

import re

__all__ = ["crossgl_named_vector_spelling", "crossgl_named_vector_type"]

_GENERIC_VECTOR = re.compile(r"^vec([234])\s*<\s*([A-Za-z0-9_]+)\s*>$")

# CrossGL scalar spelling -> prefix of the named vector (``u16`` -> ``u16vec``).
_NAMED_VECTOR_PREFIXES = {
    "float": "",
    "f32": "",
    "int": "i",
    "i32": "i",
    "uint": "u",
    "u32": "u",
    "bool": "b",
    "double": "d",
    "f64": "d",
    "half": "f16",
    "f16": "f16",
    "float16_t": "f16",
    "bfloat16": "bfloat16",
    "bfloat16_t": "bfloat16",
    "i8": "i8",
    "int8_t": "i8",
    "u8": "u8",
    "uint8_t": "u8",
    "i16": "i16",
    "int16_t": "i16",
    "u16": "u16",
    "uint16_t": "u16",
    "i64": "i64",
    "int64_t": "i64",
    "u64": "u64",
    "uint64_t": "u64",
}


def crossgl_named_vector_type(element_type, size):
    """The named CrossGL vector for ``element_type`` and ``size`` (``u16``, 2 -> ``u16vec2``), else ``None``.

    Elements without a named CrossGL vector (struct and template names) return ``None`` so the caller keeps its own
    handling.
    """
    if not isinstance(element_type, str) or str(size) not in {"2", "3", "4"}:
        return None
    prefix = _NAMED_VECTOR_PREFIXES.get(element_type.strip())
    if prefix is None:
        return None
    return f"{prefix}vec{size}"


def crossgl_named_vector_spelling(type_name):
    """``type_name`` with a generic vector spelling replaced by its named one (``vec2<i64>`` -> ``i64vec2``)."""
    if not isinstance(type_name, str):
        return type_name
    match = _GENERIC_VECTOR.match(type_name.strip())
    if match is None:
        return type_name
    named = crossgl_named_vector_type(match.group(2), match.group(1))
    return type_name if named is None else named
