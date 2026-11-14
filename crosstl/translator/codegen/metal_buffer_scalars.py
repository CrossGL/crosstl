"""Scalar types a Metal kernel can take by value from a buffer slot.

A kernel parameter without an attribute is a stage input, and a kernel has no stage inputs, so ``xcrun metal``
rejects ``uint64_t size`` ("invalid type for input declaration in a kernel function"). A scalar the host binds to
a buffer slot is passed as ``constant T& name [[buffer(n)]]`` (the specification's "Attributes to Locate Buffer,
Texture, and Sampler Arguments"), whatever its width: MSL has 8, 16, 32 and 64-bit integers and 16 and 32-bit
floats (section 2.1, "Scalar Data Types"), plus ``bfloat`` from MSL 3.1.

The reference keeps the source's width. The generator computes on 8 and 16-bit integers as 32-bit ``int`` or
``uint`` with truncating casts, but a ``constant uint&`` bound to a host's two-byte ``ushort`` reads two bytes
past it, so :func:`metal_buffer_scalar_type` names the exact-width type.
"""

from __future__ import annotations

from ..validation import is_numeric_scalar_type

__all__ = ["is_metal_buffer_scalar_type_name", "metal_buffer_scalar_type"]

_METAL_BUFFER_SCALARS = frozenset(
    {
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "half",
        "float",
        "bfloat",
    }
)


def is_metal_buffer_scalar_type_name(type_name, map_type) -> bool:
    """Whether ``type_name`` (CrossGL or Metal) is a numeric scalar that a buffer slot can carry."""
    if not type_name or "[" in str(type_name):
        return False
    if is_numeric_scalar_type(type_name):
        return True
    return str(map_type(type_name)) in _METAL_BUFFER_SCALARS


# CrossGL and C spellings of the 8 and 16-bit integers, which the generator otherwise maps to int or uint.
_EXACT_WIDTH = {
    **dict.fromkeys(("i8", "int8", "int8_t", "char", "signed char"), "char"),
    **dict.fromkeys(("u8", "uint8", "uint8_t", "uchar", "unsigned char"), "uchar"),
    **dict.fromkeys(("i16", "int16", "int16_t", "short", "signed short"), "short"),
    **dict.fromkeys(("u16", "uint16", "uint16_t", "ushort", "unsigned short"), "ushort"),
}


def metal_buffer_scalar_type(type_name, mapped_type: str) -> str:
    """The Metal type of a buffer-bound scalar of CrossGL type ``type_name``, which the generator maps to ``mapped_type``."""
    return _EXACT_WIDTH.get(str(type_name).strip(), mapped_type)
