"""Metal builtin arguments of 8- and 16-bit integer scalars.

The Metal generator keeps an 8- or 16-bit integer scalar register in a 32-bit ``int`` or ``uint`` and reduces each
value written to it (``uint(uchar(e))``), while an element of a narrow buffer or array keeps its own type
(``device uchar*``). Metal overloads ``min``, ``max`` and ``clamp`` for each integer width but applies no usual
arithmetic conversions to their arguments, so ``min(ids[i], uint(uchar(n)))`` mixed ``uchar`` and
``uint`` and matched no overload. Under CrossGL's rules a narrow scalar operand is promoted to ``int`` (C++
[conv.prom]), so the call computes on the 32-bit register type; each narrow scalar argument is converted to it
explicitly.
"""

from __future__ import annotations

__all__ = ["NARROW_REGISTER_UNIFIED_BUILTINS", "narrow_scalar_register_type"]

# Builtins whose integer overloads take every argument at one type.
NARROW_REGISTER_UNIFIED_BUILTINS = frozenset({"clamp", "max", "min"})

# CrossGL spellings of the 8- and 16-bit integer scalars -> the Metal register type that holds them.
_NARROW_SCALAR_REGISTERS = {
    "i8": "int",
    "int8": "int",
    "int8_t": "int",
    "char": "int",
    "signed char": "int",
    "i16": "int",
    "int16": "int",
    "int16_t": "int",
    "short": "int",
    "u8": "uint",
    "uint8": "uint",
    "uint8_t": "uint",
    "uchar": "uint",
    "unsigned char": "uint",
    "u16": "uint",
    "uint16": "uint",
    "uint16_t": "uint",
    "ushort": "uint",
    "unsigned short": "uint",
}


def narrow_scalar_register_type(type_name):
    """The 32-bit register type of an 8- or 16-bit integer scalar type spelling, else ``None``."""
    if not isinstance(type_name, str):
        return None
    return _NARROW_SCALAR_REGISTERS.get(" ".join(type_name.split()))
