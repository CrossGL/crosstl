"""The width of HLSL ``half``: binary32 unless the program is compiled with native 16-bit types.

DXC maps ``half`` to ``float`` unless ``-enable-16bit-types`` is given (Shader Model 6.2, HLSL 2018 and later); the
HLSL scalar-types documentation says "Direct3D 10 shader targets map all half data types to float data types". With
the option, ``half`` is ``float16_t``, a binary16 value. A source that names ``float16_t``, ``int16_t`` or
``uint16_t`` compiles only with the option, so in such a program ``half`` is binary16 too; any other program is read
as DXC reads it by default, and ``half`` computes in binary32.

The importer used to give every ``half`` binary16, so a program compiled without the option computed in binary32 on
Direct3D and in binary16 after translation: ``half v = half(x); v * v`` differed in its low 13 bits.

:func:`resolve_native_16_bit_types` decides the width for a preprocessed source, and :func:`binary32_half_token`
respells a ``half``, ``halfN`` or ``halfRxC`` token as its ``float`` form when the program has no native 16-bit
types. The source option ``enable_16bit_types`` states the compilation mode explicitly; with it, the preprocessor
also defines ``__HLSL_ENABLE_16_BIT`` as DXC does.
"""

from __future__ import annotations

import re

__all__ = [
    "ENABLE_16_BIT_MACRO",
    "binary32_half_token",
    "names_native_16_bit_type",
    "resolve_native_16_bit_types",
]

# DXC defines this macro (value 1) when it compiles with -enable-16bit-types.
ENABLE_16_BIT_MACRO = "__HLSL_ENABLE_16_BIT"
# The types that exist only under -enable-16bit-types: scalars, vectors and matrices.
_NATIVE_16_BIT = re.compile(r"\b(?:float16_t|int16_t|uint16_t)(?:[1-4](?:x[1-4])?)?\b")
_HALF_TOKEN = re.compile(r"half((?:[1-4])(?:x[1-4])?)?")
_HALF_TOKEN_TYPES = frozenset({"HALF", "FVECTOR", "MATRIX"})


def names_native_16_bit_type(code: str) -> bool:
    """Whether preprocessed HLSL ``code`` names a type that only ``-enable-16bit-types`` declares."""
    return _NATIVE_16_BIT.search(code) is not None


def resolve_native_16_bit_types(code: str, enable_16bit_types: bool | None) -> bool:
    """Whether ``half`` is binary16 in ``code``: the explicit option, otherwise whether the source needs it."""
    if enable_16bit_types is not None:
        return bool(enable_16bit_types)
    return names_native_16_bit_type(code)


def binary32_half_token(token_type: str, text: str) -> tuple[str, str] | None:
    """The ``float`` token that a ``half`` token means without native 16-bit types, or ``None`` for other tokens."""
    if token_type not in _HALF_TOKEN_TYPES:
        return None
    match = _HALF_TOKEN.fullmatch(text)
    if match is None:
        return None
    shape = match.group(1) or ""
    if not shape:
        return "FLOAT", "float"
    return token_type, f"float{shape}"
