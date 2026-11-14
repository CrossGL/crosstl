"""The type operand of a CUDA or HIP ``sizeof``, spelled as the CrossGL type it imports as.

``sizeof(long long)``, ``sizeof(size_t)`` or ``sizeof(__nv_bfloat16)`` used to reach CrossGL with the source
spelling. CrossGL cannot parse a multi-word C type there, and it has no ``size_t`` or ``__nv_bfloat16``, so the
operand is converted like any declared type. The value is then folded per target, and a type whose layout is the
target's (a three-component vector, a structure) is refused there rather than here.
"""

from __future__ import annotations

import re

from ...translator.semantics.sizeof_values import (
    SizeofUnsupportedError,
    sizeof_type_bytes,
)

__all__ = ["sizeof_type_operand"]

# Built-in arithmetic type spellings of CUDA C++ and HIP: C keywords, <cstdint> names and the vector types.
_BUILTIN_TYPE = re.compile(
    r"(?:(?:unsigned|signed|long|short|int|char|float|double|bool)\b\s*)+"
    r"|size_t|ptrdiff_t|half2?|__half2?|__nv_bfloat162?"
    r"|u?int(?:8|16|32|64)_t"
    r"|(?:u?char|u?short|u?int|u?long|u?longlong|float|double|half)[1-4]"
)


def sizeof_type_operand(operand, convert_type):
    """Return the CrossGL spelling of a built-in ``sizeof`` type operand, or ``None`` to keep the operand.

    A spelling CrossGL already sizes, such as ``float`` or ``float4``, is kept as written.
    """
    text = " ".join(str(operand).split())
    if not _BUILTIN_TYPE.fullmatch(text):
        return None
    try:
        sizeof_type_bytes(text)
        return None
    except SizeofUnsupportedError:
        pass
    converted = convert_type(text)
    return converted if converted and converted != text else None
