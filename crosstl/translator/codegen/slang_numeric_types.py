"""Numeric types that Slang output cannot represent.

Slang's floating-point scalars are ``half``, ``float`` and ``double``; it has no bfloat16 type. The Slang generator
printed a CrossGL ``bfloat16`` (an MSL ``bfloat``) unchanged, so ``bfloat16 x`` or ``RWStructuredBuffer<bfloat16>``
reached slangc as an undefined identifier. Writing it as ``half`` or ``float`` would change its range, precision and
storage size. :func:`slang_unrepresentable_type_reason` names such a type so the generator refuses the program
instead. Emulating bfloat16 on 16-bit storage, as the HLSL generator does, is the lowering that would replace the
refusal.
"""

from __future__ import annotations

import re

__all__ = ["slang_unrepresentable_type_reason"]

_BFLOAT16 = re.compile(r"(?:bfloat16(?:_t)?|bfloat|bf16)(?:vec[234]|[234](?:x[234])?)?")


def slang_unrepresentable_type_reason(type_name) -> str | None:
    """Why Slang cannot spell the CrossGL type ``type_name``, or ``None`` when it can."""
    spelling = " ".join(str(type_name).split())
    if _BFLOAT16.fullmatch(spelling):
        return (
            f"Slang has no bfloat16 type for '{spelling}'; half or float would change its range, precision "
            "and storage size"
        )
    return None
