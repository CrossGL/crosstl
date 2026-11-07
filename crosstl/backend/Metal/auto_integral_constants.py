"""The integral type of a ``constexpr auto`` local, for the compile-time integral evaluator.

Kernels name a file-scope constant locally through a placeholder: PyTorch's ``ScaledMM.metal`` writes
``constexpr auto tile = scaled_mm_tile;`` and then ``constexpr uint block = tile / sg_grid;`` and
``static_assert(frags * frag * sg_grid == tile, ...)``. The integral-constant collector accepted only declarations
spelled with an integral type, so ``tile`` and everything computed from it stayed unresolved and the assertion
refused the kernel.

``auto`` takes the type of its initializer ([dcl.type.auto.deduct]). :func:`deduced_integral_auto_type` gives that
type when it is evident without overload resolution or usual arithmetic conversions: an initializer that names a
visible constant whose integral type is known, or a single integer or Boolean literal ([lex.icon]: no suffix is
``int``, then ``long``; ``u`` is ``uint``, then ``ulong``; ``l`` is ``long``; ``ul`` is ``ulong``). Any other
initializer gives ``None``, which leaves the local unresolved as before.
"""

from __future__ import annotations

import re
from typing import Mapping

__all__ = ["deduced_integral_auto_type"]

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_INTEGER_LITERAL = re.compile(
    r"(?P<digits>0[xX][0-9A-Fa-f]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)(?P<suffix>[uUlL]*)"
)
_SIGNED_INT_MAX = 2**31 - 1
_UNSIGNED_INT_MAX = 2**32 - 1


def _literal_type(text: str) -> str | None:
    if text in {"true", "false"}:
        return "bool"
    match = _INTEGER_LITERAL.fullmatch(text)
    if match is None:
        return None
    suffix = match.group("suffix").lower()
    digits = match.group("digits")
    value = int(digits, 0) if not re.fullmatch(r"0[0-7]+", digits) else int(digits, 8)
    if suffix == "":
        return "int" if value <= _SIGNED_INT_MAX else "long"
    if suffix == "u":
        return "uint" if value <= _UNSIGNED_INT_MAX else "ulong"
    if suffix in {"l", "ll"}:
        return "long"
    if suffix in {"ul", "lu", "ull", "llu"}:
        return "ulong"
    return None


def deduced_integral_auto_type(
    initializer: str, integral_types: Mapping[str, str]
) -> str | None:
    """The integral type ``auto`` deduces from ``initializer``, or ``None`` when it is not evident.

    ``integral_types`` maps each visible integral constant to its integral scalar type.
    """
    text = initializer.strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if _IDENTIFIER.fullmatch(text):
        if text in {"true", "false"}:
            return "bool"
        return integral_types.get(text)
    return _literal_type(text)
