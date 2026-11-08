"""Integral conversions in Metal compile-time integral expressions.

Static assertions and constant initializers convert integral values: PyTorch's MPS headers define
``#define MAX_THREADGROUP_SIZE static_cast<uint32_t>(1024)`` and assert ``INNER_TG_SIZE <= MAX_THREADGROUP_SIZE``.
The integral folder parsed neither ``static_cast<T>(v)`` nor the functional cast ``T(v)``, so such an assertion
stayed unresolved and the kernel was refused.

A conversion folds when its result is exact: to ``bool`` it yields ``v != 0`` ([conv.bool]); to another integer type
it yields ``v`` when ``v`` is representable in that type ([conv.integral]/3). A value the type cannot represent would
wrap or be implementation-defined, so it does not fold and the caller keeps refusing. Metal's integer widths follow
the Metal Shading Language Specification, section 2.1 (``long`` and ``size_t`` are 64-bit).

:func:`rewrite_integral_casts` turns ``static_cast<T>(`` into a call of a reserved name, before the expression is
parsed; :func:`fold_integral_cast` evaluates that call, or a functional cast, once its operand is known.
"""

from __future__ import annotations

import re

__all__ = ["fold_integral_cast", "integral_cast_target", "rewrite_integral_casts"]

_CAST_PREFIX = "__crossgl_integral_cast_"

_SIGNED = {8: (-(2**7), 2**7 - 1), 16: (-(2**15), 2**15 - 1)}
_SIGNED.update({32: (-(2**31), 2**31 - 1), 64: (-(2**63), 2**63 - 1)})
_UNSIGNED = {bits: (0, 2**bits - 1) for bits in (8, 16, 32, 64)}

# Spelling (whitespace collapsed to one space) -> inclusive range, or None for bool.
_INTEGRAL_TYPES = {
    "bool": None,
    "char": _SIGNED[8],
    "signed char": _SIGNED[8],
    "int8_t": _SIGNED[8],
    "unsigned char": _UNSIGNED[8],
    "uchar": _UNSIGNED[8],
    "uint8_t": _UNSIGNED[8],
    "short": _SIGNED[16],
    "int16_t": _SIGNED[16],
    "unsigned short": _UNSIGNED[16],
    "ushort": _UNSIGNED[16],
    "uint16_t": _UNSIGNED[16],
    "int": _SIGNED[32],
    "signed": _SIGNED[32],
    "signed int": _SIGNED[32],
    "int32_t": _SIGNED[32],
    "unsigned": _UNSIGNED[32],
    "unsigned int": _UNSIGNED[32],
    "uint": _UNSIGNED[32],
    "uint32_t": _UNSIGNED[32],
    "long": _SIGNED[64],
    "long long": _SIGNED[64],
    "int64_t": _SIGNED[64],
    "ptrdiff_t": _SIGNED[64],
    "unsigned long": _UNSIGNED[64],
    "unsigned long long": _UNSIGNED[64],
    "ulong": _UNSIGNED[64],
    "uint64_t": _UNSIGNED[64],
    "size_t": _UNSIGNED[64],
}

_STATIC_CAST_RE = re.compile(
    r"\bstatic_cast\s*<\s*(?:(?:::)?\s*metal\s*::\s*)?(?P<type>[A-Za-z_][\w ]*?)\s*>\s*\("
)


def _normalized(type_text: str) -> str:
    return " ".join(type_text.split())


def rewrite_integral_casts(expression: str) -> str:
    """``expression`` with each ``static_cast`` to an integral type spelled as a call the folder recognises."""
    if "static_cast" not in expression:
        return expression

    def replace(match: re.Match) -> str:
        spelling = _normalized(match.group("type"))
        if spelling not in _INTEGRAL_TYPES:
            return match.group(0)
        return f"{_CAST_PREFIX}{spelling.replace(' ', '_')}("

    return _STATIC_CAST_RE.sub(replace, expression)


def integral_cast_target(callee: str) -> str | None:
    """The integral type a call named ``callee`` converts to, or ``None`` when it is no integral conversion."""
    if callee.startswith(_CAST_PREFIX):
        spelling = callee[len(_CAST_PREFIX) :].replace("_", " ")
        # Spellings such as ``uint32_t`` contain an underscore of their own.
        for candidate in (spelling, callee[len(_CAST_PREFIX) :]):
            if candidate in _INTEGRAL_TYPES:
                return candidate
        return None
    return callee if callee in _INTEGRAL_TYPES and " " not in callee else None


def fold_integral_cast(target: str, value: int) -> int | None:
    """The exact result of converting ``value`` to the integral type ``target``, or ``None`` when it is not exact."""
    bounds = _INTEGRAL_TYPES[target]
    if bounds is None:
        return int(value != 0)
    low, high = bounds
    return value if low <= value <= high else None
