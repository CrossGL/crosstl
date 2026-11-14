"""Result types of built-in arithmetic for implicit template materialization.

``_metal_expression_type`` types call arguments so an implicit template call can be materialized
(``g(u + l)`` becomes ``g<T>`` with ``T`` the argument's type). Its binary rule used to return the "widest"
operand from a fixed preference list and ignored untyped operands, so ``uint + long`` became ``uint`` (C++
gives ``long``), ``float4 * float`` became ``float`` and ``uint + <unknown>`` became ``uint``. Each of these
materialized the wrong specialization and converted the argument to it, silently.

The result now follows the Metal compiler (checked with ``xcrun metal -std=metal3.1``):

* every operand must be typed;
* integral scalars use the integral promotions and the usual arithmetic conversions;
* a floating operand converts an integral one; ``float`` outranks ``half`` and ``bfloat``, which are unordered;
* a vector with a scalar has the vector type, except an integral vector with a floating scalar, a ``half``
  vector with a ``bfloat`` scalar and a ``bfloat`` vector with a ``half`` or ``float`` scalar (ill-formed);
* vectors of different types, ``bool`` vectors, and ``%``, ``|``, ``&``, ``^`` on floating operands are untyped.

A pointer plus or minus an integral offset has the pointer's type, wherever the pointer comes from (a member
``self.dst``, a subscript, a call). Other operand types (structs, matrices) are typed only when all operands have
the same type.
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = ["arithmetic_result_type"]

_FLOATING = {
    "half": "half",
    "float16_t": "half",
    "bfloat16_t": "bfloat",
    "float": "float",
}
_FLOATING_RANK = {"half": 1, "bfloat": 1, "float": 2}
_INTEGRAL_ONLY_OPERATORS = frozenset({"%", "|", "&", "^"})


def _value_type(type_text: str) -> str:
    """A reference or qualified name used as a value: ``constant half&`` is ``half``."""
    text = re.sub(r"\s*&+\s*$", "", type_text)
    text = re.sub(
        r"\b(?:const|volatile|constant|device|thread|threadgroup)\b", " ", text
    )
    return " ".join(text.split())


def _operand(preprocessor, type_text: str):
    if "*" in type_text or "&" in _value_type(type_text):
        return None
    decomposed = preprocessor._scalar_and_width(_value_type(type_text))
    if decomposed is None:
        return None
    base, width = decomposed
    if base in _FLOATING:
        kind = _FLOATING[base]
    elif base in preprocessor._METAL_INTEGRAL_SCALAR_TYPES:
        kind = "integral"
    else:
        return None
    if width != 1 and base == "bool":
        return None
    return kind, width


def _vector_accepts(vector_kind: str, scalar_kind: str) -> bool:
    if vector_kind == "integral":
        return scalar_kind == "integral"
    if vector_kind == "float":
        return True
    if vector_kind == "half":
        return scalar_kind != "bfloat"
    return scalar_kind in {"integral", "bfloat"}


def _combine(preprocessor, left: str, right: str, operator: str) -> str | None:
    if _operand(preprocessor, left) is not None:
        left = _value_type(left)
    if _operand(preprocessor, right) is not None:
        right = _value_type(right)
    left_operand = _operand(preprocessor, left)
    right_operand = _operand(preprocessor, right)
    if left_operand is None or right_operand is None:
        # Not a built-in arithmetic type: only one type with itself is known.
        return (
            left
            if left == right and left_operand is None and right_operand is None
            else None
        )
    left_kind, left_width = left_operand
    right_kind, right_width = right_operand
    if operator in _INTEGRAL_ONLY_OPERATORS and "integral" != left_kind:
        return None
    if operator in _INTEGRAL_ONLY_OPERATORS and "integral" != right_kind:
        return None
    if left_width != 1 and right_width != 1:
        return left if left == right else None
    if left_width != 1 or right_width != 1:
        if left_width != 1:
            vector, vector_kind, scalar_kind = left, left_kind, right_kind
        else:
            vector, vector_kind, scalar_kind = right, right_kind, left_kind
        return vector if _vector_accepts(vector_kind, scalar_kind) else None
    if left_kind == "integral" and right_kind == "integral":
        return preprocessor._integral_arithmetic_operand_result_type(left, right)
    if left_kind == "integral":
        return right
    if right_kind == "integral":
        return left
    if left_kind == right_kind:
        return left
    if _FLOATING_RANK[left_kind] == _FLOATING_RANK[right_kind]:
        return None
    return left if _FLOATING_RANK[left_kind] > _FLOATING_RANK[right_kind] else right


def _pointer_type(type_text: str) -> str | None:
    """``type_text`` as a pointer value (``device T* const&`` is ``device T*``), or None for a non-pointer."""
    text = re.sub(r"\s*&+\s*$", "", type_text)
    text = re.sub(r"\*(?:\s*\b(?:const|volatile)\b)+\s*$", "*", text).strip()
    return text if text.endswith("*") else None


def _pointer_arithmetic_type(
    preprocessor, operand_types: Sequence[str | None], operator: str
) -> str | None:
    """The type of ``p + n``, ``n + p`` or ``p - n``: the type of the pointer ``p`` ([expr.add]/4).

    An offset added to a pointer has integral or unscoped enumeration type, because a pointer plus a pointer is
    ill-formed, so an untyped ``n`` of ``+`` is still an offset (``self.dst + i * factor()``). ``p - q`` of two
    pointers is a ``ptrdiff_t``, so every operand subtracted from a pointer must be typed as an integral scalar.
    A typed non-integral offset, a second pointer and every other operator are untyped.
    """
    pointers = [
        index
        for index, operand in enumerate(operand_types)
        if operand and _pointer_type(operand) is not None
    ]
    if operator not in {"+", "-"} or len(pointers) != 1:
        return None
    position = pointers[0]
    if operator == "-" and position != 0:
        return None
    for index, operand in enumerate(operand_types):
        if index == position:
            continue
        if not operand:
            if operator == "-":
                return None
            continue
        if _operand(preprocessor, operand) != ("integral", 1):
            return None
    return _pointer_type(operand_types[position])


def arithmetic_result_type(
    preprocessor, operand_types: Sequence[str | None], operator: str
) -> str | None:
    """The type of ``a op b op c ...`` (left to right) for operands of the given types."""
    if any(operand and _pointer_type(operand) for operand in operand_types):
        return _pointer_arithmetic_type(preprocessor, operand_types, operator)
    if not operand_types or any(not operand for operand in operand_types):
        return None
    result = operand_types[0]
    for operand in operand_types[1:]:
        result = _combine(preprocessor, result, operand, operator)
        if result is None:
            return None
    return result
