"""Result types of built-in arithmetic between Metal scalar and vector types of different types.

``MetalPreprocessor._infer_binary_arithmetic_type`` types ``a op b`` for overload resolution. Equal operand
types and integral scalars already have rules. This module decides the remaining built-in mixes, following the
behaviour of the Metal compiler (MSL 3.1; checked with ``xcrun metal`` on non-constant operands and literals):

* scalar with scalar: the usual arithmetic conversions ([expr.arith.conv]/1). A floating operand converts an
  integral one to its type, and ``float`` outranks ``half`` and ``bfloat``. ``half`` and ``bfloat`` have
  unordered ranks, so mixing them is ill-formed.
* vector with scalar: the scalar converts to the vector's element type and the result has the vector type
  (``half4 * float`` is ``half4``, ``int4 * long`` is ``int4``). An integral vector with a floating scalar, a
  ``half`` vector with a ``bfloat`` scalar, and a ``bfloat`` vector with a ``half`` or ``float`` scalar are
  ill-formed.
* vector with vector of a different type: ill-formed, because vectors have no implicit conversions.

Literals follow the same rules, typed as the importer types them (an unsuffixed floating literal is ``float``).
``bool`` vectors, ``double``, pointers, matrices and aggregates are left to the existing rules.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["builtin_arithmetic_conversion"]

_FLOATING = {
    "half": "half",
    "float16_t": "half",
    "bfloat16_t": "bfloat",
    "float": "float",
}
_FLOATING_RANK = {"half": 1, "bfloat": 1, "float": 2}

# (decided, result type). A decided ``None`` is an ill-formed expression, so no
# other rule may type it.
Decision = Tuple[bool, Optional[str]]
_UNDECIDED: Decision = (False, None)
_ILL_FORMED: Decision = (True, None)


def _operand(preprocessor, type_text: str):
    if not type_text or "*" in type_text:
        return None
    decomposed = preprocessor._scalar_and_width(type_text)
    if decomposed is None:
        return None
    base, width = decomposed
    if base in _FLOATING:
        kind = _FLOATING[base]
    elif base in preprocessor._METAL_INTEGRAL_SCALAR_TYPES:
        kind = "integral"
    else:
        return None
    normalized = preprocessor._normalize_inferred_type(type_text)
    if not normalized:
        return None
    return kind, width, normalized


def _vector_with_scalar(vector_kind: str, scalar_kind: str) -> bool:
    """Whether a scalar of ``scalar_kind`` converts to elements of ``vector_kind``."""
    if vector_kind == "integral":
        return scalar_kind == "integral"
    if vector_kind == "float":
        return True
    if vector_kind == "half":
        return scalar_kind != "bfloat"
    # bfloat elements accept integral and bfloat scalars only.
    return scalar_kind in {"integral", "bfloat"}


def builtin_arithmetic_conversion(
    preprocessor, left_type: str, right_type: str, operator: str
) -> Decision:
    """Decide the result type of ``left op right`` for two built-in arithmetic types."""
    left = _operand(preprocessor, left_type)
    right = _operand(preprocessor, right_type)
    if left is None or right is None:
        return _UNDECIDED
    left_kind, left_width, left_text = left
    right_kind, right_width, right_text = right
    if (left_width != 1 and left_text.startswith("bool")) or (
        right_width != 1 and right_text.startswith("bool")
    ):
        return _UNDECIDED
    if operator == "%" and (left_kind != "integral" or right_kind != "integral"):
        # `%` requires integral operands ([expr.mul]/2); Metal has fmod for floats.
        return _ILL_FORMED

    if left_width != 1 and right_width != 1:
        if (left_kind, left_width, left_text) == (right_kind, right_width, right_text):
            return True, left_text
        return _ILL_FORMED
    if left_width != 1 or right_width != 1:
        if left_width != 1:
            vector_kind, vector_text, scalar_kind = left_kind, left_text, right_kind
        else:
            vector_kind, vector_text, scalar_kind = right_kind, right_text, left_kind
        if _vector_with_scalar(vector_kind, scalar_kind):
            return True, vector_text
        return _ILL_FORMED

    if left_kind == "integral" and right_kind == "integral":
        # Integral promotions and conversions have their own rule.
        return _UNDECIDED
    if left_kind == "integral":
        return True, right_text
    if right_kind == "integral":
        return True, left_text
    if left_kind == right_kind:
        return True, left_text
    if _FLOATING_RANK[left_kind] == _FLOATING_RANK[right_kind]:
        return _ILL_FORMED
    return True, (
        left_text
        if _FLOATING_RANK[left_kind] > _FLOATING_RANK[right_kind]
        else right_text
    )
