"""Evaluate Metal's ``is_scalar``, ``is_vector`` and ``is_arithmetic`` type traits.

``metal_type_traits`` defines all three on ``remove_qualifiers_t<T>``, which removes cv and address-space
qualifiers but not references or pointers:

* ``is_scalar`` is clang's ``__is_scalar``: arithmetic scalars (``bool``, the integer types, ``half``, ``bfloat``
  and ``float``), pointers and enumerations.
* ``is_vector`` holds for ``ext_vector_type`` and ``packed_vector_type`` types (``float4``, ``packed_float3``).
* ``is_arithmetic`` is clang's ``__is_arithmetic`` and is specialized for both vector kinds, so it holds for
  arithmetic scalars and their vectors.

All three are false for references, arrays, matrices and class types. These results were checked against
``xcrun metal -std=metal3.1``. Enumerations and other named types are not decided, because the caller cannot
tell an enumeration (a scalar) from a class (not a scalar); those operands fail closed.
"""

from __future__ import annotations

import re

from crosstl.translator.refusals import MetalSourceRefusal

__all__ = ["CATEGORY_TRAITS", "evaluate_category_trait"]

CATEGORY_TRAITS = frozenset({"is_scalar", "is_vector", "is_arithmetic"})

_QUALIFIERS_RE = re.compile(
    r"\b(?:const|volatile|device|constant|thread|threadgroup|threadgroup_imageblock|ray_data|object_data)\b"
)
_PACKED_VECTOR_RE = re.compile(
    r"(?:(?:::)?metal::)?packed_(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|bfloat|float)[234]"
)
_MATRIX_RE = re.compile(r"(?:(?:::)?metal::)?(?:half|float|bfloat)[234]x[234]")


def _category(preprocessor, operand: str) -> str | None:
    """``scalar``, ``vector``, ``pointer`` or ``other`` for a decided operand, else None."""
    text = preprocessor._normalize_template_argument_text(operand or "")
    if not text:
        return None
    if re.search(r"&\s*$", text) or "[" in text:
        return "other"
    unqualified = preprocessor._normalize_template_argument_text(
        _QUALIFIERS_RE.sub(" ", text)
    )
    if not unqualified:
        return None
    if unqualified.endswith("*"):
        return "pointer"
    if _PACKED_VECTOR_RE.fullmatch(unqualified):
        return "vector"
    if _MATRIX_RE.fullmatch(unqualified) or re.fullmatch(
        r"(?:(?:::)?metal::)?matrix<.+>", unqualified
    ):
        return "other"
    decomposed = preprocessor._scalar_and_width(unqualified)
    if decomposed is None:
        return None
    base, width = decomposed
    if base == "double":
        # Metal has no double type.
        return None
    return "scalar" if width == 1 else "vector"


def evaluate_category_trait(preprocessor, trait: str, operand: str) -> bool | None:
    """The value of ``trait<operand>``, or None when the operand is not decided."""
    category = _category(preprocessor, operand)
    if category is None:
        return None
    if trait == "is_scalar":
        return category in {"scalar", "pointer"}
    if trait == "is_vector":
        return category == "vector"
    if trait == "is_arithmetic":
        return category in {"scalar", "vector"}
    raise MetalSourceRefusal(f"not a category trait: {trait}")
