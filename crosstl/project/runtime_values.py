"""Bulk handling of flat runtime value lists.

Native dispatch requests carry buffer contents as Python lists of scalars, often
hundreds of thousands long. The general code walks every element through
``collections.abc`` checks, ``copy.deepcopy`` and one ``struct`` call each.
These helpers give the same results for lists of exact built-in scalars in
bulk, and leave every other value to the element-wise code, so the general
paths still decide every case they reject and report the same errors.
"""

from __future__ import annotations

import copy
import math
import struct
from collections.abc import Iterable, Sequence
from typing import Any

# Exact types whose element-wise flattening is ``[item]``.
_LEAF_TYPES = frozenset({int, float, bool, str, bytes, bytearray})
# Exact types ``copy.deepcopy`` returns unchanged.
_ATOMIC_TYPES = frozenset({int, float, bool, str, bytes, type(None)})
_NUMERIC_TYPES = frozenset({int, float})
_INT_TYPES = frozenset({int})
_BOOL_TYPES = frozenset({bool})
_FLOAT_FORMATS = {"float16": "e", "float32": "f"}
_INTEGER_RANGES = {
    **{
        f"int{width}": (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        for width in (8, 16, 32, 64)
    },
    **{f"uint{width}": (0, (1 << width) - 1) for width in (8, 16, 32, 64)},
}


def exact_types_within(values: Iterable[Any], allowed: frozenset[type]) -> bool:
    """Whether the exact type of every item in ``values`` is in ``allowed``.

    The type lookup runs in C over the whole sequence, which is several times
    faster than an ``all`` over a generator for lists of millions of scalars.
    Subclasses do not match: ``bool`` is not accepted where only ``int`` is.
    """
    return allowed.issuperset(map(type, values))


def flatten_values(value: Any, *, keep_none: bool = False) -> list[Any]:
    """Flatten nested sequences of runtime values in order.

    Strings and byte strings are leaves. ``None`` flattens to nothing, or to
    ``[None]`` when ``keep_none`` is set.
    """
    if value is None:
        return [None] if keep_none else []
    if type(value) is list and exact_types_within(value, _LEAF_TYPES):
        # A flat list of leaves is its own flattening.
        return list(value)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return [value]
    result: list[Any] = []
    append = result.append
    for item in value:
        if type(item) in _LEAF_TYPES:
            append(item)
        else:
            result.extend(flatten_values(item, keep_none=keep_none))
    return result


def copy_values(values: Any) -> Any:
    """``copy.deepcopy(values)``, as one list copy for a list of atomic scalars."""
    if type(values) is list and exact_types_within(values, _ATOMIC_TYPES):
        return list(values)
    return copy.deepcopy(values)


def scalars_fit(values: Sequence[Any], dtype: str) -> bool:
    """Whether every value is an exact built-in scalar that ``dtype`` stores.

    ``bool`` buffers take ``bool`` values, integer buffers in-range ``int``
    values, and ``float16``/``float32`` buffers finite ``int`` or ``float``
    values that pack at that width. ``False`` decides nothing: the caller's
    element-wise check must then judge the values.
    """
    if dtype == "bool":
        return exact_types_within(values, _BOOL_TYPES)
    if dtype in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[dtype]
        return exact_types_within(values, _INT_TYPES) and (
            not values or (low <= min(values) and max(values) <= high)
        )
    if dtype in _FLOAT_FORMATS:
        if not exact_types_within(values, _NUMERIC_TYPES):
            return False
        try:
            if not all(map(math.isfinite, values)):
                return False
            struct.pack(f"<{len(values)}{_FLOAT_FORMATS[dtype]}", *values)
        except (OverflowError, struct.error, ValueError):
            return False
        return True
    return False
