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
_PACK_FORMATS = {
    "bool": "?",
    **{
        f"int{width}": code
        for width, code in ((8, "b"), (16, "h"), (32, "i"), (64, "q"))
    },
    **{
        f"uint{width}": code
        for width, code in ((8, "B"), (16, "H"), (32, "I"), (64, "Q"))
    },
    **_FLOAT_FORMATS,
}


def exact_types_within(values: Iterable[Any], allowed: frozenset[type]) -> bool:
    """Whether the exact type of every item in ``values`` is in ``allowed``.

    The type lookup runs in C over the whole sequence, which is several times
    faster than an ``all`` over a generator for lists of millions of scalars.
    Subclasses do not match: ``bool`` is not accepted where only ``int`` is.
    """
    return allowed.issuperset(map(type, values))


def scalar_types(values: Iterable[Any]) -> frozenset[type]:
    """The exact types of the items of ``values``, found in one pass in C.

    The helpers below take these as ``types`` so that a caller that checks one
    list against several type sets scans it once.
    """
    return frozenset(map(type, values))


def _within(
    values: Iterable[Any], allowed: frozenset[type], types: frozenset[type] | None
) -> bool:
    return (
        types <= allowed if types is not None else exact_types_within(values, allowed)
    )


def flatten_values(
    value: Any, *, keep_none: bool = False, types: frozenset[type] | None = None
) -> list[Any]:
    """Flatten nested sequences of runtime values in order.

    Strings and byte strings are leaves. ``None`` flattens to nothing, or to
    ``[None]`` when ``keep_none`` is set. ``types``, when given, are the exact
    types of the items of the list ``value`` (``scalar_types``).
    """
    if value is None:
        return [None] if keep_none else []
    if type(value) is list and _within(value, _LEAF_TYPES, types):
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


def flatten_scalars(
    value: Any, *, types: frozenset[type] | None = None
) -> tuple[list[Any], frozenset[type]]:
    """``flatten_values(value)`` and the exact types of its items.

    A flat list of leaves is scanned once for both; ``types``, when given, are
    the exact types of the items of the list ``value``.
    """
    if type(value) is list:
        if types is None:
            types = scalar_types(value)
        if types <= _LEAF_TYPES:
            return list(value), types
    flat = flatten_values(value)
    return flat, scalar_types(flat)


def copy_values(values: Any, *, types: frozenset[type] | None = None) -> Any:
    """``copy.deepcopy(values)``, as one list copy for a list of atomic scalars.

    ``types``, when given, are the exact types of the items of the list
    ``values``.
    """
    if type(values) is list and _within(values, _ATOMIC_TYPES, types):
        return list(values)
    return copy.deepcopy(values)


def scalars_fit(
    values: Sequence[Any], dtype: str, *, types: frozenset[type] | None = None
) -> bool:
    """Whether every value is an exact built-in scalar that ``dtype`` stores.

    ``bool`` buffers take ``bool`` values, integer buffers in-range ``int``
    values, and ``float16``/``float32`` buffers finite ``int`` or ``float``
    values that pack at that width. ``False`` decides nothing: the caller's
    element-wise check must then judge the values. ``types``, when given, are
    the exact types of the items of ``values`` (``scalar_types``).
    """
    return _checked_storage(values, dtype, types, pack=False) is not None


def pack_scalars(
    values: Sequence[Any], dtype: str, *, types: frozenset[type] | None = None
) -> bytes | None:
    """The little-endian ``dtype`` storage of ``values`` when ``scalars_fit``.

    The bytes are those of packing each value with its ``struct`` code in
    order. ``None`` decides nothing, as a ``False`` from ``scalars_fit`` does.
    """
    return _checked_storage(values, dtype, types, pack=True)


def _checked_storage(
    values: Sequence[Any], dtype: str, types: frozenset[type] | None, *, pack: bool
) -> bytes | None:
    if dtype == "bool":
        if not _within(values, _BOOL_TYPES, types):
            return None
    elif dtype in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[dtype]
        if not _within(values, _INT_TYPES, types) or (
            values and not (low <= min(values) and max(values) <= high)
        ):
            return None
    elif dtype in _FLOAT_FORMATS:
        if not _within(values, _NUMERIC_TYPES, types):
            return None
        try:
            # A NaN or infinity makes the sum NaN or infinite, and values that
            # pack at 16 or 32 bits cannot sum past the float64 range, so a
            # finite sum decides finiteness as ``all(map(math.isfinite, ...))``
            # does for every list that packs below.
            if not math.isfinite(sum(values)):
                return None
            # Packing is the range check, so its bytes come free.
            return struct.pack(f"<{len(values)}{_FLOAT_FORMATS[dtype]}", *values)
        except (OverflowError, struct.error, ValueError):
            return None
    else:
        return None
    if not pack:
        return b""
    return struct.pack(f"<{len(values)}{_PACK_FORMATS[dtype]}", *values)
