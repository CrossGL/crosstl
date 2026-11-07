"""Storage encodings for typed runtime buffers."""

from __future__ import annotations

from typing import Any, Sequence

from crosstl.project.runtime_values import exact_types_within, integer_storage

FLOAT16_BITS = "ieee754-binary16"
FLOAT32_BITS = "ieee754-binary32"
BFLOAT16_BITS = "bfloat16-bits"
_INT_TYPES = frozenset({int})
_FORMATS = {
    FLOAT16_BITS: ("float16", 16),
    FLOAT32_BITS: ("float32", 32),
    BFLOAT16_BITS: ("bfloat16", 16),
}


def _flat_words(values: Any, width: int, types: frozenset[type] | None) -> bytes | None:
    """The packed words of a flat list of in-range exact ``int`` words, or ``None``.

    ``types``, when given, are the exact types of the items of the list
    ``values``; one C-level ``array`` pass decides the range.
    """
    if type(values) is not list or not (
        types <= _INT_TYPES
        if types is not None
        else exact_types_within(values, _INT_TYPES)
    ):
        return None
    return integer_storage(values, f"uint{width}")


def storage_words(
    values: Any, *, width: int = 32, types: frozenset[type] | None = None
) -> list[int]:
    """Flatten exact storage words without conversion through host floats.

    ``types``, when given, are the exact types of the items of the list
    ``values`` (``scalar_types``).
    """
    if width not in (16, 32):
        raise ValueError("Storage word width must be 16 or 32 bits.")
    if _flat_words(values, width, types) is not None:
        # A flat list of in-range words is its own flattening.
        return list(values)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return [word for value in values for word in storage_words(value, width=width)]
    if (
        not isinstance(values, int)
        or isinstance(values, bool)
        or not 0 <= values < (1 << width)
    ):
        raise ValueError(
            f"Binary{width} storage values must be unsigned {width}-bit integers."
        )
    return [values]


def storage_bytes(
    values: Any, *, width: int = 32, types: frozenset[type] | None = None
) -> bytes:
    """``storage_words(values)`` packed as little-endian ``width``-bit words.

    It accepts and refuses exactly what ``storage_words`` does, with the same
    errors; a flat list of in-range words is checked and packed in one pass.
    """
    if width not in (16, 32):
        raise ValueError("Storage word width must be 16 or 32 bits.")
    packed = _flat_words(values, width, types)
    if packed is None:
        packed = integer_storage(storage_words(values, width=width), f"uint{width}")
    return packed


def _encoding_width(encoding: Any, dtype: str | None) -> int:
    if (
        not isinstance(encoding, str)
        or encoding not in _FORMATS
        or dtype != _FORMATS[encoding][0]
    ):
        raise ValueError(
            "Storage encoding requires matching float16, float32 or bfloat16 buffer storage."
        )
    return _FORMATS[encoding][1]


def encoded_storage(
    encoding: str,
    dtype: str | None,
    values: Any,
    *,
    types: frozenset[type] | None = None,
) -> bytes:
    """Check ``values`` as ``encoding`` storage and pack them little-endian.

    It raises exactly the errors ``validate_value_encoding`` raises for the
    same arguments. ``types``, when given, are the exact types of the items of
    the list ``values``.
    """
    return storage_bytes(values, width=_encoding_width(encoding, dtype), types=types)


def validate_value_encoding(
    encoding: str | None,
    dtype: str | None,
    values: Any = None,
    *,
    types: frozenset[type] | None = None,
) -> None:
    """Refuse an encoding that does not match ``dtype``, or words it cannot hold.

    ``types``, when given, are the exact types of the items of the list
    ``values`` (``scalar_types``), so a caller that already scanned them does
    not scan them again.
    """
    if encoding is None:
        if dtype == "bfloat16":
            raise ValueError("Bfloat16 buffers require explicit bfloat16-bits storage.")
        return
    width = _encoding_width(encoding, dtype)
    if values is not None:
        storage_bytes(values, width=width, types=types)
