"""Storage encodings for typed runtime buffers."""

from __future__ import annotations

from typing import Any, Sequence

from crosstl.project.runtime_values import exact_types_within

FLOAT16_BITS = "ieee754-binary16"
FLOAT32_BITS = "ieee754-binary32"
BFLOAT16_BITS = "bfloat16-bits"
_INT_TYPES = frozenset({int})


def storage_words(values: Any, *, width: int = 32) -> list[int]:
    """Flatten exact storage words without conversion through host floats."""
    if width not in (16, 32):
        raise ValueError("Storage word width must be 16 or 32 bits.")
    if (
        type(values) is list
        and exact_types_within(values, _INT_TYPES)
        and (not values or (min(values) >= 0 and max(values) < (1 << width)))
    ):
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


def validate_value_encoding(
    encoding: str | None, dtype: str | None, values: Any = None
) -> None:
    if encoding is None:
        if dtype == "bfloat16":
            raise ValueError("Bfloat16 buffers require explicit bfloat16-bits storage.")
        return
    formats = {
        FLOAT16_BITS: ("float16", 16),
        FLOAT32_BITS: ("float32", 32),
        BFLOAT16_BITS: ("bfloat16", 16),
    }
    if (
        not isinstance(encoding, str)
        or encoding not in formats
        or dtype != formats[encoding][0]
    ):
        raise ValueError(
            "Storage encoding requires matching float16, float32 or bfloat16 buffer storage."
        )
    if values is not None:
        storage_words(values, width=formats[encoding][1])
