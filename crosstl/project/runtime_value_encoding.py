"""Storage encodings for typed runtime buffers."""

from __future__ import annotations

from typing import Any, Sequence

FLOAT32_BITS = "ieee754-binary32"


def storage_words(values: Any) -> list[int]:
    """Flatten exact binary32 words without conversion through host floats."""
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return [word for value in values for word in storage_words(value)]
    if (
        not isinstance(values, int)
        or isinstance(values, bool)
        or not 0 <= values <= 0xFFFFFFFF
    ):
        raise ValueError("Binary32 storage values must be unsigned 32-bit integers.")
    return [values]


def validate_value_encoding(
    encoding: str | None, dtype: str | None, values: Any = None
) -> None:
    if encoding is None:
        return
    if encoding != FLOAT32_BITS or dtype != "float32":
        raise ValueError(
            "The ieee754-binary32 encoding requires float32 buffer storage."
        )
    if values is not None:
        storage_words(values)
