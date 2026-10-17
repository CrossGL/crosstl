"""Storage encodings for typed runtime buffers."""

from __future__ import annotations

from typing import Any, Sequence

FLOAT16_BITS = "ieee754-binary16"
FLOAT32_BITS = "ieee754-binary32"
BFLOAT16_BITS = "bfloat16-bits"


def storage_words(values: Any, *, width: int = 32) -> list[int]:
    """Flatten exact storage words without conversion through host floats."""
    if width not in (16, 32):
        raise ValueError("Storage word width must be 16 or 32 bits.")
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
