"""Packed little-endian element storage as a typed runtime buffer value.

A buffer's ``values`` may be one ``bytes`` object instead of a list: the
buffer's elements packed little-endian at the dtype's width, in order, exactly
as a native runtime uploads them. Packed values skip the per-element type
checks, copies and conversions of a value list, which dominate the cost of
uploading and reading back buffers of millions of elements.

Every bit pattern of an integer element is a value, and Boolean elements are
canonical zero or one bytes. Floating-point elements must name their bit
``encoding``, so the bytes are their storage words and no NaN payload, signed
zero or subnormal is rewritten by a conversion through host floats. A native
readback of a binding whose upload or expected output is packed is packed as
well, so a caller that hands over bytes receives bytes.
"""

from __future__ import annotations

import base64
import binascii
import struct
from typing import Any

from crosstl.project.runtime_value_encoding import validate_value_encoding

ELEMENT_SIZES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
    "int64": 8,
    "uint64": 8,
}
_FLOATING = frozenset({"float16", "float32", "bfloat16"})
# The ``struct`` code of one element, or of one encoded storage word.
_ELEMENT_CODES = {
    "bool": "?",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "float16": "e",
    "bfloat16": "H",
    "int32": "i",
    "uint32": "I",
    "float32": "f",
    "int64": "q",
    "uint64": "Q",
}
_WORD_CODES = {"float16": "H", "bfloat16": "H", "float32": "I"}


class PackedStorageError(ValueError):
    """Packed storage that does not hold its declared elements.

    ``reason`` is ``value-encoding-invalid``, ``value-size-mismatch`` or
    ``value-data-invalid``, the reasons a value list is refused for.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def is_packed(values: Any) -> bool:
    """Whether ``values`` is packed storage rather than a value list."""
    return type(values) is bytes


def check_packed_storage(
    data: bytes, dtype: str | None, encoding: str | None, count: int
) -> None:
    """Refuse ``data`` unless it is ``count`` packed ``dtype`` elements.

    Raises ``PackedStorageError``. The encoding must match the dtype as it
    must for a value list, and floating-point storage must name one.
    """
    if not is_packed(data):
        raise PackedStorageError(
            "value-data-invalid", "Packed storage must be a bytes object."
        )
    size = ELEMENT_SIZES.get(dtype) if isinstance(dtype, str) else None
    if size is None:
        raise PackedStorageError(
            "value-data-invalid",
            f"Packed storage does not support {dtype!r} elements.",
        )
    try:
        validate_value_encoding(encoding, dtype)
    except ValueError as exc:
        raise PackedStorageError("value-encoding-invalid", str(exc)) from exc
    if dtype in _FLOATING and encoding is None:
        raise PackedStorageError(
            "value-encoding-invalid",
            "Packed floating-point storage requires its bit encoding.",
        )
    if type(count) is not int or count < 0 or len(data) != count * size:
        raise PackedStorageError(
            "value-size-mismatch",
            f"Packed storage holds {len(data)} bytes, not {count} {dtype} elements.",
        )
    if dtype == "bool" and data.translate(None, b"\x00\x01"):
        raise PackedStorageError(
            "value-data-invalid",
            "Packed boolean storage must contain canonical zero or one bytes.",
        )


def unpack_storage(data: bytes, dtype: str, encoding: str | None) -> list[Any]:
    """The value list whose upload is the packed ``data``.

    Encoded floating-point storage yields its unsigned storage words, other
    storage its elements, as a native readback of the same bytes as a list.
    """
    codes = _WORD_CODES if encoding is not None else _ELEMENT_CODES
    code = codes.get(dtype) if isinstance(dtype, str) else None
    if code is None:
        raise PackedStorageError(
            "value-data-invalid",
            f"Packed storage does not support {dtype!r} elements.",
        )
    size = struct.calcsize(code)
    if len(data) % size:
        raise PackedStorageError(
            "value-size-mismatch",
            f"Packed storage of {len(data)} bytes is not whole {dtype} elements.",
        )
    return list(struct.unpack(f"<{len(data) // size}{code}", data))


def packed_json(data: bytes) -> str:
    """The JSON form of packed storage: its standard base64 text."""
    return base64.b64encode(data).decode("ascii")


def parse_packed_json(text: Any) -> bytes:
    """Packed storage from its strict base64 JSON form."""
    if not isinstance(text, str):
        raise PackedStorageError(
            "value-data-invalid", "Packed values must be base64 text."
        )
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise PackedStorageError(
            "value-data-invalid", "Packed values must be strict base64 text."
        ) from exc
