"""Lossless transport between MLX bfloat16 words and native buffer carriers.

Buffers hold millions of words, so a list's word types and range are checked
in bulk, in C, and the OpenGL carriers' 16-bit halves move through ``array``
slices rather than per-word Python arithmetic.
"""

import sys
from array import array

from crosstl.project.runtime_value_encoding import BFLOAT16_BITS, FLOAT32_BITS
from crosstl.project.runtime_values import exact_types_within

GUARD = [0x3EAB] * 32
_INT = frozenset({int})
# The array type codes of exact 16-bit and 32-bit unsigned words.
_U16 = next(code for code in "HI" if array(code).itemsize == 2)
_U32 = next(code for code in "ILH" if array(code).itemsize == 4)
# The OpenGL carrier's 16-bit halves, low half first, in native order.
_HIGH = slice(1, None, 2) if sys.byteorder == "little" else slice(0, None, 2)
_LOW = slice(0, None, 2) if sys.byteorder == "little" else slice(1, None, 2)


def _words_within(values, limit):
    """Whether ``values`` holds only exact ``int`` words in ``[0, limit]``."""
    return exact_types_within(values, _INT) and (
        not values or (min(values) >= 0 and max(values) <= limit)
    )


def encoding(target):
    if target not in {"metal", "directx", "opengl"}:
        raise ValueError("Unsupported bfloat storage target")
    return {"metal": BFLOAT16_BITS, "directx": None, "opengl": FLOAT32_BITS}[target]


def pack(values, target):
    encoding(target)
    values = list(values)
    if not _words_within(values, 0xFFFF):
        raise ValueError("Bfloat storage requires unsigned 16-bit words")
    if target != "opengl":
        return values
    # Each word moves to the high half of its binary32 carrier.
    halves = array(_U16, bytes(4 * len(values)))
    halves[_HIGH] = array(_U16, values)
    return array(_U32, halves.tobytes()).tolist()


def unpack(values, target):
    """Reject inexact carriers rather than rounding a native result on the host."""
    encoding(target)
    values = list(values)
    if target == "opengl":
        if not _words_within(values, 0xFFFFFFFF):
            raise ValueError(
                "Native bfloat carrier is not an exact bfloat16 representation"
            )
        halves = array(_U16, array(_U32, values).tobytes())
        low = halves[_LOW]
        if low.count(0) != len(low):
            raise ValueError(
                "Native bfloat carrier is not an exact bfloat16 representation"
            )
        return halves[_HIGH].tolist()
    return pack(values, target)
