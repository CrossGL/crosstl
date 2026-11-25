"""Lossless transport between MLX binary16 words and native buffer carriers.

Buffers hold hundreds of thousands of words, so lists are converted through
two tables built on first use: the binary32 carrier of every binary16 word and
its inverse. A list is first checked in bulk to hold only exact ``int`` words
in range; any other list goes through the element-wise functions, which raise
the same errors as before.
"""

from crosstl.project.runtime_value_encoding import FLOAT16_BITS, FLOAT32_BITS
from crosstl.project.runtime_values import exact_types_within

GUARD = [0x3555] * 32
_INT = frozenset({int})
# The binary32 carrier of every binary16 word, and the inverse mapping.
_WIDE = None
_NARROW = None


def _carrier(word):
    sign = (word & 0x8000) << 16
    exponent, fraction = (word >> 10) & 31, word & 1023
    if exponent == 0:
        if not fraction:
            return sign
        high = fraction.bit_length() - 1
        return sign | ((103 + high) << 23) | ((fraction - (1 << high)) << (23 - high))
    return sign | ((255 if exponent == 31 else exponent + 112) << 23) | (fraction << 13)


def _tables():
    global _WIDE, _NARROW
    if _WIDE is None:
        wide = tuple(_carrier(word) for word in range(0x10000))
        _NARROW = {carrier: word for word, carrier in enumerate(wide)}
        _WIDE = wide
    return _WIDE, _NARROW


def _words_within(values, limit):
    """Whether ``values`` holds only exact ``int`` words in ``[0, limit]``."""
    return exact_types_within(values, _INT) and (
        not values or (min(values) >= 0 and max(values) <= limit)
    )


def widen(word):
    if type(word) is not int or not 0 <= word <= 0xFFFF:
        raise ValueError("Half storage requires unsigned 16-bit words")
    return _carrier(word)


def narrow(word):
    """Decode only exact binary16 carriers; never perform numeric rounding."""
    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        raise ValueError("Half carrier requires unsigned 32-bit words")
    sign = (word >> 16) & 0x8000
    exponent, fraction = (word >> 23) & 255, word & 0x7FFFFF
    if exponent == 255:
        value = sign | 0x7C00 | (fraction >> 13)
    elif 113 <= exponent <= 142:
        value = sign | ((exponent - 112) << 10) | (fraction >> 13)
    elif 103 <= exponent <= 112:
        value = sign | ((0x800000 | fraction) >> (126 - exponent))
    else:
        value = sign
    if _carrier(value) != word:
        raise ValueError("Native half carrier is not an exact binary16 representation")
    return value


def widen_all(values):
    """``[widen(word) for word in values]``, by table for valid lists."""
    values = list(values)
    if not _words_within(values, 0xFFFF):
        return [widen(word) for word in values]
    return list(map(_tables()[0].__getitem__, values))


def narrow_all(values):
    """``[narrow(word) for word in values]``, by table for exact carriers."""
    values = list(values)
    if _words_within(values, 0xFFFFFFFF):
        try:
            return list(map(_tables()[1].__getitem__, values))
        except KeyError:
            pass
    # Raises the element-wise error for the first invalid carrier.
    return [narrow(word) for word in values]


def _check_words(values):
    if not _words_within(values, 0xFFFF):
        for word in values:
            widen(word)


def encoding(target):
    if target not in {"metal", "directx", "opengl"}:
        raise ValueError("Unsupported half storage target")
    return FLOAT32_BITS if target == "opengl" else FLOAT16_BITS


def pack(values, target):
    encoding(target)
    if target == "opengl":
        return widen_all(values)
    values = list(values)
    _check_words(values)
    return values


def unpack(values, target):
    encoding(target)
    if target == "opengl":
        return narrow_all(values)
    values = list(values)
    _check_words(values)
    return values
