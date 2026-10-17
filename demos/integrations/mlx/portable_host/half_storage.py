"""Lossless transport between MLX binary16 words and native buffer carriers."""

from crosstl.project.runtime_value_encoding import FLOAT16_BITS, FLOAT32_BITS

GUARD = [0x3555] * 32


def widen(word):
    if type(word) is not int or not 0 <= word <= 0xFFFF:
        raise ValueError("Half storage requires unsigned 16-bit words")
    sign = (word & 0x8000) << 16
    exponent, fraction = (word >> 10) & 31, word & 1023
    if exponent == 0:
        if not fraction:
            return sign
        high = fraction.bit_length() - 1
        return sign | ((103 + high) << 23) | ((fraction - (1 << high)) << (23 - high))
    return sign | ((255 if exponent == 31 else exponent + 112) << 23) | (fraction << 13)


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
    if widen(value) != word:
        raise ValueError("Native half carrier is not an exact binary16 representation")
    return value


def encoding(target):
    if target not in {"metal", "directx", "opengl"}:
        raise ValueError("Unsupported half storage target")
    return FLOAT32_BITS if target == "opengl" else FLOAT16_BITS


def pack(values, target):
    encoding(target)
    values = list(values)
    for word in values:
        widen(word)
    return [widen(word) for word in values] if target == "opengl" else list(values)


def unpack(values, target):
    encoding(target)
    values = list(values)
    if target == "opengl":
        return [narrow(word) for word in values]
    for word in values:
        widen(word)
    return list(values)
