"""Inputs and oracles of the integer cases.

``*_extended_arithmetic``: GLSL's ``uaddCarry``, ``usubBorrow``, ``umulExtended`` and ``imulExtended`` (GLSL 4.60
section 8.8) and SPIR-V's ``OpIAddCarry``, ``OpISubBorrow``, ``OpUMulExtended`` and ``OpSMulExtended`` (SPIR-V 1.6
section 3.52.13) are exact: the sum or difference modulo 2^32 with its carry or borrow, and the high and low words of
the 64-bit product.

``s_integer_signedness``: SPIR-V takes the signedness of an integer instruction from its opcode, not from its operand
types (SPIR-V 1.6 section 3.3.6), so a logical shift of an ``int``, an unsigned comparison of ``int`` values or a
signed division of ``uint`` values reads the operand bits with the opcode's signedness. Every input word is used both
as a ``uint`` and, reinterpreted, as an ``int``.
"""

from __future__ import annotations

import numpy as np

THREADS = 64
EXTENDED_ARITHMETIC_OUTPUTS = 16
SIGNEDNESS_OUTPUTS = 14
_MASK = 0xFFFFFFFF
_EDGES = [
    0,
    1,
    2,
    3,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFF,
    0xFFFFFFFE,
    0xFFFF,
    0x10000,
    0xFFFF0000,
    0x12345678,
    0x9ABCDEF0,
    0xFFFFFFF0,
    0x0000FFFF,
]


def integer_pair_inputs():
    """``x = a[i]`` and ``y = a[i + 64]``: edge words against edge words, then random words."""
    rng = np.random.default_rng(20261020)
    first = rng.integers(0, 2**32, THREADS, dtype=np.uint64)
    second = rng.integers(0, 2**32, THREADS, dtype=np.uint64)
    count = len(_EDGES)
    for thread in range(3 * count):
        first[thread] = _EDGES[thread % count]
        second[thread] = _EDGES[(thread * 7 + thread // count) % count]
    return np.concatenate([first, second]).astype(np.uint32)


def _pairs(inputs):
    words = [int(word) for word in inputs]
    return list(zip(words[:THREADS], words[THREADS:]))


def _signed(word, bits=32):
    word &= (1 << bits) - 1
    return word - (1 << bits) if word >> (bits - 1) else word


def _add(x, y):
    total = x + y
    return [total & _MASK, total >> 32]


def _sub(x, y):
    return [(x - y) & _MASK, int(x < y)]


def _umul(x, y):
    product = x * y
    return [product >> 32, product & _MASK]


def _imul(x, y):
    product = (_signed(x) * _signed(y)) & 0xFFFFFFFFFFFFFFFF
    return [product >> 32, product & _MASK]


def extended_arithmetic_reference(inputs):
    """Sixteen words per thread, in the order the ``g_`` and ``s_`` sources store them."""
    out = []
    for x, y in _pairs(inputs):
        # The vector forms: uaddCarry(uvec2(x, y), uvec2(y, x ^ y)) and imulExtended(ivec2(x, y), ivec2(y, x)).
        (sum0, carry0), (sum1, carry1) = _add(x, y), _add(y, x ^ y)
        (high0, low0), (high1, low1) = _imul(x, y), _imul(y, x)
        out += _add(x, y) + _sub(x, y) + _umul(x, y) + _imul(x, y)
        out += [sum0, sum1, carry0, carry1, high0, high1, low0, low1]
    return np.array(out, np.uint32)


def _truncated_quotient(x, y):
    quotient = abs(x) // abs(y)
    return quotient if (x < 0) == (y < 0) else -quotient


def signedness_reference(inputs):
    """Fourteen words per thread, in the order ``s_integer_signedness`` stores them."""
    out = []
    for x, y in _pairs(inputs):
        sx, sy = _signed(x), _signed(y)
        count = y & 31
        divisor = _signed((y & 0xFFFFFFF0) | 5)
        quotient = _truncated_quotient(sx, divisor)
        remainder = sx - divisor * quotient
        modulus = (
            remainder + divisor
            if remainder and (remainder < 0) != (divisor < 0)
            else remainder
        )
        out += [
            x >> count,
            (sx >> count) & _MASK,
            int(x < y),
            int(sx < sy),
            int(x >= 3),
            remainder & _MASK,
            modulus & _MASK,
            quotient & _MASK,
            int(np.float32(sx >> 8).view(np.uint32)),
            int(np.float32(x >> 8).view(np.uint32)),
            (sx >> 8) & _MASK,
            (x + y) & _MASK,
            int(x < y),
            int(y < x),
        ]
    return np.array(out, np.uint32)


def _clz(word, bits=32):
    word &= (1 << bits) - 1
    return bits - word.bit_length()


def _ctz(word, bits=32):
    word &= (1 << bits) - 1
    return bits if word == 0 else (word & -word).bit_length() - 1


def _find_msb(word, signed=False):
    """GLSL findMSB: the highest set bit, or for a negative int the highest clear bit; -1 when there is none."""
    word &= _MASK
    if signed and word & 0x80000000:
        word = ~word & _MASK
    return word.bit_length() - 1


def _find_lsb(word):
    word &= _MASK
    return -1 if word == 0 else _ctz(word)


def bit_scan_inputs():
    """One word per thread: the first half of ``integer_pair_inputs``, edge words first."""
    return integer_pair_inputs()[:THREADS]


def metal_bit_scan_reference(inputs):
    """Eight words per thread, in the order ``m_bit_scans`` stores them."""
    out = []
    for x in (int(word) for word in inputs):
        s = _signed(x)
        out += [_clz(x), _ctz(x), _clz(x), _ctz(x), _clz(x), _clz(~x)]
        out += [_ctz(~s), _ctz(x >> 16)]
    return np.array(out, np.uint32)


def glsl_bit_scan_reference(inputs):
    """Eight words per thread, in the order ``g_bit_scans`` stores them; -1 is stored as its two's complement."""
    out = []
    for x in (int(word) for word in inputs):
        _signed(x)
        out += [_find_msb(x), _find_lsb(x), _find_msb(x, True), _find_lsb(x)]
        out += [
            _find_msb(x, True),
            _find_msb(~x, True),
            _find_lsb(~x),
            _find_lsb(x >> 16),
        ]
    return np.array([value & _MASK for value in out], np.uint32)
