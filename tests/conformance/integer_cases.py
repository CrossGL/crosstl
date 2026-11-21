"""Inputs and oracles of the integer cases.

``*_extended_arithmetic``: GLSL's ``uaddCarry``, ``usubBorrow``, ``umulExtended`` and ``imulExtended`` (GLSL 4.60
section 8.8) and SPIR-V's ``OpIAddCarry``, ``OpISubBorrow``, ``OpUMulExtended`` and ``OpSMulExtended`` (SPIR-V 1.6
section 3.52.13) are exact: the sum or difference modulo 2^32 with its carry or borrow, and the high and low words of
the 64-bit product.

``m_narrow_bit_functions``: MSL ``popcount`` and ``reverse_bits`` (MSL 3.2 section 6.13) of ``short``, ``ushort``,
``char`` and ``uchar`` scalars and vectors count and reverse the bits of the value's own width, whatever register
width a target computes them in.

``x_integer_combinatorics``: CrossGL's ``gcd``, ``lcm``, ``factorial``, ``comb`` and ``perm``
(``crosstl/translator/semantics/integer_math.py``) are exact where their results fit the type: Python's ``math``
functions of the same names, with ``comb`` and ``perm`` 0 for a negative ``k``.

``s_integer_signedness``: SPIR-V takes the signedness of an integer instruction from its opcode, not from its operand
types (SPIR-V 1.6 section 3.3.6), so a logical shift of an ``int``, an unsigned comparison of ``int`` values or a
signed division of ``uint`` values reads the operand bits with the opcode's signedness. Every input word is used both
as a ``uint`` and, reinterpreted, as an ``int``.
"""

from __future__ import annotations

import math

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


def _reverse(word, bits):
    word &= (1 << bits) - 1
    return int(format(word, f"0{bits}b")[::-1], 2)


def _popcount(word, bits):
    return bin(word & ((1 << bits) - 1)).count("1")


def narrow_bit_function_reference(inputs):
    """Eight words per thread, in the order ``m_narrow_bit_functions`` stores them."""
    out = []
    for x in (int(word) for word in inputs):
        out += [_popcount(x, 16), _reverse(x, 16), _reverse(x >> 16, 16)]
        out += [_reverse(x >> 4, 16), _popcount(x >> 8, 8)]
        out += [_reverse(x, 8) | (_reverse(x >> 24, 8) << 8)]
        out += [_popcount(x >> 4, 16), _popcount(x >> 16, 8)]
    return np.array(out, np.uint32)


# Edge words of x_integer_combinatorics: zeros, ones, signs, the extremes, shared factors and large primes. The most
# negative word is paired only with operands whose gcd with it fits ``int``.
_COMBINATORICS_EDGES = [
    (0, 0),
    (0, 7),
    (7, 0),
    (1, 1),
    (-1, 1),
    (12, 18),
    (-12, 18),
    (12, -18),
    (-12, -18),
    (0x7FFFFFFF, 0x7FFFFFFF),
    (0x7FFFFFFF, -0x7FFFFFFF),
    (-(2**31), 6),
    (-(2**31), -(2**30)),
    (2147483629, 2147483587),
    (1 << 30, 3 << 28),
    (0x7FFFFFF0, 21),
    (63, 31),
    (62, 33),
    (20, 20),
    (15, 15),
    (31, 17),
    (5, 9),
]


def combinatorics_inputs():
    """``a = inp[i]`` and ``b = inp[i + 64]``: the edge pairs, then random words."""
    rng = np.random.default_rng(20261121)
    first = rng.integers(-(2**31) + 1, 2**31, THREADS, dtype=np.int64)
    second = rng.integers(-(2**31) + 1, 2**31, THREADS, dtype=np.int64)
    for thread, (a, b) in enumerate(_COMBINATORICS_EDGES):
        first[thread], second[thread] = a, b
    return np.concatenate([first, second]).astype(np.int32)


def _comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def _perm(n, k):
    return math.perm(n, k) if 0 <= k <= n else 0


def combinatorics_reference(inputs):
    """Sixteen words per thread, in the order ``x_integer_combinatorics`` stores them."""
    words = [int(word) for word in inputs]
    out = []
    for a, b in zip(words[:THREADS], words[THREADS:]):
        ua, ub = a & _MASK, b & _MASK
        wide = math.lcm(a, b)
        product = math.factorial(ua % 21)
        binomial = _comb(ua & 63, ub & 63)
        arranged = _perm(ua & 15, ub & 15)
        out += [math.gcd(a, b), math.gcd(ua, ub), math.lcm(a >> 20, b >> 20)]
        out += [wide, wide >> 32, product, product >> 32, binomial, binomial >> 32]
        out += [arranged, arranged >> 32, math.factorial(ua & 7)]
        out += [_comb(ua & 31, (ub & 31) - 2), math.gcd(a, 12), math.gcd(b, 18)]
        out += [math.gcd(_signed16(a >> 17), _signed16(b >> 17))]
    return np.array([value & _MASK for value in out], np.uint32)


def _signed16(value):
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value
