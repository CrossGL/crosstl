"""Inputs and oracles of the portable math cases: CrossGL ``sinpi``, ``cospi``, ``fdim``, ``nextafter``, ``ilogb``
and ``copysign`` on binary32, as C17 section 7.12 and IEEE 754-2019 section 9.2 define them.

``crosstl/translator/semantics/portable_math.py`` computes them with CrossGL helpers on targets that lack them. The
oracles here share nothing with those helpers: the pi functions reduce the operand exactly in binary64 and evaluate
the binary64 ``sin`` and ``cos``, the others use numpy's binary32 ``nextafter``, ``copysign`` and ``frexp``.
"""

from __future__ import annotations

import math

import numpy as np

NAN, INF = float("nan"), float("inf")
INT_MIN, INT_MAX = -(2**31), 2**31 - 1
_F32 = np.float32


def _up(value):
    return float(np.nextafter(_F32(value), _F32(INF)))


def _down(value):
    return float(np.nextafter(_F32(value), _F32(-INF)))


def pi_inputs(n=64, seed=29):
    """Zeros, integers, half-integers, their neighbours, small and huge values, the 2^22..2^24 boundaries.

    No operand or result is subnormal, so a target that flushes subnormals computes the same values.
    """
    special = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        3.0,
        -7.0,
        0.5,
        -0.5,
        1.5,
        -2.5,
        3.5,
        0.25,
        -0.75,
        _up(1.0),
        _down(2.0),
        _up(0.5),
        _down(-0.5),
        -(2.0**-12),
        1e-6,
        2.0**22 + 0.5,
        -(2.0**22) - 1.5,
        2.0**23,
        2.0**23 + 1.0,
        -(2.0**23) - 3.0,
        2.0**24,
        1e10,
        3.4028234663852886e38,
        INF,
        -INF,
        NAN,
    ]
    fixed = np.array(special, np.float32)
    rng = np.random.default_rng(seed)
    count = n - fixed.size
    spread = rng.uniform(-8.0, 8.0, count).astype(np.float32)
    return np.concatenate([fixed, spread])[:n]


def _pi_function(x, sine):
    if math.isnan(x) or math.isinf(x):
        return NAN
    nearest = round(
        2.0 * x
    )  # exact: x is a binary32 value, 2x and the halves are exact in binary64
    t = x - nearest / 2.0
    quadrant = int(nearest) % 4
    if sine:
        if t == 0.0 and quadrant % 2 == 0:
            return math.copysign(0.0, x)
        value = math.sin(math.pi * t) if quadrant % 2 == 0 else math.cos(math.pi * t)
        return value if quadrant < 2 else -value
    if t == 0.0 and quadrant % 2 == 1:
        return 0.0
    value = math.cos(math.pi * t) if quadrant % 2 == 0 else math.sin(math.pi * t)
    return value if quadrant in (0, 3) else -value


def pi_reference(inputs):
    """Per operand: sinpi(x), sinpi(-x), cospi(x), cospi(-x), each rounded once to binary32."""
    out = []
    for x in np.asarray(inputs, np.float32).astype(np.float64):
        x = float(x)
        out.extend(
            [
                _pi_function(x, True),
                _pi_function(-x, True),
                _pi_function(x, False),
                _pi_function(-x, False),
            ]
        )
    return np.array(out, np.float64).astype(np.float32)


def pair_inputs(n=64, seed=31):
    """Interleaved (x, y) operand pairs: NaN, both zeros, infinities, subnormals, the largest finite value."""
    tiny = 1.401298464324817e-45
    largest = 3.4028234663852886e38
    special = [
        (NAN, 1.0),
        (1.0, NAN),
        (0.0, -0.0),
        (-0.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (-0.0, 1.0),
        (tiny, -1.0),
        (-tiny, 1.0),
        (tiny, 1.0),
        (1.17549435e-38, 0.0),
        (largest, INF),
        (-largest, -INF),
        (INF, 0.0),
        (-INF, 0.0),
        (INF, INF),
        (1.0, 2.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (-1.0, -3.0),
        (-3.0, -1.0),
        (1.5, -0.0),
        (-2.5, 0.0),
        (largest, -largest),
    ]
    rng = np.random.default_rng(seed)
    count = n - len(special)
    exponents = rng.integers(-20, 20, (count, 2)).astype(np.float64)
    signs = np.where(rng.random((count, 2)) < 0.5, -1.0, 1.0)
    values = signs * rng.uniform(1.0, 2.0, (count, 2)) * np.exp2(exponents)
    pairs = np.array(special, np.float64).reshape(-1, 2)
    return np.concatenate([pairs, values]).astype(np.float32).reshape(-1)[: 2 * n]


def _fdim(x, y):
    if np.isnan(x) or np.isnan(y):
        return _F32(NAN)
    return _F32(x - y) if x > y else _F32(0.0)


def pair_reference(inputs):
    """Per pair: fdim(x, y), fdim(y, x), nextafter(x, y), copysign(x, y) in binary32."""
    out = []
    with np.errstate(over="ignore", invalid="ignore"):
        for x, y in np.asarray(inputs, np.float32).reshape(-1, 2):
            out.extend(
                [_fdim(x, y), _fdim(y, x), np.nextafter(x, y), np.copysign(x, y)]
            )
    return np.array(out, np.float32)


def ilogb_inputs(n=64, seed=37):
    """Zeros, subnormals of every magnitude, normal binades, infinities and NaN."""
    special = [
        0.0,
        -0.0,
        INF,
        -INF,
        NAN,
        1.401298464324817e-45,
        2.0**-140,
        2.0**-127,
        2.0**-126,
        1.0,
        0.75,
    ]
    special += [_down(2.0**-126), 3.4028234663852886e38, -1e-40, 1e-30, 12345.0]
    rng = np.random.default_rng(seed)
    count = n - len(special)
    spread = (
        np.where(rng.random(count) < 0.5, -1.0, 1.0)
        * rng.uniform(1.0, 2.0, count)
        * np.exp2(rng.integers(-149, 128, count).astype(np.float64))
    )
    return np.concatenate([np.array(special, np.float64), spread]).astype(np.float32)[
        :n
    ]


def _ilogb(x):
    if np.isnan(x) or x == 0:
        return INT_MIN
    if np.isinf(x):
        return INT_MAX
    return int(np.frexp(np.float64(x))[1]) - 1


def ilogb_reference(inputs):
    """Per operand: ilogb(x), ilogb(-x)."""
    out = []
    for x in np.asarray(inputs, np.float32):
        out.extend([_ilogb(x), _ilogb(-x)])
    return np.array(out, np.int32)
