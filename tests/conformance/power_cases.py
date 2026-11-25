"""Inputs and oracle of ``x_integer_powers``, CrossGL's ``powi`` (``crosstl/translator/semantics/powi_helpers.py``).

``powi(x, n)`` follows the C rules for an integral exponent (C17 F.10.4.4): ``powi(x, 0)`` is 1 for every ``x``,
including a NaN; a negative base keeps its sign for an odd exponent; zero to a negative power is an infinity of the
zero's sign for an odd exponent and ``+inf`` otherwise; an infinite base gives an infinity or a zero of the same
signs. The oracle rounds the exact rational power to binary32.

Every exponent is at most 9 in magnitude, so a power by repeated squaring rounds at most 9 times and takes at most
one reciprocal (at most 9 ulp, plus the reciprocal's 2.5 ulp bound in GLSL 4.60 section 4.7.1 and MSL 3.2 section
7.4); the case bounds the result at 16 ulp and the sign of every zero. No result is subnormal, because whether those
flush is the target's choice.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

THREADS = 64
OUTPUTS = 6
_EDGES = [
    (0.0, 0),
    (0.0, -1),
    (-0.0, -3),
    (-0.0, 3),
    (-0.0, 2),
    (0.0, -2),
    (float("nan"), 0),
    (float("inf"), -2),
    (float("-inf"), 3),
    (float("-inf"), -3),
    (-2.0, 7),
    (2.0, -8),
    (-1.0, -7),
    (1.0, 8),
    (0.75, 0),
    (-1.5, -1),
]


def power_inputs():
    """``x = inp[i]`` and ``n = int(inp[i + 64])``: the edge pairs, then signed bases in [0.5, 2) with |n| <= 8."""
    rng = np.random.default_rng(20261122)
    bases = rng.uniform(0.5, 2.0, THREADS) * rng.choice([-1.0, 1.0], THREADS)
    exponents = rng.integers(-8, 9, THREADS).astype(np.float64)
    for thread, (base, exponent) in enumerate(_EDGES):
        bases[thread] = base
        exponents[thread] = exponent
    return np.concatenate([bases, exponents]).astype(np.float32)


def _power(x, n):
    """``x`` to the power ``n``, rounded to binary32, with the C rules for its zeros, infinities and NaNs."""
    if n == 0:
        return 1.0
    if math.isnan(x):
        return x
    negative = math.copysign(1.0, x) < 0 and n % 2 == 1
    if x == 0 or math.isinf(x):
        huge = (x == 0) == (n < 0)
        return (
            -math.inf
            if negative and huge
            else (math.inf if huge else (-0.0 if negative else 0.0))
        )
    exact = Fraction(x) ** n
    with np.errstate(over="ignore"):
        return float(np.float32(float(exact)))


def power_reference(inputs):
    """Six words per thread: ``powi`` of a scalar, both lanes of a vector, a splat exponent, a uint and ``powif``."""
    values = [float(value) for value in inputs]
    out = []
    for thread in range(THREADS):
        x, n = values[thread], int(values[thread + THREADS])
        x2 = float(np.float32(x * 2.0))
        out += [
            _power(x, n),
            _power(x, n),
            _power(-x, 1 - n),
            _power(x2, n + 1),
            _power(x, n & 7),
            _power(x, n),
        ]
    return np.array(out, dtype=np.float32)
