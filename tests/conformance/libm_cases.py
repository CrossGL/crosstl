"""Inputs and oracles of the libm cases: C-family ``sinf`` and ``cosf`` over the whole binary32 range.

CUDA and HIP bound ``sinf`` and ``cosf`` at 2 ulp over every finite operand, and OpenCL C at 4 ulp; ``sin`` keeps the
sign of a zero, and infinities and NaN give NaN. Graphics targets bound their own ``sin`` on ``[-pi, pi]`` at best,
so these inputs reach far outside it: both zeros, subnormals, the 2^-12 boundary below which ``sin(x) = x``, the
binary32 values nearest to multiples of ``pi/2`` (where the reduction cancels), every few binades up to the largest
finite value, both infinities and NaN.

The oracles evaluate the binary64 functions and round once to binary32; they share nothing with the integer helpers
in crosstl/translator/semantics/libm_trig.py.
"""

from __future__ import annotations

import math

import numpy as np

NAN, INF = float("nan"), float("inf")


def _near_half_pi_multiples():
    values = []
    for k in (1, 2, 3, 4, 7, 100, 1000, 12345, 102943, 1 << 20):
        nearest = np.float32(k * math.pi / 2)
        values.append(nearest)
        values.append(np.nextafter(nearest, np.float32(INF)))
    return values


def range_inputs(n=64, seed=11):
    special = [
        0.0,
        -0.0,
        INF,
        -INF,
        NAN,
        1e-45,
        -1e-45,
        1.17549435e-38,
        2.0**-12,
        -(2.0**-12),
        float(np.nextafter(np.float32(2.0**-12), np.float32(0.0))),
        0.785398185253143310546875,
        3.4028234663852886e38,
        -3.4028234663852886e38,
        16367173.0 * 2.0**72,
        1e10,
    ]
    fixed = np.array(special + _near_half_pi_multiples(), np.float32)
    rng = np.random.default_rng(seed)
    count = n - fixed.size
    exponents = rng.integers(-12, 128, count)
    mantissas = rng.uniform(1.0, 2.0, count)
    signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    spread = (signs * mantissas * np.exp2(exponents.astype(np.float64))).astype(
        np.float32
    )
    return np.concatenate([fixed, spread])[:n]


def _reference(function, inputs):
    with np.errstate(invalid="ignore"):
        return function(np.asarray(inputs, np.float32).astype(np.float64)).astype(
            np.float32
        )


def sin_reference(inputs):
    return _reference(np.sin, inputs)


def cos_reference(inputs):
    return _reference(np.cos, inputs)
