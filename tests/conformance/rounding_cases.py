"""Inputs and oracles of the rounding cases: C11 7.12.9.6 ``round`` and 7.12.9.4 ``rint`` on binary32 values.

``round`` (CUDA ``roundf``, OpenCL C ``round``) rounds halfway cases away from zero and keeps the sign of a zero
result; ``rint`` (CUDA ``rintf``) rounds them to even in the default rounding mode. The inputs are the halfway values
around zero, the largest binary32 value below one half (where ``floor(x + 0.5)`` gives 1), negative values that round
to -0.0, the halfway value below 2^23, values that are already integral, both zeros, both infinities and NaN.

The oracles compute in binary64, where every step is exact for binary32 operands, and do not reuse the formula of the
``round_away`` helper in crosstl/translator/semantics/rule_table.py.
"""

from __future__ import annotations

import math

import numpy as np

NAN, INF = float("nan"), float("inf")
BELOW_HALF = float(np.nextafter(np.float32(0.5), np.float32(0.0)))


def _tile(values, n=64):
    values = np.asarray(values, np.float32)
    return np.tile(values, int(np.ceil(n / values.size)))[:n]


def halfway_inputs():
    return _tile(
        [
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            3.5,
            -3.5,
            4.5,
            -4.5,
            BELOW_HALF,
            -BELOW_HALF,
            -0.25,
            -0.0,
            0.0,
            0.75,
            -2.75,
            1.25,
            8388607.5,
            -8388607.5,
            8388608.0,
            16777216.0,
            1e30,
            -1e30,
            INF,
            -INF,
            NAN,
            1e-45,
            -1e-45,
            12345.5,
            -12344.5,
            float(np.nextafter(np.float32(2.5), np.float32(3.0))),
        ]
    )


def _round_away(value: float) -> float:
    if not math.isfinite(value):
        return value
    magnitude = math.floor(
        abs(value) + 0.5
    )  # exact in binary64 for every binary32 value
    return math.copysign(float(magnitude), value)


def round_away_reference(inputs):
    return np.array([_round_away(float(v)) for v in inputs], np.float64).astype(
        np.float32
    )


def _round_even(value: float) -> float:
    if not math.isfinite(value):
        return value
    # Python's round() of a float rounds halfway cases to even, exactly; it returns +0 for -0.4, so keep the sign.
    return math.copysign(float(round(value)), value)


def round_even_reference(inputs):
    return np.array([_round_even(float(v)) for v in inputs], np.float64).astype(
        np.float32
    )
