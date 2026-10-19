"""Inputs and oracles of the OpenCL C saturated-conversion cases (``c_convert_*``), OpenCL C section 6.4.3.3.

A saturated conversion maps NaN to 0 and clamps every other value, after rounding in the named mode, to the range of
the destination type. The inputs cover NaN of both signs, both infinities, the binary32 values next to each bound,
ties for round-to-nearest-even, negative values that round to zero and a subnormal.
"""

from __future__ import annotations

import math

import numpy as np

NAN, INF = float("nan"), float("inf")
_NEG_NAN = float(np.array([0xFFC00000], np.uint32).view(np.float32)[0])


def _tile(values, n=64):
    values = np.asarray(values, np.float32)
    return np.tile(values, int(np.ceil(n / values.size)))[:n]


def uint_rte_inputs():
    return _tile(
        [
            NAN,
            _NEG_NAN,
            INF,
            -INF,
            0.5,
            1.5,
            2.5,
            3.5,
            -0.5,
            -1.5,
            0.49999997,
            16777215.0,
            2147483648.0,
            4294967040.0,
            5e9,
            -3e9,
        ]
    )


def char_inputs():
    return _tile(
        [
            NAN,
            _NEG_NAN,
            INF,
            -INF,
            127.9,
            128.0,
            -128.9,
            -129.0,
            2.7,
            -2.7,
            3e9,
            -3e9,
            -0.5,
            1e-40,
            100.0,
            -100.5,
        ]
    )


def saturated(values, low, high, rounding):
    """``values`` (binary32) rounded by ``rounding`` and clamped to [low, high]; NaN gives 0."""
    out = []
    for value in np.asarray(values, np.float64):
        if math.isnan(value):
            out.append(0)
        elif value >= high:
            out.append(high)
        elif value <= low:
            out.append(low)
        else:
            out.append(min(max(int(rounding(value)), low), high))
    return np.array(out, np.int64)


def uint_rte_reference(values):
    return saturated(values, 0, 2**32 - 1, np.rint).astype(np.uint32)


def char_reference(values):
    return saturated(values, -128, 127, math.trunc).astype(np.int32)
