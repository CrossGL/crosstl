"""Inputs and oracle of ``x_bfloat16_arithmetic``: bfloat16 results rounded to bfloat16, ties to even.

MSL's ``bfloat`` (MSL section 2.1) and CUDA's ``__nv_bfloat16`` operators compute an arithmetic result in binary32 and
round it to bfloat16, to nearest with ties to even. Every bfloat16 value is a binary32 value, and the operands here
are bfloat16 values of at most eight significant bits within a factor of 2^8 of each other, so the binary32 product,
sum and difference are exact and the oracle rounds once. A translation that keeps the binary32 result of a bfloat16
operation and widens it to ``float`` gives up to 8 more significant bits.
"""

from __future__ import annotations

import numpy as np

RESULTS = 4  # x * y, x + y, x - y, and x * y assigned to a float


def round_bfloat16(values: np.ndarray) -> np.ndarray:
    """Binary32 ``values`` rounded to bfloat16, to nearest with ties to even, as binary32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def inputs(count: int = 256) -> np.ndarray:
    """``count`` pairs of bfloat16 values, a quarter of them with products just above a bfloat16 value."""
    rng = np.random.default_rng(16_2026)
    significands = rng.integers(128, 256, size=(count, 2))
    exponents = rng.integers(-4, 4, size=(count, 2))
    signs = rng.choice([-1.0, 1.0], size=(count, 2))
    pairs = signs * significands * np.exp2(exponents - 7.0)
    # x = 1 + 2^-7 k with k odd and y = 1 + 2^-7 have the product 1 + 2^-7 (k + 1) + 2^-14 k: the bfloat16 rounding
    # discards up to 7 significant bits, from 1 to 127 binary32 ulps of 2^-23 ... 2^-16.
    ties = np.arange(count // 4)
    pairs[ties, 0] = 1.0 + (2 * (ties % 64) + 1) * 2.0**-7
    pairs[ties, 1] = 1.0 + 2.0**-7
    flat = pairs.astype(np.float32).reshape(-1)
    assert np.array_equal(round_bfloat16(flat), flat)
    return flat


def reference(inp: np.ndarray) -> np.ndarray:
    x, y = inp[0::2], inp[1::2]
    out = np.empty((x.size, RESULTS), np.float32)
    out[:, 0] = round_bfloat16(x * y)
    out[:, 1] = round_bfloat16(x + y)
    out[:, 2] = round_bfloat16(x - y)
    out[:, 3] = round_bfloat16(x * y)
    return out.reshape(-1)
