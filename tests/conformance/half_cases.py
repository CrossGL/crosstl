"""Inputs and oracles of the binary16 arithmetic cases (``m_half_*``): exact rational references, rounded once.

MSL computes ``half`` arithmetic in binary16 and rounds every result to nearest, ties to even (MSL section 8.2; IEEE
754-2019 sections 4.3.1 and 5.4.1 for ``fma``). Denormalized halves "may be flushed to zero" (MSL section 8.1), so the
oracle also accepts a zero where the exact result is subnormal. The operands are normal binary16 values; the division,
square-root and fused multiply-add inputs include results within a few binary32 ULP of a binary16 midpoint, where an
uncorrected binary32 operation (2.5 ULP in Vulkan and WGSL) can round to the wrong neighbour.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

_EDGE = Fraction(65520)
_SUBNORMAL = 2.0**-14


def round_half(value: Fraction) -> float:
    """``value`` rounded to binary16, ties to even (overflow to infinity), as a Python float."""
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    if magnitude >= _EDGE:
        return sign * float("inf")
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    ulp = Fraction(2) ** (max(exponent, -14) - 10)
    return sign * float(round(magnitude / ulp) * ulp)


def _halves(values):
    return np.asarray(values, np.float64).astype(np.float16).astype(np.float32)


def _normal(values):
    values = _halves(values)
    return values[(np.abs(values) >= _SUBNORMAL) & np.isfinite(values)]


def div_inputs(count=512):
    """Pairs (a, b): half of them have quotients two binary32 ULP from a binary16 midpoint, the rest are random."""
    rng = np.random.default_rng(1601)
    hard = []
    # a = A 2^-10, b = B 2^-10 and the midpoint m = M 2^-11 (M odd) with A 2^11 - M B = +-1.
    for b in range(1025, 2048, 2):
        inverse = pow(b, -1, 2048)
        for sign in (1, -1):
            m = 2048 + (-sign * inverse) % 2048
            numerator = m * b + sign
            if numerator % 2048 == 0 and 1024 <= numerator // 2048 < 2048:
                hard.append((numerator // 2048, b))
    hard = hard[: count // 2]
    pairs = []
    for index, (a, b) in enumerate(hard):
        scale_a, scale_b = rng.integers(-6, 7, 2)
        flip = -1.0 if index % 3 == 0 else 1.0
        pairs.append(
            (flip * a * 2.0 ** (int(scale_a) - 10), b * 2.0 ** (int(scale_b) - 10))
        )
    while len(pairs) < count:
        a, b = _normal(rng.uniform(-4, 4, 2) * 2.0 ** rng.integers(-8, 8, 2))
        if len(_normal([a, b])) == 2:
            pairs.append((float(a), float(b)))
    return _halves([value for pair in pairs[:count] for value in pair])


def div_reference(inputs):
    pairs = inputs.astype(np.float64).reshape(-1, 2)
    return [round_half(Fraction(a) / Fraction(b)) for a, b in pairs]


def sqrt_inputs():
    """Every binary16 value in [1, 4) and 512 random positive normal values."""
    every = np.arange(0x3C00, 0x4400, dtype=np.uint16).view(np.float16)
    rng = np.random.default_rng(1602)
    extra = _normal(rng.uniform(1, 2, 2048) * 2.0 ** rng.integers(-14, 15, 2048))[:512]
    return np.concatenate([every.astype(np.float32), extra])


def sqrt_reference(inputs):
    # A binary16 square root is never within 2^-23 (relative) of a binary16 midpoint: rounding the binary64 root
    # exactly is the correctly rounded result.
    return [
        round_half(Fraction(float(np.sqrt(np.float64(value))))) for value in inputs
    ]


def fma_inputs(count=512):
    """Triples (a, b, c): sums next to a binary16 midpoint, exact cancellations and random triples."""
    rng = np.random.default_rng(1603)
    triples = [
        (1.5, 0.6669921875, 2.0**-14),
        (1.5, 0.6669921875, -(2.0**-14)),
        (1.5, 0.6669921875, -1.0),
        (256.0, 256.0, -16.0),
        (255.875, 256.0, -65504.0),
        (65504.0, 1.0, 16.0),
        (65504.0, 1.0, 15.0),
    ]
    # products that are binary16 midpoints, with an addend at most half a binary32 ULP of the product: the binary32
    # sum rounds back onto the midpoint, and only the exact sum decides the direction
    for m in range(2049, 4096, 2):
        factor = next((f for f in range(3, 46, 2) if m % f == 0 and m // f < 2048), None)
        if factor is None:
            continue
        scale = int(rng.integers(0, 3))
        a, b = factor * 2.0 ** int(rng.integers(-4, 1)), m // factor * 1.0
        b *= 2.0**scale / (a / factor)
        sign = -1.0 if len(triples) % 2 else 1.0
        c = sign * 2.0 ** int(rng.integers(-14, scale - 12))
        triples.append((-a if len(triples) % 3 == 0 else a, b, c))
        if len(triples) >= count // 4:
            break
    while len(triples) < count // 2:
        exponent = int(rng.integers(-10, 14))
        midpoint = (2 * int(rng.integers(1024, 2048)) + 1) * 2.0 ** (exponent - 11)
        c = rng.uniform(-1, 1) * 2.0 ** int(rng.integers(-14, exponent))
        a = rng.uniform(1, 2) * 2.0 ** int(rng.integers(-5, 6))
        a, c = (float(v) for v in _halves([a, c]))
        b = float(_halves([(midpoint - c) / a])[0])
        if len(_normal([a, b, c])) == 3:
            triples.append((a, b, c))
    while len(triples) < count:
        a, b = _halves(rng.uniform(-2, 2, 2) * 2.0 ** rng.integers(-6, 6, 2))
        c = float(_halves([-a * b if len(triples) % 4 == 0 else rng.uniform(-8, 8)])[0])
        if len(_normal([a, b, c])) == 3:
            triples.append((float(a), float(b), c))
    return _halves([value for triple in triples for value in triple])


def fma_reference(inputs):
    triples = inputs.astype(np.float64).reshape(-1, 3)
    return [
        round_half(Fraction(a) * Fraction(b) + Fraction(c)) for a, b, c in triples
    ]


def round_inputs(count=512):
    """binary32 values on and next to binary16 midpoints, from the subnormal range to overflow, and random ones."""
    rng = np.random.default_rng(1604)
    midpoints = [(2 * k + 1) * 2.0**-25 for k in (0, 1, 2, 3, 511, 1022, 1023)]
    for exponent in (-14, -13, -1, 0, 1, 10, 15):
        midpoints += [(2 * k + 1) * 2.0 ** (exponent - 11) for k in (1024, 1025, 1536, 2047)]
    values = []
    for midpoint in midpoints:
        exact = np.float32(midpoint)
        values += [np.nextafter(exact, np.float32(-np.inf)), exact, np.nextafter(exact, np.float32(np.inf))]
    values += [np.float32(v) for v in (65504.0, 65519.996, 65520.0, 65536.0, 2.0**-26)]
    values = np.array(values, np.float32)
    signs = np.where(np.arange(values.size) % 2 == 0, 1.0, -1.0).astype(np.float32)
    values = values * signs
    patterns = rng.integers(0x33000000, 0x47800000, count, dtype=np.int64).astype(np.uint32)
    random = patterns.view(np.float32) * np.where(np.arange(count) % 2, -1, 1).astype(np.float32)
    return np.concatenate([values, random])[:count].astype(np.float32)


def round_reference(inputs):
    return [round_half(Fraction(float(value))) for value in inputs]


def check(reference):
    """A parametric oracle: bit-exact binary16 results, or a zero where the exact result is subnormal."""

    def compare(inputs, out):
        want = np.array(reference(inputs), np.float32)
        got = out.astype(np.float32)[: want.size]
        exact = got.view(np.uint32) == want.view(np.uint32)
        flushed = (got == 0) & (np.abs(want) < _SUBNORMAL)
        bad = np.nonzero(~(exact | flushed))[0]
        detail = {"mismatches": int(bad.size), "n": int(want.size)}
        if bad.size:
            first = int(bad[0])
            detail.update(
                first_index=first, got=repr(float(got[first])), want=repr(float(want[first]))
            )
        return bad.size == 0, detail

    return compare
