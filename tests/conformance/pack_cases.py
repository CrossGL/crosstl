"""Inputs and oracles of the GLSL packing cases (``g_pack``, ``g_unpack``), GLSL 4.60 section 8.4.

``packUnorm`` writes ``round(clamp(c, 0, 1) * (2^n - 1))`` and ``packSnorm`` writes ``round(clamp(c, -1, 1) *
(2^(n-1) - 1))``, the first component in the least significant bits. GLSL lets ``round`` take either direction at
an exact half, so the pack inputs exclude every value whose scaled form is a tie. ``packHalf2x16`` leaves the
rounding of an inexact value to the implementation, so its inputs are exactly representable in binary16.

``unpackUnorm`` is ``f / (2^n - 1)`` and ``unpackSnorm`` is ``clamp(f / (2^(n-1) - 1), -1, 1)``; GLSL division is
accurate to 2.5 ULP, so the unpack case allows 3. The unpack inputs keep every binary16 half finite, because a NaN
payload is not preserved by every conversion.
"""

from __future__ import annotations

import numpy as np

THREADS = 64
PACK_OUTPUTS = 5
UNPACK_OUTPUTS = 14
_SCALES = (
    np.float32(255.0),
    np.float32(127.0),
    np.float32(65535.0),
    np.float32(32767.0),
)


def _is_tie(value):
    for scale in _SCALES:
        scaled = np.float32(min(max(value, -1.0), 1.0)) * scale
        if scaled - np.floor(scaled) == np.float32(0.5):
            return True
    return False


def pack_inputs():
    """Four binary32 lanes per thread: the edges first, then multiples of 2^-10 in [-1.5, 1.5] that are no tie."""
    edges = [
        -2.0,
        -1.0,
        -0.25,
        0.0,
        0.25,
        1.0,
        1.5,
        2.0,
        -0.0,
        1.0 / 1024.0,
        -1.0 / 1024.0,
        0.75,
    ]
    rng = np.random.default_rng(20261020)
    values = [np.float32(value) for value in edges]
    while len(values) < THREADS * 4:
        candidate = np.float32(int(rng.integers(-1536, 1537)) / 1024.0)
        if not _is_tie(candidate):
            values.append(candidate)
    values = np.array(values, np.float32)
    assert not any(_is_tie(value) for value in values)
    return values


def _quantize(values, scale, signed):
    low = -1.0 if signed else 0.0
    scaled = np.clip(values, low, 1.0).astype(np.float32) * np.float32(scale)
    quantized = np.rint(scaled).astype(np.int64)
    return quantized & (0xFF if scale < 256 else 0xFFFF)


def _join(parts, width):
    word = np.zeros(parts.shape[0], np.uint64)
    for index in range(parts.shape[1]):
        word |= parts[:, index].astype(np.uint64) << np.uint64(width * index)
    return word.astype(np.uint32)


def pack_reference(inputs):
    lanes = inputs.reshape(-1, 4)
    halves = lanes[:, 1:3].astype(np.float16).view(np.uint16)
    columns = (
        _join(_quantize(lanes, 255, False), 8),
        _join(_quantize(lanes, 127, True), 8),
        _join(_quantize(lanes[:, :2], 65535, False), 16),
        _join(_quantize(lanes[:, 2:], 32767, True), 16),
        _join(halves, 16),
    )
    return np.stack(columns, axis=1).reshape(-1).astype(np.uint32)


def unpack_inputs():
    """One word per thread: signed extremes, both zeros and random words whose binary16 halves are finite."""
    edges = [
        0x00000000,
        0x80808080,
        0x81818181,
        0x7F7F0000,
        0x01FF7F80,
        0x80008000,
        0x7BFF8001,
        0x3C00BC00,
    ]
    rng = np.random.default_rng(20261021)
    words = rng.integers(0, 2**32, THREADS - len(edges), dtype=np.uint64)
    # Clear the top exponent bit of each binary16 half, so no half is an infinity or a NaN.
    words &= np.uint64(~0x40004000 & 0xFFFFFFFF)
    return np.concatenate([np.array(edges, np.uint64), words]).astype(np.uint32)


def _fields(words, width, count, signed):
    mask = (1 << width) - 1
    fields = np.stack(
        [(words.astype(np.int64) >> (width * index)) & mask for index in range(count)],
        axis=1,
    )
    if signed:
        fields = np.where(fields >= 1 << (width - 1), fields - (1 << width), fields)
    return fields.astype(np.float32)


def unpack_reference(inputs):
    words = inputs.astype(np.uint32)
    halves = np.stack([words & 0xFFFF, words >> 16], axis=1).astype(np.uint16)
    columns = (
        _fields(words, 8, 4, False) / np.float32(255.0),
        np.clip(_fields(words, 8, 4, True) / np.float32(127.0), -1.0, 1.0),
        _fields(words, 16, 2, False) / np.float32(65535.0),
        np.clip(_fields(words, 16, 2, True) / np.float32(32767.0), -1.0, 1.0),
        halves.view(np.float16).astype(np.float32),
    )
    return np.concatenate(columns, axis=1).astype(np.float32).reshape(-1)


def bitfield_inputs():
    """One word per thread: all-zero, all-one and sign-bit patterns first, then random words."""
    edges = [0x00000000, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x55555555, 0xAAAAAAAA]
    rng = np.random.default_rng(20261022)
    words = rng.integers(0, 2**32, THREADS - len(edges), dtype=np.uint64)
    return np.concatenate([np.array(edges, np.uint64), words]).astype(np.uint32)


def _extract(value, offset, bits, signed):
    if bits == 0:
        return 0
    field = (value >> offset) & ((1 << bits) - 1)
    if signed and field >> (bits - 1):
        field -= 1 << bits
    return field & 0xFFFFFFFF


def _insert(base, insert, offset, bits):
    mask = ((1 << bits) - 1) << offset
    return (base & ~mask & 0xFFFFFFFF) | ((insert << offset) & mask)


def bitfield_reference(inputs):
    out = []
    for index, value in enumerate(int(word) for word in inputs):
        offset = index % 33
        bits = min((index * 7) % 33, 32 - offset)
        hashed = (index * 2654435761) & 0xFFFFFFFF
        out += [
            _extract(value, offset, bits, False),
            _extract(value, offset, bits, True),
            _insert(value, ~value & 0xFFFFFFFF, offset, bits),
            _insert(value, hashed, offset, bits),
        ]
    return np.array(out, np.uint32)


def mix_select_inputs():
    """Operand pairs in which the operand that is not selected is often infinite or NaN."""
    specials = [np.inf, -np.inf, np.nan, 3.0e38, -0.0, 1.0]
    rng = np.random.default_rng(20261023)
    values = rng.uniform(-8.0, 8.0, 2 * THREADS).astype(np.float32)
    values[::3] = np.array(specials * THREADS, np.float32)[: values[::3].size]
    return values


def mix_select_reference(inputs):
    pairs = inputs.reshape(-1, 2)
    out = []
    for index, (x, y) in enumerate(pairs):
        first, second = index % 3 == 0, index % 3 == 1
        out += [y if index & 1 else x, y if first else x, x if second else y]
    return np.array(out, np.float32)


def exponent_inputs():
    """Zero, powers of two, and normal values whose magnitude stays within [2^-60, 2^60].

    The values are normal, so no device flushes them, and ``ldexp`` by at most 2^20 stays normal, so every result is
    exact. Negative values are never integral: ``modf`` of a negative integer has a zero fraction whose sign C, MSL
    and HLSL fix as negative while ``x - trunc(x)`` gives positive zero, and GLSL does not pin the sign of zero.
    """
    edges = [
        0.0,
        1.0,
        0.5,
        2.0,
        0.75,
        3.0,
        -0.75,
        -1.5,
        1024.5,
        -3.25,
        2.0**-60,
        2.0**60,
        -(2.0**-30) * 1.5,
    ]
    rng = np.random.default_rng(20261022)
    values = [np.float32(value) for value in edges]
    while len(values) < THREADS:
        magnitude = np.float32(2.0 ** rng.uniform(-60.0, 60.0))
        if rng.integers(0, 2):
            if np.trunc(magnitude) == magnitude:
                continue
            magnitude = -magnitude
        values.append(magnitude)
    return np.array(values, np.float32)


def exponent_reference(inputs):
    significands, exponents = np.frexp(inputs)
    indices = np.arange(inputs.shape[0])
    scaled = np.ldexp(inputs, (indices % 41) - 20).astype(np.float32)
    fractions, wholes = np.modf(inputs)
    columns = (
        significands.astype(np.float32).view(np.uint32),
        exponents.astype(np.int64).astype(np.uint32),
        scaled.view(np.uint32),
        fractions.astype(np.float32).view(np.uint32),
        wholes.astype(np.float32).view(np.uint32),
    )
    return np.stack(columns, axis=1).reshape(-1).astype(np.uint32)


def hlsl_exponent_inputs():
    """Zero, powers of two and normal values in [2^-30, 2^30], so ``modf``'s whole part fits an int."""
    edges = [
        0.0,
        1.0,
        0.5,
        2.0,
        0.75,
        3.0,
        -0.75,
        -1.5,
        1024.5,
        -3.25,
        2.0**-30,
        2.0**30,
        -12345.678,
    ]
    rng = np.random.default_rng(20261023)
    values = [np.float32(value) for value in edges]
    while len(values) < THREADS:
        magnitude = np.float32(2.0 ** rng.uniform(-30.0, 30.0))
        if rng.integers(0, 2):
            if np.trunc(magnitude) == magnitude:
                continue
            magnitude = -magnitude
        values.append(magnitude)
    return np.array(values, np.float32)


def hlsl_exponent_reference(inputs):
    significands, exponents = np.frexp(inputs)
    fractions, wholes = np.modf(inputs)
    doubled, doubled_exponents = np.frexp(inputs * np.float32(2.0))
    indices = np.arange(inputs.shape[0])
    scales = np.exp2((indices % 9) * 0.5 - 2.0)
    columns = (
        significands,
        exponents.astype(np.float64),
        fractions,
        wholes,
        # frexp(0.5) has exponent 0, so the third lane subtracts nothing.
        (doubled.astype(np.float32) + doubled_exponents.astype(np.float32)).astype(
            np.float32
        ),
        inputs.astype(np.float64) * scales,
    )
    return (
        np.stack([np.asarray(c, np.float64) for c in columns], axis=1)
        .reshape(-1)
        .astype(np.float32)
    )
