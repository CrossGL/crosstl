"""Constants blocks, inputs and oracles of the constant-buffer layout cases.

Each case reads one constants block whose members the source languages place differently: GLSL ``std140`` and,
for SPIR-V push constants, ``std430`` (GLSL 4.60 sections 7.6.2.2 and 7.6.2.3), HLSL constant-buffer packing (the "Packing rules for constant variables" page of the
HLSL reference) and the natural MSL layout of a ``constant`` struct (the size and alignment tables of MSL 4.1
chapter 2: a ``float3`` takes 16 bytes, a ``packed_float3`` 12 with 4-byte alignment, a ``float2`` array has an
8-byte stride and ``float3x3`` columns are 16 bytes apart). The host uploads the block in the source's layout, so a translation that moves a member reads
another member or the canary words that fill every padding byte.

The offsets below are written out from those rules by hand, independently of the translator's layout code. All
values are small integers, so every product and sum is exact in binary32 with or without contraction.
"""

from __future__ import annotations

import numpy as np

THREADS = 64
OUTPUTS = 8
BLOCK_BYTES = (
    256  # a multiple of the 256 bytes a Direct3D 12 constant-buffer view covers
)
CANARY = 0x7F7F7F7F

SCALE = (3.0, 5.0, 7.0)
BIAS = 11.0
OFFSETS = ((13.0, 17.0), (19.0, 23.0), (29.0, 31.0))
ROT = ((1.0, 2.0, 3.0), (4.0, 6.0, 8.0), (9.0, 10.0, 12.0))  # columns
MODE = 37
WEIGHTS = (41.0, 43.0)
TAIL = (47.0, 53.0)
K = 59.0  # HLSL only
SHIFT = (61.0, 67.0, 71.0)  # Metal only

# (byte offset, values, dtype) per source layout.
STD140 = (
    (0, SCALE, np.float32),
    (12, (BIAS,), np.float32),
    *((16 + 16 * index, value, np.float32) for index, value in enumerate(OFFSETS)),
    *((64 + 16 * index, column, np.float32) for index, column in enumerate(ROT)),
    (112, (MODE,), np.int32),
    *((128 + 16 * index, (value,), np.float32) for index, value in enumerate(WEIGHTS)),
    (160, TAIL, np.float32),
)
STD430 = (
    (0, SCALE, np.float32),
    (12, (BIAS,), np.float32),
    *((16 + 8 * index, value, np.float32) for index, value in enumerate(OFFSETS)),
    *((48 + 16 * index, column, np.float32) for index, column in enumerate(ROT)),
    (96, (MODE,), np.int32),
    *((100 + 4 * index, (value,), np.float32) for index, value in enumerate(WEIGHTS)),
    (112, TAIL, np.float32),
)
HLSL = (
    (0, SCALE, np.float32),
    (12, (BIAS,), np.float32),
    *((16 + 16 * index, value, np.float32) for index, value in enumerate(OFFSETS)),
    (56, TAIL, np.float32),
    *((64 + 16 * index, column, np.float32) for index, column in enumerate(ROT)),
    (108, (K,), np.float32),
    (112, (MODE,), np.int32),
    *((128 + 16 * index, (value,), np.float32) for index, value in enumerate(WEIGHTS)),
)
# A row_major float3x3 stores the rows of the same matrix: row r holds component r of every column.
ROWS = tuple(tuple(column[row] for column in ROT) for row in range(3))
HLSL_ROW_MAJOR = (
    *HLSL[:6],
    *((64 + 16 * index, row, np.float32) for index, row in enumerate(ROWS)),
    *HLSL[9:],
)
MSL = (
    (0, SCALE, np.float32),
    (16, (BIAS,), np.float32),
    (20, SHIFT, np.float32),
    *((32 + 8 * index, value, np.float32) for index, value in enumerate(OFFSETS)),
    *((64 + 16 * index, column, np.float32) for index, column in enumerate(ROT)),
    (112, (MODE,), np.int32),
    *((116 + 4 * index, (value,), np.float32) for index, value in enumerate(WEIGHTS)),
    (128, TAIL, np.float32),
)


def block(fields):
    """The block's bytes: each member's little-endian values at its offset, canary words everywhere else."""
    data = np.full(BLOCK_BYTES // 4, CANARY, dtype=np.uint32).view(np.uint8)
    taken = np.zeros(BLOCK_BYTES, dtype=bool)
    for offset, values, dtype in fields:
        raw = np.asarray(values, dtype=dtype).astype("<" + np.dtype(dtype).str[1:])
        raw = raw.view(np.uint8)
        end = offset + raw.size
        if end > BLOCK_BYTES or taken[offset:end].any():
            raise ValueError(
                f"Member at offset {offset} overlaps or overflows the block"
            )
        data[offset:end] = raw
        taken[offset:end] = True
    return data.tobytes()


def inputs():
    return np.arange(THREADS, dtype=np.float32)


def _rotated(v, lane):
    """``(rot * vec3(v, 1, 2))[lane]`` with ``rot`` given by its columns."""
    return ROT[0][lane] * v + ROT[1][lane] + 2.0 * ROT[2][lane]


def _reference(inp, row):
    out = np.zeros((THREADS, OUTPUTS), dtype=np.float32)
    for i, v in enumerate(inp.astype(np.float64)):
        out[i] = row(i, v)
    return out.reshape(-1)


def glsl_reference(inp):
    return _reference(
        inp,
        lambda i, v: (
            v * SCALE[0] + BIAS,
            v * SCALE[1] + SCALE[2],
            OFFSETS[i % 3][0],
            OFFSETS[i % 3][1],
            _rotated(v, i % 3),
            MODE,
            WEIGHTS[i % 2],
            TAIL[1],
        ),
    )


def hlsl_reference(inp):
    return _reference(
        inp,
        lambda i, v: (
            v * SCALE[0] + BIAS,
            v * SCALE[1] + SCALE[2],
            OFFSETS[i % 3][0],
            OFFSETS[i % 3][1] + TAIL[1],
            _rotated(v, i % 3),
            MODE + K,
            WEIGHTS[i % 2],
            TAIL[0],
        ),
    )


def msl_reference(inp):
    return _reference(
        inp,
        lambda i, v: (
            v * SCALE[0] + BIAS,
            v * SCALE[1] + SCALE[2] + SHIFT[1],
            OFFSETS[i % 3][0],
            OFFSETS[i % 3][1] + SHIFT[2],
            _rotated(v, i % 3),
            MODE + SHIFT[0],
            WEIGHTS[i % 2],
            TAIL[1],
        ),
    )
