"""Constants block, inputs and oracle of the case that passes arrays to helper functions by value.

GLSL passes an ``in`` array parameter by value (GLSL 4.60 section 6.1.1), so a helper reads the caller's elements
wherever they are stored and its writes stay in its own copy. The std140 offsets below are written out by hand
from GLSL 4.60 section 7.6.2.2: every array element, matrix column and row-major matrix row starts a 16-byte slot.
All values are small integers, so every sum and product is exact in binary32.
"""

from __future__ import annotations

import numpy as np

from .constants_cases import block as _block

THREADS = 64
OUTPUTS = 8
V = ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))
W = (9.0, 10.0, 11.0)
# m[k][column][row] of the column-major mat2x3 array.
M = (
    ((12.0, 13.0, 14.0), (15.0, 16.0, 17.0)),
    ((18.0, 19.0, 20.0), (21.0, 22.0, 23.0)),
)
# r[k][column][row] of the row-major mat2 array, stored one row per slot.
R = (((24.0, 25.0), (26.0, 27.0)), ((28.0, 29.0), (30.0, 31.0)))


def block():
    fields = [(16 * k, V[k], np.float32) for k in range(2)]
    fields += [(32 + 16 * k, (W[k],), np.float32) for k in range(3)]
    fields += [
        (80 + 32 * k + 16 * column, M[k][column], np.float32)
        for k in range(2)
        for column in range(2)
    ]
    fields += [
        (144 + 32 * k + 16 * row, (R[k][0][row], R[k][1][row]), np.float32)
        for k in range(2)
        for row in range(2)
    ]
    return _block(fields)


def inputs():
    return np.arange(THREADS, dtype=np.float32)


def reference(inp):
    x = inp.astype(np.float64)
    s = 2.0 * x
    out = np.zeros((THREADS, OUTPUTS), dtype=np.float32)
    for i in range(THREADS):
        nxt = (i + 1) % THREADS
        out[i] = (
            V[i % 2][i % 4],
            W[i % 3],
            M[i % 2][(i // 2) % 2][i % 3],
            R[i % 2][(i // 2) % 2][(i // 4) % 2],
            V[(i + 1) % 2][(i + 1) % 4],
            2.0 * W[i % 3] + 100.0,
            2.0 * s[nxt] + 100.0,
            W[(i + 1) % 3] * x[i],
        )
    return out.reshape(-1)
