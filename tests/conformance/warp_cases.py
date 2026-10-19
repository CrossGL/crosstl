"""Oracle of the CUDA warp-shuffle case (``x_warp_shuffle``), written from the CUDA C++ Programming Guide.

A warp is 32 threads of consecutive index. With ``width`` < 32 each group of ``width`` consecutive lanes behaves as
a separate warp whose logical lanes start at 0 ("Warp Shuffle Functions"):

- ``__shfl_sync(m, v, s, w)`` returns ``v`` of logical lane ``s`` modulo ``w``;
- ``__shfl_up_sync(m, v, d, w)`` returns ``v`` of logical lane ``l - d``; the lower ``d`` lanes keep their own;
- ``__shfl_down_sync(m, v, d, w)`` returns ``v`` of logical lane ``l + d``; the upper ``d`` lanes keep their own;
- ``__shfl_xor_sync(m, v, x, w)`` returns ``v`` of lane ``l ^ x``, or the caller's own value when that lane lies
  in a later group of ``w`` lanes.

The kernel stores eight results per thread: a shuffle-down tree sum, then the seven shuffles of the source in order.
"""

from __future__ import annotations

import numpy as np

WARP = 32
RESULTS = 8


def _down(values, delta, width=WARP):
    lanes = np.arange(WARP)
    base, sub = lanes & ~(width - 1), lanes & (width - 1)
    source = np.where(sub + delta < width, base + sub + delta, lanes)
    return values[source]


def _up(values, delta, width=WARP):
    lanes = np.arange(WARP)
    sub = lanes & (width - 1)
    return values[np.where(sub >= delta, lanes - delta, lanes)]


def _xor(values, mask, width=WARP):
    lanes = np.arange(WARP)
    partner = lanes ^ mask
    later = (partner & ~(width - 1)) > (lanes & ~(width - 1))
    return values[np.where(later, lanes, partner)]


def _index(values, source, width=WARP):
    lanes = np.arange(WARP)
    return values[(lanes & ~(width - 1)) + np.asarray(source) % width]


def shuffle_reference(inputs):
    out = []
    for warp in inputs.astype(np.int64).reshape(-1, WARP):
        total = warp.copy()
        for offset in (16, 8, 4, 2, 1):
            total = total + _down(total, offset)
        lanes = np.arange(WARP)
        columns = (
            total,
            _up(warp, 3),
            _xor(warp, 5),
            _index(warp, lanes * 7 + 3),
            _down(warp, 3, 8),
            _up(warp, 2, 4),
            _xor(warp, 17, 16),
            _index(warp, np.full(WARP, 5), 4),
        )
        out.append(np.stack(columns, axis=1))
    return np.concatenate(out).reshape(-1).astype(np.int32)
