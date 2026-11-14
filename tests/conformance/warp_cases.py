"""Oracles of the CUDA warp cases (``x_warp_shuffle``, ``x_warp_vote``), written from the CUDA C++ Programming Guide.

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


VOTE_GROUP = 48
VOTE_RESULTS = 10


def vote_reference(inputs, group=VOTE_GROUP):
    """Oracle of ``x_warp_vote``: CUDA "Warp Vote Functions", "Warp Reduce Functions" and ``__match_any_sync``.

    Each workgroup of ``group`` threads is cut into warps of 32 consecutive local indices; the last is partial and
    has only the lanes that exist. A vote or reduction returns the same value to every lane of its warp; the sum
    wraps modulo 2^32, and min/max compare as the operand's type (signed ``v``, unsigned ``uint(v)``).
    """
    values = inputs.astype(np.int64)
    out = np.zeros((values.size, VOTE_RESULTS), np.uint64)
    for start in range(0, values.size, group):
        for base in range(start, min(start + group, values.size), WARP):
            v = values[base : min(base + WARP, start + group, values.size)]
            unsigned = v & 0xFFFFFFFF
            lanes = np.arange(v.size, dtype=np.uint64)
            ballot = int(np.sum(np.where(v > 0, np.uint64(1) << lanes, 0)))
            and_all = 0xFFFFFFFF
            for item in unsigned | 0xFFFFF800:
                and_all &= int(item)
            or_all, xor_all = 0, 0
            for item in (v + 1000) & 0xFFFFFFFF:
                or_all |= int(item)
            for item in unsigned:
                xor_all ^= int(item)
            row = [
                ballot,
                int(np.any(v > 900)),
                int(np.all(v > -990)),
                int(np.sum(v * 1048576)) & 0xFFFFFFFF,
                int(v.min()) & 0xFFFFFFFF,
                int(unsigned.max()),
                and_all,
                or_all,
                xor_all,
            ]
            for lane, item in enumerate(v):
                matches = np.sum(
                    np.where((v & 3) == (item & 3), np.uint64(1) << lanes, 0)
                )
                out[base + lane] = row + [int(matches)]
    return out.reshape(-1).astype(np.uint32).view(np.int32)
