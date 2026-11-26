"""Oracle of ``x_workgroup_collectives``: CrossGL's reductions, scans, votes and broadcast over a whole workgroup.

``crosstl/translator/workgroup_operations.py`` defines them: ``workgroup<Op>`` combines the values of every
invocation of the workgroup, ``workgroupInclusive<Op>`` those whose local invocation index is at most the caller's,
``workgroupExclusive<Op>`` those below it, starting from the operator's identity (the smallest ``int`` for ``Max``,
1 for ``Mul``), ``workgroupAll`` and ``workgroupAny`` vote, and ``workgroupBroadcast(x, id)`` returns invocation
``id``'s value. Integer arithmetic wraps; the float sum adds halves of integers, which binary32 represents exactly.
"""

from __future__ import annotations

import numpy as np

GROUP = 64
RESULTS = 12
_INT_MIN = -(2**31)


def _wrap(value):
    return ((int(value) + 2**31) % 2**32) - 2**31


def reference(inp):
    """The 12 results of each invocation, in invocation order."""
    out = []
    for group in inp.astype(np.int64).reshape(-1, GROUP):
        values = [int(v) for v in group]
        bits = [(v & 0xFFFFFFFF) & 255 for v in values]
        total = sum(values)
        votes = int(all(v > -990 for v in values)) + 2 * int(
            any(v > 990 for v in values)
        )
        xor = 0
        for b in bits:
            xor ^= b
        floats = sum(v * 0.5 for v in values) * 2.0
        for lane, x in enumerate(values):
            before, upto = values[:lane], values[: lane + 1]
            ored = 0
            for b in bits[: lane + 1]:
                ored |= b
            product = 1
            for b in bits[:lane]:
                product = (product * ((b & 1) + 1)) % 2**32
            pair = max(upto) * 3 + max(-v for v in upto)
            out.extend(
                [
                    total,
                    sum(upto),
                    sum(before),
                    min(values) * 7 + max(values),
                    min(upto),
                    max(before) if before else _INT_MIN,
                    values[5],
                    votes,
                    xor ^ (ored << 8),
                    _wrap(product),
                    int(floats),
                    pair,
                ]
            )
    return np.array([_wrap(v) for v in out], dtype=np.int32)
