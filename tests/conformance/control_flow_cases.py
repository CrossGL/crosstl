"""Inputs and oracle of the structured control flow cases.

``s_structured_control_flow`` and ``s_structured_control_flow_phi`` are one GLSL compute kernel compiled by glslang,
the second also optimized by ``spirv-opt -O`` so that its loop variables and switch results are ``OpPhi``
instructions rather than function variables. Each thread writes eight words from its input word ``x``:

0. nested loops: an early ``continue`` in the outer loop and a ``break`` in the inner one;
1. a switch whose cases fall through, with the default between them;
2. a do-while loop with an ``if``/``else`` in its body and a compound exit condition;
3. and 4. two loop-carried values that swap every iteration (a parallel copy of phis);
5. and 6. a storage-buffer word read once and then overwritten, both uses taking the value read;
7. an endless loop left only by a ``return`` inside an ``if``.

SPIR-V 1.6 section 2.11 defines the constructs; the oracle runs the same statements in Python.
"""

from __future__ import annotations

import numpy as np

THREADS = 64
OUTPUTS = 8
_MASK = 0xFFFFFFFF


def structured_control_flow_inputs():
    """Every switch residue and every loop-trip pattern among the first words, then random words."""
    rng = np.random.default_rng(20261021)
    words = rng.integers(0, 2**32, THREADS, dtype=np.uint64)
    edges = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 31, 63, 64, 0x5555, 0xFFFFFFFF, 0x8000001F]
    words[: len(edges)] = edges
    return words.astype(np.uint32)


def _thread(x):
    acc = 0
    for r in range(5):
        if ((x >> r) & 1) == 0:
            continue
        for c in range(7):
            if c * r > (x & 15):
                break
            acc += c + r

    residue = x & 7
    switch = {0: 11, 1: 10, 5: 11000, 6: 10000}.get(residue, 100)

    n = x & 31
    steps = 0
    while True:
        n = n * 3 + 1 if n & 1 else n >> 1
        steps += 1
        if not (n > 1 and steps < 40):
            break

    p, q = x, x ^ 0x5555
    for k in range(x & 3):
        p, q = q, (p + k) & _MASK

    return [acc, switch, steps, p, q, (x + 1) & _MASK, (x * 2) & _MASK, (x & 63) + 1]


def structured_control_flow_reference(inputs):
    """Eight words per thread, in the order the kernel stores them."""
    out = []
    for word in np.asarray(inputs, np.uint32).tolist():
        out.extend(_thread(int(word)))
    return np.array(out, np.uint32)


SHORT_CIRCUIT_THREADS = 64


def short_circuit_inputs():
    """Words whose low four bits take every value, then words outside the 64-element buffer."""
    words = list(range(40)) + [-1, -64, 64, 65, 1000, 1 << 20, -(1 << 30), 2**31 - 1]
    words += [3, 5, 12, 15, 9, 13, 11, 14, 7, 6, 2, 1, 0, 8, 10, 4]
    assert len(words) == SHORT_CIRCUIT_THREADS
    return np.array(words, np.int32)


def short_circuit_reference(inputs):
    """One word per thread: the number of bump() calls and the values the guarded operands decided."""
    words = np.asarray(inputs, np.int32).tolist()
    out = []
    for x in words:
        calls = 0

        def bump():
            nonlocal calls
            calls += 1
            return calls > 2

        both = (x & 1) == 1 and bump()
        either = (x & 2) == 2 or bump()
        chosen = (100 if bump() else 200) if (x & 4) == 4 else 7
        trips = 0
        while trips < (x & 3) and bump():
            trips += 1
        while True:
            trips += 10
            if not ((x & 8) == 8 and trips < 40 and bump()):
                break
        guarded = 1 if 0 <= x < len(words) and words[x] == x else 0
        out.append(
            calls * 1000
            + int(both)
            + 2 * int(either)
            + chosen * 10
            + trips * 100000
            + guarded * 4
        )
    return np.array(out, np.int32)
