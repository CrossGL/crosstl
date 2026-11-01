"""Recorded MLX generator seeds for the upstream tests verify_upstream_ops runs."""

import argparse
import hashlib


def test_seed(seed, name):
    """The MLX generator seed of upstream test ``name`` in a run seeded ``seed``.

    MLX seeds its generator from the clock and NumPy from the system, and most
    upstream tests draw their inputs without a seed of their own. The verifier
    picks a run seed instead and records it, and each test seeds both
    generators from a seed derived from it and the test's name, so a failing
    test replays alone with the same inputs.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def parse_seed(text):
    """An unsigned 64-bit run seed from the command line."""
    value = int(text)
    if not 0 <= value < 1 << 64:
        raise argparse.ArgumentTypeError("the seed must be an unsigned 64-bit integer")
    return value
