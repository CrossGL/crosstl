"""Recorded MLX generator seeds for the upstream tests verify_upstream_ops runs."""

import argparse
import hashlib

# Upstream tests whose reference is less accurate than their bound for some
# inputs, so that MLX's own Metal backend fails them there. They draw the
# inputs of run seed 0, on which stock MLX passes them on Metal and on the CPU.
#
# test_softmax compares float16 results with a NumPy softmax that it also
# evaluates in float16. That reference can be 1.1e-3 from the exact softmax,
# beyond the test's 1e-3 tolerance. Stock MLX 0.32.4 on Metal, whose float16
# results are closer to the exact values (4e-4 on the same inputs), fails
# the axis-0 case for 15 of 400 NumPy seeds. The CPU backend evaluates like
# the reference and fails none.
FIXED_INPUT_TESTS = frozenset(("test_ops.TestOps.test_softmax",))
FIXED_RUN_SEED = 0


def test_seed(seed, name):
    """The MLX generator seed of upstream test ``name`` in a run seeded ``seed``.

    MLX seeds its generator from the clock and NumPy from the system, and most
    upstream tests draw their inputs without a seed of their own. The verifier
    picks a run seed instead and records it, and each test seeds both
    generators from a seed derived from it and the test's name, so a failing
    test replays alone with the same inputs. The tests in FIXED_INPUT_TESTS
    always derive theirs from FIXED_RUN_SEED.
    """
    if name in FIXED_INPUT_TESTS:
        seed = FIXED_RUN_SEED
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def parse_seed(text):
    """An unsigned 64-bit run seed from the command line."""
    value = int(text)
    if not 0 <= value < 1 << 64:
        raise argparse.ArgumentTypeError("the seed must be an unsigned 64-bit integer")
    return value
