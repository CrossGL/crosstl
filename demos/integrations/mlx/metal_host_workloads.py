"""Exercise translated complex-power entries through ordinary MLX array operations."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--dataset",
        choices=("ordinary", "positive-zero", "negative-zero"),
        required=True,
    )
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    offsets = np.arange(512, dtype=np.float32)
    a = (0.75 + offsets * 0.005 + 1j * (-0.25 + offsets * 0.002)).astype(np.complex64)
    b = (-0.3 + offsets * 0.001 + 1j * (0.2 - offsets * 0.0005)).astype(np.complex64)
    if args.dataset != "ordinary":
        # Assign components separately so construction cannot erase negative zero.
        a.real = -4.0 - offsets * 0.005
        a.imag = -0.0 if args.dataset == "negative-zero" else 0.0
        b.real = 0.5 + (offsets % 3) * 0.125
        b.imag = 0.0
    cases = [
        ("ss", a[0], b[0]),
        ("sv", a[0], b[:7]),
        ("vs", a[:7], b[0]),
        ("vv", a[:7], b[:7]),
    ]
    results = []

    def execute(name, lhs, rhs, a_view, b_view):
        expected = np.power(a_view, b_view)
        actual = np.asarray(mx.power(lhs, rhs))
        bound = 5e-5 * np.maximum(1.0, np.abs(expected))
        error = np.abs(actual - expected)
        matched = (
            actual.shape == expected.shape
            and actual.dtype == np.complex64
            and np.all(np.isfinite(actual))
            and np.all(error <= bound)
        )

        def pairs(values):
            return [[float(v.real), float(v.imag)] for v in values.ravel()]

        results.append(
            {
                "dataset": args.dataset,
                "shape": name,
                "arrayShape": list(actual.shape),
                "dtype": str(actual.dtype),
                "outputs": int(actual.size),
                "base": pairs(np.broadcast_to(a_view, expected.shape)),
                "exponent": pairs(np.broadcast_to(b_view, expected.shape)),
                "expected": pairs(expected),
                "actual": pairs(actual),
                "maximumError": float(np.max(error)),
                "matched": bool(matched),
            }
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if not matched:
            raise AssertionError(
                f"MLX host workload {args.dataset}/{name} does not match the reference"
            )

    for name, lhs, rhs in cases:
        execute(name, mx.array(lhs), mx.array(rhs), lhs, rhs)
    for name, shape, a_strides, b_strides in (
        ("g1", (7,), (2,), (3,)),
        ("g2", (3, 5), (11, 2), (0, 1)),
        ("g3", (2, 3, 5), (50, 13, 2), (50, 0, 1)),
        ("gn2", (2, 2, 3, 5), (120, 50, 13, 2), (130, 51, 0, 1)),
    ):
        for strides in (a_strides, b_strides):
            assert sum(
                (size - 1) * stride for size, stride in zip(shape, strides)
            ) < len(a)
        lhs = mx.as_strided(mx.array(a), shape=shape, strides=a_strides)
        rhs = mx.as_strided(mx.array(b), shape=shape, strides=b_strides)
        a_view = np.lib.stride_tricks.as_strided(
            a, shape=shape, strides=tuple(s * a.itemsize for s in a_strides)
        )
        b_view = np.lib.stride_tricks.as_strided(
            b, shape=shape, strides=tuple(s * b.itemsize for s in b_strides)
        )
        execute(name, lhs, rhs, a_view, b_view)


if __name__ == "__main__":
    main()
