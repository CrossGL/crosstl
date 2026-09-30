"""Exercise translated complex-power entries through ordinary MLX array operations."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    mx.set_default_device(mx.gpu)
    offsets = np.arange(512, dtype=np.float32)
    a = (0.75 + offsets * 0.005 + 1j * (-0.25 + offsets * 0.002)).astype(np.complex64)
    b = (-0.3 + offsets * 0.001 + 1j * (0.2 - offsets * 0.0005)).astype(np.complex64)
    cases = [
        ("ss", a[0], b[0]),
        ("sv", a[0], b[:7]),
        ("vs", a[:7], b[0]),
        ("vv", a[:7], b[:7]),
    ]
    results = []

    def execute(name, lhs, rhs, expected):
        actual = np.asarray(mx.power(lhs, rhs))
        bound = 5e-5 * np.maximum(1.0, np.abs(expected))
        error = np.abs(actual - expected)
        if (
            actual.shape != expected.shape
            or not np.all(np.isfinite(actual))
            or not np.all(error <= bound)
        ):
            raise AssertionError(
                f"MLX host workload {name} does not match the reference"
            )
        results.append(
            {
                "shape": name,
                "outputs": int(actual.size),
                "maximumError": float(np.max(error)),
                "matched": True,
            }
        )

    for name, lhs, rhs in cases:
        execute(name, mx.array(lhs), mx.array(rhs), np.power(lhs, rhs))
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
        execute(name, lhs, rhs, np.power(a_view, b_view))
    Path(sys.argv[1]).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
