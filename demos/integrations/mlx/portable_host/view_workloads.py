"""Check shared-buffer views and translated unary dispatch through those views."""

from __future__ import annotations

import json


def _record(name, value):
    return {
        "case": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "values": value.reshape(-1).tolist(),
    }


def expected_records():
    import numpy as np

    base = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    flat = np.arange(1, 17, dtype=np.float32)
    integer = np.array([[2**60, 2**60 + 1], [-(2**60), -(2**60) + 1]], dtype=np.int64)
    cases = [
        ("matrix", base, True),
        ("transpose", base.T, True),
        ("reshape", base.reshape(2, 6), True),
        ("expand", base[:, None, :], True),
        ("squeeze", base, True),
        ("copy", base, True),
        ("stop-gradient", base, True),
        ("unflatten", base.reshape(3, 2, 2), True),
        ("broadcast-row", np.broadcast_to(base[0], (3, 4)), True),
        ("broadcast-scalar", np.full((2, 3), 2, dtype=np.float32), True),
        ("offset", flat[3:9], True),
        ("reverse", flat[11::-1], True),
        ("gapped", flat[:12:2], True),
        ("split-a", flat[:2], True),
        ("split-b", flat[2:7], True),
        ("split-c", flat[7:12], True),
        ("empty", np.empty((0, 3), dtype=np.float32), False),
        ("empty-transpose", np.empty((3, 0), dtype=np.float32), False),
        ("depends", base, True),
        ("custom", base, True),
        ("int64-transpose", integer.T, False),
        ("slice-positive", flat[2:14:3], True),
        ("slice-reverse", flat[14:1:-2], True),
        ("slice-matrix", base[::-1, 3:0:-2], True),
        ("slice-nested", flat[1:15:2][::-2], True),
        ("slice-transpose", base.T[1:4:2, ::-1], True),
        ("slice-broadcast", np.broadcast_to(base[0:1], (3, 4))[::-1, ::2], True),
        ("slice-row", base[1:2], True),
        ("slice-empty", flat[4:4], True),
        ("slice-int64", integer[::-1, ::-1], False),
    ]
    records = []
    for name, value, square in cases:
        records.append(_record(name, value))
        if square:
            records.append(_record(name + "/square", value * value))
    records.extend(
        [
            _record("dependency", np.array([3, -2, -5], dtype=np.float32)),
            _record("source-preserved", base),
            _record("offset-source-preserved", flat),
        ]
    )
    return records


def run(mx, np):
    base = mx.array(np.arange(1, 13, dtype=np.float32).reshape(3, 4))
    flat = mx.array(np.arange(1, 17, dtype=np.float32))
    integer = mx.array(
        np.array([[2**60, 2**60 + 1], [-(2**60), -(2**60) + 1]], dtype=np.int64)
    )
    expanded = mx.expand_dims(base, 1)
    splits = mx.split(mx.reshape(base, (12,)), [2, 7])
    empty = mx.array(np.empty((0, 3), dtype=np.float32))
    dependency = mx.negative(mx.array([-3.0, 2.0, 5.0]))

    @mx.custom_function
    def identity(value):
        return value

    cases = [
        ("matrix", base, True),
        ("transpose", mx.transpose(base), True),
        ("reshape", mx.reshape(base, (2, 6)), True),
        ("expand", expanded, True),
        ("squeeze", mx.squeeze(expanded, 1), True),
        ("copy", +base, True),
        ("stop-gradient", mx.stop_gradient(base), True),
        ("unflatten", mx.unflatten(base, 1, (2, 2)), True),
        (
            "broadcast-row",
            mx.broadcast_to(mx.array([[1.0, 2.0, 3.0, 4.0]]), (3, 4)),
            True,
        ),
        ("broadcast-scalar", mx.broadcast_to(mx.array(2.0), (2, 3)), True),
        ("offset", mx.as_strided(flat, (6,), (1,), 3), True),
        ("reverse", mx.as_strided(flat, (12,), (-1,), 11), True),
        ("gapped", mx.as_strided(flat, (6,), (2,)), True),
        ("split-a", splits[0], True),
        ("split-b", splits[1], True),
        ("split-c", splits[2], True),
        ("empty", mx.reshape(empty, (0, 3)), False),
        ("empty-transpose", mx.transpose(empty), False),
        ("depends", mx.depends([base], [dependency])[0], True),
        ("custom", identity(base), True),
        ("int64-transpose", mx.transpose(integer), False),
        ("slice-positive", flat[2:14:3], True),
        ("slice-reverse", flat[14:1:-2], True),
        ("slice-matrix", base[::-1, 3:0:-2], True),
        ("slice-nested", flat[1:15:2][::-2], True),
        ("slice-transpose", mx.transpose(base)[1:4:2, ::-1], True),
        (
            "slice-broadcast",
            mx.broadcast_to(base[0:1], (3, 4))[::-1, ::2],
            True,
        ),
        ("slice-row", base[1:2], True),
        ("slice-empty", flat[4:4], True),
        ("slice-int64", integer[::-1, ::-1], False),
    ]
    records = []
    for name, value, square in cases:
        records.append(_record(name, np.array(value)))
        if square:
            records.append(_record(name + "/square", np.array(mx.square(value))))
    records.extend(
        [
            _record("dependency", np.array(dependency)),
            _record("source-preserved", np.array(base)),
            _record("offset-source-preserved", np.array(flat)),
        ]
    )
    validate(records)
    return records


def validate(records):
    if json.dumps(records, sort_keys=True, allow_nan=False) != json.dumps(
        expected_records(), sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("Incomplete or incorrect shared-buffer view readbacks")


def dispatches():
    # Broadcasts compute stored elements once; metadata reuses the result.
    square = "v_Squarefloat32float32"
    copy = "ggn2_dynamic_copyuint32uint32"
    counts = [12] * 8 + [4, 1, 6]
    return (
        [(square, count) for count in counts]
        + [
            (copy, 12),
            (square, 12),
            (copy, 6),
            (square, 6),
            (square, 2),
            (square, 5),
            (square, 5),
            ("v_Negativefloat32float32", 3),
            (square, 12),
            (square, 12),
        ]
        + [
            entry
            for count in (4, 7, 6, 4, 6, 6)
            for entry in ((copy, count), (square, count))
        ]
        + [(square, 4)]
    )
