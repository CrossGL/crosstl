"""Check array filling through translated copies, including storage payloads."""

import json
import math

from demos.integrations.mlx.portable_host.copy_workloads import record, source_words
from demos.integrations.mlx.portable_host.packages import COPY_ENTRY

DTYPES = ("float32", "int32", "uint32")
CASES = (
    ("scalar", (), (), 1, (), "full"),
    ("fill-vector", (), (), 8, (257,), "full"),
    ("fill-matrix", (), (), 3, (3, 5), "full"),
    ("row-broadcast", (5,), (2,), 0, (3, 5), "full"),
    ("column-broadcast", (3, 1), (5, 0), 1, (3, 5), "full"),
    ("tile-broadcast", (2, 1, 5), (13, 0, 2), 2, (2, 3, 5), "full"),
    ("transpose", (5, 3), (1, 5), 0, (5, 3), "full"),
    ("reverse", (17,), (-2,), 38, (17,), "full"),
    ("negative-4d", (2, 3, 4, 5), (100, 7, -21, -1), 80, (2, 3, 4, 5), "full"),
    ("like-broadcast", (5,), (-2,), 12, (3, 5), "full_like"),
    ("empty", (), (), 1, (0, 3), "full"),
)
CONSTANT_SHAPES = ((), (0, 3), (7,), (3, 5))


def expected_records():
    import numpy as np

    words = source_words()
    records = []
    for dtype in DTYPES:
        for name, shape, strides, offset, output, _ in CASES:
            view = np.ndarray(
                shape,
                dtype=dtype,
                buffer=words,
                offset=offset * 4,
                strides=tuple(stride * 4 for stride in strides),
            )
            records.append(record(dtype + "/" + name, np.broadcast_to(view, output)))
        for operation in ("zeros", "ones"):
            for index, shape in enumerate(CONSTANT_SHAPES):
                records.append(
                    record(
                        f"{dtype}/{operation}-{index}",
                        getattr(np, operation)(shape, dtype=dtype),
                    )
                )
        records.append(record(dtype + "/source", words.view(dtype)))
    records.append(
        record(
            "mixed-full", np.broadcast_to(np.array([1, 3, 5], dtype=np.float32), (2, 3))
        )
    )
    records.append(
        record(
            "array-fill-dtype",
            np.broadcast_to(np.array([1, 3, 5], dtype=np.int32), (2, 3)),
        )
    )
    return records


def run(mx, np):
    words = source_words()
    records = []
    for dtype in DTYPES:
        base = mx.array(words.view(dtype))
        for name, shape, strides, offset, output, operation in CASES:
            value = mx.as_strided(base, shape, strides, offset)
            if operation == "full_like":
                like = mx.as_strided(base, output, (1, 3))
                result = mx.full_like(like, value)
            else:
                result = mx.full(output, value)
            records.append(record(dtype + "/" + name, np.array(result)))
        for operation in ("zeros", "ones"):
            for index, shape in enumerate(CONSTANT_SHAPES):
                result = getattr(mx, operation)(shape, dtype=getattr(mx, dtype))
                records.append(record(f"{dtype}/{operation}-{index}", np.array(result)))
        records.append(record(dtype + "/source", np.array(base)))
    integers = mx.array([1, 3, 5], dtype=mx.int32)
    promoted = mx.broadcast_to(integers, (2, 3)).astype(mx.float32)
    records.append(record("mixed-full", np.array(mx.full((2, 3), promoted))))
    # The pinned Python binding retains the dtype of an array-valued fill.
    retained = mx.full((2, 3), integers, dtype=mx.float32)
    records.append(record("array-fill-dtype", np.array(retained)))
    return records


def validate(records):
    if json.dumps(records, sort_keys=True, allow_nan=False) != json.dumps(
        expected_records(), sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("Incomplete or incorrect Full readbacks")


def dispatches():
    sequence = []
    for _ in DTYPES:
        sequence.extend(
            (COPY_ENTRY, math.prod(output))
            for _, _, _, _, output, _ in CASES
            if math.prod(output)
        )
        sequence.extend(
            (COPY_ENTRY, math.prod(shape))
            for _ in ("zeros", "ones")
            for shape in CONSTANT_SHAPES
            if math.prod(shape)
        )
    sequence.extend(
        [(COPY_ENTRY, 6), ("v_copyint32float32", 6), (COPY_ENTRY, 6), (COPY_ENTRY, 6)]
    )
    return sequence
