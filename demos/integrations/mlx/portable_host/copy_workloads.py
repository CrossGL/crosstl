"""Check layout-changing host copies with independent storage-word references."""

import json
import math

from demos.integrations.mlx.portable_host.packages import COPY_ENTRY

CASES = (
    ("transpose", (5, 3), (1, 5), 0, "contiguous"),
    ("reverse", (17,), (-2,), 38, "contiguous"),
    ("gapped", (7,), (3,), 2, "contiguous"),
    ("broadcast", (3, 5), (0, 2), 0, "contiguous"),
    ("broadcast-scalar", (2, 3), (0, 0), 1, "contiguous"),
    ("permute", (3, 5, 7), (5, 1, 15), 0, "contiguous"),
    ("reverse-4d", (2, 3, 4, 5), (100, 7, -21, -1), 80, "contiguous"),
    ("reshape-transpose", (5, 3), (1, 5), 0, "reshape"),
    ("flatten-transpose", (5, 3), (1, 5), 0, "flatten"),
    ("empty", (0, 3), (1, 0), 0, "contiguous"),
)
DTYPES = ("float32", "int32", "uint32")


def source_words():
    import numpy as np

    words = np.arange(257, dtype=np.uint32) * np.uint32(2654435761)
    words[:12] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0xFFC12345,
        0x7F812345,
        0xFF812345,
    ]
    return words


def record(name, value):
    import numpy as np

    return {
        "case": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "words": np.ascontiguousarray(value).reshape(-1).view(np.uint32).tolist(),
    }


def expected_records():
    import numpy as np

    words = source_words()
    records = []
    for dtype in DTYPES:
        for name, shape, strides, offset, operation in CASES:
            view = np.ndarray(
                shape,
                dtype=dtype,
                buffer=words,
                offset=offset * 4,
                strides=tuple(stride * 4 for stride in strides),
            )
            expected = view if operation == "contiguous" else view.reshape(-1)
            records.append(record(dtype + "/" + name, expected))
        records.append(record(dtype + "/source", words.view(dtype)))
    return records


def run(mx, np):
    words = source_words()
    records = []
    for dtype in DTYPES:
        base = mx.array(words.view(dtype))
        for name, shape, strides, offset, operation in CASES:
            view = mx.as_strided(base, shape, strides, offset)
            if operation == "reshape":
                copied = mx.reshape(view, (math.prod(shape),))
            elif operation == "flatten":
                copied = mx.flatten(view)
            else:
                copied = mx.contiguous(view)
            records.append(record(dtype + "/" + name, np.array(copied)))
        records.append(record(dtype + "/source", np.array(base)))
    validate(records)
    return records


def validate(records):
    if json.dumps(records, sort_keys=True, allow_nan=False) != json.dumps(
        expected_records(), sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("Incomplete or incorrect layout-copy readbacks")


def dispatches():
    return [
        (COPY_ENTRY, math.prod(shape))
        for _ in DTYPES
        for _, shape, _, _, _ in CASES
        if math.prod(shape)
    ]
