"""Check translated casts and mixed-type arithmetic with exact binary32 results."""

import json

from demos.integrations.mlx.portable_host.binary_workloads import LAYOUTS, mlx_operand
from demos.integrations.mlx.portable_host.packages import CAST_ENTRIES, COPY_ENTRY


def operand(source, destination, layout):
    import numpy as np

    if source == "float32":
        pattern = (
            [
                0.0,
                -0.0,
                0.75,
                1.5,
                255.875,
                65535.5,
                16777215.0,
                2147483520.0,
                4294967040.0,
                1.1754942e-38,
                1.4012985e-45,
            ]
            if destination == "uint32"
            else [
                0.0,
                -0.0,
                -0.75,
                0.75,
                -1.5,
                1.5,
                -255.875,
                255.875,
                -2147483648.0,
                2147483520.0,
                -1.1754942e-38,
                1.4012985e-45,
            ]
        )
    elif source == "int32":
        pattern = [
            0,
            1,
            -1,
            16777215,
            16777216,
            16777217,
            16777219,
            -16777217,
            -16777219,
            2147483647,
            -2147483648,
        ]
    else:
        pattern = [
            0,
            1,
            16777215,
            16777216,
            16777217,
            16777219,
            2147483647,
            2147483648,
            4294967295,
        ]
    values = np.resize(np.asarray(pattern, dtype=source), 257)
    if layout == "empty":
        return values[:0]
    if layout == "scalar":
        return values[:1].reshape(())
    if layout == "vector":
        return values[:7]
    if layout == "tail":
        return values
    if layout == "matrix":
        return values[:15].reshape(3, 5)
    if layout == "transpose":
        return values[:15].reshape(3, 5).T
    if layout == "broadcast":
        return np.broadcast_to(values[:5], (3, 5))
    if layout == "reverse":
        return values[34:0:-2]
    raise ValueError(f"Unknown cast layout: {layout}")


def record(entry, layout, source, result):
    import numpy as np

    return {
        "entry": entry,
        "layout": layout,
        "sourceType": str(source.dtype),
        "dtype": str(result.dtype),
        "shape": list(result.shape),
        "sourceWords": (
            np.ascontiguousarray(source).reshape(-1).view(np.uint32).tolist()
        ),
        "words": np.ascontiguousarray(result).reshape(-1).view(np.uint32).tolist(),
    }


def expected_records():
    import numpy as np

    records = []
    for entry, (source, destination) in CAST_ENTRIES.items():
        for layout in LAYOUTS:
            value = operand(source, destination, layout)
            records.append(record(entry, layout, value, value.astype(destination)))
    for source in ("int32", "uint32"):
        value = np.array([0, 1, 3, 5, 7, 9, 11], dtype=source)
        records.append(
            record(
                "mixed-add", source, value, value.astype("float32") + np.float32(0.5)
            )
        )
    return records


def run(mx, np):
    records = []
    for entry, (source, destination) in CAST_ENTRIES.items():
        for layout in LAYOUTS:
            value = mlx_operand(mx, np, operand(source, destination, layout))
            result = np.array(value.astype(getattr(mx, destination)))
            records.append(record(entry, layout, np.array(value), result))
    for source in ("int32", "uint32"):
        value = mx.array(np.array([0, 1, 3, 5, 7, 9, 11], dtype=source))
        other = mx.array(np.full(7, 0.5, dtype=np.float32))
        result = np.array(mx.add(value, other))
        records.append(record("mixed-add", source, np.array(value), result))
    return records


def validate(records):
    if json.dumps(records, sort_keys=True, allow_nan=False) != json.dumps(
        expected_records(), sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("Incomplete or incorrect cast readbacks")


def dispatches():
    result = []
    for entry, (source, destination) in CAST_ENTRIES.items():
        for layout in LAYOUTS:
            value = operand(source, destination, layout)
            if not value.size:
                continue
            if not value.flags.c_contiguous:
                result.append((COPY_ENTRY, value.size))
            result.append((entry, value.size))
    for source in ("int32", "uint32"):
        result.extend([(f"v_copy{source}float32", 7), ("vv_Addfloat32", 7)])
    return result
