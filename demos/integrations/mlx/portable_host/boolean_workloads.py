"""Check Boolean storage, comparisons and logical operations through MLX."""

import json

from demos.integrations.mlx.portable_host.binary_workloads import LAYOUTS, mlx_operand
from demos.integrations.mlx.portable_host.packages import (
    BOOLEAN_CAST_ENTRIES,
    BOOLEAN_COPY_ENTRY,
    COMPARISON_ENTRIES,
    COPY_ENTRY,
    LOGICAL_NOT_ENTRY,
)

OPERATIONS = {
    "Equal": "equal",
    "NaNEqual": "equal",
    "NotEqual": "not_equal",
    "Less": "less",
    "LessEqual": "less_equal",
    "Greater": "greater",
    "GreaterEqual": "greater_equal",
    "LogicalAnd": "logical_and",
    "LogicalOr": "logical_or",
}


def operand(dtype, layout, *, second=False):
    import numpy as np

    if layout in {"nan-scalar", "nan-other-scalar"}:
        return np.array(
            0.0 if second and layout == "nan-other-scalar" else float("nan"),
            dtype=dtype,
        )
    if dtype == "bool_":
        pattern = (
            [False, True, True, False, True, False, False]
            if second
            else [True, False, True, False, False, True, True]
        )
    elif second:
        pattern = [0, 1, 3, 2, 7, 0, 1]
    elif dtype == "uint32":
        pattern = [0, 1, 2, 3, 0, 2147483647, 1]
    else:
        pattern = [0, -1, 2, -3, 0, 1, 2147483647]
    if layout == "nonfinite":
        pattern = [float("nan"), 0.0, -0.0, float("inf"), float("-inf"), 1.0, -1.0]
        if second:
            pattern = [float("nan"), -0.0, 0.0, float("inf"), 0.0, -1.0, 1.0]
        return np.array(pattern, dtype=dtype)
    values = np.resize(np.array(pattern, dtype=dtype), 257)
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
    raise ValueError(f"Unknown Boolean layout: {layout}")


def definitions():
    for entry, dtype in COMPARISON_ENTRIES.items():
        operation = entry[3 : -len(dtype)]
        layouts = (
            ("scalar", "nan-scalar", "nan-other-scalar")
            if operation == "NaNEqual"
            else (*LAYOUTS, *(("nonfinite",) if dtype == "float32" else ()))
        )
        for layout in layouts:
            yield entry, operation, dtype, "bool_", layout
    for entry, (source, destination) in BOOLEAN_CAST_ENTRIES.items():
        for layout in (*LAYOUTS, *(("nonfinite",) if source == "float32" else ())):
            yield entry, "cast", source, destination, layout
    for operation, entry in (
        ("not", LOGICAL_NOT_ENTRY),
        ("copy", BOOLEAN_COPY_ENTRY),
        ("full", BOOLEAN_COPY_ENTRY),
    ):
        for layout in LAYOUTS:
            yield entry, operation, "bool_", "bool_", layout


def record(entry, operation, layout, a, b, result):
    import numpy as np

    def raw(value):
        return np.ascontiguousarray(value).reshape(-1).view(np.uint8).tolist()

    return {
        "entry": entry,
        "operation": operation,
        "layout": layout,
        "sourceType": str(a.dtype),
        "dtype": str(result.dtype),
        "shape": list(result.shape),
        "aBytes": raw(a),
        "bBytes": raw(b),
        "bytes": raw(result),
    }


def expected_records():
    import numpy as np

    records = []
    for entry, operation, source, destination, layout in definitions():
        a, b = operand(source, layout), operand(source, layout, second=True)
        if operation in {"copy", "full", "cast"}:
            result = a.astype(destination)
        elif operation == "not":
            result = np.logical_not(a)
        elif operation == "NaNEqual":
            result = np.equal(a, b) | (np.isnan(a) & np.isnan(b))
        else:
            result = getattr(np, OPERATIONS[operation])(a, b)
        records.append(record(entry, operation, layout, a, b, result))
    return records


def run(mx, np):
    records = []
    for entry, operation, source, destination, layout in definitions():
        a = mlx_operand(mx, np, operand(source, layout))
        b = mlx_operand(mx, np, operand(source, layout, second=True))
        if operation == "cast":
            result = a.astype(getattr(mx, destination))
        elif operation == "not":
            result = mx.logical_not(a)
        elif operation == "copy":
            result = mx.contiguous(a)
        elif operation == "full":
            result = mx.full(a.shape, a)
        elif operation == "NaNEqual":
            result = mx.array_equal(a, b, equal_nan=True)
        else:
            result = getattr(mx, OPERATIONS[operation])(a, b)
        actual = np.array(result)
        records.append(
            record(entry, operation, layout, np.array(a), np.array(b), actual)
        )
    return records


def validate(records):
    if json.dumps(records, sort_keys=True, allow_nan=False) != json.dumps(
        expected_records(), sort_keys=True, allow_nan=False
    ):
        raise RuntimeError("Incomplete or incorrect Boolean readbacks")


def dispatches():
    result = []
    for entry, operation, source, _, layout in definitions():
        value = operand(source, layout)
        if not value.size:
            continue
        if operation == "not" and layout in {"transpose", "broadcast"}:
            result.append((entry, 5 if layout == "broadcast" else value.size))
            continue
        contiguous = value.flags.c_contiguous
        copy = BOOLEAN_COPY_ENTRY if source == "bool_" else COPY_ENTRY
        if operation == "copy":
            if not contiguous:
                result.append((copy, value.size))
            continue
        if operation != "full" and not contiguous:
            result.extend(
                [(copy, value.size)] * (2 if entry in COMPARISON_ENTRIES else 1)
            )
        result.append((entry, value.size))
    return result
