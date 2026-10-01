"""Exercise translated binary arithmetic through MLX layouts and copy dispatch."""

import math

from demos.integrations.mlx.portable_host.packages import BINARY_ENTRIES, COPY_ENTRY
from demos.integrations.mlx.portable_host.runtime import wire_value

OPERATIONS = {
    "Add": "add",
    "Subtract": "subtract",
    "Multiply": "multiply",
    "Minimum": "minimum",
    "Maximum": "maximum",
    "Divide": "divide",
}
LAYOUTS = (
    "empty",
    "scalar",
    "vector",
    "tail",
    "matrix",
    "transpose",
    "broadcast",
    "reverse",
)


def operands(dtype, layout):
    import numpy as np

    if layout == "nonfinite":
        return np.array(
            [-np.inf, np.inf, np.nan, 0.0, -0.0, 1.0, -1.0], dtype=np.float32
        ), np.array([1.0, np.inf, 2.0, -0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    count = {
        "empty": 0,
        "scalar": 1,
        "vector": 7,
        "tail": 257,
        "matrix": 15,
        "transpose": 15,
        "broadcast": 15,
        "reverse": 17,
    }[layout]
    a = np.asarray(
        [
            (i * 7 % 29) - (0 if dtype == "uint32" else 14)
            for i in range(max(count, 36))
        ],
        dtype=dtype,
    )
    b = np.asarray([i % 7 + 1 for i in range(max(count, 36))], dtype=dtype)
    if dtype == "float32":
        a *= np.float32(0.25)
        b *= np.float32(0.5)
    if layout == "scalar":
        return a[:1].reshape(()), b[:1].reshape(())
    if layout == "matrix":
        return a[:15].reshape(3, 5), b[:15].reshape(3, 5)
    if layout == "transpose":
        return a[:15].reshape(3, 5).T, b[:15].reshape(5, 3)
    if layout == "broadcast":
        return np.broadcast_to(a[:5], (3, 5)), np.broadcast_to(b[:3, None], (3, 5))
    if layout == "reverse":
        return a[34:0:-2], b[1:35:2]
    return a[:count], b[:count]


def definitions():
    for entry, dtype in BINARY_ENTRIES.items():
        operation = entry[3 : -len(dtype)]
        for layout in (*LAYOUTS, *(("nonfinite",) if dtype == "float32" else ())):
            yield entry, dtype, operation, layout


def record(entry, layout, a, b, result):
    import numpy as np

    return {
        "entry": entry,
        "layout": layout,
        "dtype": str(result.dtype),
        "shape": list(result.shape),
        "aWords": np.ascontiguousarray(a).reshape(-1).view(np.uint32).tolist(),
        "bWords": np.ascontiguousarray(b).reshape(-1).view(np.uint32).tolist(),
        "a": [
            wire_value(float(x) if result.dtype.kind == "f" else int(x))
            for x in a.reshape(-1)
        ],
        "b": [
            wire_value(float(x) if result.dtype.kind == "f" else int(x))
            for x in b.reshape(-1)
        ],
        "values": [
            wire_value(float(x) if result.dtype.kind == "f" else int(x))
            for x in result.reshape(-1)
        ],
    }


def expected_records():
    import numpy as np

    records = []
    for entry, dtype, operation, layout in definitions():
        a, b = operands(dtype, layout)
        with np.errstate(all="ignore"):
            if dtype == "float32" and operation in {"Minimum", "Maximum"}:
                # MLX selects the second operand on ties, including signed zero.
                comparison = np.less if operation == "Minimum" else np.greater
                result = np.where(np.isnan(a) | comparison(a, b), a, b)
            else:
                result = getattr(np, OPERATIONS[operation])(a, b)
        records.append(record(entry, layout, a, b, result))
    return records


def mlx_operand(mx, np, value):
    # Keep the source physical layout; passing a NumPy view directly would copy it.
    if value.size == 0 or value.ndim == 0 or value.flags.c_contiguous:
        return mx.array(value)
    base = value
    while isinstance(base.base, np.ndarray):
        base = base.base
    offset = (value.ctypes.data - base.ctypes.data) // value.itemsize
    source = mx.array(base.reshape(-1))
    return mx.as_strided(
        source,
        value.shape,
        tuple(stride // value.itemsize for stride in value.strides),
        offset,
    )


def run(mx, np):
    records = []
    for entry, dtype, operation, layout in definitions():
        a, b = operands(dtype, layout)
        left, right = mlx_operand(mx, np, a), mlx_operand(mx, np, b)
        result = np.array(getattr(mx, OPERATIONS[operation])(left, right))
        records.append(record(entry, layout, np.array(left), np.array(right), result))
    return records


def validate(records):
    expected = expected_records()
    if not isinstance(records, list) or len(records) != len(expected):
        raise RuntimeError("Incomplete binary readbacks")
    for got, want in zip(records, expected):
        if (
            not isinstance(got, dict)
            or set(got) != set(want)
            or any(got[key] != want[key] for key in want if key != "values")
            or len(got["values"]) != len(want["values"])
        ):
            raise RuntimeError("Binary readback identity changed")
        for index, (actual, reference) in enumerate(zip(got["values"], want["values"])):
            context = f"{want['entry']} [{want['layout']}] element {index}: {actual!r}, expected {reference!r}"
            if want["dtype"] != "float32":
                if type(actual) is not int or actual != reference:
                    raise RuntimeError(f"Incorrect integer binary result: {context}")
            elif isinstance(reference, str):
                if actual != reference:
                    raise RuntimeError(f"Incorrect nonfinite binary result: {context}")
            elif (
                type(actual) not in {int, float}
                or not math.isfinite(actual)
                or (
                    reference == 0
                    and (
                        actual != 0
                        or math.copysign(1, actual) != math.copysign(1, reference)
                    )
                )
                or abs(actual - reference) > 1e-6 + 2e-6 * abs(reference)
            ):
                raise RuntimeError(f"Incorrect float32 binary result: {context}")


def dispatches():
    result = []
    for entry, dtype, _, layout in definitions():
        a, b = operands(dtype, layout)
        if not a.size:
            continue
        for value in (a, b):
            if not value.flags.c_contiguous:
                result.append((COPY_ENTRY, value.size))
        result.append((entry, a.size))
    return result
