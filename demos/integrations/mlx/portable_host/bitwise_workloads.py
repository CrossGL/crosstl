"""Exact bitwise readbacks through upstream MLX operations and physical layouts."""

from demos.integrations.mlx.portable_host.binary_workloads import LAYOUTS, mlx_operand
from demos.integrations.mlx.portable_host.packages import (
    BITWISE_ENTRIES,
    BITWISE_INVERT_ENTRIES,
    BITWISE_PACKAGE_ENTRIES,
    BOOLEAN_COPY_ENTRY,
    COPY_ENTRY,
)
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
)

OPERATIONS = {
    "BitwiseAnd": "bitwise_and",
    "BitwiseOr": "bitwise_or",
    "BitwiseXor": "bitwise_xor",
    "LeftShift": "left_shift",
    "RightShift": "right_shift",
    "BitwiseInvert": "invert",
}


def cases():
    for entry, dtype in BITWISE_PACKAGE_ENTRIES.items():
        for layout in LAYOUTS:
            yield {
                "entry": entry,
                "dtype": dtype,
                "operation": (
                    "BitwiseInvert"
                    if entry in BITWISE_INVERT_ENTRIES
                    else entry[3 : -len(dtype)]
                ),
                "layout": layout,
            }
    for entry, dtype in BITWISE_PACKAGE_ENTRIES.items():
        yield {
            "entry": entry,
            "dtype": dtype,
            "operation": (
                "BitwiseInvert"
                if entry in BITWISE_INVERT_ENTRIES
                else entry[3 : -len(dtype)]
            ),
            "layout": "batched",
        }


def operands(np, case):
    count = {
        "empty": 0,
        "scalar": 1,
        "vector": 16,
        "tail": 257,
        "matrix": 15,
        "transpose": 15,
        "broadcast": 15,
        "reverse": 17,
        "batched": 65536,
    }[case["layout"]]
    size = max(count, 36)
    words = [
        0,
        1,
        0xFFFFFFFF,
        0x80000000,
        0x7FFFFFFF,
        0xAAAAAAAA,
        0x55555555,
        0x12345678,
    ]
    a = np.asarray([words[i % len(words)] for i in range(size)], dtype="uint32")
    b = np.asarray(
        [words[(i * 3 + 2) % len(words)] for i in range(size)], dtype="uint32"
    )
    if case["operation"] in {"LeftShift", "RightShift"}:
        b = np.asarray(
            [[0, 1, 15, 16, 30, 31][i % 6] for i in range(size)], dtype="uint32"
        )
        if case["operation"] == "LeftShift" and case["dtype"] == "int32":
            # Avoid undefined signed-source overflow; the unsigned cases exercise all bits.
            a = np.asarray([i % 2 for i in range(size)], dtype="uint32")
            b = np.minimum(b, np.uint32(30))
    if case["dtype"] == "int32":
        a, b = a.view("int32"), b.view("int32")
    elif case["dtype"] == "bool_":
        a = np.asarray([bool(i % 2) for i in range(size)])
        b = np.asarray([bool(i % 4 >= 2) for i in range(size)])
    layout = case["layout"]
    if layout == "scalar":
        return a[2:3].reshape(()), b[2:3].reshape(())
    if layout == "matrix":
        return a[:15].reshape(3, 5), b[:15].reshape(3, 5)
    if layout == "transpose":
        return a[:15].reshape(3, 5).T, b[:15].reshape(5, 3)
    if layout == "broadcast":
        return np.broadcast_to(a[:5], (3, 5)), np.broadcast_to(b[:3, None], (3, 5))
    if layout == "reverse":
        return a[34:0:-2], b[1:35:2]
    return a[:count], b[:count]


def reference(np, case):
    a, b = operands(np, case)
    if case["operation"] == "BitwiseInvert":
        b = None
        result = np.invert(a)
    else:
        result = getattr(np, OPERATIONS[case["operation"]])(a, b)
    # A separate integer oracle catches accidental signedness or width changes.
    values = []
    for left, right in zip(a.reshape(-1), [0] * a.size if b is None else b.reshape(-1)):
        left, right = int(left), int(right)
        operation = case["operation"]
        value = {
            "BitwiseAnd": lambda: left & right,
            "BitwiseOr": lambda: left | right,
            "BitwiseXor": lambda: left ^ right,
            "LeftShift": lambda: left << right,
            "RightShift": lambda: left >> right,
            "BitwiseInvert": lambda: ~left,
        }[operation]()
        value &= 0xFFFFFFFF
        if case["dtype"] == "int32" and value >= 0x80000000:
            value -= 0x100000000
        values.append(bool(value) if case["dtype"] == "bool_" else value)
    if result.reshape(-1).tolist() != values:
        raise ValueError("NumPy and integer bitwise references disagree")
    return a, b, result


def dispatches(np, case):
    a, b = operands(np, case)
    if not a.size:
        return []
    copy = BOOLEAN_COPY_ENTRY if case["dtype"] == "bool_" else COPY_ENTRY
    result = []
    if case["entry"] in BITWISE_INVERT_ENTRIES:
        if case["layout"] == "reverse":
            result.append((copy, a.size, [(a.size + 1) // 2, 1, 1]))
        count = a.shape[-1] if case["layout"] == "broadcast" else a.size
        return result + [
            (
                case["entry"],
                min(65535, count - first),
                [min(65535, count - first), 1, 1],
            )
            for first in range(0, count, 65535)
        ]
    for value in (a, b):
        if not value.flags.c_contiguous:
            shape = (1, *value.shape) if value.ndim == 1 else value.shape
            result.append((copy, value.size, [(shape[-1] + 1) // 2, shape[-2], 1]))
    return result + [
        (case["entry"], min(65535, a.size - first), [min(65535, a.size - first), 1, 1])
        for first in range(0, a.size, 65535)
    ]


def stored_values(case, expected):
    if case["entry"] in BITWISE_INVERT_ENTRIES:
        if case["layout"] == "broadcast":
            return expected[0].tolist()
        if case["layout"] == "transpose":
            return expected.ravel(order="F").tolist()
    return expected.reshape(-1).tolist()


def collect(mx, np, *, observe, dispatch_count=None):
    records = []
    for case in cases():
        a, b, expected = reference(np, case)
        left = mlx_operand(mx, np, a)
        right = mlx_operand(mx, np, b) if b is not None else None
        start = dispatch_count() if dispatch_count else 0
        result = (
            mx.bitwise_invert(left)
            if b is None
            else getattr(mx, OPERATIONS[case["operation"]])(left, right)
        )
        actual = np.array(result)
        record = {
            **case,
            "a": a.tolist(),
            "b": b.tolist() if b is not None else None,
            "actual": actual.tolist(),
            "expected": expected.tolist(),
            "resultShape": list(actual.shape),
            "resultDtype": str(result.dtype),
            "inputUnchanged": bool(
                np.array_equal(np.array(left), a)
                and (b is None or np.array_equal(np.array(right), b))
            ),
            "dispatchCount": dispatch_count() - start if dispatch_count else 0,
        }
        records.append(record)
        observe(record)
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not np.array_equal(actual, expected)
        ):
            raise RuntimeError(f"Bitwise numerical mismatch: {case}")
    return records


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if not isinstance(records, list) or len(records) != len(required):
        raise ValueError("Bitwise evidence does not cover every case")
    cursor = 0
    for record, case in zip(records, required):
        a, b, expected = reference(np, case)
        calls = dispatches(np, case) if native else []
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("a") != a.tolist()
            or record.get("b") != (b.tolist() if b is not None else None)
            or record.get("expected") != expected.tolist()
            or record.get("actual") != expected.tolist()
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype")
            != "mlx.core." + case["dtype"].removesuffix("_")
            or record.get("inputUnchanged") is not True
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != len(calls)
        ):
            raise ValueError(
                "Bitwise readback identity, values or dispatch count differ"
            )
        actual = np.asarray(record["actual"], dtype=object).reshape(-1)
        if any(
            type(value) is not (bool if case["dtype"] == "bool_" else int)
            for value in actual
        ):
            raise ValueError(
                "Bitwise readbacks must preserve integer and Boolean types"
            )
        first = 0
        for entry, count, grid in calls:
            if cursor >= len(trace):
                raise ValueError("Bitwise native trace is incomplete")
            event = trace[cursor]
            cursor += 1
            if (
                event.get("entry") != entry
                or event.get("threads") != count
                or event.get("target") not in {"metal", "opengl", "directx"}
                or event.get("dispatchVersion") != DISPATCH_VERSION
                or event.get("workgroupCount") != grid
                or event.get("workgroupSize") != [1, 1, 1]
                or "threadGridSize" in event
            ):
                raise ValueError(
                    "Bitwise native trace has the wrong operation or geometry"
                )
            guard = COPY_GUARD
            if case["dtype"] == "bool_":
                guard = (
                    BOOLEAN_GUARD
                    if event["target"] == "metal"
                    else [int(v) for v in BOOLEAN_GUARD]
                )
            if (
                event.get(
                    "binaryGuardValues"
                    if entry in BITWISE_ENTRIES
                    else (
                        "unaryGuardValues"
                        if entry in BITWISE_INVERT_ENTRIES
                        else "copyGuardWords"
                    )
                )
                != guard
            ):
                raise ValueError("Bitwise native output guard differs")
            if entry in BITWISE_PACKAGE_ENTRIES:
                values = stored_values(case, expected)
                values = values[first : first + count]
                first += count
                if case["dtype"] == "bool_" and event["target"] != "metal":
                    values = [int(value) for value in values]
                if event.get("bitwiseValues") != values:
                    raise ValueError(
                        "Bitwise native values differ from the MLX readback"
                    )
        if native and first != len(stored_values(case, expected)):
            raise ValueError("Bitwise native batches do not cover the output")
    if cursor != len(trace):
        raise ValueError("Bitwise trace contains unexpected dispatches")
