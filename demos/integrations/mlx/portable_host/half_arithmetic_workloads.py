"""Binary16 arithmetic and comparisons through the public MLX array API."""

from demos.integrations.mlx.portable_host.binary_workloads import (
    LAYOUTS,
    mlx_operand,
    operands,
)
from demos.integrations.mlx.portable_host.half_workloads import widen_reference
from demos.integrations.mlx.portable_host.packages import (
    HALF_ARITHMETIC_ENTRIES,
    HALF_COPY_ENTRY,
)

OPERATIONS = {
    "Add": "add",
    "Subtract": "subtract",
    "Multiply": "multiply",
    "Divide": "divide",
    "Minimum": "minimum",
    "Maximum": "maximum",
    "Equal": "equal",
    "NotEqual": "not_equal",
    "Less": "less",
    "LessEqual": "less_equal",
    "Greater": "greater",
    "GreaterEqual": "greater_equal",
    "Abs": "abs",
    "NaNEqual": "array_equal",
}


def cases():
    return [
        {"entry": entry, "operation": operation, "layout": layout}
        for entry in HALF_ARITHMETIC_ENTRIES
        for operation in ["Abs" if entry.startswith("v_") else entry[3:-7]]
        for layout in (
            ("scalar", "nan-scalar", "nan-other-scalar")
            if operation == "NaNEqual"
            else (*LAYOUTS, "boundary")
        )
    ]


def inputs(np, case):
    layout = case["layout"]
    if layout in {"nan-scalar", "nan-other-scalar"}:
        return np.asarray(np.nan, dtype=np.float16), np.asarray(
            np.nan if layout == "nan-scalar" else 1, dtype=np.float16
        )
    if layout == "boundary":
        a = np.asarray(
            [
                0,
                0x8000,
                1,
                0x8001,
                0x3FF,
                0x400,
                0x3C01,
                0xBC01,
                0x7BFF,
                0xFBFF,
                0x3555,
                0xB555,
                0x3BFF,
                0x4001,
            ],
            dtype=np.uint16,
        ).view(np.float16)
        b = np.asarray(
            [1, -1, 2, 2, 0.5, 0.5, 1.5, 1.5, 2, 2, 3, 3, 1, 1], dtype=np.float16
        )
        if case["operation"] in {
            "Equal",
            "NotEqual",
            "Less",
            "LessEqual",
            "Greater",
            "GreaterEqual",
        }:
            a = np.concatenate(
                (a, np.asarray([np.nan, np.inf, -np.inf, 0, -0.0], dtype=np.float16))
            )
            b = np.concatenate(
                (b, np.asarray([np.nan, np.inf, -np.inf, -0.0, 0], dtype=np.float16))
            )
        return a, b
    return operands("float16", layout)


def reference(np, case, a, b):
    operation = case["operation"]
    with np.errstate(all="ignore"):
        if operation in {"Minimum", "Maximum"}:
            comparison = np.less if operation == "Minimum" else np.greater
            return np.where(np.isnan(a) | comparison(a, b), a, b)
        if operation == "NaNEqual":
            return np.asarray(np.equal(a, b) | (np.isnan(a) & np.isnan(b)))
        if operation == "Abs":
            return np.abs(a)
        return getattr(np, OPERATIONS[operation])(a, b)


def words(np, array):
    flat = np.ascontiguousarray(array).reshape(-1)
    return (
        flat.astype(np.uint8).tolist()
        if array.dtype == np.bool_
        else flat.view(np.uint16).tolist()
    )


def sequence(case, a, b):
    if not a.size:
        return []
    if case["operation"] == "Abs" and case["layout"] in {"transpose", "broadcast"}:
        return [case["entry"]]
    sources = (a,) if case["operation"] == "Abs" else (a, b)
    return [HALF_COPY_ENTRY for value in sources if not value.flags.c_contiguous] + [
        case["entry"]
    ]


def native_result(case, expected):
    # Unary MLX outputs retain contiguous column-major and broadcast storage.
    if case["operation"] == "Abs":
        if case["layout"] == "transpose":
            return expected.T
        if case["layout"] == "broadcast":
            return expected[0]
    return expected


def record(np, case, a, b, result, count):
    return {
        **case,
        "aWords": words(np, a),
        "bWords": words(np, b),
        "resultWords": words(np, result),
        "shape": list(result.shape),
        "dtype": result.dtype.name,
        "dispatchCount": count,
    }


def run(mx, np, host, save):
    records = []
    for case in cases():
        a, b = inputs(np, case)
        left, right = mlx_operand(mx, np, a), mlx_operand(mx, np, b)
        start = host.dispatch_count if host else 0
        if case["operation"] == "Abs":
            result = mx.abs(left)
        elif case["operation"] == "NaNEqual":
            result = mx.array_equal(left, right, equal_nan=True)
        else:
            result = getattr(mx, OPERATIONS[case["operation"]])(left, right)
        observed = np.array(result)
        records.append(
            record(
                np,
                case,
                np.array(left),
                np.array(right),
                observed,
                host.dispatch_count - start if host else 0,
            )
        )
        save(records)
    return records


def validate(records, trace, *, native):
    import numpy as np

    if not isinstance(records, list) or len(records) != len(cases()):
        raise ValueError("Half arithmetic workload inventory is incomplete")
    cursor = 0
    for item, case in zip(records, cases()):
        a, b = inputs(np, case)
        expected = reference(np, case, a, b)
        entries = sequence(case, a, b) if native else []
        if item != record(np, case, a, b, expected, len(entries)):
            raise ValueError(f"Half arithmetic result differs: {case}")
        events = trace[cursor : cursor + len(entries)]
        cursor += len(entries)
        if [event.get("entry") for event in events] != entries:
            raise ValueError("Half arithmetic dispatch sequence differs")
        if not events:
            continue
        event = events[-1]
        stored = native_result(case, expected)
        bits = words(np, stored)
        boolean = expected.dtype == np.bool_
        guard = (
            [int(index % 2 == 0) for index in range(32)] if boolean else [0x3555] * 32
        )
        target = event["target"]
        physical, physical_guard = bits, guard
        storage = "float16"
        encoding = "ieee754-binary16"
        if boolean:
            storage = "bool" if target == "metal" else "uint32"
            encoding = None
            if target == "metal":
                physical, physical_guard = list(map(bool, bits)), list(map(bool, guard))
        elif target == "opengl":
            storage, encoding = "float32", "ieee754-binary32"
            physical = list(map(widen_reference, bits))
            physical_guard = list(map(widen_reference, guard))
        if (
            event.get("halfStorage") != {
                "logicalType": "bool_" if boolean else "float16",
                "physicalType": storage,
                "encoding": encoding,
                "values": physical,
                "guardValues": physical_guard,
                "logicalWords": bits,
            }
            or event.get("threads") != stored.size
            or event.get("workgroupCount") != [stored.size, 1, 1]
            or event.get("workgroupSize") != [1, 1, 1]
            or event.get("dispatchVersion") != 3
        ):
            raise ValueError("Half arithmetic native readback or dispatch differs")
    if cursor != len(trace):
        raise ValueError("Half arithmetic has unexpected native dispatches")
