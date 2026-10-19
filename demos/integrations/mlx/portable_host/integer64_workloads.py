"""Exact 64-bit values through translated MLX copies, casts and arithmetic."""

from demos.integrations.mlx.portable_host.absolute_workloads import physical
from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.packages import (
    INTEGER64_ABSOLUTE_ENTRIES,
    INTEGER64_BINARY_ENTRIES,
    INTEGER64_CAST_ENTRIES,
    INTEGER64_COMPARISON_ENTRIES,
    INTEGER64_COPY_ENTRIES,
    INTEGER64_ENTRIES,
)
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
)

OPERATIONS = {
    "Add": "add",
    "Subtract": "subtract",
    "Multiply": "multiply",
    "Minimum": "minimum",
    "Maximum": "maximum",
    "Equal": "equal",
    "NotEqual": "not_equal",
    "Less": "less",
    "LessEqual": "less_equal",
    "Greater": "greater",
    "GreaterEqual": "greater_equal",
}
LAYOUTS = (
    "empty",
    "scalar",
    "vector",
    "tail",
    "matrix",
    "transpose",
    "reverse",
    "broadcast",
)
FILL_LAYOUTS = ("full-scalar", "full-broadcast", "zeros", "ones")


def cases():
    for entry in INTEGER64_ENTRIES:
        for layout in LAYOUTS:
            yield {"entry": entry, "layout": layout}
    for entry in INTEGER64_COPY_ENTRIES:
        for layout in FILL_LAYOUTS:
            yield {"entry": entry, "layout": layout}


def layout(np, data, name):
    base = np.tile(data, 40)
    if name == "empty":
        return base[:0]
    if name == "scalar":
        return base[:1].reshape(())
    if name == "vector":
        return base[: len(data)]
    if name == "tail":
        return base[:257]
    if name == "matrix":
        return base[:15].reshape(3, 5)
    if name == "transpose":
        return base[:15].reshape(3, 5).T
    if name == "reverse":
        return base[34:0:-2]
    if name == "broadcast":
        return np.broadcast_to(base[:5], (3, 5))
    raise ValueError("Unknown integer64 layout")


def inputs(np, case):
    entry = case["entry"]
    if case["layout"] in FILL_LAYOUTS:
        dtype = INTEGER64_COPY_ENTRIES[entry]
        value = -(2**53) - 1 if dtype == "int64" else 2**63 + 1
        if case["layout"] == "full-broadcast":
            return [np.asarray([value + index for index in range(5)], dtype=dtype)]
        if case["layout"] in {"zeros", "ones"}:
            value = int(case["layout"] == "ones")
        return [np.asarray(value, dtype=dtype)]
    if entry in INTEGER64_CAST_ENTRIES:
        source, destination = INTEGER64_CAST_ENTRIES[entry]
    else:
        source = INTEGER64_ENTRIES[entry]
        destination = source
    data = {
        "int64": [
            -(2**63),
            -(2**63) + 1,
            -(2**53) - 1,
            -(2**32) - 1,
            -1,
            0,
            1,
            2**53 + 1,
            2**63 - 1,
        ],
        "uint64": [
            0,
            1,
            2**32 - 1,
            2**32 + 1,
            2**53 + 1,
            2**63 - 1,
            2**63,
            2**64 - 2,
            2**64 - 1,
        ],
        "int32": [-(2**31), -65535, -1, 0, 1, 65535, 2**31 - 1],
        "uint32": [0, 1, 65535, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1],
        "bool_": [False, True, False, True, True, False, True],
        "float32": [0.0, 1.75, 17.5, 65535.5, float(2**32), float(2**40), float(2**62)],
    }[source]
    if source == "float32" and destination == "int64":
        data = [-float(2**62), -65535.5, -1.75, *data]
    if entry in INTEGER64_BINARY_ENTRIES:
        # Keep signed arithmetic in range; still expose lost high bits.
        data = [0, 1, 2**32 + 1, 2**53 + 1, 2**60 - 1, 17, 2**40 + 3]
        if source == "int64":
            data[:2] = [-(2**60) + 1, -(2**53) - 1]
    left = layout(np, np.asarray(data, dtype=source), case["layout"])
    if entry in INTEGER64_BINARY_ENTRIES or entry in INTEGER64_COMPARISON_ENTRIES:
        right = np.asarray([1, 2, 3, 1, 0, 2, 1, 3, 0], dtype=source)
        # Comparison cases exercise equal values as well as both orderings.
        if entry in INTEGER64_COMPARISON_ENTRIES:
            right = np.asarray(data, dtype=source)[::-1].copy()
        right = np.resize(right, left.size).reshape(left.shape)
        return [left, right]
    return [left]


def reference(np, case, arrays):
    entry = case["entry"]
    if case["layout"] in FILL_LAYOUTS:
        return np.broadcast_to(arrays[0], (3, 5)).copy()
    if entry in INTEGER64_COPY_ENTRIES:
        return np.array(arrays[0], copy=True, order="C")
    if entry in INTEGER64_CAST_ENTRIES:
        return arrays[0].astype(INTEGER64_CAST_ENTRIES[entry][1])
    if entry in INTEGER64_ABSOLUTE_ENTRIES:
        # Signed minimum retains its two's-complement representation in MLX.
        return np.abs(arrays[0])
    dtype = INTEGER64_ENTRIES[entry]
    operation = entry.removeprefix("vv_").removesuffix(dtype)
    return getattr(np, OPERATIONS[operation])(*arrays)


def payload(np, value):
    return np.ascontiguousarray(value).tobytes().hex()


def trace_entries(case, arrays):
    if case["layout"] in FILL_LAYOUTS:
        return [case["entry"]]
    if not arrays[0].size:
        return []
    entry = case["entry"]
    if entry in INTEGER64_COPY_ENTRIES:
        # ascontiguousarray reuses already-contiguous input storage.
        return [] if arrays[0].flags.c_contiguous else [entry]
    source = arrays[0]
    copy = f"ggn2_dynamic_copy{source.dtype.name}{source.dtype.name}"
    if entry in INTEGER64_ABSOLUTE_ENTRIES:
        return ([copy] if case["layout"] == "reverse" else []) + [entry]
    if not source.flags.c_contiguous:
        if source.dtype.name not in {"int64", "uint64"}:
            copy = (
                "ggn2_dynamic_copybool_bool_"
                if source.dtype.name == "bool"
                else "ggn2_dynamic_copyuint32uint32"
            )
        return [copy, entry]
    return [entry]


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Integer64 workload coverage is incomplete")
    cursor, exercised = 0, set()
    for record, case in zip(records, required):
        arrays = inputs(np, case)
        expected = reference(np, case, arrays)
        entries = trace_entries(case, arrays) if native else []
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("inputPayloads") != [payload(np, value) for value in arrays]
            or record.get("inputUnchanged") is not True
            or record.get("resultPayload") != payload(np, expected)
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype") != expected.dtype.name
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != len(entries)
        ):
            raise ValueError("Integer64 result, input or dispatch count differs")
        events = trace[cursor : cursor + len(entries)]
        cursor += len(entries)
        if [event.get("entry") for event in events] != entries:
            raise ValueError("Integer64 dispatch sequence differs")
        for event in events:
            exercised.add(event["entry"])
            if event.get("dispatchVersion") != DISPATCH_VERSION or event.get(
                "workgroupSize"
            ) != [1, 1, 1]:
                raise ValueError("Integer64 dispatch ABI or workgroup differs")
        if not events:
            continue
        event = events[-1]
        actual = expected
        if case["entry"] in INTEGER64_ABSOLUTE_ENTRIES:
            actual = np.abs(physical(np, arrays[0]))
        values = actual.reshape(-1).tolist()
        if expected.dtype == np.bool_:
            guard = BOOLEAN_GUARD
            if event.get("target") != "metal":
                values, guard = [int(value) for value in values], [
                    int(value) for value in guard
                ]
        elif expected.dtype == np.float32:
            guard = np.asarray(COPY_GUARD, dtype="uint32").view("float32").tolist()
        else:
            guard = COPY_GUARD
        if (
            event.get("integer64Values") != values
            or event.get("integer64GuardValues") != guard
            or any(
                type(got) is not type(want)
                for got, want in zip(event.get("integer64Values", []), values)
            )
            or any(
                type(got) is not type(want)
                for got, want in zip(event.get("integer64GuardValues", []), guard)
            )
            or event.get("threads") != actual.size
            or (
                case["entry"] not in INTEGER64_COPY_ENTRIES
                and event.get("workgroupCount") != [actual.size, 1, 1]
            )
        ):
            raise ValueError("Integer64 native values, guards or launch differ")
    if cursor != len(trace) or (native and not set(INTEGER64_ENTRIES) <= exercised):
        raise ValueError("Integer64 trace coverage differs")


def run(mx, np, host, save):
    records = []
    for case in cases():
        arrays = inputs(np, case)
        operands = [mlx_operand(mx, np, value) for value in arrays]
        start = host.dispatch_count if host else 0
        entry = case["entry"]
        if case["layout"] in FILL_LAYOUTS:
            dtype = getattr(mx, INTEGER64_COPY_ENTRIES[entry])
            if case["layout"] in {"zeros", "ones"}:
                output = getattr(mx, case["layout"])((3, 5), dtype=dtype)
            else:
                output = mx.full((3, 5), operands[0], dtype=dtype)
        elif entry in INTEGER64_COPY_ENTRIES:
            output = mx.contiguous(operands[0])
        elif entry in INTEGER64_CAST_ENTRIES:
            output = operands[0].astype(getattr(mx, INTEGER64_CAST_ENTRIES[entry][1]))
        elif entry in INTEGER64_ABSOLUTE_ENTRIES:
            output = mx.abs(operands[0])
        else:
            operation = entry.removeprefix("vv_").removesuffix(INTEGER64_ENTRIES[entry])
            output = getattr(mx, OPERATIONS[operation])(*operands)
        result = np.array(output)
        records.append(
            {
                **case,
                "inputPayloads": [payload(np, value) for value in arrays],
                "inputUnchanged": all(
                    payload(np, np.array(value)) == payload(np, original)
                    for value, original in zip(operands, arrays)
                ),
                "resultPayload": payload(np, result),
                "resultShape": list(result.shape),
                "resultDtype": result.dtype.name,
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records
