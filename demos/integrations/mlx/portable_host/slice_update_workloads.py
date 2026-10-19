"""Numerical and storage checks for unchanged MLX slice-update operations."""

from demos.integrations.mlx.portable_host import padding_workloads as copies
from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
    wire_value,
)

METHODS = {"sum": "add", "prod": "multiply", "min": "minimum", "max": "maximum"}
LAYOUTS = (
    "matrix",
    "transpose",
    "reverse",
    "reverse-update",
    "broadcast",
    "negative-destination",
    "empty-update",
    "alias",
    "negative-bounds",
    "clipped-stop",
    "singleton",
    "rank3",
    "special-storage",
)


def cases():
    for dtype in copies.DTYPES:
        for operation in METHODS:
            for layout in LAYOUTS:
                if layout == "special-storage" and dtype != "float32":
                    continue
                yield {"dtype": dtype, "operation": operation, "layout": layout}
        for layout in ("negative-bounds", "clipped-stop"):
            yield {"dtype": dtype, "operation": "replace", "layout": layout}


def inputs(np, case):
    dtype, layout = case["dtype"], case["layout"]
    values = np.arange(24) % 9
    if dtype == "bool_":
        values = values % 2 == 0
    elif not dtype.startswith("uint"):
        values = values - 4
    base = values.astype(dtype).reshape(4, 6)
    if dtype == "float32":
        base = base / np.float32(4)
    if dtype in {"int64", "uint64"}:
        base = base + np.asarray(2**54, dtype=dtype)
    if layout == "transpose":
        base = base.reshape(6, 4).T
    elif layout == "reverse":
        base = base[::-1, ::-1]
    update = np.asarray([1, 0, 1, 1, 0, 1], dtype=dtype).reshape(2, 3)
    region = (slice(1, 4, 2), slice(1, 6, 2))
    if layout == "reverse-update":
        update = update[::-1, ::-1]
    elif layout == "broadcast":
        update = np.broadcast_to(update[:1, :1], (2, 3))
    elif layout == "negative-destination":
        region = (slice(3, 0, -2), slice(5, 0, -2))
    elif layout == "empty-update":
        region = (slice(1, 1), slice(1, 6, 2))
        update = update[:0]
    elif layout == "alias":
        if case["operation"] == "prod" and dtype in {"int64", "uint64"}:
            # Avoid signed overflow without losing wide-value cases elsewhere.
            base = (np.arange(24) % 9).astype(dtype).reshape(4, 6)
        update = base[:2, :3]
    elif layout == "negative-bounds":
        region = (slice(-3, -1), slice(-5, -1, 2))
        update = update[:, :2]
    elif layout == "clipped-stop":
        region = (slice(1, 100, 2), slice(-100, 100, 2))
    elif layout == "singleton":
        region = (slice(-1, None, -7), slice(1, 100, 7))
        update = update[:1, :1]
    elif layout == "rank3":
        base = base.reshape(2, 3, 4)
        update = update.reshape(2, 3, 1)
        region = (slice(None), slice(None), slice(1, 2))
    elif layout == "special-storage":
        # Only finite slots participate in arithmetic; all other words must survive.
        region = (slice(None), slice(1, 2))
        update = np.full((4, 1), 0.5, dtype="float32")
        words = [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x7FC12345,
            0xFFC12345,
            0x7FA00001,
            0xFFA00001,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x3F800000,
        ]
        untouched = [i for i in range(base.size) if i % 6 != 1]
        base.view("uint32").reshape(-1)[untouched[: len(words)]] = words
    return base, update, region


def reference(np, case):
    base, update, region = inputs(np, case)
    output = np.array(base, copy=True, order="C")
    if case["operation"] == "replace":
        output[region] = update
    else:
        fn = {"sum": np.add, "prod": np.multiply, "min": np.minimum, "max": np.maximum}
        output[region] = fn[case["operation"]](output[region], update)
    return output


def copy_event(np, target, source, output, strides, offset=0, preserve=False):
    dtype = source.dtype.name
    dtype = "bool_" if dtype == "bool" else dtype
    storage = dtype if dtype in {"bool_", "int64", "uint64"} else "uint32"
    info = copies.metadata(source, strides, offset, preserve, output)
    values = copies.words(np, output)
    guard = BOOLEAN_GUARD if storage == "bool_" else COPY_GUARD
    if storage == "bool_" and target != "metal":
        values, guard = list(map(int, values)), list(map(int, guard))
    return {
        "entry": f"ggn2_dynamic_copy{storage}{storage}",
        "target": target,
        "dispatchVersion": DISPATCH_VERSION,
        "threads": source.size,
        "workgroupSize": [1, 1, 1],
        "workgroupCount": info["workgroupCount"],
        "copyMetadata": info,
        "copyValues": values,
        "copyGuardWords": guard,
    }


def expected_trace(np, case, target):
    base, update, region = inputs(np, case)
    initial = np.array(base, copy=True, order="C")
    output = reference(np, case)
    strides = [stride // base.itemsize for stride in initial.strides]
    events = [copy_event(np, target, base, initial, strides)]
    if not update.size:
        return events
    normalized = [part.indices(size) for part, size in zip(region, base.shape)]
    offset = sum(start * stride for (start, _, _), stride in zip(normalized, strides))
    destination = [stride * step for stride, (_, _, step) in zip(strides, normalized)]
    # MLX simplifies strides of singleton update axes to their direction.
    destination = [
        stride * (-1 if step < 0 and start > 0 else 1) if size == 1 else value
        for size, stride, (start, _, step), value in zip(
            update.shape, strides, normalized, destination
        )
    ]
    if case["operation"] == "replace":
        return events + [
            copy_event(np, target, update, output, destination, offset, True)
        ]
    if not update.flags.c_contiguous:
        dense = np.array(update, copy=True, order="C")
        events.append(
            copy_event(
                np, target, update, dense, [s // update.itemsize for s in dense.strides]
            )
        )
    values = [wire_value(value) for value in output.reshape(-1).tolist()]
    guard = COPY_GUARD
    if case["dtype"] == "float32":
        guard = np.asarray(COPY_GUARD, dtype="uint32").view("float32").tolist()
    elif case["dtype"] == "bool_":
        guard = BOOLEAN_GUARD
        if target != "metal":
            values, guard = list(map(int, values)), list(map(int, guard))
    events.append(
        {
            "entry": "slice_update_" + case["operation"] + case["dtype"],
            "target": target,
            "dispatchVersion": DISPATCH_VERSION,
            "threads": update.size,
            "workgroupSize": [1, 1, 1],
            "workgroupCount": [update.size, 1, 1],
            "sliceUpdateValues": values,
            "sliceUpdateGuardValues": guard,
            **(
                {
                    "sliceUpdateStorageWords": (
                        output.view("uint32").reshape(-1).tolist()
                    ),
                    "sliceUpdateGuardWords": list(COPY_GUARD),
                }
                if case["dtype"] == "float32"
                else {}
            ),
            "sliceUpdateMetadata": {
                "shape": list(update.shape),
                "destinationStrides": destination,
                "destinationOffset": offset,
                "destinationCount": output.size,
            },
        }
    )
    return events


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Slice update case coverage is incomplete")
    cursor = 0
    for record, case in zip(records, required):
        base, update, _ = inputs(np, case)
        expected = reference(np, case)
        events = []
        if native:
            if cursor >= len(trace) or trace[cursor].get("target") not in {
                "metal",
                "opengl",
                "directx",
            }:
                raise ValueError("Slice update native trace is incomplete")
            events = expected_trace(np, case, trace[cursor]["target"])
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("inputPayloads")
            != [copies.payload(np, item) for item in (base, update)]
            or record.get("inputUnchanged") is not True
            or record.get("resultPayload") != copies.payload(np, expected)
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype") != expected.dtype.name
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != len(events)
        ):
            raise ValueError("Slice update result, input or dispatch count differs")
        for expected_event, actual in zip(events, trace[cursor:]):
            for key, wanted in expected_event.items():
                got = actual.get(key)
                if (
                    got != wanted
                    or (
                        key.endswith(("Values", "Words"))
                        and any(type(a) is not type(b) for a, b in zip(got, wanted))
                    )
                    or (
                        case["dtype"] == "float32"
                        and key in {"sliceUpdateValues", "sliceUpdateGuardValues"}
                        and np.asarray(got, dtype="float32").tobytes()
                        != np.asarray(wanted, dtype="float32").tobytes()
                    )
                ):
                    raise ValueError(f"Slice update native evidence differs: {key}")
        cursor += len(events)
        if cursor > len(trace):
            raise ValueError("Slice update native trace is incomplete")
    if cursor != len(trace):
        raise ValueError("Slice update trace contains unexpected dispatches")


def run(mx, np, host, save):
    records = []
    for case in cases():
        base, value, region = inputs(np, case)
        operand = mlx_operand(mx, np, base)
        update = (
            operand[:2, :3] if case["layout"] == "alias" else mlx_operand(mx, np, value)
        )
        start = host.dispatch_count if host else 0
        if case["operation"] == "replace":
            result = mx.array(operand)
            result[region] = update
        else:
            result = getattr(operand.at[region], METHODS[case["operation"]])(update)
        actual = np.array(result)
        records.append(
            {
                **case,
                "inputPayloads": [copies.payload(np, item) for item in (base, value)],
                "inputUnchanged": (
                    copies.payload(np, np.array(operand)) == copies.payload(np, base)
                    and copies.payload(np, np.array(update))
                    == copies.payload(np, value)
                ),
                "resultPayload": copies.payload(np, actual),
                "resultShape": list(actual.shape),
                "resultDtype": actual.dtype.name,
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records
