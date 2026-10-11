"""Exact slice updates whose destination storage exceeds one dispatch axis."""

import math

from demos.integrations.mlx.portable_host import slice_update_workloads as small
from demos.integrations.mlx.portable_host.copy_layout import destination_indices
from demos.integrations.mlx.portable_host.runtime import BOOLEAN_GUARD, COPY_GUARD


def cases():
    for dtype in small.copies.DTYPES:
        yield {"dtype": dtype, "operation": "sum", "layout": "batched"}
    for operation, layout in (
        ("replace", "column"),
        ("replace", "batched"),
        ("prod", "reverse"),
        ("min", "strided"),
        ("max", "broadcast"),
        ("sum", "alias"),
        ("sum", "rank3"),
    ):
        yield {"dtype": "float32", "operation": operation, "layout": layout}
    for count in (65535, 65536, 65537):
        yield {
            "dtype": "float32",
            "operation": "sum",
            "layout": "boundary",
            "count": count,
        }


def arrays(np, case):
    layout, dtype = case["layout"], case["dtype"]
    shape = {
        "column": (2048, 32),
        "strided": (2048, 64),
        "rank3": (4, 257, 256),
        "boundary": (1, case.get("count", 1)),
    }.get(layout, (1026, 128))
    # Retain guard rows around the view and use nonuniform values within it.
    root_shape = (shape[0] + 2, *shape[1:])
    data = (np.arange(math.prod(root_shape)) % 17).reshape(root_shape)
    if dtype == "bool_":
        data = data % 3 == 0
    elif dtype == "float32":
        data = (data - 8) / 4
    elif dtype in {"int64", "uint64"}:
        data = data + 2**54
    root = data.astype(dtype)
    base = root[1:-1]
    region = (slice(1, -1),) + tuple(slice(None) for _ in shape[1:])
    if layout == "column":
        region = (slice(None), slice(0, 1))
    elif layout == "strided":
        region = (slice(None), slice(None, None, 2))
    elif layout == "reverse":
        region = (slice(-2, 0, -1), slice(None, None, -1))
    elif layout == "boundary":
        region = (slice(None), slice(-1, None))
    selected = base[region]
    update = ((np.arange(selected.size) % 3) + 1).reshape(selected.shape).astype(dtype)
    if layout in {"column", "broadcast"}:
        update = np.broadcast_to(np.asarray(6, dtype=dtype), selected.shape)
    elif layout == "alias":
        update = base[1:-1, ::-1]
    return root, update, region


def reference(np, case):
    root, update, region = arrays(np, case)
    output = root[1:-1].copy()
    if case["operation"] == "replace":
        output[region] = update
    else:
        function = {
            "sum": np.add,
            "prod": np.multiply,
            "min": np.minimum,
            "max": np.maximum,
        }[case["operation"]]
        output[region] = function(output[region], update)
    return output


def storage_values(np, value, target):
    if value.dtype == np.float32:
        return np.ascontiguousarray(value).view("uint32").reshape(-1).tolist()
    values = value.reshape(-1).tolist()
    return (
        list(map(int, values))
        if value.dtype == np.bool_ and target != "metal"
        else values
    )


def compare_event(actual, expected):
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Slice capacity copy evidence differs")


def same_values(actual, expected):
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(type(a) is type(b) and a == b for a, b in zip(actual, expected))
    )


def validate_case(case, events, target):
    import numpy as np

    root, update, region = arrays(np, case)
    base = root[1:-1]
    output = reference(np, case)
    strides = [stride // base.itemsize for stride in output.strides]
    if (
        target not in {"metal", "opengl", "directx"}
        or not events
        or any(event.get("target") != target for event in events)
    ):
        raise ValueError("Slice capacity native trace is incomplete")
    compare_event(events[0], small.copy_event(np, target, base, base, strides))
    normalized = [part.indices(size) for part, size in zip(region, base.shape)]
    offset = sum(start * stride for (start, _, _), stride in zip(normalized, strides))
    destination = [stride * step for stride, (_, _, step) in zip(strides, normalized)]
    if case["operation"] == "replace":
        if len(events) != 2:
            raise ValueError("Slice capacity replacement dispatch count differs")
        compare_event(
            events[1],
            small.copy_event(np, target, update, output, destination, offset, True),
        )
        return
    cursor = 1
    if not update.flags.c_contiguous:
        dense = np.array(update, copy=True, order="C")
        if len(events) <= cursor:
            raise ValueError("Slice capacity update materialization is missing")
        compare_event(
            events[cursor],
            small.copy_event(
                np,
                target,
                update,
                dense,
                [stride // dense.itemsize for stride in dense.strides],
            ),
        )
        cursor += 1
    expected_indices = np.arange(base.size).reshape(base.shape)[region].reshape(-1)
    current = base.copy().reshape(-1)
    wanted = output.reshape(-1)
    first = 0
    for event in events[cursor:]:
        count = event.get("threads")
        metadata = event.get("sliceUpdateMetadata", {})
        if (
            type(count) is not int
            or not 0 < count <= 65535
            or event.get("entry") != "slice_update_" + case["operation"] + case["dtype"]
            or event.get("workgroupCount") != [count, 1, 1]
            or event.get("workgroupSize") != [1, 1, 1]
            or metadata.get("destinationCount") != base.size
            or metadata.get("destinationStrides") != destination
            or math.prod(metadata.get("shape", [])) != count
        ):
            raise ValueError("Slice capacity batch metadata differs")
        indices = list(destination_indices(metadata))
        if indices != expected_indices[first : first + count].tolist():
            raise ValueError("Slice capacity batch coverage differs")
        uploads = event.get("inputs", {})
        source = uploads.get("updates", uploads.get("updatesBuffer", {}))
        if not same_values(
            source.get("values"),
            storage_values(np, update.reshape(-1)[first : first + count], target),
        ):
            raise ValueError("Slice capacity uploaded updates differ")
        current[indices] = wanted[indices]
        key = (
            "sliceUpdateStorageWords"
            if case["dtype"] == "float32"
            else "sliceUpdateValues"
        )
        guard_key = (
            "sliceUpdateGuardWords"
            if case["dtype"] == "float32"
            else "sliceUpdateGuardValues"
        )
        guard = list(BOOLEAN_GUARD if case["dtype"] == "bool_" else COPY_GUARD)
        if case["dtype"] == "bool_" and target != "metal":
            guard = list(map(int, guard))
        if not same_values(
            event.get(key), storage_values(np, current, target)
        ) or not same_values(event.get(guard_key), guard):
            raise ValueError("Slice capacity batch readback or guard differs")
        first += count
    if first != update.size:
        raise ValueError("Slice capacity update coverage is incomplete")


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Slice capacity case coverage is incomplete")
    cursor = 0
    for record, case in zip(records, required):
        count = record.get("dispatchCount")
        expected = reference(np, case)
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("inputUnchanged") is not True
            or record.get("resultPayload") != small.copies.payload(np, expected)
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype") != expected.dtype.name
            or type(count) is not int
            or (count < 2 if native else count != 0)
        ):
            raise ValueError("Slice capacity result evidence differs")
        if native:
            events = trace[cursor : cursor + count]
            if len(events) != count:
                raise ValueError("Slice capacity native trace is incomplete")
            validate_case(case, events, events[0].get("target"))
        cursor += count
    if cursor != len(trace):
        raise ValueError("Slice capacity trace contains unexpected dispatches")


def run(mx, np, host, save):
    records = []
    for case in cases():
        root, values, region = arrays(np, case)
        original = mx.array(root)
        operand = original[1:-1]
        update = (
            operand[1:-1, ::-1]
            if case["layout"] == "alias"
            else small.mlx_operand(mx, np, values)
        )
        start = host.dispatch_count if host else 0
        if case["operation"] == "replace":
            result = mx.array(operand)
            result[region] = update
        else:
            result = getattr(operand.at[region], small.METHODS[case["operation"]])(
                update
            )
        actual = np.array(result)
        records.append(
            {
                **case,
                "inputUnchanged": (
                    small.copies.payload(np, np.array(original))
                    == small.copies.payload(np, root)
                    and small.copies.payload(np, np.array(update))
                    == small.copies.payload(np, values)
                ),
                "resultPayload": small.copies.payload(np, actual),
                "resultShape": list(actual.shape),
                "resultDtype": actual.dtype.name,
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records
