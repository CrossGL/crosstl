"""Bit-exact padding and slice replacement through translated copy kernels."""

import math

from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
)

DTYPES = ("float32", "int32", "uint32", "bool_", "int64", "uint64")
PAD_LAYOUTS = (
    "matrix",
    "transpose",
    "reverse",
    "broadcast",
    "empty-input",
    "empty-output",
    "singleton",
    "zero-width",
    "limit",
)
UPDATE_LAYOUTS = (
    "matrix",
    "transpose",
    "reverse",
    "reverse-update",
    "broadcast",
    "negative-destination",
    "empty-update",
    "alias",
)


def cases():
    for dtype in DTYPES:
        for operation, layouts in (("pad", PAD_LAYOUTS), ("update", UPDATE_LAYOUTS)):
            for layout in layouts:
                yield {"dtype": dtype, "operation": operation, "layout": layout}


def payload(np, value):
    return np.ascontiguousarray(value).tobytes().hex()


def inputs(np, case):
    dtype = case["dtype"]
    words = np.asarray(
        [0, 0x80000000, 1, 0xFFFFFFFF, 0x7FC12345, 0x7F800000, 0x3F800000, 0x55555555],
        dtype="uint32",
    )
    if dtype == "bool_":
        data = (words & 1).astype("bool")
    elif dtype in {"int64", "uint64"}:
        data = np.asarray(
            [0, 2**63, 1, 2**64 - 1, 2**53 + 1, 2**63 - 1, 17, 2**63 + 9],
            dtype="uint64",
        ).view(dtype)
    else:
        data = words.view(dtype)
    data = np.tile(data, 8192)
    layout = case["layout"]
    if case["operation"] == "pad":
        base = data[:15].reshape(3, 5)
        widths = ((1, 2), (3, 1))
        if layout == "transpose":
            base = data[:15].reshape(5, 3).T
        elif layout == "reverse":
            base = base[::-1, ::-1]
        elif layout == "broadcast":
            base = np.broadcast_to(data[:5], (3, 5))
        elif layout in {"empty-input", "empty-output"}:
            base = base[:0]
            if layout == "empty-output":
                widths = ((0, 0), (3, 1))
        elif layout == "singleton":
            base = data[4:5].reshape(1, 1)
        elif layout == "zero-width":
            widths = ((0, 0), (0, 0))
        elif layout == "limit":
            base = data[:65025].reshape(255, 255)
            widths = ((0, 0), (1, 1))
        return base, data[4:5].reshape(()), widths
    base = data[:24].reshape(4, 6)
    if layout == "transpose":
        base = data[:24].reshape(6, 4).T
    elif layout == "reverse":
        base = base[::-1, ::-1]
    update = data[17:23].reshape(2, 3)
    selection = (slice(1, 4, 2), slice(1, 6, 2))
    if layout == "reverse-update":
        update = update[::-1, ::-1]
    elif layout == "broadcast":
        update = np.broadcast_to(data[4:5], (2, 3))
    elif layout == "negative-destination":
        selection = (slice(3, 0, -2), slice(5, 0, -2))
    elif layout == "empty-update":
        selection = (slice(1, 1), slice(1, 6, 2))
        update = update[:0]
    elif layout == "alias":
        update = base[:2, :3]
    return base, update, selection


def stages(np, case):
    base, value, region = inputs(np, case)
    if case["operation"] == "pad":
        shape = tuple(
            size + low + high for size, (low, high) in zip(base.shape, region)
        )
        initial = np.full(shape, value, dtype=base.dtype)
        output = np.pad(base, region, constant_values=value)
        strides = [stride // base.itemsize for stride in initial.strides]
        offset = sum(low * stride for (low, _), stride in zip(region, strides))
        operations = [
            (np.broadcast_to(value, shape), strides, 0, False, initial),
            (base, strides, offset, True, output),
        ]
    else:
        initial = np.array(base, copy=True, order="C")
        output = initial.copy()
        output[region] = value
        strides = [stride // base.itemsize for stride in initial.strides]
        normalized = [item.indices(size) for item, size in zip(region, base.shape)]
        offset = sum(
            start * stride for (start, _, _), stride in zip(normalized, strides)
        )
        operations = [
            (base, strides, 0, False, initial),
            (
                value,
                [stride * item[2] for stride, item in zip(strides, normalized)],
                offset,
                True,
                output,
            ),
        ]
    return output, [item for item in operations if item[0].size]


def words(np, value):
    array = np.ascontiguousarray(value)
    return (
        array.reshape(-1).tolist()
        if array.dtype in (np.bool_, np.int64, np.uint64)
        else array.view("uint32").reshape(-1).tolist()
    )


def metadata(source, strides, offset, preserve, output):
    shape = list(source.shape)
    src_strides = [
        stride // source.itemsize if size != 1 else 0
        for size, stride in zip(source.shape, source.strides)
    ]
    extents = [(size - 1) * stride for size, stride in zip(shape, src_strides)]
    low, high = sum(min(v, 0) for v in extents), sum(max(v, 0) for v in extents)
    return {
        "shape": shape,
        "sourceStrides": src_strides,
        "destinationStrides": strides,
        "sourceOffset": -low,
        "destinationOffset": offset,
        "sourceCount": high - low + 1,
        "destinationCount": output.size,
        "preserveDestination": preserve,
        "workgroupCount": [(shape[-1] + 1) // 2, shape[-2], math.prod(shape[:-2])],
    }


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Padding case coverage is incomplete")
    cursor = 0
    for record, case in zip(records, required):
        base, value, _ = inputs(np, case)
        expected, operations = stages(np, case)
        operations = operations if native else []
        if (
            any(record.get(key) != val for key, val in case.items())
            or record.get("inputPayloads")
            != [payload(np, item) for item in (base, value)]
            or record.get("inputUnchanged") is not True
            or record.get("resultPayload") != payload(np, expected)
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype") != expected.dtype.name
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != len(operations)
        ):
            raise ValueError("Padding result, input or dispatch count differs")
        events = trace[cursor : cursor + len(operations)]
        cursor += len(operations)
        if len(events) != len(operations):
            raise ValueError("Padding copy trace is incomplete")
        dtype = (
            case["dtype"] if case["dtype"] in {"bool_", "int64", "uint64"} else "uint32"
        )
        for event, operation in zip(events, operations):
            source, _, _, _, destination = operation
            info = metadata(*operation)
            actual_words = words(np, destination)
            guard = BOOLEAN_GUARD if dtype == "bool_" else COPY_GUARD
            if dtype == "bool_" and event.get("target") != "metal":
                actual_words, guard = [int(v) for v in actual_words], [
                    int(v) for v in guard
                ]
            if (
                event.get("entry") != f"ggn2_dynamic_copy{dtype}{dtype}"
                or event.get("target") not in {"metal", "opengl", "directx"}
                or event.get("dispatchVersion") != DISPATCH_VERSION
                or event.get("threads") != source.size
                or event.get("workgroupSize") != [1, 1, 1]
                or event.get("workgroupCount") != info["workgroupCount"]
                or event.get("copyMetadata") != info
                or event.get("copyValues") != actual_words
                or event.get("copyGuardWords") != guard
                or any(
                    type(a) is not type(b)
                    for a, b in zip(event.get("copyValues", []), actual_words)
                )
            ):
                raise ValueError("Padding copy storage, guards or addressing differ")
    if cursor != len(trace):
        raise ValueError("Padding trace contains unexpected dispatches")


def run(mx, np, host, save):
    records = []
    for case in cases():
        base, value, region = inputs(np, case)
        operand = mlx_operand(mx, np, base)
        update = (
            operand[:2, :3] if case["layout"] == "alias" else mlx_operand(mx, np, value)
        )
        start = host.dispatch_count if host else 0
        if case["operation"] == "pad":
            result = mx.pad(operand, region, constant_values=update)
        else:
            result = mx.array(operand)
            result[region] = update
        actual = np.array(result)
        records.append(
            {
                **case,
                "inputPayloads": [payload(np, item) for item in (base, value)],
                "inputUnchanged": (
                    payload(np, np.array(operand)) == payload(np, base)
                    and payload(np, np.array(update)) == payload(np, value)
                ),
                "resultPayload": payload(np, actual),
                "resultShape": list(actual.shape),
                "resultDtype": actual.dtype.name,
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records
