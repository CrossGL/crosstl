"""Typed selection workloads with source layouts and exact finite readbacks."""

from collections import Counter

from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.packages import BOOLEAN_COPY_ENTRY, COPY_ENTRY


def cases():
    for dtype in ("float32", "int32", "uint32", "bool_"):
        for layout in (
            "empty",
            "scalar",
            "vector",
            "tail",
            "matrix",
            "transpose",
            "broadcast",
            "reverse",
        ):
            yield {
                "id": f"{dtype}-{layout}",
                "dtype": dtype,
                "layout": layout,
                "mask": "mixed",
            }
        for mask in ("true", "false"):
            yield {
                "id": f"{dtype}-{mask}",
                "dtype": dtype,
                "layout": "matrix",
                "mask": mask,
            }
    for mask in ("mixed", "true", "false"):
        yield {
            "id": f"float32-special-{mask}",
            "dtype": "float32",
            "layout": "special",
            "mask": mask,
        }
    for layout in ("numeric-condition", "promotion"):
        yield {"id": layout, "dtype": "float32", "layout": layout, "mask": "mixed"}


def operands(np, case):
    dtype, layout = case["dtype"], case["layout"]
    values = np.arange(96, dtype="uint32")
    if dtype == "bool_":
        left, right = values % 2 == 0, values % 2 != 0
    elif dtype == "float32":
        left, right = (
            values.astype("float32") / 4 - 12,
            24 - values.astype("float32") / 8,
        )
    elif dtype == "int32":
        left, right = values.astype("int32") - 48, 48 - values.astype("int32")
    else:
        left, right = values * np.uint32(0x01010101), np.uint32(0xFFFFFFFF) - values
    if layout == "empty":
        left, right = left[:0], right[:0]
    elif layout == "scalar":
        left, right = left[0].reshape(()), right[0].reshape(())
    elif layout in {"matrix", "promotion"}:
        left, right = left[:12].reshape(3, 4), right[:12].reshape(3, 4)
    elif layout == "transpose":
        left, right = left[:12].reshape(4, 3).T, right[:12].reshape(3, 4)
    elif layout == "broadcast":
        left, right = left[:3].reshape(3, 1), right[:4].reshape(1, 4)
    elif layout == "reverse":
        left, right = left[16:1:-3], right[:5]
    elif layout == "special":
        words = np.asarray(
            [
                0,
                0x80000000,
                1,
                0x80000001,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0x3F800000,
            ],
            dtype="uint32",
        )
        left, right = words.view("float32"), words[::-1].copy().view("float32")
    else:
        size = 33 if layout == "tail" else 5
        left, right = left[:size], right[:size]
    shape = np.broadcast_shapes(left.shape, right.shape)
    condition = (np.arange(np.prod(shape), dtype="int32") % 3 == 0).reshape(shape)
    if layout == "reverse":
        condition = condition[::-1]
    if case["mask"] != "mixed":
        condition = np.asarray(case["mask"] == "true")
    if layout == "numeric-condition":
        condition = np.asarray([0, -1, 2, 0, -3], dtype="int32")
    if layout == "promotion":
        left = left.astype("int32")
    return condition, left, right


def dtype_name(value):
    return "bool_" if value.dtype.name == "bool" else value.dtype.name


def words(np, value):
    data = np.ascontiguousarray(value)
    return (
        data.reshape(-1).tolist()
        if data.dtype == np.bool_
        else data.view("uint32").reshape(-1).tolist()
    )


def equal_words(actual, expected, dtype):
    if len(actual) != len(expected):
        return False
    for got, want in zip(actual, expected):
        if type(got) is not type(want):
            return False
        if dtype == "float32" and (want & 0x7FFFFFFF) > 0x7F800000:
            if not 0 <= got <= 0xFFFFFFFF or (got & 0x7FFFFFFF) <= 0x7F800000:
                return False
        elif got != want:
            return False
    return True


def expected_entries(np, case):
    arrays = operands(np, case)
    shape = np.broadcast_shapes(*(value.shape for value in arrays))
    if not np.prod(shape):
        return []
    entries, converted = [], []
    for value, dtype in zip(arrays, ("bool_", case["dtype"], case["dtype"])):
        if dtype_name(value) != dtype:
            if not value.flags.c_contiguous:
                entries.append(
                    BOOLEAN_COPY_ENTRY if value.dtype == np.bool_ else COPY_ENTRY
                )
            entries.append(f"v_copy{dtype_name(value)}{dtype}")
            value = np.ascontiguousarray(value.astype(dtype))
        converted.append(np.broadcast_to(value, shape))
    entries.extend(
        BOOLEAN_COPY_ENTRY if value.dtype == np.bool_ else COPY_ENTRY
        for value in converted
        if not value.flags.c_contiguous
    )
    return entries + ["v_Select" + case["dtype"]]


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Selection workload coverage is incomplete")
    cursor = 0
    for record, case in zip(records, required):
        arrays = operands(np, case)
        condition, left, right = arrays
        expected = np.where(
            condition, left.astype(case["dtype"]), right.astype(case["dtype"])
        )
        entries = expected_entries(np, case) if native else []
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("inputWords") != [words(np, value) for value in arrays]
            or record.get("inputUnchanged") is not True
            or record.get("resultShape") != list(expected.shape)
            or record.get("resultDtype") != case["dtype"]
            or not equal_words(
                record.get("resultWords", []), words(np, expected), case["dtype"]
            )
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != len(entries)
        ):
            raise ValueError(f"Selection result or input storage differs: {case['id']}")
        events = trace[cursor : cursor + len(entries)]
        cursor += len(entries)
        if Counter(event["entry"] for event in events) != Counter(entries):
            raise ValueError("Selection dispatch sequence differs")
        if entries:
            event = events[-1]
            from demos.integrations.mlx.portable_host.runtime import (
                BOOLEAN_GUARD,
                COPY_GUARD,
                DISPATCH_VERSION,
            )

            guard = BOOLEAN_GUARD if case["dtype"] == "bool_" else COPY_GUARD
            if case["dtype"] == "float32":
                guard = np.asarray(guard, dtype="uint32").view("float32").tolist()
            elif case["dtype"] == "bool_" and event.get("target") != "metal":
                guard = [int(value) for value in guard]
            actual = (
                np.asarray(
                    [float(value) for value in event["selectionValues"]],
                    dtype="float32",
                )
                if case["dtype"] == "float32"
                else np.asarray(event["selectionValues"], dtype=case["dtype"])
            )
            if (
                event["entry"] != entries[-1]
                or event["threads"] != expected.size
                or event.get("dispatchVersion") != DISPATCH_VERSION
                or event.get("selectionGuardValues") != guard
                or event["workgroupCount"] != [expected.size, 1, 1]
                or event["workgroupSize"] != [1, 1, 1]
                or not equal_words(
                    words(np, actual), words(np, expected), case["dtype"]
                )
            ):
                raise ValueError("Selection native readback differs")
    if cursor != len(trace):
        raise ValueError("Selection trace contains unexpected dispatches")


def run(mx, np, host, save):
    records = []
    for case in cases():
        arrays = operands(np, case)
        values = [mlx_operand(mx, np, value) for value in arrays]
        start = host.dispatch_count if host else 0
        result = np.array(mx.where(*values))
        records.append(
            {
                **case,
                "inputWords": [words(np, value) for value in arrays],
                "inputUnchanged": all(
                    words(np, np.array(value)) == words(np, original)
                    for value, original in zip(values, arrays)
                ),
                "resultWords": words(np, result),
                "resultShape": list(result.shape),
                "resultDtype": dtype_name(result),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records
