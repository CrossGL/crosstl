"""Source-typed bfloat arithmetic, conversion and layout workloads."""

import math

from demos.integrations.mlx.portable_host import bfloat_storage
from demos.integrations.mlx.portable_host.binary_workloads import LAYOUTS, operands
from demos.integrations.mlx.portable_host.half_arithmetic_workloads import OPERATIONS
from demos.integrations.mlx.portable_host.packages import (
    BFLOAT_CAST_ENTRIES,
    BFLOAT_COMPARISON_ENTRIES,
    BFLOAT_COPY_ENTRY,
    BFLOAT_ENTRIES,
)
from demos.integrations.mlx.portable_host.runtime import physical_dtype


def cases():
    return [
        {"entry": entry, "layout": layout}
        for entry in BFLOAT_ENTRIES
        for layout in (
            ("scalar",) if entry == "vv_NaNEqualbfloat16" else (*LAYOUTS, "batched")
        )
    ]


def inputs(np, case):
    if case["layout"] == "batched":
        a = np.resize(np.asarray([-3.5, -0.0, 1.0078125, 2.5], dtype=np.float32), 65537)
        return a, np.full(a.shape, 1.5, dtype=np.float32)
    a, b = operands("float32", case["layout"])
    if case["entry"] == "v_copyfloat32bfloat16":
        # Adjacent midpoint ties must select the even bfloat significand.
        a = np.asarray(a + np.float32(0.00390625))
    return a, b


def round_words(np, values):
    words = np.ascontiguousarray(values, dtype=np.float32).reshape(-1).view(np.uint32)
    return (
        ((words.astype(np.uint64) + 0x7FFF + ((words >> 16) & 1)) >> 16) & 0xFFFF
    ).astype(np.uint16)


def source_values(np, values, dtype):
    if dtype == "float32":
        return values
    words = round_words(np, values)
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(values.shape)


def result(np, case, a, b):
    entry = case["entry"]
    if entry == BFLOAT_COPY_ENTRY:
        return a, "bfloat16"
    if entry in BFLOAT_CAST_ENTRIES:
        return a, BFLOAT_CAST_ENTRIES[entry][1]
    operation = "Abs" if entry.startswith("v_") else entry[3:-8]
    if operation == "Abs":
        value = np.abs(a)
    elif operation == "NaNEqual":
        value = np.asarray(np.equal(a, b) | (np.isnan(a) & np.isnan(b)))
    elif operation in {"Minimum", "Maximum"}:
        cmp = np.less if operation == "Minimum" else np.greater
        value = np.where(np.isnan(a) | cmp(a, b), a, b)
    else:
        value = getattr(np, OPERATIONS[operation])(a, b)
    return value, "bool_" if entry in BFLOAT_COMPARISON_ENTRIES else "bfloat16"


def words(np, values, dtype):
    if dtype == "bool_":
        return np.asarray(values, dtype=np.bool_).reshape(-1).astype(np.uint8).tolist()
    if dtype == "bfloat16":
        return round_words(np, values).tolist()
    return (
        np.ascontiguousarray(values, dtype=np.float32)
        .reshape(-1)
        .view(np.uint32)
        .tolist()
    )


def operand(mx, np, values, dtype):
    base = values
    while isinstance(base.base, np.ndarray):
        base = base.base
    source = mx.array(np.ravel(base, order="K").tolist(), dtype=getattr(mx, dtype))
    offset = (values.ctypes.data - base.ctypes.data) // values.itemsize
    return mx.as_strided(
        source,
        values.shape,
        tuple(s // values.itemsize for s in values.strides),
        offset,
    )


def sequence(case, a, b):
    if not a.size:
        return []
    entry = case["entry"]
    if entry == BFLOAT_COPY_ENTRY:
        return [] if a.flags.c_contiguous else [entry]
    casts = entry in BFLOAT_CAST_ENTRIES
    absolute = entry.startswith("v_") and not casts
    copy = (
        "ggn2_dynamic_copyuint32uint32"
        if entry == "v_copyfloat32bfloat16"
        else BFLOAT_COPY_ENTRY
    )
    sources = (a,) if casts or absolute else (a, b)
    if absolute and case["layout"] in {"transpose", "broadcast"}:
        return [entry]
    return [copy for value in sources if not value.flags.c_contiguous] + [
        entry
    ] * math.ceil(a.size / 65535)


def run(mx, np, host, save):
    records = []
    for case in cases():
        entry = case["entry"]
        a, b = inputs(np, case)
        source = BFLOAT_CAST_ENTRIES.get(entry, ("bfloat16",))[0]
        left, right = operand(mx, np, a, source), operand(mx, np, b, source)
        mx.eval(left, right)
        start = host.dispatch_count if host else 0
        if entry == BFLOAT_COPY_ENTRY:
            output = mx.contiguous(left)
        elif entry in BFLOAT_CAST_ENTRIES:
            output = left.astype(getattr(mx, BFLOAT_CAST_ENTRIES[entry][1]))
        elif entry == "v_Absbfloat16bfloat16":
            output = mx.abs(left)
        elif entry == "vv_NaNEqualbfloat16":
            output = mx.array_equal(left, right, equal_nan=True)
        else:
            output = getattr(mx, OPERATIONS[entry[3:-8]])(left, right)
        mx.eval(output)
        expected, dtype = result(
            np, case, source_values(np, a, source), source_values(np, b, source)
        )
        # tolist reads the completed allocation without inserting a GPU cast or view.
        observed = np.asarray(
            output.tolist(), dtype=np.bool_ if dtype == "bool_" else np.float32
        ).reshape(output.shape)
        records.append(
            {
                **case,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "inputWords": words(np, np.asarray(left.tolist()), source),
                "resultWords": words(np, observed, dtype),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records


def validate(records, trace, *, native):
    import numpy as np

    if not isinstance(records, list) or len(records) != len(cases()):
        raise ValueError("Bfloat workload inventory differs")
    cursor = 0
    for record, case in zip(records, cases()):
        source = BFLOAT_CAST_ENTRIES.get(case["entry"], ("bfloat16",))[0]
        a, b = inputs(np, case)
        expected, dtype = result(
            np, case, source_values(np, a, source), source_values(np, b, source)
        )
        entries = sequence(case, a, b) if native else []
        wanted = {
            **case,
            "shape": list(expected.shape),
            "dtype": "mlx.core." + dtype.rstrip("_"),
            "inputWords": words(np, a, source),
            "resultWords": words(np, expected, dtype),
            "dispatchCount": len(entries),
        }
        if record != wanted or any(
            type(word) is not int
            for name in ("inputWords", "resultWords")
            for word in record[name]
        ):
            raise ValueError(f"Bfloat workload result differs: {case}")
        events = trace[cursor : cursor + len(entries)]
        cursor += len(entries)
        if [event["entry"] for event in events] != entries:
            raise ValueError("Bfloat workload dispatch sequence differs")
        result_events = [event for event in events if event["entry"] == case["entry"]]
        if not result_events:
            continue
        stored = expected
        if case["entry"] == "v_Absbfloat16bfloat16":
            if case["layout"] == "transpose":
                stored = expected.T
            elif case["layout"] == "broadcast":
                stored = expected[0]
        actual_words = []
        for event in result_events:
            data = event["bfloatStorage"]
            target = event["target"]
            logical = data["logicalWords"]
            actual_words.extend(logical)
            if dtype == "bfloat16":
                physical = bfloat_storage.pack(logical, target)
                guard = bfloat_storage.pack(bfloat_storage.GUARD, target)
                encoding = bfloat_storage.encoding(target)
            elif dtype == "bool_":
                physical = list(map(bool, logical)) if target == "metal" else logical
                guard = [
                    bool(i % 2 == 0) if target == "metal" else int(i % 2 == 0)
                    for i in range(32)
                ]
                encoding = None
            else:
                physical, guard, encoding = (
                    logical,
                    [0x6A15BEEF] * 32,
                    "ieee754-binary32",
                )
            if (
                data != {
                    "logicalType": dtype,
                    "physicalType": physical_dtype(dtype, target),
                    "encoding": encoding,
                    "values": physical,
                    "guardValues": guard,
                    "logicalWords": logical,
                }
                or event["dispatchVersion"] != 3
                or event["workgroupSize"] != [1, 1, 1]
            ):
                raise ValueError("Bfloat native storage, guards or launch differs")
            if any(
                type(got) is not type(want)
                for got, want in zip(
                    data["values"] + data["guardValues"], physical + guard
                )
            ) or any(type(word) is not int for word in logical):
                raise ValueError("Bfloat native storage word types differ")
            if event["threads"] != len(logical):
                raise ValueError("Bfloat native output coverage differs")
        if actual_words != words(np, stored, dtype):
            raise ValueError(
                "Bfloat native readback differs from the independent reference"
            )
    if cursor != len(trace):
        raise ValueError("Bfloat workload has unexpected native dispatches")
