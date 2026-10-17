"""Exercise upstream column plans, including native intermediate readbacks."""

import math
import struct

from demos.integrations.mlx.portable_host.reduction_packages import COLUMN_ENTRIES
from demos.integrations.mlx.portable_host.row_workloads import values_match, wire_values


def cases():
    for entry, dtype in COLUMN_ENTRIES.items():
        two_pass = "2pass" in entry
        dimension = int(entry.split("_")[3])
        rows = {1: 257, 2: 129, 5: 65}[dimension] if two_pass else 33
        prefix = () if dimension == 1 else (2, 2) if dimension == 2 else (2, 2, 2, 2)
        for columns in (32, 33):
            case = {
                "id": f"{entry}-{columns}",
                "entry": entry,
                "dtype": dtype,
                "operation": entry.rsplit("reduce_", 1)[1].removesuffix(dtype),
                "shape": (*prefix, rows, columns),
                "axes": tuple(range(0, len(prefix) + 1, 2)),
                "layout": "dense",
            }
            yield case
        if dimension == 1:
            yield case | {"id": f"{entry}-65-columns", "shape": (rows, 65)}
            for boundary in ((1025, 1985) if two_pass else (32, 256)):
                yield case | {
                    "id": f"{entry}-{boundary}-rows",
                    "shape": (boundary, 33),
                }
        if dimension == 2:
            for layout in ("slice", "broadcast"):
                yield case | {"id": f"{entry}-{layout}", "layout": layout}
        if dtype == "float32":
            for profile in (
                "early-nan",
                "late-nan",
                "positive-infinity",
                "negative-zero",
            ):
                yield case | {"id": f"{entry}-{profile}", "profile": profile}


def reduce(np, values, operation, axes):
    name = {"and": "all", "or": "any"}.get(operation, operation)
    kwargs = {"dtype": values.dtype} if name in {"sum", "prod"} else {}
    if name in {"min", "max"}:
        limits = (
            (-math.inf, math.inf)
            if values.dtype.kind == "f"
            else (np.iinfo(values.dtype).min, np.iinfo(values.dtype).max)
        )
        kwargs["initial"] = limits[1 if name == "min" else 0]
    return getattr(np, name)(values, axis=axes, **kwargs)


def reference(np, case):
    shape = case["shape"]
    base_shape = (
        (4, *shape[1:])
        if case["layout"] == "slice"
        else (1, *shape[1:]) if case["layout"] == "broadcast" else shape
    )
    indexes = np.arange(math.prod(base_shape), dtype=np.int64)
    column = indexes % shape[-1]
    dtype, operation = case["dtype"], case["operation"]
    if dtype == "bool_":
        values = np.where(
            column % 3 == 0, True, np.where(column % 3 == 1, False, indexes % 7 != 0)
        )
    elif operation == "prod":
        values = np.where(indexes % 3 == 0, -1 if dtype != "uint32" else 1, 1)
        values = np.where((column % 3 == 1) & (indexes // shape[-1] == 0), 0, values)
    else:
        values = indexes % 23 - (0 if dtype == "uint32" else 11)
        if operation in {"min", "max"}:
            values = values + (column % 11) * 24
        if dtype == "float32":
            values = values * 0.25
    base = values.astype(np.bool_ if dtype == "bool_" else dtype).reshape(base_shape)
    if "profile" in case:
        base.fill(1)
        profile = case["profile"]
        if profile in {"early-nan", "late-nan"}:
            base.flat[0 if profile == "early-nan" else -1] = np.nan
        elif profile == "positive-infinity":
            base.fill(np.inf)
        elif profile == "negative-zero":
            base.fill(-0.0)
        else:
            raise ValueError("Unknown column numerical profile")
    logical = base[1::2] if case["layout"] == "slice" else np.broadcast_to(base, shape)
    expected = reduce(np, logical, operation, case["axes"])
    partials = None
    if "2pass" in case["entry"]:
        remaining = tuple(
            axis for axis in range(logical.ndim) if axis not in case["axes"]
        )
        rows = np.transpose(logical, case["axes"] + remaining).reshape(
            (-1, *expected.shape)
        )
        # Each native block reduces 32-row tiles, revisiting every 1024 rows.
        partitions = (np.arange(len(rows)) // 32) % 32
        partials = np.stack(
            [reduce(np, rows[partitions == block], operation, 0) for block in range(32)]
        )
    return base, logical, expected, partials


def collect(mx, np, *, observe=None, dispatch_count=None):
    records = []
    for case in cases():
        base, logical, expected, _ = reference(np, case)
        start = dispatch_count() if dispatch_count is not None else None
        source = mx.array(base, dtype=getattr(mx, case["dtype"]))
        if case["layout"] == "slice":
            source = source[1::2]
        elif case["layout"] == "broadcast":
            source = mx.broadcast_to(source, case["shape"])
        name = {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
        result = getattr(mx, name)(source, axis=case["axes"])
        actual = np.array(result)
        record = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in case.items()
        } | {
            "actual": wire_values(actual.tolist()),
            "expected": wire_values(expected.tolist()),
            "resultShape": list(result.shape),
            "logicalShape": list(logical.shape),
            "resultDtype": str(result.dtype),
        }
        if dispatch_count is not None:
            record.update(dispatchStart=start, dispatchEnd=dispatch_count())
        records.append(record)
        if observe is not None:
            observe(record)
        if result.dtype != getattr(mx, case["dtype"]) or not values_match(
            record["actual"], expected.tolist(), case["dtype"]
        ):
            raise RuntimeError(f"Incorrect column reduction: {case['id']}")
    return records


def validate(records, *, trace=None):
    import numpy as np

    expected_cases = list(cases())
    if not isinstance(records, list) or len(records) != len(expected_cases):
        raise ValueError("Column evidence must cover every workload")
    cursor = 0
    for case, record in zip(expected_cases, records):
        _, logical, expected, partials = reference(np, case)
        dtype = case["dtype"]
        identity = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in case.items()
        }
        if (
            any(record.get(key) != value for key, value in identity.items())
            or not values_match(record.get("actual"), expected.tolist(), dtype)
            or not values_match(record.get("expected"), expected.tolist(), dtype)
            or record.get("resultShape") != list(expected.shape)
            or record.get("logicalShape") != list(logical.shape)
            or record.get("resultDtype") != "mlx.core." + dtype.removesuffix("_")
        ):
            raise ValueError("Column evidence is missing or numerically incorrect")
        if trace is None:
            continue
        passes = [expected] if partials is None else [partials, expected]
        start, end = record.get("dispatchStart"), record.get("dispatchEnd")
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or end != cursor + len(passes)
            or end > len(trace)
        ):
            raise ValueError(
                "Column workload did not execute exactly its upstream passes"
            )
        cursor = end
        for stage, values in enumerate(passes):
            dispatch = trace[start + stage]
            stride = logical.shape[-1] if stage == 0 else expected.size
            outer = expected.size // stride
            entry = (
                case["entry"]
                if stage == 0
                else "col_reduce_looped_1_32_32_reduce_" + case["operation"] + dtype
            )
            groups = [
                (stride + 31) // 32,
                outer * (32 if stage == 0 and partials is not None else 1),
                1,
            ]
            guards = (
                [index % 2 == 0 for index in range(32)]
                if dtype == "bool_"
                else [
                    (
                        struct.unpack("=f", struct.pack("=I", 0x6A15BEEF))[0]
                        if dtype == "float32"
                        else 0x6A15BEEF
                    )
                ]
                * 32
            )
            if (
                dispatch.get("entry") != entry
                or dispatch.get("dispatchVersion") != 3
                or type(dispatch.get("dispatchVersion")) is not int
                or type(dispatch.get("threads")) is not int
                or dispatch.get("threads")
                != (logical.size if stage == 0 else partials.size)
                or dispatch.get("workgroupSize") != [256, 1, 1]
                or dispatch.get("workgroupCount") != groups
                or any(
                    type(value) is not int
                    for key in ("workgroupSize", "workgroupCount")
                    for value in dispatch[key]
                )
                or not values_match(
                    dispatch.get("reductionValues"),
                    values.reshape(-1).tolist(),
                    dtype,
                    physical_boolean=dispatch.get("target") in {"opengl", "directx"},
                )
                or dispatch.get("reductionGuardValues") != guards
            ):
                raise ValueError(
                    "Column trace does not match its native pass or readback"
                )
    if trace is not None and cursor != len(trace):
        raise ValueError("Column trace contains unaccounted dispatches")
