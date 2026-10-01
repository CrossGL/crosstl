"""Execute row reductions through MLX with independently calculated references."""

import math
import struct

from demos.integrations.mlx.portable_host.reduction_packages import ROW_ENTRIES
from demos.integrations.mlx.portable_host.reduction_workloads import matches
from demos.integrations.mlx.portable_host.row_reduction_layout import width


def cases(widths):
    for entry, dtype in ROW_ENTRIES.items():
        operation = entry.removesuffix(dtype).rsplit("_", 1)[1]
        for group_width in widths:
            row_sizes = (
                (65, 512)
                if group_width == 32
                else (
                    (513, 1024)
                    if group_width == 128
                    else (group_width * 4 - 127, group_width * 4)
                )
            )
            for row_size in row_sizes:
                if "simple" in entry:
                    rows = 33 if 33 * row_size <= 65535 else 32
                    shape, axes = (rows, row_size), (1,)
                elif "looped_1_" in entry:
                    shape, axes = (2, 3, row_size), (0, 2)
                elif "looped_2_" in entry:
                    shape, axes = (2, 2, 2, 2, row_size), (0, 2, 4)
                else:
                    shape, axes = (2, 2, 2, 2, 2, 2, row_size), (0, 2, 4, 6)
                if math.prod(shape) > 65535:
                    continue
                assert width(row_size) == group_width
                yield {
                    "id": f"{entry}-{row_size}",
                    "entry": entry,
                    "dtype": dtype,
                    "operation": operation,
                    "shape": shape,
                    "axes": axes,
                    "width": group_width,
                    "layout": "dense",
                }
        if "looped_1_" in entry and 32 in widths:
            for layout in ("contiguous", "slice", "transpose"):
                shape = (
                    (6, 65)
                    if layout == "slice"
                    else (2, 3, 65) if layout == "transpose" else (3, 65)
                )
                yield {
                    "id": f"{entry}-{layout}",
                    "entry": entry,
                    "dtype": dtype,
                    "operation": operation,
                    "shape": shape,
                    "axes": (len(shape) - 1,),
                    "width": 32,
                    "layout": layout,
                }
        if "looped_1_" in entry and 1024 in widths:
            yield {
                "id": f"{entry}-capped-width",
                "entry": entry,
                "dtype": dtype,
                "operation": operation,
                "shape": (2, 32767),
                "axes": (1,),
                "width": 1024,
                "layout": "dense",
            }


def reference(np, case):
    dtype = case["dtype"]
    indexes = np.arange(math.prod(case["shape"]), dtype=np.int64)
    source_rows = indexes // case["shape"][-1]
    if dtype == "bool_":
        values = np.where(
            source_rows % 3 == 0,
            True,
            np.where(source_rows % 3 == 1, False, indexes % 7 != 0),
        )
    elif case["operation"] == "prod":
        values = np.where(indexes % 3 == 0, -1 if dtype != "uint32" else 1, 1)
        values = np.where(
            (source_rows % 3 == 1)
            & (indexes % case["shape"][-1] == case["shape"][-1] - 1),
            0,
            values,
        )
    else:
        values = indexes % 23 - (0 if dtype == "uint32" else 11)
        if case["operation"] in {"min", "max"}:
            values = values + (source_rows % 11) * 24
        if dtype == "float32":
            values = values * 0.25
    base = values.astype(np.bool_ if dtype == "bool_" else dtype).reshape(case["shape"])
    values = (
        base[1::2]
        if case["layout"] == "slice"
        else base.transpose(1, 0, 2) if case["layout"] == "transpose" else base
    )
    name = {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
    kwargs = {"dtype": values.dtype} if name in {"sum", "prod"} else {}
    return base, values, getattr(np, name)(values, axis=case["axes"], **kwargs)


def collect(mx, np, widths, *, observe=None, dispatch_count=None):
    records = []
    for case in cases(widths):
        dtype = case["dtype"]
        base, values, expected = reference(np, case)
        start = dispatch_count() if dispatch_count is not None else None
        source = mx.array(base, dtype=getattr(mx, dtype))
        if case["layout"] == "slice":
            source = source[1::2]
        elif case["layout"] == "transpose":
            source = mx.transpose(source, (1, 0, 2))
        name = {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
        result = getattr(mx, name)(source, axis=case["axes"])
        actual = np.array(result)
        record = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in case.items()
        } | dict(
            logicalShape=list(values.shape),
            expected=expected.tolist(),
            actual=actual.tolist(),
            resultShape=list(result.shape),
            resultDtype=str(result.dtype),
        )
        if dispatch_count is not None:
            record.update(dispatchStart=start, dispatchEnd=dispatch_count())
        records.append(record)
        if observe is not None:
            observe(record)
        if (
            result.dtype != getattr(mx, dtype)
            or actual.shape != expected.shape
            or any(
                not matches(a.item(), b.item(), dtype)
                for a, b in zip(actual.flat, expected.flat)
            )
        ):
            raise RuntimeError(f"Incorrect row reduction: {case['id']}")
    return records


def validate(records, widths, *, trace=None):
    import numpy as np

    expected = list(cases(widths))
    if len(records) != len(expected) or {case["entry"] for case in expected} != set(
        ROW_ENTRIES
    ):
        raise ValueError("Row evidence must cover every supported entry")
    if trace is not None and len(trace) != len(records):
        raise ValueError("Row trace must retain every dispatch exactly once")
    for ordinal, (record, case) in enumerate(zip(records, expected)):
        _, logical, reference_values = reference(np, case)
        identity = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in case.items()
        }
        if (
            any(record.get(key) != value for key, value in identity.items())
            or record.get("actual") != reference_values.tolist()
            or record.get("expected") != reference_values.tolist()
            or record.get("resultShape") != list(reference_values.shape)
            or record.get("logicalShape") != list(logical.shape)
            or record.get("resultDtype")
            != "mlx.core." + case["dtype"].removesuffix("_")
        ):
            raise ValueError("Row evidence is missing or numerically incorrect")
        if trace is not None:
            start, end = record.get("dispatchStart"), record.get("dispatchEnd")
            if (
                type(start) is not int
                or type(end) is not int
                or start != ordinal
                or end != ordinal + 1
            ):
                raise ValueError(
                    "Row workload did not execute exactly one native dispatch"
                )
            dispatch = trace[start]
            rows = reference_values.size
            grid = [1, (rows + 3) // 4 if "simple" in case["entry"] else rows, 1]
            guards = dispatch.get("reductionGuardValues")
            expected_guards = (
                [index % 2 == 0 for index in range(32)]
                if case["dtype"] == "bool_"
                else [
                    (
                        struct.unpack("=f", struct.pack("=I", 0x6A15BEEF))[0]
                        if case["dtype"] == "float32"
                        else 0x6A15BEEF
                    )
                ]
                * 32
            )
            dimensions = [dispatch.get("workgroupSize"), dispatch.get("workgroupCount")]
            if (
                any(
                    not isinstance(items, list)
                    or len(items) != 3
                    or any(type(item) is not int for item in items)
                    for items in dimensions
                )
                or type(dispatch.get("threads")) is not int
            ):
                raise ValueError("Row trace launch dimensions must be integers")
            if (
                dispatch.get("entry") != case["entry"]
                or dispatch.get("workgroupSize") != [case["width"], 1, 1]
                or dispatch.get("workgroupCount") != grid
                or dispatch.get("threads") != logical.size
                or type(dispatch.get("dispatchVersion")) is not int
                or dispatch.get("dispatchVersion") != 2
                or dispatch.get("reductionValues")
                != reference_values.reshape(-1).tolist()
                or guards != expected_guards
            ):
                raise ValueError("Row trace does not match the source entry and width")
