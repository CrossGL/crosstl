"""Compose MLX reductions across independently packaged kernel families."""

import struct

from demos.integrations.mlx.portable_host.packages import BOOLEAN_COPY_ENTRY, COPY_ENTRY
from demos.integrations.mlx.portable_host.reduction_packages import ENTRIES
from demos.integrations.mlx.portable_host.row_workloads import values_match, wire_values


def cases():
    for entry, dtype in ENTRIES.items():
        operation = entry.removeprefix("all_reduce_").removesuffix(dtype)
        for direction in ("row-first", "scalar-first"):
            yield {
                "id": f"{entry}-{direction}",
                "entry": entry,
                "rowEntry": "row_reduce_simple_" + operation + dtype,
                "dtype": dtype,
                "operation": operation,
                "direction": direction,
            }


def reference(np, case):
    dtype, operation = case["dtype"], case["operation"]
    count = 33 * 65 if case["direction"] == "row-first" else 127
    indexes = np.arange(count)
    if dtype == "bool_":
        values = (indexes // 65) % 3 != 1 if count > 127 else indexes % 7 != 0
    elif operation == "prod":
        values = np.where(indexes % 3 == 0, -1 if dtype != "uint32" else 1, 1)
    else:
        values = indexes % 7 + 1
        if dtype == "float32":
            values = values * 0.25
    source = values.astype(np.bool_ if dtype == "bool_" else dtype)
    name = {"and": "all", "or": "any"}.get(operation, operation)
    kwargs = {"dtype": source.dtype} if name in {"sum", "prod"} else {}
    seed = getattr(np, name)(source, **kwargs) if count == 127 else None
    matrix = (
        np.broadcast_to(seed, (33, 65)) if seed is not None else source.reshape(33, 65)
    )
    rows = getattr(np, name)(matrix, axis=1, **kwargs)
    final = getattr(np, name)(rows, **kwargs)
    return source, seed, rows, final


def collect(mx, np, *, observe=None, dispatch_count=None):
    records = []
    for case in cases():
        source, expected_seed, expected_rows, expected_final = reference(np, case)
        start = dispatch_count() if dispatch_count else None
        value = mx.array(source, dtype=getattr(mx, case["dtype"]))
        operation = getattr(
            mx, {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
        )
        seed = operation(value) if case["direction"] == "scalar-first" else None
        matrix = (
            mx.broadcast_to(seed, (33, 65))
            if seed is not None
            else value.reshape(33, 65)
        )
        rows = operation(matrix, axis=1)
        final = operation(rows)
        mx.eval(final)
        record = case | {
            "rows": wire_values(np.array(rows).tolist()),
            "final": wire_values(np.array(final).item()),
            "seed": wire_values(np.array(seed).item()) if seed is not None else None,
            "resultDtype": str(final.dtype),
            "resultShape": list(final.shape),
        }
        if dispatch_count:
            record.update(dispatchStart=start, dispatchEnd=dispatch_count())
        records.append(record)
        if observe:
            observe(record)
        if (
            not values_match(record["rows"], expected_rows.tolist(), case["dtype"])
            or not values_match(record["final"], expected_final.item(), case["dtype"])
            or (
                seed is not None
                and not values_match(
                    record["seed"], expected_seed.item(), case["dtype"]
                )
            )
        ):
            raise RuntimeError(f"Mixed reduction mismatch: {case['id']}")
    return records


def validate(records, *, trace=None, offset=0):
    import numpy as np

    expected_cases = list(cases())
    if not isinstance(records, list) or len(records) != len(expected_cases):
        raise ValueError("Mixed reduction evidence must cover every case")
    cursor = offset
    for record, case in zip(records, expected_cases):
        _, seed, rows, final = reference(np, case)
        dtype = case["dtype"]
        if (
            any(record.get(key) != value for key, value in case.items())
            or not values_match(record.get("rows"), rows.tolist(), dtype)
            or not values_match(record.get("final"), final.item(), dtype)
            or record.get("resultDtype") != "mlx.core." + dtype.removesuffix("_")
            or record.get("resultShape") != []
            or (seed is None and record.get("seed") is not None)
            or (
                seed is not None
                and not values_match(record.get("seed"), seed.item(), dtype)
            )
        ):
            raise ValueError("Mixed reduction results are incomplete or incorrect")
        if trace is None:
            continue
        stages = []
        if seed is not None:
            stages.extend(
                [
                    (case["entry"], 127, [1, 1, 1], [32, 1, 1], [seed.item()]),
                    (
                        BOOLEAN_COPY_ENTRY if dtype == "bool_" else COPY_ENTRY,
                        2145,
                        [33, 33, 1],
                        [1, 1, 1],
                        None,
                    ),
                ]
            )
        stages.extend(
            [
                (case["rowEntry"], 2145, [1, 9, 1], [32, 1, 1], rows.tolist()),
                (case["entry"], 33, [1, 1, 1], [32, 1, 1], [final.item()]),
            ]
        )
        start, end = record.get("dispatchStart"), record.get("dispatchEnd")
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or end != start + len(stages)
            or end > len(trace)
        ):
            raise ValueError("Mixed reduction dispatch intervals are incomplete")
        guard = (
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
        for dispatch, (entry, size, groups, width, values) in zip(
            trace[start:end], stages
        ):
            dimensions = [dispatch.get("workgroupCount"), dispatch.get("workgroupSize")]
            if (
                any(
                    not isinstance(items, list)
                    or len(items) != 3
                    or any(type(item) is not int for item in items)
                    for items in dimensions
                )
                or type(dispatch.get("threads")) is not int
                or type(dispatch.get("dispatchVersion")) is not int
                or dispatch.get("dispatchVersion") != 2
                or dispatch.get("entry") != entry
                or dispatch.get("threads") != size
                or dimensions != [groups, width]
            ):
                raise ValueError(
                    "Mixed reduction did not preserve source dispatch order"
                )
            if values is None:
                if dispatch.get("copyGuardWords") != (
                    [index % 2 == 0 for index in range(32)]
                    if dtype == "bool_"
                    else [0x6A15BEEF] * 32
                ):
                    raise ValueError("Mixed reduction copy guard was changed")
            elif (
                not values_match(
                    dispatch.get("reductionValues"),
                    values,
                    dtype,
                    physical_boolean=dispatch.get("target") in {"directx", "opengl"},
                )
                or dispatch.get("reductionGuardValues") != guard
            ):
                raise ValueError(
                    "Mixed reduction intermediate values or guards are incorrect"
                )
        cursor = end
    if trace is not None and cursor != len(trace):
        raise ValueError("Mixed reduction trace contains unexpected dispatches")
