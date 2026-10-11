"""Whole-array MLX reductions with exact finite and special-value references."""

from __future__ import annotations

import math
import struct

from demos.integrations.mlx.portable_host import reduction_layout
from demos.integrations.mlx.portable_host.reduction_packages import ENTRIES, WIDTHS


def cases(widths=WIDTHS):
    counts = sorted(
        {width * 4 - 3 for width in widths}
        | {width * 4 for width in widths}
        | {4097, 8191, 65535}
    )
    for entry, dtype in ENTRIES.items():
        operation = entry.removeprefix("all_reduce_").removesuffix(dtype)
        for count in counts:
            if dtype == "bool_":
                values = [index % 7 != 0 for index in range(count)]
                expected = all(values) if operation == "and" else any(values)
            elif operation == "prod":
                values = [
                    1 if dtype == "uint32" or index % 3 else -1
                    for index in range(count)
                ]
                expected = math.prod(values)
            else:
                values = [
                    index % 23 - (0 if dtype == "uint32" else 11)
                    for index in range(count)
                ]
                if dtype == "float32":
                    values = [value * 0.25 for value in values]
                expected = {"sum": sum, "min": min, "max": max}[operation](values)
            yield {
                "id": f"{entry}-{count}",
                "entry": entry,
                "dtype": dtype,
                "count": count,
                "operation": operation,
                "inputs": values,
                "expected": expected,
            }

    for dtype in ("float32", "int32", "uint32", "bool_"):
        for layout in ("reverse", "transpose", "broadcast"):
            count = 256
            values = [
                bool((index + 1) % 3) if dtype == "bool_" else 1 + index % 11
                for index in range(count)
            ]
            if layout == "broadcast":
                values = values[:1]
                logical = values * count
            else:
                logical = values
            operation = "or" if dtype == "bool_" else "sum"
            entry = f"all_reduce_{operation}{dtype}"
            yield {
                "id": f"{entry}-{layout}",
                "entry": entry,
                "dtype": dtype,
                "count": count,
                "operation": operation,
                "inputs": values,
                "layout": layout,
                "expected": any(logical) if dtype == "bool_" else sum(logical),
            }

    for operation in ("sum", "prod", "min", "max"):
        for profile, values in (
            ("nan", [float("nan")] + [1.0] * 508),
            ("late-nan", [1.0] * 8190 + [float("nan")]),
            ("positive-infinity", [float("inf")] * 509),
            ("negative-zero", [-0.0] * 509),
        ):
            if "nan" in profile:
                expected = float("nan")
            elif profile == "negative-zero":
                expected = 0.0 if operation == "sum" else -0.0
            else:
                expected = float("inf")
            entry = f"all_reduce_{operation}float32"
            yield {
                "id": f"{entry}-{profile}",
                "entry": entry,
                "dtype": "float32",
                "count": len(values),
                "operation": operation,
                "inputs": values,
                "expected": expected,
            }

    for dtype in ("int32", "uint32"):
        for operation in ("sum", "prod"):
            values = ([2**31 - 1, 7, 13, 3] * 32) + [3]
            exact = sum(values) if operation == "sum" else math.prod(values)
            expected = exact % 2**32
            if dtype == "int32" and expected >= 2**31:
                expected -= 2**32
            entry = f"all_reduce_{operation}{dtype}"
            yield {
                "id": f"{entry}-overflow",
                "entry": entry,
                "dtype": dtype,
                "count": len(values),
                "operation": operation,
                "inputs": values,
                "expected": expected,
            }

    for operation in ("and", "or"):
        for value in (False, True):
            entry = f"all_reduce_{operation}bool_"
            yield {
                "id": f"{entry}-uniform-{value}",
                "entry": entry,
                "dtype": "bool_",
                "count": 8191,
                "operation": operation,
                "inputs": [value] * 8191,
                "expected": value,
            }

    # Append boundary cases without changing the established workload inventory.
    for entry, dtype in ENTRIES.items():
        operation = entry.removeprefix("all_reduce_").removesuffix(dtype)
        for count in (65536, 131075):
            if reduction_layout.stage(count, dtype)["workgroupSize"][0] not in widths:
                continue
            identity = (
                3 if operation == "min" else 1 if operation in {"prod", "and"} else 0
            )
            tail = 0 if operation in {"and", "min"} else 2 if operation == "prod" else 1
            values = [identity] * count
            values[-1] = tail
            if dtype == "bool_":
                values = list(map(bool, values))
            expected = {
                "sum": sum,
                "prod": math.prod,
                "min": min,
                "max": max,
                "and": all,
                "or": any,
            }[operation](values)
            yield {
                "id": f"{entry}-large-offset-{count}",
                "entry": entry,
                "dtype": dtype,
                "count": count,
                "operation": operation,
                "inputs": values,
                "offset": 7,
                "expected": expected,
            }


def wire(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "+infinity" if value > 0 else "-infinity"
    return value


def matches(actual, expected, dtype):
    if dtype == "float32":
        return isinstance(actual, float) and (
            math.isnan(actual)
            if math.isnan(expected)
            else struct.pack("<f", actual) == struct.pack("<f", expected)
        )
    return type(actual) is (bool if dtype == "bool_" else int) and actual == expected


def partial_reference(values, operation, dtype, plan):
    """Reduce exact-valued boundary inputs independently for each source row."""
    reduce = {
        "sum": sum,
        "prod": math.prod,
        "min": min,
        "max": max,
        "and": all,
        "or": any,
    }[operation]
    row_size = plan["rowSize"]
    return [
        float(value) if dtype == "float32" else value
        for row in range(plan["workgroupCount"][1])
        for value in [reduce(values[row * row_size : (row + 1) * row_size])]
    ]


def validate_large_pass(dispatch, count, plan, expected, dtype):
    from demos.integrations.mlx.portable_host.runtime import BOOLEAN_GUARD, COPY_GUARD

    target = dispatch.get("target")
    if target not in {"metal", "opengl", "directx"}:
        raise RuntimeError("Large reduction target is missing")
    guard = (
        (list(BOOLEAN_GUARD) if target == "metal" else list(map(int, BOOLEAN_GUARD)))
        if dtype == "bool_"
        else (
            [struct.unpack("<f", struct.pack("<I", word))[0] for word in COPY_GUARD]
            if dtype == "float32"
            else COPY_GUARD
        )
    )
    metadata = dispatch.get("reductionMetadata")
    if (
        not isinstance(metadata, dict)
        or metadata != {"in_size": count, "row_size": plan["rowSize"]}
        or any(type(value) is not int for value in metadata.values())
        or dispatch.get("reductionGuardValues") != guard
    ):
        raise RuntimeError("Large reduction metadata or output guards changed")
    actual = dispatch.get("reductionValues")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise RuntimeError("Large reduction intermediate readback is incomplete")
    if dtype == "bool_" and target != "metal":
        if any(type(value) is not int or value not in (0, 1) for value in actual):
            raise RuntimeError("Large reduction Boolean storage is invalid")
        actual = list(map(bool, actual))
    if any(
        not matches(value, reference, dtype)
        for value, reference in zip(actual, expected)
    ):
        raise RuntimeError(
            "Large reduction intermediate readback differs from reference"
        )


def collect(mx, widths=WIDTHS, *, observe=None, dispatch_count=None):
    records = []
    for case in cases(widths):
        start = dispatch_count() if dispatch_count is not None else None
        offset = case.get("offset", 0)
        source = mx.array(
            [3] * offset + case["inputs"] + [3] * offset,
            dtype=getattr(mx, case["dtype"]),
        )
        if offset:
            source = source[offset : offset + case["count"]]
        if case.get("layout") == "reverse":
            source = source[::-1]
        elif case.get("layout") == "transpose":
            source = source.reshape(16, 16).T
        elif case.get("layout") == "broadcast":
            source = mx.broadcast_to(source, (case["count"],))
        operation = getattr(
            mx, {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
        )
        result = operation(source)
        actual = result.item()
        record = {
            key: wire(value) for key, value in case.items() if key != "inputs"
        } | {
            "actual": wire(actual),
            "resultDtype": str(result.dtype),
            "resultShape": list(result.shape),
        }
        if dispatch_count is not None:
            record.update(dispatchStart=start, dispatchEnd=dispatch_count())
        records.append(record)
        if observe is not None:
            observe(record)
        if (
            result.dtype != source.dtype
            or result.shape != ()
            or not matches(actual, case["expected"], case["dtype"])
        ):
            raise RuntimeError(
                f"Incorrect native reduction: {case['entry']} count={case['count']}: {actual!r} != {case['expected']!r}"
            )
    return records


def validate(records, widths=WIDTHS, *, trace=None):
    expected_cases = list(cases(widths))
    if not isinstance(records, list) or len(records) != len(expected_cases):
        raise RuntimeError("Reduction evidence does not cover every workload")
    previous_end = None
    for case, record in zip(expected_cases, records):
        expected = {key: wire(value) for key, value in case.items() if key != "inputs"}
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Reduction workload identity changed")
        actual = record.get("actual")
        if isinstance(actual, str):
            actual = {
                "nan": float("nan"),
                "+infinity": float("inf"),
                "-infinity": -float("inf"),
            }.get(actual)
        dtype_name = "bool" if case["dtype"] == "bool_" else case["dtype"]
        if (
            record.get("resultShape") != []
            or record.get("resultDtype") != f"mlx.core.{dtype_name}"
            or not matches(actual, case["expected"], case["dtype"])
        ):
            raise RuntimeError(f"Incorrect reduction readback: {case['id']}")
        if trace is None:
            continue
        start, end = record.get("dispatchStart"), record.get("dispatchEnd")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(trace)
            or (previous_end is not None and start != previous_end)
        ):
            raise RuntimeError("Reduction dispatch interval is missing or inconsistent")
        previous_end = end
        dispatches = trace[start:end]
        if case.get("layout") in {"reverse", "broadcast"}:
            expected_copy = (
                "ggn2_dynamic_copybool_bool_"
                if case["dtype"] == "bool_"
                else "ggn2_dynamic_copyuint32uint32"
            )
            if dispatches[0].get("entry") != expected_copy:
                raise RuntimeError("Noncontiguous reduction input was not materialized")
            dispatches = dispatches[1:]
        rows = reduction_layout.stage(case["count"], case["dtype"])["workgroupCount"][1]
        counts = [case["count"]] + ([rows] if rows > 1 else [])
        if len(dispatches) != len(counts):
            raise RuntimeError("Reduction did not execute every upstream pass")
        partials = case["inputs"]
        for dispatch, count in zip(dispatches, counts):
            plan = reduction_layout.stage(count, case["dtype"])
            if (
                dispatch.get("entry") != case["entry"]
                or dispatch.get("threads") != count
                or any(
                    dispatch.get(key) != plan[key]
                    for key in ("workgroupCount", "workgroupSize")
                )
                or dispatch.get("dispatchVersion") != 3
                or type(dispatch.get("dispatchVersion")) is not int
                or type(dispatch.get("threads")) is not int
                or any(
                    type(value) is not int
                    for key in ("workgroupCount", "workgroupSize")
                    for value in dispatch[key]
                )
                or "reductionGuardValues" not in dispatch
            ):
                raise RuntimeError("Reduction dispatch does not match the source plan")
            if case.get("offset"):
                partials = partial_reference(
                    partials, case["operation"], case["dtype"], plan
                )
                validate_large_pass(dispatch, count, plan, partials, case["dtype"])
    if trace is not None and previous_end != len(trace):
        raise RuntimeError("Native trace contains unaccounted reduction dispatches")
