"""Exercise upstream column plans, including native intermediate readbacks."""

import math
import struct

from demos.integrations.mlx.portable_host import column_small_layout
from demos.integrations.mlx.portable_host.column_small_layout import (
    SMALL_COLUMN_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import COLUMN_ENTRIES
from demos.integrations.mlx.portable_host.row_workloads import values_match, wire_values

# Fewer than 32 reduced elements per output. Each template's reduction rank,
# with a partial last threadgroup of columns for the rank-one template.
SMALL_SHAPES = {
    1: ((5, 130), (0,)),
    2: ((2, 2, 3, 33), (0, 2)),
    5: ((2, 2, 2, 2, 3, 32), (0, 2, 4)),
}
# Further sum geometries: one lane, narrow lanes, the widest threadgroup and
# 31 reduced rows, the largest small plan, 131,072 elements, and a stride of
# 65,536 columns over two axes, past the bound of one axis.
SMALL_SUM_SHAPES = (
    (2, 3),
    (3, 16),
    (31, 32),
    (8, 128),
    (8, 2048),
    (8, 16384),
    (8, 128, 512),
)
# Looped and two-pass plans over more than 65,535 elements, for these entries'
# rank-one templates.
WIDE_OPERATIONS = ("sumfloat32", "maxuint32")
WIDE_SHAPES = {"looped": (40, 2048), "2pass": (300, 256)}
SMALL_SUM_ENTRY = "col_reduce_small_1_reduce_sumfloat32"


def _small_cases():
    for entry, dtype in SMALL_COLUMN_ENTRIES.items():
        dimension = int(entry.split("_")[3])
        shape, axes = SMALL_SHAPES[dimension]
        case = {
            "id": f"{entry}-small",
            "entry": entry,
            "dtype": dtype,
            "operation": entry.rsplit("reduce_", 1)[1].removesuffix(dtype),
            "shape": shape,
            "axes": axes,
            "layout": "dense",
        }
        yield case
        if entry == SMALL_SUM_ENTRY:
            for shape in SMALL_SUM_SHAPES:
                yield case | {
                    "id": f"{entry}-{'x'.join(map(str, shape))}",
                    "shape": shape,
                }
        if dtype == "float32" and dimension == 1:
            for profile in ("early-nan", "late-nan", "positive-infinity"):
                yield case | {"id": f"{entry}-{profile}", "profile": profile}


def small_variants():
    """The small column variants the workloads launch, as cache keys."""
    keys = set()
    for case in _small_cases():
        size, stride, _ = small_plan(case)
        lanes, rows = column_small_layout.geometry(stride, size)
        keys.add(column_small_layout.variant(case["entry"], (lanes, rows, 1)))
    return keys


def column_stride(case):
    """The reduction stride MLX plans: the dense axes after the last reduced one."""
    return math.prod(case["shape"][max(case["axes"]) + 1 :])


def small_plan(case):
    """``(reduced elements, columns, outputs)`` of one small column case."""
    shape, axes = case["shape"], case["axes"]
    total = math.prod(shape[axis] for axis in axes)
    columns = column_stride(case)
    return total, columns, math.prod(shape) // total // columns


def cases():
    yield from _looped_cases()
    yield from _small_cases()


def _looped_cases():
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
            if entry.endswith(WIDE_OPERATIONS):
                mode = "2pass" if two_pass else "looped"
                yield case | {"id": f"{entry}-wide", "shape": WIDE_SHAPES[mode]}
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
            stride = column_stride(case) if stage == 0 else expected.size
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
            size = [256, 1, 1]
            if entry in SMALL_COLUMN_ENTRIES:
                total = logical.size // expected.size
                lanes, rows = column_small_layout.geometry(stride, total)
                blocks = -(-stride // column_small_layout.N_READS)
                groups, size = [-(-blocks // lanes), outer, 1], [lanes, rows, 1]
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
                or dispatch.get("workgroupSize") != size
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
