"""Integer and Boolean absolute-value storage contracts used by MLX comparisons."""

from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.packages import BOOLEAN_COPY_ENTRY, COPY_ENTRY
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
)
from demos.integrations.mlx.portable_host.selection_workloads import words


def cases(np):
    for dtype in ("int32", "uint32", "bool_"):
        values = {
            "int32": [-2147483648, -2147483647, -17, -1, 0, 1, 2147483647],
            "uint32": [0, 1, 17, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 31],
            "bool_": [False, True, True, False, True, False, False],
        }[dtype]
        base = np.asarray(values * 5, dtype=dtype)
        layouts = {
            "empty": base[:0],
            "scalar": base[0].reshape(()),
            "vector": base[:7],
            "tail": base[:33],
            "matrix": base[:15].reshape(3, 5),
            "transpose": base[:15].reshape(3, 5).T,
            "broadcast": np.broadcast_to(base[:5], (3, 5)),
            "reverse": base[16:1:-1],
        }
        for layout, value in layouts.items():
            yield dtype, layout, value


def expected(np, value):
    if value.dtype.name != "int32":
        return value.copy()
    return (
        np.asarray([abs(int(v)) for v in value.reshape(-1)], dtype="uint32")
        .view("int32")
        .reshape(value.shape)
    )


def physical(np, value):
    compact = (
        value[
            tuple(
                slice(0, 1) if stride == 0 else slice(None) for stride in value.strides
            )
        ]
        if value.ndim
        else value
    )
    return (
        compact.ravel(order="K")
        if compact.flags.c_contiguous or compact.flags.f_contiguous
        else np.ascontiguousarray(value).reshape(-1)
    )


def validate(records, trace, *, native):
    import numpy as np

    required = list(cases(np))
    if len(records) != len(required):
        raise ValueError("Absolute-value coverage is incomplete")
    cursor = 0
    for record, (dtype, layout, value) in zip(records, required):
        entries = (
            (
                [BOOLEAN_COPY_ENTRY if dtype == "bool_" else COPY_ENTRY]
                if layout == "reverse"
                else []
            )
            + [f"v_Abs{dtype}{dtype}"]
            if native and value.size
            else []
        )
        if (
            record.get("dtype") != dtype
            or record.get("layout") != layout
            or record.get("resultDtype") != ("bool" if dtype == "bool_" else dtype)
            or record.get("shape") != list(value.shape)
            or record.get("inputWords") != words(np, value)
            or record.get("inputUnchanged") is not True
            or record.get("resultWords") != words(np, expected(np, value))
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != len(entries)
        ):
            raise ValueError("Absolute-value result or storage differs")
        events = trace[cursor : cursor + len(entries)]
        cursor += len(entries)
        if [event["entry"] for event in events] != entries:
            raise ValueError("Absolute-value dispatch sequence differs")
        if entries:
            event = events[-1]
            values = expected(np, physical(np, value)).tolist()
            guard = BOOLEAN_GUARD if dtype == "bool_" else COPY_GUARD
            if dtype == "bool_" and event.get("target") != "metal":
                values, guard = [int(v) for v in values], [int(v) for v in guard]
            if (
                event.get("absoluteValues") != values
                or event.get("unaryGuardValues") != guard
                or event.get("threads") != len(values)
                or event.get("dispatchVersion") != DISPATCH_VERSION
                or event.get("workgroupCount") != [len(values), 1, 1]
                or event.get("workgroupSize") != [1, 1, 1]
            ):
                raise ValueError("Absolute-value native readback differs")
    if cursor != len(trace):
        raise ValueError("Absolute-value trace contains unexpected dispatches")


def run(mx, np, host, save):
    records = []
    for dtype, layout, value in cases(np):
        operand = mlx_operand(mx, np, value)
        start = host.dispatch_count if host else 0
        result = np.array(mx.abs(operand))
        records.append(
            {
                "dtype": dtype,
                "layout": layout,
                "shape": list(result.shape),
                "resultDtype": result.dtype.name,
                "inputWords": words(np, value),
                "inputUnchanged": words(np, np.array(operand)) == words(np, value),
                "resultWords": words(np, result),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records
