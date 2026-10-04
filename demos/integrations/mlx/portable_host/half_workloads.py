"""Exact half copies and float casts through the public MLX array API."""

import struct

from demos.integrations.mlx.portable_host.binary_workloads import mlx_operand
from demos.integrations.mlx.portable_host.packages import (
    COPY_ENTRY,
    HALF_CAST_ENTRIES,
    HALF_COPY_ENTRY,
    HALF_ENTRIES,
)

LAYOUTS = (
    "empty",
    "scalar",
    "vector",
    "tail",
    "matrix",
    "transpose",
    "reverse",
    "broadcast",
)
WORDS = [
    0,
    0x8000,
    1,
    0x8001,
    0x3FF,
    0x400,
    0x3C00,
    0xBC00,
    0x7BFF,
    0xFBFF,
    0x7C00,
    0xFC00,
    0x7E01,
    0xFE55,
    0x7D01,
    0xFD55,
]


def cases():
    return [
        dict(entry=entry, layout=layout) for entry in HALF_ENTRIES for layout in LAYOUTS
    ] + [
        dict(entry=HALF_COPY_ENTRY, layout="exhaustive", start=start)
        for start in range(0, 65536, 16384)
    ]


def inputs(np, case):
    if case["layout"] == "exhaustive":
        return np.arange(case["start"], case["start"] + 16384, dtype=np.uint16).view(
            np.float16
        )[::-1]
    if case["entry"] == HALF_COPY_ENTRY:
        data = np.asarray(WORDS, dtype=np.uint16).view(np.float16)
    elif HALF_CAST_ENTRIES[case["entry"]][0] == "float16":
        data = np.asarray(WORDS[:10], dtype=np.uint16).view(np.float16)
    else:
        data = np.asarray(
            [
                0.0,
                -0.0,
                1.0,
                -2.0,
                1.00048828125,
                1.00146484375,
                2**-24,
                -(2**-24),
                2**-25,
                65504.0,
                -65504.0,
                0.33333334,
            ],
            dtype=np.float32,
        )
    data = np.tile(data, 40)
    layout = case["layout"]
    if layout == "empty":
        return data[:0]
    if layout == "scalar":
        return data[:1].reshape(())
    if layout == "vector":
        return data[:16]
    if layout == "tail":
        return data[:257]
    if layout == "matrix":
        return data[:15].reshape(3, 5)
    if layout == "transpose":
        return data[:15].reshape(3, 5).T
    if layout == "reverse":
        return data[34:0:-2]
    if layout == "broadcast":
        return np.broadcast_to(data[:5], (3, 5))
    raise ValueError("Unknown half workload layout")


def reference(np, case, source):
    return (
        source.copy()
        if case["entry"] == HALF_COPY_ENTRY
        else source.astype(HALF_CAST_ENTRIES[case["entry"]][1])
    )


def words(np, value):
    return (
        np.ascontiguousarray(value)
        .reshape(-1)
        .view("uint16" if value.dtype == np.float16 else "uint32")
        .tolist()
    )


def entries(case, source):
    if not source.size:
        return []
    if case["entry"] == HALF_COPY_ENTRY:
        return [] if source.flags.c_contiguous else [HALF_COPY_ENTRY]
    copy = HALF_COPY_ENTRY if source.dtype.name == "float16" else COPY_ENTRY
    return ([] if source.flags.c_contiguous else [copy]) + [case["entry"]]


def run(mx, np, host, save):
    records = []
    for case in cases():
        source = inputs(np, case)
        operand = mlx_operand(mx, np, source)
        start = host.dispatch_count if host else 0
        result = (
            mx.contiguous(operand)
            if case["entry"] == HALF_COPY_ENTRY
            else operand.astype(getattr(mx, HALF_CAST_ENTRIES[case["entry"]][1]))
        )
        actual = np.array(result)
        records.append(
            {
                **case,
                "inputWords": words(np, np.array(operand)),
                "resultWords": words(np, actual),
                "shape": list(actual.shape),
                "dtype": actual.dtype.name,
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        save(records)
    return records


def widen_reference(word):
    if word & 0x7C00 == 0x7C00:
        return ((word & 0x8000) << 16) | 0x7F800000 | ((word & 1023) << 13)
    return struct.unpack(
        "<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", word))[0])
    )[0]


def validate(records, trace, *, native):
    import numpy as np

    if not isinstance(records, list) or len(records) != len(cases()):
        raise ValueError("Half workload inventory is incomplete")
    cursor = 0
    for record, case in zip(records, cases()):
        source = inputs(np, case)
        expected = reference(np, case, source)
        sequence = entries(case, source) if native else []
        wanted = {
            **case,
            "inputWords": words(np, source),
            "resultWords": words(np, expected),
            "shape": list(expected.shape),
            "dtype": expected.dtype.name,
            "dispatchCount": len(sequence),
        }
        if record != wanted:
            raise ValueError(f"Half workload result differs: {case}")
        events = trace[cursor : cursor + len(sequence)]
        cursor += len(sequence)
        if [event.get("entry") for event in events] != sequence:
            raise ValueError("Half workload dispatch sequence differs")
        for event in events:
            if event.get("dispatchVersion") != 3 or event.get("workgroupSize") != [
                1,
                1,
                1,
            ]:
                raise ValueError("Half workload launch contract differs")
        if not events:
            continue
        event = events[-1]
        half = expected.dtype == np.float16
        bits = words(np, expected)
        guard = [0x3555 if half else 0x6A15BEEF] * 32
        physical, physical_guard = bits, guard
        if half and event["target"] == "opengl":
            physical = [widen_reference(word) for word in bits]
            physical_guard = [widen_reference(word) for word in guard]
        storage = "float32" if not half or event["target"] == "opengl" else "float16"
        if (
            event.get("halfStorage")
            != {
                "logicalType": expected.dtype.name,
                "physicalType": storage,
                "encoding": (
                    "ieee754-binary32" if storage == "float32" else "ieee754-binary16"
                ),
                "values": physical,
                "guardValues": physical_guard,
                "logicalWords": bits,
            }
            or event.get("threads") != expected.size
        ):
            raise ValueError("Half native readback, guards or logical storage differs")
    if cursor != len(trace):
        raise ValueError("Half workload contains unexpected dispatches")
