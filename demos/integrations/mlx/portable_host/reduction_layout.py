"""Validate bounded whole-array reduction passes against upstream geometry."""

import ctypes


def stage(count):
    if type(count) is not int or not 1 <= count <= 65535:
        raise ValueError("Whole-array reductions require 1 to 65535 stored elements")
    rows = 1 if count <= 4096 else 128
    row_size = (count + rows - 1) // rows
    width = min(1024, ((row_size + 127) // 128) * 32)
    return {
        "rowSize": row_size,
        "workgroupCount": [1, rows, 1],
        "workgroupSize": [width, 1, 1],
    }


def validate(buffers, count, execution):
    plan = stage(count)
    if execution != {key: plan[key] for key in ("workgroupCount", "workgroupSize")}:
        raise ValueError("Native reduction launch does not match the upstream plan")
    for name, expected in (("in_size", count), ("row_size", plan["rowSize"])):
        buffer = buffers[name]
        value = ctypes.cast(buffer.data, ctypes.POINTER(ctypes.c_uint64))[0]
        if value != expected:
            raise ValueError(
                "Native reduction metadata does not match the upstream plan"
            )
    return plan
