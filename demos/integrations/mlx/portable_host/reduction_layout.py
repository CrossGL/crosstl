"""Validate whole-array reduction passes against upstream geometry."""

import ctypes

# Generated buffer indices share the copy adapter's signed 32-bit bound.
MAX_ELEMENTS = 2**31 - 1
ITEM_SIZES = {"float32": 4, "int32": 4, "uint32": 4, "bool_": 1, "bfloat16": 2}


def stage(count, dtype="float32"):
    if type(count) is not int or not 1 <= count <= MAX_ELEMENTS:
        raise ValueError(
            f"Whole-array reductions require 1 to {MAX_ELEMENTS} stored elements"
        )
    if not isinstance(dtype, str) or dtype not in ITEM_SIZES:
        raise ValueError("Unsupported whole-array reduction dtype")
    # MLX plans using logical bytes, not the target's widened storage.
    rows = 1 if count <= 4096 else 128 if count * ITEM_SIZES[dtype] <= 2**26 else 4096
    row_size = (count + rows - 1) // rows
    width = min(1024, ((row_size + 127) // 128) * 32)
    return {
        "rowSize": row_size,
        "workgroupCount": [1, rows, 1],
        "workgroupSize": [width, 1, 1],
    }


def validate(buffers, count, execution, dtype="float32"):
    plan = stage(count, dtype)
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
