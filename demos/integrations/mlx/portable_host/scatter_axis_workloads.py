"""Replacement and additive axis-scatter through public MLX operations."""

import math

LAYOUTS = (
    "dense",
    "transposed",
    "source-strided",
    "source-broadcast",
    "source-reversed",
    "update-strided",
    "update-broadcast",
    "update-reversed",
    "index-strided",
    "index-reversed",
    "both-strided",
    "flattened",
    "alias",
    "empty",
)

# Workloads past one launch axis: each maps to its output shape, the scattered
# axis, its index rows along that axis, and whether the update and index views
# are strided. The first is the fused cross-entropy gradient reference's shape.
WIDE = {
    "wide-output": ((4, 7, 4096), 2, 1, False),
    "wide-rows": ((2,), 0, 70001, True),
    "wide-leading": ((131073, 2), 1, 1, False),
    "wide-slab": ((300, 300, 2), 2, 1, True),
    "wide-split": ((3, 2), 1, 70000, False),
}


def cases():
    signed = []
    for dtype in ("int32", "uint32"):
        for operation in ("none", "sum"):
            for i, layout in enumerate(LAYOUTS):
                case = {
                    "id": f"{dtype}-{operation}-{layout}",
                    "dtype": dtype,
                    "operation": operation,
                    "layout": layout,
                    "axis": (
                        None
                        if layout == "flattened"
                        else (-1 if layout == "alias" else i % 3)
                    ),
                    "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
                }
                yield case
                if dtype == "int32":
                    signed.append(
                        {**case, "id": case["id"] + "-signed", "values": "signed"}
                    )
    yield from signed
    yield from float_cases()
    yield from wide_cases()


def wide_cases():
    """Int32 and float32 workloads whose launches split past 65535 groups."""
    for i, layout in enumerate(WIDE):
        for dtype in ("int32", "float32"):
            for operation in ("none", "sum"):
                yield {
                    "id": f"{dtype}-{operation}-{layout}",
                    "dtype": dtype,
                    "operation": operation,
                    "layout": layout,
                    "axis": WIDE[layout][1],
                    "index_dtype": ("int32", "uint32", "int64", "uint64", "int32")[i],
                }


def wide_arrays(xp, np, case):
    shape, axis, rows, strided = WIDE[case["layout"]]
    floating = case["dtype"] == "float32"
    values = np.arange(math.prod(shape)).reshape(shape) % 97 - 48
    source = xp.array((values / 4 if floating else values).astype(case["dtype"]))
    index_shape = list(shape)
    index_shape[axis] = rows
    raw = np.arange(math.prod(index_shape)).reshape(index_shape)
    indices = raw % shape[axis]
    # Duplicate replacements carry identical values for every possible ordering.
    updates = indices + 17 if case["operation"] == "none" else raw % 11 + 1
    if floating:
        updates = (updates - 6 if case["operation"] == "sum" else 5 - updates) / 4
    updates = updates.astype(case["dtype"])
    if case["index_dtype"].startswith("int"):
        indices = np.where(raw % 2, indices - shape[axis], indices)
    indices = indices.astype(case["index_dtype"])
    if strided:
        # Stride the last axis that has more than one element, so the views'
        # strides are observable rather than those of a unit dimension.
        step = max(i for i, size in enumerate(index_shape) if size > 1)
        view = (slice(None),) * step + (slice(None, None, 2),)
        return (
            source,
            xp.array(np.repeat(indices, 2, axis=step))[view],
            xp.array(np.repeat(updates, 2, axis=step))[view],
        )
    return source, xp.array(indices), xp.array(updates)


def float_cases():
    """Float32 replacements and sums of quarter-step values.

    Every partial sum of these values is exact, so additive updates agree in
    any atomic order.
    """
    for operation in ("none", "sum"):
        for i, layout in enumerate(LAYOUTS):
            yield {
                "id": f"float32-{operation}-{layout}",
                "dtype": "float32",
                "operation": operation,
                "layout": layout,
                "axis": (
                    None
                    if layout == "flattened"
                    else (-1 if layout == "alias" else i % 3)
                ),
                "index_dtype": ("int32", "uint32", "int64", "uint64")[(i + 1) % 4],
            }


def arrays(xp, np, case):
    if case["layout"] in WIDE:
        return wide_arrays(xp, np, case)
    signed = case.get("values") == "signed"
    floating = case["dtype"] == "float32"
    values = np.arange(48).reshape(4, 3, 4) - (24 if signed or floating else 0)
    source = xp.array((values / 4 if floating else values).astype(case["dtype"]))
    layout = case["layout"]
    if layout == "transposed":
        source = xp.transpose(source, (1, 0, 2))
    elif layout == "source-strided":
        source = source[::2]
    elif layout == "source-broadcast":
        source = xp.broadcast_to(source[:1], source.shape)
    elif layout == "source-reversed":
        source = source[::-1]
    elif layout == "flattened":
        source = xp.reshape(source, (-1,))
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    shape = list(source.shape)
    shape[axis] = 0 if layout == "empty" else 5
    if layout == "alias":
        updates = source[..., :2]
        indices = xp.broadcast_to(
            xp.array(np.array([1, 0], dtype=case["index_dtype"])), updates.shape
        )
        return source, indices, updates
    count = int(np.prod(shape))
    raw = np.arange(count).reshape(shape)
    indices = raw % source.shape[axis]
    # Duplicate replacements carry identical values for every possible ordering.
    updates = indices + 17 if case["operation"] == "none" else raw % 11 + 1
    if floating:
        # Negative and fractional quarter steps; sums also cross zero.
        updates = (updates - 6 if case["operation"] == "sum" else 5 - updates) / 4
    updates = updates.astype(case["dtype"])
    if signed:
        updates = -updates
    if case["index_dtype"].startswith("int"):
        indices = np.where(raw % 2, indices - source.shape[axis], indices)
    indices = indices.astype(case["index_dtype"])
    if layout in {"update-strided", "both-strided"}:
        updates = xp.array(np.repeat(updates, 2, axis=-1))[..., ::2]
    elif layout == "update-broadcast":
        updates = xp.broadcast_to(
            xp.array(
                np.array(
                    -5.75 if floating else -23 if signed else 23, dtype=case["dtype"]
                )
            ),
            tuple(shape),
        )
    else:
        updates = xp.array(updates)
    if layout in {"index-strided", "both-strided"}:
        indices = xp.array(np.repeat(indices, 2, axis=-1))[..., ::2]
    else:
        indices = xp.array(indices)
    if layout in {"update-reversed", "index-reversed"}:
        # Reversing both retains matching update/destination identities while
        # exercising negative-stride materialization before native dispatch.
        updates, indices = updates[::-1], indices[::-1]
    return source, indices, updates


def reference(np, case):
    source, indices, updates = arrays(np, np, case)
    expected = source.copy() if case["operation"] == "none" else np.zeros_like(source)
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    for coordinate in np.ndindex(indices.shape):
        destination = list(coordinate)
        destination[axis] = int(indices[coordinate]) % source.shape[axis]
        destination = tuple(destination)
        if case["operation"] == "none":
            expected[destination] = updates[coordinate]
        else:
            expected[destination] += updates[coordinate]
    return source, indices, updates, expected


def expression(mx, np, case):
    source, indices, updates = arrays(mx, np, case)
    if case["operation"] == "none":
        result = mx.put_along_axis(source, indices, updates, axis=case["axis"])
    else:
        _, gradients = mx.vjp(
            lambda value: mx.take_along_axis(value, indices, axis=case["axis"]),
            [source],
            [updates],
        )
        result = gradients[0]
    return source, result
