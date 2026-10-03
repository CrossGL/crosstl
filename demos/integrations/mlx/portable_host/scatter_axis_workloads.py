"""Replacement and additive axis-scatter through public MLX operations."""

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


def cases():
    for dtype in ("int32", "uint32"):
        for operation in ("none", "sum"):
            for i, layout in enumerate(LAYOUTS):
                yield {
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


def arrays(xp, np, case):
    source = xp.array(np.arange(48).reshape(4, 3, 4).astype(case["dtype"]))
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
    updates = (indices + 17 if case["operation"] == "none" else raw % 11 + 1).astype(
        case["dtype"]
    )
    if case["index_dtype"].startswith("int"):
        indices = np.where(raw % 2, indices - source.shape[axis], indices)
    indices = indices.astype(case["index_dtype"])
    if layout in {"update-strided", "both-strided"}:
        updates = xp.array(np.repeat(updates, 2, axis=-1))[..., ::2]
    elif layout == "update-broadcast":
        updates = xp.broadcast_to(
            xp.array(np.array(23, dtype=case["dtype"])), tuple(shape)
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
