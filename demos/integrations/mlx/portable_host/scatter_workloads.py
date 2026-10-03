"""General indexed updates through public MLX array operations."""

LAYOUTS = (
    "dense",
    "source-transposed",
    "source-broadcast",
    "source-reversed",
    "update-strided",
    "update-broadcast",
    "index-strided",
    "index-reversed",
    "multiple",
    "scalar",
    "alias",
    "empty",
)


def cases():
    for dtype in ("int32", "uint32"):
        for operation in ("none", "sum", "min", "max"):
            for i, layout in enumerate(LAYOUTS):
                yield {
                    "id": f"{dtype}-{operation}-{layout}",
                    "dtype": dtype,
                    "operation": operation,
                    "layout": layout,
                    "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
                }
    for work in (4, 8, 16, 32):
        for operation in ("none", "sum", "min", "max"):
            yield {
                "id": f"int32-{operation}-work{work}",
                "dtype": "int32",
                "operation": operation,
                "layout": f"work{work}",
                "index_dtype": "int64",
            }
    for dtype in ("int32", "uint32"):
        for i, layout in enumerate((*LAYOUTS, "work4", "work8", "work16", "work32")):
            yield {
                "id": f"{dtype}-prod-{layout}",
                "dtype": dtype,
                "operation": "prod",
                "layout": layout,
                "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
            }


def arrays(xp, np, case):
    layout = case["layout"]
    shape = (2, 2) if layout.startswith("work") else (3, 4)
    source = xp.array(
        (np.arange(np.prod(shape)).reshape(shape) + 10).astype(case["dtype"])
    )
    if layout == "source-transposed":
        source = xp.transpose(source)
    elif layout == "source-broadcast":
        source = xp.broadcast_to(source[:1], source.shape)
    elif layout == "source-reversed":
        source = source[::-1]
    index_shape = {
        "scalar": (),
        "empty": (0,),
        "alias": (2,),
        "multiple": (2, 3),
        "work4": (3, 5),
        "work8": (3, 7),
        "work16": (5, 13),
        "work32": (7, 19),
    }.get(layout, (5,))
    raw = np.arange(np.prod(index_shape, dtype=int)).reshape(index_shape)
    first = raw % source.shape[0]
    two = layout == "multiple" or layout.startswith("work")
    destinations = [first]
    if two:
        destinations.append((raw // 2) % source.shape[1])
    indices = []
    for axis, index in enumerate(destinations):
        if case["index_dtype"].startswith("int"):
            index = np.where(raw % 2, index - source.shape[axis], index)
        index = index.astype(case["index_dtype"])
        if layout == "index-strided":
            index = xp.array(np.repeat(index, 2, axis=-1))[..., ::2]
        else:
            index = xp.array(index)
        indices.append(index)
    if two:
        update = first * source.shape[1] + destinations[1] + 20
    else:
        update = (
            np.expand_dims(first, -1) * source.shape[1]
            + np.arange(source.shape[1])
            + 20
        )
    # Repeated replacements have identical values, independent of thread order.
    if case["operation"] == "prod":
        positions = np.arange(update.size).reshape(update.shape)
        # Bound intermediate products independently of contending update order.
        update = np.where(positions < 8, 2, 1)
        if case["dtype"] == "int32":
            update = np.where(positions % 7 == 1, -update, update)
        update = np.where(positions == 0, 0, update)
    elif case["operation"] != "none":
        update = np.arange(update.size).reshape(update.shape) % 19 + 1
    update = update.astype(case["dtype"])
    if layout == "update-strided":
        update = xp.array(np.repeat(update, 2, axis=-1))[..., ::2]
    elif layout == "update-broadcast":
        update = xp.broadcast_to(
            xp.array(
                np.array(2 if case["operation"] == "prod" else 7, dtype=case["dtype"])
            ),
            update.shape,
        )
    elif layout == "alias":
        update = source[:2]
        indices = [xp.array(np.array([1, 0], dtype=case["index_dtype"]))]
    else:
        update = xp.array(update)
    if layout == "index-reversed":
        indices = [index[::-1] for index in indices]
        update = update[::-1]
    return source, indices, update


def reference(np, case):
    source, indices, updates = arrays(np, np, case)
    expected = source.copy()
    operation = case["operation"]
    for coordinate in np.ndindex(indices[0].shape):
        destination = tuple(int(index[coordinate]) for index in indices)
        if operation == "none":
            expected[destination] = updates[coordinate]
        elif operation == "sum":
            expected[destination] += updates[coordinate]
        elif operation == "prod":
            expected[destination] *= updates[coordinate]
        elif operation == "min":
            expected[destination] = np.minimum(
                expected[destination], updates[coordinate]
            )
        else:
            expected[destination] = np.maximum(
                expected[destination], updates[coordinate]
            )
    return source, indices, updates, expected


def expression(mx, np, case):
    source, indices, updates = arrays(mx, np, case)
    index = tuple(indices)
    if case["operation"] == "none":
        result = mx.array(source)
        result[index] = updates
    else:
        method = {"sum": "add", "prod": "multiply", "min": "minimum", "max": "maximum"}[
            case["operation"]
        ]
        result = getattr(source.at[index], method)(updates)
    return source, result
