"""General indexed updates through public MLX array operations.

The ``narrow`` profile scatters into the 8- and 16-bit integers, float16 and
bfloat16, whose destinations MLX updates as lanes of packed atomic words.
NumPy has no bfloat16, so its arrays here hold bfloat16 values as float32;
``storage_words`` turns any profile's array into its storage words.
"""

from demos.integrations.mlx.portable_host import scatter_arithmetic

FLOATS = frozenset({"float32", "float16", "bfloat16"})
NARROW_TYPES = ("uint8", "int8", "uint16", "int16", "float16", "bfloat16")
OPERATIONS = ("none", "sum", "prod", "min", "max")
INDEX_TYPES = ("int32", "uint32", "int64", "uint64")
# Layouts whose neighbouring destinations share packed words across threads.
NARROW_LAYOUTS = (
    "dense",
    "update-broadcast",
    "update-strided",
    "index-reversed",
    "multiple",
    "scalar",
    "work4",
)
PROFILES = ("integer", "float32", "narrow")

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

    for dtype in ("int32", "uint32"):
        yield {
            "id": f"{dtype}-none-capacity",
            "dtype": dtype,
            "operation": "none",
            "layout": "capacity",
            "index_dtype": "int64",
        }


def float_cases():
    for operation in ("none", "sum", "prod", "min", "max"):
        for i, layout in enumerate((*LAYOUTS, "work4", "work8", "work16", "work32")):
            yield {
                "id": f"float32-{operation}-{layout}",
                "dtype": "float32",
                "operation": operation,
                "layout": layout,
                "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
            }
    yield {
        "id": "float32-none-payloads",
        "dtype": "float32",
        "operation": "none",
        "layout": "payloads",
        "index_dtype": "int64",
    }


def narrow_cases():
    for dtype in NARROW_TYPES:
        for operation in OPERATIONS:
            for i, layout in enumerate(NARROW_LAYOUTS):
                yield {
                    "id": f"{dtype}-{operation}-{layout}",
                    "dtype": dtype,
                    "operation": operation,
                    "layout": layout,
                    "index_dtype": INDEX_TYPES[i % 4],
                }
    # Integer sums and products wrap within their lanes, never into a neighbour.
    for dtype in NARROW_TYPES[:4]:
        for operation in ("sum", "prod"):
            yield {
                "id": f"{dtype}-{operation}-wrap",
                "dtype": dtype,
                "operation": operation,
                "layout": "wrap",
                "index_dtype": "int64",
            }
    # Contending 16-bit float sums that round run in index order.
    for dtype in ("float16", "bfloat16"):
        yield {
            "id": f"{dtype}-sum-rounding",
            "dtype": dtype,
            "operation": "sum",
            "layout": "rounding",
            "index_dtype": "int64",
        }


def profile_cases(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown scatter workload profile: {profile!r}")
    return {"integer": cases, "float32": float_cases, "narrow": narrow_cases}[profile]()


def host_array(xp, np, values, dtype):
    """``values`` as ``dtype``; NumPy keeps bfloat16 values as float32."""
    if dtype != "bfloat16":
        return xp.array(np.asarray(values).astype(dtype))
    values = np.asarray(values).astype(np.float32)
    if xp is np:
        scatter_arithmetic.storage_words(np, dtype, values)
        return values
    return xp.array(values).astype(xp.bfloat16)


def storage_words(np, dtype, array):
    """The unsigned storage words of a NumPy result or workload array."""
    if dtype == "bfloat16":
        return scatter_arithmetic.storage_words(np, dtype, array)
    array = np.ascontiguousarray(array)
    unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
    return array.view(unsigned[array.dtype.itemsize]).reshape(-1).tolist()


def host_words(mx, np, array):
    """The unsigned storage words of an MLX array of any profile's type."""
    if array.dtype == mx.bfloat16:
        array = array.view(mx.uint16)
    return storage_words(np, None, np.array(array))


# Destination rows of the wrapping and rounding workloads' six index rows.
SPECIAL_INDEX = (1, 0, 1, 1, 2, 1)


def ordered_runs(case):
    """The launches of a rounding sum as ``(first, rows)``, else ``None``.

    scatter.inc issues a contended float sum that some order would round as
    maximal runs of consecutive index rows whose destinations are disjoint, so
    each destination receives its updates in index order. Every row of
    SPECIAL_INDEX updates a whole destination row, so two rows share a
    destination exactly when their indices are equal.
    """
    if case["layout"] != "rounding":
        return None
    runs, touched, first = [], set(), 0
    for row, destination in enumerate(SPECIAL_INDEX):
        if destination in touched:
            runs.append((first, row - first))
            touched.clear()
            first = row
        touched.add(destination)
    runs.append((first, len(SPECIAL_INDEX) - first))
    return runs


def special_arrays(xp, np, case):
    """Wrapping integer and rounding float sums, mostly into row 1.

    Their 3x5 destinations end inside a packed word, so the last word keeps
    padding lanes that no update may change.
    """
    dtype, layout = case["dtype"], case["layout"]
    index = host_array(xp, np, np.array(SPECIAL_INDEX, dtype=np.int64), "int64")
    if layout == "wrap":
        # Row 1's odd lanes overflow; a carry into a neighbouring lane shows.
        source = np.arange(15).reshape(3, 5) + 10
        bits = 8 * np.dtype(dtype).itemsize
        prod = case["operation"] == "prod"
        update = np.full((6, 5), (1 << bits // 2) + 3 if prod else 3 << (bits - 3))
        update[:, ::2] = 1 if prod else 0
        update = update.astype(np.int64).astype(dtype)
        return (
            host_array(xp, np, source, dtype),
            [index],
            host_array(xp, np, update, dtype),
        )
    # Terms of half and quarter units of 1.0 tie and round to even; row 1's
    # first two lanes reach different sums in reverse order.
    precision = 11 if dtype == "float16" else 8
    source = np.full((3, 5), 1.0, dtype=np.float32)
    update = np.array(
        [
            [0.5, -0.25, 0.5, 1.5, -0.25],
            [1.0, -1.0, 2.0, 0.5, 4.0],
            [0.5, -0.25, 1.0, 0.5, -0.25],
            [1.0, -0.5, 0.5, -1.0, -0.5],
            [3.0, 2.0, -1.0, -2.0, 0.5],
            [0.0, 0.0, -0.5, 0.25, 0.75],
        ],
        dtype=np.float32,
    ) * np.float32(2.0 ** (1 - precision))
    return host_array(xp, np, source, dtype), [index], host_array(xp, np, update, dtype)


def arrays(xp, np, case):
    layout = case["layout"]
    if layout in ("wrap", "rounding"):
        return special_arrays(xp, np, case)
    shape = (2, 2) if layout.startswith("work") else (3, 4)
    if layout == "capacity":
        shape = (65535,)
    raw_source = np.arange(np.prod(shape)).reshape(shape)
    if case["dtype"] in FLOATS:
        raw_source = (raw_source + 1) * np.where(raw_source % 2, -0.5, 0.5)
    else:
        raw_source = raw_source + 10
    source = host_array(xp, np, raw_source, case["dtype"])
    if layout == "capacity":
        return (
            source,
            [xp.array(np.array([65534, -1, 0], dtype=np.int64))],
            xp.array(np.array([123, 123, 321], dtype=case["dtype"])),
        )
    if layout == "payloads":
        payloads = (
            np.array(
                [
                    0,
                    0x80000000,
                    1,
                    0x80000001,
                    0x007FFFFF,
                    0x807FFFFF,
                    0x7F800000,
                    0xFF800000,
                    0x7FC12345,
                    0xFFC12345,
                    0x7FA00001,
                    0xFFA00001,
                ],
                dtype=np.uint32,
            )
            .view(np.float32)
            .reshape(shape)
        )
        return (
            xp.array(payloads[::-1].copy()),
            [xp.array(np.array([2, 0, 1], dtype=np.int64))],
            xp.array(payloads),
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
        if case["dtype"] in ("int32", "int8", "int16"):
            update = np.where(positions % 7 == 1, -update, update)
        update = np.where(positions == 0, 0, update)
        if case["dtype"] in FLOATS:
            update = np.where(
                positions < 8, np.take([-1.0, 0.5, -2.0, 1.0], positions % 4), 1.0
            )
    elif case["operation"] != "none":
        update = np.arange(update.size).reshape(update.shape) % 19 + 1
    if case["dtype"] in FLOATS and case["operation"] != "prod":
        update = -update / 4 if case["operation"] == "none" else (update - 10) / 4
    if layout == "update-strided":
        update = host_array(xp, np, np.repeat(update, 2, axis=-1), case["dtype"])
        update = update[..., ::2]
    elif layout == "update-broadcast":
        update = xp.broadcast_to(
            host_array(
                xp,
                np,
                np.array(
                    (0.5 if case["operation"] == "prod" else -1.75)
                    if case["dtype"] in FLOATS
                    else (2 if case["operation"] == "prod" else 7)
                ),
                case["dtype"],
            ),
            update.shape,
        )
    elif layout == "alias":
        update = source[:2]
        indices = [xp.array(np.array([1, 0], dtype=case["index_dtype"]))]
    else:
        update = host_array(xp, np, update, case["dtype"])
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
        if case["dtype"] in NARROW_TYPES:
            # Lane by lane, as MLX's packed atomic update applies them.
            current = expected[destination]
            update = np.broadcast_to(updates[coordinate], np.shape(current))
            for offset in np.ndindex(np.shape(current)):
                expected[destination + offset] = scatter_arithmetic.combine(
                    np, case["dtype"], operation, current[offset], update[offset]
                )
        elif operation == "none":
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
