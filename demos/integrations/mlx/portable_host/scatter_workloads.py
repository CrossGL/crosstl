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
# Binary16 and bfloat16 outputs pack two elements into each atomic word; an odd
# output leaves the high lane of its last word unused.
HALF_TYPES = ("float16", "bfloat16")
FLOAT_TYPES = ("float32", *HALF_TYPES)
HALF_LAYOUTS = (*LAYOUTS, "odd", "work4", "work8", "work16", "work32")
# Signed zeros, the extreme finite values, infinities and quiet NaN payloads,
# with binary16 subnormals. GLSL carries both formats in float registers, where
# signaling NaNs need not be supported and float32 subnormals (which every
# bfloat16 subnormal is) may be flushed to zero (GLSL 4.60, section 4.7.1).
HALF_PAYLOADS = {
    "float16": (
        *(0x0000, 0x8000, 0x0001, 0x83FF, 0x0400, 0x7BFF),
        *(0xFBFF, 0x7C00, 0xFC00, 0x7E01, 0xFE55, 0x3C00),
    ),
    "bfloat16": (
        *(0x0000, 0x8000, 0x0080, 0x8080, 0x3F80, 0x7F7F),
        *(0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFD5, 0xBF81),
    ),
}


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


def half_cases():
    for dtype in HALF_TYPES:
        for operation in ("none", "sum", "prod", "min", "max"):
            for i, layout in enumerate(HALF_LAYOUTS):
                yield {
                    "id": f"{dtype}-{operation}-{layout}",
                    "dtype": dtype,
                    "operation": operation,
                    "layout": layout,
                    "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
                }
        yield {
            "id": f"{dtype}-none-payloads",
            "dtype": dtype,
            "operation": "none",
            "layout": "payloads",
            "index_dtype": "int64",
        }
        yield {
            "id": f"{dtype}-sum-rounding",
            "dtype": dtype,
            "operation": "sum",
            "layout": "rounding",
            "index_dtype": "uint32",
        }


def profile_cases(*, float32=False, half=False):
    if float32 and half:
        raise ValueError("Choose one scatter workload profile")
    return half_cases() if half else float_cases() if float32 else cases()


def half_words(np, values, dtype):
    """Return the raw words of values that ``dtype`` represents exactly."""
    exact = np.asarray(values, dtype=np.float64)
    if dtype == "float16":
        rounded = exact.astype(np.float16)
        lanes = rounded.view(np.uint16)
    else:
        rounded = exact.astype(np.float32)
        bits = rounded.view(np.uint32)
        if (bits & 0xFFFF).any():
            raise ValueError("Bfloat scatter workload values must be exact")
        lanes = (bits >> 16).astype(np.uint16)
    if not np.array_equal(rounded.astype(np.float64), exact):
        raise ValueError("16-bit scatter workload values must be exact")
    return lanes


def half_values(np, lanes, dtype):
    """Widen raw 16-bit words to the float32 values they hold exactly."""
    lanes = np.asarray(lanes, dtype=np.uint16)
    if dtype == "float16":
        return lanes.view(np.float16).astype(np.float32)
    return (lanes.astype(np.uint32) << 16).view(np.float32)


def round_half(np, values, dtype):
    """Round float32 values to the nearest 16-bit words, ties to even."""
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return values.astype(np.float16).view(np.uint16)
    bits = values.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return np.where(np.isnan(values), (bits >> 16) | 0x40, rounded).astype(np.uint16)


def combine(np, dtype, operation, current, update):
    """Apply the pinned atomic.h update ``op(update, current)`` to raw words.

    The compare-and-swap loop stores only while ``Op::condition`` holds: a
    product never changes a zero, and min and max store only a strict change.
    """
    current = np.asarray(current, dtype=np.uint16)
    update = np.asarray(update, dtype=np.uint16)
    if operation == "none":
        return update.copy()
    a, b = half_values(np, update, dtype), half_values(np, current, dtype)
    with np.errstate(all="ignore"):
        result = {"sum": a + b, "prod": a * b, "min": a, "max": a}[operation]
        store = {
            "sum": np.ones(a.shape, dtype=bool),
            "prod": b != 0,
            "min": a < b,
            "max": a > b,
        }[operation]
    return np.where(store, round_half(np, result, dtype), current).astype(np.uint16)


def typed(xp, np, values, dtype):
    """Build ``dtype`` storage; 16-bit floats are built from their raw words."""
    values = np.asarray(values)
    if dtype not in HALF_TYPES:
        return xp.array(values.astype(dtype))
    lanes = half_words(np, values, dtype)
    return lanes if xp is np else xp.array(lanes).view(getattr(xp, dtype))


def host_array(mx, np, array):
    """Return NumPy storage of an MLX array; 16-bit floats as their raw words."""
    if array.dtype in (mx.float16, mx.bfloat16):
        array = array.view(mx.uint16)
    return np.array(array)


def half_layout(xp, np, case, shape):
    """The payload and rounding workloads, which index rows exactly once."""
    dtype = case["dtype"]

    def build(lanes):
        return lanes.copy() if xp is np else xp.array(lanes).view(getattr(xp, dtype))

    if case["layout"] == "payloads":
        lanes = np.array(HALF_PAYLOADS[dtype], dtype=np.uint16).reshape(shape)
        return (
            build(lanes[::-1].copy()),
            [xp.array(np.array([2, 0, 1], dtype=case["index_dtype"]))],
            build(lanes),
        )
    # Each sum needs rounding: below, at and above a half-unit tie, with even
    # and odd neighbours, at and below a binade boundary and on negative values.
    precision = 11 if dtype == "float16" else 8
    unit = 2.0 ** (1 - precision)
    pairs = (
        *((1, unit / 2), (1 + unit, unit / 2), (1, 3 * unit / 4), (1, unit / 4)),
        *((-1, -unit / 2), (-(1 + unit), -unit / 2), (2**precision, 1)),
        *((2**precision + 2, 1), (0.5, unit / 4), (3, unit), (1, -unit / 4)),
        (1, -3 * unit / 4),
    )
    initial, update = (np.array(column).reshape(shape) for column in zip(*pairs))
    return (
        build(half_words(np, initial, dtype)),
        [xp.array(np.array([0, 1, 2], dtype=case["index_dtype"]))],
        build(half_words(np, update, dtype)),
    )


def arrays(xp, np, case):
    layout = case["layout"]
    shape = (
        (2, 2) if layout.startswith("work") else (3, 3) if layout == "odd" else (3, 4)
    )
    if layout == "capacity":
        shape = (65535,)
    if case["dtype"] in HALF_TYPES and layout in ("payloads", "rounding"):
        return half_layout(xp, np, case, shape)
    raw_source = np.arange(np.prod(shape)).reshape(shape)
    if case["dtype"] in FLOAT_TYPES:
        raw_source = (raw_source + 1) * np.where(raw_source % 2, -0.5, 0.5)
    else:
        raw_source = raw_source + 10
    source = typed(xp, np, raw_source, case["dtype"])
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
        if case["dtype"] == "int32":
            update = np.where(positions % 7 == 1, -update, update)
        update = np.where(positions == 0, 0, update)
        if case["dtype"] in FLOAT_TYPES:
            update = np.where(
                positions < 8, np.take([-1.0, 0.5, -2.0, 1.0], positions % 4), 1.0
            )
    elif case["operation"] == "sum" and case["dtype"] in HALF_TYPES:
        # Every partial sum, in any order, keeps within bfloat16's 8 bits.
        update = (np.arange(update.size).reshape(update.shape) % 5 - 2) / 2
    elif case["operation"] != "none":
        update = np.arange(update.size).reshape(update.shape) % 19 + 1
    if (
        case["dtype"] in FLOAT_TYPES
        and case["operation"] != "prod"
        and not (case["operation"] == "sum" and case["dtype"] in HALF_TYPES)
    ):
        update = -update / 4 if case["operation"] == "none" else (update - 10) / 4
    if case["dtype"] not in HALF_TYPES:
        update = update.astype(case["dtype"])
    if layout == "update-strided":
        update = typed(xp, np, np.repeat(update, 2, axis=-1), case["dtype"])[..., ::2]
    elif layout == "update-broadcast":
        update = xp.broadcast_to(
            typed(
                xp,
                np,
                np.array(
                    (0.5 if case["operation"] == "prod" else -1.75)
                    if case["dtype"] in FLOAT_TYPES
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
        update = typed(xp, np, update, case["dtype"])
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
        if case["dtype"] in HALF_TYPES:
            expected[destination] = combine(
                np, case["dtype"], operation, expected[destination], updates[coordinate]
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
