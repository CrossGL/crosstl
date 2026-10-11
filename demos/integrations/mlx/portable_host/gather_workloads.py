"""Exact-storage indexing workloads for the portable MLX host backend."""

DTYPES = ("float32", "int32", "uint32", "int64", "uint64", "bool_")
LAYOUTS = (
    "dense",
    "transposed",
    "strided",
    "broadcast",
    "reversed",
    "scalar",
    "multiple",
)


def cases():
    for dtype in DTYPES:
        for layout in LAYOUTS:
            yield {"id": f"{dtype}-{layout}", "dtype": dtype, "layout": layout}


def data(np, dtype):
    values = np.arange(24).reshape(4, 3, 2)
    if dtype == "float32":
        words = (
            0x7FC01234,
            0x80000000,
            0,
            1,
            0x7F800000,
            0xFF800000,
            0x3F800000,
            0xBF800000,
        )
        return (
            np.resize(np.array(words, dtype=np.uint32), 24)
            .reshape(4, 3, 2)
            .view(np.float32)
        )
    if dtype == "int64":
        return (values.astype(np.int64) + 2**54) * np.where(values % 2, -1, 1)
    if dtype == "uint64":
        return values.astype(np.uint64) + np.uint64(2**63)
    if dtype == "bool_":
        return values % 3 == 1
    return (values - 12 if dtype == "int32" else values).astype(dtype)


def reference(np, case):
    source = data(np, case["dtype"])
    layout = case["layout"]
    if layout == "transposed":
        source = source.transpose(1, 0, 2)
    elif layout == "strided":
        source = source[::2]
    elif layout == "broadcast":
        source = np.broadcast_to(source[0:1], source.shape)
    elif layout == "reversed":
        source = source[::-1]
    if layout == "multiple":
        first = np.array([[0, -1, 1], [2, 0, -2]], dtype=np.int32)
        second = np.array([[0, 1, -1], [-2, 0, 1]], dtype=np.int32)
        return source, (first, second), source[first, second]
    index_dtype = {
        "transposed": "uint32",
        "strided": "int64",
        "broadcast": "uint64",
    }.get(layout, "int32")
    indices = np.array(
        1 if layout == "scalar" else [source.shape[0] - 1, 0, 1], dtype=index_dtype
    )
    return source, (indices,), np.take(source, indices, axis=0)


def expression(mx, np, case):
    source = mx.array(data(np, case["dtype"]))
    layout = case["layout"]
    if layout == "transposed":
        source = mx.transpose(source, (1, 0, 2))
    elif layout == "strided":
        source = source[::2]
    elif layout == "broadcast":
        source = mx.broadcast_to(source[0:1], source.shape)
    elif layout == "reversed":
        source = source[::-1]
    _, indices, _ = reference(np, case)
    arrays = [mx.array(index) for index in indices]
    result = (
        source[tuple(arrays)]
        if layout == "multiple"
        else mx.take(source, arrays[0], axis=0)
    )
    return source, result


def words(np, array):
    array = np.ascontiguousarray(array)
    return (
        array.view({1: np.uint8, 4: np.uint32, 8: np.uint64}[array.dtype.itemsize])
        .reshape(-1)
        .tolist()
    )
