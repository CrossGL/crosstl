"""Axis-gather workloads with exact storage and noncontiguous index views."""

from demos.integrations.mlx.portable_host.gather_workloads import DTYPES, data

LAYOUTS = (
    "dense",
    "transposed",
    "strided",
    "broadcast",
    "reversed",
    "index-strided",
    "both-strided",
    "flattened",
    "index-broadcast",
    "index-reversed",
)


def cases():
    for dtype in DTYPES:
        for i, layout in enumerate(LAYOUTS):
            yield {
                "id": f"{dtype}-{layout}",
                "dtype": dtype,
                "layout": layout,
                "axis": None if layout == "flattened" else (0, 1, -1)[i % 3],
                "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
            }


def arrays(xp, np, case):
    source = xp.array(data(np, case["dtype"]))
    layout = case["layout"]
    if layout == "transposed":
        source = xp.transpose(source, (1, 0, 2))
    elif layout in {"strided", "both-strided"}:
        source = source[::2]
    elif layout == "broadcast":
        source = xp.broadcast_to(source[:1], source.shape)
    elif layout == "reversed":
        source = source[::-1]
    elif layout == "flattened":
        source = xp.reshape(source, (-1,))
    axis = 0 if case["axis"] is None else case["axis"] % source.ndim
    shape = list(source.shape)
    shape[axis] = 3
    index_strided = layout in {"index-strided", "both-strided"}
    if index_strided:
        shape[-1] *= 2
    elif layout == "index-broadcast":
        shape[0] = 1
    count = int(np.prod(shape))
    indices = np.arange(count).reshape(shape) % source.shape[axis]
    if case["index_dtype"].startswith("int"):
        indices = np.where(
            np.arange(count).reshape(shape) % 2, indices - source.shape[axis], indices
        )
    indices = xp.array(indices.astype(case["index_dtype"]))
    if index_strided:
        indices = indices[..., ::2]
    elif layout == "index-broadcast":
        target = list(source.shape)
        target[axis] = 3
        indices = xp.broadcast_to(indices, tuple(target))
    elif layout == "index-reversed":
        indices = indices[::-1]
    return source, indices


def reference(np, case):
    source, indices = arrays(np, np, case)
    return source, (indices,), np.take_along_axis(source, indices, axis=case["axis"])


def expression(mx, np, case):
    source, indices = arrays(mx, np, case)
    return source, mx.take_along_axis(source, indices, axis=case["axis"])
