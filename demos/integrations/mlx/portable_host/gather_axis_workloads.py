"""Axis-gather workloads with exact storage and noncontiguous index views."""

import math

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

# Workloads whose launches split past 65,535 workgroups per axis: the source
# shape, the gathered axis, its index rows and whether the source and index
# views are strided. Trailing extents past one launch axis run through the
# strided specializations, per leading position and index row.
WIDE = {
    "wide-trailing": ((2, 70001), 0, 3, False),
    "wide-trailing-slab": ((2, 257, 257), 0, 2, True),
    "wide-rows": ((5,), 0, 70001, True),
    "wide-leading": ((131073, 2), 1, 1, False),
    "wide-slab": ((300, 300, 2), 2, 3, True),
    "wide-split": ((3, 2), 1, 70000, False),
}


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
    for i, layout in enumerate(WIDE):
        for dtype in ("float32", "int64"):
            yield {
                "id": f"{dtype}-{layout}",
                "dtype": dtype,
                "layout": layout,
                "axis": WIDE[layout][1],
                "index_dtype": ("int32", "uint32", "int64", "uint64")[i % 4],
            }


def wide_arrays(xp, np, case):
    shape, axis, rows, strided = WIDE[case["layout"]]
    values = np.arange(math.prod(shape)).reshape(shape) % 97 - 48
    if case["dtype"] == "float32":
        values = values / 4
    values = values.astype(case["dtype"])
    index_shape = list(shape)
    index_shape[axis] = rows
    raw = np.arange(math.prod(index_shape)).reshape(index_shape)
    indices = raw % shape[axis]
    if case["index_dtype"].startswith("int"):
        indices = np.where(raw % 2, indices - shape[axis], indices)
    indices = indices.astype(case["index_dtype"])
    if not strided:
        return xp.array(values), xp.array(indices)
    # Stride the last axis that has more than one element, so the views'
    # strides are observable rather than those of a unit dimension.
    views = []
    for array in (values, indices):
        step = max(i for i, size in enumerate(array.shape) if size > 1)
        view = (slice(None),) * step + (slice(None, None, 2),)
        views.append(xp.array(np.repeat(array, 2, axis=step))[view])
    return tuple(views)


def arrays(xp, np, case):
    if case["layout"] in WIDE:
        return wide_arrays(xp, np, case)
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
