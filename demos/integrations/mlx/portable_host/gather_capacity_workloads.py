"""Indexed views larger than one native dispatch axis, with exact storage oracles."""

import math

from demos.integrations.mlx.portable_host import gather_workloads

# Slices wider than one launch axis: a scalar index (the shape of
# test_blas.test_gather_mm_blocks' reference), a 2-D index block, and a
# gathered middle axis of a strided view. Each maps to its source root shape,
# its index root and the gathered axis.
SLICES = {
    "large-slice": ((10, 129, 512), (7, -5, 1), 0),
    "large-slice-rows": ((6, 257, 256), (9, -1, 9, 2, 9, -4, 9, 0, 9), 0),
    "large-slice-axis": ((259, 5, 512), (9, 3, 9, -4, 9), 1),
}


def cases():
    yield from gather_workloads.cases()
    for dtype in gather_workloads.DTYPES:
        yield {"id": f"{dtype}-large-vector", "dtype": dtype, "layout": "large-vector"}
    for layout in ("large-column", "large-row", "large-tail", "large-multiple"):
        yield {"id": f"float32-{layout}", "dtype": "float32", "layout": layout}
    for count in (65534, 65535, 131071):
        yield {
            "id": f"float32-table-{count}",
            "dtype": "float32",
            "layout": "large-table",
            "count": count,
        }
    for layout in SLICES:
        yield {"id": f"float32-{layout}", "dtype": "float32", "layout": layout}


def slice_arrays(np, case):
    shape, index, _ = SLICES[case["layout"]]
    source = np.resize(gather_workloads.data(np, case["dtype"]), shape)
    return source, np.array(index, dtype=np.int32)


def slice_views(module, source, index, layout):
    if layout == "large-slice-axis":
        return source[1:-1, :, ::2], index[1:-1:2]
    view = index[1:2] if layout == "large-slice" else index[1:-1:2]
    shape = () if layout == "large-slice" else (2, 2)
    return source[1:-1], module.reshape(view, shape)


def arrays(np, case):
    multiple = case["layout"] == "large-multiple"
    source_shape = (259, 263, 2) if multiple else (65538, 2)
    if case["layout"] == "large-table":
        source_shape = (18, 2)
    source = np.resize(gather_workloads.data(np, case["dtype"]), source_shape)
    shape = {
        "large-vector": (65536,),
        "large-column": (65536, 2),
        "large-row": (2, 65536),
        "large-tail": (2, 257, 256),
        "large-multiple": (2, 257, 256),
        "large-table": (case.get("count", 1),),
    }[case["layout"]]
    if multiple:
        first = (np.arange(2 * 257, dtype=np.int32) % 257 - 257).reshape(2, 257, 1)
        second = (np.arange(512, dtype=np.int32) % 263 - 263).reshape(1, 1, 512)
        return source, (first, second), shape
    count = math.prod(shape)
    extent = source_shape[0] - 2
    indices = (np.arange(2 * count + 2, dtype=np.int32) * 37 % extent) - extent
    return source, (indices,), shape


def views(module, source, indices, shape, multiple):
    source = source[1:-1]
    if multiple:
        first, second = indices
        indices = (
            module.broadcast_to(first, shape),
            module.broadcast_to(second[:, :, ::2], shape),
        )
    else:
        indices = (module.reshape(indices[0][1:-1:2], shape),)
    return source, indices


def reference(np, case):
    if not case["layout"].startswith("large-"):
        return gather_workloads.reference(np, case)
    if case["layout"] in SLICES:
        source, index = slice_views(np, *slice_arrays(np, case), case["layout"])
        axis = SLICES[case["layout"]][2]
        return source, (index,), np.take(source, index, axis=axis)
    source, indices, shape = arrays(np, case)
    source, indices = views(
        np, source, indices, shape, case["layout"] == "large-multiple"
    )
    result = (
        source[indices] if len(indices) > 1 else np.take(source, indices[0], axis=0)
    )
    return source, indices, result


def expression(mx, np, case):
    if not case["layout"].startswith("large-"):
        return gather_workloads.expression(mx, np, case)
    if case["layout"] in SLICES:
        source_data, index = slice_arrays(np, case)
        index_data = (index,)
    else:
        source_data, index_data, shape = arrays(np, case)
    source_root = mx.array(source_data)
    index_roots = tuple(mx.array(index) for index in index_data)
    if case["layout"] in SLICES:
        source, index = slice_views(mx, source_root, index_roots[0], case["layout"])
        indices = (index,)
        axis = SLICES[case["layout"]][2]
    else:
        source, indices = views(
            mx, source_root, index_roots, shape, case["layout"] == "large-multiple"
        )
        axis = 0
    result = (
        source[indices] if len(indices) > 1 else mx.take(source, indices[0], axis=axis)
    )
    mx.eval(result)
    for root, original in zip((source_root, *index_roots), (source_data, *index_data)):
        if gather_workloads.words(np, np.array(root)) != gather_workloads.words(
            np, original
        ):
            raise ValueError(
                "Gather changed an input allocation, including its unused storage"
            )
    return source, result
