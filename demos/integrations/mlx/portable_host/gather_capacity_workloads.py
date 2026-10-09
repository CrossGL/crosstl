"""Indexed views larger than one native dispatch axis, with exact storage oracles."""

import math

from demos.integrations.mlx.portable_host import gather_workloads


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
    source_data, index_data, shape = arrays(np, case)
    source_root = mx.array(source_data)
    index_roots = tuple(mx.array(index) for index in index_data)
    source, indices = views(
        mx, source_root, index_roots, shape, case["layout"] == "large-multiple"
    )
    result = (
        source[indices] if len(indices) > 1 else mx.take(source, indices[0], axis=0)
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
