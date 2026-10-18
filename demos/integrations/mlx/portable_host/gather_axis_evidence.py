"""Reconstruct axis-gather outputs from retained native uploads."""

import math

from demos.integrations.mlx.portable_host.gather_axis_layout import METADATA, signature
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_result,
    require,
    storage,
    view,
)
from demos.integrations.mlx.portable_host.gather_workloads import words


def audit_event(np, event):
    _, index_dtype, source_contiguous, index_contiguous = signature(event["entry"])
    request = event["details"]["request"]
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(event["entry"] + "_")
        member = "out" if member == "out_" else member
        require(member not in inputs, "Duplicate axis-gather binding")
        require(
            binding["dtype"] == value["dtype"] == layout["elementType"]
            and binding["shape"] == value["shape"]
            and binding.get("encoding") == value.get("encoding"),
            "Axis-gather binding layout changed",
        )
        inputs[member] = value
    require(
        set(inputs) == {"src", "indices", "out", *METADATA},
        "Axis-gather bindings are incomplete",
    )
    arrays = {name: storage(np, value) for name, value in inputs.items()}
    for name, dtype in METADATA.items():
        require(inputs[name]["dtype"] == dtype, "Axis-gather metadata storage changed")
    require(
        inputs["indices"]["dtype"] == index_dtype, "Axis-gather index storage changed"
    )
    scalars = ("ndim", "axis", "axis_size", "src_ax_stride", "idx_ax_stride")
    require(
        all(arrays[name].size == 1 for name in scalars),
        "Axis-gather scalar metadata changed",
    )
    ndim, axis, axis_size, source_step, index_step = (
        int(arrays[name][0]) for name in scalars
    )
    require(
        0 <= ndim < 64 and 0 <= axis <= ndim and axis_size > 0,
        "Axis-gather rank or axis changed",
    )
    require(
        all(
            arrays[name].size == max(1, ndim)
            for name in ("shape", "src_strides", "idx_strides")
        ),
        "Axis-gather metadata length changed",
    )
    shape = arrays["shape"][:ndim].tolist()
    require(all(size > 0 for size in shape), "Axis-gather outer shape changed")
    output_size = len(event["gatherValues"])
    require(
        output_size > 0 and output_size % math.prod(shape) == 0,
        "Axis-gather output shape changed",
    )
    index_size = output_size // math.prod(shape)
    source_shape, index_shape = shape.copy(), shape.copy()
    source_shape.insert(axis, axis_size)
    index_shape.insert(axis, index_size)
    source_strides, index_strides = (
        arrays["src_strides"][:ndim].tolist(),
        arrays["idx_strides"][:ndim].tolist(),
    )
    source_strides.insert(axis, source_step)
    index_strides.insert(axis, index_step)
    source = view(np, arrays["src"], source_shape, source_strides)
    indices = view(np, arrays["indices"], index_shape, index_strides)
    require(
        not source_contiguous or source.flags.c_contiguous,
        "Axis-gather source contiguity changed",
    )
    require(
        not index_contiguous or indices.flags.c_contiguous,
        "Axis-gather index contiguity changed",
    )
    require(
        all(-axis_size <= int(value) < axis_size for value in indices.flat),
        "Axis-gather index is outside its source",
    )
    expected = words(np, np.take_along_axis(source, indices, axis=axis))
    grid = [math.prod(shape[axis:]), index_size, math.prod(shape[:axis])]
    metadata = {
        "sourceShape": source_shape,
        "sourceStrides": source_strides,
        "indexShape": index_shape,
        "indexStrides": index_strides,
        "sourceContiguous": source_contiguous,
        "indexContiguous": index_contiguous,
        "axis": axis,
        "sourceCount": arrays["src"].size,
        "outputCount": output_size,
        "maximumIndex": (
            max(
                value.size if name != "out" else output_size
                for name, value in arrays.items()
            )
            - 1
        ),
    }
    return audit_result(np, event, inputs, source, [indices], expected, grid, metadata)
