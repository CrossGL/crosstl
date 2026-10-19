"""Reconstruct scatter results from retained uploads and native readbacks."""

import hashlib
import math

from demos.integrations.mlx.portable_host import scatter_axis_workloads as workloads
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_input_bindings,
    audit_native_execution,
    require,
    view,
)
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.portable_host.runtime import COPY_GUARD
from demos.integrations.mlx.portable_host.scatter_axis_layout import METADATA, signature


def audit_event(np, event):
    dtype, index_dtype, operation, update_contiguous, index_contiguous = signature(
        event["entry"]
    )
    require(
        event["target"] in {"metal", "directx", "opengl"}, "Unknown axis-scatter target"
    )
    request = event["details"]["request"]
    audit_input_bindings(event)
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(event["entry"] + "_")
        member = "out" if member == "out_" else member
        require(member not in inputs, "Duplicate axis-scatter binding")
        require(
            binding["dtype"] == value["dtype"] == layout["elementType"]
            and binding["shape"] == value["shape"]
            and binding.get("encoding") is None
            and "encoding" not in value,
            "Axis-scatter binding layout changed",
        )
        require(
            value["shape"]
            == (
                [len(value["values"]), 1] if member == "out" else [len(value["values"])]
            ),
            "Axis-scatter upload shape changed",
        )
        if member == "out":
            require(
                layout.get("componentCount") == 1
                and layout.get("elementStrideBytes") == 4
                and layout.get("structMembers")
                == [
                    {
                        "name": "val",
                        "offsetBytes": 0,
                        "physicalType": "int" if dtype == "int32" else "uint",
                    }
                ],
                "Axis-scatter atomic layout changed",
            )
        inputs[member] = value
    dtypes = {**METADATA, "upd": dtype, "indices": index_dtype, "out": dtype}
    require(set(inputs) == set(dtypes), "Axis-scatter bindings are incomplete")
    for name, kind in dtypes.items():
        require(inputs[name]["dtype"] == kind, "Axis-scatter storage type changed")
    arrays = {
        name: np.array(value["values"], dtype=value["dtype"])
        for name, value in inputs.items()
    }
    scalars = ("ndim", "axis", "out_axis_size", "upd_ax_stride", "idx_ax_stride")
    require(
        all(arrays[name].size == 1 for name in scalars),
        "Axis-scatter scalar length changed",
    )
    ndim, axis, axis_size, update_step, index_step = (
        int(arrays[name][0]) for name in scalars
    )
    require(
        0 <= ndim < 64 and 0 <= axis <= ndim and axis_size > 0,
        "Axis-scatter axis or rank changed",
    )
    require(
        all(
            arrays[name].size == max(1, ndim)
            for name in ("shape", "upd_strides", "idx_strides")
        ),
        "Axis-scatter metadata lengths changed",
    )
    shape = arrays["shape"][:ndim].tolist()
    grid = event["workgroupCount"]
    require(
        len(grid) == 3
        and all(type(size) is int and size > 0 for size in grid)
        and grid[0] == math.prod(shape[axis:])
        and grid[2] == math.prod(shape[:axis]),
        "Axis-scatter launch changed",
    )
    index_shape, output_shape = shape.copy(), shape.copy()
    index_shape.insert(axis, grid[1])
    output_shape.insert(axis, axis_size)
    update_strides, index_strides = (
        arrays[name][:ndim].tolist() for name in ("upd_strides", "idx_strides")
    )
    update_strides.insert(axis, update_step)
    index_strides.insert(axis, index_step)
    updates = view(np, arrays["upd"], index_shape, update_strides)
    indices = view(np, arrays["indices"], index_shape, index_strides)
    require(
        not update_contiguous or updates.flags.c_contiguous,
        "Axis-scatter update contiguity changed",
    )
    require(
        not index_contiguous or indices.flags.c_contiguous,
        "Axis-scatter index contiguity changed",
    )
    size = math.prod(output_shape)
    require(
        arrays["out"].size == size + len(COPY_GUARD),
        "Axis-scatter output extent changed",
    )
    initial = arrays["out"][:size].reshape(output_shape).copy()
    expected = initial.copy()
    for coordinate in np.ndindex(indices.shape):
        index = int(indices[coordinate])
        require(
            -axis_size <= index < axis_size,
            "Axis-scatter index exceeds its destination",
        )
        destination = list(coordinate)
        destination[axis] = index % axis_size
        destination = tuple(destination)
        if operation == "none":
            expected[destination] = updates[coordinate]
        else:
            expected[destination] += updates[coordinate]
    readback = event["scatterValues"]
    limits = np.iinfo(dtype)
    require(
        isinstance(readback, list)
        and len(readback) == size
        and all(
            type(value) is int and limits.min <= value <= limits.max
            for value in readback
        ),
        "Axis-scatter readback is outside its storage type",
    )
    raw = np.array(readback, dtype=dtype)
    actual = words(np, raw)
    require(
        actual == words(np, expected),
        "Axis-scatter native readback disagrees with uploads",
    )
    require(
        event["scatterGuardValues"] == arrays["out"][size:].tolist() == COPY_GUARD,
        "Axis-scatter output guards changed",
    )
    metadata = {
        "operation": operation,
        "outputShape": output_shape,
        "updateShape": index_shape,
        "updateStrides": update_strides,
        "indexShape": index_shape,
        "indexStrides": index_strides,
        "updateContiguous": update_contiguous,
        "indexContiguous": index_contiguous,
        "axis": axis,
        "updateCount": math.prod(grid),
        "outputCount": size,
        "maximumIndex": size - 1,
    }
    target = event["target"]
    require(
        event["scatterMetadata"] == metadata
        and event["scatterStorageType"] == dtype
        and event["threads"] == size
        and event["dispatchVersion"] == 3
        and event["workgroupSize"] == [1, 1, 1],
        "Axis-scatter execution metadata changed",
    )
    entry = {"directx": "CSMain", "opengl": "main", "metal": event["entry"]}[target]
    dispatch = request["dispatch"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and dispatch.get("entryPoint") == entry
        and dispatch.get("workgroupCount") == grid
        and dispatch.get("workgroupSize") == [1, 1, 1]
        and dispatch.get("globalSize") == dispatch.get("gridSize") == grid
        and "threadGridSize" not in dispatch,
        "Axis-scatter native request changed",
    )
    require(
        hashlib.sha256(raw.tobytes()).hexdigest() == event["outputHash"],
        "Axis-scatter host output hash changed",
    )
    audit_native_execution(event)
    return initial, indices, updates, actual


def validate(np, records, trace):
    from demos.integrations.mlx.portable_host.verify_bitwise import (
        verify_native_identity,
    )

    cases = list(workloads.cases())
    require(len(records) == len(cases), "Axis-scatter workload set is incomplete")
    cursor, count, targets = 0, 0, set()
    for case, record in zip(cases, records):
        require(
            all(record.get(key) == value for key, value in case.items())
            and record.get("inputUnchanged") is True
            and record.get("resultDtype") == "mlx.core." + case["dtype"],
            "Axis-scatter workload identity changed",
        )
        require(
            type(record["dispatchStart"]) is int
            and record["dispatchStart"] == cursor
            and type(record["dispatchCount"]) is int
            and record["dispatchCount"] > 0,
            "Axis-scatter dispatch boundary changed",
        )
        end = cursor + record["dispatchCount"]
        require(end <= len(trace), "Axis-scatter trace ends before its workload")
        events = [
            item
            for item in trace[cursor:end]
            if item["entry"].startswith("scatter_axis")
        ]
        require(
            len(events) == (0 if case["layout"] == "empty" else 1),
            "Axis-scatter dispatch count changed",
        )
        source, expected_indices, expected_updates, expected = workloads.reference(
            np, case
        )
        require(
            record["actual"] == words(np, expected)
            and record["shape"] == list(expected.shape),
            "Axis-scatter workload result changed",
        )
        for event in events:
            initial, indices, updates, actual = audit_event(np, event)
            initial_expected = (
                source if case["operation"] == "none" else np.zeros_like(source)
            )
            require(
                initial.shape == initial_expected.shape
                and words(np, initial) == words(np, initial_expected),
                "Axis-scatter initialization does not belong to its workload",
            )
            for name, actual_array, expected_array in (
                ("index", indices, expected_indices),
                ("update", updates, expected_updates),
            ):
                require(
                    actual_array.shape == expected_array.shape
                    and words(np, actual_array) == words(np, expected_array),
                    "Axis-scatter uploads do not belong to their workload",
                )
                physical = (
                    np.ascontiguousarray(expected_array)
                    if any(step < 0 for step in expected_array.strides)
                    else expected_array
                )
                require(
                    event["scatterMetadata"][name + "Strides"]
                    == [step // physical.itemsize for step in physical.strides],
                    "Axis-scatter workload strides changed",
                )
            require(
                actual == record["actual"] == words(np, expected),
                "Axis-scatter readback and MLX result disagree",
            )
            count += 1
        targets.update(event["target"] for event in trace[cursor:end])
        cursor = end
    require(
        cursor == len(trace) and len(targets) == 1,
        "Axis-scatter trace is incomplete or mixes targets",
    )
    target = targets.pop()
    verify_native_identity(
        [event for event in trace if not event["entry"].startswith("scatter_axis")],
        target,
    )
    return {"target": target, "scatterDispatchCount": count}
