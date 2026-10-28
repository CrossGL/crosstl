"""Reconstruct general-scatter results from retained uploads and native readbacks."""

import hashlib
import math

from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host import (
    copy_evidence,
    gather_storage,
    scatter_arithmetic,
    scatter_storage,
)
from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_input_bindings,
    audit_native_execution,
    require,
    view,
)
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.portable_host.runtime import COPY_GUARD, LAZY_COPY_ENTRIES
from demos.integrations.mlx.portable_host.scatter_layout import (
    ATOMIC_TYPES,
    METADATA,
    atomic_storage_dtype,
    signature,
)


def word_binding_matches(np, binding, layout, value):
    """A value carried as its own words matches its binding exactly."""
    return (
        binding["dtype"] == value["dtype"] == layout["elementType"]
        and binding["shape"] == value["shape"]
        and binding.get("encoding")
        == (FLOAT32_BITS if value["dtype"] == "float32" else None)
        and (
            value.get("encoding") == FLOAT32_BITS
            if value["dtype"] == "float32"
            else "encoding" not in value
        )
        and layout["elementStrideBytes"]
        == (1 if value["dtype"] == "bool" else np.dtype(value["dtype"]).itemsize)
    )


def audit_binding(np, dtype, target, member, binding, layout, value):
    """Require one upload to match its reflected binding and storage type."""
    packed = member == "out" and scatter_storage.packed(dtype)
    if member == "updates" and gather_storage.extended(dtype):
        matches = (
            binding["dtype"] == value["dtype"]
            and binding["shape"] == value["shape"] == [len(value["values"])]
            and binding.get("encoding") == value.get("encoding")
            and gather_storage.reflected_layout_matches(layout, value, dtype)
        )
    elif packed:
        matches = (
            binding["dtype"] == value["dtype"] == "uint32"
            and binding["shape"] == value["shape"]
            and binding.get("encoding") is None
            and "encoding" not in value
            and scatter_storage.layout_matches(layout)
        )
    else:
        matches = word_binding_matches(np, binding, layout, value)
    require(matches, "Scatter binding layout changed")
    require(
        value["shape"]
        == ([len(value["values"]), 1] if member == "out" else [len(value["values"])]),
        "Scatter upload shape changed",
    )
    if member == "out" and not packed:
        require(
            layout.get("componentCount") == 1
            and layout.get("structMembers")
            == [
                {
                    "name": "val",
                    "offsetBytes": 0,
                    "physicalType": ATOMIC_TYPES[atomic_storage_dtype(dtype, target)],
                }
            ],
            "Scatter atomic storage layout changed",
        )


def upload_arrays(np, event, dtype, index_dtype, count):
    """Audit and decode every upload; values become their logical storage."""
    target = event["target"]
    request = event["details"]["request"]
    audit_input_bindings(event)
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(event["entry"] + "_")
        member = "out" if member == "out_" else member
        require(member not in inputs, "Duplicate scatter binding")
        audit_binding(np, dtype, target, member, binding, layout, value)
        inputs[member] = value
    dtypes = {
        **METADATA,
        "updates": dtype,
        "out": atomic_storage_dtype(dtype, target),
        **{f"idx{i}": index_dtype for i in range(count)},
    }
    dtypes["idx_contigs"] = "bool" if target == "metal" else "uint32"
    require(set(inputs) == set(dtypes), "Scatter bindings are incomplete")
    extended = gather_storage.extended(dtype)
    for name, kind in dtypes.items():
        if name != "updates" or not extended:
            require(inputs[name]["dtype"] == kind, "Scatter storage type changed")
    require(
        all(value in (0, 1) for value in inputs["idx_contigs"]["values"]),
        "Scatter Boolean upload changed",
    )
    arrays = {}
    for name, value in inputs.items():
        if extended and name == "updates":
            arrays[name] = scatter_arithmetic.values(
                np, dtype, gather_storage.decoded(np, dtype, target, value)
            )
        elif dtype == "float32" and name in {"out", "updates"}:
            require(
                isinstance(value["values"], list)
                and all(
                    type(word) is int and 0 <= word <= 0xFFFFFFFF
                    for word in value["values"]
                ),
                "Scatter float upload is not raw binary32 storage",
            )
            arrays[name] = np.array(value["values"], dtype=np.uint32).view(np.float32)
        elif name != "out" or not scatter_storage.packed(dtype):
            arrays[name] = np.array(value["values"], dtype=value["dtype"])
    return inputs, arrays


def audit_word_readback(np, event, dtype, size, expected, arrays):
    """Native words of a 32-bit destination must equal the recomputed result."""
    readback = event["scatterValues"]
    limits = np.iinfo("uint32" if dtype == "float32" else dtype)
    require(
        isinstance(readback, list)
        and len(readback) == size
        and all(
            type(value) is int and limits.min <= value <= limits.max
            for value in readback
        ),
        "Scatter readback is outside its storage type",
    )
    raw = np.array(readback, dtype="uint32" if dtype == "float32" else dtype)
    actual = words(np, raw)
    require(
        actual == words(np, expected), "Scatter native readback disagrees with uploads"
    )
    require(
        event["scatterGuardValues"] == words(np, arrays["out"][size:]) == COPY_GUARD,
        "Scatter output guards changed",
    )
    require(
        hashlib.sha256(raw.tobytes()).hexdigest() == event["outputHash"],
        "Scatter output hash changed",
    )
    return actual


def audit_packed_readback(np, event, dtype, size, expected, trailing):
    """Lanes of packed atomic words must equal the recomputed logical words."""
    readback = event["scatterValues"]
    ctype = scatter_storage.CTYPES[dtype]
    require(
        isinstance(readback, list)
        and len(readback) == size
        and all(
            type(value) is int and ctype(value).value == value for value in readback
        ),
        "Scatter readback is outside its storage type",
    )
    require(
        readback == scatter_arithmetic.storage_words(np, dtype, expected),
        "Scatter native readback disagrees with uploads",
    )
    require(event["scatterGuardValues"] == trailing, "Scatter output guards changed")
    native = scatter_storage.native_words(dtype, readback)
    require(
        hashlib.sha256(bytes(native)).hexdigest() == event["outputHash"],
        "Scatter output hash changed",
    )
    mask = (1 << 8 * scatter_storage.element_size(dtype)) - 1
    return [word & mask for word in readback]


def audit_event(np, event):
    dtype, index_dtype, count, operation, contiguous, work = signature(event["entry"])
    target = event["target"]
    require(target in {"metal", "directx", "opengl"}, "Unknown scatter target")
    request = event["details"]["request"]
    inputs, arrays = upload_arrays(np, event, dtype, index_dtype, count)
    packed = scatter_storage.packed(dtype)
    scalar_names = ("upd_ndim", "upd_size", "out_ndim", "idx_ndim", "idx_size")
    require(
        all(arrays[name].size == 1 for name in scalar_names),
        "Scatter scalar lengths changed",
    )
    update_rank, slice_size, rank, index_rank, index_size = (
        int(arrays[name][0]) for name in scalar_names
    )
    require(
        1 <= rank <= 64
        and 0 <= index_rank <= 64
        and update_rank == rank + index_rank <= 64,
        "Scatter ranks changed",
    )
    (
        shape,
        steps,
        update_shape,
        update_steps,
        axes,
        index_shapes,
        index_steps,
        contigs,
    ) = (
        arrays[name].tolist()
        for name in (
            "out_shape",
            "out_strides",
            "upd_shape",
            "upd_strides",
            "axes",
            "idx_shapes",
            "idx_strides",
            "idx_contigs",
        )
    )
    require(
        len(shape) == rank
        and len(update_shape) == update_rank
        and len(axes) == len(set(axes)) == count
        and all(0 <= axis < rank for axis in axes),
        "Scatter shape or axes changed",
    )
    require(
        len(index_shapes) == len(index_steps) == max(1, count * index_rank)
        and len(contigs) == count + int(index_rank == 0)
        and all(flag in (0, 1) for flag in contigs),
        "Scatter index metadata changed",
    )
    size = math.prod(shape)
    slices, index_shape = update_shape[index_rank:], update_shape[:index_rank]
    require(
        size > 0
        and math.prod(index_shape) == index_size
        and math.prod(slices) == slice_size
        and all(0 < length <= extent for length, extent in zip(slices, shape)),
        "Scatter extents changed",
    )
    require(
        steps == [math.prod(shape[i + 1 :]) for i in range(rank)],
        "Scatter output strides changed",
    )
    if packed:
        try:
            logical, trailing = scatter_storage.split(
                dtype, inputs["out"]["values"], size
            )
        except ValueError as error:
            raise ValueError("Scatter output extent changed") from error
        require(
            trailing == scatter_storage.tail(dtype, size),
            "Scatter output guards changed",
        )
        initial = scatter_arithmetic.values(np, dtype, logical).reshape(shape).copy()
    else:
        require(
            arrays["out"].size == size + len(COPY_GUARD),
            "Scatter output extent changed",
        )
        initial = arrays["out"][:size].reshape(shape).copy()
    expected = initial.copy()
    updates = view(np, arrays["updates"], update_shape, update_steps)
    require(
        not contiguous or updates.flags.c_contiguous,
        "Scatter update contiguity changed",
    )
    indices = []
    for i in range(count):
        require(
            index_shapes[i * index_rank : (i + 1) * index_rank] == index_shape,
            "Scatter index shapes changed",
        )
        index = view(
            np,
            arrays[f"idx{i}"],
            index_shape,
            index_steps[i * index_rank : (i + 1) * index_rank],
        )
        require(
            not contigs[i] or index.flags.c_contiguous,
            "Scatter index contiguity changed",
        )
        indices.append(index)
    for coordinate in np.ndindex(tuple(index_shape)):
        starts = [0] * rank
        for axis, index in zip(axes, indices):
            start = int(index[coordinate])
            start = start + shape[axis] if start < 0 else start
            require(
                0 <= start <= shape[axis] - slices[axis],
                "Scatter uploaded index exceeds output",
            )
            starts[axis] = start
        for offset in np.ndindex(tuple(slices)):
            destination = tuple(start + step for start, step in zip(starts, offset))
            value = updates[coordinate + offset]
            if packed:
                expected[destination] = scatter_arithmetic.combine(
                    np, dtype, operation, expected[destination], value
                )
            elif operation == "none":
                expected[destination] = value
            elif operation == "sum":
                expected[destination] += value
            elif operation == "prod":
                expected[destination] *= value
            elif operation == "min":
                expected[destination] = min(expected[destination], value)
            else:
                expected[destination] = max(expected[destination], value)
    ratio = index_size // size
    planned_work = (
        1
        if index_rank <= 1 or ratio < 1
        else 4 if ratio <= 4 else 8 if ratio < 16 else 16 if ratio < 32 else 32
    )
    require(work == planned_work, "Scatter work-per-thread changed")
    grid = [slice_size, (index_size + work - 1) // work, 1]
    metadata = {
        "operation": operation,
        "outputShape": shape,
        "outputStrides": steps,
        "updateShape": update_shape,
        "updateStrides": update_steps,
        "updateContiguous": contiguous,
        "sliceSizes": slices,
        "axes": axes,
        "indexShape": index_shape,
        "indexStrides": index_steps,
        "indexContiguous": contigs,
        "workPerThread": work,
        "updateCount": index_size * slice_size,
        "outputCount": size,
        "maximumIndex": (
            max(
                [size] + [value.size for name, value in arrays.items() if name != "out"]
            )
            - 1
        ),
    }
    if packed:
        actual = audit_packed_readback(np, event, dtype, size, expected, trailing)
    else:
        actual = audit_word_readback(np, event, dtype, size, expected, arrays)
    require(
        event["scatterMetadata"] == metadata
        and event["scatterStorageType"] == dtype
        and event["threads"] == size
        and event["dispatchVersion"] == 3
        and event["workgroupSize"] == [1, 1, 1]
        and event["workgroupCount"] == grid,
        "Scatter execution metadata changed",
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
        "Scatter native request changed",
    )
    audit_native_execution(event)
    return initial, indices, updates, actual


def _run_bounds(run):
    first, rows = run
    return first, first + rows


def validate(np, records, trace, upstream, *, profile="integer", lazy_copies=None):
    """Audit a native scatter run; returns its dispatch summary.

    ``lazy_copies`` is ``(installed_entries, cache)``: copies and casts outside
    the installed families, which narrow dtypes need, are then checked against
    their retained values, native modules and cached package rather than an
    installed package's identity.
    """
    from demos.integrations.mlx.portable_host.verify_bitwise import (
        verify_native_identity,
    )
    from demos.integrations.mlx.portable_host.verify_gather import validate_upstream
    from demos.integrations.mlx.portable_host.verify_scatter import (
        UPSTREAM_TESTS,
        validate_records,
    )

    validate_records(np, records, native=True, profile=profile)
    validate_upstream(upstream, native=True, tests=UPSTREAM_TESTS)
    cursor, count, targets = 0, 0, set()
    cases = workloads.profile_cases(profile)
    for case, record in zip(cases, records):
        end = cursor + record["dispatchCount"]
        require(end <= len(trace), "Scatter trace ends before its workload")
        events = [
            item for item in trace[cursor:end] if item["entry"].startswith("scatter")
        ]
        # A rounding sum runs as index-order launches; any other one launch.
        runs = workloads.ordered_runs(case)
        require(
            len(events)
            == (0 if case["layout"] == "empty" else len(runs) if runs else 1),
            "Scatter dispatch count changed",
        )
        source, expected_indices, expected_updates, expected = workloads.reference(
            np, case
        )
        # Each launch starts from the previous one's result and binds its own
        # index rows and their updates; the last result is the workload's.
        state = workloads.storage_words(np, case["dtype"], source)
        for position, event in enumerate(events):
            rows = slice(*_run_bounds(runs[position])) if runs else slice(None)
            initial, indices, updates, actual = audit_event(np, event)
            require(
                initial.shape == source.shape
                and workloads.storage_words(np, case["dtype"], initial) == state,
                "Scatter initialization does not belong to workload",
            )
            require(
                len(indices) == len(expected_indices),
                "Scatter workload index count changed",
            )
            for index, expected_index in zip(indices, expected_indices):
                # MLX's Python indexing promotes scalar indices to one dimension.
                if expected_index.ndim == 0:
                    expected_index = expected_index.reshape(1)
                expected_index = expected_index[rows]
                require(
                    index.shape == expected_index.shape
                    and index.dtype == expected_index.dtype
                    and words(np, index) == words(np, expected_index),
                    "Scatter index upload does not belong to workload",
                )
            tile_updates = expected_updates[rows]
            require(
                workloads.storage_words(np, case["dtype"], updates)
                == workloads.storage_words(np, case["dtype"], tile_updates)
                and updates.size == tile_updates.size,
                "Scatter updates do not belong to workload",
            )
            state = actual
            count += 1
        require(
            not events
            or state
            == record["actual"]
            == workloads.storage_words(np, case["dtype"], expected),
            "Scatter native and MLX results disagree",
        )
        targets.update(item["target"] for item in trace[cursor:end])
        cursor = end
    require(
        upstream["dispatchStart"] == cursor
        and cursor + upstream["dispatchCount"] == len(trace),
        "Scatter upstream trace boundary changed",
    )
    upstream_count = 0
    for event in trace[cursor:]:
        targets.add(event["target"])
        if event["entry"].startswith("scatter"):
            audit_event(np, event)
            upstream_count += 1
    require(
        len(targets) == 1 and upstream_count > 0,
        "Scatter target or upstream dispatch evidence missing",
    )
    target = targets.pop()
    installed, copies = lazy_copies if lazy_copies is not None else (None, None)
    others = []
    for event in trace:
        if event["entry"].startswith("scatter"):
            continue
        if (
            copies is not None
            and event["entry"] in LAZY_COPY_ENTRIES
            and event["entry"] not in installed
        ):
            copy_evidence.audit_event(event, copies)
        else:
            others.append(event)
    verify_native_identity(others, target)
    return {
        "target": target,
        "scatterDispatchCount": count,
        "upstreamScatterDispatchCount": upstream_count,
    }
