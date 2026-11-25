"""Reconstruct general-scatter results from retained uploads and native readbacks.

A complex64 sum uploads its updates as ``element_storage`` float32 lane pairs
and its output as two raw 32-bit atomic words per element, which the audit
reads back as complex64 values. A float16 or bfloat16 scatter uploads its
updates as exact 16-bit carriers and its output as packed atomic words, two
elements per word, whose unused high lane keeps the padding word.
"""

import ctypes
import hashlib
import math

from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.translator.resource_storage import encoded_storage_dtype
from demos.integrations.mlx.portable_host import bfloat_storage, half_storage
from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host.element_storage import (
    COMPLEX,
    reflected_layout_matches,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_input_bindings,
    audit_native_execution,
    require,
    view,
)
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.portable_host.runtime import (
    BFLOAT_COPY_ENTRY,
    COPY_GUARD,
    HALF_COPY_ENTRY,
    TYPES,
    physical_dtype,
)
from demos.integrations.mlx.portable_host.scatter_layout import (
    ATOMIC_LANES,
    ATOMIC_TYPES,
    ENCODINGS,
    METADATA,
    PACKED_TYPES,
    PADDING,
    atomic_storage_dtype,
    signature,
    storage_count,
    unpack_lanes,
)

# The 16-bit general copies that initialize packed outputs keep their
# retained native modules and report their storage beside the readback.
WORD_COPIES = {
    HALF_COPY_ENTRY: ("float16", "halfStorage", half_storage),
    BFLOAT_COPY_ENTRY: ("bfloat16", "bfloatStorage", bfloat_storage),
}


def audit_word_copy(event):
    """Check a 16-bit copy's compilation and its exact logical words."""
    dtype, key, storage = WORD_COPIES[event["entry"]]
    target = event["target"]
    audit_input_bindings(event)
    audit_native_execution(event)
    record = event.get(key)
    require(
        isinstance(record, dict)
        and record.get("logicalType") == dtype
        and record.get("physicalType") == physical_dtype(dtype, target)
        and record.get("encoding") == storage.encoding(target)
        and isinstance(record.get("values"), list)
        and len(record["values"]) == event["threads"],
        "Scatter 16-bit copy storage changed",
    )
    try:
        logical = storage.unpack(record["values"], target)
        guard = storage.unpack(record["guardValues"], target)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Scatter 16-bit copy carriers changed") from error
    require(
        logical == record.get("logicalWords") and guard == storage.GUARD,
        "Scatter 16-bit copy readback or guard changed",
    )


def update_lanes(dtype, value, target):
    """Recover raw 16-bit update words from their exact native carriers."""
    storage = half_storage if dtype == "float16" else bfloat_storage
    try:
        return storage.unpack(value["values"], target)
    except (TypeError, ValueError) as error:
        raise ValueError("Scatter 16-bit update carriers changed") from error


def raw_words(value):
    """Whether ``value`` holds only raw 32-bit words."""
    return isinstance(value["values"], list) and all(
        type(word) is int and 0 <= word <= 0xFFFFFFFF for word in value["values"]
    )


def audit_event(np, event):
    dtype, index_dtype, count, operation, contiguous, work = signature(event["entry"])
    target = event["target"]
    require(target in {"metal", "directx", "opengl"}, "Unknown scatter target")
    packed = dtype in PACKED_TYPES
    request = event["details"]["request"]
    audit_input_bindings(event)
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(event["entry"] + "_")
        member = "out" if member == "out_" else member
        require(member not in inputs, "Duplicate scatter binding")
        if dtype == COMPLEX and member == "updates":
            require(
                binding["dtype"] == value["dtype"]
                and binding["shape"] == value["shape"] == [len(value["values"])]
                and binding.get("encoding") == FLOAT32_BITS
                and reflected_layout_matches(layout, value, COMPLEX),
                "Scatter binding layout changed",
            )
            inputs[member] = value
            continue
        require(
            value["dtype"] == "bool" or value["dtype"] in TYPES,
            "Scatter storage type changed",
        )
        try:
            reflected = encoded_storage_dtype(
                layout,
                target=target,
                resource_kind=binding["binding"].get("kind", ""),
                logical_dtype=value["dtype"],
            )
        except ValueError as error:
            raise ValueError("Scatter binding storage encoding changed") from error
        encoding = ENCODINGS.get(value["dtype"])
        require(
            binding["dtype"] == value["dtype"]
            and layout["elementType"] == reflected
            and binding["shape"] == value["shape"]
            and binding.get("encoding") == encoding
            and (
                value.get("encoding") == encoding
                if encoding
                else "encoding" not in value
            )
            and layout["elementStrideBytes"]
            == (
                1 if value["dtype"] == "bool" else ctypes.sizeof(TYPES[value["dtype"]])
            ),
            "Scatter binding layout changed",
        )
        require(
            value["shape"]
            == (
                [len(value["values"]), 1] if member == "out" else [len(value["values"])]
            ),
            "Scatter upload shape changed",
        )
        if member == "out":
            require(
                layout.get("componentCount") == 1
                and layout.get("structMembers")
                == [
                    {
                        "name": "val",
                        "offsetBytes": 0,
                        "physicalType": ATOMIC_TYPES[
                            atomic_storage_dtype(dtype, target)
                        ],
                    }
                ],
                "Scatter atomic storage layout changed",
            )
        inputs[member] = value
    if dtype == COMPLEX:
        # complex64 updates travel as float32 lane pairs.
        update_storage = "float32"
    else:
        update_storage = physical_dtype(dtype, target) if packed else dtype
    dtypes = {
        **METADATA,
        "updates": update_storage,
        "out": atomic_storage_dtype(dtype, target),
        **{f"idx{i}": index_dtype for i in range(count)},
    }
    dtypes["idx_contigs"] = "bool" if target == "metal" else "uint32"
    require(set(inputs) == set(dtypes), "Scatter bindings are incomplete")
    for name, kind in dtypes.items():
        require(inputs[name]["dtype"] == kind, "Scatter storage type changed")
    require(
        all(value in (0, 1) for value in inputs["idx_contigs"]["values"]),
        "Scatter Boolean upload changed",
    )
    arrays = {}
    for name, value in inputs.items():
        if packed and name == "out":
            # Validated as packed words once the output extent is known.
            arrays[name] = value["values"]
        elif packed and name == "updates":
            arrays[name] = np.array(update_lanes(dtype, value, target), dtype=np.uint16)
        elif dtype in {"float32", COMPLEX} and name in {"out", "updates"}:
            require(
                raw_words(value), "Scatter float upload is not raw binary32 storage"
            )
            arrays[name] = np.array(value["values"], dtype=np.uint32)
            # The output keeps its words so its trailing guard stays exact.
            if name == "updates" or dtype == "float32":
                arrays[name] = arrays[name].view(
                    np.complex64 if dtype == COMPLEX else np.float32
                )
        else:
            arrays[name] = np.array(value["values"], dtype=value["dtype"])
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
    # One atomic word per element, two per complex64 element, or one per two
    # packed 16-bit elements.
    words_count = storage_count(dtype, size)
    if packed:
        uploaded = arrays["out"]
        require(
            isinstance(uploaded, list)
            and len(uploaded) == words_count + len(COPY_GUARD),
            "Scatter output extent changed",
        )
        try:
            lanes, padding = unpack_lanes(uploaded[:words_count], size)
        except ValueError as error:
            raise ValueError("Scatter packed output upload changed") from error
        require(
            padding == [PADDING[dtype]] * len(padding),
            "Scatter packed padding lane changed",
        )
        guard_words = uploaded[words_count:]
        arrays["out"] = np.array(lanes, dtype=np.uint16)
        initial = arrays["out"]
    else:
        require(
            arrays["out"].size == words_count + len(COPY_GUARD),
            "Scatter output extent changed",
        )
        guard_words = words(np, arrays["out"][words_count:])
        initial = arrays["out"][:words_count]
        if dtype == COMPLEX:
            initial = initial.view(np.complex64)
    initial = initial.reshape(shape).copy()
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
                expected[destination] = workloads.combine(
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
        # A complex64 output's atomic words are its largest index space; a
        # packed 16-bit output indexes no further than its elements.
        "maximumIndex": (
            max(
                value.size if name != "out" else size * ATOMIC_LANES.get(dtype, 1)
                for name, value in arrays.items()
            )
            - 1
        ),
    }
    readback = event["scatterValues"]
    storage = "uint32" if dtype in {"float32", COMPLEX} or packed else dtype
    limits = np.iinfo(storage)
    require(
        isinstance(readback, list)
        and len(readback) == words_count
        and all(
            type(value) is int and limits.min <= value <= limits.max
            for value in readback
        ),
        "Scatter readback is outside its storage type",
    )
    if packed:
        lanes, padding = unpack_lanes(readback, size)
        require(
            padding == [PADDING[dtype]] * len(padding),
            "Scatter native readback changed its packed padding lane",
        )
        raw = np.array(lanes, dtype=np.uint16)
    else:
        raw = np.array(readback, dtype=storage)
    actual = words(np, raw.view(np.complex64) if dtype == COMPLEX else raw)
    require(
        actual == words(np, expected), "Scatter native readback disagrees with uploads"
    )
    require(
        event["scatterGuardValues"] == guard_words == COPY_GUARD,
        "Scatter output guards changed",
    )
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
    require(
        hashlib.sha256(raw.tobytes()).hexdigest() == event["outputHash"],
        "Scatter output hash changed",
    )
    audit_native_execution(event)
    return initial, indices, updates, actual


def validate(np, records, trace, upstream, *, float32=False, half=False):
    from demos.integrations.mlx.portable_host.verify_bitwise import (
        verify_native_identity,
    )
    from demos.integrations.mlx.portable_host.verify_gather import validate_upstream
    from demos.integrations.mlx.portable_host.verify_scatter import (
        UPSTREAM_TESTS,
        validate_records,
    )

    validate_records(np, records, native=True, float32=float32, half=half)
    validate_upstream(upstream, native=True, tests=UPSTREAM_TESTS)
    cursor, count, targets = 0, 0, set()
    cases = workloads.profile_cases(float32=float32, half=half)
    for case, record in zip(cases, records):
        end = cursor + record["dispatchCount"]
        require(end <= len(trace), "Scatter trace ends before its workload")
        events = [
            item for item in trace[cursor:end] if item["entry"].startswith("scatter")
        ]
        require(
            len(events) == (0 if case["layout"] == "empty" else 1),
            "Scatter dispatch count changed",
        )
        source, expected_indices, expected_updates, expected = workloads.reference(
            np, case
        )
        for event in events:
            initial, indices, updates, actual = audit_event(np, event)
            require(
                initial.shape == source.shape
                and words(np, initial) == words(np, source),
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
                require(
                    index.shape == expected_index.shape
                    and index.dtype == expected_index.dtype
                    and words(np, index) == words(np, expected_index),
                    "Scatter index upload does not belong to workload",
                )
            require(
                words(np, updates) == words(np, expected_updates)
                and updates.size == expected_updates.size,
                "Scatter updates do not belong to workload",
            )
            require(
                actual == record["actual"] == words(np, expected),
                "Scatter native and MLX results disagree",
            )
            count += 1
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
    copies = [event for event in trace if event["entry"] in WORD_COPIES]
    for event in copies:
        audit_word_copy(event)
    verify_native_identity(
        [
            event
            for event in trace
            if not event["entry"].startswith("scatter")
            and event["entry"] not in WORD_COPIES
        ],
        target,
    )
    return {
        "target": target,
        "scatterDispatchCount": count,
        "upstreamScatterDispatchCount": upstream_count,
    }
