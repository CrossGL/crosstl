"""Reconcile retained gather uploads, native readbacks and MLX results."""

import hashlib
import math
from pathlib import Path

from demos.integrations.mlx.portable_host import gather_workloads
from demos.integrations.mlx.portable_host.gather_layout import signature
from demos.integrations.mlx.portable_host.runtime import BOOLEAN_GUARD, COPY_GUARD


def require(condition, message):
    if not condition:
        raise ValueError(message)


def storage(np, value):
    dtype = value["dtype"]
    words = value["values"]
    require(value["shape"] == [len(words)], "Gather upload shape changed")
    if dtype == "float32":
        require(
            value.get("encoding") == "ieee754-binary32", "Gather float encoding changed"
        )
        return np.array(words, dtype=np.uint32).view(np.float32)
    require("encoding" not in value, "Unexpected gather storage encoding")
    return np.array(words, dtype=dtype)


def view(np, array, shape, strides):
    require(len(shape) == len(strides), "Gather rank and strides disagree")
    require(
        all(type(size) is int and size > 0 for size in shape), "Invalid gather shape"
    )
    require(
        all(type(step) is int and step >= 0 for step in strides),
        "Invalid gather stride",
    )
    require(
        1 + sum((size - 1) * step for size, step in zip(shape, strides)) == array.size,
        "Gather upload storage span changed",
    )
    return np.ndarray(
        shape,
        dtype=array.dtype,
        buffer=array,
        strides=tuple(step * array.itemsize for step in strides),
    )


def audit_event(np, event):
    if event["entry"].startswith("gather_axis"):
        from demos.integrations.mlx.portable_host.gather_axis_evidence import (
            audit_event as audit_axis,
        )

        return audit_axis(np, event)
    dtype, index_dtype, count, ndim = signature(event["entry"])
    target = event["target"]
    require(target in {"metal", "directx", "opengl"}, "Unknown gather target")
    request = event["details"]["request"]
    inputs = {}
    for name, value in event["inputs"].items():
        binding = request["buffers"][name]
        layout = binding["binding"]["metadata"]["scalarLayout"]
        member = layout.get("memberName", name).removeprefix(
            event["entry"].rstrip("_") + "_"
        )
        member = "out" if member == "out_" else member
        require(member not in inputs, "Duplicate gather binding")
        require(
            binding["dtype"] == value["dtype"] == layout["elementType"]
            and binding["shape"] == value["shape"]
            and binding.get("encoding") == value.get("encoding"),
            "Gather binding layout changed",
        )
        inputs[member] = value
    required = {
        "src",
        "out",
        "src_shape",
        "src_strides",
        "src_ndim",
        "slice_sizes",
        "axes",
        "idx_shapes",
        "idx_strides",
        "idx_contigs",
        "idx_ndim",
    }
    require(
        set(inputs) == required | {f"idx{i}" for i in range(count)},
        "Gather bindings are incomplete",
    )
    arrays = {name: storage(np, value) for name, value in inputs.items()}
    shape = arrays["src_shape"].tolist()
    strides = arrays["src_strides"].tolist()
    slices = arrays["slice_sizes"].tolist()
    axes = arrays["axes"].tolist()
    idx_shape = arrays["idx_shapes"][:ndim].tolist()
    steps = arrays["idx_strides"].tolist()
    require(
        arrays["src_ndim"].tolist() == [len(shape)]
        and arrays["idx_ndim"].tolist() == [ndim],
        "Gather uploaded ranks changed",
    )
    require(
        len(axes) == count
        and len(set(axes)) == count
        and all(0 <= axis < len(shape) for axis in axes),
        "Gather uploaded axes changed",
    )
    require(
        len(slices) == len(shape)
        and all(0 < size <= extent for size, extent in zip(slices, shape)),
        "Gather uploaded slice sizes changed",
    )
    source = view(np, arrays["src"], shape, strides)
    indices = []
    for i in range(count):
        require(
            inputs[f"idx{i}"]["dtype"] == index_dtype, "Gather index storage changed"
        )
        require(
            arrays["idx_shapes"][i * ndim : (i + 1) * ndim].tolist() == idx_shape,
            "Gather index shapes disagree",
        )
        indices.append(
            view(np, arrays[f"idx{i}"], idx_shape, steps[i * ndim : (i + 1) * ndim])
        )
    expected = []
    for coordinate in np.ndindex(tuple(idx_shape)):
        starts = [0] * len(shape)
        for axis, index in zip(axes, indices):
            start = int(index[coordinate])
            start = start + shape[axis] if start < 0 else start
            require(
                0 <= start <= shape[axis] - slices[axis],
                "Gather uploaded index is outside its source",
            )
            starts[axis] = start
        part = source[
            tuple(slice(start, start + size) for start, size in zip(starts, slices))
        ]
        expected.extend(gather_workloads.words(np, part))
    grid = [idx_shape[0] if ndim else 1, math.prod(idx_shape[1:]), math.prod(slices)]
    metadata = {
        "sourceShape": shape,
        "sourceStrides": strides,
        "sliceSizes": slices,
        "axes": axes,
        "indexShape": idx_shape,
        "indexStrides": steps,
        "indexContiguous": arrays["idx_contigs"].tolist(),
        "sourceCount": arrays["src"].size,
        "outputCount": len(expected),
        "maximumIndex": (
            max(
                value.size if name != "out" else len(expected)
                for name, value in arrays.items()
            )
            - 1
        ),
    }
    return audit_result(np, event, inputs, source, indices, expected, grid, metadata)


def audit_result(np, event, inputs, source, indices, expected, grid, metadata):
    """Check native execution evidence independently of the indexing reference."""
    from demos.integrations.mlx.portable_host.gather_packages import (
        signature as entry_signature,
    )

    dtype = entry_signature(event["entry"])[0]
    target = event["target"]
    require(target in {"metal", "directx", "opengl"}, "Unknown gather target")
    request = event["details"]["request"]
    physical = (
        ("bool" if target == "metal" else "uint32") if dtype == "bool_" else dtype
    )
    require(
        inputs["src"]["dtype"] == inputs["out"]["dtype"] == physical,
        "Gather value storage changed",
    )
    guard = list(BOOLEAN_GUARD if dtype == "bool_" else COPY_GUARD)
    if dtype == "bool_":
        require(
            all(value in (0, 1) for value in inputs["src"]["values"]),
            "Noncanonical gather Boolean upload",
        )
        expected = [bool(word) if target == "metal" else int(word) for word in expected]
        guard = [bool(word) if target == "metal" else int(word) for word in guard]
    actual = event["gatherValues"]
    actual_words = gather_workloads.words(
        np, storage(np, dict(inputs["out"], shape=[len(actual)], values=actual))
    )
    require(
        actual_words == expected, "Gather native result disagrees with retained uploads"
    )
    require(
        event["gatherGuardValues"] == guard
        and inputs["out"]["values"] == [guard[0]] * len(actual) + guard,
        "Gather guard or output initialization changed",
    )
    require(
        event["threads"] == math.prod(grid) == len(actual)
        and event["dispatchVersion"] == 3
        and event["gatherStorageType"] == dtype
        and event["workgroupCount"] == grid
        and event["workgroupSize"] == [1, 1, 1],
        "Gather dispatch geometry changed",
    )
    expected_entry = {"opengl": "main", "directx": "CSMain"}.get(target, event["entry"])
    require(
        request["target"] == target
        and request["entryPoint"] == expected_entry
        and request["dispatch"]["workgroupCount"] == grid
        and request["dispatch"]["workgroupSize"] == [1, 1, 1],
        "Gather native request changed",
    )
    raw = storage(np, dict(inputs["out"], shape=[len(actual)], values=actual))
    if dtype == "bool_":
        raw = raw.astype(np.bool_)
    require(
        hashlib.sha256(raw.tobytes()).hexdigest() == event["outputHash"],
        "Gather host output hash changed",
    )
    require(event["gatherMetadata"] == metadata, "Gather metadata and uploads disagree")
    modules = [event["details"]["module"], *event["details"]["validationModules"]]
    extensions = {Path(module["file"]).suffix for module in modules}
    require(
        {"metal": {".air", ".metallib"}, "directx": {".dxil"}, "opengl": {".glsl"}}[
            target
        ]
        <= extensions,
        "Gather native compiler evidence is incomplete",
    )
    for module in modules:
        data = Path(module["file"]).read_bytes()
        require(
            data and hashlib.sha256(data).hexdigest() == module["sha256"],
            "Retained gather module identity changed",
        )
    artifact = event["artifact"]
    path = Path(event["packageRoot"]) / artifact["packagePath"]
    data = path.read_bytes()
    require(
        len(data) == artifact["sizeBytes"]
        and hashlib.sha256(data).hexdigest() == artifact["hash"]["value"],
        "Retained gather source identity changed",
    )
    identity = event["details"]["artifactIdentityVerification"]
    expected_identity = {key: artifact[key] for key in ("hash", "sizeBytes")}
    steps = event["details"]["adapterSteps"]
    compiler = {
        "metal": {"compile-metal-for-native-runtime", "link-metal-for-native-runtime"},
        "directx": {"compile-hlsl-for-directx-runtime"},
        "opengl": {"validate-glsl-for-opengl-runtime"},
    }[target]
    require(
        identity.get("verificationStatus") == "verified"
        and identity.get("target") == target
        and identity.get("expectedIdentity")
        == identity.get("observedIdentity")
        == expected_identity
        and event["details"]["nativeRuntimeDispatch"] == request
        and all(step.get("status") == "passed" for step in steps)
        and compiler <= {step.get("action") for step in steps},
        "Gather native compilation or artifact verification is incomplete",
    )
    return source, indices, expected


def validate(np, records, trace, upstream, *, workloads=gather_workloads):
    require(
        len(records) == len(list(workloads.cases())),
        "Gather workload set is incomplete",
    )
    cursor, targets = 0, set()
    for case, record in zip(workloads.cases(), records):
        require(
            record["dispatchStart"] == cursor,
            "Gather workload dispatch boundary changed",
        )
        end = cursor + record["dispatchCount"]
        events = [
            event for event in trace[cursor:end] if event["entry"].startswith("gather")
        ]
        require(
            len(events) == 1, "Each gather workload requires exactly one native gather"
        )
        event = events[0]
        source, indices, actual = audit_event(np, event)
        expected_source, expected_indices, expected = workloads.reference(np, case)
        layout_source = (
            np.ascontiguousarray(expected_source)
            if any(stride < 0 for stride in expected_source.strides)
            else expected_source
        )
        require(
            list(source.shape) == list(expected_source.shape)
            and gather_workloads.words(np, source)
            == gather_workloads.words(np, expected_source),
            "Gather source upload does not belong to its workload",
        )
        require(
            event["gatherMetadata"]["sourceStrides"]
            == [stride // layout_source.itemsize for stride in layout_source.strides],
            "Gather workload source layout changed",
        )
        require(
            len(indices) == len(expected_indices)
            and all(
                left.shape == right.shape and left.tolist() == right.tolist()
                for left, right in zip(indices, expected_indices)
            ),
            "Gather index upload does not belong to its workload",
        )
        index_layouts = [
            (
                np.ascontiguousarray(index)
                if any(stride < 0 for stride in index.strides)
                else index
            )
            for index in expected_indices
        ]
        require(
            event["gatherMetadata"]["indexStrides"]
            == (
                [
                    stride // index.itemsize
                    for index in index_layouts
                    for stride in index.strides
                ]
                or [0]
            ),
            "Gather workload index layout changed",
        )
        require(
            actual == record["actual"] == gather_workloads.words(np, expected),
            "Gather readback and MLX result disagree",
        )
        targets.add(event["target"])
        cursor = end
    require(
        upstream["dispatchStart"] == cursor
        and upstream["dispatchCount"] > 0
        and cursor + upstream["dispatchCount"] == len(trace),
        "Upstream gather trace is incomplete",
    )
    remaining = [
        event for event in trace[cursor:] if event["entry"].startswith("gather")
    ]
    require(remaining, "Upstream indexing did not execute native gather")
    for event in remaining:
        audit_event(np, event)
        targets.add(event["target"])
    require(
        len(targets) == 1 and all(event["target"] in targets for event in trace),
        "Gather trace mixes native targets",
    )
    target = targets.pop()
    from demos.integrations.mlx.portable_host.verify_bitwise import (
        verify_native_identity,
    )

    verify_native_identity(
        [event for event in trace if not event["entry"].startswith("gather")], target
    )
    return {"target": target, "gatherDispatchCount": len(records) + len(remaining)}
