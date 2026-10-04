"""Dispatch validated indexing metadata and copy back native result storage."""

import ctypes
import hashlib
import json
from pathlib import Path

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.project.runtime_verification import (
    RuntimeExecutionState,
    RuntimeExecutorResult,
)
from demos.integrations.mlx.portable_host import (
    gather_axis_layout,
    gather_layout,
    scatter_axis_layout,
    scatter_layout,
)


def execute(host, request):
    adapter = host.executor.runtime_adapter
    availability = host.executor.is_available(request)
    if not availability.available:
        raise RuntimeError(f"Native gather runtime unavailable: {availability.reason}")
    if request.execution_plan.diagnostics:
        raise ValueError("Native gather execution plan contains diagnostics")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    retained = host.trace.parent / "native-modules"
    retained.mkdir(parents=True, exist_ok=True)

    def save(path):
        data = path.read_bytes()
        if not data:
            raise RuntimeError("Native gather compiler produced an empty module")
        digest = hashlib.sha256(data).hexdigest()
        destination = retained / (digest + path.suffix)
        destination.write_bytes(data)
        return {"file": str(destination), "sha256": digest}

    try:
        native = adapter.prepare_buffers(state)
        module = save(native.module_path)
        validation = [
            save(path)
            for directory in state.temporary_directories
            for path in Path(directory.name).rglob("*")
            if path.is_file() and path.suffix in {".spv", ".air", ".metallib", ".dxil"}
        ]
        outputs = adapter.dispatch(state, native)
        return RuntimeExecutorResult(
            outputs=outputs,
            details={
                "module": module,
                "validationModules": validation,
                "request": native.to_json(),
                "adapterSteps": [step.to_json() for step in state.adapter_steps],
                **state.details,
            },
        )
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import (
        BOOLEAN_GUARD,
        COPY_GUARD,
        DISPATCH_VERSION,
        boolean_values,
        physical_dtype,
    )

    if host.gathers is None or launch is None:
        raise ValueError(
            "Gather dispatch requires the pinned source root and launch geometry"
        )
    scatter = entry.startswith("scatter")
    scatter_axis = entry.startswith("scatter_axis")
    axis = entry.startswith("gather_axis") or scatter_axis
    layout_module = (
        scatter_axis_layout
        if scatter_axis
        else (
            scatter_layout if scatter else gather_axis_layout if axis else gather_layout
        )
    )
    parameters = layout_module.signature(entry)
    dtype, indices = parameters[0], parameters[2]
    if count != (11 if axis else (15 if scatter else 11) + indices) or not buffers:
        raise ValueError("Native gather buffer count does not match its entry")
    supplied = {}
    for i in range(count):
        buffer = buffers[i]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Native gather buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Native gather buffer names must be unique")
        supplied[name] = buffer
    execution = launch.execution()
    metadata = layout_module.validate(entry, supplied, threads, execution)
    descriptor, directory = host.gathers.get(entry, metadata["maximumIndex"])
    inputs, outputs, matched = {}, {}, set()
    guard = list(BOOLEAN_GUARD if dtype == "bool_" else COPY_GUARD)
    if dtype == "bool_" and host.target != "metal":
        guard = [int(value) for value in guard]
    output_name = None
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        layout = binding["scalarLayout"]
        member = layout.get("memberName", binding["name"])
        member = member.removeprefix(entry.rstrip("_") + "_")
        name = "out" if member == "out_" else member
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected gather binding identity does not match")
        matched.add(name)
        buffer = supplied[name]
        kind = buffer.dtype.decode("ascii")
        storage = physical_dtype(kind, host.target)
        if scatter and name == "out":
            storage = scatter_layout.atomic_storage_dtype(kind, host.target)
        size = 1 if storage == "bool" else ctypes.sizeof(gather_layout.TYPES[storage])
        if layout["elementType"] != storage or layout["elementStrideBytes"] != size:
            raise ValueError("Native and reflected gather layouts disagree")
        if name == "out" and not scatter:
            values = [guard[0]] * buffer.count
        elif kind == "float32":
            values = list(
                ctypes.cast(
                    buffer.data, ctypes.POINTER(ctypes.c_uint32 * buffer.count)
                ).contents
            )
        else:
            values = gather_layout.values(buffer)
        if name == "out":
            values += guard
        if kind == "bool_":
            values = boolean_values(values, "uint32" if name != "out" else storage)
            values = [
                bool(value) if storage == "bool" else int(value) for value in values
            ]
        shape = [len(values), 1] if scatter and name == "out" else [len(values)]
        if (
            scatter
            and name == "out"
            and (
                layout.get("componentCount") != 1
                or layout.get("structMembers")
                != [
                    {
                        "name": "val",
                        "offsetBytes": 0,
                        "physicalType": scatter_layout.ATOMIC_TYPES[storage],
                    }
                ]
            )
        ):
            raise ValueError("Native scatter atomic storage layout does not match")
        value = {"dtype": storage, "shape": shape, "values": values}
        if storage == "float32":
            value["encoding"] = FLOAT32_BITS
        inputs[binding["name"]] = value
        if name == "out":
            output_name = binding["name"]
            outputs[output_name] = {
                key: item for key, item in value.items() if key != "values"
            }
    if matched != set(supplied) or output_name is None:
        raise ValueError("Reflected gather bindings do not cover the operation")
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        outputs,
        execution,
        expected_target=host.target,
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != {output_name}:
        raise RuntimeError("Native gather executor did not return its output")
    output = result.outputs[output_name]
    storage = physical_dtype(dtype, host.target)
    if scatter:
        storage = scatter_layout.atomic_storage_dtype(dtype, host.target)
    size = threads + len(guard)
    if (
        output.get("dtype") != storage
        or output.get("shape") != ([size, 1] if scatter else [size])
        or output.get("encoding") != (FLOAT32_BITS if storage == "float32" else None)
        or not isinstance(output.get("values"), list)
        or len(output["values"]) != size
    ):
        raise RuntimeError("Native gather readback layout does not match")
    readback = output["values"]
    if dtype == "bool_":
        boolean_values(readback, storage)
    else:
        bits = (
            32 if dtype == "float32" else ctypes.sizeof(gather_layout.TYPES[dtype]) * 8
        )
        low, high = (
            (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            if dtype.startswith("int")
            else (0, 2**bits - 1)
        )
        if any(
            type(value) is not int or not low <= value <= high for value in readback
        ):
            raise RuntimeError("Native gather readback is outside its storage type")
    if readback[threads:] != guard:
        raise RuntimeError("Native gather changed the output buffer guard")
    ctype = ctypes.c_uint32 if dtype == "float32" else gather_layout.TYPES[dtype]
    native = (ctype * threads)(*readback[:threads])
    record = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        ("scatterMetadata" if scatter else "gatherMetadata"): metadata,
        ("scatterStorageType" if scatter else "gatherStorageType"): dtype,
        ("scatterValues" if scatter else "gatherValues"): readback[:threads],
        ("scatterGuardValues" if scatter else "gatherGuardValues"): readback[threads:],
        "inputs": inputs,
        "outputHash": hashlib.sha256(bytes(native)).hexdigest(),
    }
    with host.trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    ctypes.memmove(supplied["out"].data, native, ctypes.sizeof(native))
    host.dispatch_count += 1
