"""Dispatch validated indexing metadata and copy back native result storage."""

import ctypes
import hashlib
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
    gather_storage,
    scatter_axis_layout,
    scatter_layout,
)
from demos.integrations.mlx.portable_host.retained_modules import retain
from demos.integrations.mlx.portable_host.trace_records import write_trace


def execute(host, request):
    adapter = host.executor.runtime_adapter
    availability = host.executor.is_available(request)
    if not availability.available:
        raise RuntimeError(f"Native gather runtime unavailable: {availability.reason}")
    if request.execution_plan.diagnostics:
        raise ValueError("Native gather execution plan contains diagnostics")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    retained = host.trace.parent / "native-modules"

    def save(path):
        return retain(retained, path.read_bytes(), path.suffix)

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


def word_value(host, buffer, layout, name, scatter, guard):
    """A word-type buffer's runtime value: its own storage words."""
    from demos.integrations.mlx.portable_host.runtime import (
        boolean_values,
        physical_dtype,
    )

    kind = buffer.dtype.decode("ascii")
    storage = physical_dtype(kind, host.target)
    if scatter and name == "out":
        storage = scatter_layout.atomic_storage_dtype(kind, host.target)
    size = 1 if storage == "bool" else ctypes.sizeof(gather_layout.TYPES[storage])
    if layout["elementType"] != storage or layout["elementStrideBytes"] != size:
        raise ValueError("Native and reflected gather layouts disagree")
    if name == "out" and not scatter:
        values = [guard[0]] * buffer.count
    elif scatter and name == "out" and kind in scatter_layout.PACKED_TYPES:
        # Raw 16-bit words, two per atomic word; an odd tail pads its high lane.
        lanes = list(
            ctypes.cast(
                buffer.data, ctypes.POINTER(ctypes.c_uint16 * buffer.count)
            ).contents
        )
        values = scatter_layout.pack_lanes(lanes, scatter_layout.PADDING[kind])
    elif kind in scatter_layout.ATOMIC_LANES and (
        kind == "float32" or name == "out" and scatter
    ):
        # Raw 32-bit words: float32 bits, or a complex64 sum's float lanes.
        words = buffer.count * scatter_layout.ATOMIC_LANES[kind]
        values = list(
            ctypes.cast(buffer.data, ctypes.POINTER(ctypes.c_uint32 * words)).contents
        )
    else:
        values = gather_layout.values(buffer)
    if name == "out":
        values += guard
    if kind == "bool_":
        values = boolean_values(values, "uint32" if name != "out" else storage)
        values = [bool(value) if storage == "bool" else int(value) for value in values]
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
    return value


def element_value(name, dtype, scatter):
    """Whether buffer ``name`` travels as ``gather_storage`` element words.

    A scatter's output is its atomic words, so only its updates do.
    """
    values = {"updates"} if scatter else gather_storage.VALUES
    return name in values and gather_storage.extended(dtype)


def bind(host, entry, descriptor, supplied, dtype, scatter, guard):
    """Map reflected bindings to the supplied buffers and their packets."""
    inputs, outputs, matched = {}, {}, set()
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
        if element_value(name, dtype, scatter):
            value = gather_storage.bind_value(
                dtype, host.target, buffer, layout, name == "out"
            )
        else:
            value = word_value(host, buffer, layout, name, scatter, guard)
        inputs[binding["name"]] = value
        if name == "out":
            output_name = binding["name"]
            outputs[output_name] = {
                key: item for key, item in value.items() if key != "values"
            }
    if matched != set(supplied) or output_name is None:
        raise ValueError("Reflected gather bindings do not cover the operation")
    return inputs, outputs, output_name


def word_readback(host, dtype, scatter, threads, guard, output):
    """Check a word-type readback and return its values and host storage."""
    from demos.integrations.mlx.portable_host.runtime import (
        boolean_values,
        physical_dtype,
    )

    storage = physical_dtype(dtype, host.target)
    elements = threads
    packed = scatter and dtype in scatter_layout.PACKED_TYPES
    if scatter:
        storage = scatter_layout.atomic_storage_dtype(dtype, host.target)
        # The output's atomic words: two per complex64 element, one per two
        # packed 16-bit elements, otherwise one per element.
        threads = scatter_layout.storage_count(dtype, threads)
    # float32 bits, a complex64 element's lanes and packed 16-bit pairs all
    # travel as 32-bit words.
    wide = (
        dtype == "float32"
        or packed
        or scatter
        and scatter_layout.ATOMIC_LANES.get(dtype, 1) > 1
    )
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
        bits = 32 if wide else ctypes.sizeof(gather_layout.TYPES[dtype]) * 8
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
    if packed:
        lanes, padding = scatter_layout.unpack_lanes(readback[:threads], elements)
        if padding != [scatter_layout.PADDING[dtype]] * len(padding):
            raise RuntimeError("Native scatter changed its packed padding lane")
        return readback, (ctypes.c_uint16 * elements)(*lanes)
    ctype = ctypes.c_uint32 if wide else gather_layout.TYPES[dtype]
    return readback, (ctype * threads)(*readback[:threads])


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import (
        BOOLEAN_GUARD,
        COPY_GUARD,
        DISPATCH_VERSION,
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
    # A scatter's output is atomic words even when its values are not.
    extended = gather_storage.extended(dtype) and not scatter
    if extended:
        guard = gather_storage.guard_words(dtype)
    else:
        guard = list(BOOLEAN_GUARD if dtype == "bool_" else COPY_GUARD)
        if dtype == "bool_" and host.target != "metal":
            guard = [int(value) for value in guard]
    inputs, outputs, output_name = bind(
        host, entry, descriptor, supplied, dtype, scatter, guard
    )
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
    if extended:
        readback = gather_storage.readback(
            dtype, host.target, outputs[output_name], output, threads
        )
        native = gather_storage.native_words(dtype, readback[:threads])
    else:
        readback, native = word_readback(host, dtype, scatter, threads, guard, output)
    # Readback words before the guard: one per element, a complex64 sum's two
    # atomic words per element, or one per two packed 16-bit elements.
    words = scatter_layout.storage_count(dtype, threads) if scatter else threads
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
        ("scatterValues" if scatter else "gatherValues"): readback[:words],
        ("scatterGuardValues" if scatter else "gatherGuardValues"): readback[words:],
        "inputs": inputs,
        "outputHash": hashlib.sha256(bytes(native)).hexdigest(),
    }
    write_trace(host, record)
    ctypes.memmove(supplied["out"].data, native, ctypes.sizeof(native))
    host.dispatch_count += 1
