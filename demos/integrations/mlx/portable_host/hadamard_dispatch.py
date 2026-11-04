"""Execute Hadamard stage packages and commit guarded MLX outputs."""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import hadamard_layout
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.matmul_dispatch import (
    GUARD_COUNT,
    POISON,
    _words,
    encode_binding,
)
from demos.integrations.mlx.portable_host.quantization_dispatch import (
    GUARDS,
    WORD_FORMATS,
    decode,
)
from demos.integrations.mlx.portable_host.trace_records import write_trace


def _storage(info, supplied, elements):
    """Each parameter's words: the input and scale, then a guarded output."""
    dtype = info["dtype"]
    return {
        "in": _words(supplied["in"], dtype) + [POISON[dtype]] * GUARD_COUNT,
        "out": [GUARDS[dtype]] * (elements + GUARD_COUNT),
        "scale": [ctypes.c_uint32.from_address(supplied["scale"].data).value],
    }


def _inputs(host, info, descriptor, storage):
    """Each reflected binding's value, the output binding and the bound names."""
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            # The runtime supplies DirectX's dispatch-size constants itself.
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if name not in storage or name in bound or binding["name"] in inputs:
            raise ValueError("Reflected Hadamard binding differs from its source")
        bound.add(name)
        inputs[binding["name"]] = encode_binding(
            "float32" if name == "scale" else info["dtype"],
            storage[name],
            host.target,
            binding["scalarLayout"],
            binding.get("kind", ""),
        )
        if name == "out":
            if binding.get("access") != "read_write":
                raise ValueError("Hadamard output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if bound != set(hadamard_layout.PARAMETERS) or output is None:
        raise ValueError("Hadamard reflection is incomplete")
    return inputs, observed, output


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    names = hadamard_layout.PARAMETERS
    if host.hadamards is None or count != len(names) or not buffers or launch is None:
        raise ValueError(
            "Hadamard dispatch requires pinned packages, buffers and geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Hadamard buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Hadamard buffer names must be unique")
        supplied[name] = buffer
    execution = launch.execution()
    metadata = hadamard_layout.validate(entry, supplied, threads, execution)
    info = hadamard_layout.signature(entry)
    dtype = info["dtype"]
    storage = _storage(info, supplied, threads)
    descriptor, directory = host.hadamards.get(entry)
    inputs, observed, output = _inputs(host, info, descriptor, storage)
    outputs = {output: inputs[output]}
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        outputs,
        execution,
        expected_target=host.target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != {output}:
        raise RuntimeError("Native Hadamard execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError("Native Hadamard output layout differs")
    words = decode(value, dtype, initial["dtype"])
    if (
        len(words) != threads + GUARD_COUNT
        or words[threads:] != [GUARDS[dtype]] * GUARD_COUNT
    ):
        raise RuntimeError("Native Hadamard output guard differs")
    itemsize = hadamard_layout.ITEM_SIZES[dtype]
    payload = struct.pack("<" + str(threads) + WORD_FORMATS[itemsize], *words[:threads])
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "hadamardMetadata": metadata,
        "inputHash": (
            hashlib.sha256(
                ctypes.string_at(supplied["in"].data, threads * itemsize)
            ).hexdigest()
        ),
        "outputValues": words[:threads] if threads <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied["out"].data, payload, len(payload))
    host.dispatch_count += 1
