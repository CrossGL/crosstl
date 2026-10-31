"""Execute floating-point dot-product packages and commit guarded partial sums."""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import dot_layout
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
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.portable_host.trace_records import write_trace

# The bindings the translation must reflect: the kernel reads nothing else
# that is not a scalar.
REQUIRED = {"a", "b", "output"}


def _storage(entry, supplied, counts, parameters):
    """Every kernel parameter's words, with guards after the vectors."""
    dtype = dot_layout.signature(entry)["dtype"]
    storage = {
        "output": [GUARDS["float32"]] * (counts["output"] + GUARD_COUNT),
        "n": [parameters["n"]],
    }
    for name in dot_layout.VECTOR_PARAMETERS:
        storage[name] = _words(supplied[name], dtype) + [POISON[dtype]] * GUARD_COUNT
    if set(storage) != set(dot_layout.PARAMETERS):
        raise ValueError("Dot-product storage does not cover every parameter")
    return storage


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    names = dot_layout.HOST_PARAMETERS
    if host.dots is None or count != len(names) or not buffers or launch is None:
        raise ValueError(
            "Dot-product dispatch requires pinned packages, buffers and geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Dot-product buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Dot-product buffer names must be unique")
        supplied[name] = buffer
    counts, parameters = dot_layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = dot_layout.plan(entry, counts, parameters, threads, execution)
    dtype = dot_layout.signature(entry)["dtype"]
    storage = _storage(entry, supplied, counts, parameters)
    descriptor, directory = host.dots.get(entry)
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name not in storage
            or name in bound
            or binding["name"] in inputs
        ):
            raise ValueError("Reflected dot-product binding differs from its source")
        bound.add(name)
        inputs[binding["name"]] = encode_binding(
            dot_layout.parameter_type(entry, name),
            storage[name],
            host.target,
            binding["scalarLayout"],
            binding.get("kind", ""),
        )
        if name == "output":
            if binding.get("access") != "read_write":
                raise ValueError("Dot-product output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if not REQUIRED <= bound or output is None:
        raise ValueError("Dot-product reflection is incomplete")
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
        raise RuntimeError("Native dot-product execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError("Native dot-product output layout differs")
    words = decode(value, "float32", initial["dtype"])
    size = counts["output"]
    if (
        len(words) != size + GUARD_COUNT
        or words[size:] != [GUARDS["float32"]] * GUARD_COUNT
    ):
        raise RuntimeError("Native dot-product output guard differs")
    payload = struct.pack("<" + str(size) + WORD_FORMATS[4], *words[:size])
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "dotMetadata": metadata,
        "parameters": parameters,
        "boundParameters": sorted(bound),
        "inputHashes": {
            name: (
                hashlib.sha256(
                    ctypes.string_at(
                        supplied[name].data, parameters["n"] * ITEM_SIZES[dtype]
                    )
                ).hexdigest()
            )
            for name in dot_layout.VECTOR_PARAMETERS
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied["output"].data, payload, len(payload))
    host.dispatch_count += 1
