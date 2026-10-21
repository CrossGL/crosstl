"""Execute floating-point matrix-vector packages and commit guarded MLX outputs."""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import gemv_layout
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

# The bindings the translation must reflect: the kernels read nothing else
# that is not a scalar.
REQUIRED = {"mat", "in_vec", "out_vec"}
INERT_TYPES = {
    "alpha": "float32",
    "beta": "float32",
    "bias_batch_stride": "int64",
    "bias_stride": "int32",
}


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _storage(entry, supplied, counts, parameters, metadata):
    """Every kernel parameter's words, with guards after the matrix buffers."""
    dtype = gemv_layout.signature(entry)["dtype"]
    storage = {
        "out_vec": [GUARDS[dtype]] * (counts["out_vec"] + GUARD_COUNT),
        "bias": [POISON[dtype]] * GUARD_COUNT,
    }
    for name in ("mat", "in_vec"):
        storage[name] = _words(supplied[name], dtype) + [POISON[dtype]] * GUARD_COUNT
    for name, value in parameters.items():
        storage[name] = [value] if isinstance(value, int) else list(value)
    for name, value in metadata["inert"].items():
        values = value if isinstance(value, list) else [value]
        storage[name] = (
            [_float_word(item) for item in values]
            if INERT_TYPES[name] == "float32"
            else list(values)
        )
    if set(storage) != set(gemv_layout.PARAMETERS):
        raise ValueError("Matrix-vector storage does not cover every parameter")
    return storage


def _type(entry, name):
    return INERT_TYPES.get(name) or (
        gemv_layout.signature(entry)["dtype"]
        if name == "bias"
        else gemv_layout.parameter_type(entry, name)
    )


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    names = gemv_layout.HOST_PARAMETERS
    if host.gemvs is None or count != len(names) or not buffers or launch is None:
        raise ValueError(
            "Matrix-vector dispatch requires pinned packages, buffers and geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Matrix-vector buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Matrix-vector buffer names must be unique")
        supplied[name] = buffer
    counts, parameters = gemv_layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = gemv_layout.plan(entry, counts, parameters, threads, execution)
    dtype = gemv_layout.signature(entry)["dtype"]
    storage = _storage(entry, supplied, counts, parameters, metadata)
    descriptor, directory = host.gemvs.get(entry)
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
            raise ValueError("Reflected matrix-vector binding differs from its source")
        bound.add(name)
        inputs[binding["name"]] = encode_binding(
            _type(entry, name),
            storage[name],
            host.target,
            binding["scalarLayout"],
            binding.get("kind", ""),
        )
        if name == "out_vec":
            if binding.get("access") != "read_write":
                raise ValueError("Matrix-vector output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if not REQUIRED <= bound or output is None:
        raise ValueError("Matrix-vector reflection is incomplete")
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
        raise RuntimeError("Native matrix-vector execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError("Native matrix-vector output layout differs")
    words = decode(value, dtype, initial["dtype"])
    size = counts["out_vec"]
    if (
        len(words) != size + GUARD_COUNT
        or words[size:] != [GUARDS[dtype]] * GUARD_COUNT
    ):
        raise RuntimeError("Native matrix-vector output guard differs")
    payload = struct.pack(
        "<" + str(size) + WORD_FORMATS[ITEM_SIZES[dtype]], *words[:size]
    )
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "gemvMetadata": {key: item for key, item in metadata.items() if key != "inert"},
        "parameters": parameters,
        "boundParameters": sorted(bound),
        "inputHashes": {
            name: (
                hashlib.sha256(
                    ctypes.string_at(
                        supplied[name].data, supplied[name].count * ITEM_SIZES[dtype]
                    )
                ).hexdigest()
            )
            for name in ("mat", "in_vec")
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied["out_vec"].data, payload, len(payload))
    host.dispatch_count += 1
