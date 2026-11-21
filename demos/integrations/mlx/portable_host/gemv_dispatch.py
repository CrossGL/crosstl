"""Execute floating-point and complex64 matrix-vector packages and commit guarded MLX outputs."""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import element_storage, gemv_layout
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

COMPLEX = gemv_layout.COMPLEX
# (1.5, 1.5): exact in complex64, and no route may read it into an output.
COMPLEX_POISON = 0x3FC000003FC00000
FORMATS = {**WORD_FORMATS, 8: "Q"}

# The bindings the translation must reflect: the kernels read nothing else
# that is not a scalar, and the bias only with axpby.
REQUIRED = {"mat", "in_vec", "out_vec"}
AXPBY_REQUIRED = REQUIRED | {"bias"}
INERT_TYPES = {
    "alpha": "float32",
    "beta": "float32",
    "bias_batch_stride": "int64",
    "bias_stride": "int32",
}


def _float_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _guard(dtype):
    return element_storage.GUARDS[COMPLEX] if dtype == COMPLEX else GUARDS[dtype]


def _poison(dtype):
    return COMPLEX_POISON if dtype == COMPLEX else POISON[dtype]


def _matrix_words(buffer, dtype):
    """A matrix buffer's storage words: complex64 as one 64-bit word each."""
    if dtype == COMPLEX:
        return element_storage.buffer_words(buffer, dtype)
    return _words(buffer, dtype)


def _decode(value, dtype, target, physical):
    """The output's logical storage words, refusing a carrier they do not fill."""
    if dtype != COMPLEX:
        return decode(value, dtype, physical)
    words = value.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native matrix-vector readback words are missing")
    words = element_storage.unpack(dtype, target, words)
    if element_storage.packet(dtype, target, words) != value:
        raise RuntimeError("Native matrix-vector readback storage differs")
    return words


def _storage(entry, supplied, counts, parameters, metadata):
    """Every kernel parameter's words, with guards after the matrix buffers."""
    info = gemv_layout.signature(entry)
    dtype = info["dtype"]
    storage = {
        "out_vec": [_guard(dtype)] * (counts["out_vec"] + GUARD_COUNT),
        # Without axpby the kernels never read the bias.
        "bias": [_poison(dtype)] * GUARD_COUNT,
    }
    matrices = ("mat", "in_vec", "bias") if info["axpby"] else ("mat", "in_vec")
    for name in matrices:
        storage[name] = _matrix_words(supplied[name], dtype) + (
            [_poison(dtype)] * GUARD_COUNT
        )
    for name, value in parameters.items():
        if name in gemv_layout.FLOAT_PARAMETERS:
            storage[name] = [_float_word(value)]
        else:
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

    names = gemv_layout.host_parameters(entry)
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
    matrices = ("mat", "in_vec", "bias") if "axpby" in metadata else ("mat", "in_vec")
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
    required = AXPBY_REQUIRED if "axpby" in metadata else REQUIRED
    if not required <= bound or output is None:
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
    words = _decode(value, dtype, host.target, initial["dtype"])
    size = counts["out_vec"]
    if (
        len(words) != size + GUARD_COUNT
        or words[size:] != [_guard(dtype)] * GUARD_COUNT
    ):
        raise RuntimeError("Native matrix-vector output guard differs")
    itemsize = gemv_layout.ELEMENT_SIZES[dtype]
    payload = struct.pack("<" + str(size) + FORMATS[itemsize], *words[:size])
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
                        supplied[name].data, supplied[name].count * itemsize
                    )
                ).hexdigest()
            )
            for name in matrices
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied["out_vec"].data, payload, len(payload))
    host.dispatch_count += 1
