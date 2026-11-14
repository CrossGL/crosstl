"""Execute steel GEMM packages and commit guarded MLX outputs."""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import gemm_layout
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
# without has_batch or use_out_source.
REQUIRED = {"A", "B", "D", "params"}
# Reflected struct bindings and their GEMMParams and GEMMAddMMParams strides.
RECORD_STRIDES = {"params": 72, "addmm_params": 24}


def _record(binding, values):
    """One reflected struct record as uint32 words."""
    layout = binding["scalarLayout"]
    if (
        layout.get("elementStrideBytes")
        != RECORD_STRIDES[binding["provenance"]["sourceResource"]["parameter"]]
    ):
        raise ValueError("Steel GEMM reflected record stride differs")
    words = pack_storage_records(layout, [values])
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _storage(entry, supplied, counts, parameters):
    """Every array parameter's words, with guards after the matrix buffers."""
    dtype = gemm_layout.signature(entry)["dtype"]
    storage = {
        "D": [GUARDS[dtype]] * (counts["D"] + GUARD_COUNT),
        # use_out_source is false, so the kernel never reads C.
        "C": [POISON[dtype]] * GUARD_COUNT,
        "batch_shape": list(parameters["batch_shape"]),
        "batch_strides": list(parameters["batch_strides"]),
    }
    for name in ("A", "B"):
        storage[name] = _words(supplied[name], dtype) + [POISON[dtype]] * GUARD_COUNT
    return storage


def _type(entry, name):
    if name in ("C", "A", "B", "D"):
        return gemm_layout.signature(entry)["dtype"]
    return gemm_layout.VECTOR_TYPES[name]


def _inputs(host, entry, descriptor, storage, records):
    """Each reflected binding's value, the output binding and the bound names."""
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name not in gemm_layout.PARAMETERS
            or name in bound
            or binding["name"] in inputs
        ):
            raise ValueError("Reflected steel GEMM binding differs from its source")
        bound.add(name)
        if name in records:
            inputs[binding["name"]] = _record(binding, records[name])
        else:
            inputs[binding["name"]] = encode_binding(
                _type(entry, name),
                storage[name],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        if name == "D":
            if binding.get("access") != "read_write":
                raise ValueError("Steel GEMM output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if not REQUIRED <= bound or output is None:
        raise ValueError("Steel GEMM reflection is incomplete")
    return inputs, observed, output, bound


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    names = gemm_layout.HOST_PARAMETERS
    if host.gemms is None or count != len(names) or not buffers or launch is None:
        raise ValueError(
            "Steel GEMM dispatch requires pinned packages, buffers and geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Steel GEMM buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Steel GEMM buffer names must be unique")
        supplied[name] = buffer
    counts, parameters = gemm_layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = gemm_layout.plan(entry, counts, parameters, threads, execution)
    dtype = gemm_layout.signature(entry)["dtype"]
    storage = _storage(entry, supplied, counts, parameters)
    records = {"params": parameters["params"], "addmm_params": metadata["inert"]}
    descriptor, directory = host.gemms.get(entry)
    inputs, observed, output, bound = _inputs(host, entry, descriptor, storage, records)
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
        raise RuntimeError("Native steel GEMM execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError("Native steel GEMM output layout differs")
    words = decode(value, dtype, initial["dtype"])
    size = counts["D"]
    if (
        len(words) != size + GUARD_COUNT
        or words[size:] != [GUARDS[dtype]] * GUARD_COUNT
    ):
        raise RuntimeError("Native steel GEMM output guard differs")
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
        "gemmMetadata": {key: item for key, item in metadata.items() if key != "inert"},
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
            for name in ("A", "B")
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied["D"].data, payload, len(payload))
    host.dispatch_count += 1
