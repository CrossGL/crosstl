"""Execute translated MLX convolution packages and commit guarded MLX outputs.

One dispatcher serves the unfold kernels and the separable depthwise kernel.
Each launch is checked against ``conv_layout``'s plan, run through its
translated package with guarded outputs, and recorded in the trace before
the MLX output changes. The steel GEMM launches of the explicit routes take
``gemm_dispatch`` and ``gemm_splitk_dispatch``.
"""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import conv_layout as layout
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

ENTRY_PREFIXES = layout.ENTRY_PREFIXES
LABEL = "Convolution"


def is_entry(entry):
    try:
        layout.signature(entry)
    except ValueError:
        return False
    return True


def _storage(entry, supplied, counts, parameters):
    """Every bound parameter's words, with guards after the array buffers."""
    dtype = layout.signature(entry)["dtype"]
    storage = {layout.OUTPUT: [GUARDS[dtype]] * (counts[layout.OUTPUT] + GUARD_COUNT)}
    for name, buffer in supplied.items():
        if name == layout.OUTPUT:
            continue
        if name in counts:
            storage[name] = _words(buffer, dtype) + [POISON[dtype]] * GUARD_COUNT
        elif name != "params":
            value = parameters[name]
            storage[name] = list(value) if isinstance(value, list) else [value]
    return storage


def _params(binding, params, spatial):
    """MLXConvParams<N> as one reflected storage record."""
    scalar = binding["scalarLayout"]
    if (
        scalar.get("elementType") != "record"
        or scalar.get("elementStrideBytes") != layout.PARAMS_SIZES[spatial]
    ):
        raise ValueError(f"{LABEL} reflected parameter record differs")
    words = pack_storage_records(scalar, [dict(params)])
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _inputs(host, entry, descriptor, storage, parameters):
    """Each reflected binding's value, the output binding and the bound names."""
    info = layout.signature(entry)
    names = set(layout.host_parameters(entry))
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if provenance.get("kind") == "generated-execution-input":
            # threads_per_grid: the request derives it from the launch.
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name not in names
            or name in bound
            or binding["name"] in inputs
        ):
            raise ValueError(f"Reflected {LABEL} binding differs from its source")
        bound.add(name)
        if name == "params":
            inputs[binding["name"]] = _params(
                binding, parameters["params"], info["spatial"]
            )
        else:
            inputs[binding["name"]] = encode_binding(
                layout.parameter_type(entry, name),
                storage[name],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        if name == layout.OUTPUT:
            if binding.get("access") != "read_write":
                raise ValueError(f"{LABEL} output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if bound != names or output is None:
        raise ValueError(f"{LABEL} reflection is incomplete")
    return inputs, observed, output, bound


def _supplied(buffers, count, names):
    if count != len(names) or not buffers:
        raise ValueError(f"{LABEL} dispatch requires pinned packages and buffers")
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError(f"{LABEL} buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError(f"{LABEL} buffer names must be unique")
        supplied[name] = buffer
    return supplied


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    dtype = layout.signature(entry)["dtype"]
    if host.convs is None or launch is None:
        raise ValueError(f"{LABEL} dispatch requires pinned packages and geometry")
    supplied = _supplied(buffers, count, layout.host_parameters(entry))
    counts, parameters = layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = layout.plan(entry, counts, parameters, threads, execution)
    storage = _storage(entry, supplied, counts, parameters)
    descriptor, directory = host.convs.get(entry)
    inputs, observed, output, bound = _inputs(
        host, entry, descriptor, storage, parameters
    )
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
        raise RuntimeError(f"Native {LABEL} execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError(f"Native {LABEL} output layout differs")
    size = counts[layout.OUTPUT]
    words = decode(value, dtype, initial["dtype"])
    guard = GUARDS[dtype]
    if len(words) != size + GUARD_COUNT or words[size:] != [guard] * GUARD_COUNT:
        raise RuntimeError(f"Native {LABEL} output guard differs")
    itemsize = ITEM_SIZES[dtype]
    payload = struct.pack("<" + str(size) + WORD_FORMATS[itemsize], *words[:size])
    arrays = sorted(name for name in counts if name != layout.OUTPUT)
    event = {
        "entry": entry,
        "target": host.target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "convMetadata": metadata,
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
            for name in arrays
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied[layout.OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
