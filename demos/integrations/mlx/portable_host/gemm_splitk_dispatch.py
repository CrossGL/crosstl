"""Execute split-K steel GEMM packages and commit guarded MLX outputs.

``gemm_splitk`` writes the float32 ``C_split`` temporary and
``gemm_splitk_accum`` reads it into the MLX output; each dispatch commits its
output only after the native result and its guards check out.
"""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request, pack_storage_records
from demos.integrations.mlx.portable_host import gemm_splitk_layout as layout
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

# GEMMSpiltKParams holds thirteen 32-bit ints.
PARAMS_STRIDE = 52


def _params(binding, params):
    """GEMMSpiltKParams in the binding's reflected storage.

    The record holds only 32-bit ints, so targets may reflect it as an int32
    array in member order rather than as a packed record.
    """
    layout_ = binding["scalarLayout"]
    members = [member.get("name") for member in layout_.get("structMembers", ())]
    if layout_.get("elementStrideBytes") != PARAMS_STRIDE or members != list(
        layout.PARAMS_FIELDS
    ):
        raise ValueError("Split-K GEMM reflected parameter record differs")
    if layout_.get("elementType") == "record":
        words = pack_storage_records(layout_, [params])
        return {"dtype": "uint32", "shape": [len(words)], "values": words}
    if (
        layout_.get("elementType") != "int32"
        or layout_.get("componentCount") != len(layout.PARAMS_FIELDS)
        or [member.get("offsetBytes") for member in layout_["structMembers"]]
        != list(range(0, PARAMS_STRIDE, 4))
    ):
        raise ValueError("Split-K GEMM reflected parameter record differs")
    values = [params[name] for name in layout.PARAMS_FIELDS]
    return {"dtype": "int32", "shape": [len(values)], "values": values}


def _storage(entry, supplied, counts, parameters):
    """Every array and scalar parameter's words, guards after the arrays."""
    info = layout.signature(entry)
    output = layout.OUTPUTS[info["kind"]]
    storage = {}
    for name, count in counts.items():
        dtype = layout.parameter_type(entry, name)
        if name == output:
            storage[name] = [GUARDS[dtype]] * (count + GUARD_COUNT)
        else:
            storage[name] = (
                _words(supplied[name], dtype) + [POISON[dtype]] * GUARD_COUNT
            )
    for name in layout.SCALARS:
        if name in parameters:
            storage[name] = [parameters[name]]
    return storage


def _inputs(host, entry, descriptor, storage, parameters):
    """Each reflected binding's value, the output binding and the bound names."""
    info = layout.signature(entry)
    names, output_name = layout.PARAMETERS[info["kind"]], layout.OUTPUTS[info["kind"]]
    inputs, observed, output, bound = {}, [], None, set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        name = provenance.get("sourceResource", {}).get("parameter")
        if (
            "executionInput" in provenance
            or name not in names
            or name in bound
            or binding["name"] in inputs
        ):
            raise ValueError("Reflected split-K GEMM binding differs from its source")
        bound.add(name)
        if name == "params":
            inputs[binding["name"]] = _params(binding, parameters["params"])
        else:
            inputs[binding["name"]] = encode_binding(
                layout.parameter_type(entry, name),
                storage[name],
                host.target,
                binding["scalarLayout"],
                binding.get("kind", ""),
            )
        if name == output_name:
            if binding.get("access") != "read_write":
                raise ValueError("Split-K GEMM output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    if bound != set(names) or output is None:
        raise ValueError("Split-K GEMM reflection is incomplete")
    return inputs, observed, output, bound


def _supplied(buffers, count):
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Split-K GEMM buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Split-K GEMM buffer names must be unique")
        supplied[name] = buffer
    return supplied


def _hash(buffer, dtype):
    return hashlib.sha256(
        ctypes.string_at(buffer.data, buffer.count * ITEM_SIZES[dtype])
    ).hexdigest()


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    info = layout.signature(entry)
    names = layout.PARAMETERS[info["kind"]]
    if (
        host.splitk_gemms is None
        or count != len(names)
        or not buffers
        or launch is None
    ):
        raise ValueError(
            "Split-K GEMM dispatch requires pinned packages, buffers and geometry"
        )
    supplied = _supplied(buffers, count)
    counts, parameters = layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = layout.plan(entry, counts, parameters, threads, execution)
    output_name = layout.OUTPUTS[info["kind"]]
    dtype = layout.parameter_type(entry, output_name)
    storage = _storage(entry, supplied, counts, parameters)
    descriptor, directory = host.splitk_gemms.get(entry)
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
        raise RuntimeError("Native split-K GEMM execution returned no output")
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError("Native split-K GEMM output layout differs")
    words = decode(value, dtype, initial["dtype"])
    size = counts[output_name]
    if (
        len(words) != size + GUARD_COUNT
        or words[size:] != [GUARDS[dtype]] * GUARD_COUNT
    ):
        raise RuntimeError("Native split-K GEMM output guard differs")
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
        "splitkMetadata": metadata,
        "parameters": parameters,
        "boundParameters": sorted(bound),
        "inputHashes": {
            name: _hash(supplied[name], layout.parameter_type(entry, name))
            for name in counts
            if name != output_name
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied[output_name].data, payload, len(payload))
    host.dispatch_count += 1
