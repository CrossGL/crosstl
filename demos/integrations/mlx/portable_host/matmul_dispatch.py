"""Execute affine matrix-vector packages and commit guarded MLX outputs."""

import ctypes
import hashlib
import struct

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator.resource_storage import encoded_storage_dtype
from demos.integrations.mlx.portable_host import matmul_layout
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.quantization_dispatch import (
    GUARDS,
    WORD_FORMATS,
    WORD_TYPES,
    decode,
    encode,
)
from demos.integrations.mlx.portable_host.quantization_layout import ITEM_SIZES
from demos.integrations.mlx.portable_host.trace_records import write_trace

GUARD_COUNT = 8
# Exactly representable in every element type; no route may read it into an
# output that MLX keeps.
POISON = {"float32": 0x3FC00000, "float16": 0x3E00, "bfloat16": 0x3FC0}
WORD_POISON = 0xA5A5A5A5
STRIDES = {"uint8": 1, "uint16": 2, "uint32": 4, "float16": 2}
STRIDES.update(bfloat16=2, float32=4, int32=4, int64=8)


def encode_binding(dtype, words, host_target, layout, kind):
    """The runtime value of one reflected binding holding ``dtype`` words."""
    physical = layout["elementType"]
    if dtype in ("float32", "float16", "bfloat16"):
        storage = "float16" if dtype == "float16" and physical == "uint16" else physical
        if (
            encoded_storage_dtype(
                layout, target=host_target, resource_kind=kind, logical_dtype=storage
            )
            != physical
        ):
            raise ValueError("Matrix-vector reflected encoding differs")
        value = encode(words, dtype, storage)
    else:
        if physical != dtype:
            raise ValueError("Matrix-vector reflected integer type differs")
        value = {"dtype": dtype, "shape": [len(words)], "values": list(words)}
    if layout["elementStrideBytes"] != STRIDES.get(physical):
        raise ValueError("Matrix-vector reflected element stride differs")
    return value


def _value(entry, name, words, host_target, layout, kind):
    """The runtime value of one reflected binding."""
    return encode_binding(
        matmul_layout.parameter_type(entry, name), words, host_target, layout, kind
    )


def _words(buffer, dtype):
    size = ITEM_SIZES.get(dtype, 4)
    return list(
        ctypes.cast(
            buffer.data, ctypes.POINTER(WORD_TYPES[size] * buffer.count)
        ).contents
    )


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    names = matmul_layout.parameter_names(entry)
    if (
        host.quantized_matmuls is None
        or count != len(names)
        or not buffers
        or launch is None
    ):
        raise ValueError(
            "Quantized matrix-vector dispatch requires pinned packages, buffers and geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Quantized matrix-vector buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Quantized matrix-vector buffer names must be unique")
        supplied[name] = buffer
    counts, parameters = matmul_layout.read_storage(entry, supplied)
    execution = launch.execution()
    metadata = matmul_layout.plan(entry, counts, parameters, threads, execution)
    dtype = matmul_layout.signature(entry)["dtype"]
    storage = {
        name: (
            [GUARDS[dtype]] * (counts["y"] + GUARD_COUNT)
            if name == "y"
            else _words(supplied[name], "uint32" if name == "w" else dtype)
            + [WORD_POISON if name == "w" else POISON[dtype]]
            * (GUARD_COUNT + metadata["overreadPadding"].get(name, 0))
        )
        for name in matmul_layout.MATRIX_PARAMETERS
    }
    for name, value in metadata["parameters"].items():
        storage[name] = [value] if isinstance(value, int) else list(value)
    descriptor, directory = host.quantized_matmuls.get(entry)
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
            raise ValueError(
                "Reflected quantized matrix-vector binding differs from its source"
            )
        bound.add(name)
        value = _value(
            entry,
            name,
            storage[name],
            host.target,
            binding["scalarLayout"],
            binding.get("kind", ""),
        )
        inputs[binding["name"]] = value
        if name == "y":
            if binding.get("access") != "read_write":
                raise ValueError("Quantized matrix-vector output is not writable")
            output = binding["name"]
        else:
            observed.append(binding["name"])
    # The translation reflects every kernel parameter, batched or not.
    if bound != set(storage) or output is None:
        raise ValueError("Quantized matrix-vector reflection is incomplete")
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
        raise RuntimeError(
            "Native quantized matrix-vector execution returned no output"
        )
    initial, value = outputs[output], result.outputs[output]
    if {key: item for key, item in value.items() if key != "values"} != {
        key: item for key, item in initial.items() if key != "values"
    }:
        raise RuntimeError("Native quantized matrix-vector output layout differs")
    words = decode(value, dtype, initial["dtype"])
    size = counts["y"]
    if (
        len(words) != size + GUARD_COUNT
        or words[size:] != [GUARDS[dtype]] * GUARD_COUNT
    ):
        raise RuntimeError("Native quantized matrix-vector output guard differs")
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
        "matmulMetadata": {
            key: item for key, item in metadata.items() if key != "parameters"
        },
        "parameters": metadata["parameters"],
        "inputHashes": {
            name: (
                hashlib.sha256(
                    ctypes.string_at(
                        supplied[name].data,
                        supplied[name].count
                        * ITEM_SIZES.get(matmul_layout.parameter_type(entry, name), 4),
                    )
                ).hexdigest()
            )
            for name in ("w", "scales", "biases", "x")
        },
        "outputValues": words[:size] if size <= 4096 else None,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    # Validate the result and persist evidence before changing the MLX output.
    write_trace(host, event)
    ctypes.memmove(supplied["y"].data, payload, len(payload))
    host.dispatch_count += 1
