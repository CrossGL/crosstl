"""Execute affine packages and commit all guarded outputs to MLX storage."""

import ctypes
import hashlib
import json
import struct

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
)
from crosstl.translator.resource_storage import encoded_storage_dtype
from demos.integrations.mlx.portable_host import half_storage, quantization_layout
from demos.integrations.mlx.portable_host.gather_dispatch import execute

GUARDS = {"uint8": 0xA5, "float32": 0x3EAAAAAB, "float16": 0x3555, "bfloat16": 0x3EAB}
GUARD_COUNT = 17
WORD_TYPES = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32}
WORD_FORMATS = {1: "B", 2: "H", 4: "I"}


def encode(words, logical, physical):
    words = list(words)
    size = quantization_layout.ITEM_SIZES[logical]
    if any(type(word) is not int or not 0 <= word < 1 << (8 * size) for word in words):
        raise ValueError("Invalid affine storage word")
    if logical == "uint8" and physical in {"uint8", "uint32"}:
        return {"dtype": physical, "shape": [len(words)], "values": words}
    if logical == "bfloat16" and physical == "uint16":
        return {"dtype": physical, "shape": [len(words)], "values": words}
    if logical == "float32" and physical == "float32":
        encoding = FLOAT32_BITS
    elif logical == "float16" and physical in {"float16", "float32"}:
        encoding = FLOAT16_BITS if physical == "float16" else FLOAT32_BITS
        if physical == "float32":
            words = [half_storage.widen(word) for word in words]
    elif logical == "bfloat16" and physical in {"bfloat16", "float32"}:
        encoding = BFLOAT16_BITS if physical == "bfloat16" else FLOAT32_BITS
        if physical == "float32":
            words = [word << 16 for word in words]
    else:
        raise ValueError("Unsupported affine physical storage")
    return {
        "dtype": physical,
        "shape": [len(words)],
        "encoding": encoding,
        "values": words,
    }


def decode(value, logical, physical):
    words = value.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native affine readback words are missing")
    if logical == "float16" and physical == "float32":
        words = [half_storage.narrow(word) for word in words]
    elif logical == "bfloat16" and physical == "float32":
        if any(
            type(word) is not int or not 0 <= word <= 0xFFFFFFFF or word & 0xFFFF
            for word in words
        ):
            raise RuntimeError("Native affine bfloat carrier is not exact")
        words = [word >> 16 for word in words]
    if encode(words, logical, physical) != value:
        raise RuntimeError("Native affine readback storage differs")
    return words


def dispatch(host, entry, buffers, count, elements, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.quantization is None or count != 4 or not buffers or launch is None:
        raise ValueError(
            "Affine dispatch requires pinned packages, four buffers and geometry"
        )
    supplied = {}
    for i in range(count):
        buffer = buffers[i]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Affine buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Affine buffer names must be unique")
        supplied[name] = buffer
    execution = launch.execution()
    metadata = quantization_layout.validate(entry, supplied, elements, execution)
    descriptor, directory = host.quantization.get(entry)
    inputs, outputs, matched = {}, {}, {}
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"]).removeprefix(entry + "_")
        name = "out" if name == "out_" else name
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError("Affine reflection does not match its buffer contract")
        buffer = supplied[name]
        logical, physical = buffer.dtype.decode("ascii"), layout["elementType"]
        size = quantization_layout.ITEM_SIZES[logical]
        storage = (
            "float16" if logical == "float16" and physical == "uint16" else physical
        )
        if (
            encoded_storage_dtype(
                layout,
                target=host.target,
                resource_kind=binding.get("kind", ""),
                logical_dtype=storage,
            )
            != physical
        ):
            raise ValueError("Affine reflected storage encoding differs")
        if layout["elementStrideBytes"] != {
            "uint8": 1,
            "uint16": 2,
            "uint32": 4,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }.get(physical):
            raise ValueError("Affine reflected element stride differs")
        if buffer.output:
            words = [GUARDS[logical]] * (buffer.count + GUARD_COUNT)
        else:
            words = list(
                ctypes.cast(
                    buffer.data, ctypes.POINTER(WORD_TYPES[size] * buffer.count)
                ).contents
            )
        value = encode(words, logical, storage)
        inputs[binding["name"]] = value
        matched[name] = binding["name"]
        if buffer.output:
            outputs[binding["name"]] = value
    if set(matched) != set(supplied):
        raise ValueError("Affine reflected buffers are incomplete")
    request = build_native_loader_dispatch_request(
        descriptor, directory, inputs, outputs, execution, expected_target=host.target
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != set(outputs):
        raise RuntimeError("Native affine execution returned incomplete outputs")
    readbacks, storage = {}, {}
    for name, buffer in supplied.items():
        if not buffer.output:
            continue
        reflected = matched[name]
        logical = buffer.dtype.decode("ascii")
        initial = outputs[reflected]
        output = result.outputs[reflected]
        if {key: item for key, item in output.items() if key != "values"} != {
            key: item for key, item in initial.items() if key != "values"
        }:
            raise RuntimeError("Native affine output layout differs")
        words = decode(output, logical, initial["dtype"])
        if (
            len(words) != buffer.count + GUARD_COUNT
            or words[buffer.count :] != [GUARDS[logical]] * GUARD_COUNT
        ):
            raise RuntimeError("Native affine output guard differs")
        readbacks[name] = words
        storage[name] = struct.pack(
            "<"
            + str(buffer.count)
            + WORD_FORMATS[quantization_layout.ITEM_SIZES[logical]],
            *words[: buffer.count],
        )
    event = {
        "entry": entry,
        "target": host.target,
        "threads": elements,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "quantizationMetadata": metadata,
        "inputs": inputs,
        "outputs": result.outputs,
        "quantizationReadbacks": readbacks,
        "outputHashes": {
            name: hashlib.sha256(data).hexdigest() for name, data in storage.items()
        },
    }
    # Validate every result and persist evidence before changing any MLX output.
    with host.trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    for name, data in storage.items():
        ctypes.memmove(supplied[name].data, data, len(data))
    host.dispatch_count += 1
