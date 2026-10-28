"""Dispatch checked scan buffers and retain exact native storage evidence."""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host import (
    bfloat_storage,
    half_storage,
    scan_layout,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import write_trace


def guard(dtype):
    if dtype in {"float16", "bfloat16"}:
        return [0x3555] * 32
    if dtype == "bool_":
        return [index % 2 for index in range(32)]
    return [0x6A15BEEF] * 32


def packet(dtype, target, words):
    from demos.integrations.mlx.portable_host.runtime import physical_dtype

    encoding = None
    if dtype in {"float16", "bfloat16"}:
        storage = half_storage if dtype == "float16" else bfloat_storage
        words, encoding = storage.pack(words, target), storage.encoding(target)
    elif dtype == "bool_":
        if any(type(word) is not int or word not in (0, 1) for word in words):
            raise ValueError("Scan Boolean storage must contain zero or one")
        words = [bool(word) for word in words] if target == "metal" else list(words)
    elif dtype == "float32":
        encoding = FLOAT32_BITS
    return {
        "dtype": physical_dtype(dtype, target),
        "shape": [len(words)],
        "values": list(words),
        **({"encoding": encoding} if encoding else {}),
    }


def unpack(dtype, target, words):
    if dtype in {"float16", "bfloat16"}:
        storage = half_storage if dtype == "float16" else bfloat_storage
        return storage.unpack(words, target)
    if dtype == "bool_":
        kind = bool if target == "metal" else int
        if any(type(word) is not kind or word not in (0, 1) for word in words):
            raise ValueError("Native scan returned invalid Boolean storage")
        return [int(word) for word in words]
    ctype = scan_layout.TYPES[dtype]
    if any(type(word) is not int or ctype(word).value != word for word in words):
        raise ValueError("Native scan returned invalid scalar storage")
    return list(words)


def dispatch(host, entry, buffers, count, size, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    layout, _, source, destination = scan_layout.signature(entry)
    if (
        host.scans is None
        or launch is None
        or not buffers
        or count != (3 if layout == "contig" else 5)
    ):
        raise ValueError(
            "Scan dispatch requires complete buffers, pinned sources and launch geometry"
        )
    supplied = {}
    for index in range(count):
        buffer = buffers[index]
        if not buffer.name or not buffer.dtype:
            raise ValueError("Scan buffer identity is missing")
        name = buffer.name.decode("ascii")
        if name in supplied:
            raise ValueError("Scan buffer names must be unique")
        supplied[name] = buffer
    execution = launch.execution()
    metadata = scan_layout.validate(entry, supplied, size, execution)
    descriptor, directory = host.scans.get(entry, metadata, execution)
    inputs, outputs, observed, matched = {}, {}, [], set()
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError(
                "Reflected scan binding does not match its source parameter"
            )
        matched.add(name)
        buffer = supplied[name]
        dtype = buffer.dtype.decode("ascii")
        words = (
            [guard(destination)[0]] * size
            if name == "out"
            else scan_layout.values(buffer)
        )
        if name in {"in", "out"}:
            words += guard(dtype)
        value = packet(dtype, host.target, words)
        scalar = binding["scalarLayout"]
        physical_size = (
            1
            if value["dtype"] == "bool"
            else (
                2
                if value["dtype"] == "uint16"
                else ctypes.sizeof(scan_layout.TYPES[value["dtype"]])
            )
        )
        if (
            scalar["elementType"] != value["dtype"]
            or scalar["elementStrideBytes"] != physical_size
        ):
            raise ValueError("Reflected scan storage differs from the native buffer")
        inputs[binding["name"]] = value
        if name == "out":
            outputs[binding["name"]] = {
                key: item for key, item in value.items() if key != "values"
            }
        else:
            observed.append(binding["name"])
    if matched != set(supplied) or len(outputs) != 1:
        raise ValueError("Scan reflection does not cover the complete source contract")
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
    if result.status != "ok" or set(result.outputs) != set(outputs):
        raise RuntimeError("Native scan did not return its output")
    (output_name,) = outputs
    output = result.outputs[output_name]
    expected = outputs[output_name]
    if any(
        output.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native scan output layout differs")
    words = output.get("values")
    if not isinstance(words, list) or len(words) != size + 32:
        raise RuntimeError("Native scan output extent differs")
    words = unpack(destination, host.target, words)
    if words[size:] != guard(destination):
        raise RuntimeError("Native scan changed its output guard")
    storage = (scan_layout.TYPES[destination] * size)(*words[:size])
    payload = bytes(storage)
    event = {
        "entry": entry,
        "target": host.target,
        "threads": size,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "scanMetadata": metadata,
        "inputs": inputs,
        "outputValues": words[:size],
        "outputGuards": words[size:],
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    write_trace(host, event)
    ctypes.memmove(supplied["out"].data, payload, len(payload))
    host.dispatch_count += 1
