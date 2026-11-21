"""Execute the kernels MLX generates for fused tapes through translated packages.

The native host passes each launch of a compiled variant with the kernel's
source as a ``kernel_source`` buffer, every source parameter as a buffer of
its own name, and one-thread groups. The launch is checked against the
parsed parameters before upload, every output carries a trailing guard, and
the readback is decoded exactly into MLX's logical storage. An output may be
a donated input: every input is read before any output is written back.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import compiled_layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    buffer_words,
    guard_words,
    packet,
    reflected_layout_matches,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import write_trace

SOURCE_BUFFER = "kernel_source"


def kernel_source(supplied):
    """Remove and decode the ``kernel_source`` buffer the host appends."""
    buffer = supplied.pop(SOURCE_BUFFER, None)
    if buffer is None or buffer.dtype != b"utf8" or not buffer.data or buffer.output:
        raise ValueError("Compiled dispatch requires its kernel source")
    return ctypes.string_at(buffer.data, buffer.count).decode("utf-8")


def bind(descriptor, parameters, uploads, outputs):
    """Map reflected bindings to source parameters through their origins."""
    inputs, expected, observed, matched = {}, {}, [], {}
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if name not in uploads or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected compiled binding does not match its source")
        matched[name] = binding["name"]
        value = uploads[name]
        if not reflected_layout_matches(
            binding["scalarLayout"], value, parameters[name]
        ):
            raise ValueError(
                "Reflected compiled storage differs from the native buffer"
            )
        inputs[binding["name"]] = value
        if name in outputs:
            if binding.get("access") != "read_write":
                raise ValueError("Compiled output is not writable")
            expected[binding["name"]] = {
                k: v for k, v in value.items() if k != "values"
            }
        else:
            observed.append(binding["name"])
    if set(matched) != set(uploads):
        raise ValueError(
            "Compiled reflection does not cover the complete source contract"
        )
    return inputs, expected, observed, matched


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.compiled is None or launch is None or not buffers:
        raise ValueError(
            "Compiled dispatch requires complete buffers, pinned headers and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Compiled")
    source = kernel_source(supplied)
    listed = compiled_layout.parse(entry, source)
    compiled_layout.require_defined(compiled_layout.operations(source))
    execution = launch.execution()
    metadata = compiled_layout.validate(
        entry, source, listed, supplied, threads, execution
    )
    target = host.target
    parameters = {name: dtype for name, dtype, _ in listed}
    outputs = [name for name, _, direction in listed if direction == 1]
    values = {
        name: buffer_words(supplied[name], dtype)
        for name, dtype, direction in listed
        if direction == 0
    }
    # A fresh output is filled with guards, so the readback is never stale.
    initial = {
        name: guard_words(parameters[name], supplied[name].count) for name in outputs
    }
    tail = {name: guard_words(parameters[name]) for name in outputs}
    uploads = {
        **{
            name: packet(parameters[name], target, words)
            for name, words in values.items()
        },
        **{
            name: packet(parameters[name], target, initial[name] + tail[name])
            for name in outputs
        },
    }
    descriptor, directory = host.compiled.get(entry, source)
    inputs, expected, observed, matched = bind(descriptor, parameters, uploads, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        expected,
        execution,
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != {
        matched[name] for name in outputs
    }:
        raise RuntimeError("Native compiled kernel did not return its outputs")
    payloads, words_out = {}, {}
    for name in outputs:
        returned = result.outputs[matched[name]]
        if any(
            returned.get(key) != expected[matched[name]].get(key)
            for key in ("dtype", "encoding", "shape")
        ):
            raise RuntimeError("Native compiled output layout differs")
        words = returned.get("values")
        if not isinstance(words, list):
            raise RuntimeError("Native compiled output extent differs")
        words = unpack(parameters[name], target, words)
        size = supplied[name].count
        if len(words) != size + GUARD_COUNT:
            raise RuntimeError("Native compiled output extent differs")
        if words[size:] != tail[name]:
            raise RuntimeError("Native compiled kernel changed its output guard")
        words_out[name] = words[:size]
        payloads[name] = bytes((CTYPES[parameters[name]] * size)(*words[:size]))
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "compiledMetadata": metadata,
        "sourceHash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters": [list(item) for item in listed],
        "operandValues": values,
        "outputValues": words_out,
        "outputHashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }
    # Validate every result and persist evidence before changing MLX outputs.
    write_trace(host, event)
    for name, payload in payloads.items():
        ctypes.memmove(supplied[name].data, payload, len(payload))
    host.dispatch_count += 1
