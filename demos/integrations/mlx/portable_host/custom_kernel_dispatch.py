"""Execute the kernels ``mx.fast.metal_kernel`` builds through translated packages.

The native host passes each launch with the kernel's source as a
``kernel_source`` buffer, its math mode as ``compile_options``, every buffer
slot of the source's signature under the slot's name, and the exact thread
grid with Metal's threadgroup shape. The buffers are checked against the
parsed signature before upload. Every output carries a trailing guard and,
unless the kernel has an initial value, starts as guards, so the readback is
never stale; the readback is decoded exactly into MLX's storage. Metal runs
the exact grid; Direct3D and OpenGL run its dispatch regions in one native
sequence.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import custom_kernel_layout as layout
from demos.integrations.mlx.portable_host import region_dispatch
from demos.integrations.mlx.portable_host.compiled_dispatch import kernel_source
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


def bind(descriptor, parameters, uploads, outputs):
    """Map reflected bindings to the source's parameters through their origins.

    Every output must be bound. The translator binds every input the kernel
    reads; an input it does not bind is never read, so it is not uploaded.
    """
    inputs, expected, observed, matched = {}, {}, [], {}
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            continue
        name = provenance.get("sourceResource", {}).get("parameter")
        if name not in uploads or name in matched or binding["name"] in inputs:
            raise ValueError(
                "Reflected custom kernel binding does not match its source"
            )
        matched[name] = binding["name"]
        value = uploads[name]
        if not reflected_layout_matches(
            binding["scalarLayout"], value, parameters[name]
        ):
            raise ValueError(
                "Reflected custom kernel storage differs from the native buffer"
            )
        inputs[binding["name"]] = value
        if name in outputs:
            if binding.get("access") != "read_write":
                raise ValueError("Custom kernel output is not writable")
            expected[binding["name"]] = {
                key: item for key, item in value.items() if key != "values"
            }
        else:
            observed.append(binding["name"])
    if not set(outputs) <= set(matched):
        raise ValueError("Custom kernel reflection does not bind every output")
    return inputs, expected, observed, matched


def _execute(host, packages, inputs, expected, observed, execution):
    descriptor, directory = packages[0]
    if host.target == "metal":
        request = build_native_loader_dispatch_request(
            descriptor,
            directory,
            inputs,
            expected,
            execution,
            expected_target=host.target,
            observe_inputs=tuple(observed),
        )
        return execute(host, request)
    return region_dispatch.run_regions(host, packages, inputs, expected, execution)


def _readback(result, signature, supplied, expected, matched, tail, target):
    """Each output's words, with its guard tail checked and removed."""
    if result.status != "ok" or set(result.outputs) != {
        matched[item.name] for item in signature.outputs
    }:
        raise RuntimeError("Native custom kernel did not return its outputs")
    words_out = {}
    for item in signature.outputs:
        binding = matched[item.name]
        returned = result.outputs[binding]
        if any(
            returned.get(key) != expected[binding].get(key)
            for key in ("dtype", "encoding", "shape")
        ):
            raise RuntimeError("Native custom kernel output layout differs")
        words = returned.get("values")
        if not isinstance(words, list):
            raise RuntimeError("Native custom kernel output extent differs")
        words = unpack(item.dtype, target, words)
        size = supplied[item.buffer].count
        if len(words) != size + GUARD_COUNT:
            raise RuntimeError("Native custom kernel output extent differs")
        if words[size:] != tail[item.name]:
            raise RuntimeError("Native custom kernel changed its output guard")
        words_out[item.name] = words[:size]
    return words_out


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.custom_kernels is None or launch is None or not buffers:
        raise ValueError(
            "Custom kernel dispatch requires complete buffers, pinned headers and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Custom kernel")
    source = kernel_source(supplied)
    mode = layout.math_mode(supplied)
    signature = layout.parse(entry, source)
    execution = launch.execution()
    metadata = layout.validate(entry, signature, supplied, threads, execution, mode)
    target = host.target
    parameters = {item.name: item.dtype for item in signature.parameters}
    outputs = {item.name for item in signature.outputs}
    values = {
        item.name: buffer_words(supplied[item.buffer], item.dtype)
        for item in signature.parameters
        if item.kind != "output"
    }
    initial = {
        item.name: (
            buffer_words(supplied[item.buffer], item.dtype)
            if supplied[item.buffer].output == 2
            else guard_words(item.dtype, supplied[item.buffer].count)
        )
        for item in signature.outputs
    }
    tail = {item.name: guard_words(item.dtype) for item in signature.outputs}
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
    packages = host.custom_kernels.get(
        entry,
        source,
        math_mode=mode,
        thread_grid_size=metadata["threadGridSize"],
        workgroup_size=metadata["workgroupSize"],
    )
    descriptor, directory = packages[0]
    inputs, expected, observed, matched = bind(descriptor, parameters, uploads, outputs)
    result = _execute(host, packages, inputs, expected, observed, execution)
    words_out = _readback(result, signature, supplied, expected, matched, tail, target)
    payloads = {
        item.name: bytes(
            (CTYPES[item.dtype] * len(words_out[item.name]))(*words_out[item.name])
        )
        for item in signature.outputs
    }
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "customKernel": metadata,
        "sourceHash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters": [item.to_json() for item in signature.parameters],
        "boundParameters": sorted(matched),
        "operandValues": values,
        "initialValues": {name: initial[name] for name in sorted(outputs)},
        "outputValues": words_out,
        "outputHashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }
    # Validate every result and persist evidence before changing MLX outputs.
    write_trace(host, event)
    for item in signature.outputs:
        payload = payloads[item.name]
        ctypes.memmove(supplied[item.buffer].data, payload, len(payload))
    host.dispatch_count += 1
