"""Dispatch checked casts and general copies through lazily translated kernels.

The native host passes ``v_copy`` conversions of contiguous vectors and
``ggn2_dynamic_copy`` copies between strided layouts for every GPU dtype,
including the narrow integers, the 16-bit float pairs and ``complex64`` that
no prebuilt family covers. Every buffer is checked against its kernel's source contract
before upload, the destination carries a trailing guard, elements a copy does
not write must keep their uploaded words, and the readback is decoded exactly
into MLX's logical storage.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import copy_layout
from demos.integrations.mlx.portable_host.copy_packages import (
    ENTRY_PREFIXES,
    VECTOR,
    signature,
)
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

__all__ = ("ENTRY_PREFIXES", "dispatch", "validate_vector")
# backend.cpp splits casts into launches of at most 65535 one-thread groups.
MAX_VECTOR = 65535
METADATA_CTYPES = {
    "uint32": ctypes.c_uint32,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}


def _overlap(first, second, first_size, second_size):
    return max(first.data, second.data) < min(
        first.data + first.count * first_size,
        second.data + second.count * second_size,
    )


def validate_vector(buffers, threads, execution, source, destination):
    """Check a ``v_copy`` launch: one invocation per element of both vectors."""
    if type(threads) is not int or not 0 < threads <= MAX_VECTOR:
        raise ValueError("Cast size exceeds the host contract")
    if set(buffers) != {"src", "dst", "size"}:
        raise ValueError("Cast buffers do not match their kernel")
    expected = {"src": (source, 0), "dst": (destination, 1), "size": ("uint32", 0)}
    for name, buffer in buffers.items():
        dtype, output = expected[name]
        size = ctypes.sizeof(CTYPES.get(dtype) or METADATA_CTYPES[dtype])
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % size
            or buffer.count != (1 if name == "size" else threads)
            or buffer.output != output
        ):
            raise ValueError("Cast buffer type, extent, alignment or direction differs")
    if ctypes.cast(buffers["size"].data, ctypes.POINTER(ctypes.c_uint32))[0] != threads:
        raise ValueError("Cast size does not match the launch")
    if execution != {"workgroupCount": [threads, 1, 1], "workgroupSize": [1, 1, 1]}:
        raise ValueError("Cast launch does not match its vector")
    if _overlap(
        buffers["src"],
        buffers["dst"],
        ctypes.sizeof(CTYPES[source]),
        ctypes.sizeof(CTYPES[destination]),
    ):
        raise ValueError("Cast source and destination allocations overlap")
    return {"kernel": VECTOR, "source": source, "destination": destination}


def validate(entry, buffers, threads, execution):
    kernel, source, destination = signature(entry)
    if kernel == VECTOR:
        return validate_vector(buffers, threads, execution, source, destination)
    if set(buffers) != set(copy_layout.DTYPES):
        raise ValueError("Copy buffers do not match their kernel")
    for name, buffer in buffers.items():
        directions = {1, copy_layout.INOUT} if name == "dst" else {0}
        if not buffer.data or buffer.output not in directions:
            raise ValueError("Copy buffer direction differs")
        size = ctypes.sizeof(
            CTYPES[source]
            if name in {"src", "dst"}
            else METADATA_CTYPES[copy_layout.DTYPES[name]]
        )
        if buffer.data % size:
            raise ValueError("Copy buffer alignment differs")
    metadata = copy_layout.validate(buffers, threads, dtype=source)
    if execution != {
        "workgroupCount": metadata["workgroupCount"],
        "workgroupSize": [1, 1, 1],
    }:
        raise ValueError("Copy launch does not match its layout")
    return {"kernel": kernel, "source": source, "destination": destination, **metadata}


def _bind(entry, descriptor, supplied, uploads):
    """Map reflected bindings to the source parameters and their packets."""
    prefix = entry + "_"
    _, source, destination = signature(entry)
    inputs, outputs, observed, matched = {}, {}, [], {}
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        scalar = binding["scalarLayout"]
        name = scalar.get("memberName", binding["name"]).removeprefix(prefix)
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected copy binding does not match its source")
        matched[name] = binding["name"]
        if name in uploads:
            value = uploads[name]
        else:
            buffer = supplied[name]
            kind = buffer.dtype.decode("ascii")
            value = {
                "dtype": kind,
                "shape": [buffer.count],
                "values": list(
                    ctypes.cast(
                        buffer.data,
                        ctypes.POINTER(METADATA_CTYPES[kind] * buffer.count),
                    ).contents
                ),
            }
        logical = {"src": source, "dst": destination}.get(name)
        logical = logical or supplied[name].dtype.decode("ascii")
        if not reflected_layout_matches(scalar, value, logical):
            raise ValueError("Reflected copy storage differs from the native buffer")
        inputs[binding["name"]] = value
        if name == "dst":
            outputs[binding["name"]] = {k: v for k, v in value.items() if k != "values"}
        else:
            observed.append(binding["name"])
    if set(matched) != set(supplied):
        raise ValueError("Copy reflection does not cover the complete source contract")
    return inputs, outputs, observed, matched


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    kernel, source, destination = signature(entry)
    if host.copies is None or launch is None or not buffers or count not in {3, 8}:
        raise ValueError(
            "Copy dispatch requires complete buffers, pinned sources and launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Copy")
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    target = host.target
    source_words = buffer_words(supplied["src"], source)
    dst = supplied["dst"]
    preserve = dst.output == copy_layout.INOUT
    # A fresh destination is filled with guards, so a skipped write is visible.
    initial = (
        buffer_words(dst, destination)
        if preserve
        else guard_words(destination, dst.count)
    )
    tail = guard_words(destination)
    uploads = {
        "src": packet(source, target, source_words),
        "dst": packet(destination, target, initial + tail),
    }
    descriptor, directory = host.copies.get(entry)
    inputs, outputs, observed, matched = _bind(entry, descriptor, supplied, uploads)
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        outputs,
        execution,
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    if result.status != "ok" or set(result.outputs) != {matched["dst"]}:
        raise RuntimeError("Native copy did not return its destination")
    output = result.outputs[matched["dst"]]
    expected = outputs[matched["dst"]]
    if any(
        output.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native copy output layout differs")
    words = output.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native copy output extent differs")
    words = unpack(destination, target, words)
    if len(words) != dst.count + GUARD_COUNT:
        raise RuntimeError("Native copy output extent differs")
    if words[dst.count :] != tail:
        raise RuntimeError("Native copy changed its output guard")
    words = words[: dst.count]
    written = (
        range(threads)
        if kernel == VECTOR
        else set(copy_layout.destination_indices(metadata))
    )
    if len(written) != dst.count and any(
        word != initial[index]
        for index, word in enumerate(words)
        if index not in written
    ):
        raise RuntimeError("Native copy changed untouched destination storage")
    payload = bytes((CTYPES[destination] * dst.count)(*words))
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "copyMetadata": metadata,
        "inputs": inputs,
        "sourceValues": source_words,
        "initialValues": initial,
        "outputValues": words,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    write_trace(host, event)
    ctypes.memmove(dst.data, payload, len(payload))
    host.dispatch_count += 1
