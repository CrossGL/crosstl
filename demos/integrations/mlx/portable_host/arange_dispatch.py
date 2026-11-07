"""Dispatch lazily translated ``arange`` kernels.

The native host passes ``start`` and ``step`` as one-element constants of the
output type and ``out`` as one element per invocation, in a launch of at most
65535 one-thread groups. The output carries a trailing guard and the readback
is decoded exactly into MLX's storage.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host.arange_packages import signature
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    buffer_words,
    guard_words,
    packet,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.elementwise_dispatch import bind
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import write_trace

# backend.cpp refuses an arange of more than 65535 elements.
MAX_VECTOR = 65535
SCALARS = ("start", "step")
OUTPUT = "out"


def operand_types(entry):
    """Every source parameter's element type; all three are the output type."""
    dtype = signature(entry)
    return {name: dtype for name in (*SCALARS, OUTPUT)}


def validate(entry, buffers, threads, execution):
    """Check one launch: two scalar constants and one element per invocation."""
    dtype = signature(entry)
    if type(threads) is not int or not 0 < threads <= MAX_VECTOR:
        raise ValueError("Arange size exceeds the host contract")
    if set(buffers) != {*SCALARS, OUTPUT}:
        raise ValueError("Arange buffers do not match their kernel")
    size = ctypes.sizeof(CTYPES[dtype])
    for name, buffer in buffers.items():
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % size
            or buffer.count != (threads if name == OUTPUT else 1)
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Arange buffer type, extent, alignment or direction differs"
            )
    if execution != {"workgroupCount": [threads, 1, 1], "workgroupSize": [1, 1, 1]}:
        raise ValueError("Arange launch does not match its output")
    return {"dtype": dtype}


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.aranges is None or launch is None or not buffers or count != 3:
        raise ValueError(
            "Arange dispatch requires complete buffers, pinned sources and launch "
            "geometry"
        )
    supplied = supplied_buffers(buffers, count, "Arange")
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    dtype, target = metadata["dtype"], host.target
    values = {name: buffer_words(supplied[name], dtype) for name in SCALARS}
    # A fresh output is filled with guards, so the readback is never stale.
    initial = guard_words(dtype, threads)
    tail = guard_words(dtype)
    uploads = {
        **{name: packet(dtype, target, words) for name, words in values.items()},
        OUTPUT: packet(dtype, target, initial + tail),
    }
    descriptor, directory = host.aranges.get(entry)
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, uploads, OUTPUT, operand_types(entry)
    )
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
    binding = matched[OUTPUT]
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError("Native arange kernel did not return its output")
    returned, expected = result.outputs[binding], outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native arange output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native arange output extent differs")
    words = unpack(dtype, target, words)
    if len(words) != threads + GUARD_COUNT:
        raise RuntimeError("Native arange output extent differs")
    if words[threads:] != tail:
        raise RuntimeError("Native arange kernel changed its output guard")
    words = words[:threads]
    payload = bytes((CTYPES[dtype] * threads)(*words))
    write_trace(
        host,
        {
            "entry": entry,
            "target": target,
            "threads": threads,
            **execution,
            "dispatchVersion": DISPATCH_VERSION,
            "artifact": descriptor["artifact"],
            "packageRoot": str(directory),
            "details": result.details,
            "arangeMetadata": metadata,
            "inputs": inputs,
            "operandValues": values,
            "initialValues": initial,
            "outputValues": words,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
