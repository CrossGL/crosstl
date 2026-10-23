"""Dispatch contiguous unary and binary vectors through lazily translated kernels.

The native host passes ``v_<op>`` and ``vv_<op>`` launches of at most 65535
one-thread groups for every dtype MLX instantiates on the GPU. Each buffer is checked
against its kernel's source contract before upload: one element per invocation
in every vector, the reflected storage of every binding, and an output that
either has its own allocation or is exactly a donated input. The output
carries a trailing guard, and the readback is decoded exactly into MLX's
logical storage.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    INEXACT,
    buffer_words,
    guard_words,
    packet,
    reflected_layout_matches,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ENTRIES,
    UNARY,
    signature,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import write_trace

__all__ = ("ENTRIES", "dispatch", "operands", "validate")
# backend.cpp splits unary and binary vectors into launches of 65535 groups.
MAX_VECTOR = 65535


def operands(kernel):
    """The source parameters that read elements, then the one that writes."""
    return (("in",), "out") if kernel == UNARY else (("a", "b"), "c")


def _span(buffer, dtype):
    return buffer.data, buffer.data + buffer.count * ctypes.sizeof(CTYPES[dtype])


def require_defined(operation, dtype, values):
    """Refuse operands whose result Metal, HLSL and GLSL leave undefined.

    Shifts promote 8- to 32-bit operands to 32 bits, and integer or Boolean
    division and remainders need a nonzero divisor without signed overflow.
    """
    if operation in {"LeftShift", "RightShift"}:
        limit = 64 if ctypes.sizeof(CTYPES[dtype]) == 8 else 32
        if any(not 0 <= count < limit for count in values["b"]):
            raise ValueError(
                "CrossTL shifts require counts in [0, 31], or [0, 63] for 64-bit "
                "integers"
            )
    if operation in {"Divide", "Remainder"} and dtype not in INEXACT:
        minimum = (
            -(1 << (8 * ctypes.sizeof(CTYPES[dtype]) - 1))
            if dtype.startswith("int")
            else None
        )
        if any(
            divisor == 0 or (divisor == -1 and dividend == minimum)
            for dividend, divisor in zip(values["a"], values["b"])
        ):
            raise ValueError(
                "CrossTL integer divisions and remainders require nonzero divisors "
                "without signed overflow"
            )


def validate(entry, buffers, threads, execution):
    """Check a vector launch: one invocation per element of every vector."""
    kernel, operation, source, destination = signature(entry)
    inputs, output = operands(kernel)
    if type(threads) is not int or not 0 < threads <= MAX_VECTOR:
        raise ValueError("Elementwise size exceeds the host contract")
    if set(buffers) != {*inputs, output, "size"}:
        raise ValueError("Elementwise buffers do not match their kernel")
    expected = {
        **{name: (source, 0) for name in inputs},
        output: (destination, 1),
        "size": ("uint32", 0),
    }
    for name, buffer in buffers.items():
        dtype, direction = expected[name]
        size = ctypes.sizeof(CTYPES[dtype])
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.data % size
            or buffer.count != (1 if name == "size" else threads)
            or buffer.output != direction
        ):
            raise ValueError(
                "Elementwise buffer type, extent, alignment or direction differs"
            )
    if ctypes.cast(buffers["size"].data, ctypes.POINTER(ctypes.c_uint32))[0] != threads:
        raise ValueError("Elementwise size does not match the launch")
    if execution != {"workgroupCount": [threads, 1, 1], "workgroupSize": [1, 1, 1]}:
        raise ValueError("Elementwise launch does not match its vector")
    written = _span(buffers[output], destination)
    for name in inputs:
        read = _span(buffers[name], source)
        # MLX donates an input to the output only as the identical allocation.
        if read != written and max(read[0], written[0]) < min(read[1], written[1]):
            raise ValueError("Elementwise input and output allocations overlap")
    return {
        "kernel": kernel,
        "operation": operation,
        "source": source,
        "destination": destination,
    }


def _bind(entry, descriptor, supplied, uploads):
    """Map reflected bindings to the source parameters and their packets."""
    # Direct3D prefixes constants with the entry; "in" and "out" are keywords.
    prefix = entry.rstrip("_") + "_"
    inputs, outputs, observed, matched = {}, {}, [], {}
    kernel, _, source, destination = signature(entry)
    _, output = operands(kernel)
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        scalar = binding["scalarLayout"]
        name = scalar.get("memberName", binding["name"]).removeprefix(prefix)
        name = {"out_": "out", "in_": "in"}.get(name, name)
        if name not in supplied or name in matched or binding["name"] in inputs:
            raise ValueError("Reflected elementwise binding does not match its source")
        matched[name] = binding["name"]
        if name in uploads:
            value = uploads[name]
        else:
            buffer = supplied[name]
            value = {
                "dtype": "uint32",
                "shape": [buffer.count],
                "values": list(
                    ctypes.cast(
                        buffer.data, ctypes.POINTER(ctypes.c_uint32 * buffer.count)
                    ).contents
                ),
            }
        logical = (
            "uint32" if name == "size" else destination if name == output else source
        )
        if not reflected_layout_matches(scalar, value, logical):
            raise ValueError(
                "Reflected elementwise storage differs from the native buffer"
            )
        inputs[binding["name"]] = value
        if name == output:
            outputs[binding["name"]] = {k: v for k, v in value.items() if k != "values"}
        else:
            observed.append(binding["name"])
    if set(matched) != set(supplied):
        raise ValueError(
            "Elementwise reflection does not cover the complete source contract"
        )
    return inputs, outputs, observed, matched


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    kernel, _, source, destination = signature(entry)
    names, output = operands(kernel)
    if (
        host.elementwise is None
        or launch is None
        or not buffers
        or count != len(names) + 2
    ):
        raise ValueError(
            "Elementwise dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Elementwise")
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    target = host.target
    values = {name: buffer_words(supplied[name], source) for name in names}
    require_defined(metadata["operation"], source, values)
    # A fresh output is filled with guards, so the readback is never stale.
    initial = guard_words(destination, threads)
    tail = guard_words(destination)
    uploads = {
        **{name: packet(source, target, words) for name, words in values.items()},
        output: packet(destination, target, initial + tail),
    }
    descriptor, directory = host.elementwise.get(entry)
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
    if result.status != "ok" or set(result.outputs) != {matched[output]}:
        raise RuntimeError("Native elementwise kernel did not return its output")
    returned = result.outputs[matched[output]]
    expected = outputs[matched[output]]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native elementwise output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native elementwise output extent differs")
    words = unpack(destination, target, words)
    if len(words) != threads + GUARD_COUNT:
        raise RuntimeError("Native elementwise output extent differs")
    if words[threads:] != tail:
        raise RuntimeError("Native elementwise kernel changed its output guard")
    words = words[:threads]
    payload = bytes((CTYPES[destination] * threads)(*words))
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "elementwiseMetadata": metadata,
        "inputs": inputs,
        "operandValues": values,
        "initialValues": initial,
        "outputValues": words,
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    write_trace(host, event)
    ctypes.memmove(supplied[output].data, payload, len(payload))
    host.dispatch_count += 1
