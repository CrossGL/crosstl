"""Dispatch contiguous unary, binary and ternary vectors through lazy kernels.

The native host passes ``v_<op>``, ``vv_<op>``, ``sv_<op>``, ``vs_<op>`` and
``v_Select`` launches of at most 65535 one-thread groups for every dtype MLX
instantiates on the GPU; ``sv_`` and ``vs_`` read their first or second operand
from one scalar element. Each buffer is checked
against its kernel's source contract before upload: one element per invocation
in every vector, the reflected storage of every binding, and an output that
either has its own allocation or is exactly a donated input. The output
carries a trailing guard, and the readback is decoded exactly into MLX's
logical storage.

A dispatch whose records keep no values, and whose every operand the target
carries as MLX's own storage bytes, uploads and reads back those bytes as
packed storage. Its record is the compact record the word-list path writes.
"""

import ctypes
import hashlib
import sys

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    GUARD_COUNT,
    INEXACT,
    STORAGE_CODES,
    buffer_storage,
    buffer_words,
    guard_words,
    native_bytes,
    packed_layout,
    packed_packet,
    packet,
    packet_code,
    reflected_layout_matches,
    storage_bytes,
    supplied_buffers,
    unpack,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    BINARY_KERNELS,
    ENTRIES,
    SCALAR_VECTOR,
    UNARY,
    VECTOR_SCALAR,
    operand_types,
    signature,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import (
    compact_storage,
    compact_trace_values,
    retains_values,
    write_trace,
)

__all__ = (
    "ENTRIES",
    "bind",
    "broadcast",
    "dispatch",
    "extents",
    "operands",
    "validate",
)
# backend.cpp splits unary and binary vectors into launches of 65535 groups.
MAX_VECTOR = 65535
# Operations whose operands ``require_defined`` reads.
CHECKED_OPERATIONS = frozenset({"LeftShift", "RightShift", "Divide", "Remainder"})


def operands(kernel):
    """The source parameters that read elements, then the one that writes."""
    if kernel == UNARY:
        return ("in",), "out"
    return (("a", "b"), "c") if kernel in BINARY_KERNELS else (("a", "b", "c"), "d")


# The operand binary_sv and binary_vs read as a[0] or b[0].
SCALARS = {SCALAR_VECTOR: "a", VECTOR_SCALAR: "b"}


def extents(kernel, threads):
    """The element count of every operand and the output of one launch."""
    names, output = operands(kernel)
    scalar = SCALARS.get(kernel)
    return {
        **{name: 1 if name == scalar else threads for name in names},
        output: threads,
    }


def broadcast(kernel, values, threads):
    """The operand words each invocation reads, a scalar repeated ``threads`` times."""
    scalar = SCALARS.get(kernel)
    return {
        name: words * threads if name == scalar else words
        for name, words in values.items()
    }


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
    types = operand_types(entry)
    if type(threads) is not int or not 0 < threads <= MAX_VECTOR:
        raise ValueError("Elementwise size exceeds the host contract")
    counts = {**extents(kernel, threads), "size": 1}
    if set(buffers) != {*inputs, output, "size"}:
        raise ValueError("Elementwise buffers do not match their kernel")
    expected = {
        **{name: (types[name], 0) for name in inputs},
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
            or buffer.count != counts[name]
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
        read = _span(buffers[name], types[name])
        # MLX donates an input to the output only as the identical allocation.
        if read != written and max(read[0], written[0]) < min(read[1], written[1]):
            raise ValueError("Elementwise input and output allocations overlap")
    return {
        "kernel": kernel,
        "operation": operation,
        "source": source,
        "destination": destination,
    }


def bind(entry, descriptor, supplied, uploads, output, types):
    """Map reflected bindings to the source parameters and their packets.

    ``types`` names each parameter's logical element type; ``size`` is the
    vectors' ``uint32`` element count.
    """
    # Direct3D prefixes constants with the entry; "in" and "out" are keywords.
    prefix = entry.rstrip("_") + "_"
    inputs, outputs, observed, matched = {}, {}, [], {}
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
        logical = "uint32" if name == "size" else types[name]
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


def packed_transport(host, entry):
    """Whether ``entry`` uploads and reads back its storage as packed bytes.

    Records that keep their values go through word lists, which the audits
    recompute; so does any operand the target widens or re-encodes.
    """
    if sys.byteorder != "little" or retains_values(host, entry):
        return False
    _, _, _, destination = signature(entry)
    return all(
        packed_layout(dtype, host.target) is not None
        for dtype in (*operand_types(entry).values(), destination)
    )


def _traced_packet(value):
    """A packet as the compact record of its word-list packet holds it."""
    if type(value.get("values")) is not bytes:
        return value
    return {**value, "values": compact_storage(value["values"], packet_code(value))}


def _dispatch_packed(host, entry, supplied, threads, execution, metadata):
    """``dispatch`` through packed storage; see ``packed_transport``."""
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    kernel, operation, source, destination = signature(entry)
    names, output = operands(kernel)
    target = host.target
    types = operand_types(entry)
    storage = {name: buffer_storage(supplied[name], types[name]) for name in names}
    if operation in CHECKED_OPERATIONS:
        words = {
            name: memoryview(storage[name]).cast(STORAGE_CODES[types[name]]).tolist()
            for name in names
        }
        require_defined(operation, source, broadcast(kernel, words, threads))
    size = ctypes.sizeof(CTYPES[destination])
    initial = storage_bytes(destination, guard_words(destination, threads))
    tail = storage_bytes(destination, guard_words(destination))
    uploads = {
        **{
            name: packed_packet(types[name], target, stored)
            for name, stored in storage.items()
        },
        output: packed_packet(destination, target, initial + tail),
    }
    descriptor, directory = host.elementwise.get(entry)
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, uploads, output, types
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
    data = _returned(result, outputs, matched[output])
    if type(data) is not bytes or len(data) != (threads + GUARD_COUNT) * size:
        raise RuntimeError("Native elementwise output extent differs")
    if data[threads * size :] != tail:
        raise RuntimeError("Native elementwise kernel changed its output guard")
    payload = data[: threads * size]
    if destination == "bool_" and payload.translate(None, b"\x00\x01"):
        raise ValueError("Native readback returned a non-canonical Boolean")
    code = STORAGE_CODES[destination]
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
        "inputs": {name: _traced_packet(value) for name, value in inputs.items()},
        "operandValues": {
            name: compact_storage(stored, STORAGE_CODES[types[name]])
            for name, stored in storage.items()
        },
        "initialValues": compact_storage(initial, code),
        "outputValues": compact_storage(payload, code),
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }
    write_trace(host, compact_trace_values(event), prepared=True)
    ctypes.memmove(supplied[output].data, payload, len(payload))
    host.dispatch_count += 1


def _returned(result, outputs, name):
    """The readback values of output ``name``, after checking its layout."""
    if result.status != "ok" or set(result.outputs) != {name}:
        raise RuntimeError("Native elementwise kernel did not return its output")
    returned = result.outputs[name]
    expected = outputs[name]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native elementwise output layout differs")
    return returned.get("values")


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
    if packed_transport(host, entry):
        return _dispatch_packed(host, entry, supplied, threads, execution, metadata)
    target = host.target
    types = operand_types(entry)
    values = {name: buffer_words(supplied[name], types[name]) for name in names}
    require_defined(metadata["operation"], source, broadcast(kernel, values, threads))
    # A fresh output is filled with guards, so the readback is never stale.
    initial = guard_words(destination, threads)
    tail = guard_words(destination)
    uploads = {
        **{name: packet(types[name], target, words) for name, words in values.items()},
        output: packet(destination, target, initial + tail),
    }
    descriptor, directory = host.elementwise.get(entry)
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, uploads, output, types
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
    words = _returned(result, outputs, matched[output])
    if not isinstance(words, list):
        raise RuntimeError("Native elementwise output extent differs")
    words = unpack(destination, target, words)
    if len(words) != threads + GUARD_COUNT:
        raise RuntimeError("Native elementwise output extent differs")
    if words[threads:] != tail:
        raise RuntimeError("Native elementwise kernel changed its output guard")
    words = words[:threads]
    payload = native_bytes(CTYPES[destination], threads, words)
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
