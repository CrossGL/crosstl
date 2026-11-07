"""Dispatch contiguous unary, binary and ternary vectors through lazy kernels.

The native host passes ``v_<op>``, ``vv_<op>``, ``sv_<op>``, ``vs_<op>`` and
``v_Select`` launches of at most 65535 one-thread groups for every dtype MLX
instantiates on the GPU; ``sv_`` and ``vs_`` read their first or second operand
from one scalar element. A vector of more than 65535 elements runs through the
wide ``v2`` kernels instead, over the exact 2-D grid ``wide_vectors`` derives
from its int64 ``size``. Each buffer is checked
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
    is_wide,
    operand_types,
    signature,
    work_per_thread,
)
from demos.integrations.mlx.portable_host.gather_dispatch import execute
from demos.integrations.mlx.portable_host.trace_records import (
    compact_storage,
    compact_trace_values,
    retains_values,
    write_trace,
)
from demos.integrations.mlx.portable_host.wide_vectors import grid

__all__ = (
    "ENTRIES",
    "bind",
    "broadcast",
    "dispatch",
    "element_count",
    "extents",
    "operands",
    "validate",
    "wide_launch",
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


def wide_launch(entry, elements):
    """The exact launch of wide ``entry`` over ``elements`` elements."""
    columns, rows = grid(elements, work_per_thread(entry))
    return {"workgroupCount": [columns, rows, 1], "workgroupSize": [1, 1, 1]}


def element_count(metadata, threads):
    """The elements a validated launch covers: a wide launch's size, else ``threads``."""
    return metadata.get("elements", threads)


def _launch_elements(entry, size, threads):
    """The elements of a launch and its exact geometry, from its ``size`` buffer.

    A narrow launch runs one invocation per element and passes a ``uint32``
    size; a wide one passes an ``int64`` size and runs every invocation of
    the grid ``wide_launch`` derives from it.
    """
    wide = is_wide(entry)
    size_type = "int64" if wide else "uint32"
    if (
        size.dtype != size_type.encode("ascii")
        or not size.data
        or size.data % ctypes.sizeof(CTYPES[size_type])
        or size.count != 1
        or size.output != 0
    ):
        raise ValueError(
            "Elementwise buffer type, extent, alignment or direction differs"
        )
    elements = ctypes.cast(size.data, ctypes.POINTER(CTYPES[size_type]))[0]
    if not wide:
        if elements != threads:
            raise ValueError("Elementwise size does not match the launch")
        return elements, {"workgroupCount": [threads, 1, 1], "workgroupSize": [1, 1, 1]}
    launch = wide_launch(entry, elements)
    columns, rows, _ = launch["workgroupCount"]
    if threads != columns * rows:
        raise ValueError("Elementwise size does not match the launch")
    return elements, launch


def validate(entry, buffers, threads, execution):
    """Check a vector launch: every element of every vector, exactly once.

    A narrow launch covers at most 65535 elements, one per invocation; a wide
    one covers its ``size`` over the grid ``wide_launch`` derives from it.
    """
    kernel, operation, source, destination = signature(entry)
    inputs, output = operands(kernel)
    types = operand_types(entry)
    limit = MAX_VECTOR if not is_wide(entry) else None
    if type(threads) is not int or threads < 1 or (limit and threads > limit):
        raise ValueError("Elementwise size exceeds the host contract")
    if set(buffers) != {*inputs, output, "size"}:
        raise ValueError("Elementwise buffers do not match their kernel")
    elements, launch = _launch_elements(entry, buffers["size"], threads)
    counts = extents(kernel, elements)
    expected = {
        **{name: (types[name], 0) for name in inputs},
        output: (destination, 1),
    }
    for name, (dtype, direction) in expected.items():
        buffer = buffers[name]
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
    if execution != launch:
        raise ValueError("Elementwise launch does not match its vector")
    written = _span(buffers[output], destination)
    for name in inputs:
        read = _span(buffers[name], types[name])
        # MLX donates an input to the output only as the identical allocation.
        if read != written and max(read[0], written[0]) < min(read[1], written[1]):
            raise ValueError("Elementwise input and output allocations overlap")
    metadata = {
        "kernel": kernel,
        "operation": operation,
        "source": source,
        "destination": destination,
    }
    if is_wide(entry):
        metadata.update(elements=elements, workPerThread=work_per_thread(entry))
    return metadata


def bind(entry, descriptor, supplied, uploads, output, types):
    """Map reflected bindings to the source parameters and their packets.

    ``types`` names each parameter's logical element type; ``size`` is the
    vectors' element count, ``uint32`` for a narrow launch and ``int64`` for
    a wide one.
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
            kind = "int64" if name == "size" and buffer.dtype == b"int64" else "uint32"
            value = {
                "dtype": kind,
                "shape": [buffer.count],
                "values": list(
                    ctypes.cast(
                        buffer.data, ctypes.POINTER(CTYPES[kind] * buffer.count)
                    ).contents
                ),
            }
        logical = value["dtype"] if name == "size" else types[name]
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
    elements = element_count(metadata, threads)
    storage = {name: buffer_storage(supplied[name], types[name]) for name in names}
    if operation in CHECKED_OPERATIONS:
        words = {
            name: memoryview(storage[name]).cast(STORAGE_CODES[types[name]]).tolist()
            for name in names
        }
        require_defined(operation, source, broadcast(kernel, words, elements))
    size = ctypes.sizeof(CTYPES[destination])
    initial = storage_bytes(destination, guard_words(destination, elements))
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
    if type(data) is not bytes or len(data) != (elements + GUARD_COUNT) * size:
        raise RuntimeError("Native elementwise output extent differs")
    if data[elements * size :] != tail:
        raise RuntimeError("Native elementwise kernel changed its output guard")
    payload = data[: elements * size]
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
    elements = element_count(metadata, threads)
    values = {name: buffer_words(supplied[name], types[name]) for name in names}
    require_defined(metadata["operation"], source, broadcast(kernel, values, elements))
    # A fresh output is filled with guards, so the readback is never stale.
    initial = guard_words(destination, elements)
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
    if len(words) != elements + GUARD_COUNT:
        raise RuntimeError("Native elementwise output extent differs")
    if words[elements:] != tail:
        raise RuntimeError("Native elementwise kernel changed its output guard")
    words = words[:elements]
    payload = native_bytes(CTYPES[destination], elements, words)
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
