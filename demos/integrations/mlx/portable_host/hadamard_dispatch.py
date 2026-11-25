"""Dispatch lazily translated MLX Hadamard transform stages.

The native host passes each slice of one ``Hadamard::eval_gpu`` stage with
the buffers ``hadamard_mn_contiguous`` binds: ``in`` at ``buffer(0)``,
``out`` at ``buffer(1)`` and the stage's float ``scale``. A stage after the
first reads and writes the same array, so ``in`` and ``out`` may alias:
every input word is read before the output is written back.
``hadamard_layout`` checks the slice and its launch before any package is
translated. Storage travels as MLX's exact words through
``element_storage``; the output starts as guards, with a trailing guard that
must survive.

A slice of more than ``RETAINED_LIMIT`` elements keeps only the length and
digest of its long value lists in the trace.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import hadamard_layout as layout
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
from demos.integrations.mlx.portable_host.trace_records import (
    compact_trace_values,
    write_trace,
)
from demos.integrations.mlx.portable_host.wide_column_dispatch import (
    refuse_subnormal_bfloat,
)

ENTRIES = layout.ENTRIES
RETAINED_LIMIT = 1 << 16
LAYOUT_KEYS = ("dtype", "encoding", "shape")


def uploads(entry, target, values, scale):
    """The packets one slice binds; ``scale`` is a float32 storage word and
    the output starts as guards."""
    kind = layout.signature(entry)[2]
    count = len(values)
    return {
        layout.INPUT: packet(kind, target, values),
        layout.OUTPUT: packet(
            kind, target, guard_words(kind, count) + guard_words(kind)
        ),
        layout.SCALE: packet("float32", target, [scale]),
    }


def output_hash(entry, words):
    """The digest of the output's exact storage."""
    kind = layout.signature(entry)[2]
    return hashlib.sha256(bytes((CTYPES[kind] * len(words))(*words))).hexdigest()


def _readback(result, native_outputs, binding, kind, count, target):
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError("Native Hadamard stage did not return its output")
    returned, expected = result.outputs[binding], native_outputs[binding]
    if any(returned.get(key) != expected.get(key) for key in LAYOUT_KEYS):
        raise RuntimeError("Native Hadamard output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native Hadamard output extent differs")
    words = unpack(kind, target, words)
    if len(words) != count + GUARD_COUNT:
        raise RuntimeError("Native Hadamard output extent differs")
    if words[count:] != guard_words(kind):
        raise RuntimeError("Native Hadamard stage changed its output guard")
    return words[:count]


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    types = layout.operand_types(entry)
    if host.hadamards is None or launch is None or not buffers or count != len(types):
        raise ValueError(
            "Hadamard dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Hadamard")
    execution = launch.execution()
    size, scale = layout.validate(entry, supplied, threads, execution)
    target = host.target
    kind = types[layout.INPUT]
    values = buffer_words(supplied[layout.INPUT], kind)
    refuse_subnormal_bfloat(kind, target, values)
    packets = uploads(entry, target, values, scale)
    descriptor, directory = host.hadamards.get(entry)
    inputs, native_outputs, observed, matched = bind(
        entry, descriptor, supplied, packets, layout.OUTPUT, types
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        directory,
        inputs,
        native_outputs,
        execution,
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    result = execute(host, request)
    words = _readback(
        result, native_outputs, matched[layout.OUTPUT], kind, size, target
    )
    event = {
        "entry": entry,
        "target": target,
        "threads": threads,
        **execution,
        "dispatchVersion": DISPATCH_VERSION,
        "artifact": descriptor["artifact"],
        "packageRoot": str(directory),
        "details": result.details,
        "scaleWord": scale,
        "inputs": inputs,
        "operandValues": {layout.INPUT: values},
        "outputValues": words,
        "outputHash": output_hash(entry, words),
    }
    if size > RETAINED_LIMIT:
        event = compact_trace_values(event)
    write_trace(host, event)
    payload = bytes((CTYPES[kind] * size)(*words))
    ctypes.memmove(supplied[layout.OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
