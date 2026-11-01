"""Dispatch lazily translated MLX fused cross-entropy kernels.

The native host passes every buffer ``CrossEntropy`` or
``CrossEntropyVJP::eval_gpu`` binds: the row-contiguous logits, their int32
class indices, the losses and cotangents of a gradient, the output and
``int axis_size``. ``cross_entropy_layout`` checks them, the class indices
and the launch against cross_entropy.cpp before any package is translated.
Storage travels as MLX's exact words through ``element_storage``; the output
carries a trailing guard that must survive.

A launch over more than ``RETAINED_LIMIT`` logits keeps only the length and
digest of its long value lists in the trace.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import cross_entropy_layout as layout
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
PREFIX = layout.PREFIX
RETAINED_LIMIT = 1 << 16
LAYOUT_KEYS = ("dtype", "encoding", "shape")


def uploads(entry, target, values, outputs, axis_size):
    """The packets one launch binds.

    ``values`` holds every read element buffer's words and ``outputs`` the
    output's extent; the output starts as guards with a trailing guard.
    """
    types = layout.operand_types(entry)
    packets = {name: packet(types[name], target, values[name]) for name in values}
    kind = types[layout.OUTPUT]
    packets[layout.OUTPUT] = packet(
        kind, target, guard_words(kind, outputs) + guard_words(kind)
    )
    packets[layout.AXIS_SIZE] = packet("int32", target, [axis_size])
    return packets


def output_hash(entry, words):
    """The digest of the output's exact storage."""
    kind = layout.operand_types(entry)[layout.OUTPUT]
    return hashlib.sha256(bytes((CTYPES[kind] * len(words))(*words))).hexdigest()


def _readback(result, native_outputs, binding, kind, outputs, target):
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError("Native cross-entropy did not return its output")
    returned, expected = result.outputs[binding], native_outputs[binding]
    if any(returned.get(key) != expected.get(key) for key in LAYOUT_KEYS):
        raise RuntimeError("Native cross-entropy output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native cross-entropy output extent differs")
    words = unpack(kind, target, words)
    if len(words) != outputs + GUARD_COUNT:
        raise RuntimeError("Native cross-entropy output extent differs")
    if words[outputs:] != guard_words(kind):
        raise RuntimeError("Native cross-entropy changed its output guard")
    return words[:outputs]


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    types = layout.operand_types(entry)
    if (
        host.cross_entropies is None
        or launch is None
        or not buffers
        or count != len(types)
    ):
        raise ValueError(
            "Cross-entropy dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Cross-entropy")
    execution = launch.execution()
    axis_size = layout.validate(entry, supplied, threads, execution)
    target = host.target
    values = {
        name: buffer_words(supplied[name], types[name])
        for name in layout.inputs_of(entry)
    }
    layout.require_targets(values[layout.TARGETS], axis_size)
    refuse_subnormal_bfloat(types[layout.INPUT], target, values[layout.INPUT])
    outputs = supplied[layout.OUTPUT].count
    packets = uploads(entry, target, values, outputs, axis_size)
    descriptor, directory = host.cross_entropies.get(
        layout.variant(entry, execution["workgroupSize"][0])
    )
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
    kind = types[layout.OUTPUT]
    words = _readback(
        result, native_outputs, matched[layout.OUTPUT], kind, outputs, target
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
        "axisSize": axis_size,
        "inputs": inputs,
        "operandValues": values,
        "outputValues": words,
        "outputHash": output_hash(entry, words),
    }
    if threads > RETAINED_LIMIT:
        event = compact_trace_values(event)
    write_trace(host, event)
    payload = bytes((CTYPES[kind] * outputs)(*words))
    ctypes.memmove(supplied[layout.OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
