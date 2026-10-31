"""Dispatch lazily translated softmax and logsumexp row kernels.

The native host passes upstream's three buffers: the last-axis-contiguous
input, the output and ``int axis_size``. ``softmax_layout`` checks them and
the launch against ``Softmax::eval_gpu`` or ``LogSumExp::eval_gpu`` before
any package is translated. The kernels size their work by the launch, so
each width is its own package. Storage travels as MLX's exact words through
``element_storage``; the output carries a trailing guard that must survive.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import softmax_layout
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
from demos.integrations.mlx.portable_host.wide_column_dispatch import (
    refuse_subnormal_bfloat,
)

ENTRIES = softmax_layout.ENTRIES
INPUT, OUTPUT = softmax_layout.INPUT, softmax_layout.OUTPUT
AXIS_SIZE = softmax_layout.AXIS_SIZE
BUFFER_COUNT = 3


def uploads(dtype, target, values, outputs, axis_size):
    """The packets one launch binds: the input, a guarded output and the row length."""
    return {
        INPUT: packet(dtype, target, values),
        OUTPUT: packet(dtype, target, guard_words(dtype, outputs) + guard_words(dtype)),
        AXIS_SIZE: packet("int32", target, [axis_size]),
    }


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.softmaxes is None or launch is None or not buffers or count != BUFFER_COUNT:
        raise ValueError(
            "Softmax and logsumexp dispatch requires complete buffers, pinned "
            "sources and launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Softmax or logsumexp")
    execution = launch.execution()
    axis_size = softmax_layout.validate(entry, supplied, threads, execution)
    _, _, dtype, _ = softmax_layout.signature(entry)
    target = host.target
    values = buffer_words(supplied[INPUT], dtype)
    outputs = supplied[OUTPUT].count
    refuse_subnormal_bfloat(dtype, target, values)
    packets = uploads(dtype, target, values, outputs, axis_size)
    descriptor, directory = host.softmaxes.get(
        softmax_layout.variant(entry, execution["workgroupSize"][0])
    )
    inputs, native_outputs, observed, matched = bind(
        entry,
        descriptor,
        supplied,
        packets,
        OUTPUT,
        softmax_layout.operand_types(entry),
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
    binding = matched[OUTPUT]
    if result.status != "ok" or set(result.outputs) != {binding}:
        raise RuntimeError("Native softmax or logsumexp did not return its output")
    returned, expected = result.outputs[binding], native_outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native softmax or logsumexp output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native softmax or logsumexp output extent differs")
    words = unpack(dtype, target, words)
    if len(words) != outputs + GUARD_COUNT:
        raise RuntimeError("Native softmax or logsumexp output extent differs")
    if words[outputs:] != guard_words(dtype):
        raise RuntimeError("Native softmax or logsumexp changed its output guard")
    words = words[:outputs]
    payload = bytes((CTYPES[dtype] * outputs)(*words))
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
            "axisSize": axis_size,
            "inputs": inputs,
            "operandValues": values,
            "initialValues": guard_words(dtype, outputs),
            "outputValues": words,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
