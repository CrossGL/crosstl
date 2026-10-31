"""Dispatch lazily translated ``arg_reduce_general`` kernels.

The native host passes upstream's eight buffers: the input view, a ``uint32``
index per output row, and the shape, strides, rank and reduced axis
``ArgReduce::eval_gpu`` binds. ``arg_reduce_layout`` checks them and the
launch against upstream's plan before any package is translated. The kernel
reads its threadgroup size, so each width is its own package. Storage travels
as MLX's exact words through ``element_storage``; the index output carries a
trailing guard and is decoded exactly.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import arg_reduce_layout
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

ENTRIES = arg_reduce_layout.ENTRIES
INPUT, OUTPUT = arg_reduce_layout.INPUT, arg_reduce_layout.OUTPUT
OUTPUT_TYPE = arg_reduce_layout.OUTPUT_TYPE
METADATA = arg_reduce_layout.METADATA
BUFFER_COUNT = len(METADATA) + 2


def uploads(dtype, target, values, outputs, metadata):
    """The packets one launch binds: the input, a guarded output and metadata."""
    return {
        INPUT: packet(dtype, target, values),
        OUTPUT: packet(
            OUTPUT_TYPE,
            target,
            guard_words(OUTPUT_TYPE, outputs) + guard_words(OUTPUT_TYPE),
        ),
        **{
            name: packet(kind, target, metadata[name])
            for name, kind in METADATA.items()
        },
    }


def require_target_width(target, execution):
    """Refuse a threadgroup width ``target``'s subgroup lowering cannot run."""
    limit = arg_reduce_layout.TARGET_WIDTH_LIMITS.get(target)
    if limit is not None and execution["workgroupSize"][0] > limit:
        raise ValueError(
            f"{target} arg reductions of more than one simdgroup are not supported "
            "yet; see CrossGL/crosstl#2207"
        )


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if (
        host.arg_reductions is None
        or launch is None
        or not buffers
        or count != BUFFER_COUNT
    ):
        raise ValueError(
            "Arg reduction dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Arg reduction")
    execution = launch.execution()
    metadata = arg_reduce_layout.validate(entry, supplied, threads, execution)
    _, dtype = arg_reduce_layout.signature(entry)
    target = host.target
    require_target_width(target, execution)
    values = buffer_words(supplied[INPUT], dtype)
    outputs = supplied[OUTPUT].count
    refuse_subnormal_bfloat(dtype, target, values)
    packets = uploads(dtype, target, values, outputs, metadata)
    descriptor, directory = host.arg_reductions.get(
        arg_reduce_layout.variant(entry, execution["workgroupSize"][0])
    )
    inputs, native_outputs, observed, matched = bind(
        entry,
        descriptor,
        supplied,
        packets,
        OUTPUT,
        arg_reduce_layout.operand_types(entry),
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
        raise RuntimeError("Native arg reduction did not return its output")
    returned, expected = result.outputs[binding], native_outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native arg reduction output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native arg reduction output extent differs")
    words = unpack(OUTPUT_TYPE, target, words)
    if len(words) != outputs + GUARD_COUNT:
        raise RuntimeError("Native arg reduction output extent differs")
    if words[outputs:] != guard_words(OUTPUT_TYPE):
        raise RuntimeError("Native arg reduction changed its output guard")
    words = words[:outputs]
    payload = bytes((CTYPES[OUTPUT_TYPE] * outputs)(*words))
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
            "argReduceMetadata": metadata,
            "inputs": inputs,
            "operandValues": values,
            "initialValues": guard_words(OUTPUT_TYPE, outputs),
            "outputValues": words,
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
