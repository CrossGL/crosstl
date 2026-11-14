"""Dispatch and audit the translated ``compute_dynamic_offset`` kernels.

``DynamicSlice`` and ``DynamicSliceUpdate`` pass the start indices, the
strides of the sliced array, the sliced axes and their count, as
``mlx/backend/metal/slicing.cpp`` binds them, and one int64 ``offset`` output,
in a launch of one thread. The output carries a trailing guard. The host then
hands the native offset to the dynamic copy kernels;
``dynamic_offset_evidence`` audits each dispatch.
"""

import ctypes
import hashlib

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host.dynamic_offset_packages import (
    ENTRIES,
    signature,
)
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

__all__ = ("ENTRIES", "LAUNCH", "MAX_AXES", "dispatch", "operand_types", "validate")

OUTPUT = "offset"
# backend.cpp bounds the sliced array's rank.
MAX_AXES = 64
LAUNCH = {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]}


def operand_types(entry):
    """Every source parameter's element type."""
    return {
        "indices": signature(entry),
        OUTPUT: "int64",
        "strides": "int64",
        "axes": "int32",
        "n_axes": "int32",
    }


def validate(entry, buffers, threads, execution):
    """Check one launch and return the operands it reads."""
    types = operand_types(entry)
    if threads != 1 or execution != LAUNCH:
        raise ValueError("Dynamic offset launch differs from one thread")
    if set(buffers) != set(types):
        raise ValueError("Dynamic offset buffers do not match their kernel")
    for name, buffer in buffers.items():
        size = ctypes.sizeof(CTYPES[types[name]])
        if (
            buffer.dtype != types[name].encode("ascii")
            or not buffer.data
            or buffer.data % size
            or buffer.output != int(name == OUTPUT)
        ):
            raise ValueError(
                "Dynamic offset buffer type, alignment or direction differs"
            )
    if buffers[OUTPUT].count != 1 or buffers["n_axes"].count != 1:
        raise ValueError("Dynamic offset scalar extent differs")
    (count,) = buffer_words(buffers["n_axes"], "int32")
    strides = buffer_words(buffers["strides"], "int64")
    axes = buffer_words(buffers["axes"], "int32")
    indices = buffer_words(buffers["indices"], types["indices"])
    if (
        not 0 < count <= MAX_AXES
        or buffers["axes"].count != count
        or buffers["indices"].count != count
        or not 0 < len(strides) <= MAX_AXES
        or len(set(axes)) != count
        or any(not 0 <= axis < len(strides) for axis in axes)
    ):
        raise ValueError("Dynamic offset axes exceed their strides")
    return {
        "dtype": types["indices"],
        "indices": indices,
        "strides": strides,
        "axes": axes,
    }


def dispatch(host, entry, buffers, count, threads, launch):
    from demos.integrations.mlx.portable_host.runtime import DISPATCH_VERSION

    if host.dynamic_offsets is None or launch is None or not buffers or count != 5:
        raise ValueError(
            "Dynamic offset dispatch requires complete buffers, pinned sources and "
            "launch geometry"
        )
    supplied = supplied_buffers(buffers, count, "Dynamic offset")
    execution = launch.execution()
    metadata = validate(entry, supplied, threads, execution)
    target = host.target
    values = {
        "indices": metadata["indices"],
        "strides": metadata["strides"],
        "axes": metadata["axes"],
        "n_axes": [len(metadata["axes"])],
    }
    types = operand_types(entry)
    initial, tail = guard_words("int64", 1), guard_words("int64")
    uploads = {
        **{name: packet(types[name], target, words) for name, words in values.items()},
        OUTPUT: packet("int64", target, initial + tail),
    }
    descriptor, directory = host.dynamic_offsets.get(entry)
    inputs, outputs, observed, matched = bind(
        entry, descriptor, supplied, uploads, OUTPUT, types
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
        raise RuntimeError("Native dynamic offset kernel did not return its output")
    returned, expected = result.outputs[binding], outputs[binding]
    if any(
        returned.get(key) != expected.get(key) for key in ("dtype", "encoding", "shape")
    ):
        raise RuntimeError("Native dynamic offset output layout differs")
    words = returned.get("values")
    if not isinstance(words, list):
        raise RuntimeError("Native dynamic offset output extent differs")
    words = unpack("int64", target, words)
    if len(words) != 1 + GUARD_COUNT or words[1:] != tail:
        raise RuntimeError("Native dynamic offset kernel changed its output guard")
    payload = bytes((ctypes.c_int64 * 1)(words[0]))
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
            "dynamicOffsetMetadata": metadata,
            "inputs": inputs,
            "initialValues": initial,
            "outputValues": words[:1],
            "outputHash": hashlib.sha256(payload).hexdigest(),
        },
    )
    ctypes.memmove(supplied[OUTPUT].data, payload, len(payload))
    host.dispatch_count += 1
