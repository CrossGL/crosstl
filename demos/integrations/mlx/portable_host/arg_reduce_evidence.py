"""Audit lazily translated ``arg_reduce_general`` dispatches.

Every dispatch must name the package its arg reduction cache derives again
from the pinned source for the same threadgroup width, and its retained
native modules and request must match the event. The metadata and launch are
validated again against ``ArgReduce::eval_gpu``'s plan, and every index is
recomputed exactly from the retained input words for every dtype.

The kernel starts each lane at index zero and ``Limits::max`` (argmin) or
``Limits::min`` (argmax), folds its reads in ascending index order, taking a
value only when it is strictly better or the first NaN, and then combines
lanes under one total order: any NaN before any number, then the better
value, then the lower index. That order makes the combination independent of
the lane partition, so the result equals one ascending fold of the whole
axis from the same start, which is what ``expected_indices`` computes.
"""

import hashlib
import itertools
import math
import struct
from pathlib import Path

from demos.integrations.mlx.portable_host import arg_reduce_layout
from demos.integrations.mlx.portable_host.arg_reduce_dispatch import uploads
from demos.integrations.mlx.portable_host.element_storage import CTYPES, guard_words
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)

__all__ = ("audit_event", "expected_indices")

GEOMETRY = ("workgroupCount", "workgroupSize")
OUTPUT_TYPE = arg_reduce_layout.OUTPUT_TYPE


def _value(dtype, word):
    """The value MLX's storage ``word`` of ``dtype`` holds."""
    if dtype == "float32":
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<H", word))[0]
    if dtype == "bfloat16":
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return word


def _start(operation, dtype):
    """``Limits<T>::max`` for argmin and ``Limits<T>::min`` for argmax."""
    if dtype in ("float16", "float32", "bfloat16"):
        return math.inf if operation == "argmin" else -math.inf
    if dtype == "bool_":
        return int(operation == "argmin")
    bits = 8 * struct.calcsize(CTYPES[dtype]._type_)
    signed = dtype.startswith("int")
    if operation == "argmin":
        return (1 << (bits - signed)) - 1
    return -(1 << (bits - 1)) if signed else 0


def _isnan(value):
    return isinstance(value, float) and math.isnan(value)


def _offsets(shape, strides):
    """Every row's first element offset, last axis fastest."""
    return [
        sum(index * stride for index, stride in zip(position, strides))
        for position in itertools.product(*(range(size) for size in shape))
    ]


def expected_indices(operation, dtype, words, metadata):
    """Every output index of an arg reduction over the input ``words``."""
    require(operation in arg_reduce_layout.OPERATIONS, "Unknown arg reduction")
    ndim = metadata["ndim"][0]
    axis_size, axis_stride = metadata["axis_size"][0], metadata["axis_stride"][0]
    start = _start(operation, dtype)
    minimum = operation == "argmin"
    indices = []
    for offset in _offsets(metadata["shape"][:ndim], metadata["in_strides"][:ndim]):
        best_index, best = 0, start
        for index in range(axis_size):
            value = _value(dtype, words[offset + index * axis_stride])
            if (not _isnan(best) and _isnan(value)) or (
                value < best if minimum else value > best
            ):
                best_index, best = index, value
        indices.append(best_index)
    return indices


def audit_event(event, cache):
    """Check one arg reduction dispatch; returns the outputs checked."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    operation, dtype = arg_reduce_layout.signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require(
        "threadGridSize" not in event,
        "Arg reduction launch names an exact thread grid",
    )
    values, outputs = event["operandValues"], event["outputValues"]
    require(
        isinstance(values, list) and isinstance(outputs, list),
        "Arg reduction trace did not retain its values",
    )
    metadata = arg_reduce_layout.validate_metadata(
        entry, event["argReduceMetadata"], threads, len(values), len(outputs), execution
    )
    count = len(outputs)
    require(
        event["initialValues"] == guard_words(OUTPUT_TYPE, count)
        and event["dispatchVersion"] == 3,
        "Arg reduction readback extent differs",
    )
    expected_uploads = uploads(dtype, target, values, count, metadata)
    bound = [*event["inputs"].values()]
    require(
        all(value in bound for value in expected_uploads.values())
        and len(bound) == len(expected_uploads),
        "Arg reduction upload differs from its values",
    )
    expected = expected_indices(operation, dtype, values, metadata)
    require(len(expected) == count, "Arg reduction output extent differs")
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        require(
            actual == wanted,
            f"Arg reduction output {index} of {entry} differs from its inputs",
        )
    payload = bytes((CTYPES[OUTPUT_TYPE] * count)(*outputs))
    require(
        hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
        "Arg reduction host output hash changed",
    )
    descriptor, directory = cache.get(
        arg_reduce_layout.variant(entry, execution["workgroupSize"][0])
    )
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Arg reduction dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Arg reduction native request changed",
    )
    audit_native_execution(event)
    return count
