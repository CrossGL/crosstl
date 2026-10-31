"""Audit lazily translated small-column reductions of the wide types.

Every dispatch must name the package its small-column cache derives again
from the pinned source for the same threadgroup size, and its retained native
modules and request must match the event. The column metadata and launch are
validated again against ``strided_reduce_small``'s plan. Integer dispatches,
and all and any of every type, are recomputed exactly: output ``o * reduction_stride + c`` folds, for each
position ``reduce_shape`` and ``reduce_strides`` address, the input at column
``c`` of ``o``'s offset under ``shape`` and ``strides``, with two's-complement
wrapping in the accumulator type for sums and products. Float16, bfloat16
and complex64 sums, products and extrema round or select in their element type in an order the kernel fixes; they are
checked by package and native identity, and their values are compared with
the CPU results by the upstream tests themselves.
"""

import hashlib
import itertools
from pathlib import Path

from demos.integrations.mlx.portable_host import column_small_layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    guard_words,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.wide_column_dispatch import (
    ENTRIES,
    signature,
    uploads,
)
from demos.integrations.mlx.portable_host.wide_reduction_evidence import exact, fold
from demos.integrations.mlx.portable_host.wide_reduction_packages import accumulator

__all__ = ("audit_event", "expected_words", "recomputable")

GEOMETRY = ("workgroupCount", "workgroupSize")


def recomputable(entry):
    """Whether the audit recomputes ``entry``'s results from its inputs."""
    return exact(*signature(entry))


def _offsets(shape, strides):
    """Every element offset of ``shape`` and ``strides``, last axis fastest."""
    return [
        sum(index * stride for index, stride in zip(position, strides))
        for position in itertools.product(*(range(size) for size in shape))
    ]


def expected_words(operation, dtype, words, metadata):
    """Every output word of an exact small-column dispatch over ``words``."""
    require(
        exact(operation, dtype),
        "Wide column recompute requires an integer type or all/any",
    )
    ndim, reduce_ndim = metadata["ndim"][0], metadata["reduce_ndim"][0]
    outer = _offsets(metadata["shape"][:ndim], metadata["strides"][:ndim])
    positions = _offsets(
        metadata["reduce_shape"][:reduce_ndim],
        metadata["reduce_strides"][:reduce_ndim],
    )
    require(
        len(positions)
        == metadata["non_col_reductions"][0] * metadata["reduction_size"][0],
        "Wide column reduction loop does not cover its rows",
    )
    stride = metadata["reduction_stride"][0]
    return [
        fold(operation, dtype, [words[offset + column + p] for p in positions])
        for offset in outer
        for column in range(stride)
    ]


def audit_event(event, cache):
    """Check one wide small-column dispatch; returns the outputs checked."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    require(entry in ENTRIES, "Dispatch is not a wide small-column reduction")
    operation, dtype = signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require(
        "threadGridSize" not in event,
        "Wide small-column launch names an exact thread grid",
    )
    values, outputs = event["operandValues"], event["outputValues"]
    metadata = column_small_layout.validate_metadata(
        entry, event["columnMetadata"], threads, len(values), len(outputs), execution
    )
    count = len(outputs)
    output_type = accumulator(operation, dtype)
    require(
        event["initialValues"] == guard_words(output_type, count)
        and event["dispatchVersion"] == 3,
        "Wide small-column readback extent differs",
    )
    expected_uploads = uploads(dtype, target, values, count, metadata, output_type)
    bound = [*event["inputs"].values()]
    require(
        all(value in bound for value in expected_uploads.values())
        and len(bound) == len(expected_uploads),
        "Wide small-column upload differs from its values",
    )
    if recomputable(entry):
        expected = expected_words(operation, dtype, values, metadata)
        require(len(expected) == count, "Wide small-column output extent differs")
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            require(
                actual == wanted,
                f"Wide small-column output {index} of {entry} differs from its inputs",
            )
    payload = bytes((CTYPES[output_type] * count)(*outputs))
    require(
        hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
        "Wide small-column host output hash changed",
    )
    descriptor, directory = cache.get(
        column_small_layout.variant(entry, tuple(execution["workgroupSize"]))
    )
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Wide small-column dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Wide small-column native request changed",
    )
    audit_native_execution(event)
    return count
