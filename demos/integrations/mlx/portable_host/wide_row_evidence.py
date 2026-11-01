"""Audit lazily translated small-row reductions of the wide types.

Every dispatch must name the package its small-row cache derives again from
the pinned source for the same launch, and its native modules must match the
event: the Metal request and modules, or each Direct3D and OpenGL region.
The row metadata and launch are validated again against upstream's plan.
Integer dispatches, and all and any of every type, are recomputed exactly
(``wide_reduction_evidence.fold``): output ``r`` folds, for each of
the ``non_row_reductions`` positions ``reduce_shape`` and ``reduce_strides``
address, the ``row_size`` contiguous inputs that start at ``r``'s offset
under ``shape`` and ``strides``, with two's-complement wrapping in the
accumulator type for sums and products. Float16, bfloat16 and complex64
sums, products and extrema round in their element type in an order the kernel fixes; they are checked by package and native
identity, and their values are compared with the CPU results by the upstream
tests themselves.
"""

import hashlib
import itertools
from pathlib import Path

from demos.integrations.mlx.portable_host import row_reduction_layout
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    guard_words,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    audit_regions,
    require_unchanged,
)
from demos.integrations.mlx.portable_host.wide_reduction_evidence import exact, fold
from demos.integrations.mlx.portable_host.wide_reduction_packages import accumulator
from demos.integrations.mlx.portable_host.wide_row_dispatch import (
    ENTRIES,
    signature,
    uploads,
)

__all__ = ("audit_event", "expected_words", "recomputable")

GEOMETRY = ("workgroupCount", "workgroupSize", "threadGridSize")


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
    """Every output word of an exact small-row dispatch over ``words``."""
    require(
        exact(operation, dtype),
        "Wide row recompute requires an integer type or all/any",
    )
    ndim, reduce_ndim = metadata["ndim"][0], metadata["reduce_ndim"][0]
    rows = _offsets(metadata["shape"][:ndim], metadata["strides"][:ndim])
    positions = _offsets(
        metadata["reduce_shape"][:reduce_ndim],
        metadata["reduce_strides"][:reduce_ndim],
    )
    row_size = metadata["row_size"][0]
    return [
        fold(
            operation,
            dtype,
            [
                words[row + position + column]
                for position in positions
                for column in range(row_size)
            ],
        )
        for row in rows
    ]


def audit_event(event, cache, modules):
    """Check one wide small-row dispatch; returns the outputs checked."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    require(entry in ENTRIES, "Dispatch is not a wide small-row reduction")
    operation, dtype = signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    values, outputs = event["operandValues"], event["outputValues"]
    metadata = row_reduction_layout.validate_metadata(
        entry, event["rowMetadata"], threads, len(values), len(outputs), execution
    )
    rows = len(outputs)
    output_type = accumulator(operation, dtype)
    require(
        event["initialValues"] == guard_words(output_type, rows)
        and event["dispatchVersion"] == 3,
        "Wide small-row readback extent differs",
    )
    expected_uploads = uploads(dtype, target, values, rows, metadata, output_type)
    bound = [*event["inputs"].values()]
    require(
        all(value in bound for value in expected_uploads.values())
        and len(bound) == len(expected_uploads),
        "Wide small-row upload differs from its values",
    )
    if recomputable(entry):
        expected = expected_words(operation, dtype, values, metadata)
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            require(
                actual == wanted,
                f"Wide small-row output {index} of {entry} differs from its inputs",
            )
    payload = bytes((CTYPES[output_type] * rows)(*outputs))
    require(
        hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
        "Wide small-row host output hash changed",
    )
    packages = list(
        cache.get(
            entry,
            thread_grid_size=execution["threadGridSize"],
            workgroup_size=execution["workgroupSize"],
        )
    )
    require(packages, "Wide small-row dispatch has no cached package")
    for descriptor, directory in packages:
        require_unchanged(descriptor, directory)
    require(
        event["artifact"] == packages[0][0]["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(packages[0][1]).resolve(),
        "Wide small-row dispatch does not name its cached package",
    )
    if target != "metal":
        audit_regions(event, packages, Path(modules).resolve())
        return rows
    require(len(packages) == 1, "Wide small-row dispatch skipped its regions")
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY[:2]),
        "Wide small-row native request changed",
    )
    audit_native_execution(event)
    return rows
