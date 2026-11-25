"""Audit translated ``masked_assign`` dispatches of masked scatters.

Every dispatch must name the package its cache derives again from the pinned
template, upload exactly its retained operands, start its output from the
destination copy and keep its trailing guard, and its native request and
modules must match the event. The scatter offsets must be the exclusive
per-batch counts of the mask's set elements, which indexing.cpp's scan
computes. The output is recomputed as ``masked_assign_impl`` writes it: a set
element whose offset lies inside its batch's source receives that source
element, read at the contiguous index or at ``elem_to_loc<uint>``; every
other element keeps the destination's value.
"""

import hashlib
import math

from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    guard_words,
    packet,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.masked_scatter_dispatch import (
    MAX_ELEMENTS,
    launch_for,
    operand_types,
    row_contiguous,
    source_location,
)
from demos.integrations.mlx.portable_host.masked_scatter_packages import signature

__all__ = ("audit_event", "expected_offsets", "expected_output")

METADATA = {
    "dtype",
    "sourceContiguous",
    "mask",
    "scatterOffsets",
    "sourceShape",
    "sourceStrides",
    "sourceBatchSize",
    "maskBatchSize",
    "sourceSpan",
}


def expected_offsets(mask, batch):
    """The exclusive per-batch prefix counts of ``mask``'s set elements."""
    offsets, count = [], 0
    for index, value in enumerate(mask):
        if index % batch == 0:
            count = 0
        offsets.append(count)
        count += int(bool(value))
    return offsets


def expected_output(metadata, source, initial):
    """The output ``masked_assign_impl`` writes over the destination copy."""
    shape, strides = metadata["sourceShape"], metadata["sourceStrides"]
    batch, mask_batch = metadata["sourceBatchSize"], metadata["maskBatchSize"]
    output = list(initial)
    for index, value in enumerate(metadata["mask"]):
        offset = metadata["scatterOffsets"][index]
        if not value or offset >= batch:
            continue
        element = (index // mask_batch) * batch + offset
        location = source_location(
            element, shape, strides, metadata["sourceContiguous"]
        )
        require(0 <= location < len(source), "Masked scatter read leaves its source")
        output[index] = source[location]
    return output


def audit_event(event, cache=None, *, native=True):
    """Recompute one masked scatter dispatch from its retained operands.

    ``cache`` derives the event's package again from the pinned template;
    ``native`` also verifies the retained native modules and the request.
    """
    entry, target, threads = event["entry"], event["target"], event["threads"]
    dtype, contiguous = signature(entry)
    metadata = event["maskedScatterMetadata"]
    require(
        set(metadata) == METADATA
        and metadata["dtype"] == dtype
        and metadata["sourceContiguous"] is contiguous
        and type(threads) is int
        and 0 < threads <= MAX_ELEMENTS
        and len(metadata["mask"]) == len(metadata["scatterOffsets"]) == threads,
        "Masked scatter event metadata does not match its entry",
    )
    shape, strides = metadata["sourceShape"], metadata["sourceStrides"]
    mask_batch = metadata["maskBatchSize"]
    span = 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    require(
        0 < len(shape) == len(strides)
        and all(size >= 1 for size in shape)
        and all(stride >= 0 for stride in strides)
        and span == metadata["sourceSpan"] == len(event["sourceValues"])
        and contiguous == row_contiguous(shape, strides)
        and type(mask_batch) is int
        and mask_batch * shape[0] == threads
        and metadata["sourceBatchSize"] * shape[0] == math.prod(shape),
        "Masked scatter source layout does not match its metadata",
    )
    require(
        metadata["scatterOffsets"] == expected_offsets(metadata["mask"], mask_batch),
        "Masked scatter offsets are not the exclusive counts of its mask",
    )
    types = operand_types(entry)
    uploads = [*event["inputs"].values()]
    for name, words in (
        ("mask", metadata["mask"]),
        ("scatter_offsets", metadata["scatterOffsets"]),
        ("src", event["sourceValues"]),
        ("src_shapes", shape),
        ("src_strides", strides),
        ("src_ndim", [len(shape)]),
        ("src_batch_size", [metadata["sourceBatchSize"]]),
        ("mask_batch_size", [mask_batch]),
    ):
        require(
            packet(types[name], target, words) in uploads,
            "Masked scatter upload differs from its operands",
        )
    initial = event["initialValues"]
    require(
        len(initial) == threads
        and packet(dtype, target, initial + guard_words(dtype)) in uploads,
        "Masked scatter output upload differs from its destination",
    )
    outputs = event["outputValues"]
    require(
        outputs == expected_output(metadata, event["sourceValues"], initial),
        f"Masked scatter of {entry} differs from its operands",
    )
    require(
        hashlib.sha256(bytes((CTYPES[dtype] * threads)(*outputs))).hexdigest()
        == event.get("outputHash"),
        "Masked scatter host output hash changed",
    )
    if cache is not None:
        descriptor, directory = cache.get(entry)
        require_unchanged(descriptor, directory)
        require(
            descriptor["artifact"] == event["artifact"]
            and str(directory) == event["packageRoot"],
            "Masked scatter dispatch does not name its cached package",
        )
    if native:
        request = event["details"]["request"]
        launch = launch_for(threads)
        require(
            request["target"] == target
            and request["entryPoint"]
            == {"opengl": "main", "directx": "CSMain"}.get(target, entry)
            and request["dispatch"]["workgroupCount"]
            == event["workgroupCount"]
            == launch["workgroupCount"]
            and request["dispatch"]["workgroupSize"]
            == event["workgroupSize"]
            == launch["workgroupSize"]
            and event["dispatchVersion"] == 3,
            "Masked scatter native request changed",
        )
        audit_native_execution(event)
    return 1
