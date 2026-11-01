"""Audit lazily translated softmax and logsumexp row dispatches.

Every dispatch must name the package its softmax cache derives again from the
pinned source for the same threadgroup width, and its retained native modules
and request must match the event. The row length and launch are validated
again against ``Softmax::eval_gpu`` or ``LogSumExp::eval_gpu``, the uploads
must be the retained input words, and the output guard and hash must hold.
The kernels round ``fast::exp`` and their simdgroup sums in an order the
target fixes, so their values are compared with the CPU results by the
upstream tests themselves.
"""

import hashlib
from pathlib import Path

from demos.integrations.mlx.portable_host import softmax_layout
from demos.integrations.mlx.portable_host.element_storage import CTYPES, guard_words
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.softmax_dispatch import uploads

__all__ = ("audit_event",)

GEOMETRY = ("workgroupCount", "workgroupSize")


def audit_event(event, cache):
    """Check one softmax or logsumexp dispatch; returns the outputs checked."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    _, _, dtype, _ = softmax_layout.signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require(
        "threadGridSize" not in event,
        "Softmax or logsumexp launch names an exact thread grid",
    )
    values, outputs = event["operandValues"], event["outputValues"]
    require(
        isinstance(values, list) and isinstance(outputs, list),
        "Softmax or logsumexp trace did not retain its values",
    )
    axis_size = softmax_layout.validate_metadata(
        entry, event["axisSize"], threads, len(values), len(outputs), execution
    )
    count = len(outputs)
    require(
        event["initialValues"] == guard_words(dtype, count)
        and event["dispatchVersion"] == 3,
        "Softmax or logsumexp readback extent differs",
    )
    expected_uploads = uploads(dtype, target, values, count, axis_size)
    bound = [*event["inputs"].values()]
    require(
        all(value in bound for value in expected_uploads.values())
        and len(bound) == len(expected_uploads),
        "Softmax or logsumexp upload differs from its values",
    )
    payload = bytes((CTYPES[dtype] * count)(*outputs))
    require(
        hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
        "Softmax or logsumexp host output hash changed",
    )
    descriptor, directory = cache.get(
        softmax_layout.variant(entry, execution["workgroupSize"][0])
    )
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Softmax or logsumexp dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Softmax or logsumexp native request changed",
    )
    audit_native_execution(event)
    return count
