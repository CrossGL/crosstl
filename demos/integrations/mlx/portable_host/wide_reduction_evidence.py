"""Audit lazily translated wide whole-array reductions.

Every dispatch must name the package its cache derives again from the pinned
source for the same count and launch, and its native request and modules must
match the event. Integer passes are recomputed exactly: each row of
``row_size`` inputs, clipped to ``in_size``, folds to one output word with
two's-complement wrapping for sums and products, and an empty row yields the
operation's identity, as ``all_reduce`` initializes ``Op::init``. Float16,
bfloat16 and complex64 passes round in their element type in an order the
simdgroup tree fixes; they are checked by package and native identity like
the prebuilt float families, and their values are compared with the CPU
results by the upstream tests themselves.
"""

import hashlib
import math
from functools import reduce

from demos.integrations.mlx.portable_host.copy_evidence import integer_bounds, wrap
from demos.integrations.mlx.portable_host.element_storage import (
    CTYPES,
    INEXACT,
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
from demos.integrations.mlx.portable_host.wide_reduction_packages import (
    dispatch_contract,
    signature,
)

__all__ = ("audit_event", "expected_words", "fold", "recomputable")


def recomputable(entry):
    """Whether the audit recomputes ``entry``'s results from its inputs."""
    return signature(entry)[1] not in INEXACT


def fold(operation, dtype, words):
    low, high = integer_bounds(dtype)
    if operation == "sum":
        return wrap(sum(words), dtype)
    if operation == "prod":
        return wrap(math.prod(words), dtype)
    if operation == "min":
        return reduce(min, words, high)
    return reduce(max, words, low)


def expected_words(operation, dtype, words, rows, row_size):
    """Every output word of an integer pass over ``words``."""
    require(dtype not in INEXACT, "Wide reduction recompute requires an integer type")
    low, high = integer_bounds(dtype)
    require(
        all(type(word) is int and low <= word <= high for word in words),
        "Wide reduction input outside its storage range",
    )
    return [
        fold(operation, dtype, words[row * row_size : (row + 1) * row_size])
        for row in range(rows)
    ]


def audit_event(event, cache=None, *, native=True):
    """Check one wide reduction pass; returns the outputs checked.

    ``cache`` derives the event's package again from the pinned source;
    ``native`` also verifies the retained native modules and the request.
    """
    entry, target, threads = event["entry"], event["target"], event["threads"]
    operation, dtype = signature(entry)
    execution = {key: event[key] for key in ("workgroupCount", "workgroupSize")}
    contract = dispatch_contract(entry, threads, execution)
    rows, row_size = execution["workgroupCount"][1], contract["rowSize"]
    require(
        event["reductionMetadata"] == {
            "operation": operation,
            "dtype": dtype,
            "rows": rows,
            "in_size": threads,
            "row_size": row_size,
        },
        "Wide reduction event metadata does not match its plan",
    )
    if recomputable(entry):
        values, outputs = event["operandValues"], event["outputValues"]
        require(
            len(values) == threads
            and len(outputs) == rows
            and event["initialValues"] == guard_words(dtype, rows),
            "Wide reduction readback extent differs",
        )
        uploads = [*event["inputs"].values()]
        for value in (
            packet(dtype, target, values),
            packet(dtype, target, guard_words(dtype, rows) + guard_words(dtype)),
        ):
            require(value in uploads, "Wide reduction upload differs from its values")
        expected = expected_words(operation, dtype, values, rows, row_size)
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            require(
                actual == wanted,
                f"Wide reduction row {index} of {entry} differs from its inputs",
            )
        payload = bytes((CTYPES[dtype] * rows)(*outputs))
        require(
            hashlib.sha256(payload).hexdigest() == event.get("outputHash"),
            "Wide reduction host output hash changed",
        )
    if cache is not None:
        descriptor, directory = cache.get(entry, threads, execution)
        require_unchanged(descriptor, directory)
        require(
            descriptor["artifact"] == event["artifact"]
            and str(directory) == event["packageRoot"],
            "Wide reduction dispatch does not name its cached package",
        )
    if native:
        request = event["details"]["request"]
        require(
            request["target"] == target
            and request["entryPoint"]
            == {"opengl": "main", "directx": "CSMain"}.get(target, entry)
            and request["dispatch"]["workgroupCount"] == event["workgroupCount"]
            and request["dispatch"]["workgroupSize"] == event["workgroupSize"]
            and event["dispatchVersion"] == 3,
            "Wide reduction native request changed",
        )
        audit_native_execution(event)
    return rows
