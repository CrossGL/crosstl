"""Audit lazily translated MLX fused cross-entropy dispatches.

Every dispatch must name the package its cross-entropy cache derives again
from the pinned source at the same threadgroup width, and its retained native
modules and request must match the event. The row length and launch are
validated again against cross_entropy.cpp. A record that keeps its values
must have uploaded exactly those words, must name class indices inside their
rows and must hash to its output. A record compacted past ``RETAINED_LIMIT``
logits keeps the lengths and digests of its long lists; its extents, guard
uploads and row length are checked. The forward kernel rounds its simdgroup
reductions in an order the target fixes, so the unchanged upstream tests
compare the values with MLX's composed reference.
"""

import re
from pathlib import Path

from demos.integrations.mlx.portable_host import cross_entropy_layout as layout
from demos.integrations.mlx.portable_host.cross_entropy_dispatch import (
    RETAINED_LIMIT,
    output_hash,
    uploads,
)
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.trace_records import compact_trace_values

__all__ = ("audit_event",)

GEOMETRY = ("workgroupCount", "workgroupSize")
DIGEST = re.compile(r"[0-9a-f]{64}")


def _extent(value):
    if isinstance(value, list):
        return len(value)
    compacted = value.get("compactedValues") if isinstance(value, dict) else None
    require(
        isinstance(compacted, dict)
        and type(compacted.get("length")) is int
        and DIGEST.fullmatch(str(compacted.get("sha256"))),
        "Cross-entropy trace value list is neither retained nor compacted",
    )
    return compacted["length"]


def _without_values(packet):
    return {key: value for key, value in packet.items() if key != "values"}


def _check_uploads(event, entry, values, outputs, axis_size, retained):
    """Every bound packet is the one its values, guards or row length give."""
    target = event["target"]
    words = {
        name: value if isinstance(value, list) else [0] * _extent(value)
        for name, value in values.items()
    }
    expected = uploads(entry, target, words, outputs, axis_size)
    if not retained:
        expected = compact_trace_values(expected)
    bound = list(event["inputs"].values())
    require(
        len(bound) == len(expected), "Cross-entropy uploads differ from its buffers"
    )
    for name, packet in expected.items():
        if name not in values or isinstance(values[name], list):
            require(packet in bound, f"Cross-entropy upload of {name} differs")
            continue
        require(
            any(
                _without_values(item) == _without_values(packet)
                and _extent(item["values"]) == _extent(packet["values"])
                for item in bound
            ),
            f"Cross-entropy upload of {name} differs from its extent",
        )


def audit_event(event, cache):
    """Check one cross-entropy dispatch; returns whether it kept its values."""
    entry, threads = event["entry"], event["threads"]
    layout.signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require(
        "threadGridSize" not in event,
        "Cross-entropy launch names an exact thread grid",
    )
    values, written = event["operandValues"], event["outputValues"]
    require(
        isinstance(values, dict) and set(values) == set(layout.inputs_of(entry)),
        "Cross-entropy trace does not name its element buffers",
    )
    counts = {name: _extent(value) for name, value in values.items()}
    counts[layout.OUTPUT] = _extent(written)
    axis_size = event["axisSize"]
    layout.validate_metadata(entry, axis_size, counts, threads, execution)
    require(event["dispatchVersion"] == 3, "Cross-entropy dispatch version differs")
    retained = threads <= RETAINED_LIMIT
    lists = [*values.values(), written]
    require(
        (
            all(isinstance(value, list) for value in lists)
            if retained
            else not isinstance(values[layout.INPUT], list)
        ),
        "Cross-entropy trace retention does not match its extent",
    )
    if isinstance(values[layout.TARGETS], list):
        layout.require_targets(values[layout.TARGETS], axis_size)
    _check_uploads(event, entry, values, counts[layout.OUTPUT], axis_size, retained)
    if retained:
        require(
            output_hash(entry, written) == event["outputHash"],
            "Cross-entropy host output hash changed",
        )
    else:
        require(
            DIGEST.fullmatch(str(event["outputHash"])),
            "Cross-entropy output hash is missing",
        )
    descriptor, directory = cache.get(
        layout.variant(entry, execution["workgroupSize"][0])
    )
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Cross-entropy dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == event["target"]
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Cross-entropy native request changed",
    )
    audit_native_execution(event)
    return retained
