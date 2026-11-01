"""Audit lazily translated MLX RMS and layer normalization dispatches.

Every dispatch must name the package its normalization cache derives again
from the pinned source at the same threadgroup width, and its retained
native modules and request must match the event. The scalars and launch are
validated again against normalization.cpp. A record that keeps its values
must have uploaded exactly those words and must hash to its outputs; a
gradient without a weight must have left its ``gw`` placeholder as the guard
it uploaded. A record compacted past ``RETAINED_LIMIT`` row elements keeps
the lengths and digests of its long lists; its extents, guard uploads and
scalars are checked. The kernels round their simdgroup sums in an order the
target fixes, so the unchanged upstream tests compare their values with
MLX's composed normalization.
"""

import re
from pathlib import Path

from demos.integrations.mlx.portable_host import norm_layout
from demos.integrations.mlx.portable_host.element_storage import guard_words
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.norm_dispatch import (
    RETAINED_LIMIT,
    inputs_of,
    output_hash,
    uploads,
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
        "Normalization trace value list is neither retained nor compacted",
    )
    return compacted["length"]


def _without_values(packet):
    return {key: value for key, value in packet.items() if key != "values"}


def _check_uploads(event, entry, values, counts, metadata, retained):
    """Every bound packet is the one its values, guards or scalars give.

    A compacted record keeps the words of its short lists, so only the long
    element lists it compacted are compared by layout and extent.
    """
    target = event["target"]
    words = {
        name: value if isinstance(value, list) else [0] * _extent(value)
        for name, value in values.items()
    }
    expected = uploads(entry, target, words, counts, metadata)
    if not retained:
        expected = compact_trace_values(expected)
    bound = list(event["inputs"].values())
    require(
        len(bound) == len(expected), "Normalization uploads differ from its buffers"
    )
    for name, packet in expected.items():
        if name not in values or isinstance(values[name], list):
            require(packet in bound, f"Normalization upload of {name} differs")
            continue
        require(
            any(
                _without_values(item) == _without_values(packet)
                and _extent(item["values"]) == _extent(packet["values"])
                for item in bound
            ),
            f"Normalization upload of {name} differs from its extent",
        )


def audit_event(event, cache):
    """Check one normalization dispatch; returns whether it kept its values."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    _, gradient, _, dtype, has_w = norm_layout.signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require(
        "threadGridSize" not in event,
        "Normalization launch names an exact thread grid",
    )
    values, outputs = event["operandValues"], event["outputValues"]
    written = norm_layout.outputs(entry)
    require(
        isinstance(values, dict)
        and set(values) == set(inputs_of(entry))
        and isinstance(outputs, dict)
        and set(outputs) == set(written),
        "Normalization trace does not name its element buffers",
    )
    counts = {name: _extent(value) for name, value in {**values, **outputs}.items()}
    metadata = event["normMetadata"]
    norm_layout.validate_metadata(entry, metadata, counts, threads, execution)
    require(event["dispatchVersion"] == 3, "Normalization dispatch version differs")
    retained = threads <= RETAINED_LIMIT
    lists = [*values.values(), *outputs.values()]
    require(
        (
            all(isinstance(value, list) for value in lists)
            if retained
            else not isinstance(values["x"], list)
        ),
        "Normalization trace retention does not match its extent",
    )
    output_counts = {name: counts[name] for name in written}
    _check_uploads(event, entry, values, output_counts, metadata, retained)
    if gradient and not has_w:
        require(
            outputs["gw"] == guard_words(dtype, 1),
            "Normalization gradient wrote its weight placeholder",
        )
    if retained:
        require(
            output_hash(entry, outputs) == event["outputHash"],
            "Normalization host output hash changed",
        )
    else:
        require(
            DIGEST.fullmatch(str(event["outputHash"])),
            "Normalization output hash is missing",
        )
    descriptor, directory = cache.get(
        norm_layout.variant(entry, execution["workgroupSize"][0])
    )
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Normalization dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == norm_layout.source_entry(entry)
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Normalization native request changed",
    )
    audit_native_execution(event)
    return retained
