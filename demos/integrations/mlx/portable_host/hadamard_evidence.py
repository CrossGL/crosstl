"""Audit lazily translated MLX Hadamard transform dispatches.

Every dispatch must name the package its Hadamard cache derives again from
the pinned headers, and its retained native modules and request must match
the event. The slice and its launch are validated again against
``hadamard_mn_contiguous``. A record that keeps its values must have uploaded
exactly those words, its scale and guard-filled output, and must hash to its
output. A record compacted past ``RETAINED_LIMIT`` elements keeps the
lengths and digests of its long lists; its extents, guard upload and scale
are checked. The butterflies round in the order each target's float
arithmetic fixes, so the unchanged upstream tests compare the values with
NumPy's Hadamard products and with the MLX CPU.
"""

import re
from pathlib import Path

from demos.integrations.mlx.portable_host import hadamard_layout as layout
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.hadamard_dispatch import (
    RETAINED_LIMIT,
    output_hash,
    uploads,
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
        "Hadamard trace value list is neither retained nor compacted",
    )
    return compacted["length"]


def _without_values(packet):
    return {key: value for key, value in packet.items() if key != "values"}


def _check_uploads(event, entry, values, scale, retained):
    """Every bound packet is the one its values, scale or guards give."""
    words = values if retained else [0] * _extent(values)
    expected = uploads(entry, event["target"], words, scale)
    if not retained:
        expected = compact_trace_values(expected)
    bound = list(event["inputs"].values())
    require(len(bound) == len(expected), "Hadamard uploads differ from its buffers")
    for name, packet in expected.items():
        if retained or name != layout.INPUT:
            require(packet in bound, f"Hadamard upload of {name} differs")
            continue
        require(
            any(
                _without_values(item) == _without_values(packet)
                and _extent(item["values"]) == _extent(packet["values"])
                for item in bound
            ),
            "Hadamard upload of in differs from its extent",
        )


def audit_event(event, cache):
    """Check one Hadamard dispatch; returns whether it kept its values."""
    entry = event["entry"]
    layout.signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require("threadGridSize" not in event, "Hadamard launch names an exact thread grid")
    values, written = event["operandValues"], event["outputValues"]
    require(
        isinstance(values, dict) and set(values) == {layout.INPUT},
        "Hadamard trace does not name its input",
    )
    values = values[layout.INPUT]
    size, scale = _extent(values), event["scaleWord"]
    require(
        _extent(written) == size and type(scale) is int,
        "Hadamard trace extents or scale differ",
    )
    require(
        layout.plan(entry, size, scale) == execution
        and layout.threads(execution) == event["threads"],
        "Hadamard launch does not match the upstream plan",
    )
    require(event["dispatchVersion"] == 3, "Hadamard dispatch version differs")
    retained = size <= RETAINED_LIMIT
    require(
        isinstance(values, list) == retained and isinstance(written, list) == retained,
        "Hadamard trace retention does not match its extent",
    )
    _check_uploads(event, entry, values, scale, retained)
    if retained:
        require(
            output_hash(entry, written) == event["outputHash"],
            "Hadamard host output hash changed",
        )
    else:
        require(
            DIGEST.fullmatch(str(event["outputHash"])),
            "Hadamard output hash is missing",
        )
    descriptor, directory = cache.get(entry)
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Hadamard dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == event["target"]
        and request["entryPoint"]
        == {"opengl": "main", "directx": "CSMain"}.get(event["target"], entry)
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Hadamard native request changed",
    )
    audit_native_execution(event)
    return retained
