"""Audit lazily translated searchsorted.metal passes.

Every pass must name the package its cache derives again from the pinned
source at the same pass size, and its retained native modules and request
must match the event. The metadata and launch are validated again. Each
output is ``searchsorted_impl``'s binary search under MLX's ``LessThan`` (NaN
after every ordered value, complex values by real then imaginary part), so a
record that keeps its values must have uploaded exactly those words, and its
outputs are recomputed and compared exactly, also for an unsorted sequence.
A record compacted past ``RETAINED_LIMIT`` elements keeps the lengths and
digests of its long lists; its extents, guard uploads and metadata are
checked, and the unchanged upstream test compares the result itself.
"""

import re
from pathlib import Path

from demos.integrations.mlx.portable_host import searchsorted_layout
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.searchsorted_dispatch import (
    INPUTS,
    output_hash,
    retained,
    uploads,
)
from demos.integrations.mlx.portable_host.sort_evidence import decode, less_than
from demos.integrations.mlx.portable_host.trace_records import compact_trace_values

__all__ = ("audit_event", "expected_outputs")

GEOMETRY = ("workgroupCount", "workgroupSize")
DIGEST = re.compile(r"[0-9a-f]{64}")
UINT32_MASK = 0xFFFFFFFF


def search(sequence, value, n, stride, right):
    """``searchsorted_impl``: the first position whose element is not below."""
    low, high = 0, n
    while low < high:
        middle = low + (high - low) // 2
        element = sequence[middle * stride]
        below = not less_than(value, element) if right else less_than(element, value)
        if below:
            low = middle + 1
        else:
            high = middle
    return low & UINT32_MASK


def expected_outputs(entry, metadata, values):
    """Every value's search result over the retained sequence words."""
    dtype, right = searchsorted_layout.signature(entry)
    sequence = [decode(dtype, word) for word in values["a"]]
    return [
        search(sequence, decode(dtype, word), metadata["n"], metadata["aStride"], right)
        for word in values["v"]
    ]


def _extent(value):
    if isinstance(value, list):
        return len(value)
    compacted = value.get("compactedValues") if isinstance(value, dict) else None
    require(
        isinstance(compacted, dict)
        and type(compacted.get("length")) is int
        and DIGEST.fullmatch(str(compacted.get("sha256"))),
        "Searchsorted trace value list is neither retained nor compacted",
    )
    return compacted["length"]


def _audit_uploads(event, values, metadata, keep):
    entry, target, threads = event["entry"], event["target"], event["threads"]
    if keep:
        expected = uploads(entry, target, values, threads, metadata)
    else:
        placeholders = {name: [0] * _extent(values[name]) for name in INPUTS}
        expected = compact_trace_values(
            uploads(entry, target, placeholders, threads, metadata)
        )
    bound = list(event["inputs"].values())
    require(len(bound) == len(expected), "Searchsorted uploads differ from its buffers")
    for name, packet in expected.items():
        if keep or name not in INPUTS:
            require(
                packet in bound,
                f"Searchsorted upload of {name} differs from its values",
            )
            continue
        require(
            any(
                {k: v for k, v in item.items() if k != "values"}
                == {k: v for k, v in packet.items() if k != "values"}
                and _extent(item["values"]) == _extent(packet["values"])
                for item in bound
            ),
            f"Searchsorted upload of {name} differs from its extent",
        )


def audit_event(event, cache):
    """Check one searchsorted pass; returns whether its outputs were recomputed."""
    entry, target, threads = event["entry"], event["target"], event["threads"]
    execution = {key: event[key] for key in GEOMETRY}
    require(
        "threadGridSize" not in event, "Searchsorted launch names an exact thread grid"
    )
    require(
        execution == searchsorted_layout.execution(threads),
        "Searchsorted launch differs from its value count",
    )
    values, outputs = event["operandValues"], event["outputValues"]
    require(
        isinstance(values, dict) and set(values) == set(INPUTS),
        "Searchsorted trace does not name its element buffers",
    )
    span = _extent(values["a"])
    require(
        _extent(values["v"]) == threads and _extent(outputs) == threads,
        "Searchsorted trace extents differ from its launch",
    )
    metadata = event["searchMetadata"]
    searchsorted_layout.check_metadata(metadata, {"a": span})
    require(event["dispatchVersion"] == 3, "Searchsorted dispatch version differs")
    keep = retained(span, threads)
    require(
        keep == all(
            isinstance(value, list) for value in (values["a"], values["v"], outputs)
        ),
        "Searchsorted trace retention does not match its extent",
    )
    _audit_uploads(event, values, metadata, keep)
    if keep:
        require(
            output_hash(outputs) == event["outputHash"],
            "Searchsorted host output hash changed",
        )
        require(
            expected_outputs(entry, metadata, values) == outputs,
            "Searchsorted outputs differ from the LessThan binary search",
        )
    else:
        require(
            DIGEST.fullmatch(str(event["outputHash"])),
            "Searchsorted output hash is missing",
        )
    descriptor, directory = cache.get(searchsorted_layout.variant(entry, threads))
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Searchsorted dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == target
        and request["entryPoint"] == entry
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Searchsorted native request changed",
    )
    audit_native_execution(event)
    return keep
