"""Audit translated MLX gated-delta dispatches.

Every dispatch must name the package its gated-delta cache derives again
from the pinned sources, and its retained native modules and request must
match the event. Its buffer extents, step count and launch are validated
again against ``gated_delta_layout.plan``. Each record keeps the digest of
every input array and output. The
recurrences round in the order each target's float arithmetic fixes, and
the backward adds its query, key, decay and beta gradients atomically in
any order, so the unchanged upstream tests compare the values with the
composed MLX fallback.
"""

import re
from pathlib import Path

from demos.integrations.mlx.portable_host import gated_delta_layout as layout
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)

__all__ = ("audit_event",)

GEOMETRY = ("workgroupCount", "workgroupSize")
DIGEST = re.compile(r"[0-9a-f]{64}")


def _check_outputs(event, entry):
    hashes = event["outputHashes"]
    require(
        isinstance(hashes, dict)
        and set(hashes) == set(layout.outputs(entry))
        and all(DIGEST.fullmatch(str(value)) for value in hashes.values()),
        "Gated-delta output digests are missing",
    )


def audit_event(event, cache):
    """Check one gated-delta dispatch against its plan and cached package."""
    entry = event["entry"]
    layout.signature(entry)
    execution = {key: event[key] for key in GEOMETRY}
    require("threadGridSize" not in event, "Gated-delta launch names a thread grid")
    counts, metadata = event["counts"], event["gatedDeltaMetadata"]
    require(
        isinstance(counts, dict) and isinstance(metadata, dict),
        "Gated-delta trace does not name its buffers",
    )
    require(
        layout.plan(entry, counts, metadata.get("steps"), event["threads"], execution)
        == metadata,
        "Gated-delta launch does not match the upstream plan",
    )
    require(event["dispatchVersion"] == 3, "Gated-delta dispatch version differs")
    hashes = event["inputHashes"]
    inputs = {
        name
        for name in layout.parameters(entry)
        if name not in layout.outputs(entry) and name not in layout.SCALARS
    }
    require(
        isinstance(hashes, dict)
        and set(hashes) == inputs
        and all(DIGEST.fullmatch(str(value)) for value in hashes.values()),
        "Gated-delta input digests are missing",
    )
    _check_outputs(event, entry)
    descriptor, directory = cache.get(entry)
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Gated-delta dispatch does not name its cached package",
    )
    request = event["details"]["request"]
    require(
        request["target"] == event["target"]
        and request["entryPoint"]
        == {"opengl": "main", "directx": "CSMain"}.get(
            event["target"], layout.source_entry(entry)
        )
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Gated-delta native request changed",
    )
    audit_native_execution(event)
    return True
