"""Artifact-owned physical launch obligations for exact-grid specializations."""

from __future__ import annotations

import json

from .artifact_contracts import (
    DISPATCH_REGION_PREFIX,
    artifact_contract_headers,
    with_artifact_contract_header,
)
from .codegen.stage_utils import compute_local_size
from .dispatch_regions import DispatchRegion


def codegen_dispatch_region(ast, entries):
    """Require the selected stage to retain the specialized physical shape."""
    record = (getattr(ast, "annotations", {}) or {}).get("dispatch_region")
    if record is None:
        return None
    region = DispatchRegion.from_json(record)
    if len(entries) != 1 or entries[0][0] != "compute":
        raise ValueError("A dispatch region requires one selected compute entry.")
    size = compute_local_size(entries[0][2])
    if size is None or tuple(str(n) for n in size) != tuple(
        str(n) for n in region.workgroup_size
    ):
        raise ValueError("Dispatch region differs from the compute workgroup size.")
    return region


def with_dispatch_region_header(source, region):
    if region is None:
        return source
    if not isinstance(region, DispatchRegion):
        raise TypeError("region must be a DispatchRegion")
    payload = {"schemaVersion": 1, "region": region.to_json()}
    header = DISPATCH_REGION_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    return with_artifact_contract_header(source, header)


def parse_dispatch_region_header(source):
    """Read one bounded, unambiguous region declaration from an artifact."""
    line = artifact_contract_headers(source).get(DISPATCH_REGION_PREFIX)
    if line is None:
        return None

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate dispatch region key.")
            result[key] = value
        return result

    try:
        record = json.loads(
            line[len(DISPATCH_REGION_PREFIX) :], object_pairs_hook=unique
        )
    except (ValueError, RecursionError) as exc:
        raise ValueError("Invalid dispatch region JSON.") from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"schemaVersion", "region"}
        or type(record["schemaVersion"]) is not int
        or record["schemaVersion"] != 1
    ):
        raise ValueError("Unsupported dispatch region header schema.")
    return DispatchRegion.from_json(record["region"])


def check_dispatch_region_geometry(region, dispatch):
    """Validate resolved physical geometry without trusting caller metadata."""
    if region is None:
        return
    if dispatch is None:
        raise ValueError("A dispatch region requires explicit physical geometry.")

    def dimensions(values):
        if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 3:
            return None
        if any(type(n) is not int or n <= 0 for n in values):
            return None
        return tuple(values) + (1,) * (3 - len(values))

    expected_global = tuple(
        n * w for n, w in zip(region.workgroup_count, region.workgroup_size)
    )
    if (
        dimensions(dispatch.workgroup_count) != region.workgroup_count
        or dimensions(dispatch.workgroup_size) != region.workgroup_size
        or dispatch.thread_grid_size
        or any(
            values and dimensions(values) != expected_global
            for values in (dispatch.global_size, dispatch.grid_size)
        )
    ):
        raise ValueError(
            "Dispatch geometry differs from the artifact's specialized physical region."
        )
