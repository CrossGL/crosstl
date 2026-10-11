"""Content identity shared by the physical regions of one compute program."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

_FIELDS = frozenset(
    (
        "schemaVersion",
        "sourceEntryPoint",
        "target",
        "intermediateHash",
        "implementationHash",
        "settingsHash",
        "hash",
    )
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def translation_implementation_hash() -> str:
    """Identify installed translation code without checkout paths or Git state."""
    root = Path(__file__).resolve().parents[1]
    # Read each time: a long-lived process may translate after source updates.
    return _digest(
        {
            path.relative_to(root).as_posix(): (
                hashlib.sha256(
                    path.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest()
            )
            for path in sorted(root.rglob("*.py"))
        }
    )


def build_dispatch_region_program(
    *,
    intermediate_hash: str,
    entry_point: str,
    target: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a selected source entry and lowering policies, excluding region geometry."""
    payload = {
        "schemaVersion": 1,
        "sourceEntryPoint": entry_point,
        "target": target,
        "intermediateHash": intermediate_hash,
        "implementationHash": translation_implementation_hash(),
        "settingsHash": _digest(settings),
    }
    payload["hash"] = _digest(payload)
    validate_dispatch_region_program(payload, target=target)
    return payload


def validate_dispatch_region_program(value: Any, *, target: str) -> None:
    """Reject incomplete or internally inconsistent program identity records."""
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError(
            "dispatchRegionProgram must contain exactly its identity fields"
        )
    if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1:
        raise ValueError("dispatchRegionProgram schemaVersion must be 1")
    if value["target"] != target or target not in {"directx", "opengl"}:
        raise ValueError("dispatchRegionProgram target must match the artifact")
    entry = value["sourceEntryPoint"]
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("dispatchRegionProgram sourceEntryPoint must be nonempty")
    for key in ("intermediateHash", "implementationHash", "settingsHash", "hash"):
        if not isinstance(value[key], str) or not re.fullmatch(
            r"[0-9a-f]{64}", value[key]
        ):
            raise ValueError(f"dispatchRegionProgram {key} must be a SHA-256 digest")
    if value["hash"] != _digest(
        {key: item for key, item in value.items() if key != "hash"}
    ):
        raise ValueError(
            "dispatchRegionProgram hash does not match its identity fields"
        )
