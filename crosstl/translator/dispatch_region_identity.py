"""Content identity shared by the physical regions of one compute program."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping

# Source path -> ((mtime_ns, size, inode), sha256 of its text).
_SOURCE_DIGESTS: dict[str, tuple[tuple[int, int, int], str]] = {}
_SOURCE_DIGESTS_LOCK = threading.Lock()
# Coarser than the modification-time granularity of common file systems.
_SETTLED_SOURCE_NS = 2_000_000_000

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


def _source_files(directory: str, prefix: str):
    """Yield ``(relative path, entry)`` for every ``*.py`` file under ``directory``.

    Like ``Path.rglob``, the walk does not descend into symbolic links to
    directories.
    """
    with os.scandir(directory) as entries:
        children = sorted(entries, key=lambda entry: entry.name)
    for entry in children:
        if entry.is_dir(follow_symlinks=False):
            yield from _source_files(entry.path, prefix + entry.name + "/")
        elif entry.name.endswith(".py") and entry.is_file():
            yield prefix + entry.name, entry


def _source_digest(entry: os.DirEntry) -> str:
    status = entry.stat()
    signature = (status.st_mtime_ns, status.st_size, status.st_ino)
    with _SOURCE_DIGESTS_LOCK:
        cached = _SOURCE_DIGESTS.get(entry.path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    with open(entry.path, encoding="utf-8") as source:
        digest = hashlib.sha256(source.read().encode("utf-8")).hexdigest()
    # A file written within its timestamp's granularity of this read could
    # change again without a new signature, so its digest is not kept.
    if time.time_ns() - status.st_mtime_ns > _SETTLED_SOURCE_NS:
        with _SOURCE_DIGESTS_LOCK:
            _SOURCE_DIGESTS[entry.path] = (signature, digest)
    return digest


def source_tree_hash(root: Path) -> str:
    """Digest every ``*.py`` file's text under ``root`` by its relative path.

    Checked on each call, because a long-lived process may translate after
    source updates. A file is reread whenever its modification time, size or
    inode differs from the read its digest came from.
    """
    return _digest(
        {
            relative: _source_digest(entry)
            for relative, entry in _source_files(str(root), "")
        }
    )


def translation_implementation_hash() -> str:
    """Identify installed translation code without checkout paths or Git state."""
    return source_tree_hash(Path(__file__).resolve().parents[1])


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
