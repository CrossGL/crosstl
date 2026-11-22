#!/usr/bin/env python3
"""Compare two input-hash manifests and name the keys whose inputs changed.

An expensive CI job can record one SHA-256 per unit of work, such as a leg, a shard or a test group, over everything
that unit reads. A later run computes the same hashes and runs only the units whose hash differs from the manifest
of the newest run that passed. This module does that comparison. The producer of a manifest decides what goes into
each hash, and the keys are opaque strings to it.

A manifest is a JSON object: either a bare ``{key: sha256}`` map, or ``{"schema": ..., "toolchain": ...,
"entries": {key: sha256}}``. The comparison fails safe. A missing or unreadable baseline, or a baseline whose
``schema`` or ``toolchain`` differs from the current manifest's, selects every key of the current manifest. A
malformed current manifest is an error, because the run that wrote it is the one being planned.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_DIGEST = re.compile(r"[0-9a-f]{64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# The fields a structured manifest may carry besides its entries; anything else is rejected as a likely typo.
_FIELDS = {"schema", "toolchain", "entries"}


@dataclass(frozen=True)
class Manifest:
    """The entries of one manifest, with the schema and toolchain that must match for its hashes to compare."""

    entries: dict[str, str]
    schema: object = None
    toolchain: object = None


@dataclass(frozen=True)
class Comparison:
    """Keys of the current manifest by how they compare with the baseline, each sorted."""

    added: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    # Why every current key was selected without comparing hashes, or None when the hashes were compared.
    reason: str | None = None
    selected: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", tuple(sorted(self.added + self.changed)))

    def as_json(self) -> dict[str, object]:
        return {
            "selected": list(self.selected),
            "added": list(self.added),
            "changed": list(self.changed),
            "removed": list(self.removed),
            "unchanged": len(self.unchanged),
            "reason": self.reason,
        }


def parse(data: object) -> Manifest:
    """The manifest in decoded JSON ``data``. Raises ``ValueError`` naming the first problem."""
    if not isinstance(data, dict):
        raise ValueError("a manifest is a JSON object")
    if "entries" in data:
        unknown = sorted(set(data) - _FIELDS)
        if unknown:
            raise ValueError(f"unknown manifest fields: {', '.join(unknown)}")
        entries, schema, toolchain = (
            data["entries"],
            data.get("schema"),
            data.get("toolchain"),
        )
        if not isinstance(entries, dict):
            raise ValueError("'entries' is a JSON object")
    else:
        entries, schema, toolchain = data, None, None
    for key, digest in entries.items():
        if not key or key != key.strip() or _CONTROL.search(key):
            raise ValueError(
                f"key {key!r} is empty, has surrounding space or a control character"
            )
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError(
                f"{key}: {digest!r} is not a lowercase hexadecimal SHA-256"
            )
    return Manifest(dict(entries), schema, toolchain)


def load(path: Path) -> Manifest:
    """The manifest in the JSON file at ``path``. Raises ``OSError`` or ``ValueError``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not JSON: {error}") from error
    try:
        return parse(data)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _select_all(
    baseline: Manifest | None, current: Manifest, reason: str
) -> Comparison:
    before = baseline.entries if baseline else {}
    return Comparison(
        added=tuple(sorted(key for key in current.entries if key not in before)),
        changed=tuple(sorted(key for key in current.entries if key in before)),
        removed=tuple(sorted(key for key in before if key not in current.entries)),
        reason=reason,
    )


def compare(baseline: Manifest | None, current: Manifest) -> Comparison:
    """How ``current`` differs from ``baseline``; ``None`` stands for a baseline that could not be read."""
    if baseline is None:
        return _select_all(None, current, "no baseline manifest")
    if baseline.schema != current.schema:
        return _select_all(baseline, current, "the manifest schema differs")
    if baseline.toolchain != current.toolchain:
        return _select_all(baseline, current, "the toolchain differs")
    before, after = baseline.entries, current.entries
    return Comparison(
        added=tuple(sorted(key for key in after if key not in before)),
        changed=tuple(
            sorted(key for key in after if key in before and before[key] != after[key])
        ),
        unchanged=tuple(
            sorted(key for key in after if key in before and before[key] == after[key])
        ),
        removed=tuple(sorted(key for key in before if key not in after)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.ci_manifest", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "baseline",
        type=Path,
        help="manifest of the newest passing run; a missing or unreadable file selects every key",
    )
    parser.add_argument("current", type=Path, help="manifest of this run")
    parser.add_argument(
        "--format",
        choices=("json", "lines"),
        default="json",
        help="json: the full comparison; lines: one selected key per line (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        current = load(args.current)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        baseline: Manifest | None = load(args.baseline)
    except FileNotFoundError:
        baseline = None
    except (OSError, ValueError) as error:
        print(
            f"warning: ignoring the baseline, every key is selected: {error}",
            file=sys.stderr,
        )
        baseline = None
    result = compare(baseline, current)
    if args.format == "lines":
        for key in result.selected:
            print(key)
    else:
        print(json.dumps(result.as_json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
