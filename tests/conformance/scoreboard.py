"""Per-cell regression comparison; missing or unavailable proof cannot pass.

A scoreboard maps each cell key (case, language, target, configuration, route, Vulkan environment, runtime) to
its outcome and the hashes of the source and input data it was measured on. A run is checked against the cells of
the baseline that fall inside the run's declared scope; a cell in scope that is missing, measured on different
data, or worse than recorded is a failure. Better outcomes pass and are reported so the baseline can be advanced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SCHEMA = 2
KEYS = ("case", "language", "target", "config", "route", "environment", "runtime")
IMPROVEMENTS = {
    "translator-error": {
        "refused",
        "invalid",
        "interface-changed",
        "diverged",
        "preserved",
    },
    "invalid": {"refused", "diverged", "preserved"},
    "interface-changed": {"refused", "diverged", "preserved"},
    "diverged": {"refused", "preserved"},
    "memory-corruption": {"refused", "preserved"},
    "driver-rejected": {"refused", "diverged", "preserved"},
    "timeout": {"refused", "diverged", "preserved"},
    "runtime-error": {"refused", "diverged", "preserved"},
    "refused": {"preserved"},
}


def key(record):
    return "|".join(str(record[name]) for name in KEYS)


def snapshot(records):
    cells = {}
    for record in records:
        identity = key(record)
        if identity in cells:
            raise ValueError(f"Duplicate conformance result: {identity}")
        cells[identity] = {
            name: record[name] for name in ("outcome", "sourceSHA256", "inputSHA256")
        }
    counts = dict(sorted(Counter(record["outcome"] for record in records).items()))
    return {
        "schemaVersion": SCHEMA,
        "keys": list(KEYS),
        "cells": dict(sorted(cells.items())),
        "counts": counts,
    }


def in_scope(identity, scope):
    if scope is None:
        return True
    values = dict(zip(KEYS, identity.split("|")))
    return all(values[name] in allowed for name, allowed in scope.items())


def check(baseline, records, scope=None):
    """Return failure messages; ``scope`` maps key fields to the values this run claims to cover."""
    if baseline.get("schemaVersion") != SCHEMA or not baseline.get("cells"):
        raise ValueError(f"A nonempty version {SCHEMA} scoreboard is required")
    if scope is not None and set(scope) - set(KEYS):
        raise ValueError(f"Unknown scope fields: {sorted(set(scope) - set(KEYS))}")
    current = snapshot(records)["cells"]
    failures = []
    covered = [identity for identity in baseline["cells"] if in_scope(identity, scope)]
    if not covered:
        failures.append("The baseline has no cells in this run's scope")
    for identity in covered:
        expected, actual = baseline["cells"][identity], current.get(identity)
        if actual is None:
            failures.append(f"Missing conformance cell: {identity}")
            continue
        for name in ("sourceSHA256", "inputSHA256"):
            if actual[name] != expected[name]:
                failures.append(f"Changed {name}: {identity}")
        if actual["outcome"] != expected["outcome"] and actual[
            "outcome"
        ] not in IMPROVEMENTS.get(expected["outcome"], set()):
            failures.append(
                f"Regression {identity}: {expected['outcome']} -> {actual['outcome']}"
            )
    return failures


def improvements(baseline, records):
    current = snapshot(records)["cells"]
    return [
        f"Improved {identity}: {expected['outcome']} -> {current[identity]['outcome']}"
        for identity, expected in baseline["cells"].items()
        if identity in current
        and current[identity]["outcome"] != expected["outcome"]
        and current[identity]["outcome"] in IMPROVEMENTS.get(expected["outcome"], set())
    ]


def merge(baseline, records, scope):
    """Replace the baseline cells inside ``scope`` with this run's cells and keep every other cell."""
    kept = {
        identity: cell
        for identity, cell in baseline.get("cells", {}).items()
        if not in_scope(identity, scope)
    }
    fresh = snapshot(records)["cells"]
    if set(kept) & set(fresh):
        raise ValueError("The run produced cells outside its declared scope")
    cells = dict(sorted({**kept, **fresh}.items()))
    counts = dict(sorted(Counter(cell["outcome"] for cell in cells.values()).items()))
    return {
        "schemaVersion": SCHEMA,
        "keys": list(KEYS),
        "cells": cells,
        "counts": counts,
    }


def read_results(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="+", help="results.jsonl files of one or more runs"
    )
    parser.add_argument(
        "--write", type=Path, required=True, help="Scoreboard to create or update"
    )
    args = parser.parse_args(argv)
    records = [record for path in args.results for record in read_results(path)]
    scope = {
        name: {str(record[name]) for record in records}
        for name in ("runtime", "target", "case")
    }
    baseline = json.loads(args.write.read_text()) if args.write.is_file() else {}
    args.write.write_text(
        json.dumps(merge(baseline, records, scope), indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
