"""Report inventory coverage: items per category and cell states per target.

python -m tests.conformance.inventory                     # text report of all four inventories
python -m tests.conformance.inventory --json              # the same counts as JSON
python -m tests.conformance.inventory --require-complete  # exit 1 while any cell is unknown
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from . import load_all


def summary(inventories):
    return {
        language: {
            "items": len(inventory.items),
            "categories": dict(
                sorted(Counter(item["category"] for item in inventory.items).items())
            ),
            "cells": inventory.counts(),
        }
        for language, inventory in inventories.items()
    }


def text(report):
    lines = []
    for language, entry in report.items():
        categories = ", ".join(
            f"{name} {count}" for name, count in entry["categories"].items()
        )
        lines.append(f"{language}: {entry['items']} items ({categories})")
        for target, counts in entry["cells"].items():
            states = ", ".join(f"{state} {count}" for state, count in counts.items())
            lines.append(f"  -> {target}: {states}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    inventories = load_all()
    report = summary(inventories)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else text(report))
    unknown = sum(inventory.unknown() for inventory in inventories.values())
    if args.require_complete and unknown:
        print(f"{unknown} cells are unknown", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
