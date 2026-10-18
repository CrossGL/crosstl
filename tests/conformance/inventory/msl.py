"""Build the MSL inventory skeleton from the Metal Shading Language Specification, Version 4.1.

Apple publishes the specification only as a PDF at an unversioned URL, so the input is pinned by hash: the PDF
(sha256 below) converted with ``pdftotext -layout`` (poppler 26.10.0), whose text output is pinned as well. The
numbered section table is read from the specification's table of contents; every item of ``curated/metal.tsv``
is cited to a section by its exact title, and a title that is absent from the pinned text is an error.

This is a skeleton: it lists the language's constructs per category, with representative built-in functions and
built-in variable attributes. Deriving the complete standard library (every function and member function of
``metal_stdlib``) and the complete attribute tables is the next step; the Metal toolchain's headers recorded in
TOOLCHAINS.md are the machine-readable source for it.

Because the PDF URL is not versioned, CI validates the committed inventory but does not re-derive it.

    pdftotext -layout Metal-Shading-Language-Specification.pdf msl.txt
    python -m tests.conformance.inventory.msl --text msl.txt --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from . import DATA, InventoryError, curated, dumps, validate

PDF_URL = "https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf"
PDF_SHA256 = "41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5"
TEXT_SHA256 = "6b5e4c211e45ca42b1412fa4cb54029f1e414b15ec0b8dc35d12beef268ff8ee"
TOC_END = "Tables and Figures"
TOC_LINE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}\s*\d+)?\s*$")


def table_of_contents(text):
    sections = {}
    for line in text.splitlines():
        if line.strip() == TOC_END:
            break
        match = TOC_LINE.match(line)
        if match and not re.match(r"^\d+ of \d+$", match.group(2)):
            sections.setdefault(
                match.group(1), re.sub(r"\s+", " ", match.group(2)).strip()
            )
    return sections


def derive(text, cells=None):
    toc = table_of_contents(text)
    by_title = {}
    for number, title in toc.items():
        by_title.setdefault(title, number)
    items, cited = [], {}
    for row in curated("metal"):
        number = by_title.get(row["reference"])
        if number is None:
            raise InventoryError(
                f"section {row['reference']!r} does not exist in the pinned specification"
            )
        cited[number] = toc[number]
        item = {key: row[key] for key in ("id", "name", "category", "kind")}
        item["cite"] = f"msl-4.1 §{number}"
        if row["note"]:
            item["note"] = row["note"]
        items.append(item)
    items.sort(key=lambda item: item["id"])
    cells = dict(cells or {})
    orphaned = sorted(set(cells) - {item["id"] for item in items})
    if orphaned:
        raise InventoryError(f"classified items vanished: {', '.join(orphaned)}")
    return {
        "schemaVersion": 1,
        "language": "metal",
        "title": "Metal Shading Language 4.1 (skeleton)",
        "sources": {
            "msl-4.1": {
                "title": "Metal Shading Language Specification, Version 4.1",
                "url": PDF_URL,
                "version": "4.1",
                "sections": dict(
                    sorted(
                        cited.items(), key=lambda kv: [int(p) for p in kv[0].split(".")]
                    )
                ),
            }
        },
        "generated": {
            "by": "tests/conformance/inventory/msl.py",
            "pdfSha256": PDF_SHA256,
            "textSha256": TEXT_SHA256,
            "converter": "pdftotext -layout (poppler 26.10.0)",
            "scope": (
                "skeleton: categories with representative library sections; full standard library pending"
            ),
        },
        "items": items,
        "cells": cells,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--text",
        type=Path,
        required=True,
        help="pdftotext -layout output of the pinned PDF",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA / "metal.json")
    args = parser.parse_args(argv)
    data = args.text.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TEXT_SHA256:
        raise InventoryError(
            f"{args.text} has sha256 {digest}, not the pinned {TEXT_SHA256}"
        )
    existing = json.loads(args.output.read_text()) if args.output.is_file() else {}
    inventory = derive(data.decode(), existing.get("cells"))
    errors = validate(inventory, language="metal")
    if errors:
        raise InventoryError("\n".join(errors))
    text = dumps(inventory)
    if args.write:
        args.output.write_text(text)
        print(f"Wrote {args.output}: {len(inventory['items'])} items")
        return 0
    if args.output.is_file() and args.output.read_text() == text:
        print(
            f"{args.output} matches the pinned specification ({len(inventory['items'])} items)"
        )
        return 0
    print(f"{args.output} is out of date; regenerate with --write", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
