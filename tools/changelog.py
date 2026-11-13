#!/usr/bin/env python3
"""Keep unreleased changelog entries in their own files until a release is prepared.

A pull request adds ``changelog.d/<name>.<kind>.md`` with its entry, one or two sentences on what changed for people
who install CrossTL, where ``<kind>`` is one of ``KINDS``. Pull requests that all edit the top of ``CHANGELOG.md``
conflict with each other in the merge queue; new files never do.

``check`` validates the entries, and with ``--folded`` also fails while one is left, which the release workflow
uses on a tag. ``fold``, run when a release is prepared, moves the entries into the ``[Unreleased]`` section of
``CHANGELOG.md`` and deletes their files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = "changelog.d"
UNRELEASED = "## [Unreleased]"
# File name kinds and the headings they go under, in the order the headings appear in a release.
KINDS = {
    "breaking": "Breaking Changes",
    "added": "Added",
    "improved": "Improved",
    "fixed": "Fixed",
    "removed": "Removed",
}
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\.(" + "|".join(KINDS) + r")\.md")
_HEADING = re.compile(r"#+\s")


def entries(root: Path = ROOT) -> dict[str, list[str]]:
    """The entries in ``changelog.d`` by kind, in file name order. Raises ``ValueError`` for a file it cannot use."""
    found: dict[str, list[str]] = {kind: [] for kind in KINDS}
    directory = root / DIRECTORY
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        name = f"{DIRECTORY}/{path.name}"
        match = _NAME.fullmatch(path.name)
        if match is None or not path.is_file():
            kinds = ", ".join(KINDS)
            raise ValueError(
                f"{name}: name it <name>.<kind>.md in lower case, where <kind> is one of {kinds}"
            )
        found[match.group(1)] += _items(path.read_text(encoding="utf-8"), name)
    return found


def _items(text: str, name: str) -> list[str]:
    # Each line that starts with "- " begins an entry, other lines continue it, and the first needs no "- ".
    items: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _HEADING.match(line):
            raise ValueError(f"{name}: an entry is a sentence or two, without headings")
        if line.startswith("- ") or not items:
            items.append([line.removeprefix("- ").strip()])
        else:
            items[-1].append(line)
    if not items:
        raise ValueError(f"{name}: the file has no entry")
    return [" ".join(words) for words in items]


def fold(root: Path = ROOT) -> int:
    """Move the entries into the ``[Unreleased]`` section of ``CHANGELOG.md``; returns how many moved."""
    found = entries(root)
    count = sum(map(len, found.values()))
    if not count:
        return 0
    path = root / "CHANGELOG.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    if UNRELEASED not in lines:
        first = next(
            (i for i, line in enumerate(lines) if line.startswith("## [")), len(lines)
        )
        lines[first:first] = [UNRELEASED, "", "---", ""]
    start = lines.index(UNRELEASED) + 1
    end = next(
        (
            i
            for i in range(start, len(lines))
            if lines[i] == "---" or lines[i].startswith("## [")
        ),
        len(lines),
    )
    lines[start:end] = _section(lines[start:end], found)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    directory = root / DIRECTORY
    for entry in directory.iterdir():
        entry.unlink()
    directory.rmdir()
    return count


def _section(lines: list[str], found: dict[str, list[str]]) -> list[str]:
    preamble: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("### "):
            blocks.append((line[4:].strip(), []))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            preamble.append(line)
    order = list(KINDS.values())
    for kind, heading in KINDS.items():
        new = [f"- {item}" for item in found[kind]]
        if not new:
            continue
        body = next((body for name, body in blocks if name == heading), None)
        if body is not None:
            # The section lists the newest change first, and every entry here is newer than those already in it.
            body[:] = new + _trimmed(body)
            continue
        later = (
            i
            for i, (name, _) in enumerate(blocks)
            if name in order and order.index(name) > order.index(heading)
        )
        blocks.insert(next(later, len(blocks)), (heading, new))
    result = [""]
    if _trimmed(preamble):
        result += [*_trimmed(preamble), ""]
    for name, body in blocks:
        result += [f"### {name}", "", *_trimmed(body), ""]
    return result


def _trimmed(lines: list[str]) -> list[str]:
    start = next((i for i, line in enumerate(lines) if line.strip()), len(lines))
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help=f"validate the entries in {DIRECTORY}")
    check.add_argument(
        "--folded",
        action="store_true",
        help=f"also fail while {DIRECTORY} has an entry",
    )
    commands.add_parser(
        "fold", help="move the entries into the [Unreleased] section of CHANGELOG.md"
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fold":
            print(f"Moved {fold(root)} entries into CHANGELOG.md")
            return 0
        count = sum(map(len, entries(root).values()))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if arguments.folded and count:
        print(
            f"{DIRECTORY} has {count} entries that are not in CHANGELOG.md yet; "
            "run python tools/changelog.py fold",
            file=sys.stderr,
        )
        return 1
    print(f"{DIRECTORY} has {count} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
