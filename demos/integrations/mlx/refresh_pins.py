#!/usr/bin/env python3
"""Refresh MLX generated-artifact pins from report-mode drift records.

Report pin mode (``tests/pins.py``) lets merge-queue, push and nightly runs go on
when a translator change on ``main`` alters the bytes of a pinned MLX artifact. Each
changed pin is printed as one ``MLX-PIN-DRIFT`` JSON line. This tool reads those
lines from job logs, drift directories or a whole workflow run, and rewrites the
pins under ``demos/integrations/mlx``:

* every drifted SHA-256, wherever it is recorded: contracts, tests, harness tables,
  ``expected-gaps.json`` and the README;
* the size recorded with it: a ``sizeBytes``, ``size_bytes`` or
  ``generatedSizeBytes`` field of the same object, a compact ``"<sha>:<size>"``
  value, or an ``N-byte``/``N bytes`` phrase in the same Markdown paragraph;
* contract summaries derived from the contract's ``entries``
  (``generatedSizeBytesTotal`` and ``generatedSizeRange``), together with their
  mirrors in the tests and the README that name the contract;
* the SHA-256 and size of every file the refresh changes, wherever they are pinned.

A value changes only where its old value is found exactly. An ambiguous or missing
size is reported and left alone, so the exit status is non-zero until every drift
record is accounted for. The tool does not judge the new bytes: take the drift from
a run whose translation, compilation and numerical checks passed, and review the
refreshed diff. Recorded native identities such as ``dxilSha256`` are not
recomputed; entries that keep one next to a refreshed hash are listed in the
summary.

Usage::

    python -m demos.integrations.mlx.refresh_pins --run 40862351526 \\
        --save-records drift.jsonl
    python -m demos.integrations.mlx.refresh_pins drift.jsonl --write
"""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCOPE = Path("demos/integrations/mlx")
SCANNED_SUFFIXES = frozenset((".json", ".md", ".py", ".toml"))
DRIFT_PREFIX = "MLX-PIN-DRIFT "
HASH_FIELD = "generatedHash"
SIZE_FIELD = "generatedSizeBytes"
SIZE_KEYS = ("sizeBytes", "size_bytes", "generatedSizeBytes")
NATIVE_IDENTITY_KEYS = ("dxilSha256",)
TOTAL_KEY = "generatedSizeBytesTotal"
RANGE_KEY = "generatedSizeRange"
MAX_CASCADE_ROUNDS = 10

_DRIFT_LINE = re.compile(r"MLX-PIN-DRIFT (\S.*?)\s*$")
_SHA256 = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_SIZE_FIELD = re.compile(
    r"(?<![\w])(?P<q>[\"']?)(?:" + "|".join(SIZE_KEYS) + r")(?P=q)\s*[:=]\s*"
    r"(?P<value>\d+)(?![\d.])"
)
_COMPACT_SIZE = re.compile(r":(\d+)(?![\d.])")
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
# A size in prose, alone or in a list: "175,436-byte", "7,135 and 7,197 bytes".
_PROSE_SIZE = re.compile(
    rf"(?<![\d,.])(?P<value>{_NUMBER})"
    rf"(?=(?:(?:,\s+|,?\s+and\s+){_NUMBER})*(?:-byte\b|\s+bytes?\b))"
)
# Python statements that pin a hash and its size separately.
_ASSIGNED_HASH = re.compile(
    r"(?<![\w])(?P<prefix>[A-Z][A-Z0-9_]*_)SHA256\s*=\s*\(?\s*$"
)
_SUBSCRIPTED_HASH = re.compile(
    r"(?P<subject>[\w.]+(?:\[[^\[\]\n]+\])*)\[(?P<q>[\"'])sha256(?P=q)\]\s*==\s*\(?\s*$"
)
SIBLING_WINDOW = 600
_TOKENS = {
    False: re.compile(r'"(?:[^"\\\n]|\\.)*"|[()\[\]{}]'),
    True: re.compile(
        r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:[^"\\\n]|\\.)*"'
        r"|'(?:[^'\\\n]|\\.)*'|#[^\n]*|[()\[\]{}]"
    ),
}


class RefreshError(ValueError):
    """Drift records contradict each other or cannot be read."""


@dataclass(frozen=True)
class Pin:
    """One drifted artifact: its pinned and its current SHA-256 and size."""

    expected_sha256: str
    actual_sha256: str
    expected_size: int | None = None
    actual_size: int | None = None


@dataclass
class Report:
    """What a refresh changed and what it could not resolve."""

    records: int = 0
    pins: int = 0
    applied: Counter = field(default_factory=Counter)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    native_identities: Counter = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "pins": self.pins,
            "applied": dict(sorted(self.applied.items())),
            "files": dict(sorted(self.files.items())),
            "unresolved": self.unresolved,
            "unmatchedPins": sorted(self.unmatched),
            "staleNativeIdentities": dict(sorted(self.native_identities.items())),
        }

    @property
    def complete(self) -> bool:
        return not self.unresolved and not self.unmatched


# ---------------------------------------------------------------------------
# Drift records


def parse_drift_text(text: str, *, prefixed_only: bool = False) -> list[dict[str, Any]]:
    """Return the drift records in a job log or a ``drift-*.jsonl`` file.

    Log lines carry an ``MLX-PIN-DRIFT`` prefix after the runner's timestamp; the
    files ``tests/pins.py`` writes hold one bare JSON record per line.
    """

    records = []
    for line in text.splitlines():
        match = _DRIFT_LINE.search(line)
        if match and match.group(1).startswith("{"):
            try:
                record = json.loads(match.group(1))
            except json.JSONDecodeError as error:
                raise RefreshError(f"Unreadable drift record: {line[:200]}") from error
        elif not prefixed_only and line.lstrip().startswith("{"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
        else:
            continue
        if isinstance(record, dict) and {"field", "expected", "actual"} <= set(record):
            records.append(record)
    return records


def load_drift_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read drift records from files and, recursively, from directories."""

    records = []
    for path in paths:
        files = (
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
        for file in files:
            records.extend(
                parse_drift_text(file.read_text(encoding="utf-8", errors="replace"))
            )
    return records


def _gh(arguments: Sequence[str], attempts: int = 3) -> str:
    for attempt in range(attempts):
        result = subprocess.run(
            ["gh", "api", *arguments], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return result.stdout
        time.sleep(2 * (attempt + 1))
    raise RefreshError(f"gh api {' '.join(arguments)} failed: {result.stderr[-300:]}")


def fetch_run_records(repo: str, run_id: int, workers: int = 6) -> list[dict[str, Any]]:
    """Collect the drift records of every successful job of a workflow run."""

    jobs = []
    for page in range(1, 100):
        listing = json.loads(
            _gh(
                [
                    f"repos/{repo}/actions/runs/{run_id}/jobs"
                    f"?per_page=100&filter=latest&page={page}"
                ]
            )
        )
        batch = listing.get("jobs", [])
        jobs.extend(job["id"] for job in batch if job.get("conclusion") == "success")
        if len(batch) < 100:
            break

    def job_records(job_id: int) -> list[dict[str, Any]]:
        log = _gh(
            ["--allow-escape-sequences", f"repos/{repo}/actions/jobs/{job_id}/logs"]
        )
        return parse_drift_text(log, prefixed_only=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [record for batch in pool.map(job_records, jobs) for record in batch]


def _merge_pin(existing: Pin | None, pin: Pin) -> Pin:
    if existing is None:
        return pin
    if existing.actual_sha256 != pin.actual_sha256:
        raise RefreshError(
            f"Pin {pin.expected_sha256} drifted to both "
            f"{existing.actual_sha256} and {pin.actual_sha256}"
        )
    sizes = {
        (p.expected_size, p.actual_size)
        for p in (existing, pin)
        if p.actual_size is not None
    }
    if len(sizes) > 1:
        raise RefreshError(
            f"Pin {pin.expected_sha256} drifted to different sizes: {sizes}"
        )
    if not sizes:
        return existing
    ((expected_size, actual_size),) = sizes
    return Pin(pin.expected_sha256, pin.actual_sha256, expected_size, actual_size)


def build_pins(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Pin], list[dict]]:
    """Pair hash and size drift per pinned subject and key them by old SHA-256.

    Returns the pins and the size records without a hash record, which cannot be
    located on their own.
    """

    groups: dict[tuple[str, str], dict[str, tuple[Any, Any]]] = defaultdict(dict)
    for record in records:
        name = record["field"]
        if name not in (HASH_FIELD, SIZE_FIELD):
            raise RefreshError(f"Unknown drift field {name!r}")
        key = (str(record.get("subject", "")), str(record.get("test", "")))
        value = (record["expected"], record["actual"])
        if groups[key].get(name, value) != value:
            raise RefreshError(f"Conflicting {name} drift for {key[0]}")
        groups[key][name] = value
    pins: dict[str, Pin] = {}
    size_only = []
    for (subject, test), fields in sorted(groups.items()):
        digest = fields.get(HASH_FIELD)
        size = fields.get(SIZE_FIELD)
        if digest is None:
            size_only.append({"subject": subject, "test": test, "size": size})
            continue
        for value in digest:
            if not (isinstance(value, str) and _SHA256.fullmatch(value)):
                raise RefreshError(f"{subject}: {value!r} is not a SHA-256")
        if size is not None and not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in size
        ):
            raise RefreshError(f"{subject}: sizes {size!r} are not positive integers")
        pin = Pin(digest[0], digest[1], *(size or (None, None)))
        pins[pin.expected_sha256] = _merge_pin(pins.get(pin.expected_sha256), pin)
    return pins, size_only


# ---------------------------------------------------------------------------
# Locating pins in text


@dataclass
class _Structure:
    """String tokens and bracket intervals of JSON or Python source text."""

    string_starts: list[int]
    strings: list[tuple[int, int, int]]
    intervals: list[list[int]]
    children: dict[int, list[int]]

    def string_at(self, position: int) -> tuple[int, int, int] | None:
        index = bisect.bisect_right(self.string_starts, position) - 1
        if index >= 0 and self.strings[index][1] > position:
            return self.strings[index]
        return None


def _structure(text: str, python: bool) -> _Structure:
    strings: list[tuple[int, int, int]] = []
    intervals: list[list[int]] = []
    children: dict[int, list[int]] = defaultdict(list)
    stack: list[int] = []
    for token in _TOKENS[python].finditer(text):
        value = token.group()
        parent = stack[-1] if stack else -1
        if value[0] in "\"'":
            strings.append((token.start(), token.end(), parent))
        elif value in "([{":
            intervals.append([token.start(), len(text) - 1, parent])
            children[parent].append(len(intervals) - 1)
            stack.append(len(intervals) - 1)
        elif value in ")]}" and stack:
            intervals[stack.pop()][1] = token.start()
    return _Structure([s[0] for s in strings], strings, intervals, children)


def _object_span(text: str, structure: _Structure, string: tuple[int, int, int]):
    """Return the bracket interval that owns a string token, or ``None``."""

    index = string[2]
    if index < 0:
        return None
    start, end, parent = structure.intervals[index]
    if text[start + 1 : end].strip().rstrip(",").strip() == text[string[0] : string[1]]:
        # A parenthesized string such as ``"sha256": ("<digest>")``.
        if parent < 0:
            return None
        index = parent
    return index


def _direct_size_fields(text, structure, index, expected):
    start, end, _ = structure.intervals[index]
    nested = [
        structure.intervals[child][:2] for child in structure.children.get(index, ())
    ]
    found = []
    for match in _SIZE_FIELD.finditer(text, start, end):
        if int(match.group("value")) != expected:
            continue
        if any(a < match.start() < b for a, b in nested):
            continue
        found.append(match.span("value"))
    return found


def _python_sibling_sizes(text: str, string_start: int, expected: int):
    """Sizes pinned by the statement that sits beside a Python hash statement.

    Handles ``NAME_SHA256 = "<digest>"`` with ``NAME_SIZE_BYTES = <size>``, and
    ``subject["sha256"] == "<digest>"`` with ``subject["size_bytes"] == <size>``.
    """

    line_start = text.rfind("\n", 0, string_start) + 1
    before = text[max(0, line_start - 200) : string_start]
    assigned = _ASSIGNED_HASH.search(before)
    if assigned:
        pattern = re.compile(
            rf"^{re.escape(assigned.group('prefix'))}SIZE_BYTES\s*=\s*(\d+)\b", re.M
        )
        return [
            m.span(1) for m in pattern.finditer(text) if int(m.group(1)) == expected
        ]
    subscripted = _SUBSCRIPTED_HASH.search(before)
    if subscripted:
        keys = "|".join(SIZE_KEYS)
        pattern = re.compile(
            re.escape(subscripted.group("subject"))
            + rf"\[([\"'])(?:{keys})\1\]\s*==\s*(\d+)\b"
        )
        window = text[string_start : string_start + SIBLING_WINDOW]
        return [
            (string_start + m.start(2), string_start + m.end(2))
            for m in pattern.finditer(window)
            if int(m.group(2)) == expected
        ]
    return []


def _paragraph_sizes(text: str, position: int, expected: int):
    start = text.rfind("\n\n", 0, position)
    end = text.find("\n\n", position)
    start = 0 if start < 0 else start
    end = len(text) if end < 0 else end
    return [
        match.span("value")
        for match in _PROSE_SIZE.finditer(text, start, end)
        if int(match.group("value").replace(",", "")) == expected
    ]


def _format_like(original: str, value: int, *, prose: bool = False) -> str:
    """Write ``value`` the way ``original`` was written; prose groups thousands."""

    grouped = "," in original or (prose and len(original) < 4)
    return f"{value:,}" if grouped else str(value)


def _line(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def leaf_edits(path: str, text: str, pins: Mapping[str, Pin], report: Report):
    """Return ``(start, end, replacement)`` edits for the drifted pins in ``text``."""

    edits = []
    occurrences = [m for m in _SHA256.finditer(text) if m.group() in pins]
    if not occurrences:
        return edits, set()
    markdown = path.endswith(".md")
    structure = None if markdown else _structure(text, path.endswith(".py"))
    for match in occurrences:
        pin = pins[match.group()]
        edits.append((match.start(), match.end(), pin.actual_sha256))
        report.applied["sha256"] += 1
        index = string = None
        if structure is not None:
            string = structure.string_at(match.start())
            index = _object_span(text, structure, string) if string else None
            if index is not None:
                _note_native_identities(text, structure, index, path, report)
        if pin.actual_size is None or pin.actual_size == pin.expected_size:
            continue
        spans = []
        compact = _COMPACT_SIZE.match(text, match.end())
        if compact and int(compact.group(1)) == pin.expected_size:
            spans = [compact.span(1)]
        elif markdown:
            spans = _paragraph_sizes(text, match.start(), pin.expected_size)
        elif index is not None:
            spans = _direct_size_fields(text, structure, index, pin.expected_size)
        if not spans and string is not None and path.endswith(".py"):
            spans = _python_sibling_sizes(text, string[0], pin.expected_size)
        if len(spans) != 1:
            report.unresolved.append(
                {
                    "path": path,
                    "line": _line(text, match.start()),
                    "sha256": pin.expected_sha256,
                    "reason": (
                        f"found {len(spans)} size fields equal to {pin.expected_size}"
                    ),
                }
            )
            continue
        ((start, end),) = spans
        edits.append(
            (start, end, _format_like(text[start:end], pin.actual_size, prose=markdown))
        )
        report.applied["size"] += 1
    return edits, {m.group() for m in occurrences}


def _note_native_identities(text, structure, index, path, report):
    start, end, _ = structure.intervals[index]
    for key in NATIVE_IDENTITY_KEYS:
        if f'"{key}"' in text[start:end]:
            report.native_identities[f"{path}:{key}"] += 1


def apply_edits(text: str, edits: Sequence[tuple[int, int, str]]) -> str:
    """Apply non-overlapping edits; identical duplicates are applied once."""

    ordered = sorted(set(edits))
    for (start, end, value), (next_start, next_end, next_value) in zip(
        ordered, ordered[1:]
    ):
        if next_start < end:
            raise RefreshError(
                f"Overlapping edits at {start}-{end} ({value!r}) and "
                f"{next_start}-{next_end} ({next_value!r})"
            )
    parts = []
    cursor = 0
    for start, end, value in ordered:
        parts.append(text[cursor:start])
        parts.append(value)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Derived contract summaries


def _entries(contract: Any) -> list[dict] | None:
    entries = contract.get("entries") if isinstance(contract, dict) else None
    if not isinstance(entries, list) or not entries:
        return None
    if not all(
        isinstance(entry, dict)
        and isinstance(entry.get("sizeBytes"), int)
        and isinstance(entry.get("entryPoint"), str)
        for entry in entries
    ):
        return None
    return entries


def size_summary(entries: Sequence[Mapping[str, Any]]) -> tuple[int, dict]:
    """Return the total and the range the complete-family tests derive from entries."""

    minimum = min(entries, key=lambda entry: (entry["sizeBytes"], entry["entryPoint"]))
    maximum = max(entries, key=lambda entry: (entry["sizeBytes"], entry["entryPoint"]))
    return sum(entry["sizeBytes"] for entry in entries), {
        "minimum": {
            "entryPoint": minimum["entryPoint"],
            "sizeBytes": minimum["sizeBytes"],
        },
        "maximum": {
            "entryPoint": maximum["entryPoint"],
            "sizeBytes": maximum["sizeBytes"],
        },
    }


def _replace_summaries(value: Any, old: tuple, new: tuple, counts: Counter) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == TOTAL_KEY and item == old[0]:
                value[key] = new[0]
                counts["totals"] += 1
            elif key == RANGE_KEY and item == old[1]:
                value[key] = json.loads(json.dumps(new[1]))
                counts["ranges"] += 1
            else:
                _replace_summaries(item, old, new, counts)
    elif isinstance(value, list):
        for item in value:
            _replace_summaries(item, old, new, counts)


def _dump_like(original: str, value: Any) -> str | None:
    """Serialize ``value`` in the layout of ``original`` if that layout is the
    standard two-space one; return ``None`` for hand-laid-out JSON."""

    try:
        if json.dumps(json.loads(original), indent=2) + "\n" != original:
            return None
    except json.JSONDecodeError:
        return None
    return json.dumps(value, indent=2) + "\n"


def refresh_summaries(path: str, old_text: str, new_text: str, report: Report):
    """Recompute entry-derived totals and ranges; return text and mirror changes."""

    if not path.endswith(".json") or old_text == new_text:
        return new_text, None
    old_entries = _entries(json.loads(old_text))
    contract = json.loads(new_text)
    new_entries = _entries(contract)
    if old_entries is None or new_entries is None:
        return new_text, None
    old, new = size_summary(old_entries), size_summary(new_entries)
    if old == new:
        return new_text, None
    counts: Counter = Counter()
    _replace_summaries(contract, old, new, counts)
    if not counts:
        return new_text, (old, new)
    dumped = _dump_like(old_text, contract)
    if dumped is None:
        report.unresolved.append(
            {"path": path, "reason": "derived size summary in hand-laid-out JSON"}
        )
        return new_text, (old, new)
    report.applied.update(counts)
    return dumped, (old, new)


def _range_mirrors(text: str, before: Mapping[str, Any]):
    """Matches of one range bound restated as a JSON object in ``text``."""

    entry = re.escape(json.dumps(before["entryPoint"]))
    pattern = (
        rf"\"entryPoint\":\s*{entry},(?P<gap>\s*)\"sizeBytes\":\s*"
        rf"{before['sizeBytes']}(?![\d.])"
    )
    return list(re.finditer(pattern, text))


def _python_literal_spans(text: str) -> tuple[ast.AST, list[int]]:
    tree = ast.parse(text)
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return tree, starts


def _span(node: ast.AST, starts: Sequence[int]) -> tuple[int, int]:
    return (
        starts[node.lineno - 1] + node.col_offset,
        starts[node.end_lineno - 1] + node.end_col_offset,
    )


def _constant(node: ast.AST, kind: type) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) is kind:
        return node.value
    return None


def _python_mirror_edits(text: str, old: tuple, new: tuple, report: Report):
    """Python literals that restate a contract total or range bound."""

    tree, starts = _python_literal_spans(text)
    bounds = [
        (old[1][bound], new[1][bound])
        for bound in ("minimum", "maximum")
        if old[1][bound] != new[1][bound]
    ]
    edits = []
    for node in ast.walk(tree):
        if old[0] != new[0] and _constant(node, int) == old[0]:
            edits.append((*_span(node, starts), str(new[0])))
            report.applied["mirroredTotals"] += 1
        for before, after in bounds:
            pair = (before["entryPoint"], before["sizeBytes"])
            if isinstance(node, ast.Tuple) and len(node.elts) == 2:
                values = tuple(
                    (
                        _constant(element, str)
                        if _constant(element, str) is not None
                        else _constant(element, int)
                    )
                    for element in node.elts
                )
                if values in (pair, pair[::-1]):
                    entry, size = json.dumps(after["entryPoint"]), after["sizeBytes"]
                    literal = (
                        f"({entry}, {size})" if values == pair else f"({size}, {entry})"
                    )
                    edits.append((*_span(node, starts), literal))
                    report.applied["mirroredRanges"] += 1
            elif isinstance(node, ast.Dict):
                keys = [_constant(key, str) if key else None for key in node.keys]
                if "entryPoint" not in keys or "sizeBytes" not in keys:
                    continue
                entry = node.values[keys.index("entryPoint")]
                size = node.values[keys.index("sizeBytes")]
                if (_constant(entry, str), _constant(size, int)) == pair:
                    edits.append(
                        (*_span(entry, starts), json.dumps(after["entryPoint"]))
                    )
                    edits.append((*_span(size, starts), str(after["sizeBytes"])))
                    report.applied["mirroredRanges"] += 1
    return edits


def mirror_edits(path: str, text: str, old: tuple, new: tuple, report: Report):
    """Edits for a test, harness table or README that restates a contract's
    derived size summary.

    Every copy restates the same contract value, so all copies change. Python is
    matched by literal value; JSON by an integer total and ``entryPoint``/
    ``sizeBytes`` bounds; Markdown by the comma-grouped total.
    """

    if path.endswith(".py"):
        return _python_mirror_edits(text, old, new, report)
    edits = []
    if old[0] != new[0]:
        if path.endswith(".md"):
            pattern = re.compile(rf"(?<![\d,.]){re.escape(f'{old[0]:,}')}(?![\d,])")
            spans = [m.span() for m in pattern.finditer(text)]
        else:
            spans = _integer_literals(text, old[0])
        for start, end in spans:
            edits.append(
                (
                    start,
                    end,
                    _format_like(text[start:end], new[0], prose=path.endswith(".md")),
                )
            )
            report.applied["mirroredTotals"] += 1
    if not path.endswith(".json"):
        return edits
    for bound in ("minimum", "maximum"):
        before, after = old[1][bound], new[1][bound]
        if before == after:
            continue
        for match in _range_mirrors(text, before):
            replacement = (
                f'"entryPoint": {json.dumps(after["entryPoint"])},'
                f'{match.group("gap")}"sizeBytes": {after["sizeBytes"]}'
            )
            edits.append((*match.span(), replacement))
            report.applied["mirroredRanges"] += 1
    return edits


# ---------------------------------------------------------------------------
# Whole-scope refresh


def _mirror_prose_sizes(
    texts: dict[str, str], prose: Mapping[str, set[str]], report: Report
):
    """Carry README size phrases (``175,436-byte``) into the tests that quote them."""

    changes = {}
    for old, values in sorted(prose.items()):
        if len(values) == 1:
            changes[old] = next(iter(values))
        else:
            report.unresolved.append(
                {"reason": f"README size {old} changed to {sorted(values)}"}
            )
    if not changes:
        return
    for path, text in list(texts.items()):
        if not path.endswith(".py"):
            continue
        edits = [
            (*match.span("value"), changes[match.group("value")])
            for match in _PROSE_SIZE.finditer(text)
            if match.group("value") in changes
        ]
        if edits:
            texts[path] = apply_edits(text, edits)
            report.applied["mirroredProse"] += len(edits)


def _scan(root: Path, scopes: Sequence[Path]) -> dict[str, str]:
    texts = {}
    for scope in scopes:
        for path in sorted((root / scope).rglob("*")):
            if (
                path.is_file()
                and path.suffix in SCANNED_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                texts[path.relative_to(root).as_posix()] = path.read_text(
                    encoding="utf-8"
                )
    return texts


def _digest(text: str) -> tuple[str, int]:
    payload = text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _integer_literals(text: str, value: int) -> list[tuple[int, int]]:
    """Spans of ``value`` written as a plain integer, not part of a longer number."""

    pattern = rf"(?<![\w.])(?<!\d,){value}(?![\w.]|,\d)"
    return [match.span() for match in re.finditer(pattern, text)]


def _cascade_file_pins(original: dict[str, str], texts: dict[str, str], report: Report):
    """Re-pin the SHA-256 and size of every changed file wherever they are recorded."""

    pinned = {path: _digest(text) for path, text in original.items()}
    for _ in range(MAX_CASCADE_ROUNDS):
        moved = {
            path: (pinned[path], _digest(texts[path]))
            for path in sorted(texts)
            if _digest(texts[path]) != pinned[path]
        }
        if not moved:
            return
        for path, ((old_sha, old_size), (new_sha, new_size)) in moved.items():
            pattern = re.compile(rf"(?<![0-9a-fA-F]){old_sha}(?![0-9a-fA-F])")
            for other, text in list(texts.items()):
                if other == path or old_sha not in text:
                    continue
                edits = [(m.start(), m.end(), new_sha) for m in pattern.finditer(text)]
                report.applied["fileSha256"] += len(edits)
                spans = (
                    [] if old_size == new_size else _integer_literals(text, old_size)
                )
                if len(spans) == 1:
                    edits.append((*spans[0], str(new_size)))
                    report.applied["fileSize"] += 1
                elif spans:
                    report.unresolved.append(
                        {
                            "path": other,
                            "reason": f"{len(spans)} copies of {path} size {old_size}",
                        }
                    )
                texts[other] = apply_edits(text, edits)
            pinned[path] = (new_sha, new_size)
    raise RefreshError("File pins did not settle; two files may pin each other")


def refresh(
    pins: Mapping[str, Pin],
    root: Path = ROOT,
    scopes: Sequence[Path] = (DEFAULT_SCOPE,),
    *,
    write: bool = False,
) -> Report:
    """Refresh every pin in ``scopes``; write the files only with ``write``."""

    report = Report(pins=len(pins))
    original = _scan(root, scopes)
    texts = dict(original)
    seen: set[str] = set()
    summaries = {}
    prose: dict[str, set[str]] = defaultdict(set)
    for path, text in original.items():
        edits, found = leaf_edits(path, text, pins, report)
        seen |= found
        if edits:
            texts[path] = apply_edits(text, edits)
        if path.endswith(".md"):
            for start, end, value in edits:
                if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text[start:end]):
                    prose[text[start:end]].add(value)
        texts[path], summary = refresh_summaries(path, text, texts[path], report)
        if summary is not None:
            summaries[Path(path).name] = summary
    for name, (old, new) in sorted(summaries.items()):
        for path, text in list(texts.items()):
            if Path(path).name != name and name in text:
                edits = mirror_edits(path, text, old, new, report)
                if edits:
                    texts[path] = apply_edits(text, edits)
    _mirror_prose_sizes(texts, prose, report)
    report.unmatched = sorted(set(pins) - seen)
    _cascade_file_pins(original, texts, report)
    for path in sorted(texts):
        if texts[path] == original[path]:
            continue
        before, after = _digest(original[path]), _digest(texts[path])
        report.files[path] = {
            "before": {"sha256": before[0], "sizeBytes": before[1]},
            "after": {"sha256": after[0], "sizeBytes": after[1]},
        }
        if write:
            (root / path).write_text(texts[path], encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "drift", nargs="*", type=Path, help="job logs, drift JSONL or directories"
    )
    parser.add_argument(
        "--run", type=int, help="collect drift from every job of this run"
    )
    parser.add_argument("--repo", default="CrossGL/crosstl")
    parser.add_argument(
        "--save-records", type=Path, help="write the collected records as JSONL"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--scope", type=Path, action="append", help="default: demos/integrations/mlx"
    )
    parser.add_argument(
        "--summary", type=Path, help="write the refresh summary as JSON"
    )
    parser.add_argument(
        "--write", action="store_true", help="rewrite the pins in place"
    )
    args = parser.parse_args(argv)
    if not args.drift and args.run is None:
        parser.error("give drift files or --run")
    records = load_drift_records(args.drift)
    if args.run is not None:
        records.extend(fetch_run_records(args.repo, args.run))
    unique = {json.dumps(record, sort_keys=True): record for record in records}
    records = [unique[key] for key in sorted(unique)]
    if args.save_records:
        args.save_records.write_text(
            "".join(key + "\n" for key in sorted(unique)), encoding="utf-8"
        )
    pins, size_only = build_pins(records)
    report = refresh(
        pins, args.root.resolve(), args.scope or [DEFAULT_SCOPE], write=args.write
    )
    report.records = len(records)
    report.unresolved.extend(
        {"subject": item["subject"], "reason": "size drift without a hash record"}
        for item in size_only
    )
    summary = report.to_json()
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "records": summary["records"],
                "pins": summary["pins"],
                "applied": summary["applied"],
                "changedFiles": len(summary["files"]),
                "unresolved": len(summary["unresolved"]),
                "unmatchedPins": len(summary["unmatchedPins"]),
                "staleNativeIdentities": sum(report.native_identities.values()),
                "written": args.write,
            },
            indent=2,
        )
    )
    for item in report.unresolved[:20]:
        print("unresolved:", json.dumps(item, sort_keys=True), file=sys.stderr)
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
