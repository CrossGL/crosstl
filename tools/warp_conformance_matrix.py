"""Summarize per-library JUnit results into a WARP conformance matrix.

The WARP conformance workflow runs the same Direct3D controls once for each
WARP library and writes ``<evidence>/<version>/results.xml`` plus
``<evidence>/runs.json``. This tool joins them into ``matrix.json`` and a
Markdown table. A failing control is a finding, not an error, except on a
library named with ``--require``: such a reference library must be present and
pass every test. The tool also fails when the evidence is incomplete, so a
missing or truncated run can never read as a conformant library.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# pytest exits 0 when every test passes and 1 when some fail; anything else
# (interrupted, internal error, usage error, no tests, timeout) means the run
# did not produce a complete result.
COMPLETE_EXIT_CODES = (0, 1)


class IncompleteEvidence(RuntimeError):
    """The matrix cannot be trusted because a run is missing or truncated."""


def _outcome(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        if case.find(tag) is not None:
            return "failed" if tag == "failure" else tag
    return "passed"


def read_results(path: Path) -> dict[str, str]:
    """Map each test node to passed, failed, error or skipped."""

    if not path.is_file():
        raise IncompleteEvidence(f"{path} is missing")
    outcomes: dict[str, str] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        node = f"{case.get('classname', '')}::{case.get('name', '')}"
        if node in outcomes:
            raise IncompleteEvidence(f"{path} reports {node} twice")
        outcomes[node] = _outcome(case)
    if not outcomes:
        raise IncompleteEvidence(f"{path} contains no test cases")
    return outcomes


def _library(evidence: Path, run: dict) -> dict:
    version = str(run["version"])
    library = {
        "version": version,
        "dllSHA256": str(run["dllSHA256"]),
        "exitCode": int(run["exitCode"]),
        "complete": True,
        "reason": None,
        "outcomes": {},
    }
    try:
        if library["exitCode"] not in COMPLETE_EXIT_CODES:
            raise IncompleteEvidence(f"the run exited with {library['exitCode']}")
        library["outcomes"] = dict(
            sorted(read_results(evidence / version / "results.xml").items())
        )
    except IncompleteEvidence as error:
        library.update(complete=False, reason=str(error))
    return library


def build_matrix(evidence: Path) -> dict:
    """Join every run; a library whose evidence is incomplete is marked so."""

    runs_path = evidence / "runs.json"
    if not runs_path.is_file():
        raise IncompleteEvidence(f"{runs_path} is missing")
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    if isinstance(runs, dict):
        runs = [runs]
    if not runs:
        raise IncompleteEvidence(f"{runs_path} lists no runs")
    libraries = [_library(evidence, run) for run in runs]
    nodes = sorted({node for library in libraries for node in library["outcomes"]})
    for library in libraries:
        missing = [node for node in nodes if node not in library["outcomes"]]
        if library["complete"] and missing:
            library.update(complete=False, reason=f"no result for {missing[0]}")
    return {
        "schema": "crosstl.warp-conformance/1",
        "complete": all(library["complete"] for library in libraries),
        "tests": nodes,
        "libraries": libraries,
    }


def required_failures(matrix: dict, required: list[str]) -> list[str]:
    """Explain why each required library is missing or did not pass."""

    libraries = {library["version"]: library for library in matrix["libraries"]}
    reasons = []
    for version in required:
        library = libraries.get(version)
        if library is None:
            reasons.append(f"required WARP {version} did not run")
            continue
        failed = [
            node
            for node, outcome in library["outcomes"].items()
            if outcome in ("failed", "error")
        ]
        # A test that executed on another library must execute here too.
        unexecuted = [
            node
            for node, outcome in library["outcomes"].items()
            if outcome == "skipped"
            and any(
                other["outcomes"].get(node, "skipped") != "skipped"
                for other in matrix["libraries"]
            )
        ]
        if failed:
            reasons.append(
                f"required WARP {version} failed {len(failed)} test(s), "
                f"first {failed[0]}"
            )
        elif unexecuted:
            reasons.append(
                f"required WARP {version} skipped {len(unexecuted)} test(s) "
                f"that other libraries executed, first {unexecuted[0]}"
            )
        elif library["exitCode"] != 0:
            reasons.append(f"required WARP {version} exited with {library['exitCode']}")
    return reasons


def markdown(matrix: dict) -> str:
    libraries = matrix["libraries"]
    executed = [
        node
        for node in matrix["tests"]
        if any(
            library["outcomes"].get(node, "skipped") != "skipped"
            for library in libraries
        )
    ]
    symbols = {"passed": "pass", "failed": "FAIL", "error": "ERROR", "skipped": "skip"}
    lines = [
        "## WARP conformance matrix",
        "",
        "| Test | " + " | ".join(library["version"] for library in libraries) + " |",
        "| --- |" + " --- |" * len(libraries),
    ]
    for node in executed:
        cells = [
            symbols[library["outcomes"][node]] if node in library["outcomes"] else "-"
            for library in libraries
        ]
        lines.append(f"| `{node.split('::', 1)[-1]}` | " + " | ".join(cells) + " |")
    for library in libraries:
        if not library["complete"]:
            lines.append(
                f"\nWARP {library['version']} is incomplete: {library['reason']}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="VERSION",
        help="a reference library that must run and pass every test",
    )
    args = parser.parse_args(argv)
    try:
        matrix = build_matrix(args.evidence)
    except IncompleteEvidence as error:
        print(f"incomplete WARP conformance evidence: {error}", file=sys.stderr)
        return 1
    (args.evidence / "matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
    )
    failures = required_failures(matrix, args.require)
    table = markdown(matrix) + "".join(f"\n{reason}\n" for reason in failures)
    print(table)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(table)
    return 0 if matrix["complete"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
