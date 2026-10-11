"""Assertions for positive workflow path-filter coverage."""

from fnmatch import fnmatchcase
from pathlib import Path

from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[1]


def _matches_path(parts, patterns):
    if not patterns:
        return not parts
    if patterns[0] == "**":
        return _matches_path(parts, patterns[1:]) or bool(
            parts and _matches_path(parts[1:], patterns)
        )
    return bool(
        parts
        and fnmatchcase(parts[0], patterns[0])
        and _matches_path(parts[1:], patterns[1:])
    )


def assert_paths_covered(filters, *required, root=ROOT):
    assert filters and not any(pattern.startswith("!") for pattern in filters)
    for path in required:
        # Before Python 3.13, a trailing ** yields directories only.
        pattern = path + "/*" if path.endswith("/**") else path
        candidates = (
            [
                item.relative_to(root).as_posix()
                for item in root.glob(pattern)
                if item.is_file()
            ]
            if any(character in path for character in "*?[")
            else [path]
        )
        assert candidates, f"No files match required workflow path: {path}"
        for candidate in candidates:
            assert any(
                _matches_path(candidate.split("/"), pattern.split("/"))
                for pattern in filters
            ), candidate


def assert_workflow_triggers(workflow, *required):
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event), *required
        )
