"""Assertions for workflow coverage and required execution prerequisites."""

import re
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[1]


def bounded_step_deadline(step, runner_os):
    """Count literal command deadlines, excluding explicit platform mismatches."""
    condition = step.get("if", "")
    # Count disjunctions conservatively rather than interpreting arbitrary expressions.
    if "||" not in condition:
        for operator, platform in re.findall(
            r"runner\.os (==|!=) '([^']+)'", condition
        ):
            if (runner_os == platform) != (operator == "=="):
                return 0
    return sum(
        int(value)
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    )


def assert_native_step_prerequisites(step, *, job, pinned=False):
    """Require exact native prerequisites without accepting optional execution."""
    assert job in ("portable-host", "metal-host", "mlx-metal-porting")
    definition = yaml.safe_load(step)
    if isinstance(definition, list):
        assert len(definition) == 1
        definition = definition[0]
    assert isinstance(definition, dict)
    assert "continue-on-error" not in definition
    if job != "portable-host":
        assert "if" not in definition
        return
    expected = (
        "${{ !cancelled() && steps.pinned-upstream.outcome == 'success' }}"
        if pinned
        else "${{ !cancelled() && (steps.portable-host-contracts.outcome == 'success' || "
        "(runner.os == 'macOS' && steps.platform-host-abi.outcome == 'success') || "
        "steps.host-warp-identity.outcome == 'success') }}"
    )
    assert " ".join(definition.get("if", "").split()) == expected


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
