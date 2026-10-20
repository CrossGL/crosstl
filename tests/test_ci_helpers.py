"""Unit tests for the workflow assertions in ``tests/ci_helpers.py``."""

import pytest
import yaml

from tests.ci_helpers import (
    assert_native_step_prerequisites,
    assert_paths_covered,
    assert_workflow_triggers,
    bounded_step_deadline,
)


@pytest.mark.parametrize(
    "pattern,path,covered",
    [
        ("tests/**", "tests/test_example.py", True),
        ("tests/**", "tests/backend/test_example.py", True),
        ("tests/*.py", "tests/test_example.py", True),
        ("tests/*.py", "tests/backend/test_example.py", False),
        ("tests/**/test_*.py", "tests/test_example.py", True),
        ("tests/**/test_*.py", "tests/backend/test_example.py", True),
        ("tests/**/test_*.py", "tests/backend/example.txt", False),
        ("requirements*.txt", "requirements-dev.txt", True),
        ("tests/**", "demos/test_example.py", False),
    ],
)
def test_workflow_path_coverage_preserves_directory_boundaries(pattern, path, covered):
    if covered:
        assert_paths_covered([pattern], path)
    else:
        with pytest.raises(AssertionError):
            assert_paths_covered([pattern], path)


def test_workflow_subtree_coverage_checks_every_existing_file(tmp_path):
    (tmp_path / "tests/backend").mkdir(parents=True)
    (tmp_path / "tests/test_top.py").touch()
    (tmp_path / "tests/backend/test_nested.py").touch()
    assert_paths_covered(["tests/**"], "tests/**", root=tmp_path)
    with pytest.raises(AssertionError, match="test_nested"):
        assert_paths_covered(["tests/*.py"], "tests/**", root=tmp_path)
    with pytest.raises(AssertionError, match="No files match"):
        assert_paths_covered(["tests/**"], "missing/**", root=tmp_path)
    with pytest.raises(AssertionError):
        assert_paths_covered(["tests/**", "!tests/backend/**"], "tests/test_top.py")


def test_workflow_path_coverage_requires_both_events():
    workflow = """on:
  pull_request:
    paths:
      - "tests/**"
  push:
    paths:
      - "docs/**"
"""
    with pytest.raises(AssertionError, match="tests/test_example.py"):
        assert_workflow_triggers(workflow, "tests/test_example.py")


@pytest.mark.parametrize(
    "condition,runner_os,expected",
    [
        ("", "Linux", 180),
        ("runner.os == 'Windows'", "Windows", 180),
        ("runner.os == 'Windows'", "Linux", 0),
        ("runner.os != 'Linux'", "macOS", 180),
        ("runner.os != 'Linux'", "Linux", 0),
        ("runner.os == 'Windows' || steps.prepare.outcome == 'success'", "Linux", 180),
    ],
)
def test_native_deadline_excludes_only_proven_platform_mismatches(
    condition, runner_os, expected
):
    step = {
        "if": condition,
        "run": (
            "python bounded.py --timeout-seconds 120\n"
            "python bounded.py --timeout-seconds 60"
        ),
    }
    assert bounded_step_deadline(step, runner_os) == expected


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        None,
        "missing",
        "always",
        "cancelled",
        "prerequisite",
        "optional",
        "explicit-false",
        "wrong-job",
        "unknown-job",
    ],
)
def test_native_step_prerequisite_assertions_reject_weakened_conditions(
    wrapped, pinned, damage
):
    expected = (
        "${{ !cancelled() && steps.pinned-upstream.outcome == 'success' }}"
        if pinned
        else "${{ !cancelled() && (steps.portable-host-contracts.outcome == 'success' || "
        "(runner.os == 'macOS' && steps.platform-host-abi.outcome == 'success') || "
        "steps.host-warp-identity.outcome == 'success') }}"
    )
    definition = {"name": "Required native check", "if": expected, "run": "pytest"}
    job = "portable-host"
    if damage == "missing":
        del definition["if"]
    elif damage == "always":
        definition["if"] = "always()"
    elif damage == "cancelled":
        definition["if"] = expected.replace("!cancelled() && ", "")
    elif damage == "prerequisite":
        definition["if"] = expected.replace("== 'success'", "!= 'cancelled'")
    elif damage in ("optional", "explicit-false"):
        definition["continue-on-error"] = damage == "optional"
    elif damage == "wrong-job":
        job = "metal-host"
    elif damage == "unknown-job":
        job = "other"
    text = yaml.safe_dump([definition] if wrapped else definition)
    if damage is None:
        assert_native_step_prerequisites(text, job=job, pinned=pinned)
    else:
        with pytest.raises(AssertionError):
            assert_native_step_prerequisites(text, job=job, pinned=pinned)


@pytest.mark.parametrize("job", ["metal-host", "mlx-metal-porting"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_step_prerequisite_assertions_preserve_unconditional_jobs(job, wrapped):
    definition = {"name": "Required native check", "run": "pytest"}
    text = yaml.safe_dump([definition] if wrapped else definition)
    assert_native_step_prerequisites(text, job=job)
