"""Select routine and full MLX validation from the files changed by an event."""

import argparse
import json
import os
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

WORKFLOW = ".github/workflows/demo-project-testing.yml"
FULL_PATHS = (
    "demos/integrations/mlx/**",
    WORKFLOW,
)
# The tier selector, its tests and the demo's prose never run inside the full
# tier, and the complete test suite runs the selector's tests on every pull
# request and queue group, so a change limited to them needs only the smoke
# tier. ``PLAN_JOB`` names a workflow change confined to the ci-plan job.
PLAN_JOB = f"{WORKFLOW}#ci-plan"
PLANNER_PATHS = (
    "demos/integrations/mlx/ci_plan.py",
    "demos/integrations/mlx/tests/test_ci_plan.py",
    "demos/integrations/mlx/*.md",
    PLAN_JOB,
)
SMOKE_PATHS = (
    "crosstl/**",
    "tests/**",
    "tools/**",
    "support/**",
    ".github/actions/**",
    ".github/actionlint.yaml",
    ".pre-commit-config.yaml",
    "conftest.py",
    "pyproject.toml",
    "requirements*.txt",
    "setup.cfg",
)


def validation_plan(event, paths, full_requested=False):
    """Select the MLX validation tier for an event and the files it changes.

    Nightly runs and dispatches always get full validation. Pushes and
    merge-queue groups get it when they change the MLX corpus or this workflow.
    A pull request gets the smoke tier for any relevant change and full
    validation only when it carries the ``ci:full`` label. The full tier is
    several hundred jobs on macOS, Windows and Linux runners that every open
    pull request shares, and the queue builds one group at a time, so a group
    that cannot affect MLX must not wait hours for it. Every merge-queue group
    still runs the smoke tier on the exact commit that lands.
    """
    if event in {"schedule", "workflow_dispatch"}:
        return {"heavy": True, "smoke": False}
    if event not in {"pull_request", "push", "merge_group"}:
        raise ValueError(f"Unsupported workflow event: {event}")
    paths = tuple(paths)
    mlx = [path for path in paths if _matches(path, (*FULL_PATHS, PLAN_JOB))]
    corpus = any(not _matches(path, PLANNER_PATHS) for path in mlx)
    heavy = bool(full_requested) if event == "pull_request" else corpus
    smoke = (
        heavy
        or bool(mlx)
        or event == "merge_group"
        or any(fnmatchcase(path, pattern) for path in paths for pattern in SMOKE_PATHS)
    )
    return {"heavy": heavy, "smoke": smoke}


def _matches(path, patterns):
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def _require_hashes(*revisions):
    if not all(re.fullmatch(r"[0-9a-f]{40}", value or "") for value in revisions):
        raise ValueError("Both diff revisions must be full Git commit hashes")


def has_commit(revision):
    """Report whether the clone contains a commit, such as a push's old tip."""
    _require_hashes(revision)
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        timeout=60,
    )
    return result.returncode == 0


def changed_paths(base, head):
    """Read the head's complete diff from its merge base with ``base``.

    Comparing from the merge base keeps commits that landed on the base branch
    after the head branched from selecting a tier, and includes the old path of
    a renamed source.
    """
    _require_hashes(base, head)
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", f"{base}...{head}", "--"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return tuple(
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    )


def _without_plan_job(text):
    """Return the workflow's lines outside its top-level ``ci-plan`` job."""
    kept, in_plan = [], False
    for line in text.splitlines():
        if re.fullmatch(r"(  )?[A-Za-z0-9_-]+:\s*", line):
            in_plan = line.rstrip() == "  ci-plan:"
        if not in_plan:
            kept.append(line)
    return kept


def _file_at(revision, path):
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        capture_output=True,
        timeout=60,
    )
    return result.stdout.decode("utf-8") if result.returncode == 0 else None


def classify_workflow_change(base, head, paths):
    """Name a workflow change that only edits the ci-plan job as ``PLAN_JOB``.

    The plan job decides which jobs run but runs none of the full tier, so
    editing it alone does not need that tier. Any change to another job, a
    trigger or the workflow's environment keeps the workflow path.
    """
    paths = tuple(paths)
    if WORKFLOW not in paths:
        return paths
    _require_hashes(base, head)
    merge_base = subprocess.run(
        ["git", "merge-base", base, head],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    before, after = _file_at(merge_base, WORKFLOW), _file_at(head, WORKFLOW)
    if before is None or after is None:
        return paths
    if _without_plan_job(before) != _without_plan_job(after):
        return paths
    return tuple(PLAN_JOB if path == WORKFLOW else path for path in paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default=os.environ.get("MLX_CI_BASE", ""))
    parser.add_argument("--head", default=os.environ.get("MLX_CI_HEAD", ""))
    parser.add_argument("--output", type=Path, default=os.environ.get("GITHUB_OUTPUT"))
    parser.add_argument(
        "--full-requested",
        choices=("true", "false"),
        default=os.environ.get("MLX_CI_FULL", "false") or "false",
        help="whether a pull request asked for full validation with the ci:full label",
    )
    args = parser.parse_args(argv)
    full_requested = args.full_requested == "true"
    if args.event in {"schedule", "workflow_dispatch"}:
        plan = validation_plan(args.event, ())
    elif args.event in {"push", "merge_group"} and (
        args.base == "0" * 40 or not has_commit(args.base)
    ):
        # A new branch has no old tip, and a rewritten push tip or an earlier
        # queue group may not be fetched: without a diff, validate everything.
        plan = {"heavy": True, "smoke": True}
    else:
        paths = classify_workflow_change(
            args.base, args.head, changed_paths(args.base, args.head)
        )
        plan = validation_plan(args.event, paths, full_requested)
    if args.output is not None:
        with args.output.open("a", encoding="utf-8") as output:
            output.writelines(
                f"{name}={str(value).lower()}\n" for name, value in plan.items()
            )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
