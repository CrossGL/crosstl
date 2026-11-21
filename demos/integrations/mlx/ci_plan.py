"""Select the MLX validation tier and its heavy legs from the changed files."""

import argparse
import json
import os
import re
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import yaml

from demos.integrations.mlx import ci_legs

WORKFLOW = ".github/workflows/demo-project-testing.yml"
FULL_PATHS = (
    "demos/integrations/mlx/**",
    WORKFLOW,
)
# The tier selector, the smoke canary, their tests and the demo's prose never
# run inside the full tier. The smoke tier runs the canary and the complete test
# suite runs their tests on every pull request and queue group, so a change
# limited to them needs only the smoke tier. ``PLAN_JOB`` names a workflow
# change confined to the ci-plan job.
PLAN_JOB = f"{WORKFLOW}#ci-plan"
PLANNER_PATHS = (
    "demos/integrations/mlx/ci_legs.py",
    "demos/integrations/mlx/ci_plan.py",
    "demos/integrations/mlx/smoke_canary.py",
    "demos/integrations/mlx/tests/test_ci_legs.py",
    "demos/integrations/mlx/tests/test_ci_plan.py",
    "demos/integrations/mlx/tests/test_smoke_canary.py",
    "demos/integrations/mlx/*.md",
    PLAN_JOB,
)
SMOKE_PATHS = (
    "crosstl/**",
    "tests/**",
    "tools/**",
    ".github/actions/**",
    ".github/actionlint.yaml",
    ".pre-commit-config.yaml",
    "conftest.py",
    "pyproject.toml",
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


def _merge_base(base, head):
    _require_hashes(base, head)
    return subprocess.run(
        ["git", "merge-base", base, head],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


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
    merge_base = _merge_base(base, head)
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
    parser.add_argument(
        "--workflow",
        type=Path,
        default=os.environ.get("MLX_CI_WORKFLOW") or WORKFLOW,
        help="the workflow file whose heavy legs the plan selects",
    )
    args = parser.parse_args(argv)
    full_requested = args.full_requested == "true"
    change = None
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
        if args.event != "pull_request":
            # A pull request reaches the full tier only through ci:full,
            # which asks for every leg.
            change = (args.base, args.head, paths)
    legs = plan_legs(args.workflow, change) if plan["heavy"] else []
    if args.output is not None:
        with args.output.open("a", encoding="utf-8") as output:
            output.writelines(
                f"{name}={str(value).lower()}\n" for name, value in plan.items()
            )
            output.write(f"legs={json.dumps(legs)}\n")
    print(json.dumps({**plan, "legs": len(legs)}, sort_keys=True))
    return 0


def plan_legs(workflow, change=None):
    """Return the heavy legs that a full-tier event runs, sorted.

    ``change`` holds the base, the head and the changed paths of a push or a
    queue group; without one, every leg runs. When the selection cannot be
    computed, every leg runs and a workflow warning names the error.
    """
    after = yaml.safe_load(Path(workflow).read_text(encoding="utf-8"))
    legs = ci_legs.workflow_legs(after)
    if change is None:
        return sorted(legs)
    try:
        return sorted(changed_legs(after, legs, *change))
    except Exception as error:  # noqa: BLE001 - any failure runs every leg
        message = f"{type(error).__name__}: {error}".replace("\n", " ")
        print(
            f"::warning title=MLX leg plan::{message}; running every heavy leg",
            file=sys.stderr,
        )
        return sorted(legs)


def changed_legs(after, legs, base, head, paths):
    """Return the legs whose definition or MLX files a change alters.

    A workflow change selects the legs whose definition differs from the merge
    base, or every leg when it changes anything outside the jobs. A changed MLX
    file selects the legs that can read it (``ci_legs``). The planner's own
    files never run in the full tier, so they select nothing here.
    """
    paths = tuple(paths)
    selected = set()
    if WORKFLOW in paths:
        text = _file_at(_merge_base(base, head), WORKFLOW)
        before = None if text is None else yaml.safe_load(text)
        changed = ci_legs.changed_legs(before, after)
        if changed is ci_legs.EVERY_LEG:
            return set(legs)
        selected |= changed
    prefix = ci_legs.MLX + "/"
    mlx = {
        path[len(prefix) :]
        for path in paths
        if path.startswith(prefix) and not _matches(path, PLANNER_PATHS)
    }
    if mlx:
        index = ci_legs.PathIndex(set(_tracked(ci_legs.MLX)) | mlx)
        closure = ci_legs.Closure(index, _reader(Path(ci_legs.MLX)))
        selected |= ci_legs.affected_legs(legs, mlx, closure)
    return selected


def _tracked(folder):
    """Return the tracked files below a folder, relative to it."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", folder],
        check=True,
        capture_output=True,
        timeout=60,
    )
    prefix = folder.rstrip("/") + "/"
    return [
        path[len(prefix) :]
        for path in result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
        if path.startswith(prefix)
    ]


def _reader(root):
    def read(path):
        try:
            return (root / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    return read


if __name__ == "__main__":
    raise SystemExit(main())
