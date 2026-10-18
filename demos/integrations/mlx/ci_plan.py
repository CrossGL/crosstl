"""Select routine and full MLX validation from the files changed by an event."""

import argparse
import json
import os
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

FULL_PATHS = (
    "demos/integrations/mlx/**",
    ".github/workflows/demo-project-testing.yml",
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

    Nightly runs, dispatches, merge-queue groups and pushes that change the MLX
    corpus or this workflow get full validation. A pull request gets the smoke tier for any relevant change
    and full validation only when it carries the ``ci:full`` label. The full tier is
    several hundred jobs on macOS, Windows and Linux runners that every open pull
    request shares, so a routine push must not queue it.
    """
    if event in {"schedule", "workflow_dispatch"}:
        return {"heavy": True, "smoke": False}
    if event == "merge_group":
        # The queue is the last gate before main: run everything, smoke included.
        return {"heavy": True, "smoke": True}
    if event not in {"pull_request", "push"}:
        raise ValueError(f"Unsupported workflow event: {event}")
    paths = tuple(paths)
    corpus = any(fnmatchcase(path, pattern) for path in paths for pattern in FULL_PATHS)
    heavy = bool(full_requested) if event == "pull_request" else corpus
    smoke = (
        heavy
        or corpus
        or any(fnmatchcase(path, pattern) for path in paths for pattern in SMOKE_PATHS)
    )
    return {"heavy": heavy, "smoke": smoke}


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
    if args.event in {"schedule", "workflow_dispatch", "merge_group"}:
        plan = validation_plan(args.event, ())
    elif args.event == "push" and (args.base == "0" * 40 or not has_commit(args.base)):
        # A new branch has no old tip, and a rewritten one may not be fetched.
        plan = {"heavy": True, "smoke": True}
    else:
        plan = validation_plan(
            args.event, changed_paths(args.base, args.head), full_requested
        )
    if args.output is not None:
        with args.output.open("a", encoding="utf-8") as output:
            output.writelines(
                f"{name}={str(value).lower()}\n" for name, value in plan.items()
            )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
