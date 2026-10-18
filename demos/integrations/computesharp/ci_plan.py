"""Select whether a workflow event runs the ComputeSharp port's validation.

Pull requests, merge-queue groups and pushes to main always start the
workflow, so its single result check can be required. This plan skips the
port's jobs when the event's changes cannot affect the port: the relevant
paths are the demo, its workflow, the packaging metadata and the CrossTL
modules that translating the generated HLSL imports.
``tests/test_workflow.py`` translates actual generated HLSL in a fresh
interpreter and requires every CrossTL module it loads to match one of these
paths, so a new dependency of the translation cannot silently skip the port.

Usage (from the repository root)::

    python demos/integrations/computesharp/ci_plan.py --event pull_request \
      --base <full base commit> --head <full head commit>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

RELEVANT_PATHS = (
    ".github/workflows/project-demo-testing.yml",
    "crosstl/*.py",
    "crosstl/backend/**",
    "crosstl/translator/**",
    "demos/integrations/computesharp/**",
    "pyproject.toml",
    "setup.py",
    # The translated target jobs dispatch these general layout checks.
    "tests/test_backend/test_directx/test_hlsl_cbuffer_layouts.py",
)
# Events without a meaningful change set validate everything.
UNCONDITIONAL_EVENTS = frozenset({"schedule", "workflow_dispatch"})
DIFFED_EVENTS = frozenset({"pull_request", "push", "merge_group"})
NO_COMMIT = "0" * 40


def path_regex(pattern: str) -> re.Pattern[str]:
    """Follow GitHub path-filter globs: ``*`` stays in one directory, ``**`` spans them."""
    parts = re.split(r"(\*\*|\*)", pattern)
    regex = "".join(
        ".*" if part == "**" else "[^/]*" if part == "*" else re.escape(part)
        for part in parts
    )
    return re.compile(regex)


_RELEVANT = tuple(path_regex(pattern) for pattern in RELEVANT_PATHS)


def is_relevant(path: str) -> bool:
    """Report whether a changed repository path can affect the port."""
    return any(pattern.fullmatch(path) for pattern in _RELEVANT)


def validation_plan(event: str, paths=()) -> dict[str, bool]:
    """Select the port's validation for an event and the files it changes."""
    if event in UNCONDITIONAL_EVENTS:
        return {"run": True}
    if event not in DIFFED_EVENTS:
        raise ValueError(f"Unsupported workflow event: {event}")
    return {"run": any(is_relevant(path) for path in paths)}


def _require_hashes(*revisions: str) -> None:
    if not all(re.fullmatch(r"[0-9a-f]{40}", value or "") for value in revisions):
        raise ValueError("Diff revisions must be full Git commit hashes")


def has_commit(revision: str) -> bool:
    """Report whether the clone contains a commit, such as a push's old tip."""
    _require_hashes(revision)
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


def changed_paths(base: str, head: str) -> tuple[str, ...]:
    """Read the head's diff from its merge base with ``base``.

    Comparing from the merge base keeps commits that reached the base branch
    after the head branched from selecting the port, and ``--no-renames``
    reports the old path of a moved source as well as the new one.
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


def plan_for_event(event: str, base: str, head: str) -> dict[str, bool]:
    if event in UNCONDITIONAL_EVENTS:
        return validation_plan(event)
    if event == "push" and (base == NO_COMMIT or not has_commit(base)):
        # A new branch has no old tip, and a rewritten one may not be fetched.
        return {"run": True}
    return validation_plan(event, changed_paths(base, head))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default=os.environ.get("COMPUTESHARP_CI_BASE", ""))
    parser.add_argument("--head", default=os.environ.get("COMPUTESHARP_CI_HEAD", ""))
    parser.add_argument(
        "--output", type=Path, default=os.environ.get("GITHUB_OUTPUT") or None
    )
    args = parser.parse_args(argv)
    plan = plan_for_event(args.event, args.base, args.head)
    if args.output is not None:
        with args.output.open("a", encoding="utf-8") as output:
            output.writelines(
                f"{name}={str(value).lower()}\n" for name, value in sorted(plan.items())
            )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
