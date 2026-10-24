"""Cancel the workflow runs of merge groups that the merge queue has discarded.

When an entry fails or leaves the merge queue, GitHub deletes the merge groups behind it and creates new ones, but
the runs it started for the deleted groups carry on and hold their runners until they finish. A merge group's run
is orphaned when the group's ``gh-readonly-queue/`` branch no longer exists and its commit did not land on the
default branch; the runs of a group that landed are kept, since they test a commit on that branch.

``python -m tools.orphaned_runs`` lists the orphaned runs; ``--apply`` cancels them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import quote

QUEUE = "gh-readonly-queue/"
# The states a run passes through before ``completed``. The runs endpoint filters on one state per request.
ACTIVE = ("requested", "waiting", "pending", "queued", "in_progress")
# Cancellations per invocation. Rebuilding a queue of ten groups orphans about a hundred runs.
LIMIT = 200
# Commits of the default branch searched for a group that landed; an older group's runs have timed out.
HISTORY = 100
PAGE = 100


@dataclass(frozen=True)
class Run:
    id: int
    workflow: str
    branch: str
    sha: str


def target(branch: str) -> str | None:
    """The branch a merge group's branch, ``gh-readonly-queue/<target>/pr-<number>-<base>``, merges into."""
    if not branch.startswith(QUEUE):
        return None
    name, separator, _ = branch.removeprefix(QUEUE).rpartition("/pr-")
    return name if separator and name else None


def orphaned(runs: Iterable[Run], groups: set[str], landed: set[str]) -> list[Run]:
    """The ``runs`` of merge groups whose branch is not in ``groups`` and whose commit is not in ``landed``, oldest
    first."""
    return sorted(
        (
            run
            for run in runs
            if target(run.branch) and run.branch not in groups and run.sha not in landed
        ),
        key=lambda run: run.id,
    )


class GitHub:
    """Calls through the ``gh`` command line, which reads ``GH_TOKEN``."""

    def __init__(
        self,
        repository: str,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.repository = repository
        self._run = run

    def _api(self, method: str, path: str) -> object:
        done = self._run(
            ["gh", "api", "--method", method, f"repos/{self.repository}/{path}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            raise RuntimeError(f"gh api {method} {path}: {done.stderr.strip()}")
        return json.loads(done.stdout) if done.stdout.strip() else None

    def active_runs(self) -> list[Run]:
        """The merge-group runs that have not completed."""
        runs: dict[int, Run] = {}
        for status in ACTIVE:
            page = 1
            while True:
                batch = self._api(
                    "GET",
                    f"actions/runs?event=merge_group&status={status}&per_page={PAGE}&page={page}",
                )["workflow_runs"]
                for item in batch:
                    runs[item["id"]] = Run(
                        item["id"], item["name"], item["head_branch"], item["head_sha"]
                    )
                if len(batch) < PAGE:
                    break
                page += 1
        return list(runs.values())

    def groups(self) -> set[str]:
        """The branches of the merge groups that exist now."""
        branches: set[str] = set()
        page = 1
        while True:
            batch = self._api(
                "GET",
                f"git/matching-refs/heads/{QUEUE}?per_page={PAGE}&page={page}",
            )
            branches |= {item["ref"].removeprefix("refs/heads/") for item in batch}
            if len(batch) < PAGE:
                return branches
            page += 1

    def landed(self, branches: Iterable[str]) -> set[str]:
        """The latest commits of ``branches``."""
        shas: set[str] = set()
        for branch in sorted(branches):
            commits = self._api(
                "GET", f"commits?sha={quote(branch, safe='')}&per_page={HISTORY}"
            )
            shas |= {commit["sha"] for commit in commits}
        return shas

    def cancel(self, run: Run) -> bool:
        """Cancel ``run``; false when it finished first."""
        try:
            self._api("POST", f"actions/runs/{run.id}/cancel")
        except RuntimeError as error:
            if "HTTP 409" in str(error):
                return False
            raise
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.orphaned_runs", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--repo", default="CrossGL/crosstl", help="owner/name (default: %(default)s)"
    )
    parser.add_argument(
        "--apply", action="store_true", help="cancel the runs instead of listing them"
    )
    parser.add_argument("--limit", type=int, default=LIMIT, help="default: %(default)s")
    args = parser.parse_args(argv)

    github = GitHub(args.repo)
    # Runs first: a group created after this call has no run in the list, and one that lands meanwhile is in landed.
    runs = github.active_runs()
    groups = github.groups()
    landed = github.landed({target(run.branch) for run in runs} - {None})
    found = orphaned(runs, groups, landed)
    cancelled = 0
    for run in found[: args.limit]:
        line = f"{run.id} {run.workflow} on {run.branch.removeprefix(QUEUE)}"
        if not args.apply:
            print(f"would cancel {line}")
        elif github.cancel(run):
            cancelled += 1
            print(f"cancelled {line}")
        else:
            print(f"finished first: {line}")
    if len(found) > args.limit:
        print(f"left {len(found) - args.limit} for the next run (limit {args.limit})")
    print(
        f"{len(found)} of {len(runs)} active merge-group runs orphaned"
        + (f", {cancelled} cancelled" if args.apply else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
