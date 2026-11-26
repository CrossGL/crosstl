"""Cancel the workflow runs that can no longer change what the merge queue does.

A run is orphaned when its merge group is gone. When an entry fails or leaves the queue, GitHub deletes the groups
behind it and creates new ones, but the runs it started for the deleted groups carry on and hold their runners
until they finish. Such a run's ``gh-readonly-queue/`` branch no longer exists and its commit did not land on the
default branch; the runs of a group that landed are kept, since they test a commit on that branch.

A run has failed when a job of its latest attempt failed while one of the branch's required checks in it, the
``if: always()`` result job, is still waiting for the other jobs. That check will fail, but the queue drops the
entry only when it reports, so a slow leg holds every group behind it. Cancelling the run lets the result job
report at once; the group's other runs are orphaned when the queue rebuilds the groups behind it. GitHub lists a
job that needs others only once they have finished, so a waiting result job is one that the run's workflow file
names and the run does not list as completed.

A run is doomed when its merge group cannot land: a required check of the group has failed or will fail (a failed
run, above), or the group is built on top of such a group. The queue builds each group on the one ahead of it, the
branch ``gh-readonly-queue/<target>/pr-<number>-<base>`` naming the head commit of that group as its base, and with
the ``ALLGREEN`` grouping every entry's own group must pass. So once a group has failed, its entry leaves the queue
or, when an entry ahead of it also fails, is rebuilt, and either way the queue rebuilds every group behind it, whose
runs then test a commit that never lands. The queue removes the failed entry only once the entries ahead of it have
finished, and until then the doomed runs hold runners the groups ahead of the failure need. Re-running a failed
group's jobs does not bring back the runs this cancelled behind it; take the entry out of the queue and add it again
instead.

A pull-request run is closed when no open pull request has its head branch. GitHub lets the runs of a pull request
that merged or closed carry on; the merge group tested what landed, and nothing reads their checks any more.

``python -m tools.orphaned_runs`` lists the four kinds; ``--apply`` cancels them, and ``--watch`` repeats the
sweep every few minutes while the queue has merge groups.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import quote

QUEUE = "gh-readonly-queue/"
# The states a run passes through before ``completed``. The runs endpoint filters on one state per request.
ACTIVE = ("requested", "waiting", "pending", "queued", "in_progress")
# Cancellations per sweep. Rebuilding a queue of ten groups orphans about a hundred runs.
LIMIT = 200
# Commits of the default branch searched for a group that landed; an older group's runs have timed out.
HISTORY = 100
PAGE = 100
# Seconds between sweeps with --watch, and the failed sweeps in a row that end it.
INTERVAL = 300
ERRORS = 3
# Seconds a gh call may take. A call that never returns would hold the sweep, and every sweep queued behind it in
# the workflow's concurrency group, until the job times out.
TIMEOUT = 120
# Job conclusions that fail a gating workflow's result job.
FAILED = ("failure", "timed_out", "startup_failure")
# The names of the latest check runs of each workflow run on a commit, listing only the failed ones.
SUITES = """
fragment suites on Commit {
  checkSuites(first: 50) {
    nodes {
      workflowRun { databaseId }
      checkRuns(first: 100, filterBy: {checkType: LATEST, conclusions: [FAILURE, TIMED_OUT, STARTUP_FAILURE]}) {
        nodes { name }
      }
    }
  }
}
"""
# A literal ``name:`` in a workflow file, which names a job or a step.
NAME = re.compile(r"^\s+(?:- )?name:\s*(?P<quote>['\"]?)(?P<name>.+?)(?P=quote)\s*$")


@dataclass(frozen=True)
class Run:
    id: int
    workflow: str
    branch: str
    sha: str
    # The full name of the repository the head branch is in; a fork for a pull request from one.
    repository: str = ""
    # The workflow file, such as .github/workflows/lint.yml.
    path: str = ""


def target(branch: str) -> str | None:
    """The branch a merge group's branch, ``gh-readonly-queue/<target>/pr-<number>-<base>``, merges into."""
    if not branch.startswith(QUEUE):
        return None
    name, separator, _ = branch.removeprefix(QUEUE).rpartition("/pr-")
    return name if separator and name else None


def base(branch: str) -> str | None:
    """The commit a merge group's branch is built on: the head commit of the group ahead of it in the queue, or of
    the target branch for the first group."""
    if not target(branch):
        return None
    number, separator, sha = branch.rpartition("/pr-")[2].partition("-")
    return sha if separator and number.isdigit() and sha else None


def doomed(groups: Mapping[str, str], broken: set[str]) -> set[str]:
    """The branches of the merge groups in ``groups``, a branch to head commit map, that cannot land: each group
    whose head commit is in ``broken``, and every group built on one of them."""
    behind: dict[str, list[str]] = {}
    for branch in groups:
        behind.setdefault(base(branch) or "", []).append(branch)
    found = {branch for branch, sha in groups.items() if sha in broken}
    stack = sorted(found)
    while stack:
        for branch in behind.get(groups[stack.pop()], []):
            if branch not in found:
                found.add(branch)
                stack.append(branch)
    return found


def orphaned(
    runs: Iterable[Run], groups: Collection[str], landed: set[str]
) -> list[Run]:
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


def closed(runs: Iterable[Run], heads: set[tuple[str, str]]) -> list[Run]:
    """The pull-request ``runs`` whose ``(repository, branch)`` is not the head of an open pull request in
    ``heads``, oldest first."""
    return sorted(
        (run for run in runs if (run.repository, run.branch) not in heads),
        key=lambda run: run.id,
    )


class GitHub:
    """Calls through the ``gh`` command line, which reads ``GH_TOKEN``."""

    def __init__(
        self,
        repository: str,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        timeout: float = TIMEOUT,
    ):
        self.repository = repository
        self._run = run
        self.timeout = timeout

    def _call(self, command: list[str]) -> subprocess.CompletedProcess:
        """Run ``command``; a call that takes longer than ``timeout`` fails like a refused one."""
        try:
            return self._run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{' '.join(command[:4])}: no answer in {self.timeout:g} s"
            ) from None

    def _gh(self, command: list[str]) -> object:
        done = self._call(command)
        if done.returncode:
            raise RuntimeError(f"{' '.join(command[:4])}: {done.stderr.strip()}")
        return json.loads(done.stdout) if done.stdout.strip() else None

    def _api(self, method: str, path: str) -> object:
        return self._gh(
            ["gh", "api", "--method", method, f"repos/{self.repository}/{path}"]
        )

    def active_runs(self, event: str = "merge_group") -> list[Run]:
        """The runs for ``event`` that have not completed."""
        runs: dict[int, Run] = {}
        for status in ACTIVE:
            page = 1
            while True:
                batch = self._api(
                    "GET",
                    f"actions/runs?event={event}&status={status}&per_page={PAGE}&page={page}",
                )["workflow_runs"]
                for item in batch:
                    runs[item["id"]] = Run(
                        item["id"],
                        item["name"],
                        item["head_branch"],
                        item["head_sha"],
                        (item.get("head_repository") or {}).get("full_name", ""),
                        item.get("path", ""),
                    )
                if len(batch) < PAGE:
                    break
                page += 1
        return list(runs.values())

    def open_heads(self) -> set[tuple[str, str]]:
        """The ``(repository, branch)`` heads of the open pull requests."""
        heads: set[tuple[str, str]] = set()
        page = 1
        while True:
            batch = self._api("GET", f"pulls?state=open&per_page={PAGE}&page={page}")
            heads |= {
                ((pull["head"]["repo"] or {}).get("full_name", ""), pull["head"]["ref"])
                for pull in batch
            }
            if len(batch) < PAGE:
                return heads
            page += 1

    def groups(self) -> dict[str, str]:
        """The branches of the merge groups that exist now, with their head commits."""
        branches: dict[str, str] = {}
        page = 1
        while True:
            batch = self._api(
                "GET",
                f"git/matching-refs/heads/{QUEUE}?per_page={PAGE}&page={page}",
            )
            branches.update(
                (item["ref"].removeprefix("refs/heads/"), item["object"]["sha"])
                for item in batch
            )
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

    def failures(self, shas: Iterable[str]) -> dict[str, dict[int, set[str]]]:
        """The workflow runs on the commits ``shas`` whose latest attempt has failed check runs, with the names of
        those check runs, by commit, in one GraphQL query."""
        shas = sorted(set(shas))
        if not shas:
            return {}
        owner, name = self.repository.split("/", 1)
        declared = "".join(f", $c{index}: GitObjectID!" for index in range(len(shas)))
        fields = " ".join(
            f"c{index}: object(oid: $c{index}) {{ ...suites }}"
            for index in range(len(shas))
        )
        query = (
            f"query($owner: String!, $name: String!{declared}) "
            f"{{ repository(owner: $owner, name: $name) {{ {fields} }} }}{SUITES}"
        )
        command = ["gh", "api", "graphql", "-f", f"query={query}"]
        values = {"owner": owner, "name": name}
        values.update({f"c{index}": sha for index, sha in enumerate(shas)})
        for key, value in values.items():
            command += ["-f", f"{key}={value}"]
        commits = self._gh(command)["data"]["repository"]
        found: dict[str, dict[int, set[str]]] = {}
        for index, sha in enumerate(shas):
            commit = commits.get(f"c{index}")
            for suite in (commit or {}).get("checkSuites", {}).get("nodes", []):
                names = {check["name"] for check in suite["checkRuns"]["nodes"]}
                if suite["workflowRun"] and names:
                    found.setdefault(sha, {})[
                        suite["workflowRun"]["databaseId"]
                    ] = names
        return found

    def required(self, branch: str) -> set[str]:
        """The names of the checks that the rules of ``branch`` require."""
        rules = self._api("GET", f"rules/branches/{quote(branch, safe='')}")
        return {
            check["context"]
            for rule in rules
            if rule["type"] == "required_status_checks"
            for check in rule["parameters"]["required_status_checks"]
        }

    def jobs(self, run_id: int) -> list[dict]:
        """The jobs of run ``run_id``'s latest attempt."""
        jobs: list[dict] = []
        page = 1
        while True:
            batch = self._api(
                "GET",
                f"actions/runs/{run_id}/jobs?filter=latest&per_page={PAGE}&page={page}",
            )["jobs"]
            jobs += batch
            if len(batch) < PAGE:
                return jobs
            page += 1

    def names(self, path: str, sha: str) -> set[str]:
        """The literal ``name:`` values, of jobs and steps, in the workflow file ``path`` at commit ``sha``."""
        command = [
            "gh",
            "api",
            "--method",
            "GET",
            f"repos/{self.repository}/contents/{quote(path)}?ref={sha}",
            "-H",
            "Accept: application/vnd.github.raw",
        ]
        done = self._call(command)
        if done.returncode:
            raise RuntimeError(f"gh api {path}@{sha[:12]}: {done.stderr.strip()}")
        return {
            match["name"]
            for line in done.stdout.splitlines()
            if (match := NAME.match(line))
        }

    def cancel(self, run: Run) -> bool:
        """Cancel ``run``; false when GitHub answers HTTP 409, because the run finished first or was never queued."""
        try:
            self._api("POST", f"actions/runs/{run.id}/cancel")
        except RuntimeError as error:
            if "HTTP 409" in str(error):
                return False
            raise
        return True


class Sweeper:
    """Finds and cancels orphaned, failed, doomed and closed runs, remembering what never needs another look."""

    def __init__(self, github: GitHub, apply: bool, limit: int, keep: set[int]):
        self.github = github
        self.apply = apply
        self.limit = limit
        # This run, and the runs already cancelled or that GitHub cannot cancel: none needs another request.
        self.keep = set(keep)
        # Failed runs without a required check left to report: they never hold the queue themselves.
        self.settled: set[int] = set()
        # The head commits of groups with a failed run that was found still holding a required check; once that run
        # is cancelled, its check fails, so the group stays failed.
        self.broken: set[str] = set()
        self._required: dict[str, set[str]] = {}
        self._names: dict[tuple[str, str], set[str]] = {}

    def required(self, branch: str) -> set[str]:
        """The required checks of the branch the merge group ``branch`` merges into."""
        name = target(branch) or branch
        if name not in self._required:
            self._required[name] = self.github.required(name)
        return self._required[name]

    def waiting(self, run: Run, jobs: list[dict], required: set[str]) -> set[str]:
        """The checks in ``required`` that ``run``, with the latest-attempt ``jobs``, has yet to report."""
        done = {job["name"] for job in jobs if job["status"] == "completed"}
        listed = {job["name"] for job in jobs}
        waiting = required & (listed - done)
        # A job that needs others is not listed until they finish; the workflow file says whether the run has one.
        if not waiting and required - listed and run.path.startswith(".github/"):
            key = (run.path, run.sha)
            if key not in self._names:
                self._names[key] = self.github.names(run.path, run.sha)
            waiting = (required & self._names[key]) - done
        return waiting

    def failed(
        self,
        runs: Iterable[Run],
        groups: Mapping[str, str],
        failures: Mapping[str, Mapping[int, set[str]]],
    ) -> list[Run]:
        """The ``runs`` of the merge groups in ``groups`` that have failed, oldest first. ``failures`` holds the
        failed check runs of the groups' head commits, as ``GitHub.failures`` returns them.
        """
        live = {run.id: run for run in runs if run.branch in groups}
        found = []
        # The commits' check runs single out the candidates; each candidate's jobs show whether a required check is
        # still waiting.
        for run_id in sorted(
            {run_id for checks in failures.values() for run_id in checks}
        ):
            run = live.get(run_id)
            if run is None or run_id in self.settled:
                continue
            required = self.required(run.branch)
            jobs = self.github.jobs(run.id)
            if not self.waiting(run, jobs, required):
                self.settled.add(run.id)
            elif any(job["conclusion"] in FAILED for job in jobs):
                found.append(run)
                self.broken.add(run.sha)
        return found

    def cut(
        self,
        groups: Mapping[str, str],
        failures: Mapping[str, Mapping[int, set[str]]],
    ) -> set[str]:
        """The branches of the merge groups in ``groups`` that cannot land: those with a required check that has
        failed, or a failed run found holding one, and every group built on one of them.
        """
        broken = set(self.broken)
        for branch, sha in groups.items():
            required = self.required(branch)
            if any(names & required for names in failures.get(sha, {}).values()):
                broken.add(sha)
        return doomed(groups, broken)

    def sweep(self) -> bool:
        """List or cancel the orphaned, failed, doomed and closed runs once; true while the queue has merge
        groups."""
        # Runs first: a group or pull request created after these calls has no run in the lists, and a group that
        # lands meanwhile is in landed.
        runs = [run for run in self.github.active_runs() if run.id not in self.keep]
        pulls = [
            run
            for run in self.github.active_runs("pull_request")
            if run.id not in self.keep
        ]
        groups = self.github.groups()
        heads = self.github.open_heads()
        landed = self.github.landed({target(run.branch) for run in runs} - {None})
        failures = self.github.failures(groups.values())
        # Failed runs first, since each one holds the queue, then the runs of the groups that cannot land, which
        # hold runners the groups ahead of them need.
        found = [(run, "failed") for run in self.failed(runs, groups, failures)]
        listed = {run.id for run, _ in found}
        cut = self.cut(groups, failures)
        found += [
            (run, "doomed")
            for run in sorted(runs, key=lambda run: run.id)
            if run.branch in cut and run.id not in listed
        ]
        found += [(run, "orphaned") for run in orphaned(runs, groups, landed)]
        found += [(run, "closed") for run in closed(pulls, heads)]
        cancelled = 0
        for run, reason in found[: self.limit]:
            line = f"{run.id} {run.workflow} on {run.branch.removeprefix(QUEUE)} ({reason})"
            if not self.apply:
                print(f"would cancel {line}")
            elif self.github.cancel(run):
                cancelled += 1
                # A cancelled run stays active until its jobs stop; it needs no second request.
                self.keep.add(run.id)
                print(f"cancelled {line}")
            else:
                # The run finished first, or GitHub never queued it and cannot cancel it: asking again changes
                # nothing.
                self.keep.add(run.id)
                print(f"not cancelled: {line}")
        if len(found) > self.limit:
            print(
                f"left {len(found) - self.limit} for the next sweep (limit {self.limit})"
            )
        counts = {
            reason: sum(kind == reason for _, kind in found)
            for reason in ("orphaned", "failed", "doomed", "closed")
        }
        print(
            f"{len(runs)} active merge-group and {len(pulls)} pull-request runs: "
            + ", ".join(f"{count} {reason}" for reason, count in counts.items())
            + (f", {cancelled} cancelled" if self.apply else "")
            + (f"; {len(cut)} of {len(groups)} groups cannot land" if cut else "")
        )
        return bool(groups)


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
    parser.add_argument(
        "--limit", type=int, default=LIMIT, help="per sweep (default: %(default)s)"
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="MINUTES",
        help=f"sweep every {INTERVAL // 60} minutes while the queue has merge groups, for at most MINUTES",
    )
    args = parser.parse_args(argv)

    # CI captures the output through a pipe; without line buffering a long --watch shows nothing until it ends.
    sys.stdout.reconfigure(line_buffering=True)
    # A sweep that runs in the merge queue must not cancel itself when its own group is discarded.
    own = os.environ.get("GITHUB_RUN_ID", "")
    keep = {int(own)} if own.isdigit() else set()
    sweeper = Sweeper(GitHub(args.repo), args.apply, args.limit, keep)
    deadline = time.monotonic() + args.watch * 60
    errors = 0
    while True:
        try:
            busy = sweeper.sweep()
            errors = 0
        except RuntimeError as error:
            errors += 1
            if not args.watch or errors == ERRORS:
                raise
            print(f"sweep failed, retrying: {error}", file=sys.stderr)
            busy = True
        if not busy or time.monotonic() + INTERVAL > deadline:
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
