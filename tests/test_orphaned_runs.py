import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import orphaned_runs
from tools.orphaned_runs import GitHub, Run, Sweeper, closed, orphaned, target

GROUP = "gh-readonly-queue/main/pr-12-" + "a" * 40
LIVE = "gh-readonly-queue/main/pr-13-" + "b" * 40
WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"
WORKFLOW = WORKFLOWS / "orphaned-runs.yml"
REQUIRED = "MLX validation result"


def test_target_is_the_branch_a_merge_group_merges_into():
    assert target(GROUP) == "main"
    assert target("gh-readonly-queue/release/4.x/pr-7-" + "c" * 40) == "release/4.x"
    assert target("main") is None
    assert target("gh-readonly-queue/main") is None


def test_only_runs_of_groups_that_are_gone_and_did_not_land_are_orphaned():
    runs = [
        Run(5, "Coverage", GROUP, "1" * 40),
        Run(3, "Lint", GROUP, "1" * 40),
        Run(4, "Lint", LIVE, "2" * 40),
        Run(6, "Corpus", "gh-readonly-queue/main/pr-11-" + "d" * 40, "3" * 40),
        Run(7, "Lint", "repo/topic", "4" * 40),
    ]
    found = orphaned(runs, groups={LIVE}, landed={"3" * 40})
    assert [run.id for run in found] == [3, 5]


class FakeGh:
    """Answers ``gh api`` calls from a table of paths, and records them."""

    def __init__(self, answers: dict[str, object]):
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def __call__(self, command, **kwargs):
        if command[2] == "graphql":
            values = dict(arg.split("=", 1) for arg in command[4::2])
            shas = sorted(value for key, value in values.items() if key[1:].isdigit())
            self.calls.append(("GRAPHQL", ",".join(shas)))
            answer = self.answers.get("graphql", {})
            answer = {"data": {"repository": answer}}
        else:
            method, path = command[3], command[4].removeprefix("repos/o/r/")
            self.calls.append((method, path))
            answer = self.answers.get(
                path, {"workflow_runs": []} if path.startswith("actions/runs?") else []
            )
        if isinstance(answer, subprocess.CompletedProcess):
            return answer
        return subprocess.CompletedProcess(command, 0, json.dumps(answer), "")


def _run(id, branch, sha, repository=None):
    run = {"id": id, "name": "Coverage", "head_branch": branch, "head_sha": sha}
    if repository is not None:
        run["head_repository"] = repository and {"full_name": repository}
    return run


def test_active_runs_reads_every_state_and_page_once_per_run(monkeypatch):
    monkeypatch.setattr(orphaned_runs, "PAGE", 2)
    first = [_run(1, GROUP, "1" * 40), _run(2, GROUP, "1" * 40)]
    gh = FakeGh(
        {
            "actions/runs?event=merge_group&status=queued&per_page=2&page=1": {
                "workflow_runs": first
            },
            "actions/runs?event=merge_group&status=queued&per_page=2&page=2": {
                "workflow_runs": [_run(3, LIVE, "2" * 40)]
            },
            # A run that starts between two requests is listed in both states.
            "actions/runs?event=merge_group&status=in_progress&per_page=2&page=1": {
                "workflow_runs": [_run(2, GROUP, "1" * 40)]
            },
        }
    )
    runs = GitHub("o/r", run=gh).active_runs()
    assert sorted(run.id for run in runs) == [1, 2, 3]
    assert {path.split("&")[1] for _, path in gh.calls} == {
        f"status={status}" for status in orphaned_runs.ACTIVE
    }


def test_pull_request_runs_carry_the_repository_of_their_head():
    gh = FakeGh(
        {
            "actions/runs?event=pull_request&status=queued&per_page=100&page=1": {
                "workflow_runs": [
                    _run(1, "topic", "1" * 40, "o/r"),
                    _run(2, "main", "2" * 40, "fork/r"),
                    # A run from a fork that has since been deleted.
                    _run(3, "topic", "3" * 40, ""),
                ]
            }
        }
    )
    runs = GitHub("o/r", run=gh).active_runs("pull_request")
    assert sorted((run.id, run.repository, run.branch) for run in runs) == [
        (1, "o/r", "topic"),
        (2, "fork/r", "main"),
        (3, "", "topic"),
    ]
    assert {path.split("&")[0] for _, path in gh.calls} == {
        "actions/runs?event=pull_request"
    }


def test_open_heads_reads_every_page(monkeypatch):
    monkeypatch.setattr(orphaned_runs, "PAGE", 2)
    gh = FakeGh(
        {
            "pulls?state=open&per_page=2&page=1": [
                {"head": {"repo": {"full_name": "o/r"}, "ref": "topic"}},
                {"head": {"repo": {"full_name": "fork/r"}, "ref": "main"}},
            ],
            "pulls?state=open&per_page=2&page=2": [
                {"head": {"repo": None, "ref": "gone"}}
            ],
        }
    )
    assert GitHub("o/r", run=gh).open_heads() == {
        ("o/r", "topic"),
        ("fork/r", "main"),
        ("", "gone"),
    }


def test_only_runs_without_an_open_pull_request_are_closed():
    runs = [
        Run(5, "Corpus", "merged", "1" * 40, "o/r"),
        Run(3, "Lint", "open", "2" * 40, "o/r"),
        # Same branch name, other repository: a fork's pull request that closed.
        Run(4, "Lint", "open", "3" * 40, "fork/r"),
        Run(2, "Lint", "abandoned", "4" * 40, "fork/r"),
    ]
    found = closed(runs, heads={("o/r", "open"), ("fork/r", "abandoned")})
    assert [run.id for run in found] == [4, 5]


def test_groups_and_landed_commits():
    gh = FakeGh(
        {
            "git/matching-refs/heads/gh-readonly-queue/?per_page=100&page=1": [
                {"ref": f"refs/heads/{LIVE}"}
            ],
            "commits?sha=release%2F4.x&per_page=100": [{"sha": "5" * 40}],
            "commits?sha=main&per_page=100": [{"sha": "6" * 40}],
        }
    )
    github = GitHub("o/r", run=gh)
    assert github.groups() == {LIVE}
    assert github.landed({"main", "release/4.x"}) == {"5" * 40, "6" * 40}


def test_cancel_reports_a_run_that_finished_first():
    finished = subprocess.CompletedProcess(
        [], 1, "", "gh: Cannot cancel a workflow run that is completed. (HTTP 409)"
    )
    gh = FakeGh({"actions/runs/9/cancel": finished, "actions/runs/8/cancel": None})
    github = GitHub("o/r", run=gh)
    assert github.cancel(Run(8, "Lint", GROUP, "1" * 40)) is True
    assert github.cancel(Run(9, "Lint", GROUP, "1" * 40)) is False
    assert gh.calls == [
        ("POST", "actions/runs/8/cancel"),
        ("POST", "actions/runs/9/cancel"),
    ]

    gh.answers["actions/runs/9/cancel"] = subprocess.CompletedProcess(
        [], 1, "", "gh: Resource not accessible by integration (HTTP 403)"
    )
    with pytest.raises(RuntimeError, match="HTTP 403"):
        github.cancel(Run(9, "Lint", GROUP, "1" * 40))


def _suites(*runs):
    return {
        "checkSuites": {
            "nodes": [
                {
                    "workflowRun": {"databaseId": id},
                    "checkRuns": {"totalCount": failed},
                }
                for id, failed in runs
            ] + [{"workflowRun": None, "checkRuns": {"totalCount": 0}}]
        }
    }


def test_failing_asks_once_for_the_failed_runs_of_every_commit():
    gh = FakeGh({"graphql": {"c0": _suites((11, 1), (12, 0)), "c1": None}})
    github = GitHub("o/r", run=gh)
    assert github.failing(["2" * 40, "1" * 40, "1" * 40]) == {11}
    assert gh.calls == [("GRAPHQL", "1" * 40 + "," + "2" * 40)]
    assert github.failing([]) == set()
    assert len(gh.calls) == 1


def test_failing_declares_one_variable_per_commit():
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, '{"data": {"repository": {}}}', ""
        )

    GitHub("o/r", run=run).failing(["1" * 40, "2" * 40])
    values = dict(arg.split("=", 1) for arg in commands[0][4::2])
    query = values.pop("query")
    assert values == {"owner": "o", "name": "r", "c0": "1" * 40, "c1": "2" * 40}
    for index in (0, 1):
        assert f"$c{index}: GitObjectID!" in query
        assert f"c{index}: object(oid: $c{index}) {{ ...suites }}" in query
    assert "fragment suites on Commit" in query


def test_required_checks_and_latest_jobs(monkeypatch):
    monkeypatch.setattr(orphaned_runs, "PAGE", 2)
    jobs = [{"name": "a", "status": "completed"}, {"name": "b", "status": "queued"}]
    gh = FakeGh(
        {
            "rules/branches/release%2F4.x": [
                {"type": "pull_request", "parameters": {}},
                {
                    "type": "required_status_checks",
                    "parameters": {
                        "required_status_checks": [
                            {"context": REQUIRED},
                            {"context": "Lint result"},
                        ]
                    },
                },
            ],
            "actions/runs/5/jobs?filter=latest&per_page=2&page=1": {"jobs": jobs},
            "actions/runs/5/jobs?filter=latest&per_page=2&page=2": {"jobs": jobs[:1]},
        }
    )
    github = GitHub("o/r", run=gh)
    assert github.required("release/4.x") == {REQUIRED, "Lint result"}
    assert github.jobs(5) == jobs + jobs[:1]


class StubGitHub:
    """A queue with one live group (LIVE) and one discarded group (GROUP)."""

    def __init__(self):
        self.runs = [
            Run(1, "Corpus", GROUP, "1" * 40),
            Run(2, "Coverage", LIVE, "2" * 40),
            Run(3, "Project Demo Testing", LIVE, "2" * 40),
            Run(4, "Corpus", LIVE, "2" * 40),
            Run(5, "Lint", LIVE, "2" * 40),
        ]
        self.pulls = [
            Run(6, "Corpus", "merged", "3" * 40, "o/r"),
            Run(7, "Lint", "open", "4" * 40, "o/r"),
        ]
        self.groups_ = {LIVE}
        self.heads = {("o/r", "open")}
        # 1 is not live; 5 was re-run and its latest attempt has not failed.
        self.failing_ = {1, 3, 4, 5}
        self.jobs_ = {
            # The required result job waits for the slowest leg.
            3: [
                {"name": "leg", "status": "completed", "conclusion": "failure"},
                {"name": "leg", "status": "in_progress", "conclusion": None},
                {"name": REQUIRED, "status": "queued", "conclusion": None},
            ],
            # Corpus is not required, so its result decides nothing.
            4: [
                {"name": "leg", "status": "completed", "conclusion": "timed_out"},
                {"name": "Corpus result", "status": "queued", "conclusion": None},
            ],
            5: [
                {"name": "leg", "status": "in_progress", "conclusion": None},
                {"name": REQUIRED, "status": "queued", "conclusion": None},
            ],
        }
        self.calls: list[tuple] = []

    def active_runs(self, event="merge_group"):
        return list(self.runs if event == "merge_group" else self.pulls)

    def groups(self):
        return set(self.groups_)

    def open_heads(self):
        return set(self.heads)

    def landed(self, branches):
        self.calls.append(("landed", tuple(sorted(branches))))
        return {"9" * 40}

    def failing(self, shas):
        self.calls.append(("failing", tuple(sorted(set(shas)))))
        return set(self.failing_)

    def required(self, branch):
        self.calls.append(("required", branch))
        return {REQUIRED}

    def jobs(self, run_id):
        self.calls.append(("jobs", run_id))
        return self.jobs_[run_id]

    def cancel(self, run):
        self.calls.append(("cancel", run.id))
        return True


def test_a_live_run_has_failed_while_a_failed_job_keeps_a_required_check_waiting():
    github = StubGitHub()
    sweeper = Sweeper(github, apply=False, limit=10, keep=set())
    assert [run.id for run in sweeper.failed(github.runs, {LIVE})] == [3]
    assert ("failing", ("2" * 40,)) in github.calls
    assert [call for call in github.calls if call[0] == "jobs"] == [
        ("jobs", 3),
        ("jobs", 4),
        ("jobs", 5),
    ]
    # Corpus's run can never hold the queue, so it is not looked at again; Lint's can still fail.
    assert sweeper.keep == {4}
    github.calls.clear()
    sweeper.failed([run for run in github.runs if run.id not in sweeper.keep], {LIVE})
    assert [call for call in github.calls if call[0] != "failing"] == [
        ("jobs", 3),
        ("jobs", 5),
    ]


def test_sweep_cancels_failed_runs_first_and_each_run_once(capsys):
    github = StubGitHub()
    sweeper = Sweeper(github, apply=True, limit=10, keep={2})
    assert sweeper.sweep() is True
    assert [call[1] for call in github.calls if call[0] == "cancel"] == [3, 1, 6]
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"cancelled 3 Project Demo Testing on {LIVE.removeprefix('gh-readonly-queue/')} (failed)",
        f"cancelled 1 Corpus on {GROUP.removeprefix('gh-readonly-queue/')} (orphaned)",
        "cancelled 6 Corpus on merged (closed)",
        "4 active merge-group and 2 pull-request runs: "
        "1 orphaned, 1 failed, 1 closed, 3 cancelled",
    ]
    # A cancelled run stays active for a while; the next sweep leaves it alone.
    github.calls.clear()
    assert sweeper.sweep() is True
    assert not [call for call in github.calls if call[0] == "cancel"]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "1 active merge-group and 1 pull-request runs: "
        "0 orphaned, 0 failed, 0 closed, 0 cancelled"
    )
    # Once the queue is empty, a watch can stop, even while pull requests have runs.
    github.runs, github.groups_ = [], set()
    assert sweeper.sweep() is False


def _main_answers(cancel=None):
    answers = {
        "actions/runs?event=merge_group&status=queued&per_page=100&page=1": {
            "workflow_runs": [
                _run(1, GROUP, "1" * 40),
                _run(2, GROUP, "1" * 40),
                _run(3, LIVE, "2" * 40),
                _run(4, LIVE, "2" * 40),
            ]
        },
        "actions/runs?event=pull_request&status=in_progress&per_page=100&page=1": {
            "workflow_runs": [
                _run(5, "merged", "3" * 40, "o/r"),
                _run(6, "open", "4" * 40, "o/r"),
            ]
        },
        "git/matching-refs/heads/gh-readonly-queue/?per_page=100&page=1": [
            {"ref": f"refs/heads/{LIVE}"}
        ],
        "pulls?state=open&per_page=100&page=1": [
            {"head": {"repo": {"full_name": "o/r"}, "ref": "open"}}
        ],
        "commits?sha=main&per_page=100": [{"sha": "9" * 40}],
        "graphql": {"c0": _suites((3, 1), (4, 0))},
        "rules/branches/main": [
            {
                "type": "required_status_checks",
                "parameters": {"required_status_checks": [{"context": REQUIRED}]},
            }
        ],
        "actions/runs/3/jobs?filter=latest&per_page=100&page=1": {
            "jobs": [
                {"name": "leg", "status": "completed", "conclusion": "failure"},
                {"name": REQUIRED, "status": "queued", "conclusion": None},
            ]
        },
    }
    answers.update(cancel or {})
    return answers


def test_main_lists_without_cancelling(monkeypatch, capsys):
    gh = FakeGh(_main_answers())
    monkeypatch.setattr(orphaned_runs, "GitHub", lambda repo: GitHub(repo, run=gh))
    assert orphaned_runs.main(["--repo", "o/r"]) == 0
    assert not [call for call in gh.calls if call[0] == "POST"]
    assert capsys.readouterr().out.splitlines() == [
        f"would cancel 3 Coverage on {LIVE.removeprefix('gh-readonly-queue/')} (failed)",
        f"would cancel 1 Coverage on {GROUP.removeprefix('gh-readonly-queue/')} (orphaned)",
        f"would cancel 2 Coverage on {GROUP.removeprefix('gh-readonly-queue/')} (orphaned)",
        "would cancel 5 Coverage on merged (closed)",
        "4 active merge-group and 2 pull-request runs: 2 orphaned, 1 failed, 1 closed",
    ]


def test_main_cancels_up_to_the_limit(monkeypatch, capsys):
    cancels = {"actions/runs/3/cancel": None, "actions/runs/1/cancel": None}
    gh = FakeGh(_main_answers(cancels))
    monkeypatch.setattr(orphaned_runs, "GitHub", lambda repo: GitHub(repo, run=gh))
    assert orphaned_runs.main(["--repo", "o/r", "--apply", "--limit", "2"]) == 0
    assert [call for call in gh.calls if call[0] == "POST"] == [
        ("POST", "actions/runs/3/cancel"),
        ("POST", "actions/runs/1/cancel"),
    ]
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == [
        "left 2 for the next sweep (limit 2)",
        "4 active merge-group and 2 pull-request runs: "
        "2 orphaned, 1 failed, 1 closed, 2 cancelled",
    ]


class ScriptedSweeper:
    """Stands in for the Sweeper class: each sweep returns or raises the next result."""

    def __init__(self, results):
        self.results = list(results)
        self.sweeps = 0

    def __call__(self, github, apply, limit, keep):
        self.keep = keep
        return self

    def sweep(self):
        self.sweeps += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def clock(monkeypatch):
    """A fake clock that ``time.sleep`` advances; returns the list of sleeps."""
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(orphaned_runs.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(orphaned_runs.time, "sleep", sleep)
    monkeypatch.setattr(orphaned_runs, "GitHub", lambda repo: None)
    return sleeps


@pytest.mark.parametrize(
    "argv,results,sweeps",
    [
        ([], [True], 1),
        # Stops once the queue is empty.
        (["--watch", "60"], [True, True, False], 3),
        # Sweeps at 0, 5 and 10 minutes; a fourth would start after the deadline.
        (["--watch", "12"], [True] * 5, 3),
    ],
)
def test_watch_sweeps_while_the_queue_has_groups(
    monkeypatch, clock, argv, results, sweeps
):
    sweeper = ScriptedSweeper(results)
    monkeypatch.setattr(orphaned_runs, "Sweeper", sweeper)
    assert orphaned_runs.main(argv) == 0
    assert sweeper.sweeps == sweeps
    assert clock == [orphaned_runs.INTERVAL] * (sweeps - 1)


def test_watch_survives_a_few_failed_sweeps(monkeypatch, clock):
    error = RuntimeError("gh api GET actions/runs: HTTP 502")
    sweeper = ScriptedSweeper([error, True, error, error, error, True])
    monkeypatch.setattr(orphaned_runs, "Sweeper", sweeper)
    with pytest.raises(RuntimeError, match="HTTP 502"):
        orphaned_runs.main(["--watch", "300"])
    assert sweeper.sweeps == 2 + orphaned_runs.ERRORS

    sweeper = ScriptedSweeper([error])
    monkeypatch.setattr(orphaned_runs, "Sweeper", sweeper)
    with pytest.raises(RuntimeError):
        orphaned_runs.main([])


def test_a_sweep_never_cancels_its_own_run(monkeypatch, clock):
    sweeper = ScriptedSweeper([False, False])
    monkeypatch.setattr(orphaned_runs, "Sweeper", sweeper)
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    orphaned_runs.main([])
    assert sweeper.keep == {77}
    monkeypatch.delenv("GITHUB_RUN_ID")
    orphaned_runs.main([])
    assert sweeper.keep == set()


def test_the_workflow_sweeps_whenever_the_queue_creates_a_group():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # Replacing discarded groups creates new ones, so merge_group is when orphans appear; schedules start late.
    assert {"merge_group", "schedule", "workflow_dispatch"} <= set(workflow[True])
    jobs = workflow["jobs"]
    cancel = jobs["cancel"]
    assert cancel["if"] == "github.event_name != 'pull_request'"
    assert cancel["permissions"] == {
        "actions": "write",
        "checks": "read",
        "contents": "read",
    }
    assert jobs["list"]["permissions"] == {
        "actions": "read",
        "checks": "read",
        "contents": "read",
    }
    # The job outlives its watch, so the last sweep is not cut off.
    watch = re.search(r"--apply --watch (\d+)$", cancel["steps"][-1]["run"].strip())
    assert watch and cancel["timeout-minutes"] >= int(watch[1]) + 10
    # Every sweep shares one group, so a burst of new groups runs one sweep and queues the newest.
    concurrency = workflow["concurrency"]
    assert concurrency["group"] == (
        "orphaned-runs-${{ github.event.pull_request.number || 'sweep' }}"
    )
    assert (
        concurrency["cancel-in-progress"]
        == "${{ github.event_name == 'pull_request' }}"
    )


def test_a_failed_job_in_the_merge_queue_fails_its_workflow():
    # A failed run is cancelled because it cannot pass; a job allowed to fail would make that wrong.
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow[True]
        if "merge_group" not in ([triggers] if isinstance(triggers, str) else triggers):
            continue
        for name, job in workflow["jobs"].items():
            assert "continue-on-error" not in job, (path.name, name)


def test_no_workflow_runs_when_a_pull_request_closes():
    # The sweep cancels the runs of closed pull requests, so a job started by closing one would never finish.
    for path in sorted(WORKFLOWS.glob("*.yml")):
        triggers = yaml.safe_load(path.read_text(encoding="utf-8"))[True]
        if not isinstance(triggers, dict):
            continue
        for event in ("pull_request", "pull_request_target"):
            types = (triggers.get(event) or {}).get("types", [])
            assert "closed" not in types, (path.name, event)
