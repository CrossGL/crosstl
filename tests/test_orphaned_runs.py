import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import orphaned_runs
from tools.orphaned_runs import GitHub, Run, orphaned, target

GROUP = "gh-readonly-queue/main/pr-12-" + "a" * 40
LIVE = "gh-readonly-queue/main/pr-13-" + "b" * 40
WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/orphaned-runs.yml"


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
        method, path = command[3], command[4].removeprefix("repos/o/r/")
        self.calls.append((method, path))
        answer = self.answers.get(
            path, {"workflow_runs": []} if path.startswith("actions/runs?") else []
        )
        if isinstance(answer, subprocess.CompletedProcess):
            return answer
        return subprocess.CompletedProcess(command, 0, json.dumps(answer), "")


def _run(id, branch, sha):
    return {"id": id, "name": "Coverage", "head_branch": branch, "head_sha": sha}


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


def _main_answers(cancel=None):
    answers = {
        "actions/runs?event=merge_group&status=queued&per_page=100&page=1": {
            "workflow_runs": [
                _run(1, GROUP, "1" * 40),
                _run(2, GROUP, "1" * 40),
                _run(3, LIVE, "2" * 40),
            ]
        },
        "git/matching-refs/heads/gh-readonly-queue/?per_page=100&page=1": [
            {"ref": f"refs/heads/{LIVE}"}
        ],
        "commits?sha=main&per_page=100": [{"sha": "9" * 40}],
    }
    answers.update(cancel or {})
    return answers


def test_main_lists_without_cancelling(monkeypatch, capsys):
    gh = FakeGh(_main_answers())
    monkeypatch.setattr(orphaned_runs, "GitHub", lambda repo: GitHub(repo, run=gh))
    assert orphaned_runs.main(["--repo", "o/r"]) == 0
    assert not [call for call in gh.calls if call[0] == "POST"]
    out = capsys.readouterr().out.splitlines()
    assert (
        out[0]
        == f"would cancel 1 Coverage on {GROUP.removeprefix('gh-readonly-queue/')}"
    )
    assert out[-1] == "2 of 3 active merge-group runs orphaned"


def test_main_cancels_up_to_the_limit(monkeypatch, capsys):
    gh = FakeGh(_main_answers({"actions/runs/1/cancel": None}))
    monkeypatch.setattr(orphaned_runs, "GitHub", lambda repo: GitHub(repo, run=gh))
    assert orphaned_runs.main(["--repo", "o/r", "--apply", "--limit", "1"]) == 0
    assert [call for call in gh.calls if call[0] == "POST"] == [
        ("POST", "actions/runs/1/cancel")
    ]
    out = capsys.readouterr().out.splitlines()
    assert out[-2:] == [
        "left 1 for the next run (limit 1)",
        "2 of 3 active merge-group runs orphaned, 1 cancelled",
    ]


def test_the_workflow_sweeps_whenever_the_queue_creates_a_group():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # Replacing discarded groups creates new ones, so merge_group is when orphans appear; schedules start late.
    assert {"merge_group", "schedule", "workflow_dispatch"} <= set(workflow[True])
    jobs = workflow["jobs"]
    assert jobs["cancel"]["if"] == "github.event_name != 'pull_request'"
    assert jobs["cancel"]["permissions"] == {"actions": "write", "contents": "read"}
    # Every sweep shares one group, so a burst of new groups runs one sweep and queues the newest.
    concurrency = workflow["concurrency"]
    assert concurrency["group"] == (
        "orphaned-runs-${{ github.event.pull_request.number || 'sweep' }}"
    )
    assert (
        concurrency["cancel-in-progress"]
        == "${{ github.event_name == 'pull_request' }}"
    )
