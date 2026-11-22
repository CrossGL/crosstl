"""The GitHub client and the ``issues`` command."""

import json

import pytest

from tools.coverage import __main__ as cli
from tools.coverage import github, issues

from .issue_cases import bug_for, coverage_report, hlsl_row


class _FakeGitHub(github.GitHub):
    def __init__(self, pages):
        super().__init__("o/r")
        self.pages = pages
        self.calls = []

    def _api(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET":
            return self.pages.pop(0)
        return {"number": 7}


def _item(number, body, **extra):
    return {
        "number": number,
        "state": "open",
        "title": "t",
        "body": body,
        "labels": [{"name": "coverage"}],
    } | extra


def test_github_lists_only_marked_issues_across_pages():
    marked = issues.MARKER.format("tracking/hlsl/output")
    first = [_item(n, "no marker") for n in range(1, 100)] + [_item(100, marked)]
    second = [_item(101, marked, pull_request={}), _item(102, None)]
    client = _FakeGitHub([first, second])
    found = client.managed()
    assert [(i.number, i.key, i.labels) for i in found] == [
        (100, "tracking/hlsl/output", ("coverage",))
    ]
    assert [call[1] for call in client.calls] == [
        "issues?state=all&labels=coverage&per_page=100&page=1",
        "issues?state=all&labels=coverage&per_page=100&page=2",
    ]


def test_github_comments_before_closing_and_adds_labels_last():
    client = _FakeGitHub([])
    action = issues.Action("close", "k", "t", "b", ("bug",), 5, "why", "completed")
    assert client.apply(action) == 5
    assert client.calls == [
        ("POST", "issues/5/comments", {"body": "why"}),
        (
            "PATCH",
            "issues/5",
            {"title": "t", "body": "b", "state": "closed", "state_reason": "completed"},
        ),
        ("POST", "issues/5/labels", {"labels": ["bug"]}),
    ]
    assert client.apply(issues.Action("create", "k", "t", "b", ("bug",))) == 7
    assert client.calls[-1] == (
        "POST",
        "issues",
        {"title": "t", "body": "b", "labels": ["bug"]},
    )


def test_github_waits_a_second_between_changes(monkeypatch):
    clock = iter([100.0, 100.0, 100.25, 100.25, 107.0, 107.0])
    slept = []
    monkeypatch.setattr(github.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(github.time, "sleep", slept.append)
    monkeypatch.setattr(
        github.subprocess,
        "run",
        lambda *args, **kwargs: github.subprocess.CompletedProcess(args, 0, "[]", ""),
    )
    client = github.GitHub("o/r")
    client._api("GET", "issues")
    client._api("PATCH", "issues/1", {})
    client._api("GET", "issues")
    client._api("POST", "issues/1/comments", {})
    client._api("PATCH", "issues/1", {})
    assert slept == [0.0, 0.75, 0.0]


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_prints_the_plan_and_applies_it_unless_dry_run(
    tmp_path, monkeypatch, capsys, dry_run
):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage_report(hlsl_row())))
    applied = []
    monkeypatch.setattr(github.GitHub, "managed", lambda self: [])
    monkeypatch.setattr(
        github.GitHub, "apply", lambda self, action: applied.append(action.key)
    )
    argv = ["issues", str(path), "--repository", "o/r"] + (
        ["--dry-run"] if dry_run else []
    )
    assert cli.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("create new issue  HLSL output coverage: 33.3 % (2 gaps)")
    # The fake repository never records the bug, so the applied run plans its follow-up pass on an empty list.
    expected = 2 if dry_run else 3
    assert out[-1] == f"{expected} issue changes" + (" planned" if dry_run else "")
    assert applied == (
        []
        if dry_run
        else ["tracking/hlsl/output", bug_for("hlsl", "b"), "tracking/hlsl/output"]
    )


def test_cli_reports_github_errors(tmp_path, monkeypatch, capsys):
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage_report()))

    def fail(self):
        raise github.GitHubError("gh api GET issues: HTTP 401")

    monkeypatch.setattr(github.GitHub, "managed", fail)
    assert cli.main(["issues", str(path)]) == 2
    assert "HTTP 401" in capsys.readouterr().err


class _Answering(github.GitHub):
    """A client whose GraphQL queries return ``answer`` and whose REST calls return ``status``."""

    def __init__(self, answer=None, status="ahead"):
        super().__init__("o/r")
        self.answer = answer
        self.status = status
        self.calls = []

    def _query(self, query, variables):
        self.calls.append(("graphql", variables))
        return self.answer

    def _api(self, method, path, payload=None):
        self.calls.append((method, path))
        return {"status": self.status}


def _closed_by(*closers):
    nodes = [{"closer": closer} for closer in closers]
    return {"repository": {"issue": {"timelineItems": {"nodes": nodes}}}}


@pytest.mark.parametrize(
    "answer, commit",
    [
        (_closed_by({"mergeCommit": {"oid": "abc"}}), "abc"),
        (_closed_by({"oid": "def"}), "def"),
        (_closed_by(None), ""),
        (_closed_by({}), ""),
        (_closed_by(), ""),
    ],
)
def test_github_names_the_commit_that_closed_an_issue(answer, commit):
    client = _Answering(answer)
    assert client.closing_commit(12) == commit
    assert client.calls == [("graphql", {"owner": "o", "name": "r", "number": 12})]


@pytest.mark.parametrize(
    "status, contained",
    [("ahead", True), ("identical", True), ("behind", False), ("diverged", False)],
)
def test_github_tells_whether_a_commit_contains_another(status, contained):
    client = _Answering(status=status)
    assert client.contains("head", "fix") is contained
    assert client.calls == [("GET", "compare/fix...head?per_page=1")]


def test_github_queries_send_their_variables(monkeypatch):
    sent = []

    def run(command, **kwargs):
        sent.append((command, json.loads(kwargs["input"])))
        return github.subprocess.CompletedProcess(command, 0, '{"data": {"x": 1}}', "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.GitHub("o/r")._query("q", {"n": 1}) == {"x": 1}
    assert sent == [
        (
            ["gh", "api", "graphql", "--input", "-"],
            {"query": "q", "variables": {"n": 1}},
        )
    ]
