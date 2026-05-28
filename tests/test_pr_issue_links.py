import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "sync_pr_issue_links.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_pr_issue_links", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, module, issues=None, assign_errors=None):
        self.module = module
        self.issues = dict(issues or {})
        self.assign_errors = dict(assign_errors or {})
        self.assigned = []
        self.updated_bodies = []

    def get_issue(self, number):
        issue = self.issues.get(number)
        if issue is None:
            raise self.module.GitHubApiError(
                "GET", "/issues/{}".format(number), 404, ""
            )
        return issue

    def add_issue_assignee(self, number, login):
        error_status = self.assign_errors.get(number)
        if error_status is not None:
            raise self.module.GitHubApiError(
                "POST", "/issues/{}/assignees".format(number), error_status, ""
            )
        self.assigned.append((number, login))
        self.issues[number].setdefault("assignees", []).append({"login": login})

    def update_pull_body(self, number, body):
        self.updated_bodies.append((number, body))


def issue(number, assignees=None, is_pull=False):
    payload = {
        "number": number,
        "assignees": [{"login": login} for login in assignees or []],
    }
    if is_pull:
        payload["pull_request"] = {}
    return payload


def test_extract_closing_issue_numbers_supports_keywords_and_same_repo_refs():
    module = load_sync_module()
    body = """
Fixes #10 and #11
Resolved: CrossGL/crosstl#12
Closes https://github.com/CrossGL/crosstl/issues/13
Related to #99
Fixes other/repo#14
`Fixes #15`
```
Closes #16
```
<!-- crossgl-pr-issue-links:start -->
## Fixing Issues
- Fixes #17
<!-- crossgl-pr-issue-links:end -->
"""

    numbers = module.extract_closing_issue_numbers(
        "Fixed GH-18", body, "CrossGL/crosstl"
    )

    assert numbers == [18, 10, 11, 12, 13]


def test_sync_assigns_unassigned_issues_and_updates_pr_body():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Improve matrix sync",
        body="Fixes #10\nResolves #11",
        author="alice",
    )
    client = FakeClient(
        module,
        {
            10: issue(10),
            11: issue(11, assignees=["alice"]),
        },
    )

    summary = module.sync_pr_issue_links(client, pr, "CrossGL/crosstl")

    assert summary == {
        "linked": 2,
        "assigned": 1,
        "assignment_skipped": 0,
        "missing_or_pull": 0,
        "body_updated": 1,
    }
    assert client.assigned == [(10, "alice")]
    assert client.updated_bodies == [
        (
            5,
            "\n".join(
                [
                    "Fixes #10",
                    "Resolves #11",
                    "",
                    "<!-- crossgl-pr-issue-links:start -->",
                    "## Fixing Issues",
                    "",
                    "This PR is marked as fixing:",
                    "- Fixes #10",
                    "- Fixes #11",
                    "<!-- crossgl-pr-issue-links:end -->",
                ]
            ),
        )
    ]


def test_sync_removes_stale_managed_section_when_closing_refs_are_removed():
    module = load_sync_module()
    body = "\n".join(
        [
            "No linked issue now.",
            "",
            "<!-- crossgl-pr-issue-links:start -->",
            "## Fixing Issues",
            "",
            "This PR is marked as fixing:",
            "- Fixes #10",
            "<!-- crossgl-pr-issue-links:end -->",
        ]
    )
    pr = module.PullRequestContext(
        number=5,
        title="Improve matrix sync",
        body=body,
        author="alice",
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(client, pr, "CrossGL/crosstl")

    assert summary["linked"] == 0
    assert summary["body_updated"] == 1
    assert client.updated_bodies == [(5, "No linked issue now.")]


def test_sync_skips_pull_request_refs_and_non_assignable_authors():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Fixes #10 and #11",
        body="",
        author="external-user",
    )
    client = FakeClient(
        module,
        {
            10: issue(10, is_pull=True),
            11: issue(11),
        },
        assign_errors={11: 422},
    )

    summary = module.sync_pr_issue_links(client, pr, "CrossGL/crosstl")

    assert summary["linked"] == 2
    assert summary["assigned"] == 0
    assert summary["assignment_skipped"] == 1
    assert summary["missing_or_pull"] == 1
    assert client.assigned == []
    assert "- Fixes #11" in client.updated_bodies[0][1]
    assert "- Fixes #10" not in client.updated_bodies[0][1]


def test_dry_run_does_not_touch_client_but_reports_body_update():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Fixes #10",
        body="",
        author="alice",
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(client, pr, "CrossGL/crosstl", dry_run=True)

    assert summary["linked"] == 1
    assert summary["body_updated"] == 1
    assert client.assigned == []
    assert client.updated_bodies == []
