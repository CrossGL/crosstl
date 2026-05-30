import importlib.util
import json
import sys
from pathlib import Path

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
    def __init__(
        self,
        module,
        issues=None,
        assign_errors=None,
        pull_files=None,
        json_files=None,
        support_issues=None,
    ):
        self.module = module
        self.issues = dict(issues or {})
        self.assign_errors = dict(assign_errors or {})
        self.pull_files = list(pull_files or [])
        self.json_files = dict(json_files or {})
        self.support_issues = list(support_issues or [])
        self.assigned = []
        self.updated_bodies = []

    def get_issue(self, number):
        issue = self.issues.get(number)
        if issue is None:
            raise self.module.GitHubApiError("GET", f"/issues/{number}", 404, "")
        return issue

    def add_issue_assignee(self, number, login):
        error_status = self.assign_errors.get(number)
        if error_status is not None:
            raise self.module.GitHubApiError(
                "POST", f"/issues/{number}/assignees", error_status, ""
            )
        self.assigned.append((number, login))
        self.issues[number].setdefault("assignees", []).append({"login": login})

    def update_pull_body(self, number, body):
        self.updated_bodies.append((number, body))

    def list_pull_files(self, number):
        return list(self.pull_files)

    def read_json_file(self, repo, path, ref):
        return self.json_files[(repo, path, ref)]

    def list_open_support_issues(self):
        return list(self.support_issues)


def issue(number, assignees=None, is_pull=False, body=""):
    payload = {
        "number": number,
        "assignees": [{"login": login} for login in assignees or []],
        "body": body,
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
        "support_closures": 0,
        "support_references": 0,
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
                    "<!-- crossgl-pr-issue-links:start -->",
                    "Closes #10",
                    "Closes #11",
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


def test_sync_deduplicates_manual_closing_lines_into_managed_section():
    module = load_sync_module()
    body = "\n".join(
        [
            "Implements reviewed support work.",
            "",
            "Closes #10",
            "- Fixes #10",
        ]
    )
    pr = module.PullRequestContext(
        number=5,
        title="Improve matrix sync",
        body=body,
        author="alice",
    )
    client = FakeClient(module, {10: issue(10)})

    summary = module.sync_pr_issue_links(client, pr, "CrossGL/crosstl")

    assert summary["linked"] == 1
    assert summary["body_updated"] == 1
    assert client.updated_bodies == [
        (
            5,
            "\n".join(
                [
                    "Implements reviewed support work.",
                    "",
                    "<!-- crossgl-pr-issue-links:start -->",
                    "Closes #10",
                    "<!-- crossgl-pr-issue-links:end -->",
                ]
            ),
        )
    ]


def test_sync_adds_support_matrix_closures_from_removed_backlog_rows(
    tmp_path, monkeypatch
):
    module = load_sync_module()
    base_matrix = {
        "backlog": [
            {
                "backend_id": "metal",
                "feature_id": "language.wave_intrinsics",
            }
        ]
    }
    head_matrix = {"backlog": []}
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(base_matrix), encoding="utf-8")
    monkeypatch.setattr(module, "SUPPORT_MATRIX_PATH", matrix_path)
    marker = (
        "<!-- crossgl-support-issue-sync: backlog:metal:language.wave_intrinsics -->"
    )
    pr = module.PullRequestContext(
        number=5,
        title="Mark Metal wave intrinsics supported",
        body="Support matrix update.",
        author="alice",
        head_repo="CrossGL/crosstl",
        head_sha="abc123",
        changed_files=("support/generated/support-matrix.json",),
    )
    client = FakeClient(
        module,
        issues={498: issue(498)},
        json_files={
            (
                "CrossGL/crosstl",
                "support/generated/support-matrix.json",
                "abc123",
            ): head_matrix
        },
        support_issues=[issue(498, body=marker)],
    )

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        sync_support_closures=True,
        enforce_support_traceability=True,
    )

    assert summary["linked"] == 1
    assert summary["support_closures"] == 1
    assert summary["traceability_satisfied"] == 1
    assert client.updated_bodies == [
        (
            5,
            "\n".join(
                [
                    "Support matrix update.",
                    "",
                    "<!-- crossgl-pr-issue-links:start -->",
                    "Closes #498",
                    "<!-- crossgl-pr-issue-links:end -->",
                ]
            ),
        )
    ]


def test_sync_adds_support_matrix_refs_from_changed_backlog_rows(tmp_path, monkeypatch):
    module = load_sync_module()
    base_matrix = {
        "features": [
            {
                "id": "texture.projected",
                "support": {
                    "metal": {
                        "status": "partial",
                        "notes": "Planar projection is supported.",
                        "evidence": ["tests/old.py::test_planar_projection"],
                    }
                },
            }
        ],
        "backlog": [
            {
                "backend_id": "metal",
                "feature_id": "texture.projected",
                "status": "partial",
                "notes": "Planar projection is supported.",
            }
        ],
    }
    head_matrix = {
        "features": [
            {
                "id": "texture.projected",
                "support": {
                    "metal": {
                        "status": "partial",
                        "notes": "Planar and cube-shadow projection are supported.",
                        "evidence": [
                            "tests/old.py::test_planar_projection",
                            "tests/new.py::test_cube_shadow_projection",
                        ],
                    }
                },
            }
        ],
        "backlog": [
            {
                "backend_id": "metal",
                "feature_id": "texture.projected",
                "status": "partial",
                "notes": "Planar and cube-shadow projection are supported.",
            }
        ],
    }
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(base_matrix), encoding="utf-8")
    monkeypatch.setattr(module, "SUPPORT_MATRIX_PATH", matrix_path)
    marker = "<!-- crossgl-support-issue-sync: backlog:metal:texture.projected -->"
    pr = module.PullRequestContext(
        number=5,
        title="Expand Metal projected texture support",
        body="Support matrix update.",
        author="alice",
        head_repo="CrossGL/crosstl",
        head_sha="abc123",
        changed_files=("support/generated/support-matrix.json",),
    )
    client = FakeClient(
        module,
        json_files={
            (
                "CrossGL/crosstl",
                "support/generated/support-matrix.json",
                "abc123",
            ): head_matrix
        },
        support_issues=[issue(432, body=marker)],
    )

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        sync_support_references=True,
        enforce_support_traceability=True,
    )

    assert summary["linked"] == 0
    assert summary["support_closures"] == 0
    assert summary["support_references"] == 1
    assert summary["traceability_satisfied"] == 1
    assert client.assigned == []
    assert client.updated_bodies == [
        (
            5,
            "\n".join(
                [
                    "Support matrix update.",
                    "",
                    "<!-- crossgl-pr-issue-links:start -->",
                    "Refs #432",
                    "<!-- crossgl-pr-issue-links:end -->",
                ]
            ),
        )
    ]


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
    assert summary["support_closures"] == 0
    assert summary["support_references"] == 0
    assert summary["assigned"] == 0
    assert summary["assignment_skipped"] == 1
    assert summary["missing_or_pull"] == 1
    assert client.assigned == []
    assert "Closes #11" in client.updated_bodies[0][1]
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


def test_github_client_lists_pull_files_across_pages():
    module = load_sync_module()
    client = module.GitHubClient("CrossGL/crosstl", "token")
    pages = []

    def fake_request(method, path, payload=None, query=None):
        assert method == "GET"
        assert path == "/repos/CrossGL/crosstl/pulls/5/files"
        assert payload is None
        pages.append(query["page"])
        if query["page"] == 1:
            return [{"filename": "tools/support_matrix.py"}] * 100, {}
        return [{"filename": "README.md"}], {}

    client.request = fake_request

    assert client.list_pull_files(5) == ["tools/support_matrix.py"] * 100 + [
        "README.md"
    ]
    assert pages == [1, 2]


def test_traceability_policy_passes_with_valid_closing_issue_for_support_files():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Improve DirectX support",
        body="Fixes #10",
        author="alice",
        changed_files=("crosstl/backend/DirectX/DirectxCrossGLCodeGen.py",),
    )
    client = FakeClient(module, {10: issue(10)})

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        enforce_support_traceability=True,
    )

    assert summary["traceability_required"] == 1
    assert summary["traceability_satisfied"] == 1
    assert summary["traceability_failed"] == 0
    assert summary["support_relevant_files"] == 1


def test_traceability_policy_passes_with_explicit_no_issue_marker():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Refresh support matrix probes",
        body="Support issue traceability: no issue closed",
        author="alice",
        changed_files=("tools/support_matrix.py",),
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        enforce_support_traceability=True,
    )

    assert summary["traceability_required"] == 1
    assert summary["traceability_satisfied"] == 1
    assert summary["traceability_failed"] == 0


def test_traceability_policy_fails_for_support_files_without_issue_or_marker():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Improve Metal support",
        body="No linked issue.",
        author="alice",
        changed_files=("tests/test_translator/test_codegen/test_metal_codegen.py",),
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        enforce_support_traceability=True,
    )

    assert summary["traceability_required"] == 1
    assert summary["traceability_satisfied"] == 0
    assert summary["traceability_failed"] == 1


def test_traceability_policy_treats_frontend_ir_paths_as_support_relevant():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Improve frontend IR metadata",
        body="No linked issue.",
        author="alice",
        changed_files=(
            "crosstl/translator/parser.py",
            "tests/test_translator/test_frontend_parser_property_contracts.py",
            "docs/source/support-matrix.rst",
            "examples/test.py",
        ),
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        enforce_support_traceability=True,
    )

    assert summary["traceability_required"] == 1
    assert summary["traceability_failed"] == 1
    assert summary["support_relevant_files"] == 4


def test_traceability_policy_ignores_non_support_paths():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Update docs",
        body="No linked issue.",
        author="alice",
        changed_files=("README.md",),
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        enforce_support_traceability=True,
    )

    assert summary["traceability_required"] == 0
    assert summary["traceability_satisfied"] == 0
    assert summary["traceability_failed"] == 0


def test_traceability_advisory_cli_warns_without_failing(tmp_path, monkeypatch, capsys):
    module = load_sync_module()
    event = {
        "pull_request": {
            "number": 5,
            "title": "Improve Metal support",
            "body": "No linked issue.",
            "user": {"login": "alice"},
        }
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    summary_path = tmp_path / "summary.md"
    client = FakeClient(
        module,
        pull_files=("tests/test_translator/test_codegen/test_metal_codegen.py",),
    )
    monkeypatch.setenv("TOKEN", "token")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    monkeypatch.setattr(module, "GitHubClient", lambda *args, **kwargs: client)

    result = module.main(
        [
            "--repo",
            "CrossGL/crosstl",
            "--event-path",
            str(event_path),
            "--token-env",
            "TOKEN",
            "--check-support-traceability",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Support traceability: required=1, satisfied=0, failed=1" in captured.out
    assert "::warning::Support-relevant PR changes" in captured.out
    assert "Support Traceability" in summary_path.read_text(encoding="utf-8")


def test_traceability_enforcement_cli_fails_when_opted_in(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    event = {
        "pull_request": {
            "number": 5,
            "title": "Improve Metal support",
            "body": "No linked issue.",
            "user": {"login": "alice"},
        }
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    client = FakeClient(
        module,
        pull_files=("tests/test_translator/test_codegen/test_metal_codegen.py",),
    )
    monkeypatch.setenv("TOKEN", "token")
    monkeypatch.setattr(module, "GitHubClient", lambda *args, **kwargs: client)

    result = module.main(
        [
            "--repo",
            "CrossGL/crosstl",
            "--event-path",
            str(event_path),
            "--token-env",
            "TOKEN",
            "--enforce-support-traceability",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "Support traceability: required=1, satisfied=0, failed=1" in captured.out
    assert "Support-relevant PR changes must include" in captured.err
