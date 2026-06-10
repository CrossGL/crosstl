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
        json_errors=None,
        support_issues=None,
    ):
        self.module = module
        self.issues = dict(issues or {})
        self.assign_errors = dict(assign_errors or {})
        self.pull_files = list(pull_files or [])
        self.json_files = dict(json_files or {})
        self.json_errors = dict(json_errors or {})
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
        error = self.json_errors.get((repo, path, ref))
        if error is not None:
            raise error
        return self.json_files[(repo, path, ref)]

    def list_support_issues(self):
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
Fixes #19, and #20
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

    assert numbers == [18, 10, 11, 12, 13, 19, 20]


def test_extract_closing_issue_numbers_ignores_html_comments():
    module = load_sync_module()
    body = """
<!--
Fixes #123
Closes https://github.com/CrossGL/crosstl/issues/124
-->
Fixes #10
"""

    numbers = module.extract_closing_issue_numbers(
        "Ignore template examples", body, "CrossGL/crosstl"
    )

    assert numbers == [10]


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


def test_sync_strips_unmanaged_reference_lines_with_oxford_comma():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Improve matrix sync",
        body="Support matrix update.\n\nRefs #10, and #11",
        author="alice",
    )
    client = FakeClient(module)

    summary = module.sync_pr_issue_links(client, pr, "CrossGL/crosstl")

    assert summary["linked"] == 0
    assert summary["body_updated"] == 1
    assert client.updated_bodies == [(5, "Support matrix update.")]


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


def test_sync_adds_mixed_support_matrix_closures_and_refs(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    base_matrix = {
        "features": [
            {
                "id": "language.wave_intrinsics",
                "support": {
                    "metal": {
                        "status": "partial",
                        "notes": "Wave active ops are missing.",
                        "evidence": ["tests/old.py::test_wave_active"],
                    }
                },
            },
            {
                "id": "texture.projected",
                "support": {
                    "metal": {
                        "status": "partial",
                        "notes": "Planar projection is supported.",
                        "evidence": ["tests/old.py::test_planar_projection"],
                    }
                },
            },
            {
                "id": "texture.gather",
                "support": {
                    "metal": {
                        "status": "unsupported",
                        "notes": "Gather is missing.",
                        "evidence": [],
                    }
                },
            },
        ],
        "backlog": [
            {"backend_id": "metal", "feature_id": "language.wave_intrinsics"},
            {"backend_id": "metal", "feature_id": "texture.projected"},
            {"backend_id": "metal", "feature_id": "texture.gather"},
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
            },
            {
                "id": "texture.gather",
                "support": {
                    "metal": {
                        "status": "unsupported",
                        "notes": "Gather is missing.",
                        "evidence": [],
                    }
                },
            },
        ],
        "backlog": [
            {"backend_id": "metal", "feature_id": "texture.projected"},
            {"backend_id": "metal", "feature_id": "texture.gather"},
        ],
    }
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(base_matrix), encoding="utf-8")
    monkeypatch.setattr(module, "SUPPORT_MATRIX_PATH", matrix_path)
    wave_marker = (
        "<!-- crossgl-support-issue-sync: backlog:metal:language.wave_intrinsics -->"
    )
    projected_marker = (
        "<!-- crossgl-support-issue-sync: backlog:metal:texture.projected -->"
    )
    gather_marker = "<!-- crossgl-support-issue-sync: backlog:metal:texture.gather -->"
    pr = module.PullRequestContext(
        number=5,
        title="Improve Metal support",
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
        support_issues=[
            issue(498, body=wave_marker),
            issue(432, body=projected_marker),
            issue(353, body=gather_marker),
        ],
    )

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        sync_support_closures=True,
        sync_support_references=True,
        enforce_support_traceability=True,
    )

    assert summary["linked"] == 1
    assert summary["support_closures"] == 1
    assert summary["support_references"] == 1
    assert summary["traceability_satisfied"] == 1
    assert summary["support_link_audit"]["inspection_failed"] is False
    assert summary["support_link_audit"]["missing_closure_keys"] == []
    assert summary["support_link_audit"]["missing_reference_keys"] == []
    assert summary["support_link_audit"]["closure_links"] == [
        {
            "key": "backlog:metal:language.wave_intrinsics",
            "issues": [498],
            "reason": "removed_from_backlog",
            "base_row": {
                "backend_id": "metal",
                "feature_id": "language.wave_intrinsics",
                "status": "partial",
                "notes": "Wave active ops are missing.",
                "evidence": ["tests/old.py::test_wave_active"],
            },
        }
    ]
    assert summary["support_link_audit"]["reference_links"] == [
        {
            "key": "backlog:metal:texture.projected",
            "issues": [432],
            "reason": "backlog_row_changed",
            "base_row": {
                "backend_id": "metal",
                "feature_id": "texture.projected",
                "status": "partial",
                "notes": "Planar projection is supported.",
                "evidence": ["tests/old.py::test_planar_projection"],
            },
            "head_row": {
                "backend_id": "metal",
                "feature_id": "texture.projected",
                "status": "partial",
                "notes": "Planar and cube-shadow projection are supported.",
                "evidence": [
                    "tests/new.py::test_cube_shadow_projection",
                    "tests/old.py::test_planar_projection",
                ],
            },
        }
    ]
    assert client.updated_bodies == [
        (
            5,
            "\n".join(
                [
                    "Support matrix update.",
                    "",
                    "<!-- crossgl-pr-issue-links:start -->",
                    "Closes #498",
                    "Refs #432",
                    "<!-- crossgl-pr-issue-links:end -->",
                ]
            ),
        )
    ]
    assert "#353" not in client.updated_bodies[0][1]

    module.emit_support_link_audit(summary)
    captured = capsys.readouterr()
    assert "Support link audit: closure_candidates=1" in captured.out
    assert "reference_candidates=1" in captured.out
    assert (
        "#498 (backlog:metal:language.wave_intrinsics; "
        "reason=removed_from_backlog; row removed)"
    ) in captured.out
    assert (
        "#432 (backlog:metal:texture.projected; reason=backlog_row_changed; "
        "notes changed; evidence: 1 -> 2)"
    ) in captured.out


def test_sync_audit_reports_missing_managed_issue_for_removed_backlog_row(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    base_matrix = {
        "backlog": [{"backend_id": "metal", "feature_id": "language.wave_intrinsics"}]
    }
    head_matrix = {"backlog": []}
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(base_matrix), encoding="utf-8")
    monkeypatch.setattr(module, "SUPPORT_MATRIX_PATH", matrix_path)
    pr = module.PullRequestContext(
        number=5,
        title="Improve Metal support",
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
        support_issues=[],
    )

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        sync_support_closures=True,
        sync_support_references=True,
        check_support_traceability=True,
    )

    assert summary["support_closures"] == 0
    assert summary["support_references"] == 0
    assert summary["traceability_failed"] == 1
    assert summary["support_link_audit"] == {
        "inspection_failed": False,
        "matrix_delta": {
            "base_backlog_rows": 1,
            "head_backlog_rows": 0,
            "removed_backlog_rows": 1,
            "changed_backlog_rows": 0,
            "has_delta": True,
        },
        "closure_links": [],
        "reference_links": [],
        "missing_closure_keys": ["backlog:metal:language.wave_intrinsics"],
        "missing_reference_keys": [],
        "changed_files": {
            "support_relevant": 1,
            "support_matrix_artifact": 1,
        },
    }

    module.emit_support_link_audit(summary)
    captured = capsys.readouterr()
    assert "missing_closure_links=1" in captured.out
    assert (
        "::warning::Support backlog rows were removed without managed support issues"
        in captured.out
    )
    assert "backlog:metal:language.wave_intrinsics" in captured.out


def test_sync_audit_reports_no_matrix_delta_for_support_relevant_pr(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix = {
        "features": [
            {
                "id": "language.wave_intrinsics",
                "support": {
                    "metal": {
                        "status": "partial",
                        "notes": "Wave active ops are missing.",
                        "evidence": ["tests/old.py::test_wave_active"],
                    }
                },
            }
        ],
        "backlog": [
            {
                "backend_id": "metal",
                "feature_id": "language.wave_intrinsics",
                "status": "partial",
                "notes": "Wave active ops are missing.",
            }
        ],
    }
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    monkeypatch.setattr(module, "SUPPORT_MATRIX_PATH", matrix_path)
    pr = module.PullRequestContext(
        number=5,
        title="Improve Metal wave support implementation",
        body="No linked issue.",
        author="alice",
        head_repo="CrossGL/crosstl",
        head_sha="abc123",
        changed_files=("crosstl/translator/codegen/metal_codegen.py",),
    )
    client = FakeClient(
        module,
        json_files={
            (
                "CrossGL/crosstl",
                "support/generated/support-matrix.json",
                "abc123",
            ): matrix
        },
        support_issues=[],
    )

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        sync_support_closures=True,
        sync_support_references=True,
        check_support_traceability=True,
    )

    assert summary["traceability_failed"] == 1
    assert summary["support_link_audit"] == {
        "inspection_failed": False,
        "matrix_delta": {
            "base_backlog_rows": 1,
            "head_backlog_rows": 1,
            "removed_backlog_rows": 0,
            "changed_backlog_rows": 0,
            "has_delta": False,
        },
        "closure_links": [],
        "reference_links": [],
        "missing_closure_keys": [],
        "missing_reference_keys": [],
        "changed_files": {
            "support_relevant": 1,
            "support_matrix_artifact": 0,
        },
    }

    module.emit_support_link_audit(summary)
    captured = capsys.readouterr()
    assert "Support matrix delta: base_backlog_rows=1" in captured.out
    assert "matrix_delta_detected=0" in captured.out
    assert (
        "::warning::No support matrix backlog delta detected for support-relevant "
        "PR files"
    ) in captured.out
    assert "support/generated/support-matrix.json was not changed" in captured.out


def test_support_link_audit_writes_step_summary(tmp_path, monkeypatch):
    module = load_sync_module()
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    summary = {
        "support_link_audit": {
            "inspection_failed": False,
            "closure_links": [
                {
                    "key": "backlog:metal:language.wave_intrinsics",
                    "issues": [498],
                }
            ],
            "reference_links": [
                {
                    "key": "backlog:metal:texture.projected",
                    "issues": [432],
                }
            ],
            "missing_closure_keys": ["backlog:metal:texture.gather"],
            "missing_reference_keys": [],
        }
    }

    module.write_support_link_step_summary(summary)

    text = summary_path.read_text(encoding="utf-8")
    assert "## Support Issue Links" in text
    assert "- Inspection: succeeded" in text
    assert "- Closure links: 1/2" in text
    assert "- Reference links: 1/1" in text
    assert "#498 (backlog:metal:language.wave_intrinsics)" in text
    assert "#432 (backlog:metal:texture.projected)" in text
    assert "backlog:metal:texture.gather" in text


def test_support_link_step_summary_includes_change_reasons(tmp_path, monkeypatch):
    module = load_sync_module()
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    summary = {
        "support_link_audit": {
            "inspection_failed": False,
            "closure_links": [
                {
                    "key": "backlog:metal:language.wave_intrinsics",
                    "issues": [498],
                    "reason": "removed_from_backlog",
                    "base_row": {
                        "status": "partial",
                        "notes": "Wave active ops are missing.",
                        "evidence": ["tests/old.py::test_wave_active"],
                    },
                }
            ],
            "reference_links": [
                {
                    "key": "backlog:metal:texture.projected",
                    "issues": [432],
                    "reason": "backlog_row_changed",
                    "base_row": {
                        "status": "partial",
                        "notes": "Planar projection is supported.",
                        "evidence": ["tests/old.py::test_planar_projection"],
                    },
                    "head_row": {
                        "status": "partial",
                        "notes": "Planar and cube projection are supported.",
                        "evidence": [
                            "tests/new.py::test_cube_projection",
                            "tests/old.py::test_planar_projection",
                        ],
                    },
                }
            ],
            "missing_closure_keys": [],
            "missing_reference_keys": [],
        }
    }

    module.write_support_link_step_summary(summary)

    text = summary_path.read_text(encoding="utf-8")
    assert (
        "#498 (backlog:metal:language.wave_intrinsics; "
        "reason=removed_from_backlog; row removed)"
    ) in text
    assert (
        "#432 (backlog:metal:texture.projected; reason=backlog_row_changed; "
        "notes changed; evidence: 1 -> 2)"
    ) in text


def test_support_link_step_summary_reports_no_matrix_delta(tmp_path, monkeypatch):
    module = load_sync_module()
    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    summary = {
        "support_link_audit": {
            "inspection_failed": False,
            "matrix_delta": {
                "base_backlog_rows": 1,
                "head_backlog_rows": 1,
                "removed_backlog_rows": 0,
                "changed_backlog_rows": 0,
                "has_delta": False,
            },
            "closure_links": [],
            "reference_links": [],
            "missing_closure_keys": [],
            "missing_reference_keys": [],
            "changed_files": {
                "support_relevant": 2,
                "support_matrix_artifact": 0,
            },
        }
    }

    module.write_support_link_step_summary(summary)

    text = summary_path.read_text(encoding="utf-8")
    assert (
        "- Matrix delta: removed 0, changed 0, base backlog 1, head backlog 1" in text
    )
    assert "- Support-relevant files: 2" in text
    assert "- Support matrix artifact changed: no" in text
    assert "no support matrix backlog delta was detected" in text


def test_sync_preserves_existing_managed_links_when_support_matrix_unavailable(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    base_matrix = {
        "backlog": [{"backend_id": "metal", "feature_id": "language.wave_intrinsics"}]
    }
    matrix_path = tmp_path / "support-matrix.json"
    matrix_path.write_text(json.dumps(base_matrix), encoding="utf-8")
    monkeypatch.setattr(module, "SUPPORT_MATRIX_PATH", matrix_path)
    body = "\n".join(
        [
            "Support matrix update.",
            "",
            "<!-- crossgl-pr-issue-links:start -->",
            "Closes #498",
            "Refs #432",
            "<!-- crossgl-pr-issue-links:end -->",
        ]
    )
    pr = module.PullRequestContext(
        number=5,
        title="Improve Metal support",
        body=body,
        author="alice",
        head_repo="CrossGL/crosstl",
        head_sha="abc123",
        changed_files=("support/generated/support-matrix.json",),
    )
    client = FakeClient(
        module,
        json_errors={
            (
                "CrossGL/crosstl",
                "support/generated/support-matrix.json",
                "abc123",
            ): module.GitHubApiError(
                "GET",
                "/repos/CrossGL/crosstl/contents/support/generated/support-matrix.json",
                502,
                "upstream unavailable",
            )
        },
    )

    summary = module.sync_pr_issue_links(
        client,
        pr,
        "CrossGL/crosstl",
        sync_support_closures=True,
        sync_support_references=True,
        enforce_support_traceability=True,
    )

    captured = capsys.readouterr()
    assert "::warning::Could not inspect PR support matrix links" in captured.out
    assert summary["body_updated"] == 0
    assert summary["traceability_failed"] == 1
    assert summary["support_link_audit"] == {
        "inspection_failed": True,
        "closure_links": [],
        "reference_links": [],
        "missing_closure_keys": [],
        "missing_reference_keys": [],
        "changed_files": {
            "support_relevant": 1,
            "support_matrix_artifact": 1,
        },
    }
    assert client.updated_bodies == []


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


def test_github_client_lists_support_issues_across_open_and_closed_states():
    module = load_sync_module()
    client = module.GitHubClient("CrossGL/crosstl", "token")
    queries = []

    def fake_request(method, path, payload=None, query=None):
        assert method == "GET"
        assert path == "/repos/CrossGL/crosstl/issues"
        assert payload is None
        queries.append(query)
        return [
            {"number": 10, "body": "managed"},
            {"number": 11, "body": "pull", "pull_request": {}},
        ], {}

    client.request = fake_request

    assert client.list_support_issues() == [{"number": 10, "body": "managed"}]
    assert queries == [
        {
            "labels": "support:matrix",
            "state": "all",
            "per_page": 100,
            "page": 1,
        }
    ]


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
    assert summary["traceability_audit"] == {
        "required": True,
        "satisfied": True,
        "satisfaction_sources": ["same_repo_closing_issue"],
        "failure_reason": None,
        "support_relevant_files": [
            {
                "path": "crosstl/backend/DirectX/DirectxCrossGLCodeGen.py",
                "reason": "prefix",
                "matched": "crosstl/backend/",
            }
        ],
        "support_relevant_count": 1,
        "support_matrix_artifact_changed": False,
        "closing_issue_numbers": [10],
        "managed_reference_issue_numbers": [],
    }


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
    assert summary["traceability_audit"]["satisfaction_sources"] == ["explicit_opt_out"]
    assert summary["traceability_audit"]["failure_reason"] is None


def test_traceability_policy_ignores_no_issue_marker_in_html_comment():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Refresh support matrix probes",
        body="\n".join(
            [
                "<!--",
                "Support issue traceability: no issue closed",
                "-->",
                "No linked issue.",
            ]
        ),
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
    assert summary["traceability_satisfied"] == 0
    assert summary["traceability_failed"] == 1
    assert summary["traceability_audit"]["satisfaction_sources"] == []
    assert summary["traceability_audit"]["failure_reason"] == "missing_issue_or_opt_out"


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
    assert summary["traceability_audit"] == {
        "required": True,
        "satisfied": False,
        "satisfaction_sources": [],
        "failure_reason": "missing_issue_or_opt_out",
        "support_relevant_files": [
            {
                "path": "tests/test_translator/test_codegen/test_metal_codegen.py",
                "reason": "prefix",
                "matched": "tests/test_translator/test_codegen/",
            }
        ],
        "support_relevant_count": 1,
        "support_matrix_artifact_changed": False,
        "closing_issue_numbers": [],
        "managed_reference_issue_numbers": [],
    }


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
    assert summary["traceability_audit"]["support_relevant_files"] == [
        {
            "path": "crosstl/translator/parser.py",
            "reason": "exact_path",
            "matched": "crosstl/translator/parser.py",
        },
        {
            "path": "tests/test_translator/test_frontend_parser_property_contracts.py",
            "reason": "prefix",
            "matched": "tests/test_translator/test_frontend_",
        },
        {
            "path": "docs/source/support-matrix.rst",
            "reason": "exact_path",
            "matched": "docs/source/support-matrix.rst",
        },
        {
            "path": "examples/test.py",
            "reason": "exact_path",
            "matched": "examples/test.py",
        },
    ]


def test_traceability_policy_treats_project_pipeline_paths_as_support_relevant():
    module = load_sync_module()
    pr = module.PullRequestContext(
        number=5,
        title="Improve project porting metadata",
        body="No linked issue.",
        author="alice",
        changed_files=(
            "crosstl/project/pipeline.py",
            "docs/source/project-porting.rst",
            "tests/test_translator/test_project_translation.py",
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
    assert summary["support_relevant_files"] == 3
    assert summary["traceability_audit"]["support_relevant_files"] == [
        {
            "path": "crosstl/project/pipeline.py",
            "reason": "prefix",
            "matched": "crosstl/project/",
        },
        {
            "path": "docs/source/project-porting.rst",
            "reason": "exact_path",
            "matched": "docs/source/project-porting.rst",
        },
        {
            "path": "tests/test_translator/test_project_translation.py",
            "reason": "exact_path",
            "matched": "tests/test_translator/test_project_translation.py",
        },
    ]


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
    assert summary["traceability_audit"] == {
        "required": False,
        "satisfied": False,
        "satisfaction_sources": [],
        "failure_reason": "no_support_relevant_files",
        "support_relevant_files": [],
        "support_relevant_count": 0,
        "support_matrix_artifact_changed": False,
        "closing_issue_numbers": [],
        "managed_reference_issue_numbers": [],
    }


def pr_issue_link_summary_fixture():
    return {
        "linked": 1,
        "support_closures": 1,
        "support_references": 0,
        "assigned": 0,
        "assignment_skipped": 0,
        "missing_or_pull": 0,
        "body_updated": 1,
        "support_link_audit": {
            "inspection_failed": False,
            "closure_links": [
                {
                    "key": "backlog:metal:language.wave_intrinsics",
                    "issues": [498],
                    "reason": "removed_from_backlog",
                }
            ],
            "reference_links": [],
            "missing_closure_keys": [],
            "missing_reference_keys": [],
            "changed_files": {
                "support_relevant": 1,
                "support_matrix_artifact": 1,
            },
        },
        "traceability_required": 1,
        "traceability_satisfied": 1,
        "traceability_failed": 0,
        "support_relevant_files": 1,
        "traceability_audit": {
            "required": True,
            "satisfied": True,
            "satisfaction_sources": ["same_repo_closing_issue"],
            "failure_reason": None,
            "support_relevant_files": [
                {
                    "path": "support/generated/support-matrix.json",
                    "reason": "exact_path",
                    "matched": "support/generated/support-matrix.json",
                }
            ],
            "support_relevant_count": 1,
            "support_matrix_artifact_changed": True,
            "closing_issue_numbers": [498],
            "managed_reference_issue_numbers": [],
        },
    }


def test_sync_summary_output_uses_stable_schema_contract(tmp_path):
    module = load_sync_module()
    summary = pr_issue_link_summary_fixture()
    output_path = tmp_path / "pr-issue-link-summary.json"

    module.write_sync_summary_output(output_path, summary)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["generator"] == "sync_pr_issue_links"
    assert payload["summary"] == summary
    assert module.validate_sync_summary_payload(payload) == []


def test_sync_summary_payload_validator_rejects_contract_drift():
    module = load_sync_module()
    payload = module.build_sync_summary_payload(pr_issue_link_summary_fixture())

    missing_generator = dict(payload)
    missing_generator.pop("generator")
    assert module.validate_sync_summary_payload(missing_generator) == [
        "generator must be sync_pr_issue_links"
    ]

    wrong_count = json.loads(json.dumps(payload))
    wrong_count["summary"]["traceability_audit"]["support_relevant_count"] = 2
    assert (
        "summary.traceability_audit.support_relevant_count must match "
        "support_relevant_files length"
    ) in module.validate_sync_summary_payload(wrong_count)

    wrong_source = json.loads(json.dumps(payload))
    wrong_source["summary"]["traceability_audit"]["satisfaction_sources"] = [
        "manual_override"
    ]
    assert (
        "summary.traceability_audit.satisfaction_sources has unknown values: "
        "manual_override"
    ) in module.validate_sync_summary_payload(wrong_source)


def test_validate_summary_cli_accepts_written_payload_without_github_context(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    output_path = tmp_path / "pr-issue-link-summary.json"
    module.write_sync_summary_output(output_path, pr_issue_link_summary_fixture())
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    result = module.main(["--validate-summary", str(output_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert f"PR issue link summary validation passed: {output_path}" in captured.out
    assert captured.err == ""


def test_validate_summary_cli_rejects_malformed_artifact(tmp_path, capsys):
    module = load_sync_module()
    output_path = tmp_path / "pr-issue-link-summary.json"
    output_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "sync_pr_issue_links",
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    result = module.main(["--validate-summary", str(output_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert f"PR issue link summary validation failed for {output_path}" in captured.err
    assert "summary.linked must be a non-negative integer" in captured.err


def test_validate_summary_cli_reports_missing_artifact(tmp_path, capsys):
    module = load_sync_module()
    output_path = tmp_path / "missing-pr-issue-link-summary.json"

    result = module.main(["--validate-summary", str(output_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "could not read" in captured.err


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
    summary_output = tmp_path / "reports" / "pr-issue-link-summary.json"
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
            "--summary-output",
            str(summary_output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Support traceability: required=1, satisfied=0, failed=1" in captured.out
    assert (
        "Support traceability audit: status=failed, sources=none, "
        "failure_reason=missing_issue_or_opt_out"
    ) in captured.out
    assert (
        "Support traceability files: "
        "tests/test_translator/test_codegen/test_metal_codegen.py "
        "(prefix:tests/test_translator/test_codegen/)"
    ) in captured.out
    assert "::warning::Support-relevant PR changes" in captured.out
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Support Traceability" in summary_text
    assert "- Status: failed" in summary_text
    assert "- Failure reason: missing_issue_or_opt_out" in summary_text
    assert (
        "- `tests/test_translator/test_codegen/test_metal_codegen.py` "
        "(prefix: `tests/test_translator/test_codegen/`)"
    ) in summary_text
    summary_json = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary_json["schema_version"] == 1
    assert summary_json["generator"] == "sync_pr_issue_links"
    assert summary_json["summary"]["traceability_failed"] == 1
    assert summary_json["summary"]["traceability_audit"]["failure_reason"] == (
        "missing_issue_or_opt_out"
    )
    assert "Wrote" in captured.out


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
    summary_output = tmp_path / "pr-issue-link-summary.json"
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
            "--summary-output",
            str(summary_output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "Support traceability: required=1, satisfied=0, failed=1" in captured.out
    assert "Support traceability audit: status=failed" in captured.out
    assert summary_output.exists()
    assert (
        json.loads(summary_output.read_text(encoding="utf-8"))["summary"][
            "traceability_failed"
        ]
        == 1
    )
    assert "Support-relevant PR changes must include" in captured.err
