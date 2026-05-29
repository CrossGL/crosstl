import importlib.util
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "sync_support_issues.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_support_issues", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_matrix():
    return {
        "summary": {
            "status_counts": {
                "directx": {
                    "supported": 1,
                    "partial": 1,
                    "diagnostic": 0,
                    "validated_rejection": 0,
                    "unsupported": 0,
                    "unknown": 0,
                }
            }
        },
        "backends": [
            {
                "id": "directx",
                "name": "DirectX / HLSL",
                "docs": [
                    {
                        "name": "HLSL reference",
                        "url": "https://example.com/hlsl",
                    }
                ],
            }
        ],
        "features": [
            {
                "id": "target.codegen",
                "name": "Code generation",
                "category": "target",
                "description": "Emit target code.",
                "support": {
                    "directx": {
                        "status": "supported",
                        "evidence": ["tests/example.py::def test_codegen"],
                    }
                },
            },
            {
                "id": "textures.gather",
                "name": "Texture gather",
                "category": "textures",
                "description": "Gather texture samples.",
                "support": {
                    "directx": {
                        "status": "partial",
                        "notes": "Cube gather is not audited.",
                        "evidence": ["tests/example.py::def test_gather"],
                    }
                },
            },
        ],
        "backlog": [
            {
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "feature_id": "textures.gather",
                "feature": "Texture gather",
                "category": "textures",
                "status": "partial",
                "notes": "Cube gather is not audited.",
            }
        ],
    }


def sample_signals():
    return {
        "summary": {
            "docs_probe": {
                "provided": True,
                "total": 1,
                "ok": 1,
                "failed": 0,
                "linked_documents": 0,
            }
        },
        "features": [
            {
                "id": "target.codegen",
                "support": {
                    "directx": {
                        "catalog_status": "supported",
                        "catalog_evidence_count": 0,
                        "state": "not_detected",
                        "docs": [],
                        "implementation": [],
                        "tests": [],
                        "unsupported": [],
                    }
                },
            },
            {
                "id": "textures.gather",
                "support": {
                    "directx": {
                        "catalog_status": "partial",
                        "catalog_evidence_count": 1,
                        "state": "tested",
                        "docs": [],
                        "implementation": [
                            {
                                "path": "crosstl/translator/codegen/directx_codegen.py",
                                "matched_terms": ["gather"],
                            }
                        ],
                        "tests": [
                            {
                                "path": "tests/example.py",
                                "symbol": "test_gather",
                                "matched_terms": ["gather"],
                            }
                        ],
                        "unsupported": [],
                    }
                },
            },
        ],
        "issues": [
            {
                "key": (
                    "extracted:directx:target.codegen:supported_without_detected_tests"
                ),
                "kind": "supported_without_detected_tests",
                "title": "Extractor did not find tests for supported row",
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "feature_id": "target.codegen",
                "feature": "Code generation",
                "category": "target",
                "status": "supported",
                "state": "not_detected",
            },
            {
                "key": (
                    "extracted:directx:docs.sv-position:documented_candidate_not_detected"
                ),
                "kind": "documented_candidate_not_detected",
                "title": "Triage missing documented API candidate",
                "backend_id": "directx",
                "backend": "DirectX / HLSL",
                "feature_id": "docs.sv-position",
                "feature": "SV_Position",
                "category": "docs",
                "status": "untracked",
                "state": "docs_only",
                "signal": {
                    "state": "docs_only",
                    "catalog_evidence_count": 0,
                    "docs": [
                        {
                            "source": "HLSL reference",
                            "url": "https://example.com/hlsl",
                            "term": "SV_Position",
                            "count": 3,
                        }
                    ],
                    "implementation": [],
                    "tests": [],
                    "unsupported": [],
                },
            },
        ],
    }


class FakeClient:
    def __init__(self, existing=None, subissues=None):
        self.existing = list(existing or [])
        self.subissues = dict(subissues or {})
        self.created = []
        self.updated = []
        self.closed = []
        self.attached = []
        self.labels = []
        self.next_number = 100
        self.next_id = 1000

    def list_managed_issues(self):
        return list(self.existing)

    def ensure_label(self, name, color, description):
        self.labels.append((name, color, description))

    def create_issue(self, desired):
        issue = {
            "id": self.next_id,
            "number": self.next_number,
            "title": desired.title,
            "body": desired.body,
            "state": "open",
            "labels": [{"name": label} for label in desired.labels],
        }
        self.next_id += 1
        self.next_number += 1
        self.created.append(issue)
        return issue

    def update_issue(self, issue, desired):
        updated = dict(issue)
        updated.update(
            {
                "title": desired.title,
                "body": desired.body,
                "state": "open",
                "labels": [{"name": label} for label in desired.labels],
            }
        )
        self.updated.append(updated)
        return updated

    def close_issue(self, issue, body):
        closed = dict(issue)
        closed["state"] = "closed"
        closed["body"] = body
        self.closed.append(closed)
        return closed

    def add_sub_issue(self, parent, child):
        self.attached.append((parent["number"], child["number"]))
        self.subissues.setdefault(parent["number"], []).append(child)

    def list_sub_issues(self, parent):
        return list(self.subissues.get(parent["number"], []))


def issue(number, key, state="open", labels=None):
    module = load_sync_module()
    labels = labels or [module.LABEL_MANAGED]
    return {
        "id": number + 1000,
        "number": number,
        "title": "stale",
        "body": module.marker_for(key),
        "state": state,
        "labels": [{"name": label} for label in labels],
    }


def test_build_desired_issues_creates_backend_frontend_and_backlog_entries():
    module = load_sync_module()

    desired = module.build_desired_issues(sample_matrix())

    assert set(desired) == {
        "parent:directx",
        "parent:frontend",
        "backlog:directx:textures.gather",
    }
    child = desired["backlog:directx:textures.gather"]
    assert child.parent_key == "parent:directx"
    assert module.LABEL_BACKLOG in child.labels
    assert "Cube gather is not audited." in child.body
    assert "`tests/example.py::def test_gather`" in child.body


def test_build_desired_issues_includes_generated_support_signal_issues():
    module = load_sync_module()

    desired = module.build_desired_issues(sample_matrix(), sample_signals())

    key = "extracted:directx:target.codegen:supported_without_detected_tests"
    assert key in desired
    assert desired[key].parent_key == "parent:directx"
    assert module.LABEL_EXTRACTED in desired[key].labels
    assert "Extractor did not find tests for supported row" in desired[key].body

    candidate_key = (
        "extracted:directx:docs.sv-position:documented_candidate_not_detected"
    )
    assert candidate_key in desired
    assert desired[candidate_key].parent_key == "parent:directx"
    assert "SV_Position" in desired[candidate_key].body
    assert "count=3" in desired[candidate_key].body

    child = desired["backlog:directx:textures.gather"]
    assert "Extractor state: `tested`" in child.body
    assert "test_gather" in child.body


def test_desired_issue_counts_summarizes_planned_parent_backlog_and_signal_issues():
    module = load_sync_module()

    desired = module.build_desired_issues(sample_matrix(), sample_signals())

    assert module.desired_issue_counts(desired) == {
        "total": 5,
        "parents": 2,
        "backlog": 1,
        "extracted": 2,
    }


def test_validate_desired_issues_accepts_complete_plan():
    module = load_sync_module()
    matrix = sample_matrix()
    signals = sample_signals()
    desired = module.build_desired_issues(matrix, signals)

    assert (
        module.validate_desired_issues(
            matrix,
            signals,
            desired,
            min_desired_issues=5,
        )
        == []
    )


def test_validate_desired_issues_catches_empty_or_incomplete_plan():
    module = load_sync_module()

    errors = module.validate_desired_issues(
        sample_matrix(),
        sample_signals(),
        {},
        min_desired_issues=5,
    )

    assert "desired issue plan has 0 issues, below minimum 5" in errors
    assert "missing desired parent issue: parent:directx" in errors
    assert "missing desired parent issue: parent:frontend" in errors
    assert "missing desired backlog issue: backlog:directx:textures.gather" in errors
    assert (
        "missing desired extracted issue: extracted:directx:target.codegen:supported_without_detected_tests"
        in errors
    )


def test_validate_desired_issues_catches_parent_and_marker_regressions():
    module = load_sync_module()
    desired = {
        "backlog:directx:textures.gather": module.DesiredIssue(
            key="backlog:directx:textures.gather",
            title="bad",
            body="missing marker",
            labels=(module.LABEL_MANAGED, module.LABEL_BACKLOG),
            parent_key="parent:missing",
        )
    }

    errors = module.validate_desired_issues(
        sample_matrix(),
        None,
        desired,
        min_desired_issues=1,
    )

    assert (
        "desired issue backlog:directx:textures.gather body is missing its sync marker"
        in errors
    )
    assert (
        "desired issue backlog:directx:textures.gather references missing parent parent:missing"
        in errors
    )


def test_sync_issues_updates_existing_creates_missing_closes_stale_and_attaches():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    stale_child = issue(2, "backlog:directx:old.feature")
    client = FakeClient(existing=[existing_parent, stale_child])

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=True,
        throttle_seconds=0,
    )

    assert summary["updated"] == 1
    assert summary["created"] == 2
    assert summary["closed"] == 1
    assert summary["attached"] == 1
    assert client.updated[0]["number"] == existing_parent["number"]
    assert client.closed[0]["number"] == stale_child["number"]
    assert client.attached == [(existing_parent["number"], client.created[1]["number"])]


def test_sync_issues_skips_existing_sub_issue_relationships():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    existing_child = issue(3, "backlog:directx:textures.gather")
    client = FakeClient(
        existing=[existing_parent, existing_child],
        subissues={existing_parent["number"]: [existing_child]},
    )

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=True,
        throttle_seconds=0,
    )

    assert summary["attached"] == 0
    assert client.attached == []


def test_sync_issues_closes_duplicate_managed_markers():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    duplicate_parent = issue(2, "parent:directx")
    client = FakeClient(existing=[existing_parent, duplicate_parent])

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=False,
        throttle_seconds=0,
    )

    assert summary["closed"] == 1
    assert client.closed[0]["number"] == duplicate_parent["number"]
    assert "# Duplicate Managed Support Issue" in client.closed[0]["body"]
    assert module.marker_for("parent:directx") in client.closed[0]["body"]


def test_sync_issues_preserves_stale_extracted_issues_when_signals_are_not_clean():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    stale_extracted = issue(
        5,
        "extracted:directx:docs.old-candidate:documented_candidate_not_detected",
    )
    client = FakeClient(existing=[stale_extracted])

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=False,
        close_extracted_issues=False,
        throttle_seconds=0,
    )

    assert summary["closed"] == 0
    assert summary["unchanged"] == 1
    assert client.closed == []


def test_signals_allow_extracted_closure_only_for_clean_docs_probe():
    module = load_sync_module()

    assert module.signals_allow_extracted_closure(sample_signals()) is True
    assert module.signals_allow_extracted_closure(None) is False
    assert (
        module.signals_allow_extracted_closure(
            {
                "summary": {
                    "docs_probe": {
                        "provided": True,
                        "failed": 1,
                    }
                }
            }
        )
        is False
    )
    assert (
        module.signals_allow_extracted_closure(
            {
                "summary": {
                    "docs_probe": {
                        "provided": False,
                        "failed": 0,
                    }
                }
            }
        )
        is False
    )


def test_dry_run_does_not_touch_client():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    client = FakeClient()

    summary = module.sync_issues(client, desired, dry_run=True)

    assert summary == {
        "created": 0,
        "updated": 0,
        "closed": 0,
        "attached": 0,
        "unchanged": 0,
    }
    assert client.created == []
    assert client.labels == []


def test_github_client_retries_secondary_rate_limit(monkeypatch):
    module = load_sync_module()
    attempts = {"count": 0}
    sleeps = []

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(_req, timeout):
        assert timeout == 30
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise module.error.HTTPError(
                url="https://api.example.test",
                code=403,
                msg="Forbidden",
                hdrs={},
                fp=io.BytesIO(b'{"message":"secondary rate limit"}'),
            )
        return FakeResponse()

    monkeypatch.setattr(module.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: sleeps.append(seconds))

    client = module.GitHubClient(
        "owner/repo",
        "token",
        api_url="https://api.example.test",
        max_retries=1,
        retry_base_seconds=0.5,
    )

    data, _headers = client.request("POST", "/repos/owner/repo/issues", {"title": "x"})

    assert data == {"ok": True}
    assert attempts["count"] == 2
    assert sleeps == [0.5]


def test_github_client_uses_retry_after_header_for_retry_delay():
    module = load_sync_module()
    client = module.GitHubClient("owner/repo", "token", retry_base_seconds=30.0)

    assert client.retry_delay_seconds({"retry-after": "7"}, attempt=3) == 7.0


def test_parse_args_exposes_rate_limit_controls():
    module = load_sync_module()

    args = module.parse_args(
        [
            "--repo",
            "owner/repo",
            "--max-retries",
            "6",
            "--retry-base-seconds",
            "60",
            "--retry-max-seconds",
            "600",
            "--throttle-seconds",
            "2",
            "--min-desired-issues",
            "10",
        ]
    )

    assert args.max_retries == 6
    assert args.retry_base_seconds == 60
    assert args.retry_max_seconds == 600
    assert args.throttle_seconds == 2
    assert args.min_desired_issues == 10
