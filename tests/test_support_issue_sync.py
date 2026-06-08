import importlib.util
import io
import json
import sys
from pathlib import Path

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
        "schema_version": 1,
        "generator": "tools/support_matrix.py",
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
                        "current_gap": "Cube-array gather remains unaudited.",
                        "next_scope": "Add cube-array gather fixtures.",
                        "completion_criteria": (
                            "Mark supported after cube-array gather evidence is recorded."
                        ),
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
        "schema_version": 1,
        "generator": "tools/support_signals.py",
        "summary": {
            "backend_count": 1,
            "feature_count": 2,
            "state_counts": {
                "not_detected": 1,
                "tested": 1,
            },
            "issue_count": 2,
            "docs_probe": {
                "provided": True,
                "total": 1,
                "ok": 1,
                "failed": 0,
                "linked_documents": 0,
            },
            "pytest_failures": {
                "provided": False,
                "report_count": 0,
                "load_error_count": 0,
                "failed_testcase_count": 0,
                "categories": {},
                "backends": {},
            },
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


def sample_matrix_check_report(ok=True):
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py check",
        "ok": ok,
        "summary": {
            "artifact_count": 3,
            "stale_count": 0 if ok else 1,
        },
        "artifacts": [
            {
                "path": "support/generated/support-matrix.json",
                "exists": True,
                "stale": not ok,
                "actual_sha256": "actual",
                "expected_sha256": "expected",
                "diff_line_count": 0 if ok else 12,
                "diff": [] if ok else ["--- actual", "+++ expected"],
            },
            {
                "path": "docs/source/support-matrix.rst",
                "exists": True,
                "stale": False,
                "actual_sha256": "same",
                "expected_sha256": "same",
                "diff_line_count": 0,
                "diff": [],
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


def test_build_desired_issues_creates_parent_and_backlog_entries():
    module = load_sync_module()

    desired = module.build_desired_issues(sample_matrix())

    assert set(desired) == {"parent:directx", "backlog:directx:textures.gather"}
    child = desired["backlog:directx:textures.gather"]
    assert child.parent_key == "parent:directx"
    assert module.LABEL_BACKLOG in child.labels
    assert "Cube-array gather remains unaudited." in child.body
    assert "## Next Scope\n\nAdd cube-array gather fixtures." in child.body
    assert (
        "## Completion Rule\n\nMark supported after cube-array gather evidence is recorded."
        in child.body
    )
    assert "`tests/example.py::def test_gather`" in child.body


def test_build_desired_issues_routes_project_backlog_to_frontend_parent():
    module = load_sync_module()
    matrix = sample_matrix()
    matrix["summary"]["status_counts"]["opengl"] = {
        "supported": 1,
        "partial": 1,
        "diagnostic": 0,
        "validated_rejection": 0,
        "unsupported": 0,
        "unknown": 0,
    }
    matrix["backends"].append({"id": "opengl", "name": "OpenGL / GLSL", "docs": []})
    matrix["features"].append(
        {
            "id": "project.source_provenance",
            "name": "Source provenance",
            "category": "project",
            "description": "Track project source provenance.",
            "support": {
                "directx": {
                    "status": "partial",
                    "notes": "DirectX provenance is incomplete.",
                    "current_gap": "Project provenance lacks fine-grained mappings.",
                    "next_scope": "Define fine-grained source-map validation.",
                    "completion_criteria": (
                        "Mark supported after fine-grained provenance validation passes."
                    ),
                    "evidence": ["tests/project.py::def test_directx_provenance"],
                },
                "opengl": {
                    "status": "partial",
                    "notes": "OpenGL provenance is incomplete.",
                    "current_gap": "Project provenance lacks fine-grained mappings.",
                    "next_scope": "Define fine-grained source-map validation.",
                    "completion_criteria": (
                        "Mark supported after fine-grained provenance validation passes."
                    ),
                    "evidence": ["tests/project.py::def test_opengl_provenance"],
                },
            },
        }
    )
    matrix["backlog"] = [
        {
            "backend_id": "directx",
            "backend": "DirectX / HLSL",
            "feature_id": "project.source_provenance",
            "feature": "Source provenance",
            "category": "project",
            "status": "partial",
            "notes": "DirectX provenance is incomplete.",
        },
        {
            "backend_id": "opengl",
            "backend": "OpenGL / GLSL",
            "feature_id": "project.source_provenance",
            "feature": "Source provenance",
            "category": "project",
            "status": "partial",
            "notes": "OpenGL provenance is incomplete.",
        },
    ]

    desired = module.build_desired_issues(matrix)

    assert set(desired) == {
        "parent:frontend",
        "backlog:frontend:project.source_provenance",
    }
    child = desired["backlog:frontend:project.source_provenance"]
    assert child.parent_key == "parent:frontend"
    assert module.LABEL_BACKLOG in child.labels
    assert module.LABEL_PREFIX_BACKEND + module.FRONTEND_ID in child.labels
    assert "DirectX / HLSL" in child.body
    assert "OpenGL / GLSL" in child.body
    assert "| Backend | Status | Current Gap | Next Scope |" in child.body
    assert "Project provenance lacks fine-grained mappings." in child.body
    assert "Define fine-grained source-map validation." in child.body
    assert (
        "## Completion Rule\n\nMark supported after fine-grained provenance validation passes."
        in child.body
    )
    assert "`tests/project.py::def test_directx_provenance`" in child.body
    assert "`tests/project.py::def test_opengl_provenance`" in child.body
    assert "backlog:directx:project.source_provenance" not in desired
    assert "backlog:opengl:project.source_provenance" not in desired
    assert (
        module.validate_desired_issues(matrix, None, desired, min_desired_issues=2)
        == []
    )

    signals = {
        "features": [
            {
                "id": "project.source_provenance",
                "support": {
                    "directx": {
                        "catalog_status": "partial",
                        "catalog_evidence_count": 1,
                        "state": "tested",
                        "docs": [],
                        "implementation": [
                            {
                                "path": "tools/directx_project_probe.py",
                                "matched_terms": ["source_provenance"],
                            }
                        ],
                        "tests": [],
                        "unsupported": [],
                    },
                    "opengl": {
                        "catalog_status": "partial",
                        "catalog_evidence_count": 2,
                        "state": "not_detected",
                        "docs": [],
                        "implementation": [],
                        "tests": [
                            {
                                "path": "tests/project_opengl.py",
                                "symbol": "test_opengl_project_provenance",
                                "matched_terms": ["source_provenance"],
                            }
                        ],
                        "unsupported": [],
                    },
                },
            }
        ],
        "issues": [],
    }

    desired_with_signals = module.build_desired_issues(matrix, signals)
    child_with_signals = desired_with_signals[
        "backlog:frontend:project.source_provenance"
    ]
    assert "### DirectX / HLSL" in child_with_signals.body
    assert "Extractor state: `tested`" in child_with_signals.body
    assert "Catalog evidence count: `1`" in child_with_signals.body
    assert "`tools/directx_project_probe.py`" in child_with_signals.body
    assert "### OpenGL / GLSL" in child_with_signals.body
    assert "Extractor state: `not_detected`" in child_with_signals.body
    assert "Catalog evidence count: `2`" in child_with_signals.body
    assert "`test_opengl_project_provenance`" in child_with_signals.body


def test_build_desired_issues_skips_empty_parent_trackers():
    module = load_sync_module()
    matrix = sample_matrix()
    matrix["backlog"] = []

    desired = module.build_desired_issues(matrix, {"issues": []})

    assert desired == {}


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


def test_duplicate_signal_keys_do_not_create_childless_parent_trackers():
    module = load_sync_module()
    matrix = sample_matrix()
    matrix["backlog"] = []
    signals = sample_signals()
    duplicate = dict(signals["issues"][0])
    duplicate.update(
        {
            "backend_id": module.FRONTEND_ID,
            "backend": module.FRONTEND_NAME,
        }
    )
    signals["issues"] = [signals["issues"][0], duplicate]

    desired = module.build_desired_issues(matrix, signals)

    key = "extracted:directx:target.codegen:supported_without_detected_tests"
    assert set(desired) == {"parent:directx", key}
    assert desired[key].parent_key == "parent:directx"

    errors = module.validate_desired_issues(
        matrix,
        signals,
        desired,
        min_desired_issues=0,
    )
    assert errors == [
        "desired extracted issue "
        "extracted:directx:target.codegen:supported_without_detected_tests "
        "references parent parent:directx, expected parent:frontend"
    ]


def test_build_desired_issues_renders_pytest_failure_signals():
    module = load_sync_module()
    signals = sample_signals()
    signals["issues"].append(
        {
            "key": "extracted:directx:ci.pytest.backend-codegen:pytest_failure_summary",
            "kind": "pytest_failure_summary",
            "title": "Investigate CI pytest failures for backend codegen",
            "backend_id": "directx",
            "backend": "DirectX / HLSL",
            "feature_id": "ci.pytest.backend-codegen",
            "feature": "CI pytest backend codegen failures",
            "category": "ci",
            "status": "failing",
            "state": "ci_failure",
            "signal": {
                "state": "ci_failure",
                "catalog_evidence_count": 0,
                "docs": [],
                "implementation": [],
                "tests": [],
                "unsupported": [],
                "failures": [
                    {
                        "nodeid": "tests.test_directx_codegen::test_texture",
                        "path": (
                            "tests/test_translator/test_codegen/"
                            "test_directx_codegen.py"
                        ),
                        "kind": "failure",
                        "category": "backend_codegen",
                        "backend": "directx",
                        "message": "generated HLSL assertion failed",
                        "matched_terms": ["backend_codegen"],
                    }
                ],
            },
        }
    )

    desired = module.build_desired_issues(sample_matrix(), signals)
    issue = desired[
        "extracted:directx:ci.pytest.backend-codegen:pytest_failure_summary"
    ]

    assert issue.parent_key == "parent:directx"
    assert module.LABEL_EXTRACTED in issue.labels
    assert "Pytest failures" in issue.body
    assert "tests.test_directx_codegen::test_texture" in issue.body
    assert "generated HLSL assertion failed" in issue.body


def test_build_desired_issues_ignores_signal_issues_for_backends_missing_from_matrix():
    module = load_sync_module()
    signals = sample_signals()
    foreign_signal_keys = [
        "extracted:cuda:source.lexing:supported_without_detected_tests",
        "extracted:hip:source.lexing:supported_without_detected_tests",
        "extracted:slang:source.lexing:supported_without_detected_tests",
    ]
    for key in foreign_signal_keys:
        backend_id = key.split(":", maxsplit=2)[1]
        signals["issues"].append(
            {
                "key": key,
                "kind": "supported_without_detected_tests",
                "title": "Extractor did not find tests for supported row",
                "backend_id": backend_id,
                "backend": backend_id.upper(),
                "feature_id": "source.lexing",
                "feature": "Lexing",
                "category": "source",
                "status": "supported",
                "state": "not_detected",
            }
        )

    desired = module.build_desired_issues(sample_matrix(), signals)
    errors = module.validate_desired_issues(
        sample_matrix(),
        signals,
        desired,
        min_desired_issues=4,
    )

    for key in foreign_signal_keys:
        assert key not in desired
    assert errors == []


def test_pytest_failure_issues_are_preserved_without_failure_summary_input():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix(), sample_signals())
    stale_pytest = issue(
        77,
        "extracted:directx:ci.pytest.backend-codegen:pytest_failure_summary",
        labels=[module.LABEL_MANAGED, module.LABEL_EXTRACTED],
    )

    preserved_actions = module.planned_issue_actions(
        desired,
        [stale_pytest],
        close_extracted_issues=True,
        close_pytest_failure_issues=False,
    )
    preserved_closures = module.planned_issue_closures(
        desired,
        [stale_pytest],
        close_extracted_issues=True,
        close_pytest_failure_issues=False,
    )
    preserved_samples = module.planned_issue_action_samples(
        desired,
        [stale_pytest],
        close_extracted_issues=True,
        close_pytest_failure_issues=False,
    )

    assert preserved_actions["closed"] == 0
    assert preserved_closures["total"] == 0
    assert preserved_samples["preserved"][0]["reason"] == (
        "stale_pytest_failure_preserved"
    )

    closing_closures = module.planned_issue_closures(
        desired,
        [stale_pytest],
        close_extracted_issues=True,
        close_pytest_failure_issues=True,
    )
    assert closing_closures["stale_extracted"] == 1


def test_desired_issue_counts_summarizes_planned_parent_backlog_and_signal_issues():
    module = load_sync_module()

    desired = module.build_desired_issues(sample_matrix(), sample_signals())

    assert module.desired_issue_counts(desired) == {
        "total": 4,
        "parents": 1,
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
            min_desired_issues=4,
        )
        == []
    )


def test_validate_desired_issues_catches_empty_or_incomplete_plan():
    module = load_sync_module()

    errors = module.validate_desired_issues(
        sample_matrix(),
        sample_signals(),
        {},
        min_desired_issues=4,
    )

    assert "desired issue plan has 0 issues, below minimum 4" in errors
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
    operation_ledger = []

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=True,
        throttle_seconds=0,
        operation_ledger=operation_ledger,
    )

    assert summary["updated"] == 1
    assert summary["created"] == 1
    assert summary["closed"] == 1
    assert summary["attached"] == 1
    assert client.updated[0]["number"] == existing_parent["number"]
    assert client.closed[0]["number"] == stale_child["number"]
    assert client.attached == [(existing_parent["number"], client.created[0]["number"])]
    assert [entry["action"] for entry in operation_ledger] == [
        "updated",
        "created",
        "attached",
        "closed",
    ]
    assert operation_ledger[0]["key"] == "parent:directx"
    assert operation_ledger[0]["reason"] == "desired_issue_drift"
    assert operation_ledger[0]["reasons"] == ["title", "body", "labels"]
    assert operation_ledger[1]["key"] == "backlog:directx:textures.gather"
    assert operation_ledger[1]["reason"] == "missing_backlog_issue"
    assert operation_ledger[2] == {
        "action": "attached",
        "parent_key": "parent:directx",
        "parent_number": existing_parent["number"],
        "child_key": "backlog:directx:textures.gather",
        "child_number": client.created[0]["number"],
        "reason": "missing_sub_issue_relationship",
    }
    assert operation_ledger[3]["key"] == "backlog:directx:old.feature"
    assert operation_ledger[3]["reason"] == "stale_managed_marker"


def test_sync_issues_reports_partial_summary_on_mutation_failure():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")

    class FailingCreateClient(FakeClient):
        def create_issue(self, desired_issue):
            if desired_issue.key == "backlog:directx:textures.gather":
                raise RuntimeError("create failed")
            return super().create_issue(desired_issue)

    client = FailingCreateClient(existing=[existing_parent])

    try:
        module.sync_issues(
            client,
            desired,
            dry_run=False,
            manage_sub_issues=False,
            throttle_seconds=0,
        )
    except module.SupportIssueSyncMutationError as exc:
        assert exc.phase == "create_issue"
        assert exc.operation == {
            "key": "backlog:directx:textures.gather",
            "title": "[Support Matrix][DirectX / HLSL] Texture gather (partial)",
            "parent_key": "parent:directx",
        }
        assert exc.summary == {
            "created": 0,
            "updated": 1,
            "closed": 0,
            "attached": 0,
            "unchanged": 0,
        }
        assert exc.operation_ledger == [
            {
                "action": "updated",
                "key": "parent:directx",
                "number": 1,
                "title": "[Support Matrix] DirectX / HLSL coverage",
                "state": "open",
                "reason": "desired_issue_drift",
                "reasons": ["title", "body", "labels"],
            }
        ]
        assert isinstance(exc.cause, RuntimeError)
    else:
        raise AssertionError("Expected SupportIssueSyncMutationError")


def test_planned_issue_actions_reports_mutations_without_touching_client():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    stale_child = issue(2, "backlog:directx:old.feature")

    planned = module.planned_issue_actions(
        desired,
        [existing_parent, stale_child],
        manage_sub_issues=True,
        close_extracted_issues=True,
    )

    assert planned == {
        "created": 1,
        "updated": 1,
        "closed": 1,
        "attached": 1,
        "unchanged": 0,
    }


def test_planned_issue_action_samples_explain_mutation_plan():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    stale_child = issue(2, "backlog:directx:old.feature")
    duplicate_parent = issue(3, "parent:directx")

    samples = module.planned_issue_action_samples(
        desired,
        [existing_parent, stale_child, duplicate_parent],
        manage_sub_issues=True,
        close_extracted_issues=True,
        sample_limit=4,
    )

    assert samples["sample_limit"] == 4
    assert samples["created"] == [
        {
            "key": "backlog:directx:textures.gather",
            "title": "[Support Matrix][DirectX / HLSL] Texture gather (partial)",
            "reason": "missing_backlog_issue",
            "parent_key": "parent:directx",
        }
    ]
    assert samples["updated"][0]["key"] == "parent:directx"
    assert set(samples["updated"][0]["reasons"]) == {"title", "body", "labels"}
    assert samples["closed"] == [
        {
            "key": "backlog:directx:old.feature",
            "number": 2,
            "title": "stale",
            "state": "open",
            "reason": "stale_managed_marker",
        },
        {
            "key": "parent:directx",
            "number": 3,
            "title": "stale",
            "state": "open",
            "reason": "duplicate_managed_marker",
        },
    ]
    assert samples["attached"] == [
        {
            "parent_key": "parent:directx",
            "child_key": "backlog:directx:textures.gather",
            "reason": "parent_or_child_will_be_created",
        }
    ]


def test_planned_issue_closures_summarizes_closure_categories():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    stale_parent = issue(1, "parent:oldbackend")
    stale_backlog = issue(2, "backlog:directx:old.feature")
    stale_extracted = issue(
        3,
        "extracted:directx:docs.old-candidate:documented_candidate_not_detected",
    )
    duplicate_parent = issue(4, "parent:directx")
    current_parent = issue(5, "parent:directx")

    closures = module.planned_issue_closures(
        desired,
        [
            stale_parent,
            stale_backlog,
            stale_extracted,
            duplicate_parent,
            current_parent,
        ],
        close_extracted_issues=True,
    )

    assert closures == {
        "total": 4,
        "stale_parent": 1,
        "stale_backlog": 1,
        "stale_extracted": 1,
        "duplicate_marker": 1,
    }
    preserved_extracted_closures = module.planned_issue_closures(
        desired,
        [stale_extracted],
        close_extracted_issues=False,
    )
    assert preserved_extracted_closures["total"] == 0
    assert preserved_extracted_closures["stale_extracted"] == 0


def test_issue_sync_report_includes_desired_counts_and_planned_actions():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix(), sample_signals())
    existing_parent = issue(1, "parent:directx")

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=True,
        manage_sub_issues=True,
        existing_issues=[existing_parent],
        workflow_source={
            "event": "workflow_run",
            "workflow": "Backend Tests",
            "run_id": 26722241319,
            "conclusion": "success",
            "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
        },
    )

    assert report["schema_version"] == 1
    assert report["mode"] == "dry-run"
    assert report["desired"] == {
        "total": 4,
        "parents": 1,
        "backlog": 1,
        "extracted": 2,
    }
    assert report["existing"] == {
        "inspected": True,
        "managed": 1,
        "duplicates": 0,
        "unmarked": 0,
    }
    assert report["planned_actions"]["created"] == 3
    assert report["planned_actions"]["updated"] == 1
    assert report["planned_closures"] == {
        "total": 0,
        "stale_parent": 0,
        "stale_backlog": 0,
        "stale_extracted": 0,
        "duplicate_marker": 0,
    }
    assert report["planned_action_samples"]["created"][0]["key"].startswith(
        ("backlog:", "extracted:")
    )
    assert report["planned_action_samples"]["updated"][0]["key"] == "parent:directx"
    assert report["workflow_source"] == {
        "event": "workflow_run",
        "workflow": "Backend Tests",
        "run_id": "26722241319",
        "conclusion": "success",
        "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
    }


def test_github_workflow_source_reads_workflow_run_event(tmp_path):
    module = load_sync_module()
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "workflow_run": {
                    "name": "Translator Tests",
                    "id": 26722241346,
                    "conclusion": "success",
                    "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
                }
            }
        ),
        encoding="utf-8",
    )

    source = module.github_workflow_source(
        {
            "GITHUB_EVENT_NAME": "workflow_run",
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_WORKFLOW": "Support Issue Sync",
            "GITHUB_RUN_ID": "26722414948",
            "GITHUB_SHA": "ignored-current-sha",
        }
    )

    assert source == {
        "event": "workflow_run",
        "workflow": "Translator Tests",
        "run_id": "26722241346",
        "conclusion": "success",
        "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
    }


def test_github_workflow_source_falls_back_to_current_run_metadata():
    module = load_sync_module()

    source = module.github_workflow_source(
        {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_WORKFLOW": "Support Issue Sync",
            "GITHUB_RUN_ID": "26722414948",
            "GITHUB_SHA": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
        }
    )

    assert source == {
        "event": "workflow_dispatch",
        "workflow": "Support Issue Sync",
        "run_id": "26722414948",
        "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
    }


def test_issue_sync_report_samples_preserved_stale_extracted_issues():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    stale_extracted = issue(
        5,
        "extracted:directx:docs.old-candidate:documented_candidate_not_detected",
    )

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=False,
        manage_sub_issues=False,
        existing_issues=[stale_extracted],
    )

    assert report["planned_actions"]["closed"] == 0
    assert report["planned_action_samples"]["preserved"] == [
        {
            "key": (
                "extracted:directx:docs.old-candidate:"
                "documented_candidate_not_detected"
            ),
            "number": 5,
            "title": "stale",
            "state": "open",
            "reason": "stale_extracted_preserved",
        }
    ]
    assert report["managed_issue_audit"]["preserved_extracted"] == {
        "total": 1,
        "open": 1,
        "closed": 0,
        "samples": [
            {
                "key": (
                    "extracted:directx:docs.old-candidate:"
                    "documented_candidate_not_detected"
                ),
                "number": 5,
                "title": "stale",
                "state": "open",
                "reason": "stale_extracted_preserved",
            }
        ],
    }


def test_issue_sync_report_audits_stale_duplicate_and_unknown_managed_issues():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing_parent = issue(1, "parent:directx")
    stale_backlog = issue(2, "backlog:directx:old.feature")
    closed_stale_parent = issue(3, "parent:oldbackend", state="closed")
    stale_extracted = issue(
        4,
        "extracted:directx:docs.old-candidate:documented_candidate_not_detected",
    )
    duplicate_parent = issue(5, "parent:directx")
    unknown_marker = issue(6, "custom:unknown")

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=False,
        manage_sub_issues=False,
        existing_issues=[
            existing_parent,
            stale_backlog,
            closed_stale_parent,
            stale_extracted,
            duplicate_parent,
            unknown_marker,
        ],
    )

    audit = report["managed_issue_audit"]
    assert audit["sample_limit"] == module.PLANNED_ACTION_SAMPLE_LIMIT
    assert audit["stale"]["total"] == 2
    assert audit["stale"]["open"] == 1
    assert audit["stale"]["closed"] == 1
    assert audit["stale"]["samples"] == [
        {
            "key": "backlog:directx:old.feature",
            "number": 2,
            "title": "stale",
            "state": "open",
            "category": "stale_backlog",
            "reason": "stale_managed_marker",
        },
        {
            "key": "parent:oldbackend",
            "number": 3,
            "title": "stale",
            "state": "closed",
            "category": "stale_parent",
            "reason": "closed_stale_managed_marker",
        },
    ]
    assert audit["duplicates"] == {
        "total": 1,
        "open": 1,
        "closed": 0,
        "samples": [
            {
                "key": "parent:directx",
                "number": 5,
                "title": "stale",
                "state": "open",
                "reason": "duplicate_managed_marker",
            }
        ],
    }
    assert audit["preserved_extracted"]["total"] == 1
    assert audit["ignored_unknown"] == {
        "total": 1,
        "open": 1,
        "closed": 0,
        "samples": [
            {
                "key": "custom:unknown",
                "number": 6,
                "title": "stale",
                "state": "open",
                "reason": "unknown_managed_marker",
            }
        ],
    }


def test_issue_sync_report_summarizes_support_matrix_check_report():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    matrix_check_path = Path("support/generated/support-matrix-check.json")

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=True,
        manage_sub_issues=True,
        matrix_check_report=sample_matrix_check_report(ok=False),
        matrix_check_report_path=matrix_check_path,
    )

    matrix_check = report["support_matrix_check"]
    assert matrix_check["provided"] is True
    assert matrix_check["path"] == str(matrix_check_path)
    assert matrix_check["ok"] is False
    assert matrix_check["summary"]["stale_count"] == 1
    assert matrix_check["stale_artifacts"] == [
        {
            "path": "support/generated/support-matrix.json",
            "diff_line_count": 12,
            "actual_sha256": "actual",
            "expected_sha256": "expected",
        }
    ]


def test_load_optional_json_reports_bad_optional_json(tmp_path):
    module = load_sync_module()
    malformed_path = tmp_path / "support-matrix-check.json"
    wrong_shape_path = tmp_path / "wrong-shape.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    wrong_shape_path.write_text("[]", encoding="utf-8")

    malformed = module.load_optional_json(malformed_path)
    wrong_shape = module.load_optional_json(wrong_shape_path)

    assert malformed["load_error"]["path"] == str(malformed_path)
    assert malformed["load_error"]["type"] == "JSONDecodeError"
    assert "Expecting property name" in malformed["load_error"]["message"]
    assert wrong_shape["load_error"] == {
        "path": str(wrong_shape_path),
        "type": "InvalidReportType",
        "message": "expected JSON object, got list",
    }


def test_issue_sync_report_flags_invalid_support_matrix_check_report():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    matrix_check_path = Path("support/generated/support-matrix-check.json")
    invalid_report = sample_matrix_check_report(ok=True)
    invalid_report["generator"] = "tools/sync_support_issues.py"

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=True,
        manage_sub_issues=True,
        matrix_check_report=invalid_report,
        matrix_check_report_path=matrix_check_path,
    )

    matrix_check = report["support_matrix_check"]
    assert matrix_check["provided"] is True
    assert matrix_check["path"] == str(matrix_check_path)
    assert matrix_check["ok"] is False
    assert matrix_check["summary"] == {}
    assert matrix_check["stale_artifacts"] == []
    assert matrix_check["load_error"] == {
        "path": str(matrix_check_path),
        "type": "UnexpectedReportGenerator",
        "message": (
            "expected generator tools/support_matrix.py check, got "
            "tools/sync_support_issues.py"
        ),
    }


def test_issue_sync_report_includes_preflight_failure_details():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())

    failure = module.preflight_failure_summary(
        module.SupportIssueSyncPreflightError(
            "list_sub_issues",
            {
                "parent_key": "parent:directx",
                "parent_number": 17,
            },
            RuntimeError("read failed"),
        )
    )
    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=True,
        manage_sub_issues=True,
        preflight_failure=failure,
        planned_action_budget_limits={"created": 1},
    )

    assert report["existing"]["inspected"] is False
    assert report["planned_action_budget"]["evaluated"] is False
    assert report["preflight_failure"] == {
        "phase": "list_sub_issues",
        "operation": {
            "parent_key": "parent:directx",
            "parent_number": 17,
        },
        "error": {
            "type": "RuntimeError",
            "message": "read failed",
        },
    }


def test_issue_sync_report_flags_planned_action_budget_violations():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=True,
        manage_sub_issues=True,
        existing_issues=[],
        planned_action_budget_limits={
            "created": 1,
            "closed": 0,
            "total": 2,
        },
    )

    budget = report["planned_action_budget"]
    assert budget["provided"] is True
    assert budget["evaluated"] is True
    assert budget["ok"] is False
    assert budget["violations"] == [
        {
            "action": "created",
            "actual": 2,
            "limit": 1,
        },
        {
            "action": "total",
            "actual": 3,
            "limit": 2,
        },
    ]
    assert module.planned_action_budget_errors(budget) == [
        "planned issue action budget exceeded for created: 2 > 1",
        "planned issue action budget exceeded for total: 3 > 2",
    ]


def test_issue_sync_report_flags_planned_closure_budget_violations():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())

    report = module.issue_sync_report(
        desired,
        mode="dry-run",
        close_extracted_issues=True,
        manage_sub_issues=True,
        existing_issues=[
            issue(1, "parent:oldbackend"),
            issue(2, "backlog:directx:old.feature"),
        ],
        planned_closure_budget_limits={
            "stale_parent": 0,
            "stale_backlog": 0,
        },
    )

    budget = report["planned_closure_budget"]
    assert budget["provided"] is True
    assert budget["evaluated"] is True
    assert budget["ok"] is False
    assert budget["violations"] == [
        {
            "category": "stale_backlog",
            "actual": 1,
            "limit": 0,
        },
        {
            "category": "stale_parent",
            "actual": 1,
            "limit": 0,
        },
    ]
    assert module.planned_closure_budget_errors(budget) == [
        "planned issue closure budget exceeded for stale_backlog: 1 > 0",
        "planned issue closure budget exceeded for stale_parent: 1 > 0",
    ]


def test_issue_sync_report_reconciles_operation_ledger_with_plan():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())

    report = module.issue_sync_report(
        desired,
        mode="sync",
        close_extracted_issues=True,
        manage_sub_issues=True,
        existing_issues=[],
        operation_ledger=[
            {
                "action": "created",
                "key": "parent:directx",
                "number": 1,
                "reason": "missing_parent_issue",
            },
            {
                "action": "created",
                "key": "backlog:directx:textures.gather",
                "number": 2,
                "reason": "missing_backlog_issue",
            },
            {
                "action": "attached",
                "parent_key": "parent:directx",
                "parent_number": 1,
                "child_key": "backlog:directx:textures.gather",
                "child_number": 2,
                "reason": "missing_sub_issue_relationship",
            },
        ],
    )

    reconciliation = report["operation_reconciliation"]
    assert reconciliation["evaluated"] is True
    assert reconciliation["ok"] is True
    assert reconciliation["planned_actions"]["created"] == 2
    assert reconciliation["actual_actions"] == {
        "created": 2,
        "updated": 0,
        "closed": 0,
        "attached": 1,
    }
    assert reconciliation["actual_action_reasons"] == {
        "created": {
            "missing_backlog_issue": 1,
            "missing_parent_issue": 1,
        },
        "updated": {},
        "closed": {},
        "attached": {
            "missing_sub_issue_relationship": 1,
        },
    }
    assert reconciliation["actual_closures"] == {
        "total": 0,
        "stale_parent": 0,
        "stale_backlog": 0,
        "stale_extracted": 0,
        "duplicate_marker": 0,
    }
    assert reconciliation["action_overruns"] == []
    assert reconciliation["action_shortfalls"] == []
    assert reconciliation["closure_overruns"] == []
    assert reconciliation["closure_shortfalls"] == []


def test_issue_sync_report_flags_operation_ledger_overruns():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())

    report = module.issue_sync_report(
        desired,
        mode="sync",
        close_extracted_issues=True,
        manage_sub_issues=True,
        existing_issues=[],
        operation_ledger=[
            {
                "action": "closed",
                "key": "parent:directx",
                "number": 1,
                "reason": "duplicate_managed_marker",
            }
        ],
    )

    reconciliation = report["operation_reconciliation"]
    assert reconciliation["evaluated"] is True
    assert reconciliation["ok"] is False
    assert reconciliation["planned_actions"]["closed"] == 0
    assert reconciliation["actual_actions"]["closed"] == 1
    assert reconciliation["action_overruns"] == [
        {
            "action": "closed",
            "actual": 1,
            "planned": 0,
        }
    ]
    assert reconciliation["closure_overruns"] == [
        {
            "category": "duplicate_marker",
            "actual": 1,
            "planned": 0,
        },
        {
            "category": "total",
            "actual": 1,
            "planned": 0,
        },
    ]
    assert reconciliation["action_shortfalls"] == [
        {
            "action": "attached",
            "actual": 0,
            "planned": 1,
        },
        {
            "action": "created",
            "actual": 0,
            "planned": 2,
        },
    ]
    assert reconciliation["closure_shortfalls"] == []


def test_issue_sync_report_flags_operation_ledger_shortfalls():
    module = load_sync_module()

    report = module.issue_sync_report(
        {},
        mode="sync",
        close_extracted_issues=True,
        manage_sub_issues=False,
        existing_issues=[issue(1, "parent:oldbackend")],
        operation_ledger=[],
    )

    reconciliation = report["operation_reconciliation"]
    assert reconciliation["evaluated"] is True
    assert reconciliation["ok"] is False
    assert reconciliation["planned_actions"]["closed"] == 1
    assert reconciliation["actual_actions"]["closed"] == 0
    assert reconciliation["action_overruns"] == []
    assert reconciliation["action_shortfalls"] == [
        {
            "action": "closed",
            "actual": 0,
            "planned": 1,
        }
    ]
    assert reconciliation["closure_overruns"] == []
    assert reconciliation["closure_shortfalls"] == [
        {
            "category": "stale_parent",
            "actual": 0,
            "planned": 1,
        },
        {
            "category": "total",
            "actual": 0,
            "planned": 1,
        },
    ]


def test_main_writes_dry_run_plan_without_github_access(tmp_path, capsys, monkeypatch):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "missing-signals.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_check_path = tmp_path / "support-matrix-check.json"
    event_path = tmp_path / "event.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    matrix_check_path.write_text(
        json.dumps(sample_matrix_check_report(ok=True)), encoding="utf-8"
    )
    event_path.write_text(
        json.dumps(
            {
                "workflow_run": {
                    "name": "Complete Test Suite",
                    "id": 26722241325,
                    "conclusion": "success",
                    "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--matrix-check-report",
            str(matrix_check_path),
            "--repo",
            "owner/repo",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["desired"]["total"] == 2
    assert report["existing"]["inspected"] is False
    assert report["planned_actions"] is None
    assert report["support_matrix_check"]["provided"] is True
    assert report["support_matrix_check"]["ok"] is True
    assert report["workflow_source"] == {
        "event": "workflow_run",
        "workflow": "Complete Test Suite",
        "run_id": "26722241325",
        "conclusion": "success",
        "head_sha": "731cb899d2cab99dd328e4299eb65d13a97d31e3",
    }
    captured = capsys.readouterr()
    assert "Dry run: would manage 2 desired issues" in captured.out
    assert "Stale extracted support issue closure is disabled" in captured.out
    assert "Stale pytest-failure support issue closure is disabled" in captured.out
    assert "Preserving existing" not in captured.out


def test_main_dry_run_with_inspection_prints_planned_summary(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "support-signals.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    signals_path.write_text(json.dumps(sample_signals()), encoding="utf-8")
    existing = [
        issue(1, "parent:directx"),
        issue(2, "backlog:directx:old.feature"),
        issue(
            3,
            "extracted:directx:docs.old-candidate:" "documented_candidate_not_detected",
        ),
    ]
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class InspectClient(FakeClient):
        instance = None

        def __init__(self, *_args, **_kwargs):
            super().__init__(existing=existing)
            InspectClient.instance = self

    monkeypatch.setattr(module, "GitHubClient", InspectClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["existing"]["inspected"] is True
    assert report["planned_actions"]["closed"] == 2
    assert report["planned_closures"]["stale_backlog"] == 1
    assert report["planned_closures"]["stale_extracted"] == 1
    captured = capsys.readouterr()
    assert "Dry run: would manage 4 desired issues" in captured.out
    assert (
        "Support issue sync: created=3, updated=1, closed=2, " "attached=3, unchanged=0"
    ) in captured.out
    assert "Preserving existing pytest-failure support issues" not in captured.out
    assert InspectClient.instance is not None
    assert InspectClient.instance.created == []
    assert InspectClient.instance.updated == []
    assert InspectClient.instance.closed == []
    assert InspectClient.instance.attached == []
    assert InspectClient.instance.labels == []


def test_main_dry_run_with_inspection_warns_for_existing_pytest_failures(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "support-signals.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    signals_path.write_text(json.dumps(sample_signals()), encoding="utf-8")
    stale_pytest = issue(
        77,
        "extracted:directx:ci.pytest.backend-codegen:pytest_failure_summary",
        labels=[module.LABEL_MANAGED, module.LABEL_EXTRACTED],
    )
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class InspectClient(FakeClient):
        def __init__(self, *_args, **_kwargs):
            super().__init__(existing=[stale_pytest])

    monkeypatch.setattr(module, "GitHubClient", InspectClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["planned_actions"]["closed"] == 0
    assert report["planned_actions"]["unchanged"] == 1
    assert report["planned_action_samples"]["preserved"][0]["reason"] == (
        "stale_pytest_failure_preserved"
    )
    assert (
        "Preserving existing pytest-failure support issues" in capsys.readouterr().out
    )


def test_main_preserves_pytest_failures_when_closure_is_explicitly_disabled(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "support-signals.json"
    plan_path = tmp_path / "support-issue-plan.json"
    signals = sample_signals()
    signals["summary"]["pytest_failures"] = {
        "provided": True,
        "report_count": 1,
        "load_error_count": 0,
        "failed_testcase_count": 0,
        "categories": {},
        "backends": {},
    }
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    signals_path.write_text(json.dumps(signals), encoding="utf-8")
    stale_pytest = issue(
        77,
        "extracted:directx:ci.pytest.backend-codegen:pytest_failure_summary",
        labels=[module.LABEL_MANAGED, module.LABEL_EXTRACTED],
    )
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class InspectClient(FakeClient):
        def __init__(self, *_args, **_kwargs):
            super().__init__(existing=[stale_pytest])

    monkeypatch.setattr(module, "GitHubClient", InspectClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--dry-run",
            "--preserve-pytest-failure-issues",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["close_pytest_failure_issues"] is False
    assert report["planned_actions"]["closed"] == 0
    assert report["planned_action_samples"]["preserved"][0]["reason"] == (
        "stale_pytest_failure_preserved"
    )
    assert (
        "Preserving existing pytest-failure support issues" in capsys.readouterr().out
    )


def test_main_writes_dry_run_plan_with_malformed_matrix_check_report(tmp_path, capsys):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "missing-signals.json"
    matrix_check_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    matrix_check_path.write_text("{not json", encoding="utf-8")

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--matrix-check-report",
            str(matrix_check_path),
            "--repo",
            "owner/repo",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    matrix_check = report["support_matrix_check"]
    assert matrix_check["provided"] is True
    assert matrix_check["ok"] is False
    assert matrix_check["load_error"]["path"] == str(matrix_check_path)
    assert matrix_check["load_error"]["type"] == "JSONDecodeError"
    assert "Dry run: would manage 2 desired issues" in capsys.readouterr().out


def test_main_writes_plan_on_malformed_matrix_input(tmp_path, capsys):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    matrix_check_path = tmp_path / "missing-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text("{not json", encoding="utf-8")

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--matrix-check-report",
            str(matrix_check_path),
            "--repo",
            "owner/repo",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 1
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["desired"]["total"] == 0
    assert report["existing"]["inspected"] is False
    assert report["planned_actions"] is None
    assert report["input_failures"][0]["input"] == "matrix"
    assert report["input_failures"][0]["path"] == str(matrix_path)
    assert report["input_failures"][0]["error"]["type"] == "JSONDecodeError"
    stderr = capsys.readouterr().err
    assert "Support issue input is invalid" in stderr
    assert "JSONDecodeError" in stderr


def test_main_continues_with_malformed_signals_input(tmp_path, capsys):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "support-signals.json"
    matrix_check_path = tmp_path / "missing-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    signals_path.write_text("{not json", encoding="utf-8")

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--matrix-check-report",
            str(matrix_check_path),
            "--repo",
            "owner/repo",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["desired"]["total"] == 2
    assert report["close_extracted_issues"] is False
    assert report["input_failures"][0]["input"] == "signals"
    assert report["input_failures"][0]["path"] == str(signals_path)
    assert report["input_failures"][0]["error"]["type"] == "JSONDecodeError"
    captured = capsys.readouterr()
    assert "Dry run: would manage 2 desired issues" in captured.out
    assert "continuing without signals" in captured.err


def test_load_signals_reports_missing_nested_issue_fields(tmp_path):
    module = load_sync_module()
    signals_path = tmp_path / "support-signals.json"
    signals = sample_signals()
    del signals["issues"][0]["backend_id"]
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    loaded = module.load_signals(signals_path)

    assert loaded["load_error"] == {
        "path": str(signals_path),
        "type": "MissingReportFields",
        "message": "issues[0] missing required fields: backend_id",
    }


def test_load_signals_reports_invalid_nested_issue_field_type(tmp_path):
    module = load_sync_module()
    signals_path = tmp_path / "support-signals.json"
    signals = sample_signals()
    signals["issues"][0]["backend_id"] = 17
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    loaded = module.load_signals(signals_path)

    assert loaded["load_error"] == {
        "path": str(signals_path),
        "type": "InvalidReportField",
        "message": "issues[0].backend_id must be str, got int",
    }


def test_load_signals_reports_invalid_feature_support_hit(tmp_path):
    module = load_sync_module()
    signals_path = tmp_path / "support-signals.json"
    signals = sample_signals()
    signals["features"][1]["support"]["directx"]["tests"] = [17]
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    loaded = module.load_signals(signals_path)

    assert loaded["load_error"] == {
        "path": str(signals_path),
        "type": "InvalidReportField",
        "message": "features[1].support.directx.tests[0] must be object, got int",
    }


def test_load_signals_reports_invalid_docs_probe_summary(tmp_path):
    module = load_sync_module()
    signals_path = tmp_path / "support-signals.json"
    signals = sample_signals()
    signals["summary"]["docs_probe"]["failed"] = "zero"
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    loaded = module.load_signals(signals_path)

    assert loaded["load_error"] == {
        "path": str(signals_path),
        "type": "InvalidReportField",
        "message": "summary.docs_probe.failed must be int, got str",
    }


def test_load_signals_reports_invalid_docs_probe_load_error(tmp_path):
    module = load_sync_module()
    signals_path = tmp_path / "support-signals.json"
    signals = sample_signals()
    signals["summary"]["docs_probe"]["load_error"] = {
        "path": "support/generated/backend-docs-report.json",
        "type": "InvalidReportField",
        "message": 17,
    }
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    loaded = module.load_signals(signals_path)

    assert loaded["load_error"] == {
        "path": str(signals_path),
        "type": "InvalidReportField",
        "message": "summary.docs_probe.load_error.message must be str, got int",
    }


def test_main_continues_with_invalid_signals_contract(tmp_path, capsys):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "support-signals.json"
    matrix_check_path = tmp_path / "missing-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    signals = sample_signals()
    signals["issues"][0]["key"] = "directx:target.codegen:bad"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    signals_path.write_text(json.dumps(signals), encoding="utf-8")

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--matrix-check-report",
            str(matrix_check_path),
            "--repo",
            "owner/repo",
            "--dry-run",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 0
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["desired"]["total"] == 2
    assert report["desired"]["extracted"] == 0
    assert report["close_extracted_issues"] is False
    assert report["input_failures"][0]["input"] == "signals"
    assert report["input_failures"][0]["error"] == {
        "path": str(signals_path),
        "type": "InvalidReportField",
        "message": "issues[0].key must start with extracted:",
    }
    captured = capsys.readouterr()
    assert "Dry run: would manage 2 desired issues" in captured.out
    assert "continuing without signals" in captured.err


def test_main_writes_plan_on_preflight_inspection_failure(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "missing-signals.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class FailingInspectClient(FakeClient):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

        def list_managed_issues(self):
            raise RuntimeError("inspection failed")

    monkeypatch.setattr(module, "GitHubClient", FailingInspectClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--dry-run",
            "--max-planned-created",
            "1",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 1
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["mode"] == "dry-run"
    assert report["existing"]["inspected"] is False
    assert report["planned_actions"] is None
    assert report["planned_action_budget"]["evaluated"] is False
    assert report["preflight_failure"] == {
        "phase": "list_managed_issues",
        "operation": {},
        "error": {
            "type": "RuntimeError",
            "message": "inspection failed",
        },
    }
    assert (
        "support issue sync preflight failed during list_managed_issues"
        in capsys.readouterr().err
    )


def test_main_writes_plan_before_failing_planned_action_budget(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    signals_path = tmp_path / "missing-signals.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class InspectClient(FakeClient):
        def __init__(self, *_args, **_kwargs):
            super().__init__(existing=[])

    monkeypatch.setattr(module, "GitHubClient", InspectClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--signals",
            str(signals_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--dry-run",
            "--max-planned-created",
            "1",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 1
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["planned_action_budget"]["ok"] is False
    assert report["planned_action_budget"]["violations"] == [
        {
            "action": "created",
            "actual": 2,
            "limit": 1,
        }
    ]
    assert "planned issue action budget exceeded" in capsys.readouterr().err


def test_main_writes_plan_before_failing_planned_closure_budget(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    plan_path = tmp_path / "support-issue-plan.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class InspectClient(FakeClient):
        def __init__(self, *_args, **_kwargs):
            super().__init__(existing=[issue(1, "parent:oldbackend")])

    monkeypatch.setattr(module, "GitHubClient", InspectClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--dry-run",
            "--max-planned-stale-parent-closures",
            "0",
            "--plan-output",
            str(plan_path),
        ]
    )

    assert result == 1
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["planned_closures"]["stale_parent"] == 1
    assert report["planned_closure_budget"]["ok"] is False
    assert report["planned_closure_budget"]["violations"] == [
        {
            "category": "stale_parent",
            "actual": 1,
            "limit": 0,
        }
    ]
    assert "planned issue closure budget exceeded" in capsys.readouterr().err


def test_main_writes_sync_failure_summary_on_mutation_failure(
    tmp_path, monkeypatch, capsys
):
    module = load_sync_module()
    matrix_path = tmp_path / "support-matrix.json"
    summary_path = tmp_path / "support-issue-sync-summary.json"
    matrix_path.write_text(json.dumps(sample_matrix()), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    class FailingCreateClient(FakeClient):
        def __init__(self, *_args, **_kwargs):
            super().__init__(existing=[issue(1, "parent:directx")])

        def create_issue(self, desired_issue):
            if desired_issue.key == "backlog:directx:textures.gather":
                raise RuntimeError("create failed")
            return super().create_issue(desired_issue)

    monkeypatch.setattr(module, "GitHubClient", FailingCreateClient)

    result = module.main(
        [
            "--matrix",
            str(matrix_path),
            "--repo",
            "owner/repo",
            "--inspect-existing",
            "--throttle-seconds",
            "0",
            "--sync-summary-output",
            str(summary_path),
        ]
    )

    assert result == 1
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report["mode"] == "sync"
    assert report["sync_summary"] == {
        "created": 0,
        "updated": 1,
        "closed": 0,
        "attached": 0,
        "unchanged": 0,
    }
    assert report["operation_ledger"] == [
        {
            "action": "updated",
            "key": "parent:directx",
            "number": 1,
            "title": "[Support Matrix] DirectX / HLSL coverage",
            "state": "open",
            "reason": "desired_issue_drift",
            "reasons": ["title", "body", "labels"],
        }
    ]
    assert report["sync_failure"] == {
        "phase": "create_issue",
        "operation": {
            "key": "backlog:directx:textures.gather",
            "title": "[Support Matrix][DirectX / HLSL] Texture gather (partial)",
            "parent_key": "parent:directx",
        },
        "partial_summary": {
            "created": 0,
            "updated": 1,
            "closed": 0,
            "attached": 0,
            "unchanged": 0,
        },
        "operation_ledger": [
            {
                "action": "updated",
                "key": "parent:directx",
                "number": 1,
                "title": "[Support Matrix] DirectX / HLSL coverage",
                "state": "open",
                "reason": "desired_issue_drift",
                "reasons": ["title", "body", "labels"],
            }
        ],
        "error": {
            "type": "RuntimeError",
            "message": "create failed",
        },
        "recovery": {
            "rerun_safe": True,
            "strategy": (
                "Rerun support issue sync after correcting the failure; "
                "managed issue markers make completed create, update, close, "
                "and attach operations idempotent."
            ),
        },
    }
    assert "support issue sync failed during create_issue" in capsys.readouterr().err


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


def test_sync_issues_prefers_open_duplicate_marker_over_closed_issue():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    target = desired["parent:directx"]
    closed_parent = issue(1, "parent:directx", state="closed")
    open_parent = {
        "id": 1002,
        "number": 2,
        "title": target.title,
        "body": target.body,
        "state": "open",
        "labels": [{"name": label} for label in target.labels],
    }
    client = FakeClient(existing=[closed_parent, open_parent])

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=False,
        throttle_seconds=0,
    )

    assert summary["updated"] == 0
    assert summary["closed"] == 0
    assert summary["unchanged"] == 2
    assert client.updated == []
    assert client.closed == []


def test_sync_issues_skips_update_for_unchanged_existing_issues():
    module = load_sync_module()
    desired = module.build_desired_issues(sample_matrix())
    existing = []
    for number, (key, target) in enumerate(desired.items(), start=1):
        existing.append(
            {
                "id": 1000 + number,
                "number": number,
                "title": target.title,
                "body": target.body,
                "state": "open",
                "labels": [{"name": label} for label in target.labels],
            }
        )
    client = FakeClient(existing=existing)

    summary = module.sync_issues(
        client,
        desired,
        dry_run=False,
        manage_sub_issues=False,
        throttle_seconds=0,
    )

    assert summary["created"] == 0
    assert summary["updated"] == 0
    assert summary["unchanged"] == len(desired)
    assert client.created == []
    assert client.updated == []


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
                        "provided": True,
                        "failed": 0,
                        "load_error": {
                            "path": "support/generated/backend-docs-report.json",
                            "type": "InvalidReportField",
                            "message": "docs report failed validation",
                        },
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


def test_signals_allow_pytest_failure_closure_for_clean_summary_input():
    module = load_sync_module()
    signals = sample_signals()
    signals["summary"]["pytest_failures"] = {
        "provided": True,
        "report_count": 1,
        "load_error_count": 0,
        "failed_testcase_count": 0,
        "categories": {},
        "backends": {},
    }

    assert module.signals_allow_pytest_failure_closure(signals) is True


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
            "0",
            "--inspect-existing",
            "--matrix-check-report",
            "support/generated/support-matrix-check.json",
            "--planned-action-budget-mode",
            "warn",
            "--max-planned-created",
            "300",
            "--max-planned-updated",
            "300",
            "--max-planned-closed",
            "500",
            "--max-planned-attached",
            "300",
            "--max-planned-total",
            "600",
            "--max-planned-stale-parent-closures",
            "10",
            "--max-planned-stale-backlog-closures",
            "250",
            "--max-planned-stale-extracted-closures",
            "250",
            "--max-planned-duplicate-marker-closures",
            "25",
            "--plan-output",
            "support/generated/support-issue-plan.json",
            "--sync-summary-output",
            "support/generated/support-issue-sync-summary.json",
            "--preserve-pytest-failure-issues",
        ]
    )

    assert args.max_retries == 6
    assert args.retry_base_seconds == 60
    assert args.retry_max_seconds == 600
    assert args.throttle_seconds == 2
    assert args.min_desired_issues == 0
    assert args.inspect_existing is True
    assert args.matrix_check_report == Path(
        "support/generated/support-matrix-check.json"
    )
    assert args.planned_action_budget_mode == "warn"
    assert args.max_planned_created == 300
    assert args.max_planned_updated == 300
    assert args.max_planned_closed == 500
    assert args.max_planned_attached == 300
    assert args.max_planned_total == 600
    assert args.max_planned_stale_parent_closures == 10
    assert args.max_planned_stale_backlog_closures == 250
    assert args.max_planned_stale_extracted_closures == 250
    assert args.max_planned_duplicate_marker_closures == 25
    assert args.plan_output == Path("support/generated/support-issue-plan.json")
    assert args.sync_summary_output == Path(
        "support/generated/support-issue-sync-summary.json"
    )
    assert args.preserve_pytest_failure_issues is True
