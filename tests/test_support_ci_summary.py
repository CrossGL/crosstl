import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "support_ci_summary.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("support_ci_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matrix_check_report(ok=True):
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py check",
        "ok": ok,
        "summary": {
            "artifact_count": 3,
            "stale_count": 0 if ok else 1,
            "stale_artifacts": [] if ok else ["support/generated/support-matrix.json"],
            "total_diff_line_count": 0 if ok else 18,
        },
        "artifacts": [
            {
                "path": "support/generated/support-matrix.json",
                "exists": True,
                "stale": not ok,
                "actual_sha256": "actual",
                "expected_sha256": "expected",
                "diff_line_count": 18 if not ok else 0,
                "diff": [] if ok else ["--- current", "+++ expected"],
            }
        ],
    }


def evidence_check_report(missing=2):
    rows = [
        {
            "backend": "DirectX / HLSL",
            "backend_id": "directx",
            "category": "stages",
            "feature": "Vertex stage",
            "feature_id": "stage.vertex",
            "status": "supported",
            "notes": "",
            "evidence_count": 0,
            "evidence": [],
        },
        {
            "backend": "Metal",
            "backend_id": "metal",
            "category": "resources",
            "feature": "Texture sampling",
            "feature_id": "resources.texture_sampling",
            "status": "supported",
            "notes": "",
            "evidence_count": 0,
            "evidence": [],
        },
        {
            "backend": "Slang",
            "backend_id": "slang",
            "category": "types",
            "feature": "Struct declarations and construction",
            "feature_id": "types.structs",
            "status": "supported",
            "notes": "",
            "evidence_count": 1,
            "evidence": [
                "tests/test_translator/test_codegen/"
                "test_slang_codegen.py::test_struct_constructor"
            ],
        },
    ]
    if missing == 0:
        for row in rows:
            row["evidence_count"] = 1
            row["evidence"] = ["tests/test_support_matrix.py"]
    return {
        "schema_version": 1,
        "generator": "tools/support_matrix.py evidence",
        "filters": {
            "backend_ids": [],
            "categories": [],
            "statuses": ["supported"],
            "evidence": "missing",
        },
        "summary": {
            "row_count": len(rows),
            "missing_evidence_count": missing,
            "present_evidence_count": len(rows) - missing,
            "by_backend": {
                "directx": {
                    "rows": 1,
                    "present": 0 if missing else 1,
                    "missing": 1 if missing else 0,
                },
                "metal": {
                    "rows": 1,
                    "present": 0 if missing else 1,
                    "missing": 1 if missing else 0,
                },
                "slang": {
                    "rows": 1,
                    "present": 1,
                    "missing": 0,
                },
            },
            "by_status": {"supported": len(rows)},
        },
        "rows": rows,
    }


def issue_plan_report():
    return {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": "dry-run",
        "desired": {
            "total": 42,
            "parents": 10,
            "backlog": 30,
            "extracted": 2,
        },
        "existing": {
            "inspected": True,
            "managed": 40,
            "duplicates": 1,
            "unmarked": 0,
        },
        "planned_actions": {
            "created": 2,
            "updated": 3,
            "closed": 1,
            "attached": 4,
            "unchanged": 35,
        },
        "planned_closures": {
            "total": 3,
            "stale_parent": 1,
            "stale_backlog": 1,
            "stale_extracted": 0,
            "duplicate_marker": 1,
        },
        "planned_action_samples": {
            "sample_limit": 12,
            "created": [
                {
                    "key": "parent:frontend",
                    "title": "[Support Matrix] Frontend / IR / Parser coverage",
                }
            ],
            "updated": [
                {
                    "key": "parent:directx",
                    "number": 17,
                    "title": "stale DirectX parent",
                    "state": "open",
                    "reasons": ["body", "labels"],
                }
            ],
            "closed": [
                {
                    "key": "backlog:directx:old.feature",
                    "number": 18,
                    "title": "stale backlog",
                    "state": "open",
                    "reason": "stale_managed_marker",
                }
            ],
            "attached": [
                {
                    "parent_key": "parent:directx",
                    "child_key": "backlog:directx:textures.gather",
                    "reason": "missing_relationship",
                }
            ],
            "preserved": [
                {
                    "key": (
                        "extracted:directx:docs.old:documented_candidate_not_detected"
                    ),
                    "number": 19,
                    "title": "stale extracted",
                    "state": "open",
                    "reason": "stale_extracted_preserved",
                }
            ],
        },
        "planned_action_budget": {
            "provided": True,
            "mode": "fail",
            "evaluated": True,
            "ok": False,
            "limits": {
                "closed": 0,
                "total": 5,
            },
            "violations": [
                {
                    "action": "closed",
                    "actual": 1,
                    "limit": 0,
                },
                {
                    "action": "total",
                    "actual": 10,
                    "limit": 5,
                },
            ],
        },
        "planned_closure_budget": {
            "provided": True,
            "mode": "fail",
            "evaluated": True,
            "ok": False,
            "limits": {
                "stale_parent": 0,
                "duplicate_marker": 0,
            },
            "violations": [
                {
                    "category": "duplicate_marker",
                    "actual": 1,
                    "limit": 0,
                },
                {
                    "category": "stale_parent",
                    "actual": 1,
                    "limit": 0,
                },
            ],
        },
        "support_matrix_check": {
            "provided": True,
            "ok": False,
            "summary": {
                "artifact_count": 3,
                "stale_count": 1,
                "stale_artifacts": ["support/generated/support-matrix.json"],
                "total_diff_line_count": 18,
            },
        },
        "preflight_failure": {
            "phase": "list_sub_issues",
            "operation": {
                "parent_key": "parent:directx",
                "parent_number": 17,
            },
            "error": {
                "type": "RuntimeError",
                "message": "sub-issue read failed",
            },
        },
    }


def sync_summary_report():
    return {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": "sync",
        "sync_summary": {
            "created": 1,
            "updated": 2,
            "closed": 0,
            "attached": 3,
            "unchanged": 44,
        },
        "operation_ledger": [
            {
                "action": "updated",
                "key": "parent:directx",
                "number": 17,
                "title": "DirectX parent",
                "state": "open",
                "reasons": ["body", "labels"],
            },
            {
                "action": "attached",
                "parent_key": "parent:directx",
                "parent_number": 17,
                "child_key": "backlog:directx:textures.gather",
                "child_number": 22,
            },
        ],
        "operation_reconciliation": {
            "evaluated": True,
            "ok": False,
            "planned_actions": {
                "created": 1,
                "updated": 1,
                "closed": 0,
                "attached": 2,
                "unchanged": 44,
            },
            "actual_actions": {
                "created": 0,
                "updated": 2,
                "closed": 0,
                "attached": 3,
            },
            "action_overruns": [
                {
                    "action": "attached",
                    "actual": 3,
                    "planned": 2,
                },
                {
                    "action": "updated",
                    "actual": 2,
                    "planned": 1,
                },
            ],
            "action_shortfalls": [
                {
                    "action": "created",
                    "actual": 0,
                    "planned": 1,
                }
            ],
            "planned_closures": {
                "total": 0,
                "stale_parent": 0,
                "stale_backlog": 0,
                "stale_extracted": 0,
                "duplicate_marker": 0,
            },
            "actual_closures": {
                "total": 0,
                "stale_parent": 0,
                "stale_backlog": 0,
                "stale_extracted": 0,
                "duplicate_marker": 0,
            },
            "closure_overruns": [],
            "closure_shortfalls": [],
        },
        "sync_failure": {
            "phase": "create_issue",
            "operation": {
                "key": "parent:frontend",
                "title": "[Support Matrix] Frontend / IR / Parser coverage",
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
                    "number": 17,
                    "title": "DirectX parent",
                    "state": "open",
                    "reasons": ["body", "labels"],
                }
            ],
            "error": {
                "type": "RuntimeError",
                "message": "create failed",
            },
            "recovery": {
                "rerun_safe": True,
                "strategy": "Rerun support issue sync after correcting the failure.",
            },
        },
    }


def clean_issue_plan_report():
    return {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": "dry-run",
        "desired": {
            "total": 3,
            "parents": 2,
            "backlog": 1,
            "extracted": 0,
        },
        "existing": {
            "inspected": False,
            "managed": 0,
            "duplicates": 0,
        },
        "support_matrix_check": {
            "provided": True,
            "ok": True,
            "summary": {
                "artifact_count": 3,
                "stale_count": 0,
                "stale_artifacts": [],
                "total_diff_line_count": 0,
            },
        },
    }


def clean_sync_summary_report():
    return {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": "sync",
        "sync_summary": {
            "created": 0,
            "updated": 0,
            "closed": 0,
            "attached": 0,
            "unchanged": 3,
        },
    }


def test_render_summary_includes_stale_matrix_plan_and_sync_counts():
    module = load_summary_module()

    text = module.render_summary(
        matrix_check_report(ok=False),
        Path("support/generated/support-matrix-check.json"),
        issue_plan_report(),
        Path("support/generated/support-issue-plan.json"),
        sync_summary_report(),
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "# Support Automation Summary" in text
    assert "## Overall" in text
    assert "| Overall | attention |" in text
    assert "| Support matrix | fail |" in text
    assert "| Issue plan | fail |" in text
    assert "| Issue sync | fail |" in text
    assert "| Status | fail |" in text
    assert "`support/generated/support-matrix.json`: 18 diff lines" in text
    assert "| Desired total | 42 |" in text
    assert "| Planned created | 2 |" in text
    assert "| Planned closure stale_parent | 1 |" in text
    assert "| Planned action budget | fail |" in text
    assert "| Planned closure budget | fail |" in text
    assert "- total: 10 > 5" in text
    assert "Closure budget violations:" in text
    assert "- duplicate_marker: 1 > 0" in text
    assert "- stale_parent: 1 > 0" in text
    assert "| Preflight failure phase | list_sub_issues |" in text
    assert "| Preflight failure error | RuntimeError |" in text
    assert "| Preflight failure message | sub-issue read failed |" in text
    assert "Preflight failure operation:" in text
    assert "- parent_key: `parent:directx`" in text
    assert "Planned action samples:" in text
    assert "- created: `parent:frontend`" in text
    assert "- updated: `parent:directx` (#17) (reasons=body,labels)" in text
    assert (
        "- closed: `backlog:directx:old.feature` (#18) " "(reason=stale_managed_marker)"
    ) in text
    assert (
        "- attached: `backlog:directx:textures.gather` "
        "(reason=missing_relationship, parent=parent:directx)"
    ) in text
    assert (
        "- preserved: `extracted:directx:docs.old:documented_candidate_not_detected` "
        "(#19) (reason=stale_extracted_preserved)"
    ) in text
    assert "| Embedded matrix check | fail |" in text
    assert "| Sync attached | 3 |" in text
    assert "| Operation ledger entries | 2 |" in text
    assert "| Operation reconciliation | fail |" in text
    assert "| Operation action overruns | 2 |" in text
    assert "| Operation action shortfalls | 1 |" in text
    assert "| Operation closure overruns | 0 |" in text
    assert "| Operation closure shortfalls | 0 |" in text
    assert "Operation reconciliation differences:" in text
    assert "- action attached: 3 > planned 2" in text
    assert "- action updated: 2 > planned 1" in text
    assert "- action created: 0 < planned 1" in text
    assert "| Sync failure phase | create_issue |" in text
    assert "| Sync failure error | RuntimeError |" in text
    assert "| Sync recovery rerun safe | True |" in text
    assert (
        "| Sync recovery strategy | Rerun support issue sync after correcting the failure. |"
        in text
    )
    assert "- key: `parent:frontend`" in text
    assert "Operation ledger:" in text
    assert "- updated: `parent:directx` (#17) (reasons=body,labels)" in text
    assert (
        "- attached: `backlog:directx:textures.gather` (#22) "
        "(parent=parent:directx, parent_number=17)"
    ) in text


def test_render_summary_handles_missing_reports():
    module = load_summary_module()

    text = module.render_summary(
        None,
        Path("support/generated/support-matrix-check.json"),
        None,
        Path("support/generated/support-issue-plan.json"),
        None,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | incomplete |" in text
    assert "| Support matrix | missing |" in text
    assert "| Issue plan | missing |" in text
    assert "| Issue sync | missing |" in text
    assert (
        "Report: not available at `support/generated/support-matrix-check.json`."
        in text
    )
    assert (
        "Report: not available at `support/generated/support-issue-plan.json`." in text
    )
    assert (
        "Report: not available at `support/generated/support-issue-sync-summary.json`."
        in text
    )


def test_render_summary_handles_stale_matrix_with_missing_plan_and_sync():
    module = load_summary_module()

    text = module.render_summary(
        matrix_check_report(ok=False),
        Path("support/generated/support-matrix-check.json"),
        None,
        Path("support/generated/support-issue-plan.json"),
        None,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | attention |" in text
    assert "| Support matrix | fail |" in text
    assert "| Issue plan | missing |" in text
    assert "| Issue sync | missing |" in text
    assert "| Status | fail |" in text
    assert "`support/generated/support-matrix.json`: 18 diff lines" in text
    assert (
        "Report: not available at `support/generated/support-issue-plan.json`." in text
    )
    assert (
        "Report: not available at `support/generated/support-issue-sync-summary.json`."
        in text
    )


def test_render_summary_includes_support_evidence_gaps():
    module = load_summary_module()

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        clean_issue_plan_report(),
        Path("support/generated/support-issue-plan.json"),
        clean_sync_summary_report(),
        Path("support/generated/support-issue-sync-summary.json"),
        evidence_check=evidence_check_report(),
        evidence_check_path=Path("support/generated/support-evidence-check.json"),
    )

    assert "| Overall | pass |" in text
    assert "| Support evidence | warning |" in text
    assert "## Support Evidence" in text
    assert "| Report | `support/generated/support-evidence-check.json` |" in text
    assert "| Rows missing evidence | 2 |" in text
    assert "Missing evidence by backend:" in text
    assert "| directx | 1 | 0 | 1 |" in text
    assert "Missing evidence samples:" in text
    assert "- DirectX / HLSL: Vertex stage [supported]" in text
    assert "- Metal: Texture sampling [supported]" in text


def test_render_summary_marks_explicit_missing_evidence_report_incomplete():
    module = load_summary_module()

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        clean_issue_plan_report(),
        Path("support/generated/support-issue-plan.json"),
        clean_sync_summary_report(),
        Path("support/generated/support-issue-sync-summary.json"),
        evidence_check=None,
        evidence_check_path=Path("support/generated/support-evidence-check.json"),
    )

    assert "| Overall | incomplete |" in text
    assert "| Support evidence | missing |" in text
    assert (
        "Report: not available at `support/generated/support-evidence-check.json`."
        in text
    )


def test_render_issue_plan_reports_embedded_matrix_check_load_error():
    module = load_summary_module()
    plan = issue_plan_report()
    plan["support_matrix_check"] = {
        "provided": True,
        "path": "support/generated/support-matrix-check.json",
        "ok": False,
        "summary": {},
        "stale_artifacts": [],
        "load_error": {
            "type": "JSONDecodeError",
            "message": "Expecting property name",
        },
    }

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        plan,
        Path("support/generated/support-issue-plan.json"),
        None,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | attention |" in text
    assert "| Issue plan | fail |" in text
    assert "| Embedded matrix check | fail |" in text
    assert "| Embedded matrix check error | JSONDecodeError |" in text
    assert "| Embedded matrix check message | Expecting property name |" in text


def test_render_issue_plan_reports_support_input_failures():
    module = load_summary_module()
    plan = clean_issue_plan_report()
    plan["input_failures"] = [
        {
            "input": "signals",
            "path": "support/generated/support-signals.json",
            "error": {
                "type": "JSONDecodeError",
                "message": "Expecting property name",
            },
        }
    ]

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        plan,
        Path("support/generated/support-issue-plan.json"),
        None,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | attention |" in text
    assert "| Issue plan | fail |" in text
    assert "| Input failures | 1 |" in text
    assert "Input failures:" in text
    assert "- signals: JSONDecodeError - Expecting property name" in text


def test_render_issue_plan_reports_managed_issue_audit():
    module = load_summary_module()
    plan = clean_issue_plan_report()
    plan["managed_issue_audit"] = {
        "sample_limit": 12,
        "stale": {
            "total": 2,
            "open": 1,
            "closed": 1,
            "samples": [
                {
                    "key": "backlog:directx:old.feature",
                    "number": 18,
                    "title": "old backlog",
                    "state": "open",
                    "reason": "stale_managed_marker",
                    "category": "stale_backlog",
                }
            ],
        },
        "duplicates": {
            "total": 1,
            "open": 1,
            "closed": 0,
            "samples": [
                {
                    "key": "parent:directx",
                    "number": 19,
                    "title": "duplicate",
                    "state": "open",
                    "reason": "duplicate_managed_marker",
                }
            ],
        },
        "preserved_extracted": {
            "total": 1,
            "open": 1,
            "closed": 0,
            "samples": [],
        },
        "ignored_unknown": {
            "total": 1,
            "open": 1,
            "closed": 0,
            "samples": [],
        },
    }

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        plan,
        Path("support/generated/support-issue-plan.json"),
        None,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Audit stale managed | 2 |" in text
    assert "| Audit duplicate markers | 1 |" in text
    assert "| Audit preserved extracted | 1 |" in text
    assert "| Audit ignored unknown | 1 |" in text
    assert "Managed issue audit:" in text
    assert "- Stale managed: total=2, open=1, closed=1" in text
    assert (
        "  - `backlog:directx:old.feature` (#18) "
        "(reason=stale_managed_marker, stale_backlog)"
    ) in text
    assert "- Duplicate markers: total=1, open=1, closed=0" in text
    assert "  - `parent:directx` (#19) (reason=duplicate_managed_marker)" in text


def test_github_annotations_include_actionable_support_failures():
    module = load_summary_module()
    plan = clean_issue_plan_report()
    plan["input_failures"] = [
        {
            "input": "signals",
            "path": "support/generated/support-signals.json",
            "error": {
                "type": "JSONDecodeError",
                "message": "bad\njson",
            },
        }
    ]

    lines = module.github_annotation_lines(
        matrix_check_report(ok=False),
        Path("support/generated/support-matrix-check.json"),
        plan,
        Path("support/generated/support-issue-plan.json"),
        sync_summary_report(),
        Path("support/generated/support-issue-sync-summary.json"),
    )

    text = "\n".join(lines)
    assert "::error" in text
    assert "title=Support matrix check failed" in text
    assert "title=Stale support matrix artifact" in text
    assert "file=support/generated/support-signals.json" in text
    assert "signals: JSONDecodeError - bad%0Ajson" in text
    assert "title=Support issue sync failure" in text
    assert "title=Support issue sync exceeded planned actions" in text
    assert "attached: 3 > planned 2" in text


def test_github_annotations_warn_on_support_evidence_gaps():
    module = load_summary_module()

    lines = module.github_annotation_lines(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        clean_issue_plan_report(),
        Path("support/generated/support-issue-plan.json"),
        None,
        Path("support/generated/support-issue-sync-summary.json"),
        evidence_check=evidence_check_report(),
        evidence_check_path=Path("support/generated/support-evidence-check.json"),
    )

    text = "\n".join(lines)
    assert "::warning" in text
    assert "title=Support matrix evidence gaps" in text
    assert "file=support/generated/support-evidence-check.json" in text
    assert "2 supported support-matrix rows are missing evidence" in text
    assert "directx=1" in text


def test_render_summary_fails_when_sync_operations_exceed_plan():
    module = load_summary_module()
    sync_report = {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": "sync",
        "sync_summary": {
            "created": 0,
            "updated": 0,
            "closed": 1,
            "attached": 0,
            "unchanged": 0,
        },
        "operation_reconciliation": {
            "evaluated": True,
            "ok": False,
            "planned_actions": {
                "created": 0,
                "updated": 0,
                "closed": 0,
                "attached": 0,
                "unchanged": 0,
            },
            "actual_actions": {
                "created": 0,
                "updated": 0,
                "closed": 1,
                "attached": 0,
            },
            "action_overruns": [
                {
                    "action": "closed",
                    "actual": 1,
                    "planned": 0,
                }
            ],
            "action_shortfalls": [],
            "planned_closures": {
                "total": 0,
                "stale_parent": 0,
                "stale_backlog": 0,
                "stale_extracted": 0,
                "duplicate_marker": 0,
            },
            "actual_closures": {
                "total": 1,
                "stale_parent": 0,
                "stale_backlog": 0,
                "stale_extracted": 0,
                "duplicate_marker": 1,
            },
            "closure_overruns": [
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
            ],
            "closure_shortfalls": [],
        },
    }

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        clean_issue_plan_report(),
        Path("support/generated/support-issue-plan.json"),
        sync_report,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | attention |" in text
    assert "| Issue sync | fail |" in text
    assert "| Operation reconciliation | fail |" in text
    assert "- action closed: 1 > planned 0" in text
    assert "- closure duplicate_marker: 1 > planned 0" in text
    assert "- closure total: 1 > planned 0" in text


def test_render_summary_fails_when_sync_operations_miss_plan():
    module = load_summary_module()
    sync_report = {
        "schema_version": 1,
        "generator": "tools/sync_support_issues.py",
        "mode": "sync",
        "sync_summary": {
            "created": 0,
            "updated": 0,
            "closed": 0,
            "attached": 0,
            "unchanged": 0,
        },
        "operation_reconciliation": {
            "evaluated": True,
            "ok": False,
            "planned_actions": {
                "created": 1,
                "updated": 0,
                "closed": 1,
                "attached": 0,
                "unchanged": 0,
            },
            "actual_actions": {
                "created": 0,
                "updated": 0,
                "closed": 0,
                "attached": 0,
            },
            "action_overruns": [],
            "action_shortfalls": [
                {
                    "action": "created",
                    "actual": 0,
                    "planned": 1,
                },
                {
                    "action": "closed",
                    "actual": 0,
                    "planned": 1,
                },
            ],
            "planned_closures": {
                "total": 1,
                "stale_parent": 1,
                "stale_backlog": 0,
                "stale_extracted": 0,
                "duplicate_marker": 0,
            },
            "actual_closures": {
                "total": 0,
                "stale_parent": 0,
                "stale_backlog": 0,
                "stale_extracted": 0,
                "duplicate_marker": 0,
            },
            "closure_overruns": [],
            "closure_shortfalls": [
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
            ],
        },
    }

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        clean_issue_plan_report(),
        Path("support/generated/support-issue-plan.json"),
        sync_report,
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | attention |" in text
    assert "| Issue sync | fail |" in text
    assert "| Operation reconciliation | fail |" in text
    assert "| Operation action shortfalls | 2 |" in text
    assert "| Operation closure shortfalls | 2 |" in text
    assert "- action created: 0 < planned 1" in text
    assert "- action closed: 0 < planned 1" in text
    assert "- closure stale_parent: 0 < planned 1" in text
    assert "- closure total: 0 < planned 1" in text


def test_load_optional_json_reports_malformed_artifact(tmp_path):
    module = load_summary_module()
    broken_path = tmp_path / "support-issue-plan.json"
    broken_path.write_text("{not json", encoding="utf-8")

    report = module.load_optional_json(broken_path)

    assert report["load_error"]["path"] == str(broken_path)
    assert report["load_error"]["type"] == "JSONDecodeError"
    assert "Expecting property name" in report["load_error"]["message"]


def test_load_optional_json_reports_non_object_artifact(tmp_path):
    module = load_summary_module()
    wrong_shape_path = tmp_path / "support-issue-plan.json"
    wrong_shape_path.write_text("[]", encoding="utf-8")

    report = module.load_optional_json(wrong_shape_path)

    assert report["load_error"] == {
        "path": str(wrong_shape_path),
        "type": "InvalidReportType",
        "message": "expected JSON object, got list",
    }


def test_load_optional_json_reports_missing_required_fields(tmp_path):
    module = load_summary_module()
    incomplete_path = tmp_path / "support-matrix-check.json"
    incomplete_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "tools/support_matrix.py check",
            }
        ),
        encoding="utf-8",
    )

    report = module.load_optional_json(
        incomplete_path,
        expected_generator="tools/support_matrix.py check",
        required_fields=("schema_version", "generator", "ok", "summary"),
    )

    assert report["load_error"] == {
        "path": str(incomplete_path),
        "type": "MissingReportFields",
        "message": "missing required fields: ok, summary",
    }


def test_load_optional_json_reports_schema_and_generator_mismatches(tmp_path):
    module = load_summary_module()
    wrong_schema_path = tmp_path / "support-issue-plan.json"
    wrong_schema_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generator": "tools/sync_support_issues.py",
                "mode": "dry-run",
            }
        ),
        encoding="utf-8",
    )
    wrong_generator_path = tmp_path / "support-matrix-check.json"
    wrong_generator_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "tools/sync_support_issues.py",
                "ok": True,
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    wrong_schema = module.load_optional_json(
        wrong_schema_path,
        expected_generator="tools/sync_support_issues.py",
        required_fields=("schema_version", "generator", "mode"),
    )
    wrong_generator = module.load_optional_json(
        wrong_generator_path,
        expected_generator="tools/support_matrix.py check",
        required_fields=("schema_version", "generator", "ok", "summary"),
    )

    assert wrong_schema["load_error"] == {
        "path": str(wrong_schema_path),
        "type": "UnsupportedSchemaVersion",
        "message": "expected schema_version 1, got 2",
    }
    assert wrong_generator["load_error"] == {
        "path": str(wrong_generator_path),
        "type": "UnexpectedReportGenerator",
        "message": (
            "expected generator tools/support_matrix.py check, got "
            "tools/sync_support_issues.py"
        ),
    }


def test_load_optional_json_reports_invalid_matrix_check_contract(tmp_path):
    module = load_summary_module()
    matrix_path = tmp_path / "support-matrix-check.json"
    report = matrix_check_report(ok=True)
    report["summary"]["stale_count"] = "none"
    matrix_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        matrix_path,
        expected_generator=module.MATRIX_CHECK_GENERATOR,
        required_fields=module.MATRIX_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_matrix_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(matrix_path),
        "type": "InvalidReportField",
        "message": "summary.stale_count must be int, got str",
    }


def test_load_optional_json_reports_invalid_evidence_check_contract(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["summary"]["by_backend"]["directx"]["missing"] = "one"
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "summary.by_backend.directx.missing must be int, got str",
    }


def test_load_optional_json_rejects_invalid_evidence_filters(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["filters"]["statuses"] = ["supported", 7]
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "filters.statuses[1] must be str, got int",
    }


def test_load_optional_json_rejects_invalid_evidence_row_items(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["rows"][2]["evidence"] = [{"path": "tests/test_support_matrix.py"}]
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "rows[2].evidence[0] must be str, got object",
    }


def test_load_optional_json_rejects_missing_evidence_row_fields(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    del report["rows"][2]["feature_id"]
    del report["rows"][2]["evidence"]
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "MissingReportFields",
        "message": "rows[2] missing required fields: feature_id, evidence",
    }


def test_load_optional_json_rejects_evidence_count_mismatch(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["rows"][2]["evidence_count"] = 2
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "rows[2].evidence_count must match evidence length: 2 != 1",
    }


def test_load_optional_json_rejects_evidence_summary_mismatch(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["summary"]["row_count"] = 99
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "summary.row_count must match rows length: 99 != 3",
    }


def test_load_optional_json_rejects_evidence_backend_summary_mismatch(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["summary"]["by_backend"]["directx"]["missing"] = 0
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "summary.by_backend must match rows",
    }


def test_load_optional_json_rejects_evidence_status_summary_mismatch(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["summary"]["by_status"]["supported"] = 99
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert loaded["load_error"] == {
        "path": str(evidence_path),
        "type": "InvalidReportField",
        "message": "summary.by_status must match rows",
    }


def test_load_optional_json_allows_zero_evidence_status_buckets(tmp_path):
    module = load_summary_module()
    evidence_path = tmp_path / "support-evidence-check.json"
    report = evidence_check_report()
    report["summary"]["by_status"] = {
        "supported": 3,
        "partial": 0,
        "diagnostic": 0,
        "validated_rejection": 0,
        "unsupported": 0,
        "unknown": 0,
    }
    evidence_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        evidence_path,
        expected_generator=module.EVIDENCE_CHECK_GENERATOR,
        required_fields=module.EVIDENCE_CHECK_REQUIRED_FIELDS,
        contract_validator=module.validate_evidence_check_contract,
    )

    assert "load_error" not in loaded


def test_load_optional_json_reports_invalid_issue_plan_contract(tmp_path):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = issue_plan_report()
    report["planned_actions"]["created"] = "two"
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "InvalidReportField",
        "message": "planned_actions.created must be int, got str",
    }


def test_load_optional_json_reports_invalid_planned_action_sample_limit(tmp_path):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = issue_plan_report()
    report["planned_action_samples"]["sample_limit"] = "twelve"
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "InvalidReportField",
        "message": "planned_action_samples.sample_limit must be int, got str",
    }


def test_load_optional_json_reports_invalid_planned_action_sample_contract(tmp_path):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = issue_plan_report()
    report["planned_action_samples"]["updated"][0]["reasons"] = ["body", 17]
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "InvalidReportField",
        "message": "planned_action_samples.updated[0].reasons[1] must be str, got int",
    }


def test_load_optional_json_reports_missing_planned_action_sample_fields(tmp_path):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = issue_plan_report()
    del report["planned_action_samples"]["attached"][0]["child_key"]
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "MissingReportFields",
        "message": (
            "planned_action_samples.attached[0] missing required fields: child_key"
        ),
    }


def test_load_optional_json_reports_invalid_issue_plan_budget_violation_contract(
    tmp_path,
):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = issue_plan_report()
    report["planned_action_budget"]["violations"][0]["actual"] = "ten"
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "InvalidReportField",
        "message": "planned_action_budget.violations[0].actual must be int, got str",
    }


def test_load_optional_json_reports_invalid_managed_issue_audit_bucket(tmp_path):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = clean_issue_plan_report()
    report["managed_issue_audit"] = {
        "sample_limit": 12,
        "stale": {
            "total": "two",
            "open": 1,
            "closed": 1,
            "samples": [],
        },
        "duplicates": {
            "total": 0,
            "open": 0,
            "closed": 0,
            "samples": [],
        },
        "preserved_extracted": {
            "total": 0,
            "open": 0,
            "closed": 0,
            "samples": [],
        },
        "ignored_unknown": {
            "total": 0,
            "open": 0,
            "closed": 0,
            "samples": [],
        },
    }
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "InvalidReportField",
        "message": "managed_issue_audit.stale.total must be int, got str",
    }


def test_load_optional_json_reports_invalid_managed_issue_audit_sample(tmp_path):
    module = load_summary_module()
    plan_path = tmp_path / "support-issue-plan.json"
    report = clean_issue_plan_report()
    report["managed_issue_audit"] = {
        "sample_limit": 12,
        "stale": {
            "total": 1,
            "open": 1,
            "closed": 0,
            "samples": [
                {
                    "key": "backlog:directx:old.feature",
                    "number": "18",
                    "title": "old backlog",
                    "state": "open",
                    "reason": "stale_managed_marker",
                }
            ],
        },
        "duplicates": {
            "total": 0,
            "open": 0,
            "closed": 0,
            "samples": [],
        },
        "preserved_extracted": {
            "total": 0,
            "open": 0,
            "closed": 0,
            "samples": [],
        },
        "ignored_unknown": {
            "total": 0,
            "open": 0,
            "closed": 0,
            "samples": [],
        },
    }
    plan_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        plan_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=module.validate_issue_plan_contract,
    )

    assert loaded["load_error"] == {
        "path": str(plan_path),
        "type": "InvalidReportField",
        "message": "managed_issue_audit.stale.samples[0].number must be int, got str",
    }


def test_load_optional_json_reports_invalid_sync_summary_contract(tmp_path):
    module = load_summary_module()
    sync_path = tmp_path / "support-issue-sync-summary.json"
    report = sync_summary_report()
    report["operation_ledger"] = {"action": "updated"}
    sync_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        sync_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.SYNC_SUMMARY_REQUIRED_FIELDS,
        contract_validator=module.validate_sync_summary_contract,
    )

    assert loaded["load_error"] == {
        "path": str(sync_path),
        "type": "InvalidReportField",
        "message": "operation_ledger must be list, got object",
    }


def test_load_optional_json_reports_invalid_operation_ledger_contract(tmp_path):
    module = load_summary_module()
    sync_path = tmp_path / "support-issue-sync-summary.json"
    report = sync_summary_report()
    report["operation_ledger"][0]["reasons"] = ["body", 17]
    sync_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        sync_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.SYNC_SUMMARY_REQUIRED_FIELDS,
        contract_validator=module.validate_sync_summary_contract,
    )

    assert loaded["load_error"] == {
        "path": str(sync_path),
        "type": "InvalidReportField",
        "message": "operation_ledger[0].reasons[1] must be str, got int",
    }


def test_load_optional_json_reports_invalid_reconciliation_counter_contract(tmp_path):
    module = load_summary_module()
    sync_path = tmp_path / "support-issue-sync-summary.json"
    report = sync_summary_report()
    report["operation_reconciliation"]["actual_actions"]["updated"] = "two"
    sync_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        sync_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.SYNC_SUMMARY_REQUIRED_FIELDS,
        contract_validator=module.validate_sync_summary_contract,
    )

    assert loaded["load_error"] == {
        "path": str(sync_path),
        "type": "InvalidReportField",
        "message": (
            "operation_reconciliation.actual_actions.updated must be int, got str"
        ),
    }


def test_load_optional_json_reports_missing_reconciliation_fields(tmp_path):
    module = load_summary_module()
    sync_path = tmp_path / "support-issue-sync-summary.json"
    report = sync_summary_report()
    del report["operation_reconciliation"]["action_shortfalls"]
    sync_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        sync_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.SYNC_SUMMARY_REQUIRED_FIELDS,
        contract_validator=module.validate_sync_summary_contract,
    )

    assert loaded["load_error"] == {
        "path": str(sync_path),
        "type": "MissingReportFields",
        "message": (
            "operation_reconciliation missing required fields: action_shortfalls"
        ),
    }


def test_load_optional_json_reports_invalid_reconciliation_difference_contract(
    tmp_path,
):
    module = load_summary_module()
    sync_path = tmp_path / "support-issue-sync-summary.json"
    report = sync_summary_report()
    report["operation_reconciliation"]["action_shortfalls"][0]["planned"] = "one"
    sync_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_optional_json(
        sync_path,
        expected_generator=module.ISSUE_SYNC_GENERATOR,
        required_fields=module.SYNC_SUMMARY_REQUIRED_FIELDS,
        contract_validator=module.validate_sync_summary_contract,
    )

    assert loaded["load_error"] == {
        "path": str(sync_path),
        "type": "InvalidReportField",
        "message": "action_shortfalls[0].planned must be int, got str",
    }


def test_support_ci_summary_cli_survives_malformed_reports(tmp_path):
    matrix_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "support-issue-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    matrix_path.write_text(json.dumps(matrix_check_report(ok=False)), encoding="utf-8")
    plan_path.write_text("{not json", encoding="utf-8")
    sync_path.write_text("[", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Wrote" in result.stdout
    assert "| Status | fail |" in text
    assert "## Issue Plan" in text
    assert "| Overall | attention |" in text
    assert "| Issue plan | load-error |" in text
    assert "| Issue sync | load-error |" in text
    assert "Report: failed to load" in text
    assert "| Error | JSONDecodeError |" in text
    assert "## Issue Sync" in text


def test_support_ci_summary_cli_survives_non_object_reports(tmp_path):
    matrix_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "support-issue-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    matrix_path.write_text("[]", encoding="utf-8")
    plan_path.write_text(json.dumps(issue_plan_report()), encoding="utf-8")
    sync_path.write_text('"wrong shape"', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Wrote" in result.stdout
    assert "| Support matrix | load-error |" in text
    assert "| Issue sync | load-error |" in text
    assert "| Error | InvalidReportType |" in text
    assert "expected JSON object, got list" in text
    assert "expected JSON object, got str" in text


def test_support_ci_summary_cli_reports_schema_and_provenance_errors(tmp_path):
    matrix_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "support-issue-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    wrong_matrix = matrix_check_report(ok=True)
    wrong_matrix["generator"] = "tools/sync_support_issues.py"
    incomplete_plan = issue_plan_report()
    del incomplete_plan["desired"]
    wrong_sync = sync_summary_report()
    wrong_sync["schema_version"] = 2
    matrix_path.write_text(json.dumps(wrong_matrix), encoding="utf-8")
    plan_path.write_text(json.dumps(incomplete_plan), encoding="utf-8")
    sync_path.write_text(json.dumps(wrong_sync), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Wrote" in result.stdout
    assert "| Overall | attention |" in text
    assert "| Support matrix | load-error |" in text
    assert "| Issue plan | load-error |" in text
    assert "| Issue sync | load-error |" in text
    assert "| Error | UnexpectedReportGenerator |" in text
    assert "| Error | MissingReportFields |" in text
    assert "| Error | UnsupportedSchemaVersion |" in text
    assert "missing required fields: desired" in text
    assert "expected schema_version 1, got 2" in text


def test_support_ci_summary_cli_reports_contract_errors(tmp_path):
    matrix_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "support-issue-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    matrix = matrix_check_report(ok=True)
    matrix["ok"] = "yes"
    plan = issue_plan_report()
    plan["existing"]["inspected"] = "true"
    sync = sync_summary_report()
    sync["sync_summary"]["attached"] = "three"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    sync_path.write_text(json.dumps(sync), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Wrote" in result.stdout
    assert "| Overall | attention |" in text
    assert "| Support matrix | load-error |" in text
    assert "| Issue plan | load-error |" in text
    assert "| Issue sync | load-error |" in text
    assert "ok must be bool, got str" in text
    assert "existing.inspected must be bool, got str" in text
    assert "sync_summary.attached must be int, got str" in text


def test_support_ci_summary_cli_appends_summary_annotates_and_fails_on_attention(
    tmp_path,
):
    matrix_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "support-issue-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    step_summary_path = tmp_path / "step-summary.md"
    plan = clean_issue_plan_report()
    plan["input_failures"] = [
        {
            "input": "signals",
            "path": "support/generated/support-signals.json",
            "error": {
                "type": "JSONDecodeError",
                "message": "bad json",
            },
        }
    ]
    matrix_path.write_text(json.dumps(matrix_check_report(ok=True)), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    sync_path.write_text(json.dumps(sync_summary_report()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
            "--step-summary",
            str(step_summary_path),
            "--github-annotations",
            "--fail-on-attention",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    text = output_path.read_text(encoding="utf-8")
    assert step_summary_path.read_text(encoding="utf-8") == text
    assert "| Overall | attention |" in text
    assert "Wrote" in result.stdout
    assert "Appended" in result.stdout
    assert "::error" in result.stdout
    assert "title=Support issue input failure" in result.stdout


def test_support_ci_summary_cli_fail_on_attention_allows_incomplete_clean_pr_summary(
    tmp_path,
):
    matrix_path = tmp_path / "support-matrix-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "missing-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    matrix_path.write_text(json.dumps(matrix_check_report(ok=True)), encoding="utf-8")
    plan_path.write_text(json.dumps(clean_issue_plan_report()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
            "--fail-on-attention",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    text = output_path.read_text(encoding="utf-8")
    assert "| Overall | incomplete |" in text
    assert "| Issue sync | missing |" in text


def test_markdown_table_escapes_pipe_and_normalizes_newlines():
    module = load_summary_module()

    text = module.markdown_table(
        ["Field|Name", "Value"],
        [["Message", "first | second\nthird"]],
    )

    assert "| Field\\|Name | Value |" in text
    assert "| Message | first \\| second third |" in text


def test_render_summary_reports_pass_when_all_sections_are_clean():
    module = load_summary_module()

    text = module.render_summary(
        matrix_check_report(ok=True),
        Path("support/generated/support-matrix-check.json"),
        {
            "mode": "dry-run",
            "desired": {
                "total": 3,
                "parents": 2,
                "backlog": 1,
                "extracted": 0,
            },
            "existing": {
                "inspected": False,
                "managed": 0,
                "duplicates": 0,
            },
        },
        Path("support/generated/support-issue-plan.json"),
        {
            "mode": "sync",
            "sync_summary": {
                "created": 0,
                "updated": 0,
                "closed": 0,
                "attached": 0,
                "unchanged": 3,
            },
        },
        Path("support/generated/support-issue-sync-summary.json"),
    )

    assert "| Overall | pass |" in text
    assert "| Support matrix | pass |" in text
    assert "| Issue plan | pass |" in text
    assert "| Issue sync | pass |" in text


def test_support_ci_summary_cli_writes_markdown(tmp_path):
    matrix_path = tmp_path / "support-matrix-check.json"
    evidence_path = tmp_path / "support-evidence-check.json"
    plan_path = tmp_path / "support-issue-plan.json"
    sync_path = tmp_path / "support-issue-sync-summary.json"
    output_path = tmp_path / "support-issue-ci-summary.md"
    matrix_path.write_text(json.dumps(matrix_check_report(ok=True)), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence_check_report()), encoding="utf-8")
    plan_path.write_text(json.dumps(issue_plan_report()), encoding="utf-8")
    sync_path.write_text(json.dumps(sync_summary_report()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/support_ci_summary.py",
            "--matrix-check",
            str(matrix_path),
            "--support-evidence",
            str(evidence_path),
            "--issue-plan",
            str(plan_path),
            "--sync-summary",
            str(sync_path),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Wrote" in result.stdout
    assert "## Support Evidence" in text
    assert "| Rows missing evidence | 2 |" in text
    assert "| Status | pass |" in text
    assert "| Sync updated | 2 |" in text
