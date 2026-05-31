#!/usr/bin/env python3
"""Render a concise Markdown summary for support automation CI reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_CHECK_PATH = ROOT / "support" / "generated" / "support-matrix-check.json"
DEFAULT_EVIDENCE_CHECK_PATH = (
    ROOT / "support" / "generated" / "support-evidence-check.json"
)
DEFAULT_ISSUE_PLAN_PATH = ROOT / "support" / "generated" / "support-issue-plan.json"
DEFAULT_SYNC_SUMMARY_PATH = (
    ROOT / "support" / "generated" / "support-issue-sync-summary.json"
)
MATRIX_CHECK_GENERATOR = "tools/support_matrix.py check"
EVIDENCE_CHECK_GENERATOR = "tools/support_matrix.py evidence"
ISSUE_SYNC_GENERATOR = "tools/sync_support_issues.py"
SUPPORTED_SCHEMA_VERSION = 1
MATRIX_CHECK_REQUIRED_FIELDS = ("schema_version", "generator", "ok", "summary")
EVIDENCE_CHECK_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "filters",
    "summary",
    "rows",
)
ISSUE_PLAN_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "mode",
    "desired",
    "existing",
    "support_matrix_check",
)
SYNC_SUMMARY_REQUIRED_FIELDS = (
    "schema_version",
    "generator",
    "mode",
    "sync_summary",
)
MATRIX_CHECK_SUMMARY_COUNTERS = (
    "artifact_count",
    "stale_count",
    "total_diff_line_count",
)
EVIDENCE_CHECK_SUMMARY_COUNTERS = (
    "row_count",
    "missing_evidence_count",
    "present_evidence_count",
)
ISSUE_DESIRED_COUNTERS = ("total", "parents", "backlog", "extracted")
ISSUE_EXISTING_COUNTERS = ("managed", "duplicates")
ISSUE_ACTION_COUNTERS = ("created", "updated", "closed", "attached", "unchanged")
ISSUE_CLOSURE_COUNTERS = (
    "total",
    "stale_parent",
    "stale_backlog",
    "stale_extracted",
    "duplicate_marker",
)
OPERATION_LEDGER_SUMMARY_LIMIT = 12
EVIDENCE_ROW_SUMMARY_LIMIT = 12


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_error(path: Path | None, error_type: str, message: str) -> dict[str, Any]:
    return {
        "load_error": {
            "path": display_path(path),
            "type": error_type,
            "message": message,
        }
    }


def type_label(expected_type: type | tuple[type, ...]) -> str:
    if isinstance(expected_type, tuple):
        return " or ".join(type_label(item) for item in expected_type)
    if expected_type is dict:
        return "object"
    if expected_type is list:
        return "list"
    if expected_type is bool:
        return "bool"
    if expected_type is int:
        return "int"
    if expected_type is str:
        return "str"
    return expected_type.__name__


def value_type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def value_matches_type(value: Any, expected_type: type | tuple[type, ...]) -> bool:
    if isinstance(expected_type, tuple):
        return any(value_matches_type(value, item) for item in expected_type)
    if expected_type is bool:
        return type(value) is bool
    if expected_type is int:
        return type(value) is int
    return isinstance(value, expected_type)


def invalid_field_error(
    path: Path | None,
    field: str,
    expected_type: type | tuple[type, ...] | str,
    value: Any,
) -> dict[str, Any]:
    expected = (
        expected_type if isinstance(expected_type, str) else type_label(expected_type)
    )
    return load_error(
        path,
        "InvalidReportField",
        f"{field} must be {expected}, got {value_type_label(value)}",
    )


def validate_field_type(
    report: dict[str, Any],
    path: Path | None,
    field: str,
    expected_type: type | tuple[type, ...],
) -> dict[str, Any] | None:
    if field not in report:
        return None
    value = report[field]
    if not value_matches_type(value, expected_type):
        return invalid_field_error(path, field, expected_type, value)
    return None


def validate_field_types(
    report: dict[str, Any],
    path: Path | None,
    fields: dict[str, type | tuple[type, ...]],
) -> dict[str, Any] | None:
    for field, expected_type in fields.items():
        error = validate_field_type(report, path, field, expected_type)
        if error is not None:
            return error
    return None


def validate_nested_field_types(
    report: dict[str, Any],
    path: Path | None,
    prefix: str,
    fields: dict[str, type | tuple[type, ...]],
) -> dict[str, Any] | None:
    for field, expected_type in fields.items():
        if field not in report:
            continue
        value = report[field]
        if not value_matches_type(value, expected_type):
            return invalid_field_error(
                path,
                f"{prefix}.{field}",
                expected_type,
                value,
            )
    return None


def validate_counter_map(
    report: dict[str, Any],
    path: Path | None,
    field: str,
    counters: tuple[str, ...],
    *,
    allow_none: bool = False,
) -> dict[str, Any] | None:
    value = report.get(field)
    if value is None and allow_none:
        return None
    return validate_counter_map_value(
        value,
        path,
        field,
        counters,
    )


def validate_counter_map_value(
    value: Any,
    path: Path | None,
    field: str,
    counters: tuple[str, ...],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return invalid_field_error(path, field, dict, value)

    missing = [counter for counter in counters if counter not in value]
    if missing:
        return load_error(
            path,
            "MissingReportFields",
            "{} missing required counters: {}".format(
                field,
                ", ".join(missing),
            ),
        )

    for counter in counters:
        counter_value = value[counter]
        if not value_matches_type(counter_value, int):
            return invalid_field_error(path, f"{field}.{counter}", int, counter_value)
    return None


def validate_optional_object(
    report: dict[str, Any],
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    value = report.get(field)
    if value is None:
        return None
    if not isinstance(value, dict):
        return invalid_field_error(path, field, dict, value)
    return None


def validate_optional_object_list(
    report: dict[str, Any],
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    value = report.get(field)
    if value is None:
        return None
    if not isinstance(value, list):
        return invalid_field_error(path, field, list, value)
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return invalid_field_error(
                path,
                f"{field}[{index}]",
                dict,
                item,
            )
    return None


def validate_object_list_item_fields(
    report: dict[str, Any],
    path: Path | None,
    field: str,
    fields: dict[str, type | tuple[type, ...]],
) -> dict[str, Any] | None:
    error = validate_optional_object_list(report, path, field)
    if error is not None:
        return error
    value = report.get(field)
    if value is None:
        return None
    for index, item in enumerate(value):
        error = validate_nested_field_types(item, path, f"{field}[{index}]", fields)
        if error is not None:
            return error
    return None


def validate_mode(
    report: dict[str, Any],
    path: Path | None,
    field: str = "mode",
) -> dict[str, Any] | None:
    error = validate_field_type(report, path, field, str)
    if error is not None:
        return error
    mode = report.get(field)
    if mode not in {"dry-run", "sync"}:
        return load_error(
            path,
            "InvalidReportField",
            f"{field} must be dry-run or sync, got {mode}",
        )
    return None


def validate_budget_contract(
    report: dict[str, Any],
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    value = report.get(field)
    if value is None:
        return None
    if not isinstance(value, dict):
        return invalid_field_error(path, field, dict, value)
    for bool_field in ("provided", "evaluated"):
        if bool_field in value and not value_matches_type(value[bool_field], bool):
            return invalid_field_error(
                path,
                f"{field}.{bool_field}",
                bool,
                value[bool_field],
            )
    ok = value.get("ok")
    if ok is not None and not value_matches_type(ok, bool):
        return invalid_field_error(path, f"{field}.ok", bool, ok)
    if value.get("provided") and value.get("evaluated") and ok is None:
        return invalid_field_error(path, f"{field}.ok", bool, ok)
    violations = value.get("violations")
    if violations is None:
        return None
    if not isinstance(violations, list):
        return invalid_field_error(
            path,
            f"{field}.violations",
            list,
            violations,
        )
    for index, violation in enumerate(violations):
        if not isinstance(violation, dict):
            return invalid_field_error(
                path,
                f"{field}.violations[{index}]",
                dict,
                violation,
            )
        if "actual" in violation and not value_matches_type(violation["actual"], int):
            return invalid_field_error(
                path,
                f"{field}.violations[{index}].actual",
                int,
                violation["actual"],
            )
        if "limit" in violation and not value_matches_type(violation["limit"], int):
            return invalid_field_error(
                path,
                f"{field}.violations[{index}].limit",
                int,
                violation["limit"],
            )
        if "action" in violation and not value_matches_type(violation["action"], str):
            return invalid_field_error(
                path,
                f"{field}.violations[{index}].action",
                str,
                violation["action"],
            )
        if "category" in violation and not value_matches_type(
            violation["category"], str
        ):
            return invalid_field_error(
                path,
                f"{field}.violations[{index}].category",
                str,
                violation["category"],
            )
    return None


def validate_string_list_value(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_field_error(path, field, list, value)
    for index, item in enumerate(value):
        if not value_matches_type(item, str):
            return invalid_field_error(path, f"{field}[{index}]", str, item)
    return None


def validate_sample_common_fields(
    sample: dict[str, Any],
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    error = validate_nested_field_types(
        sample,
        path,
        field,
        {
            "key": str,
            "title": str,
            "state": str,
            "reason": str,
            "category": str,
            "parent_key": str,
            "child_key": str,
            "number": int,
            "parent_number": int,
            "child_number": int,
            "reasons": list,
        },
    )
    if error is not None:
        return error
    if "reasons" in sample:
        return validate_string_list_value(sample["reasons"], path, f"{field}.reasons")
    return None


def validate_required_sample_fields(
    sample: dict[str, Any],
    path: Path | None,
    field: str,
    required_fields: tuple[str, ...],
) -> dict[str, Any] | None:
    missing = [required for required in required_fields if required not in sample]
    if missing:
        return load_error(
            path,
            "MissingReportFields",
            "{} missing required fields: {}".format(field, ", ".join(missing)),
        )
    return None


def validate_sample_list(
    value: Any,
    path: Path | None,
    field: str,
    required_fields: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_field_error(path, field, list, value)
    for index, sample in enumerate(value):
        sample_field = f"{field}[{index}]"
        if not isinstance(sample, dict):
            return invalid_field_error(path, sample_field, dict, sample)
        error = validate_required_sample_fields(
            sample,
            path,
            sample_field,
            required_fields,
        )
        if error is not None:
            return error
        error = validate_sample_common_fields(sample, path, sample_field)
        if error is not None:
            return error
    return None


def validate_planned_action_samples_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    samples = report.get("planned_action_samples")
    if samples is None:
        return None
    if not isinstance(samples, dict):
        return invalid_field_error(path, "planned_action_samples", dict, samples)
    if "sample_limit" not in samples:
        return load_error(
            path,
            "MissingReportFields",
            "planned_action_samples missing required fields: sample_limit",
        )
    if not value_matches_type(samples["sample_limit"], int):
        return invalid_field_error(
            path,
            "planned_action_samples.sample_limit",
            int,
            samples["sample_limit"],
        )

    sample_requirements = {
        "created": ("key", "title", "reason"),
        "updated": ("key", "number", "title", "state", "reasons"),
        "closed": ("key", "number", "title", "state", "reason"),
        "attached": ("parent_key", "child_key", "reason"),
        "preserved": ("key", "number", "title", "state", "reason"),
    }
    missing_actions = [
        action for action in sample_requirements if action not in samples
    ]
    if missing_actions:
        return load_error(
            path,
            "MissingReportFields",
            "planned_action_samples missing required fields: {}".format(
                ", ".join(missing_actions)
            ),
        )
    for action, required_fields in sample_requirements.items():
        error = validate_sample_list(
            samples[action],
            path,
            f"planned_action_samples.{action}",
            required_fields,
        )
        if error is not None:
            return error
    return None


def validate_audit_bucket(
    bucket: Any,
    path: Path | None,
    field: str,
    *,
    sample_limit: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(bucket, dict):
        return invalid_field_error(path, field, dict, bucket)
    for counter in ("total", "open", "closed"):
        if counter not in bucket:
            return load_error(
                path,
                "MissingReportFields",
                f"{field} missing required fields: {counter}",
            )
        if not value_matches_type(bucket[counter], int):
            return invalid_field_error(
                path,
                f"{field}.{counter}",
                int,
                bucket[counter],
            )
        if bucket[counter] < 0:
            return invalid_field_error(
                path,
                f"{field}.{counter}",
                "non-negative int",
                bucket[counter],
            )
    if bucket["open"] + bucket["closed"] != bucket["total"]:
        return load_error(
            path,
            "InvalidReportField",
            "{}.total must match open + closed: {} != {}".format(
                field,
                bucket["total"],
                bucket["open"] + bucket["closed"],
            ),
        )
    if "samples" not in bucket:
        return load_error(
            path,
            "MissingReportFields",
            f"{field} missing required fields: samples",
        )
    samples = bucket["samples"]
    if isinstance(samples, list):
        if len(samples) > bucket["total"]:
            return load_error(
                path,
                "InvalidReportField",
                "{}.samples must not exceed total: {} > {}".format(
                    field,
                    len(samples),
                    bucket["total"],
                ),
            )
        if sample_limit is not None and len(samples) > sample_limit:
            return load_error(
                path,
                "InvalidReportField",
                "{}.samples must not exceed sample_limit: {} > {}".format(
                    field,
                    len(samples),
                    sample_limit,
                ),
            )
    return validate_sample_list(
        samples,
        path,
        f"{field}.samples",
        ("key", "number", "title", "state", "reason"),
    )


def validate_managed_issue_audit_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    audit = report.get("managed_issue_audit")
    if audit is None:
        return None
    if not isinstance(audit, dict):
        return invalid_field_error(path, "managed_issue_audit", dict, audit)
    if "sample_limit" not in audit:
        return load_error(
            path,
            "MissingReportFields",
            "managed_issue_audit missing required fields: sample_limit",
        )
    if not value_matches_type(audit["sample_limit"], int):
        return invalid_field_error(
            path,
            "managed_issue_audit.sample_limit",
            int,
            audit["sample_limit"],
        )
    if audit["sample_limit"] < 0:
        return invalid_field_error(
            path,
            "managed_issue_audit.sample_limit",
            "non-negative int",
            audit["sample_limit"],
        )

    for bucket in (
        "stale",
        "duplicates",
        "preserved_extracted",
        "ignored_unknown",
    ):
        if bucket not in audit:
            return load_error(
                path,
                "MissingReportFields",
                f"managed_issue_audit missing required fields: {bucket}",
            )
        error = validate_audit_bucket(
            audit[bucket],
            path,
            f"managed_issue_audit.{bucket}",
            sample_limit=audit["sample_limit"],
        )
        if error is not None:
            return error
    return None


def validate_matrix_check_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    error = validate_field_types(
        report,
        path,
        {
            "schema_version": int,
            "generator": str,
            "ok": bool,
            "summary": dict,
        },
    )
    if error is not None:
        return error
    error = validate_matrix_check_summary_value(report["summary"], path, "summary")
    if error is not None:
        return error
    error = validate_optional_object_list(report, path, "artifacts")
    if error is not None:
        return error
    for index, artifact in enumerate(report.get("artifacts") or []):
        error = validate_nested_field_types(
            artifact,
            path,
            f"artifacts[{index}]",
            {
                "path": str,
                "exists": bool,
                "stale": bool,
                "diff_line_count": int,
            },
        )
        if error is not None:
            return error
        diff = artifact.get("diff")
        if diff is not None and not isinstance(diff, list):
            return invalid_field_error(path, f"artifacts[{index}].diff", list, diff)
    return None


def validate_matrix_check_summary_value(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    error = validate_counter_map_value(
        value,
        path,
        field,
        MATRIX_CHECK_SUMMARY_COUNTERS,
    )
    if error is not None:
        return error

    stale_artifacts = value.get("stale_artifacts")
    if stale_artifacts is None:
        return load_error(
            path,
            "MissingReportFields",
            f"{field} missing required fields: stale_artifacts",
        )
    error = validate_string_list_value(
        stale_artifacts, path, f"{field}.stale_artifacts"
    )
    if error is not None:
        return error
    if value["stale_count"] != len(stale_artifacts):
        return load_error(
            path,
            "InvalidReportField",
            "{}.stale_count must match stale_artifacts length: {} != {}".format(
                field,
                value["stale_count"],
                len(stale_artifacts),
            ),
        )
    return None


def validate_evidence_check_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    error = validate_field_types(
        report,
        path,
        {
            "schema_version": int,
            "generator": str,
            "filters": dict,
            "summary": dict,
            "rows": list,
        },
    )
    if error is not None:
        return error
    filters = report["filters"]
    error = validate_nested_field_types(
        filters,
        path,
        "filters",
        {
            "backend_ids": list,
            "categories": list,
            "statuses": list,
            "evidence": str,
        },
    )
    if error is not None:
        return error
    for filter_field in ("backend_ids", "categories", "statuses"):
        if filter_field not in filters:
            return load_error(
                path,
                "MissingReportFields",
                f"filters missing required fields: {filter_field}",
            )
        error = validate_string_list_value(
            filters[filter_field], path, f"filters.{filter_field}"
        )
        if error is not None:
            return error
    if "evidence" not in filters:
        return load_error(
            path,
            "MissingReportFields",
            "filters missing required fields: evidence",
        )
    if filters["evidence"] not in {"any", "present", "missing"}:
        return load_error(
            path,
            "InvalidReportField",
            "filters.evidence must be any, present, or missing, got {}".format(
                filters["evidence"]
            ),
        )
    error = validate_counter_map(
        report,
        path,
        "summary",
        EVIDENCE_CHECK_SUMMARY_COUNTERS,
    )
    if error is not None:
        return error

    if report["summary"]["row_count"] != len(report["rows"]):
        return load_error(
            path,
            "InvalidReportField",
            "summary.row_count must match rows length: {} != {}".format(
                report["summary"]["row_count"],
                len(report["rows"]),
            ),
        )

    by_backend = report["summary"].get("by_backend", {})
    if not isinstance(by_backend, dict):
        return invalid_field_error(path, "summary.by_backend", dict, by_backend)
    for backend_id, counts in by_backend.items():
        if not isinstance(counts, dict):
            return invalid_field_error(
                path, f"summary.by_backend.{backend_id}", dict, counts
            )
        for counter in ("rows", "present", "missing"):
            if counter not in counts:
                return load_error(
                    path,
                    "MissingReportFields",
                    f"summary.by_backend.{backend_id} missing required counter: {counter}",
                )
            if not value_matches_type(counts[counter], int):
                return invalid_field_error(
                    path,
                    f"summary.by_backend.{backend_id}.{counter}",
                    int,
                    counts[counter],
                )

    by_status = report["summary"].get("by_status")
    if by_status is None:
        return load_error(
            path,
            "MissingReportFields",
            "summary missing required fields: by_status",
        )
    if not isinstance(by_status, dict):
        return invalid_field_error(path, "summary.by_status", dict, by_status)
    for status, count in by_status.items():
        if not isinstance(status, str):
            return invalid_field_error(path, "summary.by_status key", str, status)
        if not value_matches_type(count, int):
            return invalid_field_error(
                path,
                f"summary.by_status.{status}",
                int,
                count,
            )

    actual_missing = 0
    actual_present = 0
    actual_by_backend: dict[str, dict[str, int]] = {}
    actual_by_status: dict[str, int] = {}
    row_required_fields = (
        "backend",
        "backend_id",
        "category",
        "feature",
        "feature_id",
        "status",
        "evidence_count",
        "notes",
        "evidence",
    )
    for index, row in enumerate(report["rows"]):
        if not isinstance(row, dict):
            return invalid_field_error(path, f"rows[{index}]", dict, row)
        missing_row_fields = [
            field for field in row_required_fields if field not in row
        ]
        if missing_row_fields:
            return load_error(
                path,
                "MissingReportFields",
                "rows[{}] missing required fields: {}".format(
                    index,
                    ", ".join(missing_row_fields),
                ),
            )
        error = validate_nested_field_types(
            row,
            path,
            f"rows[{index}]",
            {
                "backend": str,
                "backend_id": str,
                "category": str,
                "feature": str,
                "feature_id": str,
                "status": str,
                "evidence_count": int,
                "notes": str,
                "evidence": list,
            },
        )
        if error is not None:
            return error
        evidence = row["evidence"]
        error = validate_string_list_value(evidence, path, f"rows[{index}].evidence")
        if error is not None:
            return error
        if row["evidence_count"] != len(evidence):
            return load_error(
                path,
                "InvalidReportField",
                "rows[{}].evidence_count must match evidence length: {} != {}".format(
                    index,
                    row["evidence_count"],
                    len(evidence),
                ),
            )

        backend_counts = actual_by_backend.setdefault(
            row["backend_id"], {"rows": 0, "present": 0, "missing": 0}
        )
        backend_counts["rows"] += 1
        actual_by_status[row["status"]] = actual_by_status.get(row["status"], 0) + 1
        if row["evidence_count"]:
            actual_present += 1
            backend_counts["present"] += 1
        else:
            actual_missing += 1
            backend_counts["missing"] += 1

    if report["summary"]["missing_evidence_count"] != actual_missing:
        return load_error(
            path,
            "InvalidReportField",
            "summary.missing_evidence_count must match rows: {} != {}".format(
                report["summary"]["missing_evidence_count"],
                actual_missing,
            ),
        )
    if report["summary"]["present_evidence_count"] != actual_present:
        return load_error(
            path,
            "InvalidReportField",
            "summary.present_evidence_count must match rows: {} != {}".format(
                report["summary"]["present_evidence_count"],
                actual_present,
            ),
        )
    if by_backend != actual_by_backend:
        return load_error(
            path,
            "InvalidReportField",
            "summary.by_backend must match rows",
        )
    actual_by_status_complete = {
        status: actual_by_status.get(status, 0) for status in by_status
    }
    actual_by_status_complete.update(
        {
            status: count
            for status, count in actual_by_status.items()
            if status not in actual_by_status_complete
        }
    )
    if by_status != actual_by_status_complete:
        return load_error(
            path,
            "InvalidReportField",
            "summary.by_status must match rows",
        )
    return None


def validate_report_error_summary(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return invalid_field_error(path, field, dict, value)
    error = validate_nested_field_types(
        value,
        path,
        field,
        {
            "path": (str, type(None)),
            "type": str,
            "message": str,
            "method": str,
            "status": int,
            "body": str,
        },
    )
    if error is not None:
        return error
    missing = [required for required in ("type", "message") if required not in value]
    if missing:
        return load_error(
            path,
            "MissingReportFields",
            "{} missing required fields: {}".format(field, ", ".join(missing)),
        )
    return None


def validate_operation_context_value(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return invalid_field_error(path, field, dict, value)
    for key, item in value.items():
        if not isinstance(key, str):
            return invalid_field_error(path, f"{field} key", str, key)
        if not isinstance(item, (str, int, bool)) and item is not None:
            return invalid_field_error(
                path,
                f"{field}.{key}",
                "str, int, bool, or null",
                item,
            )
    return None


def validate_input_failures_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    error = validate_optional_object_list(report, path, "input_failures")
    if error is not None:
        return error
    for index, failure in enumerate(report.get("input_failures") or []):
        field = f"input_failures[{index}]"
        missing = [
            required
            for required in ("input", "path", "error")
            if required not in failure
        ]
        if missing:
            return load_error(
                path,
                "MissingReportFields",
                "{} missing required fields: {}".format(field, ", ".join(missing)),
            )
        error = validate_nested_field_types(
            failure,
            path,
            field,
            {
                "input": str,
                "path": (str, type(None)),
                "error": dict,
            },
        )
        if error is not None:
            return error
        error = validate_report_error_summary(failure["error"], path, f"{field}.error")
        if error is not None:
            return error
    return None


def validate_preflight_failure_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    failure = report.get("preflight_failure")
    if failure is None:
        return None
    if not isinstance(failure, dict):
        return invalid_field_error(path, "preflight_failure", dict, failure)
    missing = [
        required
        for required in ("phase", "operation", "error")
        if required not in failure
    ]
    if missing:
        return load_error(
            path,
            "MissingReportFields",
            "preflight_failure missing required fields: {}".format(", ".join(missing)),
        )
    error = validate_nested_field_types(
        failure,
        path,
        "preflight_failure",
        {
            "phase": str,
            "operation": dict,
            "error": dict,
        },
    )
    if error is not None:
        return error
    error = validate_operation_context_value(
        failure["operation"], path, "preflight_failure.operation"
    )
    if error is not None:
        return error
    return validate_report_error_summary(
        failure["error"], path, "preflight_failure.error"
    )


def validate_embedded_stale_artifact_details(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_field_error(path, field, list, value)
    for index, artifact in enumerate(value):
        artifact_field = f"{field}[{index}]"
        if not isinstance(artifact, dict):
            return invalid_field_error(path, artifact_field, dict, artifact)
        missing = [
            required
            for required in ("path", "diff_line_count")
            if required not in artifact
        ]
        if missing:
            return load_error(
                path,
                "MissingReportFields",
                "{} missing required fields: {}".format(
                    artifact_field,
                    ", ".join(missing),
                ),
            )
        error = validate_nested_field_types(
            artifact,
            path,
            artifact_field,
            {
                "path": str,
                "diff_line_count": int,
                "actual_sha256": (str, type(None)),
                "expected_sha256": (str, type(None)),
            },
        )
        if error is not None:
            return error
    return None


def validate_embedded_matrix_check_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    matrix_check = report["support_matrix_check"]
    missing = [
        required for required in ("provided", "path") if required not in matrix_check
    ]
    if missing:
        return load_error(
            path,
            "MissingReportFields",
            "support_matrix_check missing required fields: {}".format(
                ", ".join(missing)
            ),
        )

    error = validate_nested_field_types(
        matrix_check,
        path,
        "support_matrix_check",
        {
            "provided": bool,
            "path": (str, type(None)),
            "ok": bool,
            "summary": dict,
            "stale_artifacts": list,
            "load_error": dict,
        },
    )
    if error is not None:
        return error

    if not matrix_check["provided"]:
        return None

    if "ok" not in matrix_check:
        return load_error(
            path,
            "MissingReportFields",
            "support_matrix_check missing required fields: ok",
        )

    load_error_summary = matrix_check.get("load_error")
    if load_error_summary is not None:
        if matrix_check["ok"] is not False:
            return load_error(
                path,
                "InvalidReportField",
                "support_matrix_check.ok must be false when load_error is present",
            )
        error = validate_report_error_summary(
            load_error_summary,
            path,
            "support_matrix_check.load_error",
        )
        if error is not None:
            return error
        if "stale_artifacts" in matrix_check:
            return validate_embedded_stale_artifact_details(
                matrix_check["stale_artifacts"],
                path,
                "support_matrix_check.stale_artifacts",
            )
        return None

    missing = [
        required
        for required in ("summary", "stale_artifacts")
        if required not in matrix_check
    ]
    if missing:
        return load_error(
            path,
            "MissingReportFields",
            "support_matrix_check missing required fields: {}".format(
                ", ".join(missing)
            ),
        )

    error = validate_matrix_check_summary_value(
        matrix_check["summary"],
        path,
        "support_matrix_check.summary",
    )
    if error is not None:
        return error
    error = validate_embedded_stale_artifact_details(
        matrix_check["stale_artifacts"],
        path,
        "support_matrix_check.stale_artifacts",
    )
    if error is not None:
        return error

    summary_paths = matrix_check["summary"]["stale_artifacts"]
    detail_paths = [artifact["path"] for artifact in matrix_check["stale_artifacts"]]
    if detail_paths != summary_paths:
        return load_error(
            path,
            "InvalidReportField",
            (
                "support_matrix_check.stale_artifacts paths must match "
                "support_matrix_check.summary.stale_artifacts"
            ),
        )
    return None


def validate_issue_plan_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    error = validate_field_types(
        report,
        path,
        {
            "schema_version": int,
            "generator": str,
            "mode": str,
            "desired": dict,
            "existing": dict,
            "close_extracted_issues": bool,
            "close_pytest_failure_issues": bool,
            "support_matrix_check": dict,
        },
    )
    if error is not None:
        return error
    for validator in (
        lambda: validate_mode(report, path),
        lambda: validate_counter_map(report, path, "desired", ISSUE_DESIRED_COUNTERS),
        lambda: validate_counter_map(
            report,
            path,
            "planned_actions",
            ISSUE_ACTION_COUNTERS,
            allow_none=True,
        ),
        lambda: validate_counter_map(
            report,
            path,
            "planned_closures",
            ISSUE_CLOSURE_COUNTERS,
            allow_none=True,
        ),
        lambda: validate_planned_action_samples_contract(report, path),
        lambda: validate_managed_issue_audit_contract(report, path),
        lambda: validate_input_failures_contract(report, path),
        lambda: validate_preflight_failure_contract(report, path),
        lambda: validate_budget_contract(report, path, "planned_action_budget"),
        lambda: validate_budget_contract(report, path, "planned_closure_budget"),
        lambda: validate_embedded_matrix_check_contract(report, path),
        lambda: validate_operation_reconciliation_contract(report, path),
        lambda: validate_workflow_source_contract(report, path),
    ):
        error = validator()
        if error is not None:
            return error

    existing = report["existing"]
    if "inspected" not in existing:
        return load_error(
            path,
            "MissingReportFields",
            "existing missing required fields: inspected",
        )
    if not value_matches_type(existing["inspected"], bool):
        return invalid_field_error(
            path, "existing.inspected", bool, existing["inspected"]
        )
    for counter in ISSUE_EXISTING_COUNTERS:
        if counter not in existing:
            return load_error(
                path,
                "MissingReportFields",
                f"existing missing required counters: {counter}",
            )
        if not value_matches_type(existing[counter], int):
            return invalid_field_error(
                path,
                f"existing.{counter}",
                int,
                existing[counter],
            )

    return None


def validate_operation_ledger_value(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return invalid_field_error(path, field, list, value)
    for index, entry in enumerate(value):
        entry_field = f"{field}[{index}]"
        if not isinstance(entry, dict):
            return invalid_field_error(path, entry_field, dict, entry)
        if "action" not in entry:
            return load_error(
                path,
                "MissingReportFields",
                f"{entry_field} missing required fields: action",
            )
        error = validate_nested_field_types(
            entry,
            path,
            entry_field,
            {
                "action": str,
                "key": str,
                "number": int,
                "title": str,
                "state": str,
                "parent_key": str,
                "parent_number": int,
                "child_key": str,
                "child_number": int,
                "reason": str,
                "reasons": list,
            },
        )
        if error is not None:
            return error
        action = entry.get("action")
        action_requirements = {
            "created": ("key", "number", "title", "state", "reason"),
            "updated": ("key", "number", "title", "state", "reason", "reasons"),
            "closed": ("key", "number", "title", "state", "reason"),
            "attached": (
                "parent_key",
                "parent_number",
                "child_key",
                "child_number",
                "reason",
            ),
        }
        required_fields = action_requirements.get(action)
        if required_fields is not None:
            missing = [
                required for required in required_fields if required not in entry
            ]
            if missing:
                return load_error(
                    path,
                    "MissingReportFields",
                    "{} missing required fields: {}".format(
                        entry_field,
                        ", ".join(missing),
                    ),
                )
        reasons = entry.get("reasons")
        if reasons is None:
            continue
        for reason_index, reason in enumerate(reasons):
            if not value_matches_type(reason, str):
                return invalid_field_error(
                    path,
                    f"{entry_field}.reasons[{reason_index}]",
                    str,
                    reason,
                )
    return None


def validate_operation_ledger_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    value = report.get("operation_ledger")
    if value is None:
        return None
    return validate_operation_ledger_value(value, path, "operation_ledger")


def operation_ledger_action_counts(
    entries: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {action: 0 for action in ISSUE_ACTION_COUNTERS[:-1]}
    for entry in entries:
        action = entry.get("action")
        if action in counts:
            counts[action] += 1
    return counts


def operation_ledger_action_reason_counts(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        action: {} for action in ISSUE_ACTION_COUNTERS[:-1]
    }
    for entry in entries:
        action = entry.get("action")
        if action not in counts:
            continue
        reason = entry.get("reason") or "unspecified"
        counts[action][reason] = counts[action].get(reason, 0) + 1
    return counts


def operation_ledger_closure_counts(
    entries: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {counter: 0 for counter in ISSUE_CLOSURE_COUNTERS}
    for entry in entries:
        if entry.get("action") != "closed":
            continue
        key = str(entry.get("key", ""))
        reason = entry.get("reason")
        if reason == "duplicate_managed_marker":
            category = "duplicate_marker"
        elif key.startswith("parent:"):
            category = "stale_parent"
        elif key.startswith("backlog:"):
            category = "stale_backlog"
        elif key.startswith("extracted:"):
            category = "stale_extracted"
        else:
            continue
        counts[category] += 1
        counts["total"] += 1
    return counts


def counter_difference_rows(
    planned: dict[str, int],
    actual: dict[str, int],
    key_field: str,
    *,
    comparison: str,
) -> list[dict[str, int | str]]:
    rows = []
    for key in sorted(actual):
        actual_count = actual[key]
        planned_count = planned.get(key, 0)
        if comparison == "overrun" and actual_count <= planned_count:
            continue
        if comparison == "shortfall" and actual_count >= planned_count:
            continue
        rows.append(
            {
                key_field: key,
                "actual": actual_count,
                "planned": planned_count,
            }
        )
    return rows


def counter_map_mismatch_error(
    path: Path | None,
    field: str,
    actual: dict[str, int],
    expected: dict[str, int],
) -> dict[str, Any]:
    return load_error(
        path,
        "InvalidReportField",
        "{} must match operation ledger: {} != {}".format(
            field,
            actual,
            expected,
        ),
    )


def validate_sync_failure_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    failure = report.get("sync_failure")
    if failure is None:
        return None
    if not isinstance(failure, dict):
        return invalid_field_error(path, "sync_failure", dict, failure)

    missing_fields = [
        field
        for field in (
            "phase",
            "operation",
            "partial_summary",
            "operation_ledger",
            "error",
            "recovery",
        )
        if field not in failure
    ]
    if missing_fields:
        return load_error(
            path,
            "MissingReportFields",
            "sync_failure missing required fields: {}".format(
                ", ".join(missing_fields)
            ),
        )

    error = validate_nested_field_types(
        failure,
        path,
        "sync_failure",
        {
            "phase": str,
            "operation": dict,
            "partial_summary": dict,
            "operation_ledger": list,
            "error": dict,
            "recovery": dict,
        },
    )
    if error is not None:
        return error

    error = validate_counter_map_value(
        failure["partial_summary"],
        path,
        "sync_failure.partial_summary",
        ISSUE_ACTION_COUNTERS,
    )
    if error is not None:
        return error

    error = validate_operation_ledger_value(
        failure["operation_ledger"],
        path,
        "sync_failure.operation_ledger",
    )
    if error is not None:
        return error

    expected_partial_actions = operation_ledger_action_counts(
        failure["operation_ledger"]
    )
    for action, expected_count in expected_partial_actions.items():
        if failure["partial_summary"][action] != expected_count:
            return load_error(
                path,
                "InvalidReportField",
                "sync_failure.partial_summary.{} must match operation ledger: {} != {}".format(
                    action,
                    failure["partial_summary"][action],
                    expected_count,
                ),
            )

    error_summary = failure["error"]
    error = validate_nested_field_types(
        error_summary,
        path,
        "sync_failure.error",
        {
            "type": str,
            "message": str,
            "method": str,
            "path": str,
            "status": int,
            "body": str,
        },
    )
    if error is not None:
        return error
    missing_error_fields = [
        field for field in ("type", "message") if field not in error_summary
    ]
    if missing_error_fields:
        return load_error(
            path,
            "MissingReportFields",
            "sync_failure.error missing required fields: {}".format(
                ", ".join(missing_error_fields)
            ),
        )

    recovery = failure["recovery"]
    error = validate_nested_field_types(
        recovery,
        path,
        "sync_failure.recovery",
        {
            "rerun_safe": bool,
            "strategy": str,
        },
    )
    if error is not None:
        return error
    missing_recovery_fields = [
        field for field in ("rerun_safe", "strategy") if field not in recovery
    ]
    if missing_recovery_fields:
        return load_error(
            path,
            "MissingReportFields",
            "sync_failure.recovery missing required fields: {}".format(
                ", ".join(missing_recovery_fields)
            ),
        )
    if not recovery["strategy"].strip():
        return load_error(
            path,
            "InvalidReportField",
            "sync_failure.recovery.strategy must not be empty",
        )
    return None


def validate_reconciliation_differences(
    reconciliation: dict[str, Any],
    path: Path | None,
    field: str,
    key_field: str,
) -> dict[str, Any] | None:
    error = validate_object_list_item_fields(
        reconciliation,
        path,
        field,
        {
            key_field: str,
            "actual": int,
            "planned": int,
        },
    )
    if error is not None:
        return error
    for index, item in enumerate(reconciliation.get(field) or []):
        missing = [
            required
            for required in (key_field, "actual", "planned")
            if required not in item
        ]
        if missing:
            return load_error(
                path,
                "MissingReportFields",
                "{}[{}] missing required fields: {}".format(
                    field,
                    index,
                    ", ".join(missing),
                ),
            )
    return None


def validate_action_reason_counts(
    value: Any,
    path: Path | None,
    field: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return invalid_field_error(path, field, dict, value)
    for action, reasons in value.items():
        if not value_matches_type(action, str):
            return invalid_field_error(path, f"{field} key", str, action)
        if action not in ISSUE_ACTION_COUNTERS[:-1]:
            return load_error(
                path,
                "InvalidReportField",
                f"{field} contains unknown action: {action}",
            )
        if not isinstance(reasons, dict):
            return invalid_field_error(path, f"{field}.{action}", dict, reasons)
        for reason, count in reasons.items():
            if not value_matches_type(reason, str):
                return invalid_field_error(
                    path,
                    f"{field}.{action} key",
                    str,
                    reason,
                )
            if not value_matches_type(count, int):
                return invalid_field_error(
                    path,
                    f"{field}.{action}.{reason}",
                    int,
                    count,
                )
    return None


def validate_operation_reconciliation_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    reconciliation = report.get("operation_reconciliation")
    if reconciliation is None:
        return None
    if not isinstance(reconciliation, dict):
        return invalid_field_error(
            path,
            "operation_reconciliation",
            dict,
            reconciliation,
        )
    missing_fields = [
        field
        for field in (
            "evaluated",
            "planned_actions",
            "actual_actions",
            "actual_action_reasons",
            "action_overruns",
            "action_shortfalls",
            "planned_closures",
            "actual_closures",
            "closure_overruns",
            "closure_shortfalls",
        )
        if field not in reconciliation
    ]
    if missing_fields:
        return load_error(
            path,
            "MissingReportFields",
            "operation_reconciliation missing required fields: {}".format(
                ", ".join(missing_fields)
            ),
        )
    if reconciliation.get("evaluated") and "ok" not in reconciliation:
        return load_error(
            path,
            "MissingReportFields",
            "operation_reconciliation missing required fields: ok",
        )

    error = validate_nested_field_types(
        reconciliation,
        path,
        "operation_reconciliation",
        {
            "evaluated": bool,
            "action_overruns": list,
            "action_shortfalls": list,
            "closure_overruns": list,
            "closure_shortfalls": list,
        },
    )
    if error is not None:
        return error
    if reconciliation.get("ok") is not None and not value_matches_type(
        reconciliation["ok"], bool
    ):
        return invalid_field_error(
            path,
            "operation_reconciliation.ok",
            bool,
            reconciliation["ok"],
        )

    for counter_field, counters in (
        ("planned_actions", ISSUE_ACTION_COUNTERS),
        ("actual_actions", ISSUE_ACTION_COUNTERS[:-1]),
        ("planned_closures", ISSUE_CLOSURE_COUNTERS),
        ("actual_closures", ISSUE_CLOSURE_COUNTERS),
    ):
        if reconciliation.get(counter_field) is None:
            continue
        error = validate_counter_map_value(
            reconciliation[counter_field],
            path,
            f"operation_reconciliation.{counter_field}",
            counters,
        )
        if error is not None:
            return error

    error = validate_action_reason_counts(
        reconciliation.get("actual_action_reasons"),
        path,
        "operation_reconciliation.actual_action_reasons",
    )
    if error is not None:
        return error

    for field in ("action_overruns", "action_shortfalls"):
        error = validate_reconciliation_differences(
            reconciliation,
            path,
            field,
            "action",
        )
        if error is not None:
            return error
    for field in ("closure_overruns", "closure_shortfalls"):
        error = validate_reconciliation_differences(
            reconciliation,
            path,
            field,
            "category",
        )
        if error is not None:
            return error

    operation_ledger = report.get("operation_ledger")
    if isinstance(operation_ledger, list):
        expected_actual_actions = operation_ledger_action_counts(operation_ledger)
        actual_actions = reconciliation.get("actual_actions")
        if actual_actions is not None and actual_actions != expected_actual_actions:
            return counter_map_mismatch_error(
                path,
                "operation_reconciliation.actual_actions",
                actual_actions,
                expected_actual_actions,
            )

        expected_action_reasons = operation_ledger_action_reason_counts(
            operation_ledger
        )
        actual_action_reasons = reconciliation.get("actual_action_reasons")
        if (
            actual_action_reasons is not None
            and actual_action_reasons != expected_action_reasons
        ):
            return load_error(
                path,
                "InvalidReportField",
                "operation_reconciliation.actual_action_reasons must match operation ledger",
            )

        expected_actual_closures = operation_ledger_closure_counts(operation_ledger)
        actual_closures = reconciliation.get("actual_closures")
        if actual_closures is not None and actual_closures != expected_actual_closures:
            return counter_map_mismatch_error(
                path,
                "operation_reconciliation.actual_closures",
                actual_closures,
                expected_actual_closures,
            )

    planned_actions = reconciliation.get("planned_actions")
    actual_actions = reconciliation.get("actual_actions")
    if planned_actions is not None and actual_actions is not None:
        expected_action_overruns = counter_difference_rows(
            planned_actions,
            actual_actions,
            "action",
            comparison="overrun",
        )
        if reconciliation["action_overruns"] != expected_action_overruns:
            return load_error(
                path,
                "InvalidReportField",
                "operation_reconciliation.action_overruns must match counters",
            )
        expected_action_shortfalls = counter_difference_rows(
            planned_actions,
            actual_actions,
            "action",
            comparison="shortfall",
        )
        if reconciliation["action_shortfalls"] != expected_action_shortfalls:
            return load_error(
                path,
                "InvalidReportField",
                "operation_reconciliation.action_shortfalls must match counters",
            )

    planned_closures = reconciliation.get("planned_closures")
    actual_closures = reconciliation.get("actual_closures")
    if planned_closures is not None and actual_closures is not None:
        expected_closure_overruns = counter_difference_rows(
            planned_closures,
            actual_closures,
            "category",
            comparison="overrun",
        )
        if reconciliation["closure_overruns"] != expected_closure_overruns:
            return load_error(
                path,
                "InvalidReportField",
                "operation_reconciliation.closure_overruns must match counters",
            )
        expected_closure_shortfalls = counter_difference_rows(
            planned_closures,
            actual_closures,
            "category",
            comparison="shortfall",
        )
        if reconciliation["closure_shortfalls"] != expected_closure_shortfalls:
            return load_error(
                path,
                "InvalidReportField",
                "operation_reconciliation.closure_shortfalls must match counters",
            )

    if reconciliation.get("evaluated"):
        expected_ok = not (
            reconciliation["action_overruns"]
            or reconciliation["action_shortfalls"]
            or reconciliation["closure_overruns"]
            or reconciliation["closure_shortfalls"]
        )
        if reconciliation.get("ok") != expected_ok:
            return load_error(
                path,
                "InvalidReportField",
                "operation_reconciliation.ok must match reconciliation differences",
            )
    elif reconciliation.get("ok") is not None:
        return load_error(
            path,
            "InvalidReportField",
            "operation_reconciliation.ok must be null when not evaluated",
        )
    return None


def validate_workflow_source_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    source = report.get("workflow_source")
    if source is None:
        return None
    if not isinstance(source, dict):
        return invalid_field_error(path, "workflow_source", dict, source)
    allowed_fields = {"event", "workflow", "run_id", "conclusion", "head_sha"}
    unknown_fields = sorted(set(source) - allowed_fields)
    if unknown_fields:
        return load_error(
            path,
            "InvalidReportField",
            "workflow_source contains unknown fields: {}".format(
                ", ".join(unknown_fields)
            ),
        )
    for field in sorted(allowed_fields):
        if field not in source:
            continue
        value = source[field]
        if not value_matches_type(value, str):
            return invalid_field_error(path, f"workflow_source.{field}", str, value)
        if not value.strip():
            return load_error(
                path,
                "InvalidReportField",
                f"workflow_source.{field} must not be empty",
            )
    return None


def validate_sync_summary_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    error = validate_field_types(
        report,
        path,
        {
            "schema_version": int,
            "generator": str,
            "mode": str,
            "close_extracted_issues": bool,
            "close_pytest_failure_issues": bool,
            "sync_summary": dict,
        },
    )
    if error is not None:
        return error
    for validator in (
        lambda: validate_mode(report, path),
        lambda: validate_counter_map(
            report, path, "sync_summary", ISSUE_ACTION_COUNTERS
        ),
        lambda: validate_operation_ledger_contract(report, path),
        lambda: validate_operation_reconciliation_contract(report, path),
        lambda: validate_sync_failure_contract(report, path),
        lambda: validate_workflow_source_contract(report, path),
    ):
        error = validator()
        if error is not None:
            return error
    operation_ledger = report.get("operation_ledger")
    if isinstance(operation_ledger, list):
        expected_actual_actions = operation_ledger_action_counts(operation_ledger)
        sync_summary = report["sync_summary"]
        for action, expected_count in expected_actual_actions.items():
            if sync_summary[action] != expected_count:
                return load_error(
                    path,
                    "InvalidReportField",
                    "sync_summary.{} must match operation ledger: {} != {}".format(
                        action,
                        sync_summary[action],
                        expected_count,
                    ),
                )
    return None


def validate_report_schema(
    report: dict[str, Any],
    path: Path | None,
    *,
    expected_generator: str | None = None,
    required_fields: tuple[str, ...] = (),
    schema_version: int = SUPPORTED_SCHEMA_VERSION,
) -> dict[str, Any] | None:
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        return load_error(
            path,
            "MissingReportFields",
            "missing required fields: {}".format(", ".join(missing_fields)),
        )

    actual_schema_version = report.get("schema_version")
    if actual_schema_version != schema_version:
        return load_error(
            path,
            "UnsupportedSchemaVersion",
            "expected schema_version {}, got {}".format(
                schema_version, actual_schema_version
            ),
        )

    actual_generator = report.get("generator")
    if expected_generator is not None and actual_generator != expected_generator:
        return load_error(
            path,
            "UnexpectedReportGenerator",
            "expected generator {}, got {}".format(
                expected_generator,
                actual_generator,
            ),
        )

    return None


def load_optional_json(
    path: Path | None,
    *,
    expected_generator: str | None = None,
    required_fields: tuple[str, ...] = (),
    schema_version: int = SUPPORTED_SCHEMA_VERSION,
    contract_validator: (
        Callable[[dict[str, Any], Path | None], dict[str, Any] | None] | None
    ) = None,
) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return load_error(path, type(exc).__name__, str(exc))
    if not isinstance(data, dict):
        return load_error(
            path,
            "InvalidReportType",
            f"expected JSON object, got {type(data).__name__}",
        )
    schema_error = validate_report_schema(
        data,
        path,
        expected_generator=expected_generator,
        required_fields=required_fields,
        schema_version=schema_version,
    )
    if schema_error is not None:
        return schema_error
    if contract_validator is not None:
        contract_error = contract_validator(data, path)
        if contract_error is not None:
            return contract_error
    return data


def resolve_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return ROOT / path


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("\n", " ").replace("|", r"\|").strip()

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def status_label(ok: bool | None) -> str:
    if ok is True:
        return "pass"
    if ok is False:
        return "fail"
    return "unknown"


def count_rows(counts: dict[str, Any] | None) -> list[list[Any]]:
    return [[key, value] for key, value in sorted((counts or {}).items())]


def sample_label(sample: dict[str, Any]) -> str:
    key = sample.get("key") or sample.get("child_key") or "unknown"
    number = sample.get("number")
    if number is not None:
        return f"`{key}` (#{number})"
    return f"`{key}`"


def sample_details(sample: dict[str, Any]) -> str:
    details = []
    if sample.get("reason"):
        details.append("reason={}".format(sample["reason"]))
    if sample.get("reasons"):
        details.append("reasons={}".format(",".join(sample["reasons"])))
    if sample.get("parent_key"):
        details.append("parent={}".format(sample["parent_key"]))
    return ", ".join(details)


def render_action_samples(samples: dict[str, Any] | None) -> list[str]:
    if not samples:
        return []
    lines = ["", "Planned action samples:"]
    for action in ("created", "updated", "closed", "attached", "preserved"):
        for sample in samples.get(action, []):
            details = sample_details(sample)
            if details:
                lines.append(f"- {action}: {sample_label(sample)} ({details})")
            else:
                lines.append(f"- {action}: {sample_label(sample)}")
    return lines if len(lines) > 2 else []


def render_managed_issue_audit(audit: dict[str, Any] | None) -> list[str]:
    if not audit:
        return []
    lines = ["", "Managed issue audit:"]
    for key, title in (
        ("stale", "Stale managed"),
        ("duplicates", "Duplicate markers"),
        ("preserved_extracted", "Preserved extracted"),
        ("ignored_unknown", "Ignored unknown markers"),
    ):
        bucket = audit.get(key, {})
        if not bucket.get("total"):
            continue
        lines.append(
            "- {}: total={}, open={}, closed={}".format(
                title,
                bucket.get("total", 0),
                bucket.get("open", 0),
                bucket.get("closed", 0),
            )
        )
        for sample in bucket.get("samples", []):
            details = sample_details(sample)
            category = sample.get("category")
            if category:
                details = ", ".join(value for value in (details, category) if value)
            if details:
                lines.append(f"  - {sample_label(sample)} ({details})")
            else:
                lines.append(f"  - {sample_label(sample)}")
    return lines if len(lines) > 2 else []


def operation_ledger_label(entry: dict[str, Any]) -> str:
    if entry.get("action") == "attached":
        child_key = entry.get("child_key", "unknown")
        child_number = entry.get("child_number")
        if child_number is not None:
            return f"`{child_key}` (#{child_number})"
        return f"`{child_key}`"
    return sample_label(entry)


def operation_ledger_details(entry: dict[str, Any]) -> str:
    details = []
    if entry.get("reason"):
        details.append("reason={}".format(entry["reason"]))
    if entry.get("reasons"):
        details.append("reasons={}".format(",".join(entry["reasons"])))
    if entry.get("parent_key"):
        details.append("parent={}".format(entry["parent_key"]))
    if entry.get("parent_number"):
        details.append("parent_number={}".format(entry["parent_number"]))
    return ", ".join(details)


def action_reason_summary(
    action_reasons: dict[str, dict[str, int]] | None,
    action: str,
) -> str:
    reasons = (action_reasons or {}).get(action) or {}
    if not reasons:
        return "none"
    return ", ".join(
        f"{reason}={count}"
        for reason, count in sorted(
            reasons.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )
    )


def render_operation_ledger(entries: list[dict[str, Any]] | None) -> list[str]:
    if not entries:
        return []
    lines = ["", "Operation ledger:"]
    for entry in entries[:OPERATION_LEDGER_SUMMARY_LIMIT]:
        details = operation_ledger_details(entry)
        action = entry.get("action", "unknown")
        if details:
            lines.append(
                "- {}: {} ({})".format(
                    action,
                    operation_ledger_label(entry),
                    details,
                )
            )
        else:
            lines.append(f"- {action}: {operation_ledger_label(entry)}")
    if len(entries) > OPERATION_LEDGER_SUMMARY_LIMIT:
        lines.append(
            "Additional operation ledger entries omitted from summary: {}".format(
                len(entries) - OPERATION_LEDGER_SUMMARY_LIMIT
            )
        )
    return lines


def render_load_error(
    title: str,
    report: dict[str, Any],
    path: Path | None,
) -> list[str]:
    error = report.get("load_error", {})
    return [
        f"## {title}",
        "",
        f"Report: failed to load `{display_path(path)}`.",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Error", error.get("type", "unknown")],
                ["Message", error.get("message", "")],
            ],
        ),
    ]


def matrix_check_status(report: dict[str, Any] | None) -> str:
    if report is None:
        return "missing"
    if report.get("load_error"):
        return "load-error"
    return status_label(report.get("ok"))


def evidence_check_status(report: dict[str, Any] | None) -> str:
    if report is None:
        return "missing"
    if report.get("load_error"):
        return "load-error"
    summary = report.get("summary", {})
    if summary.get("missing_evidence_count", 0):
        return "warning"
    return "pass"


def audit_bucket_open_count(
    audit: dict[str, Any] | None,
    bucket: str,
) -> int:
    if not audit:
        return 0
    value = audit.get(bucket)
    if not isinstance(value, dict):
        return 0
    return int(value.get("open", 0))


def issue_plan_status(report: dict[str, Any] | None) -> str:
    if report is None:
        return "missing"
    if report.get("load_error"):
        return "load-error"
    if report.get("input_failures"):
        return "fail"
    if report.get("preflight_failure"):
        return "fail"

    matrix_check = report.get("support_matrix_check", {})
    if matrix_check.get("provided") and matrix_check.get("load_error"):
        return "fail"
    if matrix_check.get("provided") and matrix_check.get("ok") is False:
        return "fail"

    for budget_key in ("planned_action_budget", "planned_closure_budget"):
        budget = report.get(budget_key, {})
        if (
            budget.get("provided")
            and budget.get("evaluated")
            and budget.get("ok") is False
        ):
            return "fail"
    if audit_bucket_open_count(report.get("managed_issue_audit"), "ignored_unknown"):
        return "fail"

    return "pass" if report.get("mode") else "unknown"


def issue_sync_status(report: dict[str, Any] | None) -> str:
    if report is None:
        return "missing"
    if report.get("load_error"):
        return "load-error"
    if report.get("sync_failure"):
        return "fail"
    reconciliation = report.get("operation_reconciliation") or {}
    if reconciliation.get("evaluated") and reconciliation.get("ok") is False:
        return "fail"
    if report.get("sync_summary") is not None:
        return "pass"
    return "unknown"


def workflow_source_label(source: dict[str, Any]) -> str:
    workflow = source.get("workflow") or "unknown"
    details = []
    run_id = source.get("run_id")
    if run_id:
        details.append(f"#{run_id}")
    conclusion = source.get("conclusion")
    if conclusion:
        details.append(str(conclusion))
    event = source.get("event")
    if event:
        details.append(f"event={event}")
    if details:
        return "{} ({})".format(workflow, ", ".join(details))
    return str(workflow)


def add_workflow_source_rows(
    rows: list[list[Any]],
    report: dict[str, Any],
) -> None:
    source = report.get("workflow_source")
    if not isinstance(source, dict) or not source:
        return
    rows.append(["Source workflow", workflow_source_label(source)])
    head_sha = source.get("head_sha")
    if head_sha:
        rows.append(["Source head SHA", str(head_sha)[:12]])


def overall_status(*statuses: str) -> str:
    if any(status in {"fail", "load-error"} for status in statuses):
        return "attention"
    if any(status in {"missing", "unknown"} for status in statuses):
        return "incomplete"
    return "pass"


def summary_status(
    matrix_check: dict[str, Any] | None,
    issue_plan: dict[str, Any] | None,
    sync_summary: dict[str, Any] | None,
    evidence_check: dict[str, Any] | None = None,
    evidence_required: bool = False,
) -> dict[str, str]:
    matrix_status = matrix_check_status(matrix_check)
    evidence_status = evidence_check_status(evidence_check)
    plan_status = issue_plan_status(issue_plan)
    sync_status = issue_sync_status(sync_summary)
    overall_inputs = [matrix_status, plan_status, sync_status]
    if evidence_required or evidence_check is not None:
        overall_inputs.append(evidence_status)
    statuses = {
        "overall": overall_status(*overall_inputs),
        "matrix": matrix_status,
        "issue_plan": plan_status,
        "sync": sync_status,
    }
    if evidence_required or evidence_check is not None:
        statuses["evidence"] = evidence_status
    return statuses


def render_overall_summary(
    matrix_check: dict[str, Any] | None,
    issue_plan: dict[str, Any] | None,
    sync_summary: dict[str, Any] | None,
    evidence_check: dict[str, Any] | None = None,
    evidence_required: bool = False,
) -> list[str]:
    statuses = summary_status(
        matrix_check,
        issue_plan,
        sync_summary,
        evidence_check=evidence_check,
        evidence_required=evidence_required,
    )
    rows = [
        ["Overall", statuses["overall"]],
        ["Support matrix", statuses["matrix"]],
    ]
    if "evidence" in statuses:
        rows.append(["Support evidence", statuses["evidence"]])
    rows.extend(
        [
            ["Issue plan", statuses["issue_plan"]],
            ["Issue sync", statuses["sync"]],
        ]
    )
    return [
        "## Overall",
        "",
        markdown_table(["Field", "Value"], rows),
    ]


def render_matrix_check(report: dict[str, Any] | None, path: Path | None) -> list[str]:
    if not report:
        return [
            "## Support Matrix",
            "",
            f"Report: not available at `{display_path(path)}`.",
        ]
    if report.get("load_error"):
        return render_load_error("Support Matrix", report, path)

    summary = report.get("summary", {})
    rows = [
        ["Report", f"`{display_path(path)}`"],
        ["Status", status_label(report.get("ok"))],
        ["Artifacts", summary.get("artifact_count", 0)],
        ["Stale artifacts", summary.get("stale_count", 0)],
    ]
    lines = ["## Support Matrix", "", markdown_table(["Field", "Value"], rows)]
    stale_artifacts = [
        artifact for artifact in report.get("artifacts", []) if artifact.get("stale")
    ]
    if stale_artifacts:
        lines.extend(["", "Stale artifacts:"])
        for artifact in stale_artifacts:
            lines.append(
                "- `{}`: {} diff lines".format(
                    artifact.get("path", "<unknown>"),
                    artifact.get("diff_line_count", 0),
                )
            )
    return lines


def evidence_backend_rows(summary: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for backend_id, counts in sorted(
        (summary.get("by_backend") or {}).items(),
        key=lambda item: (-int(item[1].get("missing", 0)), item[0]),
    ):
        rows.append(
            [
                backend_id,
                counts.get("rows", 0),
                counts.get("present", 0),
                counts.get("missing", 0),
            ]
        )
    return rows


def render_evidence_check(
    report: dict[str, Any] | None,
    path: Path | None,
) -> list[str]:
    if path is None and report is None:
        return []
    if not report:
        return [
            "## Support Evidence",
            "",
            f"Report: not available at `{display_path(path)}`.",
        ]
    if report.get("load_error"):
        return render_load_error("Support Evidence", report, path)

    summary = report.get("summary", {})
    lines = [
        "## Support Evidence",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Report", f"`{display_path(path)}`"],
                ["Status", evidence_check_status(report)],
                ["Rows", summary.get("row_count", 0)],
                ["Rows with evidence", summary.get("present_evidence_count", 0)],
                ["Rows missing evidence", summary.get("missing_evidence_count", 0)],
            ],
        ),
    ]

    backend_rows = evidence_backend_rows(summary)
    if backend_rows:
        lines.extend(
            [
                "",
                "Missing evidence by backend:",
                markdown_table(
                    ["Backend", "Rows", "With evidence", "Missing evidence"],
                    backend_rows,
                ),
            ]
        )

    missing_rows = [
        row for row in report.get("rows", []) if row.get("evidence_count", 0) == 0
    ]
    if missing_rows:
        lines.extend(["", "Missing evidence samples:"])
        for row in missing_rows[:EVIDENCE_ROW_SUMMARY_LIMIT]:
            lines.append(
                "- {backend}: {feature} [{status}]".format(
                    backend=row.get("backend", row.get("backend_id", "unknown")),
                    feature=row.get("feature", row.get("feature_id", "unknown")),
                    status=row.get("status", "unknown"),
                )
            )
        if len(missing_rows) > EVIDENCE_ROW_SUMMARY_LIMIT:
            lines.append(
                "Additional missing-evidence rows omitted from summary: {}".format(
                    len(missing_rows) - EVIDENCE_ROW_SUMMARY_LIMIT
                )
            )
    return lines


def render_issue_plan(report: dict[str, Any] | None, path: Path | None) -> list[str]:
    if not report:
        return [
            "## Issue Plan",
            "",
            f"Report: not available at `{display_path(path)}`.",
        ]
    if report.get("load_error"):
        return render_load_error("Issue Plan", report, path)

    rows = [
        ["Report", f"`{display_path(path)}`"],
        ["Mode", report.get("mode", "unknown")],
    ]
    add_workflow_source_rows(rows, report)
    if "close_extracted_issues" in report:
        rows.append(["Close stale extracted issues", report["close_extracted_issues"]])
    if "close_pytest_failure_issues" in report:
        rows.append(
            ["Close stale pytest-failure issues", report["close_pytest_failure_issues"]]
        )
    rows.extend(
        ["Desired " + key, value] for key, value in count_rows(report.get("desired"))
    )
    existing = report.get("existing", {})
    rows.extend(
        [
            ["Existing inspected", existing.get("inspected", False)],
            ["Existing managed", existing.get("managed", 0)],
            ["Existing duplicates", existing.get("duplicates", 0)],
        ]
    )
    planned = report.get("planned_actions")
    if planned is not None:
        rows.extend(["Planned " + key, value] for key, value in count_rows(planned))
    planned_closures = report.get("planned_closures")
    if planned_closures is not None:
        rows.extend(
            ["Planned closure " + key, value]
            for key, value in count_rows(planned_closures)
        )
    input_failures = report.get("input_failures") or []
    if input_failures:
        rows.append(["Input failures", len(input_failures)])
    audit = report.get("managed_issue_audit") or {}
    for key, label in (
        ("stale", "Audit stale managed"),
        ("duplicates", "Audit duplicate markers"),
        ("preserved_extracted", "Audit preserved extracted"),
        ("ignored_unknown", "Audit ignored unknown"),
    ):
        bucket = audit.get(key, {})
        if bucket.get("total"):
            rows.append([label, bucket.get("total", 0)])

    matrix_check = report.get("support_matrix_check", {})
    if matrix_check.get("provided"):
        rows.append(["Embedded matrix check", status_label(matrix_check.get("ok"))])
        load_error = matrix_check.get("load_error")
        if load_error:
            rows.extend(
                [
                    [
                        "Embedded matrix check error",
                        load_error.get("type", "unknown"),
                    ],
                    [
                        "Embedded matrix check message",
                        load_error.get("message", ""),
                    ],
                ]
            )

    budget = report.get("planned_action_budget", {})
    if budget.get("provided"):
        budget_status = (
            status_label(budget.get("ok"))
            if budget.get("evaluated")
            else "not evaluated"
        )
        rows.extend(
            [
                ["Planned action budget", budget_status],
                ["Budget mode", budget.get("mode", "unknown")],
            ]
        )
    closure_budget = report.get("planned_closure_budget", {})
    if closure_budget.get("provided"):
        closure_budget_status = (
            status_label(closure_budget.get("ok"))
            if closure_budget.get("evaluated")
            else "not evaluated"
        )
        rows.append(["Planned closure budget", closure_budget_status])
    preflight_failure = report.get("preflight_failure", {})
    if preflight_failure:
        error = preflight_failure.get("error", {})
        rows.extend(
            [
                ["Preflight failure phase", preflight_failure.get("phase", "unknown")],
                ["Preflight failure error", error.get("type", "unknown")],
                ["Preflight failure message", error.get("message", "")],
            ]
        )

    lines = ["## Issue Plan", "", markdown_table(["Field", "Value"], rows)]
    violations = budget.get("violations", []) if budget.get("provided") else []
    if violations:
        lines.extend(["", "Budget violations:"])
        for violation in violations:
            lines.append(
                "- {action}: {actual} > {limit}".format(
                    action=violation.get("action", "unknown"),
                    actual=violation.get("actual", 0),
                    limit=violation.get("limit", 0),
                )
            )
    closure_violations = (
        closure_budget.get("violations", []) if closure_budget.get("provided") else []
    )
    if closure_violations:
        lines.extend(["", "Closure budget violations:"])
        for violation in closure_violations:
            lines.append(
                "- {category}: {actual} > {limit}".format(
                    category=violation.get("category", "unknown"),
                    actual=violation.get("actual", 0),
                    limit=violation.get("limit", 0),
                )
            )
    if input_failures:
        lines.extend(["", "Input failures:"])
        for failure in input_failures:
            error = failure.get("error", {})
            lines.append(
                "- {input}: {type} - {message}".format(
                    input=failure.get("input", "unknown"),
                    type=error.get("type", "unknown"),
                    message=error.get("message", ""),
                )
            )
    operation = preflight_failure.get("operation", {}) if preflight_failure else {}
    if operation:
        lines.extend(["", "Preflight failure operation:"])
        for key, value in sorted(operation.items()):
            lines.append(f"- {key}: `{value}`")
    lines.extend(render_managed_issue_audit(audit))
    lines.extend(render_action_samples(report.get("planned_action_samples")))
    return lines


def render_sync_summary(report: dict[str, Any] | None, path: Path | None) -> list[str]:
    if not report:
        return [
            "## Issue Sync",
            "",
            f"Report: not available at `{display_path(path)}`.",
        ]
    if report.get("load_error"):
        return render_load_error("Issue Sync", report, path)

    rows = [
        ["Report", f"`{display_path(path)}`"],
        ["Mode", report.get("mode", "unknown")],
    ]
    add_workflow_source_rows(rows, report)
    if "close_extracted_issues" in report:
        rows.append(["Close stale extracted issues", report["close_extracted_issues"]])
    if "close_pytest_failure_issues" in report:
        rows.append(
            ["Close stale pytest-failure issues", report["close_pytest_failure_issues"]]
        )
    summary = report.get("sync_summary")
    if summary is None:
        rows.append(["Sync summary", "not available"])
    else:
        rows.extend(["Sync " + key, value] for key, value in count_rows(summary))
    failure = report.get("sync_failure", {})
    operation_ledger = report.get("operation_ledger")
    if operation_ledger is None and failure:
        operation_ledger = failure.get("operation_ledger")
    if operation_ledger:
        rows.append(["Operation ledger entries", len(operation_ledger)])
    reconciliation = report.get("operation_reconciliation") or {}
    if reconciliation:
        reconciliation_status = (
            status_label(reconciliation.get("ok"))
            if reconciliation.get("evaluated")
            else "not evaluated"
        )
        rows.extend(
            [
                ["Operation reconciliation", reconciliation_status],
                [
                    "Operation action overruns",
                    len(reconciliation.get("action_overruns", [])),
                ],
                [
                    "Operation action shortfalls",
                    len(reconciliation.get("action_shortfalls", [])),
                ],
                [
                    "Operation closure overruns",
                    len(reconciliation.get("closure_overruns", [])),
                ],
                [
                    "Operation closure shortfalls",
                    len(reconciliation.get("closure_shortfalls", [])),
                ],
            ]
        )
    if failure:
        error = failure.get("error", {})
        rows.extend(
            [
                ["Sync failure phase", failure.get("phase", "unknown")],
                ["Sync failure error", error.get("type", "unknown")],
                ["Sync failure message", error.get("message", "")],
            ]
        )
        recovery = failure.get("recovery", {})
        if recovery:
            rows.extend(
                [
                    ["Sync recovery rerun safe", recovery.get("rerun_safe", False)],
                    ["Sync recovery strategy", recovery.get("strategy", "")],
                ]
            )

    lines = ["## Issue Sync", "", markdown_table(["Field", "Value"], rows)]
    operation = failure.get("operation", {}) if failure else {}
    if operation:
        lines.extend(["", "Sync failure operation:"])
        for key, value in sorted(operation.items()):
            lines.append(f"- {key}: `{value}`")
    action_overruns = reconciliation.get("action_overruns", [])
    action_shortfalls = reconciliation.get("action_shortfalls", [])
    closure_overruns = reconciliation.get("closure_overruns", [])
    closure_shortfalls = reconciliation.get("closure_shortfalls", [])
    if action_overruns or action_shortfalls or closure_overruns or closure_shortfalls:
        lines.extend(["", "Operation reconciliation differences:"])
        action_reasons = reconciliation.get("actual_action_reasons") or {}
        for overrun in action_overruns:
            lines.append(
                "- action {action}: {actual} > planned {planned} (actual reasons: {reasons})".format(
                    action=overrun.get("action", "unknown"),
                    actual=overrun.get("actual", 0),
                    planned=overrun.get("planned", 0),
                    reasons=action_reason_summary(
                        action_reasons,
                        overrun.get("action", "unknown"),
                    ),
                )
            )
        for shortfall in action_shortfalls:
            lines.append(
                "- action {action}: {actual} < planned {planned} (actual reasons: {reasons})".format(
                    action=shortfall.get("action", "unknown"),
                    actual=shortfall.get("actual", 0),
                    planned=shortfall.get("planned", 0),
                    reasons=action_reason_summary(
                        action_reasons,
                        shortfall.get("action", "unknown"),
                    ),
                )
            )
        for overrun in closure_overruns:
            lines.append(
                "- closure {category}: {actual} > planned {planned}".format(
                    category=overrun.get("category", "unknown"),
                    actual=overrun.get("actual", 0),
                    planned=overrun.get("planned", 0),
                )
            )
        for shortfall in closure_shortfalls:
            lines.append(
                "- closure {category}: {actual} < planned {planned}".format(
                    category=shortfall.get("category", "unknown"),
                    actual=shortfall.get("actual", 0),
                    planned=shortfall.get("planned", 0),
                )
            )
    lines.extend(render_operation_ledger(operation_ledger))
    return lines


def github_command_escape(value: Any, *, property_value: bool = False) -> str:
    text = str(value)
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def github_annotation(
    title: str,
    message: str,
    *,
    file: str | None = None,
    level: str = "error",
) -> str:
    properties = [f"title={github_command_escape(title, property_value=True)}"]
    if file:
        properties.insert(
            0,
            f"file={github_command_escape(file, property_value=True)}",
        )
    return "::{level} {properties}::{message}".format(
        level=level,
        properties=",".join(properties),
        message=github_command_escape(message),
    )


def report_load_error_annotation(
    title: str,
    report: dict[str, Any] | None,
    path: Path | None,
) -> list[str]:
    if not report or not report.get("load_error"):
        return []
    load_error = report.get("load_error", {})
    return [
        github_annotation(
            title,
            "{}: {}".format(
                load_error.get("type", "unknown"),
                load_error.get("message", ""),
            ),
            file=display_path(path),
        )
    ]


def evidence_gap_message(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    missing = summary.get("missing_evidence_count", 0)
    backend_counts = [
        "{}={}".format(backend_id, counts.get("missing", 0))
        for backend_id, counts in sorted(
            (summary.get("by_backend") or {}).items(),
            key=lambda item: (-int(item[1].get("missing", 0)), item[0]),
        )
        if counts.get("missing", 0)
    ]
    suffix = ""
    if backend_counts:
        suffix = " ({})".format(", ".join(backend_counts[:8]))
    return f"{missing} supported support-matrix rows are missing evidence{suffix}."


def github_annotation_lines(
    matrix_check: dict[str, Any] | None,
    matrix_check_path: Path | None,
    issue_plan: dict[str, Any] | None,
    issue_plan_path: Path | None,
    sync_summary: dict[str, Any] | None,
    sync_summary_path: Path | None,
    evidence_check: dict[str, Any] | None = None,
    evidence_check_path: Path | None = None,
) -> list[str]:
    lines: list[str] = []
    lines.extend(
        report_load_error_annotation(
            "Support matrix report load error",
            matrix_check,
            matrix_check_path,
        )
    )
    if (
        matrix_check
        and not matrix_check.get("load_error")
        and matrix_check.get("ok") is False
    ):
        lines.append(
            github_annotation(
                "Support matrix check failed",
                "Generated support matrix artifacts are stale or invalid.",
                file=display_path(matrix_check_path),
            )
        )
        for artifact in matrix_check.get("artifacts", []):
            if artifact.get("stale"):
                lines.append(
                    github_annotation(
                        "Stale support matrix artifact",
                        "{} diff lines".format(artifact.get("diff_line_count", 0)),
                        file=artifact.get("path") or display_path(matrix_check_path),
                    )
                )

    lines.extend(
        report_load_error_annotation(
            "Support evidence report load error",
            evidence_check,
            evidence_check_path,
        )
    )
    if (
        evidence_check
        and not evidence_check.get("load_error")
        and evidence_check.get("summary", {}).get("missing_evidence_count", 0)
    ):
        lines.append(
            github_annotation(
                "Support matrix evidence gaps",
                evidence_gap_message(evidence_check),
                file=display_path(evidence_check_path),
                level="warning",
            )
        )

    lines.extend(
        report_load_error_annotation(
            "Support issue plan report load error",
            issue_plan,
            issue_plan_path,
        )
    )
    if issue_plan and not issue_plan.get("load_error"):
        for failure in issue_plan.get("input_failures") or []:
            error = failure.get("error", {})
            lines.append(
                github_annotation(
                    "Support issue input failure",
                    "{input}: {type} - {message}".format(
                        input=failure.get("input", "unknown"),
                        type=error.get("type", "unknown"),
                        message=error.get("message", ""),
                    ),
                    file=failure.get("path"),
                )
            )

        matrix_summary = issue_plan.get("support_matrix_check", {})
        load_error = matrix_summary.get("load_error")
        if matrix_summary.get("provided") and load_error:
            lines.append(
                github_annotation(
                    "Embedded support matrix report error",
                    "{}: {}".format(
                        load_error.get("type", "unknown"),
                        load_error.get("message", ""),
                    ),
                    file=matrix_summary.get("path"),
                )
            )

        budget = issue_plan.get("planned_action_budget", {})
        if budget.get("provided") and budget.get("evaluated"):
            for violation in budget.get("violations", []):
                lines.append(
                    github_annotation(
                        "Support issue action budget exceeded",
                        "{action}: {actual} > {limit}".format(
                            action=violation.get("action", "unknown"),
                            actual=violation.get("actual", 0),
                            limit=violation.get("limit", 0),
                        ),
                        file=display_path(issue_plan_path),
                    )
                )

        closure_budget = issue_plan.get("planned_closure_budget", {})
        if closure_budget.get("provided") and closure_budget.get("evaluated"):
            for violation in closure_budget.get("violations", []):
                lines.append(
                    github_annotation(
                        "Support issue closure budget exceeded",
                        "{category}: {actual} > {limit}".format(
                            category=violation.get("category", "unknown"),
                            actual=violation.get("actual", 0),
                            limit=violation.get("limit", 0),
                        ),
                        file=display_path(issue_plan_path),
                    )
                )

        preflight_failure = issue_plan.get("preflight_failure") or {}
        if preflight_failure:
            error = preflight_failure.get("error", {})
            lines.append(
                github_annotation(
                    "Support issue preflight failure",
                    "{phase}: {type} - {message}".format(
                        phase=preflight_failure.get("phase", "unknown"),
                        type=error.get("type", "unknown"),
                        message=error.get("message", ""),
                    ),
                    file=display_path(issue_plan_path),
                )
            )

        audit = issue_plan.get("managed_issue_audit") or {}
        ignored_unknown = audit.get("ignored_unknown") or {}
        if ignored_unknown.get("open", 0):
            lines.append(
                github_annotation(
                    "Unknown managed support issue markers",
                    (
                        "{} open managed support issues have sync markers this "
                        "tool does not understand."
                    ).format(ignored_unknown.get("open", 0)),
                    file=display_path(issue_plan_path),
                )
            )

    lines.extend(
        report_load_error_annotation(
            "Support issue sync report load error",
            sync_summary,
            sync_summary_path,
        )
    )
    if sync_summary and not sync_summary.get("load_error"):
        failure = sync_summary.get("sync_failure") or {}
        if failure:
            error = failure.get("error", {})
            lines.append(
                github_annotation(
                    "Support issue sync failure",
                    "{phase}: {type} - {message}".format(
                        phase=failure.get("phase", "unknown"),
                        type=error.get("type", "unknown"),
                        message=error.get("message", ""),
                    ),
                    file=display_path(sync_summary_path),
                )
            )
        reconciliation = sync_summary.get("operation_reconciliation") or {}
        if reconciliation.get("evaluated"):
            action_reasons = reconciliation.get("actual_action_reasons") or {}
            for overrun in reconciliation.get("action_overruns", []):
                lines.append(
                    github_annotation(
                        "Support issue sync exceeded planned actions",
                        "{action}: {actual} > planned {planned}; actual reasons: {reasons}".format(
                            action=overrun.get("action", "unknown"),
                            actual=overrun.get("actual", 0),
                            planned=overrun.get("planned", 0),
                            reasons=action_reason_summary(
                                action_reasons,
                                overrun.get("action", "unknown"),
                            ),
                        ),
                        file=display_path(sync_summary_path),
                    )
                )
            for shortfall in reconciliation.get("action_shortfalls", []):
                lines.append(
                    github_annotation(
                        "Support issue sync missed planned actions",
                        "{action}: {actual} < planned {planned}; actual reasons: {reasons}".format(
                            action=shortfall.get("action", "unknown"),
                            actual=shortfall.get("actual", 0),
                            planned=shortfall.get("planned", 0),
                            reasons=action_reason_summary(
                                action_reasons,
                                shortfall.get("action", "unknown"),
                            ),
                        ),
                        file=display_path(sync_summary_path),
                    )
                )
            for overrun in reconciliation.get("closure_overruns", []):
                lines.append(
                    github_annotation(
                        "Support issue sync exceeded planned closures",
                        "{category}: {actual} > planned {planned}".format(
                            category=overrun.get("category", "unknown"),
                            actual=overrun.get("actual", 0),
                            planned=overrun.get("planned", 0),
                        ),
                        file=display_path(sync_summary_path),
                    )
                )
            for shortfall in reconciliation.get("closure_shortfalls", []):
                lines.append(
                    github_annotation(
                        "Support issue sync missed planned closures",
                        "{category}: {actual} < planned {planned}".format(
                            category=shortfall.get("category", "unknown"),
                            actual=shortfall.get("actual", 0),
                            planned=shortfall.get("planned", 0),
                        ),
                        file=display_path(sync_summary_path),
                    )
                )

    return lines


def render_summary(
    matrix_check: dict[str, Any] | None,
    matrix_check_path: Path | None,
    issue_plan: dict[str, Any] | None,
    issue_plan_path: Path | None,
    sync_summary: dict[str, Any] | None,
    sync_summary_path: Path | None,
    evidence_check: dict[str, Any] | None = None,
    evidence_check_path: Path | None = None,
) -> str:
    lines = ["# Support Automation Summary", ""]
    evidence_required = evidence_check_path is not None
    lines.extend(
        render_overall_summary(
            matrix_check,
            issue_plan,
            sync_summary,
            evidence_check=evidence_check,
            evidence_required=evidence_required,
        )
    )
    lines.extend([""])
    lines.extend(render_matrix_check(matrix_check, matrix_check_path))
    evidence_lines = render_evidence_check(evidence_check, evidence_check_path)
    if evidence_lines:
        lines.extend([""])
        lines.extend(evidence_lines)
    lines.extend([""])
    lines.extend(render_issue_plan(issue_plan, issue_plan_path))
    lines.extend([""])
    lines.extend(render_sync_summary(sync_summary, sync_summary_path))
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-check",
        type=Path,
        default=DEFAULT_MATRIX_CHECK_PATH,
        help="support_matrix.py check JSON report",
    )
    parser.add_argument(
        "--support-evidence",
        type=Path,
        help=(
            "Optional support_matrix.py evidence JSON report. "
            "When provided, missing support evidence is summarized but not fatal."
        ),
    )
    parser.add_argument(
        "--issue-plan",
        type=Path,
        default=DEFAULT_ISSUE_PLAN_PATH,
        help="sync_support_issues.py plan JSON report",
    )
    parser.add_argument(
        "--sync-summary",
        type=Path,
        default=DEFAULT_SYNC_SUMMARY_PATH,
        help="sync_support_issues.py post-sync summary JSON report",
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    parser.add_argument(
        "--step-summary",
        type=Path,
        help="Optional GitHub Step Summary path to append the Markdown output to",
    )
    parser.add_argument(
        "--github-annotations",
        action="store_true",
        help="Emit GitHub Actions error annotations for actionable failures",
    )
    parser.add_argument(
        "--fail-on-attention",
        action="store_true",
        help="Exit non-zero when the rendered overall status is attention",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    matrix_check_path = resolve_path(args.matrix_check)
    evidence_check_path = resolve_path(args.support_evidence)
    issue_plan_path = resolve_path(args.issue_plan)
    sync_summary_path = resolve_path(args.sync_summary)
    matrix_check = load_optional_json(
        matrix_check_path,
        expected_generator=MATRIX_CHECK_GENERATOR,
        required_fields=MATRIX_CHECK_REQUIRED_FIELDS,
        contract_validator=validate_matrix_check_contract,
    )
    evidence_check = (
        load_optional_json(
            evidence_check_path,
            expected_generator=EVIDENCE_CHECK_GENERATOR,
            required_fields=EVIDENCE_CHECK_REQUIRED_FIELDS,
            contract_validator=validate_evidence_check_contract,
        )
        if evidence_check_path is not None
        else None
    )
    issue_plan = load_optional_json(
        issue_plan_path,
        expected_generator=ISSUE_SYNC_GENERATOR,
        required_fields=ISSUE_PLAN_REQUIRED_FIELDS,
        contract_validator=validate_issue_plan_contract,
    )
    sync_summary = load_optional_json(
        sync_summary_path,
        expected_generator=ISSUE_SYNC_GENERATOR,
        required_fields=SYNC_SUMMARY_REQUIRED_FIELDS,
        contract_validator=validate_sync_summary_contract,
    )
    text = render_summary(
        matrix_check,
        matrix_check_path,
        issue_plan,
        issue_plan_path,
        sync_summary,
        sync_summary_path,
        evidence_check=evidence_check,
        evidence_check_path=evidence_check_path,
    )

    output = resolve_path(args.output)
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Wrote {display_path(output)}")

    if args.step_summary is not None:
        args.step_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.step_summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
        print(f"Appended {args.step_summary}")

    if args.github_annotations:
        for annotation in github_annotation_lines(
            matrix_check,
            matrix_check_path,
            issue_plan,
            issue_plan_path,
            sync_summary,
            sync_summary_path,
            evidence_check=evidence_check,
            evidence_check_path=evidence_check_path,
        ):
            print(annotation)

    statuses = summary_status(
        matrix_check,
        issue_plan,
        sync_summary,
        evidence_check=evidence_check,
        evidence_required=evidence_check_path is not None,
    )
    if args.fail_on_attention and statuses["overall"] == "attention":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
