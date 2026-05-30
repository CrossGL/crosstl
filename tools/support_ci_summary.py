#!/usr/bin/env python3
"""Render a concise Markdown summary for support automation CI reports."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_CHECK_PATH = ROOT / "support" / "generated" / "support-matrix-check.json"
DEFAULT_ISSUE_PLAN_PATH = ROOT / "support" / "generated" / "support-issue-plan.json"
DEFAULT_SYNC_SUMMARY_PATH = (
    ROOT / "support" / "generated" / "support-issue-sync-summary.json"
)
MATRIX_CHECK_GENERATOR = "tools/support_matrix.py check"
ISSUE_SYNC_GENERATOR = "tools/sync_support_issues.py"
SUPPORTED_SCHEMA_VERSION = 1
MATRIX_CHECK_REQUIRED_FIELDS = ("schema_version", "generator", "ok", "summary")
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
        "{} must be {}, got {}".format(field, expected, value_type_label(value)),
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
                "{}.{}".format(prefix, field),
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
            return invalid_field_error(
                path, "{}.{}".format(field, counter), int, counter_value
            )
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
                "{}[{}]".format(field, index),
                dict,
                item,
            )
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
            "{} must be dry-run or sync, got {}".format(field, mode),
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
                "{}.{}".format(field, bool_field),
                bool,
                value[bool_field],
            )
    if (
        value.get("evaluated")
        and "ok" in value
        and not value_matches_type(value["ok"], bool)
    ):
        return invalid_field_error(path, "{}.ok".format(field), bool, value["ok"])
    violations = value.get("violations")
    if violations is None:
        return None
    if not isinstance(violations, list):
        return invalid_field_error(
            path,
            "{}.violations".format(field),
            list,
            violations,
        )
    for index, violation in enumerate(violations):
        if not isinstance(violation, dict):
            return invalid_field_error(
                path,
                "{}.violations[{}]".format(field, index),
                dict,
                violation,
            )
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
    error = validate_counter_map(report, path, "summary", MATRIX_CHECK_SUMMARY_COUNTERS)
    if error is not None:
        return error
    stale_artifacts = report["summary"].get("stale_artifacts")
    if stale_artifacts is None:
        return load_error(
            path,
            "MissingReportFields",
            "summary missing required fields: stale_artifacts",
        )
    if not isinstance(stale_artifacts, list):
        return invalid_field_error(
            path, "summary.stale_artifacts", list, stale_artifacts
        )
    for index, artifact_path in enumerate(stale_artifacts):
        if not isinstance(artifact_path, str):
            return invalid_field_error(
                path,
                "summary.stale_artifacts[{}]".format(index),
                str,
                artifact_path,
            )
    error = validate_optional_object_list(report, path, "artifacts")
    if error is not None:
        return error
    for index, artifact in enumerate(report.get("artifacts") or []):
        error = validate_nested_field_types(
            artifact,
            path,
            "artifacts[{}]".format(index),
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
            return invalid_field_error(
                path, "artifacts[{}].diff".format(index), list, diff
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
        lambda: validate_optional_object(report, path, "planned_action_samples"),
        lambda: validate_optional_object(report, path, "managed_issue_audit"),
        lambda: validate_optional_object_list(report, path, "input_failures"),
        lambda: validate_optional_object(report, path, "preflight_failure"),
        lambda: validate_budget_contract(report, path, "planned_action_budget"),
        lambda: validate_budget_contract(report, path, "planned_closure_budget"),
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
                "existing missing required counters: {}".format(counter),
            )
        if not value_matches_type(existing[counter], int):
            return invalid_field_error(
                path,
                "existing.{}".format(counter),
                int,
                existing[counter],
            )

    matrix_check = report["support_matrix_check"]
    if "provided" not in matrix_check:
        return load_error(
            path,
            "MissingReportFields",
            "support_matrix_check missing required fields: provided",
        )
    if not value_matches_type(matrix_check["provided"], bool):
        return invalid_field_error(
            path,
            "support_matrix_check.provided",
            bool,
            matrix_check["provided"],
        )
    if (
        "ok" in matrix_check
        and matrix_check["ok"] is not None
        and not value_matches_type(
            matrix_check["ok"],
            bool,
        )
    ):
        return invalid_field_error(
            path, "support_matrix_check.ok", bool, matrix_check["ok"]
        )
    return None


def validate_sync_failure_contract(
    report: dict[str, Any],
    path: Path | None,
) -> dict[str, Any] | None:
    failure = report.get("sync_failure")
    if failure is None:
        return None
    if not isinstance(failure, dict):
        return invalid_field_error(path, "sync_failure", dict, failure)
    error = validate_nested_field_types(
        failure,
        path,
        "sync_failure",
        {
            "phase": str,
            "operation": dict,
            "partial_summary": dict,
            "error": dict,
        },
    )
    if error is not None:
        return error
    return validate_counter_map(
        failure,
        path,
        "partial_summary",
        ISSUE_ACTION_COUNTERS,
    )


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
        lambda: validate_optional_object_list(report, path, "operation_ledger"),
        lambda: validate_optional_object(report, path, "operation_reconciliation"),
        lambda: validate_sync_failure_contract(report, path),
    ):
        error = validator()
        if error is not None:
            return error
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
            "expected JSON object, got {}".format(type(data).__name__),
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
        return "`{}` (#{})".format(key, number)
    return "`{}`".format(key)


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
                lines.append(
                    "- {}: {} ({})".format(action, sample_label(sample), details)
                )
            else:
                lines.append("- {}: {}".format(action, sample_label(sample)))
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
                lines.append("  - {} ({})".format(sample_label(sample), details))
            else:
                lines.append("  - {}".format(sample_label(sample)))
    return lines if len(lines) > 2 else []


def operation_ledger_label(entry: dict[str, Any]) -> str:
    if entry.get("action") == "attached":
        child_key = entry.get("child_key", "unknown")
        child_number = entry.get("child_number")
        if child_number is not None:
            return "`{}` (#{})".format(child_key, child_number)
        return "`{}`".format(child_key)
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
            lines.append("- {}: {}".format(action, operation_ledger_label(entry)))
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
        "## {}".format(title),
        "",
        "Report: failed to load `{}`.".format(display_path(path)),
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
) -> dict[str, str]:
    matrix_status = matrix_check_status(matrix_check)
    plan_status = issue_plan_status(issue_plan)
    sync_status = issue_sync_status(sync_summary)
    return {
        "overall": overall_status(matrix_status, plan_status, sync_status),
        "matrix": matrix_status,
        "issue_plan": plan_status,
        "sync": sync_status,
    }


def render_overall_summary(
    matrix_check: dict[str, Any] | None,
    issue_plan: dict[str, Any] | None,
    sync_summary: dict[str, Any] | None,
) -> list[str]:
    statuses = summary_status(matrix_check, issue_plan, sync_summary)
    return [
        "## Overall",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Overall", statuses["overall"]],
                ["Support matrix", statuses["matrix"]],
                ["Issue plan", statuses["issue_plan"]],
                ["Issue sync", statuses["sync"]],
            ],
        ),
    ]


def render_matrix_check(report: dict[str, Any] | None, path: Path | None) -> list[str]:
    if not report:
        return [
            "## Support Matrix",
            "",
            "Report: not available at `{}`.".format(display_path(path)),
        ]
    if report.get("load_error"):
        return render_load_error("Support Matrix", report, path)

    summary = report.get("summary", {})
    rows = [
        ["Report", "`{}`".format(display_path(path))],
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


def render_issue_plan(report: dict[str, Any] | None, path: Path | None) -> list[str]:
    if not report:
        return [
            "## Issue Plan",
            "",
            "Report: not available at `{}`.".format(display_path(path)),
        ]
    if report.get("load_error"):
        return render_load_error("Issue Plan", report, path)

    rows = [
        ["Report", "`{}`".format(display_path(path))],
        ["Mode", report.get("mode", "unknown")],
    ]
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
            lines.append("- {}: `{}`".format(key, value))
    lines.extend(render_managed_issue_audit(audit))
    lines.extend(render_action_samples(report.get("planned_action_samples")))
    return lines


def render_sync_summary(report: dict[str, Any] | None, path: Path | None) -> list[str]:
    if not report:
        return [
            "## Issue Sync",
            "",
            "Report: not available at `{}`.".format(display_path(path)),
        ]
    if report.get("load_error"):
        return render_load_error("Issue Sync", report, path)

    rows = [
        ["Report", "`{}`".format(display_path(path))],
        ["Mode", report.get("mode", "unknown")],
    ]
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
                    "Operation closure overruns",
                    len(reconciliation.get("closure_overruns", [])),
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
            lines.append("- {}: `{}`".format(key, value))
    action_overruns = reconciliation.get("action_overruns", [])
    closure_overruns = reconciliation.get("closure_overruns", [])
    if action_overruns or closure_overruns:
        lines.extend(["", "Operation reconciliation overruns:"])
        for overrun in action_overruns:
            lines.append(
                "- action {action}: {actual} > planned {planned}".format(
                    action=overrun.get("action", "unknown"),
                    actual=overrun.get("actual", 0),
                    planned=overrun.get("planned", 0),
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
    properties = ["title={}".format(github_command_escape(title, property_value=True))]
    if file:
        properties.insert(
            0,
            "file={}".format(github_command_escape(file, property_value=True)),
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


def github_annotation_lines(
    matrix_check: dict[str, Any] | None,
    matrix_check_path: Path | None,
    issue_plan: dict[str, Any] | None,
    issue_plan_path: Path | None,
    sync_summary: dict[str, Any] | None,
    sync_summary_path: Path | None,
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
            for overrun in reconciliation.get("action_overruns", []):
                lines.append(
                    github_annotation(
                        "Support issue sync exceeded planned actions",
                        "{action}: {actual} > planned {planned}".format(
                            action=overrun.get("action", "unknown"),
                            actual=overrun.get("actual", 0),
                            planned=overrun.get("planned", 0),
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

    return lines


def render_summary(
    matrix_check: dict[str, Any] | None,
    matrix_check_path: Path | None,
    issue_plan: dict[str, Any] | None,
    issue_plan_path: Path | None,
    sync_summary: dict[str, Any] | None,
    sync_summary_path: Path | None,
) -> str:
    lines = ["# Support Automation Summary", ""]
    lines.extend(render_overall_summary(matrix_check, issue_plan, sync_summary))
    lines.extend([""])
    lines.extend(render_matrix_check(matrix_check, matrix_check_path))
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
    issue_plan_path = resolve_path(args.issue_plan)
    sync_summary_path = resolve_path(args.sync_summary)
    matrix_check = load_optional_json(
        matrix_check_path,
        expected_generator=MATRIX_CHECK_GENERATOR,
        required_fields=MATRIX_CHECK_REQUIRED_FIELDS,
        contract_validator=validate_matrix_check_contract,
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
    )

    output = resolve_path(args.output)
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print("Wrote {}".format(display_path(output)))

    if args.step_summary is not None:
        args.step_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.step_summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
        print("Appended {}".format(args.step_summary))

    if args.github_annotations:
        for annotation in github_annotation_lines(
            matrix_check,
            matrix_check_path,
            issue_plan,
            issue_plan_path,
            sync_summary,
            sync_summary_path,
        ):
            print(annotation)

    statuses = summary_status(matrix_check, issue_plan, sync_summary)
    if args.fail_on_attention and statuses["overall"] == "attention":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
